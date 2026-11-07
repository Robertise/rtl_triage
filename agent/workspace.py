#!/usr/bin/env python3
"""
RTL Failure Triage Agent - Isolated Workspace Manager
Creates a strictly isolated per-run workspace containing ONLY:
  - rtl/<module>.v  (mutated RTL copied from bugs/variants/<bug_id>/<module>.v)
  - tb/visible/test_<module>.py  (copied from tb/visible/test_<module>.py)

Isolation guarantees:
- Neutral workspace directory naming (no bug_id, no 'variants', no clue words).
- Reads labels/labels.json ONLY prior to workspace setup to verify split ('dev' vs 'test')
  and obtain module/file metadata; never retains or exposes bug_type, description,
  correct_fix, alternative_fixes, or ground-truth line numbers.
- Sanitizes all simulation logs and tool outputs by replacing absolute paths with
  workspace-relative paths before sending to the LLM.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys

from eval.patcher import create_unified_diff

CREDENTIAL_PATTERNS = [
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"(?i)(aws_secret_access_key|aws_session_token|aws_access_key_id)\s*[:=]\s*\S+"),
]
MAX_SIM_LOG_CHARS = 4000
MAX_SIM_LOG_LINES = 100


@dataclass(frozen=True)
class BugRunMetadata:
    """
    Minimal metadata required to construct a workspace.
    Intentionally excludes bug_type, description, correct_fix, alternative_fixes,
    line_start, and line_end so ground-truth answers can never leak into prompts.
    """
    bug_id: str
    split: str
    module: str
    rtl_file: str


def load_bug_metadata_for_run(
    bug_id: str,
    project_root: Path | None = None,
    allow_test_split: bool = False,
) -> BugRunMetadata:
    """
    Reads labels/labels.json solely to check split and locate module/file.
    Rejects 'test' split bugs when allow_test_split is False.
    """
    root = (project_root or Path(__file__).resolve().parent.parent).resolve()
    labels_path = root / "labels" / "labels.json"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    with open(labels_path, "r", encoding="utf-8") as f:
        labels_list = json.load(f)

    matched = None
    for entry in labels_list:
        if entry.get("bug_id") == bug_id:
            matched = entry
            break

    if matched is None:
        raise ValueError(f"Unknown bug_id: {bug_id!r}")

    split = str(matched.get("split", "")).strip()
    if not allow_test_split and split != "dev":
        raise PermissionError(
            f"Refused to run bug '{bug_id}': split is '{split}'. "
            f"Only 'dev' split bugs are permitted at this stage."
        )

    module = str(matched["module"]).strip()
    rtl_file = Path(str(matched["file"]).strip()).name

    return BugRunMetadata(
        bug_id=bug_id,
        split=split,
        module=module,
        rtl_file=rtl_file,
    )


def redact_credentials(text: str) -> str:
    """Scrubs any potential AWS key or secret patterns from text."""
    result = text
    for pat in CREDENTIAL_PATTERNS:
        result = pat.sub("[REDACTED_CREDENTIAL]", result)
    return result


def truncate_sim_log(log_text: str, max_chars: int = MAX_SIM_LOG_CHARS, max_lines: int = MAX_SIM_LOG_LINES) -> str:
    """
    Truncates long simulation logs to a fixed limit and appends an explicit truncation marker.
    """
    lines = log_text.splitlines()
    truncated_by_lines = False
    if len(lines) > max_lines:
        half = max_lines // 2
        lines = (
            lines[:half]
            + [f"...[TRUNCATED: omitted {len(lines) - max_lines} middle line(s); limit={max_lines} lines]..."]
            + lines[-half:]
        )
        truncated_by_lines = True

    rejoined = "\n".join(lines)
    if len(rejoined) > max_chars:
        half_c = (max_chars - 120) // 2
        omitted = len(rejoined) - (2 * half_c)
        rejoined = (
            rejoined[:half_c]
            + f"\n...[TRUNCATED: omitted {omitted} chars; limit={max_chars} chars]...\n"
            + rejoined[-half_c:]
        )
    elif truncated_by_lines:
        rejoined += f"\n[TRUNCATED: simulation log exceeded {max_lines} lines limit]"

    return rejoined


class IsolatedWorkspace:
    """
    Manages a neutral, sandboxed workspace directory for a single run.
    """

    def __init__(
        self,
        metadata: BugRunMetadata,
        project_root: Path | None = None,
        workspace_parent: Path | None = None,
    ) -> None:
        self.metadata = metadata
        self.project_root = (project_root or Path(__file__).resolve().parent.parent).resolve()

        if workspace_parent is None:
            base_tmp = Path("/tmp/rtl_triage_workspaces") if os.name != "nt" else (self.project_root / "sim_build" / "workspaces")
        else:
            base_tmp = Path(workspace_parent).resolve()

        base_tmp.mkdir(parents=True, exist_ok=True)

        # Neutral directory name: never contains bug_id or 'variants'
        neutral_token = secrets.token_hex(6)
        self.root: Path = (base_tmp / f"ws_{neutral_token}").resolve()
        self.rtl_dir: Path = self.root / "rtl"
        self.tb_visible_dir: Path = self.root / "tb" / "visible"
        self.sim_build_dir: Path = self.root / "sim_build"

        self.rtl_path: Path = self.rtl_dir / self.metadata.rtl_file
        self.test_file_name: str = f"test_{self.metadata.module}.py"
        self.test_path: Path = self.tb_visible_dir / self.test_file_name
        self.initial_rtl_content: str = ""

    def setup(self) -> None:
        """Creates the isolated workspace and copies only the mutated RTL and visible testbench."""
        variant_src = self.project_root / "bugs" / "variants" / self.metadata.bug_id / self.metadata.rtl_file
        tb_src = self.project_root / "tb" / "visible" / self.test_file_name

        if not variant_src.is_file():
            raise FileNotFoundError(f"Mutated RTL file not found: {variant_src}")
        if not tb_src.is_file():
            raise FileNotFoundError(f"Visible testbench file not found: {tb_src}")

        self.rtl_dir.mkdir(parents=True, exist_ok=True)
        self.tb_visible_dir.mkdir(parents=True, exist_ok=True)
        self.sim_build_dir.mkdir(parents=True, exist_ok=True)

        with open(variant_src, "r", encoding="utf-8") as f:
            self.initial_rtl_content = f.read()

        with open(self.rtl_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(self.initial_rtl_content)

        with open(tb_src, "r", encoding="utf-8") as f:
            tb_content = f.read()

        with open(self.test_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(tb_content)

    def cleanup(self) -> None:
        """Removes the isolated workspace directory."""
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)

    def sanitize_text(self, text: str) -> str:
        """
        Replaces all occurrences of absolute workspace and project paths with
        relative paths, and scrubs credentials.
        Does NOT alter RTL content, patches, testbenches, logs, or model text.
        """
        if not text:
            return ""

        out = text
        # Replace workspace root variations (POSIX and Windows)
        ws_posix = self.root.as_posix()
        ws_str = str(self.root)
        for prefix in sorted({ws_posix + "/", ws_str + "\\", ws_str + "/"}, key=len, reverse=True):
            out = out.replace(prefix, "")
        for exact in sorted({ws_posix, ws_str}, key=len, reverse=True):
            out = out.replace(exact, ".")

        # Replace project root variations
        proj_posix = self.project_root.as_posix()
        proj_str = str(self.project_root)
        for prefix in sorted({proj_posix + "/", proj_str + "\\", proj_str + "/"}, key=len, reverse=True):
            out = out.replace(prefix, "")
        for exact in sorted({proj_posix, proj_str}, key=len, reverse=True):
            out = out.replace(exact, ".")

        return redact_credentials(out)

    def run_visible_simulation(self, timeout: int = 60) -> dict:
        """
        Executes scripts/run_sim.py against the workspace's rtl/<module>.v and tb/visible.
        Returns a sanitized dictionary with status, tests, failures, errors, message, and log_excerpt.
        """
        run_sim_script = self.project_root / "scripts" / "run_sim.py"
        results_xml = self.sim_build_dir / "results.xml"
        cmd = [
            sys.executable,
            "-B",
            str(run_sim_script),
            "--module",
            self.metadata.module,
            "--src-file",
            str(self.rtl_path),
            "--test-dir",
            str(self.tb_visible_dir),
            "--test-module",
            f"test_{self.metadata.module}",
            "--build-dir",
            str(self.sim_build_dir),
            "--results-xml",
            str(results_xml),
            "--timeout",
            str(timeout),
        ]

        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        try:
            raw_res = json.loads(proc.stdout)
        except json.JSONDecodeError:
            raw_res = {
                "status": "FAIL",
                "tests": 0,
                "failures": 1,
                "errors": 1,
                "log_path": "sim_build/sim.log",
                "message": proc.stderr.strip() or proc.stdout.strip() or "Simulation execution error.",
            }

        sim_log_path = self.sim_build_dir / "sim.log"
        raw_log = ""
        if sim_log_path.is_file():
            with open(sim_log_path, "r", encoding="utf-8", errors="replace") as f:
                raw_log = f.read()

        # Remove any pycache if created inside workspace tb/visible
        pycache_dir = self.tb_visible_dir / "__pycache__"
        if pycache_dir.exists():
            shutil.rmtree(pycache_dir, ignore_errors=True)

        sanitized_msg = self.sanitize_text(str(raw_res.get("message", "")))
        sanitized_log = truncate_sim_log(self.sanitize_text(raw_log))

        return {
            "status": raw_res.get("status", "FAIL"),
            "tests": int(raw_res.get("tests", 0)),
            "failures": int(raw_res.get("failures", 0)),
            "errors": int(raw_res.get("errors", 0)),
            "log_path": "sim_build/sim.log",
            "message": sanitized_msg,
            "sim_log": sanitized_log,
        }

    def read_current_rtl(self) -> str:
        """Reads current content of the RTL file in the workspace."""
        with open(self.rtl_path, "r", encoding="utf-8") as f:
            return f.read()

    def is_rtl_modified(self) -> bool:
        """Returns True if workspace RTL differs from initial mutated RTL."""
        return self.read_current_rtl() != self.initial_rtl_content

    def compute_final_diff(self) -> str:
        """
        Generates unified diff between initial mutated RTL and current workspace RTL.
        Returns empty string if workspace RTL was not modified.
        """
        current_rtl = self.read_current_rtl()
        if current_rtl == self.initial_rtl_content:
            return ""
        return create_unified_diff(
            self.initial_rtl_content,
            current_rtl,
            filename=self.metadata.rtl_file,
        )
