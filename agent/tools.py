#!/usr/bin/env python3
"""
RTL Failure Triage Agent - Sandboxed Tool Implementations
Provides the 4 tools available to the Triage Agent:
  1. read_file
  2. grep_code
  3. run_simulation
  4. apply_patch

Security & Sandboxing guarantees:
- Blocks path traversal ('..'), absolute paths, and symlinks pointing outside workspace.
- Blocks any access to 'labels/', 'tb/heldout/', 'bugs/catalog.yaml', 'docs/specs/', 'variants', '.env'.
- Restricts `apply_patch` strictly to the RTL source file under `rtl/` (testbenches are read-only).
- Restricts `run_simulation` to at most 5 executions per run; calls >= 6 are rejected without
  terminating the run so the agent can still output its final JSON answer.
"""

import json
from pathlib import Path
import re
from typing import Any

from agent.workspace import IsolatedWorkspace
from eval.patcher import apply_unified_diff, create_unified_diff

MAX_SIMULATION_CALLS = 5

FORBIDDEN_PATH_TOKENS = frozenset({
    "labels",
    "labels.json",
    "labels.lock",
    "heldout",
    "catalog.yaml",
    "variants",
    "specs",
    ".env",
})


def get_bedrock_tool_specs() -> list[dict[str, Any]]:
    """Returns Bedrock Converse API tool specifications for the 4 agent tools."""
    return [
        {
            "toolSpec": {
                "name": "read_file",
                "description": (
                    "Read a file inside the isolated workspace (e.g. 'rtl/<module>.v' or "
                    "'tb/visible/test_<module>.py'). Returns 1-based numbered lines and raw text."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Workspace-relative file path (e.g., 'rtl/counter.v' or 'tb/visible/test_counter.py').",
                            },
                            "start_line": {
                                "type": "integer",
                                "description": "Optional 1-based start line (inclusive).",
                            },
                            "end_line": {
                                "type": "integer",
                                "description": "Optional 1-based end line (inclusive).",
                            },
                        },
                        "required": ["path"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "grep_code",
                "description": (
                    "Search for a regular expression or substring pattern across files in the workspace "
                    "('rtl/' and 'tb/visible/'). Returns matching relative paths, 1-based line numbers, and line contents."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "Regex or substring pattern to search for.",
                            },
                            "path": {
                                "type": "string",
                                "description": "Optional workspace-relative file or directory path to restrict the search.",
                            },
                        },
                        "required": ["pattern"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "run_simulation",
                "description": (
                    "Compile the workspace RTL with Icarus Verilog and run the visible cocotb testbench "
                    "('tb/visible'). Maximum 5 calls allowed per run; timeout is 60 seconds per call."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {},
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "apply_patch",
                "description": (
                    "Apply a standard unified diff patch to the RTL file in 'rtl/'. "
                    "Modifying testbenches is forbidden. Context lines must match uniquely."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Workspace-relative path to the RTL file (e.g., 'rtl/counter.v').",
                            },
                            "diff": {
                                "type": "string",
                                "description": "Unified diff string with @@ hunk headers and exact context lines.",
                            },
                        },
                        "required": ["file_path", "diff"],
                    }
                },
            }
        },
    ]


class SandboxViolationError(PermissionError):
    """Raised when a tool attempts to access a forbidden or out-of-workspace path."""


def resolve_sandboxed_path(
    workspace: IsolatedWorkspace,
    raw_path: str,
    allow_rtl_basename_shorthand: bool = True,
) -> Path:
    """
    Validates that `raw_path` is a safe relative path strictly within `workspace.root`.
    Rejects:
      - empty paths
      - absolute paths (POSIX '/' or Windows drive/UNC)
      - parent traversal ('..')
      - forbidden path tokens (labels, heldout, catalog.yaml, variants, specs, .env)
      - symlinks anywhere along the path or targets resolving outside workspace.root
    """
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise SandboxViolationError("Permission denied: Path cannot be empty.")

    cleaned = raw_path.strip()

    # Block absolute paths & drive letters
    if cleaned.startswith("/") or cleaned.startswith("\\") or ":" in cleaned or Path(cleaned).is_absolute():
        raise SandboxViolationError(f"Permission denied: Absolute paths are forbidden: {raw_path!r}")

    norm_slashes = cleaned.replace("\\", "/")
    parts = [p for p in norm_slashes.split("/") if p and p != "."]

    if not parts:
        raise SandboxViolationError(f"Permission denied: Invalid path: {raw_path!r}")

    # Block '..' traversal
    if ".." in norm_slashes.split("/") or ".." in Path(cleaned).parts:
        raise SandboxViolationError(f"Permission denied: Parent directory traversal ('..') is forbidden: {raw_path!r}")

    # Block forbidden directories/files
    for part in parts:
        if part.lower() in FORBIDDEN_PATH_TOKENS:
            raise SandboxViolationError(f"Permission denied: Access to '{part}' is restricted: {raw_path!r}")

    ws_root = workspace.root.resolve()

    # Check if shorthand basename for the RTL file was passed (e.g. 'fifo.v' -> 'rtl/fifo.v')
    candidate = workspace.root.joinpath(*parts)
    if (
        allow_rtl_basename_shorthand
        and len(parts) == 1
        and parts[0] == workspace.metadata.rtl_file
        and not candidate.exists()
    ):
        candidate = workspace.rtl_dir / parts[0]
        parts = ["rtl", parts[0]]

    # Walk each prefix component to detect symlinks before resolving
    cur = workspace.root
    for part in parts:
        cur = cur / part
        if cur.is_symlink():
            raise SandboxViolationError(f"Permission denied: Symlinks are forbidden in workspace paths: {raw_path!r}")

    resolved = candidate.resolve()
    if not resolved.is_relative_to(ws_root):
        raise SandboxViolationError(f"Permission denied: Path resolves outside workspace: {raw_path!r}")

    return resolved


class WorkspaceToolExecutor:
    """
    Stateful executor for the 4 agent tools within an IsolatedWorkspace.
    Tracks simulation call count and accepted patches.
    """

    def __init__(self, workspace: IsolatedWorkspace, max_sim_calls: int = MAX_SIMULATION_CALLS) -> None:
        self.workspace = workspace
        self.max_sim_calls = max_sim_calls
        self.sim_call_count: int = 0
        self.rejected_sim_calls: int = 0
        self.first_accepted_patch: str | None = None
        self.first_accepted_patch_raw: str | None = None
        self.accepted_patches: list[str] = []

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Dispatches a tool call by name and returns a sanitized dictionary result."""
        if not isinstance(tool_input, dict):
            return {"status": "ERROR", "error": "Tool input must be a JSON object."}

        try:
            if tool_name == "read_file":
                return self.read_file(
                    path=str(tool_input.get("path", "")),
                    start_line=tool_input.get("start_line"),
                    end_line=tool_input.get("end_line"),
                )
            elif tool_name == "grep_code":
                return self.grep_code(
                    pattern=str(tool_input.get("pattern", "")),
                    path=tool_input.get("path"),
                )
            elif tool_name == "run_simulation":
                return self.run_simulation()
            elif tool_name == "apply_patch":
                return self.apply_patch(
                    file_path=str(tool_input.get("file_path", "")),
                    diff=str(tool_input.get("diff", "")),
                )
            else:
                return {"status": "ERROR", "error": f"Unknown tool: {tool_name!r}"}
        except SandboxViolationError as exc:
            return {"status": "ERROR", "error": self.workspace.sanitize_text(str(exc))}
        except Exception as exc:
            return {"status": "ERROR", "error": self.workspace.sanitize_text(f"{type(exc).__name__}: {exc}")}

    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        target = resolve_sandboxed_path(self.workspace, path)
        if not target.is_file():
            rel = target.relative_to(self.workspace.root).as_posix()
            return {"status": "ERROR", "error": f"File not found in workspace: {rel}"}

        with open(target, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.read().splitlines()

        total_lines = len(all_lines)
        s_line = 1 if start_line is None else max(1, int(start_line))
        e_line = total_lines if end_line is None else min(total_lines, int(end_line))

        if s_line > e_line and total_lines > 0:
            return {
                "status": "ERROR",
                "error": f"Invalid line range: start_line={s_line} > end_line={e_line} (total_lines={total_lines}).",
            }

        selected = all_lines[s_line - 1 : e_line]
        numbered = "\n".join(f"{idx}: {line}" for idx, line in enumerate(selected, start=s_line))
        raw_slice = "\n".join(selected)
        rel_posix = target.relative_to(self.workspace.root).as_posix()

        return {
            "status": "OK",
            "path": rel_posix,
            "start_line": s_line,
            "end_line": e_line,
            "total_lines": total_lines,
            "content": self.workspace.sanitize_text(raw_slice),
            "numbered_content": self.workspace.sanitize_text(numbered),
        }

    def grep_code(self, pattern: str, path: str | None = None) -> dict[str, Any]:
        if not pattern:
            return {"status": "ERROR", "error": "Search pattern cannot be empty."}

        search_files: list[Path] = []
        if path is not None and str(path).strip():
            resolved = resolve_sandboxed_path(self.workspace, str(path))
            if resolved.is_file():
                search_files.append(resolved)
            elif resolved.is_dir():
                for p in sorted(resolved.rglob("*")):
                    if p.is_symlink():
                        raise SandboxViolationError("Permission denied: Symlinks are forbidden in workspace.")
                    if p.is_file() and p.suffix in (".v", ".py"):
                        search_files.append(p)
            else:
                rel = resolved.relative_to(self.workspace.root).as_posix()
                return {"status": "ERROR", "error": f"Path not found in workspace: {rel}"}
        else:
            for folder in (self.workspace.rtl_dir, self.workspace.tb_visible_dir):
                if folder.is_dir():
                    for p in sorted(folder.rglob("*")):
                        if p.is_symlink():
                            raise SandboxViolationError("Permission denied: Symlinks are forbidden in workspace.")
                        if p.is_file() and p.suffix in (".v", ".py"):
                            search_files.append(p)

        try:
            regex = re.compile(pattern)
        except re.error:
            regex = re.compile(re.escape(pattern))

        matches: list[str] = []
        for file_path in search_files:
            rel_posix = file_path.relative_to(self.workspace.root).as_posix()
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                for line_no, line in enumerate(f, start=1):
                    clean_line = line.rstrip("\r\n")
                    if regex.search(clean_line):
                        matches.append(f"{rel_posix}:{line_no}:{self.workspace.sanitize_text(clean_line)}")

        return {
            "status": "OK",
            "pattern": pattern,
            "match_count": len(matches),
            "matches": matches[:200],
        }

    def run_simulation(self) -> dict[str, Any]:
        if self.sim_call_count >= self.max_sim_calls:
            self.rejected_sim_calls += 1
            return {
                "status": "REJECTED",
                "error": (
                    f"Simulation limit reached: maximum {self.max_sim_calls} run_simulation calls allowed per run. "
                    f"This simulation request was rejected, but you may still provide your final JSON answer."
                ),
                "sim_calls_used": self.sim_call_count,
                "sim_calls_remaining": 0,
            }

        self.sim_call_count += 1
        sim_res = self.workspace.run_visible_simulation(timeout=60)
        sim_res["sim_calls_used"] = self.sim_call_count
        sim_res["sim_calls_remaining"] = self.max_sim_calls - self.sim_call_count
        return sim_res

    def apply_patch(self, file_path: str, diff: str) -> dict[str, Any]:
        target = resolve_sandboxed_path(self.workspace, file_path)

        # Enforce strictly RTL-only modification (testbench modification is forbidden)
        if target != self.workspace.rtl_path.resolve():
            return {
                "status": "ERROR",
                "error": (
                    "Permission denied: apply_patch may only modify the RTL design file under 'rtl/'. "
                    "Modifying testbenches or other files is strictly forbidden."
                ),
            }

        current_rtl = self.workspace.read_current_rtl()
        patch_res = apply_unified_diff(current_rtl, diff)
        if not patch_res.success:
            return {
                "status": "ERROR",
                "error": self.workspace.sanitize_text(patch_res.error),
            }

        with open(self.workspace.rtl_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(patch_res.patched_text)

        canonical_diff_from_initial = create_unified_diff(
            self.workspace.initial_rtl_content,
            patch_res.patched_text,
            filename=self.workspace.metadata.rtl_file,
        )

        if self.first_accepted_patch is None:
            self.first_accepted_patch = canonical_diff_from_initial
            self.first_accepted_patch_raw = diff

        self.accepted_patches.append(diff)

        return {
            "status": "APPLIED",
            "file": f"rtl/{self.workspace.metadata.rtl_file}",
            "hunks_applied": patch_res.hunks_applied,
        }
