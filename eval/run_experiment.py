#!/usr/bin/env python3
"""
RTL Failure Triage Agent - Experiment Runner (run_experiment.py)
Executes systematic empirical evaluations strictly following docs/preregistration.md:
- Iterates over bugs x configs (baseline, agent) x repeats.
- Repeat order: All configs complete repeat 1 before moving to repeat 2.
- Directory layout: results/<run_id>/<config>/<bug_id>/<repeat>/trace.jsonl.
- Default --split dev; running on 'test' requires BOTH --split test AND --confirm-test-run.
- Resume capability: Skips runs that already completed without overwriting.
- Stops immediately on BUDGET_HARD_STOP ($15.00) or LLMConfigError across all runs.
- Does NOT score or print held-out test results during run.
- Generates results/<run_id>/manifest.json with all parameters and Section 8 SHA-256 table.
- Pre-flight checks: Verifies labels/labels.lock and docs/gate4_hashes.sha256.
"""

from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

# Ensure workspace root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent.budget import BudgetHardStopError, BudgetManager
from agent.llm import BedrockClient, LLMClient, LLMConfigError
from agent.loop import compute_prompt_version, run_triage_agent
from agent.workspace import load_bug_metadata_for_run
from eval.run_baseline import run_baseline_single
from scripts.check_lock import check_lockfile


def sha256_file(path: Path) -> str:
    """Computes SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_gate4_hashes(project_root: Path) -> tuple[bool, list[str]]:
    """Verifies that all files in docs/gate4_hashes.sha256 match their digests and no untracked files exist."""
    gate4_file = project_root / "docs" / "gate4_hashes.sha256"
    if not gate4_file.is_file():
        return False, ["docs/gate4_hashes.sha256 not found."]

    errors = []
    recorded_files: set[str] = set()
    with open(gate4_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                errors.append(f"Invalid format in gate4_hashes.sha256 line {line_num}: {line}")
                continue
            expected_hash, rel_path = parts[0], parts[1].strip()
            recorded_files.add(rel_path)
            target_path = project_root / rel_path
            if not target_path.is_file():
                errors.append(f"Missing file: {rel_path}")
                continue
            actual_hash = sha256_file(target_path)
            if actual_hash != expected_hash:
                errors.append(f"HASH MISMATCH: {rel_path} (expected {expected_hash[:12]}..., got {actual_hash[:12]}...)")

    # Check for unexpected extra files in agent/ and eval/
    tracked_dirs = ["agent", "eval"]
    for d_name in tracked_dirs:
        d_path = project_root / d_name
        if not d_path.is_dir():
            continue
        for p in sorted(d_path.rglob("*")):
            if p.is_file():
                if "__pycache__" in p.parts or p.name.endswith(".pyc") or p.name == ".gitkeep":
                    continue
                rel_posix = p.relative_to(project_root).as_posix()
                if rel_posix not in recorded_files:
                    errors.append(f"UNEXPECTED EXTRA FILE: {rel_posix} found in repository but not recorded in docs/gate4_hashes.sha256")

    return len(errors) == 0, errors


def generate_gate4_hashes(project_root: Path, out_path: Path) -> int:
    """Generates docs/gate4_hashes.sha256 for all files in agent/ and eval/."""
    tracked_dirs = ["agent", "eval"]
    lines = []
    for d_name in tracked_dirs:
        d_path = project_root / d_name
        if not d_path.is_dir():
            continue
        for p in sorted(d_path.rglob("*")):
            if p.is_file():
                if "__pycache__" in p.parts or p.name.endswith(".pyc") or p.name == ".gitkeep":
                    continue
                rel_posix = p.relative_to(project_root).as_posix()
                digest = sha256_file(p)
                lines.append(f"{digest}  {rel_posix}\n")

    lines.sort(key=lambda x: x.split(None, 1)[1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)
    return len(lines)


def compute_preregistration_sha256_table(project_root: Path) -> dict[str, str]:
    """
    Computes deterministic SHA-256 table according to Section 8 of preregistration:
    - rtl/
    - bugs/variants/
    - bugs/inject.py
    - tb/visible/
    - tb/heldout/
    - bugs/catalog.yaml
    - labels/labels.json
    - eval/
    - agent/
    - docs/preregistration.md
    """
    explicit_files = [
        "bugs/catalog.yaml",
        "labels/labels.json",
        "bugs/inject.py",
        "docs/preregistration.md",
    ]
    tracked_dirs = [
        "rtl",
        "bugs/variants",
        "tb/visible",
        "tb/heldout",
        "eval",
        "agent",
    ]

    table: dict[str, str] = {}
    for rel in explicit_files:
        p = project_root / rel
        if p.is_file():
            table[rel] = sha256_file(p)

    for d_rel in tracked_dirs:
        d_path = project_root / d_rel
        if not d_path.is_dir():
            continue
        for p in sorted(d_path.rglob("*")):
            if p.is_file():
                if "__pycache__" in p.parts or p.name.endswith(".pyc") or p.name == ".gitkeep":
                    continue
                rel_posix = p.relative_to(project_root).as_posix()
                table[rel_posix] = sha256_file(p)

    return dict(sorted(table.items()))


def get_tool_versions() -> dict[str, str]:
    """Captures versions of python, iverilog, and cocotb."""
    versions = {"python": sys.version.split()[0]}
    try:
        r_iv = subprocess.run(["iverilog", "-V"], capture_output=True, text=True, timeout=5)
        first_line = r_iv.stdout.splitlines()[0] if r_iv.stdout else "unknown"
        versions["iverilog"] = first_line.strip()
    except Exception:
        versions["iverilog"] = "unavailable"

    try:
        r_co = subprocess.run(["cocotb-config", "--version"], capture_output=True, text=True, timeout=5)
        versions["cocotb"] = r_co.stdout.strip() if r_co.stdout else "unknown"
    except Exception:
        versions["cocotb"] = "unavailable"

    return versions


def is_run_already_completed(run_dir: Path) -> bool:
    """Checks if a trace file already exists in run_dir and has completed."""
    trace_file = run_dir / "trace.jsonl"
    if not trace_file.is_file():
        return False
    try:
        with open(trace_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        e = json.loads(line)
                        if e.get("event_type") == "run_end":
                            status = e.get("status")
                            if status in ("COMPLETED", "FAILED", "EXCEEDED_LIMIT", "INFRA_ERROR"):
                                return True
                    except json.JSONDecodeError:
                        pass
    except Exception:
        return False
    return False


def run_experiment(
    split: str = "dev",
    confirm_test_run: bool = False,
    repeats: int = 3,
    configs: list[str] | None = None,
    bugs: list[str] | None = None,
    run_id: str | None = None,
    output_base_dir: Path | None = None,
    ledger_path: Path | None = None,
    hard_stop_usd: float = 15.0,
    llm_client: LLMClient | None = None,
    project_root: Path | None = None,
    skip_preflight: bool = False,
) -> int:
    """
    Main orchestration loop for running full benchmarks.
    Returns 0 on complete success, non-zero on failure or hard stop.
    """
    root = (project_root or Path(__file__).resolve().parent.parent).resolve()
    configs = configs or ["baseline", "agent"]
    start_time_iso = datetime.now(timezone.utc).isoformat()

    # Pre-flight Check 0: Reject --skip-preflight on test split
    if split == "test" and skip_preflight:
        sys.stderr.write(
            "[ERROR] Preflight checks cannot be skipped on split 'test'. "
            "--skip-preflight with --split test is strictly forbidden.\n"
        )
        return 1

    # Pre-flight Check 0b: Reject hard stop > $15.00 USD
    if hard_stop_usd > 15.0:
        sys.stderr.write(
            f"[ERROR] --hard-stop threshold cannot exceed $15.00 USD (got ${hard_stop_usd:.2f}). "
            f"Hard stop must be <= $15.00 USD.\n"
        )
        return 1

    # Pre-flight Check 1: Test split gate
    if split == "test" and not confirm_test_run:
        sys.stderr.write(
            "[ERROR] Split 'test' is strictly locked. "
            "You MUST pass BOTH --split test AND --confirm-test-run to execute on the test set.\n"
        )
        return 1

    # Pre-flight Check 2: Lockfile integrity
    if not skip_preflight:
        lockfile_path = root / "labels" / "labels.lock"
        lock_exit = check_lockfile(root, lockfile_path)
        if lock_exit != 0:
            sys.stderr.write("[ERROR] Pre-flight failed: labels/labels.lock verification did not pass.\n")
            return 1

        # Pre-flight Check 3: Gate 4 hashes
        gate4_ok, gate4_errs = verify_gate4_hashes(root)
        if not gate4_ok:
            sys.stderr.write(f"[ERROR] Pre-flight failed: Gate 4 hashes verification failed:\n")
            for err in gate4_errs:
                sys.stderr.write(f"  - {err}\n")
            return 1

    # Load bug metadata and filter by split
    labels_file = root / "labels" / "labels.json"
    with open(labels_file, "r", encoding="utf-8") as f:
        all_labels = json.load(f)

    valid_split_bugs = [item["bug_id"] for item in all_labels if item["split"] == split]
    if bugs:
        selected_bugs = [b for b in bugs if b in valid_split_bugs]
        if not selected_bugs:
            sys.stderr.write(f"[ERROR] None of the requested bugs {bugs} belong to split '{split}'.\n")
            return 1
    else:
        selected_bugs = sorted(valid_split_bugs)

    exp_id = run_id or f"exp_{split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_dir = (output_base_dir or (root / "results" / exp_id)).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    ledger_file = (ledger_path or (root / "results" / "budget_ledger.json")).resolve()
    budget = BudgetManager(ledger_path=ledger_file, hard_stop_usd=hard_stop_usd)

    client = llm_client or BedrockClient()

    # Prompts
    triage_prompt_file = root / "agent" / "prompts" / "triage_system.md"
    baseline_prompt_file = root / "agent" / "prompts" / "baseline_system.md"
    triage_prompt = triage_prompt_file.read_text(encoding="utf-8")
    baseline_prompt = baseline_prompt_file.read_text(encoding="utf-8")

    allow_test_split = (split == "test" and confirm_test_run)

    print("=" * 80)
    print(f"RTL Failure Triage Agent - Experiment Execution")
    print(f"Run ID:         {exp_id}")
    print(f"Split:          {split}")
    print(f"Repeats:        {repeats}")
    print(f"Configs:        {configs}")
    print(f"Bugs ({len(selected_bugs)}):     {selected_bugs}")
    print(f"Output Dir:     {base_dir}")
    print(f"Budget Ledger:  {ledger_file} (Hard Stop: ${hard_stop_usd:.2f})")
    print(f"Model ID:       {client.model_id}")
    print("=" * 80)

    # Manifest handling (create or resume)
    manifest_path = base_dir / "manifest.json"
    tool_vers = get_tool_versions()
    sha256_table = compute_preregistration_sha256_table(root)

    if manifest_path.is_file():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                existing_manifest = json.load(f)
        except Exception as e:
            sys.stderr.write(f"[ERROR] Failed to read existing manifest {manifest_path}: {e}\n")
            return 1

        old_table = existing_manifest.get("sha256_manifest_table", {})
        if old_table != sha256_table:
            sys.stderr.write(
                "[ERROR] Cannot resume experiment: sha256_manifest_table in existing manifest.json "
                "does not match current repository files.\n"
            )
            return 1

        manifest_data = existing_manifest
        resumes = manifest_data.get("resumes", [])
        if not isinstance(resumes, list):
            resumes = []
        resumes.append(start_time_iso)
        manifest_data["resumes"] = resumes
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
    else:
        manifest_data: dict[str, Any] = {
            "manifest_version": "1.0",
            "run_id": exp_id,
            "split": split,
            "repeats": repeats,
            "configs": configs,
            "bugs": selected_bugs,
            "model_id": client.model_id,
            "region": getattr(client, "region", "ap-southeast-1"),
            "inference_params": {
                "max_tokens": client.max_tokens,
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "tool_choice_agent": "auto",
                "tool_choice_baseline": "none",
            },
            "prompt_versions": {
                "triage_system": compute_prompt_version(triage_prompt),
                "baseline_system": compute_prompt_version(baseline_prompt),
            },
            "tool_versions": tool_vers,
            "start_time_utc": start_time_iso,
            "resumes": [],
            "end_time_utc": None,
            "completed": False,
            "stop_reason": None,
            "sha256_manifest_table": sha256_table,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

    total_runs = repeats * len(configs) * len(selected_bugs)
    completed_runs = 0

    try:
        # Preregistration constraint: Complete repeat 1 for both configs before repeat 2
        for repeat_idx in range(1, repeats + 1):
            print(f"\n>>> Starting Repeat {repeat_idx}/{repeats} <<<")
            for config in configs:
                print(f"  --- Configuration: {config.upper()} (Repeat {repeat_idx}) ---")
                for bug_id in selected_bugs:
                    run_dir = base_dir / config / bug_id / str(repeat_idx)
                    run_dir.mkdir(parents=True, exist_ok=True)
                    trace_path = run_dir / "trace.jsonl"

                    if trace_path.is_file():
                        if is_run_already_completed(run_dir):
                            print(f"    [SKIP] Already completed: config={config} bug={bug_id} repeat={repeat_idx}")
                            completed_runs += 1
                            continue
                        else:
                            # Incomplete trace lacks run_end -> rename with timestamp, do not delete
                            ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                            incomplete_path = run_dir / f"trace.incomplete.{ts_str}.jsonl"
                            counter = 1
                            while incomplete_path.exists():
                                incomplete_path = run_dir / f"trace.incomplete.{ts_str}_{counter}.jsonl"
                                counter += 1
                            trace_path.rename(incomplete_path)
                            print(f"    [RETRY INCOMPLETE] Renamed incomplete trace to {incomplete_path.name}")

                    # Load metadata for this run
                    metadata = load_bug_metadata_for_run(
                        bug_id=bug_id,
                        project_root=root,
                        allow_test_split=allow_test_split,
                    )

                    run_sub_id = f"{exp_id}_{config}_{bug_id}_r{repeat_idx}"
                    ws_parent = run_dir / "ws"

                    if config == "agent":
                        res = run_triage_agent(
                            metadata=metadata,
                            llm_client=client,
                            budget_manager=budget,
                            system_prompt=triage_prompt,
                            run_id=run_sub_id,
                            trace_path=trace_path,
                            project_root=root,
                            workspace_parent=ws_parent,
                        )
                    elif config == "baseline":
                        res = run_baseline_single(
                            metadata=metadata,
                            llm_client=client,
                            budget_manager=budget,
                            system_prompt=baseline_prompt,
                            run_id=run_sub_id,
                            trace_path=trace_path,
                            project_root=root,
                            workspace_parent=ws_parent,
                        )
                    else:
                        raise ValueError(f"Unknown config: {config}")

                    # Clean up workspace directory after run
                    if ws_parent.exists():
                        import shutil
                        shutil.rmtree(ws_parent, ignore_errors=True)

                    if res.hard_stop_triggered or res.status == "BUDGET_HARD_STOP":
                        sys.stderr.write(f"\n[FATAL] BUDGET HARD STOP TRIGGERED in run {run_sub_id}: {res.stop_reason}\n")
                        manifest_data["stop_reason"] = f"BUDGET_HARD_STOP: {res.stop_reason}"
                        with open(manifest_path, "w", encoding="utf-8") as f:
                            json.dump(manifest_data, f, indent=2)
                        return 1

                    if res.status == "CONFIG_ERROR":
                        sys.stderr.write(f"\n[FATAL] CONFIGURATION / PERMISSION ERROR in run {run_sub_id}: {res.stop_reason}\n")
                        manifest_data["stop_reason"] = f"LLM_CONFIG_ERROR: {res.stop_reason}"
                        with open(manifest_path, "w", encoding="utf-8") as f:
                            json.dump(manifest_data, f, indent=2)
                        return 1

                    completed_runs += 1
                    print(
                        f"    [DONE {completed_runs}/{total_runs}] {config:8s} {bug_id:5s} r{repeat_idx} -> "
                        f"status={res.status:10s} turns={res.llm_turns:2d} cost=${res.total_cost_usd:.4f}"
                    )

    except BudgetHardStopError as bhs:
        sys.stderr.write(f"\n[FATAL] BUDGET HARD STOP TRIGGERED: {bhs}\n")
        manifest_data["stop_reason"] = f"BUDGET_HARD_STOP: {bhs}"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        return 1

    except LLMConfigError as lce:
        sys.stderr.write(f"\n[FATAL] CONFIGURATION / PERMISSION ERROR: {lce}\n")
        manifest_data["stop_reason"] = f"LLM_CONFIG_ERROR: {lce}"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        return 1

    manifest_data["completed"] = True
    manifest_data["end_time_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_data["stop_reason"] = "completed"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[EXPERIMENT COMPLETED] All {completed_runs}/{total_runs} runs executed successfully.")
    print(f"Results written to: {base_dir}")
    print(f"Manifest saved to:  {manifest_path}")
    print("=" * 80)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Orchestrator for RTL Triage Empirical Experiments")
    parser.add_argument("--run-id", type=str, default=None, help="Unique identifier for experiment")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"], help="Dataset split (default: dev)")
    parser.add_argument("--confirm-test-run", action="store_true", help="Required confirmation flag to run on test split")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repetitions per bug (default: 3)")
    parser.add_argument("--configs", nargs="+", default=["baseline", "agent"], choices=["baseline", "agent"])
    parser.add_argument("--bugs", nargs="+", default=None, help="Optional subset of bug IDs to run")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to store experiment results")
    parser.add_argument("--ledger-path", type=str, default=None, help="Path to budget ledger json")
    parser.add_argument("--hard-stop", type=float, default=15.0, help="Budget hard stop threshold in USD")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip pre-flight checks (testing only)")
    parser.add_argument("--generate-gate4-hashes", action="store_true", help="Generate docs/gate4_hashes.sha256 and exit")

    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent

    if args.generate_gate4_hashes:
        out_gate4 = root / "docs" / "gate4_hashes.sha256"
        count = generate_gate4_hashes(root, out_gate4)
        print(f"[OK] Wrote {count} hashes to {out_gate4}")
        return 0

    if args.split == "test" and args.skip_preflight:
        sys.stderr.write(
            "[ERROR] Split 'test' cannot be run with --skip-preflight. "
            "Preflight verification is mandatory for test split.\n"
        )
        return 1

    if args.hard_stop > 15.0:
        sys.stderr.write(
            f"[ERROR] --hard-stop threshold cannot exceed $15.00 USD (got ${args.hard_stop:.2f}). "
            f"Hard stop must be <= $15.00 USD.\n"
        )
        return 1

    return run_experiment(
        split=args.split,
        confirm_test_run=args.confirm_test_run,
        repeats=args.repeats,
        configs=args.configs,
        bugs=args.bugs,
        run_id=args.run_id,
        output_base_dir=Path(args.output_dir) if args.output_dir else None,
        ledger_path=Path(args.ledger_path) if args.ledger_path else None,
        hard_stop_usd=args.hard_stop,
        skip_preflight=args.skip_preflight,
    )


if __name__ == "__main__":
    sys.exit(main())
