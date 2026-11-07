#!/usr/bin/env python3
"""
RTL Failure Triage Agent - CLI Entry Point
Runs the Triage Agent on a single bug inside an isolated workspace.
At Phase 4 (prior to Gate 4 lock), only 'dev' split bugs are permitted;
passing a 'test' split bug exits immediately with a non-zero error code.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from agent.budget import BudgetManager
from agent.llm import BedrockClient, LLMClient
from agent.loop import AgentRunResult, run_triage_agent
from agent.workspace import load_bug_metadata_for_run


def execute_agent_for_bug(
    bug_id: str,
    llm_client: LLMClient | None = None,
    budget_manager: BudgetManager | None = None,
    trace_path: Path | None = None,
    project_root: Path | None = None,
    allow_test_split: bool = False,
) -> AgentRunResult:
    """
    Loads bug metadata (enforcing 'dev' split restriction when allow_test_split=False)
    and executes the Triage Agent loop.
    """
    root = (project_root or Path(__file__).resolve().parent.parent).resolve()
    metadata = load_bug_metadata_for_run(
        bug_id=bug_id,
        project_root=root,
        allow_test_split=allow_test_split,
    )

    client = llm_client if llm_client is not None else BedrockClient()
    budget = budget_manager if budget_manager is not None else BudgetManager()

    if trace_path is None:
        trace_path = root / "results" / "dev_runs" / f"agent_{bug_id}_trace.jsonl"

    return run_triage_agent(
        metadata=metadata,
        llm_client=client,
        budget_manager=budget,
        run_id=f"agent_{bug_id}",
        trace_path=trace_path,
        project_root=root,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RTL Failure Triage Agent on a single dev bug.")
    parser.add_argument("--bug-id", required=True, help="Bug ID to triage (must be in 'dev' split during Phase 4).")
    parser.add_argument("--trace-out", default=None, help="Optional path for JSONL trace output.")
    parser.add_argument("--ledger-path", default=None, help="Optional path to persistent budget ledger JSON.")
    parser.add_argument("--out-json", default=None, help="Optional path to write summary JSON result.")

    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parent.parent

    try:
        # Strictly enforce dev-only split at CLI level
        metadata = load_bug_metadata_for_run(
            bug_id=args.bug_id,
            project_root=project_root,
            allow_test_split=False,
        )
    except (PermissionError, ValueError, FileNotFoundError) as exc:
        sys.stderr.write(f"[ERROR] {exc}\n")
        return 1

    budget = BudgetManager(ledger_path=args.ledger_path) if args.ledger_path else BudgetManager()
    trace_path = Path(args.trace_out).resolve() if args.trace_out else (
        project_root / "results" / "dev_runs" / f"agent_{args.bug_id}_trace.jsonl"
    )
    client = BedrockClient()

    result = run_triage_agent(
        metadata=metadata,
        llm_client=client,
        budget_manager=budget,
        run_id=f"agent_{args.bug_id}",
        trace_path=trace_path,
        project_root=project_root,
    )

    summary: dict[str, Any] = {
        "run_id": result.run_id,
        "config": result.config,
        "status": result.status,
        "stop_reason": result.stop_reason,
        "model_id": result.model_id,
        "prompt_version": result.prompt_version,
        "llm_turns": result.llm_turns,
        "sim_calls": result.sim_calls,
        "rejected_sim_calls": result.rejected_sim_calls,
        "infra_retries_used": result.infra_retries_used,
        "malformed_output_retries": result.malformed_output_retries,
        "total_input_tokens": result.total_input_tokens,
        "total_output_tokens": result.total_output_tokens,
        "total_cost_usd": result.total_cost_usd,
        "max_tokens_stop": result.max_tokens_stop,
        "patch_apply_failed": result.patch_apply_failed,
        "hard_stop_triggered": result.hard_stop_triggered,
        "first_accepted_patch": result.first_accepted_patch,
        "final_patch": result.final_patch,
        "final_patch_source": result.final_patch_source,
        "parsed_output": result.parsed_output,
        "reported_verified": result.reported_verified,
        "trace_path": str(result.trace_path) if result.trace_path else None,
    }

    out_str = json.dumps(summary, indent=2, ensure_ascii=False)
    print(out_str)

    if args.out_json:
        out_path = Path(args.out_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(out_str + "\n")

    return 0 if result.status == "COMPLETED" else 1


if __name__ == "__main__":
    sys.exit(main())
