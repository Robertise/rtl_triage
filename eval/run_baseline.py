#!/usr/bin/env python3
"""
RTL Failure Triage Agent - Single-Shot Baseline Runner
Runs the Baseline configuration on a single bug:
- Receives the exact same sanitized `tb/visible` failure log and mutated RTL as the Triage Agent.
- Invokes the LLM exactly ONCE (`tools=None`), with no tool calls and no interactive simulations.
- Uses the shared `eval/parser.py` (mode='baseline', no 'verified' field) and `eval/patcher.py`.
- Enforces the shared persistent budget ledger (`BudgetManager`) and run-level retry on `LLMInfraError`.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from agent.budget import BudgetHardStopError, BudgetManager
from agent.llm import BedrockClient, LLMClient, LLMConfigError, LLMInfraError
from agent.loop import (
    MAX_INFRA_RETRIES,
    MAX_MALFORMED_OUTPUT_RETRIES,
    AgentRunResult,
    build_initial_user_prompt,
    compute_prompt_version,
    write_trace_jsonl,
)
from agent.workspace import (
    BugRunMetadata,
    IsolatedWorkspace,
    load_bug_metadata_for_run,
    redact_credentials,
)
from eval.parser import parse_model_output
from eval.patcher import apply_unified_diff


def run_baseline_single(
    metadata: BugRunMetadata,
    llm_client: LLMClient,
    budget_manager: BudgetManager,
    system_prompt: str | None = None,
    run_id: str = "baseline_run",
    trace_path: Path | None = None,
    project_root: Path | None = None,
    workspace_parent: Path | None = None,
    max_infra_retries: int = MAX_INFRA_RETRIES,
) -> AgentRunResult:
    """
    Executes the single-shot Baseline configuration on a bug inside an isolated workspace.
    Calls the LLM at most 1 turn per attempt with tools=None.
    """
    root = (project_root or Path(__file__).resolve().parent.parent).resolve()

    if system_prompt is None:
        prompt_file = root / "agent" / "prompts" / "baseline_system.md"
        with open(prompt_file, "r", encoding="utf-8") as f:
            system_prompt = f.read()

    prompt_version = compute_prompt_version(system_prompt)
    all_trace_events: list[dict[str, Any]] = []
    cumulative_input_tokens = 0
    cumulative_output_tokens = 0
    cumulative_cost_usd = 0.0
    infra_retries_used = 0

    for attempt in range(1, max_infra_retries + 2):
        workspace = IsolatedWorkspace(
            metadata=metadata,
            project_root=root,
            workspace_parent=workspace_parent,
        )
        workspace.setup()
        try:
            # Capture initial visible failure output to construct identical input prompt as Agent
            initial_sim = workspace.run_visible_simulation(timeout=60)
            initial_user_prompt = build_initial_user_prompt(workspace, initial_sim)

            messages: list[dict[str, Any]] = [
                {
                    "role": "user",
                    "content": [{"text": initial_user_prompt}],
                }
            ]

            all_trace_events.append({
                "event_type": "run_start",
                "run_id": run_id,
                "config": "baseline",
                "attempt": attempt,
                "model_id": llm_client.model_id,
                "region": llm_client.region,
                "max_tokens": llm_client.max_tokens,
                "prompt_version": prompt_version,
                "system_prompt": workspace.sanitize_text(system_prompt),
                "initial_user_prompt": initial_user_prompt,
                "module": metadata.module,
                "rtl_file": metadata.rtl_file,
            })

            status = "FAILED"
            stop_reason = ""
            max_tokens_stop = False
            patch_apply_failed = False
            hard_stop_triggered = False
            parsed_output: dict[str, Any] | None = None
            turns_used = 0
            malformed_output_retries = 0
            consecutive_malformed = 0

            try:
                while True:
                    # 1. Pre-call budget check
                    try:
                        budget_manager.check_can_call()
                    except BudgetHardStopError as b_exc:
                        status = "BUDGET_HARD_STOP"
                        stop_reason = workspace.sanitize_text(str(b_exc))
                        hard_stop_triggered = True
                        break

                    # 2. Single-shot LLM call with NO tools
                    turns_used += 1
                    current_turn = turns_used

                    resp = llm_client.converse(
                        system_prompt=system_prompt,
                        messages=messages,
                        tools=None,
                    )

                    call_cost = budget_manager.record_usage(
                        input_tokens=resp.input_tokens,
                        output_tokens=resp.output_tokens,
                        run_id=run_id,
                        config="baseline",
                        split=metadata.split,
                        turn=current_turn,
                        model_id=llm_client.model_id,
                    )
                    cumulative_input_tokens += resp.input_tokens
                    cumulative_output_tokens += resp.output_tokens
                    cumulative_cost_usd = round(cumulative_cost_usd + call_cost, 8)

                    sanitized_assistant_text = workspace.sanitize_text(resp.content_text)

                    all_trace_events.append({
                        "event_type": "llm_turn",
                        "run_id": run_id,
                        "attempt": attempt,
                        "turn": current_turn,
                        "input_tokens": resp.input_tokens,
                        "output_tokens": resp.output_tokens,
                        "cost_usd": call_cost,
                        "cumulative_run_cost_usd": cumulative_cost_usd,
                        "stop_reason": resp.stop_reason,
                        "assistant_text": sanitized_assistant_text,
                        "tool_calls": [],
                    })

                    # Handle malformed_model_output stop reason per pre-Gate 4 rule
                    if resp.stop_reason == "malformed_model_output":
                        if consecutive_malformed < MAX_MALFORMED_OUTPUT_RETRIES:
                            consecutive_malformed += 1
                            malformed_output_retries += 1
                            # DO NOT parse, DO NOT add message. Loop continues and resends exact same request.
                            continue
                        else:
                            # Reached limit of 2 consecutive retries (3 calls total) at this position
                            status = "FAILED"
                            stop_reason = "malformed_model_output"
                            break

                    parse_res = parse_model_output(
                        raw_text=resp.content_text,
                        mode="baseline",
                        stop_reason=resp.stop_reason,
                    )
                    max_tokens_stop = parse_res.max_tokens_stop
                    if parse_res.valid and parse_res.data is not None:
                        status = "COMPLETED"
                        stop_reason = "completed"
                        parsed_output = parse_res.data
                    else:
                        status = "FAILED"
                        stop_reason = f"parse_failed: {parse_res.error}"
                    break

            except LLMConfigError as cfg_exc:
                all_trace_events.append({
                    "event_type": "config_error",
                    "run_id": run_id,
                    "attempt": attempt,
                    "error": str(cfg_exc),
                })
                status = "CONFIG_ERROR"
                stop_reason = f"config_error: {cfg_exc}"

            except LLMInfraError as infra_exc:
                all_trace_events.append({
                    "event_type": "infra_error",
                    "run_id": run_id,
                    "attempt": attempt,
                    "error": str(infra_exc),
                })
                if attempt <= max_infra_retries:
                    infra_retries_used += 1
                    continue
                else:
                    status = "INFRA_ERROR"
                    stop_reason = str(infra_exc)

            # 3. Baseline patch comes strictly from JSON `patch` field (and equals first_accepted_patch)
            final_patch = ""
            final_patch_source = "none"
            if parsed_output is not None:
                final_patch = str(parsed_output.get("patch", ""))
                final_patch_source = "json_patch"
                patch_check = apply_unified_diff(workspace.initial_rtl_content, final_patch)
                if not patch_check.success:
                    patch_apply_failed = True

            end_event = {
                "event_type": "run_end",
                "run_id": run_id,
                "config": "baseline",
                "attempt": attempt,
                "status": status,
                "stop_reason": stop_reason,
                "model_id": llm_client.model_id,
                "prompt_version": prompt_version,
                "llm_turns": turns_used,
                "sim_calls": 0,
                "rejected_sim_calls": 0,
                "infra_retries_used": infra_retries_used,
                "malformed_output_retries": malformed_output_retries,
                "total_input_tokens": cumulative_input_tokens,
                "total_output_tokens": cumulative_output_tokens,
                "total_cost_usd": cumulative_cost_usd,
                "max_tokens_stop": max_tokens_stop,
                "patch_apply_failed": patch_apply_failed,
                "hard_stop_triggered": hard_stop_triggered,
                "first_accepted_patch": redact_credentials(final_patch),
                "final_patch": redact_credentials(final_patch),
                "final_patch_source": final_patch_source,
                "parsed_output": parsed_output,
                "reported_verified": None,
            }
            all_trace_events.append(end_event)

            if trace_path is not None:
                write_trace_jsonl(trace_path, all_trace_events)

            return AgentRunResult(
                run_id=run_id,
                config="baseline",
                status=status,
                stop_reason=stop_reason,
                model_id=llm_client.model_id,
                prompt_version=prompt_version,
                llm_turns=turns_used,
                sim_calls=0,
                rejected_sim_calls=0,
                infra_retries_used=infra_retries_used,
                malformed_output_retries=malformed_output_retries,
                total_input_tokens=cumulative_input_tokens,
                total_output_tokens=cumulative_output_tokens,
                total_cost_usd=cumulative_cost_usd,
                max_tokens_stop=max_tokens_stop,
                patch_apply_failed=patch_apply_failed,
                hard_stop_triggered=hard_stop_triggered,
                first_accepted_patch=redact_credentials(final_patch),
                final_patch=redact_credentials(final_patch),
                final_patch_source=final_patch_source,
                parsed_output=parsed_output,
                reported_verified=None,
                trace_path=trace_path,
                trace_events=all_trace_events,
            )
        finally:
            workspace.cleanup()

    raise RuntimeError("Unreachable loop exit in run_baseline_single.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Single-Shot Baseline on a single dev bug.")
    parser.add_argument("--bug-id", required=True, help="Bug ID to run (must be in 'dev' split during Phase 4).")
    parser.add_argument("--trace-out", default=None, help="Optional path for JSONL trace output.")
    parser.add_argument("--ledger-path", default=None, help="Optional path to persistent budget ledger JSON.")
    parser.add_argument("--out-json", default=None, help="Optional path to write summary JSON result.")

    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parent.parent

    try:
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
        project_root / "results" / "dev_runs" / f"baseline_{args.bug_id}_trace.jsonl"
    )
    client = BedrockClient()

    result = run_baseline_single(
        metadata=metadata,
        llm_client=client,
        budget_manager=budget,
        run_id=f"baseline_{args.bug_id}",
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
