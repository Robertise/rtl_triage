#!/usr/bin/env python3
"""
RTL Failure Triage Agent - Multi-turn Tool-Calling Loop
Executes the Triage Agent on a single bug inside an IsolatedWorkspace.

Enforces preregistration constraints:
- Max 20 LLM turns per run (`EXCEEDED_LIMIT` if all 20 turns exhausted without final answer).
- Max 5 `run_simulation` executions per run (6th+ calls rejected by tool, agent can still answer).
- Pre-call budget check via BudgetManager before every LLM turn (`BUDGET_HARD_STOP`).
- Up to 2 run-level retries on transient `LLMInfraError` (`INFRA_ERROR` if all fail).
- Tracks `MAX_TOKENS_STOP` and `PATCH_APPLY_FAILED` separately.
- Generates final patch from workspace diff against initial mutated RTL (or falls back to
  JSON `patch` if workspace is unchanged, which also serves as `first_accepted_patch`).
- Writes complete, credential-redacted, path-sanitized JSONL trace.
"""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

from agent.budget import BudgetHardStopError, BudgetManager
from agent.llm import LLMClient, LLMConfigError, LLMInfraError
from agent.tools import MAX_SIMULATION_CALLS, WorkspaceToolExecutor, get_bedrock_tool_specs
from agent.workspace import BugRunMetadata, IsolatedWorkspace, redact_credentials
from eval.parser import parse_model_output
from eval.patcher import apply_unified_diff

MAX_LLM_TURNS = 20
MAX_INFRA_RETRIES = 2
MAX_MALFORMED_OUTPUT_RETRIES = 2


@dataclass
class AgentRunResult:
    run_id: str
    config: str  # "agent" or "baseline"
    status: str  # "COMPLETED", "FAILED", "EXCEEDED_LIMIT", "INFRA_ERROR", "BUDGET_HARD_STOP"
    stop_reason: str
    model_id: str
    prompt_version: str
    llm_turns: int = 0
    sim_calls: int = 0
    rejected_sim_calls: int = 0
    infra_retries_used: int = 0
    malformed_output_retries: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    max_tokens_stop: bool = False
    patch_apply_failed: bool = False
    hard_stop_triggered: bool = False
    first_accepted_patch: str = ""
    final_patch: str = ""
    final_patch_source: str = "none"  # "workspace_diff", "json_patch", or "none"
    parsed_output: dict[str, Any] | None = None
    reported_verified: bool | None = None
    trace_path: Path | None = None
    trace_events: list[dict[str, Any]] = field(default_factory=list)


def compute_prompt_version(prompt_text: str) -> str:
    """Computes a 16-char SHA-256 fingerprint of a system prompt."""
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]


def build_initial_user_prompt(workspace: IsolatedWorkspace, initial_sim: dict[str, Any]) -> str:
    """
    Constructs the sanitized initial problem prompt containing:
      1. The initial visible testbench (`tb/visible`) failure summary and log.
      2. The initial mutated RTL file (`rtl/<module>.v`) with 1-based line numbers and raw text.
    Shared identically between Triage Agent and Baseline.
    """
    rtl_rel = f"rtl/{workspace.metadata.rtl_file}"
    raw_rtl = workspace.initial_rtl_content
    numbered_lines = "\n".join(
        f"{idx}: {line}" for idx, line in enumerate(raw_rtl.splitlines(), start=1)
    )

    prompt = (
        f"## Target Module & File\n"
        f"- Module: `{workspace.metadata.module}`\n"
        f"- RTL File: `{rtl_rel}`\n"
        f"- Visible Testbench: `tb/visible/test_{workspace.metadata.module}.py`\n\n"
        f"## Initial Visible Testbench (`tb/visible`) Failure Output\n"
        f"- Status: `{initial_sim.get('status', 'FAIL')}`\n"
        f"- Tests Run: `{initial_sim.get('tests', 0)}`, Failures: `{initial_sim.get('failures', 0)}`, Errors: `{initial_sim.get('errors', 0)}`\n"
        f"- Summary Message: `{initial_sim.get('message', '')}`\n\n"
        f"### Simulation Log (`sim_build/sim.log`)\n"
        f"```text\n{initial_sim.get('sim_log', '').strip()}\n```\n\n"
        f"## Initial Unpatched RTL Source (`{rtl_rel}` - 1-based line numbers)\n"
        f"```verilog\n{numbered_lines}\n```\n\n"
        f"## Initial Unpatched RTL Source (`{rtl_rel}` - Raw Text)\n"
        f"```verilog\n{raw_rtl.rstrip()}\n```\n"
    )
    return workspace.sanitize_text(prompt)


def write_trace_jsonl(trace_path: Path, events: list[dict[str, Any]]) -> None:
    """Writes trace events as JSONL with credential redaction."""
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trace_path, "w", encoding="utf-8", newline="\n") as f:
        for ev in events:
            raw_json = json.dumps(ev, ensure_ascii=False)
            safe_json = redact_credentials(raw_json)
            f.write(safe_json + "\n")


def run_triage_agent(
    metadata: BugRunMetadata,
    llm_client: LLMClient,
    budget_manager: BudgetManager,
    system_prompt: str | None = None,
    run_id: str = "agent_run",
    trace_path: Path | None = None,
    project_root: Path | None = None,
    workspace_parent: Path | None = None,
    max_llm_turns: int = MAX_LLM_TURNS,
    max_sim_calls: int = MAX_SIMULATION_CALLS,
    max_infra_retries: int = MAX_INFRA_RETRIES,
) -> AgentRunResult:
    """
    Runs the multi-turn Triage Agent loop on a single bug inside an isolated workspace.
    Retries up to `max_infra_retries` times at the run level if `LLMInfraError` occurs.
    """
    root = (project_root or Path(__file__).resolve().parent.parent).resolve()

    if system_prompt is None:
        prompt_file = root / "agent" / "prompts" / "triage_system.md"
        with open(prompt_file, "r", encoding="utf-8") as f:
            system_prompt = f.read()

    prompt_version = compute_prompt_version(system_prompt)
    tool_specs = get_bedrock_tool_specs()

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
            initial_sim = workspace.run_visible_simulation(timeout=60)
            initial_user_prompt = build_initial_user_prompt(workspace, initial_sim)

            tool_executor = WorkspaceToolExecutor(workspace=workspace, max_sim_calls=max_sim_calls)
            messages: list[dict[str, Any]] = [
                {
                    "role": "user",
                    "content": [{"text": initial_user_prompt}],
                }
            ]

            all_trace_events.append({
                "event_type": "run_start",
                "run_id": run_id,
                "config": "agent",
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

            status = "EXCEEDED_LIMIT"
            stop_reason = "exceeded_20_llm_turns"
            max_tokens_stop = False
            patch_apply_failed = False
            hard_stop_triggered = False
            parsed_output: dict[str, Any] | None = None
            reported_verified: bool | None = None
            turns_used = 0
            malformed_output_retries = 0
            consecutive_malformed = 0

            try:
                while turns_used < max_llm_turns:
                    # 1. Check hard-stop budget strictly before calling LLM
                    try:
                        budget_manager.check_can_call()
                    except BudgetHardStopError as b_exc:
                        status = "BUDGET_HARD_STOP"
                        stop_reason = workspace.sanitize_text(str(b_exc))
                        hard_stop_triggered = True
                        break

                    # 2. Invoke LLM
                    turns_used += 1
                    current_turn = turns_used

                    resp = llm_client.converse(
                        system_prompt=system_prompt,
                        messages=messages,
                        tools=tool_specs,
                    )

                    call_cost = budget_manager.record_usage(
                        input_tokens=resp.input_tokens,
                        output_tokens=resp.output_tokens,
                        run_id=run_id,
                        config="agent",
                        split=metadata.split,
                        turn=current_turn,
                        model_id=llm_client.model_id,
                    )
                    cumulative_input_tokens += resp.input_tokens
                    cumulative_output_tokens += resp.output_tokens
                    cumulative_cost_usd = round(cumulative_cost_usd + call_cost, 8)

                    sanitized_assistant_text = workspace.sanitize_text(resp.content_text)
                    serialized_tool_calls = [
                        {
                            "tool_use_id": tc.tool_use_id,
                            "name": tc.name,
                            "input": tc.input,
                        }
                        for tc in resp.tool_calls
                    ]

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
                        "tool_calls": serialized_tool_calls,
                    })

                    # Handle malformed_model_output stop reason per pre-Gate 4 rule
                    if resp.stop_reason == "malformed_model_output":
                        if consecutive_malformed < MAX_MALFORMED_OUTPUT_RETRIES:
                            consecutive_malformed += 1
                            malformed_output_retries += 1
                            # DO NOT parse, DO NOT add message to conversation history.
                            # Loop continues and resends the exact same request.
                            continue
                        else:
                            # Reached limit of 2 consecutive retries (3 calls total) at this position
                            status = "FAILED"
                            stop_reason = "malformed_model_output"
                            break

                    # Reset consecutive malformed counter on any non-malformed response
                    consecutive_malformed = 0

                    # 3. Handle tool calls if present
                    if resp.tool_calls:
                        messages.append(resp.raw_assistant_message)
                        tool_result_blocks: list[dict[str, Any]] = []

                        for tc in resp.tool_calls:
                            tool_out = tool_executor.execute(tc.name, tc.input)
                            sanitized_out_str = workspace.sanitize_text(json.dumps(tool_out, ensure_ascii=False))
                            sanitized_out_obj = json.loads(sanitized_out_str)

                            all_trace_events.append({
                                "event_type": "tool_result",
                                "run_id": run_id,
                                "attempt": attempt,
                                "turn": current_turn,
                                "tool_use_id": tc.tool_use_id,
                                "tool_name": tc.name,
                                "tool_input": tc.input,
                                "tool_output": sanitized_out_obj,
                                "sim_call_count": tool_executor.sim_call_count,
                                "rejected_sim_calls": tool_executor.rejected_sim_calls,
                            })

                            tool_result_blocks.append({
                                "toolResult": {
                                    "toolUseId": tc.tool_use_id,
                                    "content": [{"json": sanitized_out_obj}],
                                }
                            })

                        messages.append({
                            "role": "user",
                            "content": tool_result_blocks,
                        })
                        continue

                    # 4. No tool calls -> Final response turn
                    parse_res = parse_model_output(
                        raw_text=resp.content_text,
                        mode="agent",
                        stop_reason=resp.stop_reason,
                    )
                    max_tokens_stop = parse_res.max_tokens_stop
                    if parse_res.valid and parse_res.data is not None:
                        status = "COMPLETED"
                        stop_reason = "completed"
                        parsed_output = parse_res.data
                        reported_verified = bool(parsed_output.get("verified"))
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

            # 5. Determine first_accepted_patch and final_patch per Section 4.2 of preregistration
            first_accepted_patch = tool_executor.first_accepted_patch or ""
            final_patch = ""
            final_patch_source = "none"

            if workspace.is_rtl_modified():
                final_patch = workspace.compute_final_diff()
                final_patch_source = "workspace_diff"
            elif parsed_output is not None and parsed_output.get("patch"):
                json_patch = str(parsed_output["patch"])
                final_patch = json_patch
                final_patch_source = "json_patch"
                if not first_accepted_patch:
                    first_accepted_patch = json_patch
                # Check if JSON patch applies cleanly to initial RTL
                patch_check = apply_unified_diff(workspace.initial_rtl_content, json_patch)
                if not patch_check.success:
                    patch_apply_failed = True
            elif status == "COMPLETED" and parsed_output is not None:
                # Workspace unchanged and JSON patch is empty -> cannot apply patch
                final_patch_source = "json_patch"
                patch_apply_failed = True

            end_event = {
                "event_type": "run_end",
                "run_id": run_id,
                "config": "agent",
                "attempt": attempt,
                "status": status,
                "stop_reason": stop_reason,
                "model_id": llm_client.model_id,
                "prompt_version": prompt_version,
                "llm_turns": turns_used,
                "sim_calls": tool_executor.sim_call_count,
                "rejected_sim_calls": tool_executor.rejected_sim_calls,
                "infra_retries_used": infra_retries_used,
                "malformed_output_retries": malformed_output_retries,
                "total_input_tokens": cumulative_input_tokens,
                "total_output_tokens": cumulative_output_tokens,
                "total_cost_usd": cumulative_cost_usd,
                "max_tokens_stop": max_tokens_stop,
                "patch_apply_failed": patch_apply_failed,
                "hard_stop_triggered": hard_stop_triggered,
                "first_accepted_patch": redact_credentials(first_accepted_patch),
                "final_patch": redact_credentials(final_patch),
                "final_patch_source": final_patch_source,
                "parsed_output": parsed_output,
                "reported_verified": reported_verified,
            }
            all_trace_events.append(end_event)

            if trace_path is not None:
                write_trace_jsonl(trace_path, all_trace_events)

            return AgentRunResult(
                run_id=run_id,
                config="agent",
                status=status,
                stop_reason=stop_reason,
                model_id=llm_client.model_id,
                prompt_version=prompt_version,
                llm_turns=turns_used,
                sim_calls=tool_executor.sim_call_count,
                rejected_sim_calls=tool_executor.rejected_sim_calls,
                infra_retries_used=infra_retries_used,
                malformed_output_retries=malformed_output_retries,
                total_input_tokens=cumulative_input_tokens,
                total_output_tokens=cumulative_output_tokens,
                total_cost_usd=cumulative_cost_usd,
                max_tokens_stop=max_tokens_stop,
                patch_apply_failed=patch_apply_failed,
                hard_stop_triggered=hard_stop_triggered,
                first_accepted_patch=redact_credentials(first_accepted_patch),
                final_patch=redact_credentials(final_patch),
                final_patch_source=final_patch_source,
                parsed_output=parsed_output,
                reported_verified=reported_verified,
                trace_path=trace_path,
                trace_events=all_trace_events,
            )
        finally:
            workspace.cleanup()

    raise RuntimeError("Unreachable loop exit in run_triage_agent.")
