#!/usr/bin/env python3
"""
RTL Failure Triage Agent - Phase 4 Offline Test Suite
Tests all Phase 4 components inside Docker using FakeLLM only (zero network / Bedrock calls):
1. End-to-end dev bug triage with FakeLLM (fail -> patch -> pass -> valid JSON).
2. 20 LLM turn limit (`EXCEEDED_LIMIT`) and 6th `run_simulation` rejection while still allowing final JSON answer.
3. Sandbox path blocking (`..`, absolute paths, symlinks outside workspace, `labels/`, `tb/heldout/`, `.env`).
4. `apply_patch` & `eval/patcher.py`: rejects testbench modification, malformed diffs, ambiguous multi-match contexts; multi-step patches produce accurate final workspace diff and preserve `first_accepted_patch`.
5. `eval/parser.py`: strips 1 markdown fence, normalizes `file` to basename, allows unknown `bug_type` without failing run, ignores extra fields and baseline `verified`, rejects missing fields / wrong types (`bool` as `line`), tracks `MAX_TOKENS_STOP`.
6. `agent/budget.py`: stops before an LLM call that would exceed hard stop (`BUDGET_HARD_STOP`), persists and reloads ledger accurately across restarts.
7. Split enforcement: rejects `test` split bugs in CLI and runner (`PermissionError` / exit code != 0).
8. Sanitization & zero-leakage across all 18 bugs:
   - RTL in initial prompt matches `bugs/variants/<bug_id>/<file>` byte-for-byte (raw text section).
   - Prompt, log, and `read_file` contain no `[REDACTED`.
   - Prompt contains no uppercase `B0xx` ID, no `variants`, no ground-truth `bug_type`, `description`, or `correct_fix`.
   - B018 `read_file` returns exact `3'b001` uncorrupted.
   - ALU `final_patch` applies cleanly to restore clean RTL.
9. `eval/run_baseline.py`: invokes LLM exactly once (`tools=None`), runs 0 interactive tool simulations, ignores extra `verified`.
10. `INFRA_ERROR` run-level retry (retries up to 2 times and succeeds or marks `INFRA_ERROR`).
11. Bedrock exception classification & `LLMConfigError` non-retrying termination for Agent and Baseline.
"""

import json
from pathlib import Path
import re
import socket
import sys
import pytest

from agent.budget import BudgetHardStopError, BudgetManager, compute_call_cost_usd
from agent.llm import (
    FakeLLM,
    LLMConfigError,
    LLMInfraError,
    classify_bedrock_exception,
)
from agent.loop import build_initial_user_prompt, run_triage_agent
from agent.run_agent import main as run_agent_cli_main
from agent.tools import WorkspaceToolExecutor
from agent.workspace import (
    BugRunMetadata,
    IsolatedWorkspace,
    load_bug_metadata_for_run,
    truncate_sim_log,
)
from eval.parser import parse_model_output
from eval.patcher import apply_unified_diff, create_unified_diff
from eval.run_baseline import main as run_baseline_cli_main, run_baseline_single


@pytest.fixture(autouse=True)
def forbid_network_and_bedrock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strictly blocks any socket connection or boto3 client creation during offline tests."""
    def _blocked_connect(*args, **kwargs):
        raise AssertionError("Network connections are strictly forbidden in Phase 4 offline tests!")

    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked_connect)

    try:
        import boto3
        def _blocked_boto_client(*args, **kwargs):
            raise AssertionError("boto3.client() is strictly forbidden in Phase 4 offline tests!")
        monkeypatch.setattr(boto3, "client", _blocked_boto_client)
    except ImportError:
        pass


class MockClientError(Exception):
    """Simulates a botocore ClientError with response metadata and error code."""
    def __init__(self, code: str, status_code: int, message: str = "Test error") -> None:
        super().__init__(f"ClientError: {code} (HTTP {status_code}) - {message}")
        self.response = {
            "Error": {
                "Code": code,
                "Message": message,
            },
            "ResponseMetadata": {
                "HTTPStatusCode": status_code,
            },
        }


# ---------------------------------------------------------------------------
# 1. End-to-End Dev Bug Run with FakeLLM (fail -> patch -> pass -> JSON)
# ---------------------------------------------------------------------------
def test_e2e_dev_bug_fail_patch_pass_and_no_leakage(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parent.parent
    metadata = load_bug_metadata_for_run("B015", project_root=project_root, allow_test_split=False)
    assert metadata.split == "dev"
    assert metadata.module == "counter"
    assert metadata.rtl_file == "counter.v"

    with open(project_root / "labels" / "labels.json", "r", encoding="utf-8") as f:
        labels = {entry["bug_id"]: entry for entry in json.load(f)}
    gt_b015 = labels["B015"]

    valid_diff = (
        "--- a/counter.v\n"
        "+++ b/counter.v\n"
        "@@ -23,4 +23,4 @@\n"
        "         if (!reset_n) begin\n"
        "             val_out <= {WIDTH_BITS{1'b0}};\n"
        "-        end else begin\n"
        "+        end else if (cnt_en) begin\n"
        "             if (val_out >= limit_val) begin\n"
    )

    final_json_obj = {
        "file": "rtl/counter.v",
        "line": 25,
        "bug_type": "missing_branch",
        "root_cause": "Missing enable check in sequential counter block.",
        "patch": valid_diff,
        "verified": True,
    }

    script = [
        # Turn 1: Read RTL and grep
        {
            "text": "Inspecting counter.v and searching for cnt_en.",
            "tool_calls": [
                {"tool_use_id": "call_1", "name": "read_file", "input": {"path": "rtl/counter.v"}},
                {"tool_use_id": "call_2", "name": "grep_code", "input": {"pattern": "cnt_en"}},
            ],
            "input_tokens": 1200,
            "output_tokens": 150,
        },
        # Turn 2: Run simulation on unpatched RTL -> confirm FAIL
        {
            "text": "Running visible simulation before patching.",
            "tool_calls": [
                {"tool_use_id": "call_3", "name": "run_simulation", "input": {}},
            ],
            "input_tokens": 1600,
            "output_tokens": 100,
        },
        # Turn 3: Apply patch fixing the bug
        {
            "text": "Applying unified diff patch to rtl/counter.v.",
            "tool_calls": [
                {
                    "tool_use_id": "call_4",
                    "name": "apply_patch",
                    "input": {"file_path": "rtl/counter.v", "diff": valid_diff},
                },
            ],
            "input_tokens": 1900,
            "output_tokens": 180,
        },
        # Turn 4: Run simulation on patched RTL -> confirm PASS
        {
            "text": "Verifying patched RTL with run_simulation.",
            "tool_calls": [
                {"tool_use_id": "call_5", "name": "run_simulation", "input": {}},
            ],
            "input_tokens": 2200,
            "output_tokens": 100,
        },
        # Turn 5: Return final JSON wrapped in a single markdown code fence
        {
            "text": f"```json\n{json.dumps(final_json_obj, indent=2)}\n```",
            "stop_reason": "end_turn",
            "input_tokens": 2500,
            "output_tokens": 220,
        },
    ]

    fake_llm = FakeLLM(script=script)
    ledger_path = tmp_path / "ledger.json"
    trace_path = tmp_path / "trace_b015.jsonl"
    budget = BudgetManager(ledger_path=ledger_path)

    result = run_triage_agent(
        metadata=metadata,
        llm_client=fake_llm,
        budget_manager=budget,
        run_id="test_b015_e2e",
        trace_path=trace_path,
        project_root=project_root,
        workspace_parent=tmp_path / "ws_parent",
    )

    assert result.status == "COMPLETED"
    assert result.llm_turns == 5
    assert result.sim_calls == 2
    assert result.rejected_sim_calls == 0
    assert result.parsed_output is not None
    assert result.parsed_output["file"] == "counter.v"
    assert result.parsed_output["line"] == 25
    assert result.reported_verified is True
    assert result.final_patch_source == "workspace_diff"
    assert "+        end else if (cnt_en) begin" in result.final_patch
    assert "+        end else if (cnt_en) begin" in result.first_accepted_patch

    tool_events = [e for e in result.trace_events if e.get("event_type") == "tool_result"]
    sim_events = [e for e in tool_events if e.get("tool_name") == "run_simulation"]
    assert len(sim_events) == 2
    assert sim_events[0]["tool_output"]["status"] == "FAIL"
    assert sim_events[1]["tool_output"]["status"] == "PASS"

    assert len(fake_llm.calls) == 5
    first_call_serialized = json.dumps(fake_llm.calls[0])
    assert "B015" not in first_call_serialized
    assert "variants" not in first_call_serialized.lower()
    assert gt_b015["description"] not in first_call_serialized
    assert str(project_root) not in first_call_serialized
    assert str(tmp_path) not in first_call_serialized

    for call in fake_llm.calls:
        inf_cfg = call["inferenceConfig"]
        assert inf_cfg == {"maxTokens": 8192}
        assert "temperature" not in inf_cfg
        assert "topP" not in inf_cfg
        assert "topK" not in inf_cfg
        assert call["toolConfig"]["toolChoice"] == {"auto": {}}

    assert trace_path.is_file()
    trace_lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(trace_lines) == len(result.trace_events)


# ---------------------------------------------------------------------------
# 2. Limits: 6th Simulation Rejected & 20-Turn Limit
# ---------------------------------------------------------------------------
def test_6th_simulation_rejected_but_agent_still_completes(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parent.parent
    metadata = load_bug_metadata_for_run("B012", project_root=project_root, allow_test_split=False)

    script = []
    for i in range(1, 7):
        script.append({
            "text": f"Simulation request #{i}",
            "tool_calls": [{"tool_use_id": f"sim_{i}", "name": "run_simulation", "input": {}}],
            "input_tokens": 500,
            "output_tokens": 80,
        })

    valid_diff_b012 = (
        "--- a/alu.v\n"
        "+++ b/alu.v\n"
        "@@ -17,3 +17,3 @@\n"
        " \n"
        "     // Combinational zero flag\n"
        "-    assign flag_zero = (alu_res[3:0] == 4'b0000);\n"
        "+    assign flag_zero = (alu_res == {DATA_WIDTH{1'b0}});\n"
        " \n"
    )
    script.append({
        "text": json.dumps({
            "file": "alu.v",
            "line": 19,
            "bug_type": "bit_width_mismatch",
            "root_cause": "flag_zero only checks lower 4 bits.",
            "patch": valid_diff_b012,
            "verified": False,
        }),
        "stop_reason": "end_turn",
        "input_tokens": 600,
        "output_tokens": 120,
    })

    fake_llm = FakeLLM(script=script)
    budget = BudgetManager(ledger_path=tmp_path / "ledger_sim6.json")

    result = run_triage_agent(
        metadata=metadata,
        llm_client=fake_llm,
        budget_manager=budget,
        project_root=project_root,
        workspace_parent=tmp_path / "ws_sim6",
    )

    assert result.status == "COMPLETED"
    assert result.llm_turns == 7
    assert result.sim_calls == 5
    assert result.rejected_sim_calls == 1

    sim_results = [
        e["tool_output"]
        for e in result.trace_events
        if e.get("event_type") == "tool_result" and e.get("tool_name") == "run_simulation"
    ]
    assert len(sim_results) == 6
    assert all(r["status"] == "FAIL" for r in sim_results[:5])
    assert sim_results[5]["status"] == "REJECTED"
    assert sim_results[5]["sim_calls_used"] == 5
    assert sim_results[5]["sim_calls_remaining"] == 0


def test_20_llm_turns_exceeded_limit_preserves_first_patch(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parent.parent
    metadata = load_bug_metadata_for_run("B012", project_root=project_root, allow_test_split=False)

    valid_diff_b012 = (
        "--- a/alu.v\n"
        "+++ b/alu.v\n"
        "@@ -17,3 +17,3 @@\n"
        " \n"
        "     // Combinational zero flag\n"
        "-    assign flag_zero = (alu_res[3:0] == 4'b0000);\n"
        "+    assign flag_zero = (alu_res == {DATA_WIDTH{1'b0}});\n"
        " \n"
    )

    script = [
        {
            "text": "Turn 1: apply first patch",
            "tool_calls": [
                {"tool_use_id": "t_1", "name": "apply_patch", "input": {"file_path": "rtl/alu.v", "diff": valid_diff_b012}}
            ],
            "input_tokens": 500,
            "output_tokens": 100,
        }
    ]
    for turn in range(2, 21):
        script.append({
            "text": f"Turn {turn}: reading file",
            "tool_calls": [
                {"tool_use_id": f"t_{turn}", "name": "read_file", "input": {"path": "rtl/alu.v"}}
            ],
            "input_tokens": 500,
            "output_tokens": 50,
        })

    fake_llm = FakeLLM(script=script)
    budget = BudgetManager(ledger_path=tmp_path / "ledger_20.json")

    result = run_triage_agent(
        metadata=metadata,
        llm_client=fake_llm,
        budget_manager=budget,
        project_root=project_root,
        workspace_parent=tmp_path / "ws_20",
    )

    assert result.status == "EXCEEDED_LIMIT"
    assert result.llm_turns == 20
    assert result.parsed_output is None
    assert "+    assign flag_zero = (alu_res == {DATA_WIDTH{1'b0}});" in result.first_accepted_patch


# ---------------------------------------------------------------------------
# 3. Path Sandboxing: '..', Absolute Paths, Symlinks, Forbidden Files
# ---------------------------------------------------------------------------
def test_sandbox_blocks_traversal_absolute_symlinks_and_forbidden_paths(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parent.parent
    metadata = load_bug_metadata_for_run("B005", project_root=project_root, allow_test_split=False)
    ws = IsolatedWorkspace(metadata=metadata, project_root=project_root, workspace_parent=tmp_path)
    ws.setup()
    try:
        executor = WorkspaceToolExecutor(workspace=ws)

        outside_secret = tmp_path / "secret_outside.txt"
        outside_secret.write_text("TOP_SECRET", encoding="utf-8")
        symlink_path = ws.rtl_dir / "escape_link.v"
        symlink_path.symlink_to(outside_secret)

        forbidden_targets = [
            "../secret_outside.txt",
            "rtl/../../secret_outside.txt",
            "/etc/passwd",
            str(project_root / "labels" / "labels.json"),
            "labels/labels.json",
            "tb/heldout/test_fsm_controller.py",
            "bugs/catalog.yaml",
            "docs/specs/fsm_controller_spec.md",
            ".env",
            "rtl/escape_link.v",
        ]

        for bad_path in forbidden_targets:
            read_res = executor.execute("read_file", {"path": bad_path})
            assert read_res["status"] == "ERROR", f"read_file failed to block {bad_path!r}"
            assert "Permission denied" in read_res["error"]

            grep_res = executor.execute("grep_code", {"pattern": ".*", "path": bad_path})
            assert grep_res["status"] == "ERROR", f"grep_code failed to block {bad_path!r}"
            assert "Permission denied" in grep_res["error"]

        grep_all_res = executor.execute("grep_code", {"pattern": "module"})
        assert grep_all_res["status"] == "ERROR"
        assert "Symlinks are forbidden" in grep_all_res["error"]
    finally:
        ws.cleanup()


# ---------------------------------------------------------------------------
# 4. Patcher & apply_patch: Restrictions & Multi-Step Diff
# ---------------------------------------------------------------------------
def test_apply_patch_restrictions_and_multistep_diff(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parent.parent
    metadata = load_bug_metadata_for_run("B015", project_root=project_root, allow_test_split=False)
    ws = IsolatedWorkspace(metadata=metadata, project_root=project_root, workspace_parent=tmp_path)
    ws.setup()
    try:
        executor = WorkspaceToolExecutor(workspace=ws)

        tb_diff = (
            "--- a/tb/visible/test_counter.py\n"
            "+++ b/tb/visible/test_counter.py\n"
            "@@ -1,3 +1,3 @@\n"
            "-import cocotb\n"
            "+import cocotb # modified\n"
        )
        res_tb = executor.execute("apply_patch", {"file_path": "tb/visible/test_counter.py", "diff": tb_diff})
        assert res_tb["status"] == "ERROR"
        assert "Permission denied" in res_tb["error"]

        res_bad = executor.execute("apply_patch", {"file_path": "rtl/counter.v", "diff": "not a valid diff"})
        assert res_bad["status"] == "ERROR"

        ambiguous_diff = (
            "--- a/counter.v\n"
            "+++ b/counter.v\n"
            "@@ -4,1 +4,2 @@\n"
            " \n"
            "+// ambiguous insertion at blank line\n"
        )
        res_ambig = executor.execute("apply_patch", {"file_path": "rtl/counter.v", "diff": ambiguous_diff})
        assert res_ambig["status"] == "ERROR"
        assert "Ambiguous hunk" in res_ambig["error"]

        repeated_verilog = (
            "module m;\n"
            "  always @(*) begin\n"
            "    if (a) begin\n"
            "      flag = 1'b0;\n"
            "    end\n"
            "    if (b) begin\n"
            "      flag = 1'b0;\n"
            "    end\n"
            "  end\n"
            "endmodule\n"
        )
        ambig_code_diff = (
            "--- a/m.v\n"
            "+++ b/m.v\n"
            "@@ -4,2 +4,2 @@\n"
            "-      flag = 1'b0;\n"
            "+      flag = 1'b1;\n"
            "     end\n"
        )
        ambig_code_res = apply_unified_diff(repeated_verilog, ambig_code_diff)
        assert ambig_code_res.success is False
        assert "Ambiguous hunk" in ambig_code_res.error

        step1_diff = (
            "--- a/counter.v\n"
            "+++ b/counter.v\n"
            "@@ -23,4 +23,4 @@\n"
            "         if (!reset_n) begin\n"
            "             val_out <= {WIDTH_BITS{1'b0}};\n"
            "-        end else begin\n"
            "+        end else if (reset_n) begin\n"
            "             if (val_out >= limit_val) begin\n"
        )
        res_step1 = executor.execute("apply_patch", {"file_path": "rtl/counter.v", "diff": step1_diff})
        assert res_step1["status"] == "APPLIED"

        step2_diff = (
            "--- a/counter.v\n"
            "+++ b/counter.v\n"
            "@@ -23,4 +23,4 @@\n"
            "         if (!reset_n) begin\n"
            "             val_out <= {WIDTH_BITS{1'b0}};\n"
            "-        end else if (reset_n) begin\n"
            "+        end else if (cnt_en) begin\n"
            "             if (val_out >= limit_val) begin\n"
        )
        res_step2 = executor.execute("apply_patch", {"file_path": "rtl/counter.v", "diff": step2_diff})
        assert res_step2["status"] == "APPLIED"

        assert executor.first_accepted_patch is not None
        assert "+        end else if (reset_n) begin" in executor.first_accepted_patch

        final_diff = ws.compute_final_diff()
        assert "+        end else if (cnt_en) begin" in final_diff
        assert "else if (reset_n)" not in final_diff

        patched_final = apply_unified_diff(ws.initial_rtl_content, final_diff)
        assert patched_final.success
        clean_counter = (project_root / "rtl" / "counter.v").read_text(encoding="utf-8")
        assert patched_final.patched_text == clean_counter
    finally:
        ws.cleanup()


# ---------------------------------------------------------------------------
# 5. Parser: Code Fence, Basename, Extra Fields, Baseline verified
# ---------------------------------------------------------------------------
def test_parser_rules_and_edge_cases() -> None:
    project_root = Path(__file__).resolve().parent.parent
    raw_fenced = (
        "```json\n"
        + json.dumps({
            "file": "rtl/subdir/fifo.v",
            "line": 30,
            "bug_type": "custom_unknown_category",
            "root_cause": "Some explanation",
            "patch": "--- a/fifo.v\n+++ b/fifo.v\n",
            "verified": True,
            "extra_field_1": "ignored_value",
            "notes": "extra commentary",
        })
        + "\n```"
    )
    res_ok = parse_model_output(raw_fenced, mode="agent", stop_reason="end_turn")
    assert res_ok.valid is True
    assert res_ok.status == "COMPLETED"
    assert res_ok.is_known_bug_type is False
    assert res_ok.data is not None
    assert res_ok.data["file"] == "fifo.v"
    assert "extra_field_1" not in res_ok.data

    # Baseline with extra 'verified' field must ignore it and be COMPLETED
    baseline_with_verified = json.dumps({
        "file": "rtl/alu.v",
        "line": 15,
        "bug_type": "bit_width_mismatch",
        "root_cause": "Explanation",
        "patch": "--- a/alu.v\n+++ b/alu.v\n",
        "verified": True,
        "extra_field_2": 999,
    })
    res_base = parse_model_output(baseline_with_verified, mode="baseline", stop_reason="end_turn")
    assert res_base.valid is True
    assert res_base.status == "COMPLETED"
    assert res_base.data is not None
    assert res_base.data["file"] == "alu.v"
    assert "verified" not in res_base.data
    assert "extra_field_2" not in res_base.data

    # Preamble text + exactly one ```json block -> COMPLETED (for both agent and baseline)
    text_with_one_json_fence = (
        "Here is my diagnostic analysis and solution:\n"
        "```json\n"
        + json.dumps({
            "file": "rtl/counter.v",
            "line": 26,
            "bug_type": "missing_branch",
            "root_cause": "Missing enable check.",
            "patch": "--- a/counter.v\n+++ b/counter.v\n",
            "verified": True,
        })
        + "\n```\nHope this helps!"
    )
    res_one_block = parse_model_output(text_with_one_json_fence, mode="agent", stop_reason="end_turn")
    assert res_one_block.valid is True
    assert res_one_block.status == "COMPLETED"
    assert res_one_block.data is not None
    assert res_one_block.data["file"] == "counter.v"
    assert res_one_block.data["line"] == 26
    assert res_one_block.data["bug_type"] == "missing_branch"
    assert res_one_block.data["verified"] is True

    # Same with baseline mode (ignores verified)
    res_one_block_base = parse_model_output(text_with_one_json_fence, mode="baseline", stop_reason="end_turn")
    assert res_one_block_base.valid is True
    assert res_one_block_base.status == "COMPLETED"
    assert res_one_block_base.data is not None
    assert res_one_block_base.data["file"] == "counter.v"
    assert "verified" not in res_one_block_base.data

    # Preamble text + 2 ```json blocks -> FAILED
    text_with_two_json_fences = (
        "Here is candidate 1:\n"
        "```json\n"
        + json.dumps({
            "file": "rtl/counter.v",
            "line": 26,
            "bug_type": "missing_branch",
            "root_cause": "First hypothesis.",
            "patch": "--- a/counter.v\n+++ b/counter.v\n",
            "verified": False,
        })
        + "\n```\nAnd here is candidate 2:\n"
        "```json\n"
        + json.dumps({
            "file": "rtl/counter.v",
            "line": 26,
            "bug_type": "missing_branch",
            "root_cause": "Second hypothesis.",
            "patch": "--- a/counter.v\n+++ b/counter.v\n",
            "verified": True,
        })
        + "\n```"
    )
    res_two_blocks = parse_model_output(text_with_two_json_fences, mode="agent", stop_reason="end_turn")
    assert res_two_blocks.valid is False
    assert res_two_blocks.status == "FAILED"
    assert "Ambiguous model output" in res_two_blocks.error

    # Text + JSON without code fence -> FAILED (no unfenced JSON extraction)
    text_with_unfenced_json = (
        "Here is the final JSON output:\n"
        + json.dumps({
            "file": "rtl/counter.v",
            "line": 26,
            "bug_type": "missing_branch",
            "root_cause": "Unfenced output test.",
            "patch": "--- a/counter.v\n+++ b/counter.v\n",
            "verified": True,
        })
        + "\nEnd of response."
    )
    res_unfenced = parse_model_output(text_with_unfenced_json, mode="agent", stop_reason="end_turn")
    assert res_unfenced.valid is False
    assert res_unfenced.status == "FAILED"
    assert "Invalid JSON" in res_unfenced.error

    # Real final text from B015 dev run fixture must be COMPLETED
    fixture_b015 = project_root / "tests" / "fixtures" / "agent_B015_final_output.txt"
    assert fixture_b015.is_file(), f"Missing fixture file: {fixture_b015}"
    b015_text = fixture_b015.read_text(encoding="utf-8")

    res_b015 = parse_model_output(b015_text, mode="agent", stop_reason="end_turn")
    assert res_b015.valid is True
    assert res_b015.status == "COMPLETED"
    assert res_b015.data is not None
    assert res_b015.data["file"] == "counter.v"
    assert res_b015.data["line"] == 26
    assert res_b015.data["bug_type"] == "missing_branch"
    assert res_b015.data["verified"] is True
    assert "if (cnt_en)" in res_b015.data["patch"]

    double_fenced = f"```json\n{raw_fenced}\n```"
    res_double = parse_model_output(double_fenced, mode="agent", stop_reason="end_turn")
    assert res_double.valid is False
    assert res_double.status == "FAILED"

    missing_field = json.dumps({
        "file": "fifo.v",
        "line": 30,
        "bug_type": "fifo_flag",
        "root_cause": "Explanation",
        "patch": "",
    })
    res_missing = parse_model_output(missing_field, mode="agent", stop_reason="end_turn")
    assert res_missing.valid is False
    assert res_missing.status == "FAILED"

    bool_line = json.dumps({
        "file": "fifo.v",
        "line": True,
        "bug_type": "fifo_flag",
        "root_cause": "Explanation",
        "patch": "",
        "verified": False,
    })
    res_bool_line = parse_model_output(bool_line, mode="agent", stop_reason="end_turn")
    assert res_bool_line.valid is False
    assert res_bool_line.status == "FAILED"

    truncated_json = '{"file": "fifo.v", "line": 30, "bug_type": "fifo_flag"'
    res_max_tok = parse_model_output(truncated_json, mode="agent", stop_reason="max_tokens")
    assert res_max_tok.valid is False
    assert res_max_tok.status == "FAILED"
    assert res_max_tok.max_tokens_stop is True


# ---------------------------------------------------------------------------
# 6. Budget Hard Stop & Persistent Ledger Across Restarts
# ---------------------------------------------------------------------------
def test_budget_persistence_and_hard_stop_before_call(tmp_path: Path) -> None:
    ledger_file = tmp_path / "persistent_ledger.json"

    b1 = BudgetManager(ledger_path=ledger_file, hard_stop_usd=0.05)
    # At Claude Haiku 4.5 rates ($1.00/1M in, $5.00/1M out): (10000*1.00 + 4000*5.00) / 1M = $0.030
    c1 = b1.record_usage(input_tokens=10_000, output_tokens=4_000, run_id="r1", config="agent", split="dev", turn=1)
    assert c1 == pytest.approx(0.030)
    assert b1.total_cost_usd == pytest.approx(0.030)

    b2 = BudgetManager(ledger_path=ledger_file, hard_stop_usd=0.05)
    assert b2.total_cost_usd == pytest.approx(0.030)
    assert b2.total_input_tokens == 10_000
    assert b2.total_output_tokens == 4_000
    assert b2.total_calls == 1

    with pytest.raises(BudgetHardStopError):
        b2.check_can_call(estimated_next_cost_usd=0.025)

    # Bring total cost to $0.05: (10000*1.00 + 2000*5.00) / 1M = $0.020 -> 0.030 + 0.020 = 0.050
    b2.record_usage(input_tokens=10_000, output_tokens=2_000, run_id="r1", config="agent", split="dev", turn=2)
    assert b2.total_cost_usd == pytest.approx(0.05)

    project_root = Path(__file__).resolve().parent.parent
    metadata = load_bug_metadata_for_run("B015", project_root=project_root, allow_test_split=False)
    fake_llm = FakeLLM(script=[{"text": "Should never be called"}])

    res = run_triage_agent(
        metadata=metadata,
        llm_client=fake_llm,
        budget_manager=b2,
        project_root=project_root,
        workspace_parent=tmp_path / "ws_budget",
    )
    assert res.status == "BUDGET_HARD_STOP"
    assert res.hard_stop_triggered is True
    assert res.llm_turns == 0
    assert len(fake_llm.calls) == 0


# ---------------------------------------------------------------------------
# 7. Split Restriction: Rejects Test Split Bugs in CLI and Runner
# ---------------------------------------------------------------------------
def test_rejects_test_split_bugs(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parent.parent

    with pytest.raises(PermissionError, match="split is 'test'"):
        load_bug_metadata_for_run("B001", project_root=project_root, allow_test_split=False)

    rc_agent = run_agent_cli_main(["--bug-id", "B001", "--ledger-path", str(tmp_path / "l.json")])
    assert rc_agent != 0

    rc_baseline = run_baseline_cli_main(["--bug-id", "B001", "--ledger-path", str(tmp_path / "l.json")])
    assert rc_baseline != 0


# ---------------------------------------------------------------------------
# 8. All 18 Bugs Prompt Integrity, Zero-Leakage & Sanitization Verification
# ---------------------------------------------------------------------------
def test_all_18_bugs_prompt_integrity_and_zero_leakage(tmp_path: Path) -> None:
    """
    Tests prompt construction, zero-leakage, and byte-exact RTL across all 18 bugs
    using internal allow_test_split=True parameter (without exposing test split on CLI).
    """
    project_root = Path(__file__).resolve().parent.parent
    with open(project_root / "labels" / "labels.json", "r", encoding="utf-8") as f:
        labels_list = json.load(f)
    assert len(labels_list) == 18

    labels_map = {entry["bug_id"]: entry for entry in labels_list}

    for bug_entry in labels_list:
        bug_id = bug_entry["bug_id"]
        meta = load_bug_metadata_for_run(bug_id, project_root=project_root, allow_test_split=True)
        ws = IsolatedWorkspace(metadata=meta, project_root=project_root, workspace_parent=tmp_path / bug_id)
        ws.setup()
        try:
            initial_sim = ws.run_visible_simulation(timeout=60)
            prompt = build_initial_user_prompt(ws, initial_sim)

            # 1. RTL raw text in initial prompt must be byte-for-byte identical to variants file
            raw_marker = f"## Initial Unpatched RTL Source (`rtl/{meta.rtl_file}` - Raw Text)\n```verilog\n"
            assert raw_marker in prompt, f"{bug_id}: raw RTL section marker missing from prompt"
            raw_part = prompt.split(raw_marker)[1].split("\n```\n")[0]

            expected_variant_path = project_root / "bugs" / "variants" / bug_id / meta.rtl_file
            expected_content = expected_variant_path.read_text(encoding="utf-8").rstrip()
            assert raw_part == expected_content, f"{bug_id}: Raw RTL in prompt is not byte-identical to variants file!"

            # 2. Prompt, sim log, and read_file must NOT contain '[REDACTED'
            assert "[REDACTED" not in prompt, f"{bug_id}: '[REDACTED' found in prompt"
            assert "[REDACTED" not in initial_sim.get("sim_log", ""), f"{bug_id}: '[REDACTED' found in sim log"
            assert "[REDACTED" not in initial_sim.get("message", ""), f"{bug_id}: '[REDACTED' found in sim message"

            executor = WorkspaceToolExecutor(workspace=ws)
            read_res = executor.read_file(f"rtl/{meta.rtl_file}")
            assert read_res["status"] == "OK"
            assert "[REDACTED" not in read_res["content"], f"{bug_id}: '[REDACTED' found in read_file content"
            assert "[REDACTED" not in read_res["numbered_content"], f"{bug_id}: '[REDACTED' found in read_file numbered_content"

            # 3. Prompt must NOT contain uppercase bug ID B0xx, 'variants', ground-truth bug_type, description, or correct_fix
            assert not re.search(r"\bB0\d\d\b", prompt), f"{bug_id}: uppercase bug ID found in prompt"
            assert "variants" not in prompt.lower(), f"{bug_id}: 'variants' found in prompt"
            assert bug_entry["bug_type"] not in prompt, f"{bug_id}: ground-truth bug_type leaked into prompt"
            assert bug_entry["description"] not in prompt, f"{bug_id}: ground-truth description leaked into prompt"
            assert bug_entry["correct_fix"] not in prompt, f"{bug_id}: ground-truth correct_fix leaked into prompt"

            # 4. Specifically for B018: read_file returns exact uncorrupted "3'b001"
            if bug_id == "B018":
                assert "3'b001" in read_res["content"], "B018: 3'b001 was corrupted in read_file content"
                assert "3'b001" in read_res["numbered_content"], "B018: 3'b001 was corrupted in read_file numbered_content"

            # 5. ALU final_patch test (e.g. for B012): applying patch restores clean RTL
            if bug_id == "B012":
                alu_fix_diff = (
                    "--- a/alu.v\n"
                    "+++ b/alu.v\n"
                    "@@ -17,3 +17,3 @@\n"
                    " \n"
                    "     // Combinational zero flag\n"
                    "-    assign flag_zero = (alu_res[3:0] == 4'b0000);\n"
                    "+    assign flag_zero = (alu_res == {DATA_WIDTH{1'b0}});\n"
                    " \n"
                )
                res_patch = executor.execute("apply_patch", {"file_path": "rtl/alu.v", "diff": alu_fix_diff})
                assert res_patch["status"] == "APPLIED"
                final_diff = ws.compute_final_diff()
                reapplied = apply_unified_diff(ws.initial_rtl_content, final_diff)
                assert reapplied.success
                clean_alu = (project_root / "rtl" / "alu.v").read_text(encoding="utf-8")
                assert reapplied.patched_text == clean_alu
        finally:
            ws.cleanup()


# ---------------------------------------------------------------------------
# 9. Baseline Runner: Calls LLM Exactly Once, No Tools, No Simulation Tool
# ---------------------------------------------------------------------------
def test_baseline_calls_model_once_without_tools(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parent.parent
    metadata = load_bug_metadata_for_run("B012", project_root=project_root, allow_test_split=False)

    valid_diff_b012 = (
        "--- a/alu.v\n"
        "+++ b/alu.v\n"
        "@@ -17,3 +17,3 @@\n"
        " \n"
        "     // Combinational zero flag\n"
        "-    assign flag_zero = (alu_res[3:0] == 4'b0000);\n"
        "+    assign flag_zero = (alu_res == {DATA_WIDTH{1'b0}});\n"
        " \n"
    )
    baseline_json = {
        "file": "rtl/alu.v",
        "line": 19,
        "bug_type": "bit_width_mismatch",
        "root_cause": "flag_zero compares only 4 bits.",
        "patch": valid_diff_b012,
        "verified": True,  # Extra field, should be ignored
    }

    fake_llm = FakeLLM(
        script=[
            {
                "text": json.dumps(baseline_json),
                "stop_reason": "end_turn",
                "input_tokens": 1100,
                "output_tokens": 180,
            }
        ]
    )
    budget = BudgetManager(ledger_path=tmp_path / "ledger_base.json")
    trace_path = tmp_path / "baseline_b012.jsonl"

    res = run_baseline_single(
        metadata=metadata,
        llm_client=fake_llm,
        budget_manager=budget,
        run_id="base_b012",
        trace_path=trace_path,
        project_root=project_root,
        workspace_parent=tmp_path / "ws_base",
    )

    assert res.status == "COMPLETED"
    assert res.llm_turns == 1
    assert res.sim_calls == 0
    assert res.patch_apply_failed is False
    assert res.parsed_output is not None
    assert res.parsed_output["file"] == "alu.v"
    assert "verified" not in res.parsed_output
    assert len(fake_llm.calls) == 1
    assert "toolConfig" not in fake_llm.calls[0]


# ---------------------------------------------------------------------------
# 10. INFRA_ERROR Run-Level Retry & Log Truncation Check
# ---------------------------------------------------------------------------
def test_infra_error_retries_and_log_truncation(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parent.parent
    metadata = load_bug_metadata_for_run("B012", project_root=project_root, allow_test_split=False)

    fake_llm = FakeLLM(
        script=[
            LLMInfraError("Simulated 503 #1"),
            LLMInfraError("Simulated 503 #2"),
            LLMInfraError("Simulated 503 #3"),
        ]
    )
    budget = BudgetManager(ledger_path=tmp_path / "ledger_infra.json")
    res = run_triage_agent(
        metadata=metadata,
        llm_client=fake_llm,
        budget_manager=budget,
        project_root=project_root,
        workspace_parent=tmp_path / "ws_infra",
    )
    assert res.status == "INFRA_ERROR"
    assert res.infra_retries_used == 2
    assert len(fake_llm.calls) == 3

    long_log = "\n".join(f"Line {i}: " + ("X" * 60) for i in range(250))
    truncated = truncate_sim_log(long_log)
    assert "TRUNCATED" in truncated
    assert len(truncated) <= 4100


# ---------------------------------------------------------------------------
# 11. Bedrock Exception Classification & Non-Retrying LLMConfigError
# ---------------------------------------------------------------------------
def test_bedrock_exception_classification_and_config_error(tmp_path: Path) -> None:
    # 1. Test classify_bedrock_exception for LLMInfraError
    infra_cases = [
        ConnectionError("Connection reset by peer"),
        TimeoutError("Request timed out"),
        MockClientError(code="ThrottlingException", status_code=429, message="Rate exceeded"),
        MockClientError(code="TooManyRequestsException", status_code=429, message="Too many requests"),
        MockClientError(code="InternalServerException", status_code=500, message="Server error"),
        MockClientError(code="ServiceUnavailableException", status_code=503, message="Service unavailable"),
        MockClientError(code="ModelTimeoutException", status_code=504, message="Model timeout"),
        # Status code 429 even with non-standard code
        MockClientError(code="UnknownThrottling", status_code=429, message="Throttled"),
        # Status code 502 even with non-standard code
        MockClientError(code="BadGateway", status_code=502, message="Bad gateway"),
    ]
    for exc in infra_cases:
        classified = classify_bedrock_exception(exc)
        assert isinstance(classified, LLMInfraError), f"Expected LLMInfraError for {exc}, got {type(classified)}"

    # 2. Test classify_bedrock_exception for LLMConfigError
    config_cases = [
        MockClientError(code="AccessDeniedException", status_code=403, message="Access denied"),
        MockClientError(code="ValidationException", status_code=400, message="Invalid maxTokens"),
        MockClientError(code="ResourceNotFoundException", status_code=404, message="Model not found"),
        type("NoCredentialsError", (Exception,), {})("Unable to locate credentials"),
        type("PartialCredentialsError", (Exception,), {})("Partial credentials"),
        type("ParamValidationError", (Exception,), {})("Parameter validation failed"),
        RuntimeError("Arbitrary unexpected runtime error"),
    ]
    for exc in config_cases:
        classified = classify_bedrock_exception(exc)
        assert isinstance(classified, LLMConfigError), f"Expected LLMConfigError for {exc}, got {type(classified)}"

    # 3. Test that run_triage_agent does NOT retry on LLMConfigError
    project_root = Path(__file__).resolve().parent.parent
    metadata = load_bug_metadata_for_run("B012", project_root=project_root, allow_test_split=False)
    fake_llm_cfg = FakeLLM(
        script=[
            MockClientError(code="AccessDeniedException", status_code=403, message="IAM blocked"),
            # If retry occurs, this second step would be consumed
            {"text": "Should never be called"},
        ]
    )
    budget = BudgetManager(ledger_path=tmp_path / "ledger_cfg.json")
    res_agent = run_triage_agent(
        metadata=metadata,
        llm_client=fake_llm_cfg,
        budget_manager=budget,
        project_root=project_root,
        workspace_parent=tmp_path / "ws_cfg",
    )
    assert res_agent.status == "CONFIG_ERROR"
    assert "AccessDeniedException" in res_agent.stop_reason
    assert res_agent.infra_retries_used == 0
    assert len(fake_llm_cfg.calls) == 1

    # 4. Test that run_baseline_single does NOT retry on LLMConfigError
    fake_llm_base_cfg = FakeLLM(
        script=[
            MockClientError(code="ValidationException", status_code=400, message="Bad param"),
        ]
    )
    res_base = run_baseline_single(
        metadata=metadata,
        llm_client=fake_llm_base_cfg,
        budget_manager=budget,
        project_root=project_root,
        workspace_parent=tmp_path / "ws_base_cfg",
    )
    assert res_base.status == "CONFIG_ERROR"
    assert "ValidationException" in res_base.stop_reason
    assert len(fake_llm_base_cfg.calls) == 1
