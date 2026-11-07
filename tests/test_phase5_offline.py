#!/usr/bin/env python3
"""
RTL Failure Triage Agent - Phase 5 Offline Test Suite
Tests evaluation infrastructure without calling live LLMs or Bedrock:
1. Exact execution count: bugs x configs x repeats.
2. Split guard: split='test' requires --confirm-test-run.
3. Resumption: skips completed runs without re-running or overwriting.
4. Gate 4 integrity: tampering with tracked files aborts execution.
5. Hard stop: stops entire experiment; incomplete repeats report INCOMPLETE for Decision Rule.
6. Score distinction: correct patch passes heldout, overfit patch fails heldout, FAILED gets 0 points.
7. Real B005 agent trace: scores visible PASS and held-out FAIL.
"""

import json
from pathlib import Path
import pytest
import shutil
import socket
from typing import Any

from agent.budget import BudgetManager, compute_call_cost_usd, PRICING_STATUS_NOTE
from agent.llm import BedrockClient, FakeLLM, LLMResponse, ToolCall
from eval.run_experiment import (
    generate_gate4_hashes,
    is_run_already_completed,
    run_experiment,
    verify_gate4_hashes,
)
from eval.score import (
    SingleRunScore,
    aggregate_scores,
    evaluate_patch_on_testbenches,
    load_labels_map,
    main as score_main,
    score_single_trace,
)


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blocks external network calls to guarantee purely offline tests."""
    def guarded_socket(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("External network connection attempted in offline test suite!")

    monkeypatch.setattr(socket, "socket", guarded_socket)


# ---------------------------------------------------------------------------
# 1. Exact Execution Count & Repeat Progression
# ---------------------------------------------------------------------------
def test_experiment_run_count_and_repeat_progression(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parent.parent

    # Prepare FakeLLM script for 2 bugs x 2 configs x 2 repeats = 8 runs
    # Agent: 1 turn returning final answer. Baseline: 1 turn.
    valid_answer = json.dumps({
        "file": "alu.v",
        "line": 19,
        "bug_type": "bit_width_mismatch",
        "root_cause": "Test explanation",
        "patch": "--- a/alu.v\n+++ b/alu.v\n@@ -19,1 +19,1 @@\n",
        "verified": False,
    })

    script = [{"text": valid_answer, "input_tokens": 100, "output_tokens": 50} for _ in range(8)]
    fake_llm = FakeLLM(script=script)

    out_dir = tmp_path / "exp_runs"
    ledger_file = tmp_path / "test_ledger.json"

    ret = run_experiment(
        split="dev",
        confirm_test_run=False,
        repeats=2,
        configs=["baseline", "agent"],
        bugs=["B012", "B015"],
        run_id="test_exp",
        output_base_dir=out_dir,
        ledger_path=ledger_file,
        hard_stop_usd=15.0,
        llm_client=fake_llm,
        project_root=project_root,
        skip_preflight=True,
    )

    assert ret == 0
    assert len(fake_llm.calls) == 8

    # Verify directory structure: results/<run_id>/<config>/<bug_id>/<repeat>/trace.jsonl
    expected_traces = [
        out_dir / "baseline" / "B012" / "1" / "trace.jsonl",
        out_dir / "baseline" / "B015" / "1" / "trace.jsonl",
        out_dir / "agent" / "B012" / "1" / "trace.jsonl",
        out_dir / "agent" / "B015" / "1" / "trace.jsonl",
        out_dir / "baseline" / "B012" / "2" / "trace.jsonl",
        out_dir / "baseline" / "B015" / "2" / "trace.jsonl",
        out_dir / "agent" / "B012" / "2" / "trace.jsonl",
        out_dir / "agent" / "B015" / "2" / "trace.jsonl",
    ]
    for tp in expected_traces:
        assert tp.is_file(), f"Missing trace: {tp}"

    # Manifest verification
    manifest_path = out_dir / "manifest.json"
    assert manifest_path.is_file()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["completed"] is True
    assert manifest["repeats"] == 2
    assert len(manifest["sha256_manifest_table"]) > 20


# ---------------------------------------------------------------------------
# 2. Split Guard: Test Split Without Confirmation Flag is Rejected
# ---------------------------------------------------------------------------
def test_test_split_without_confirmation_rejected(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parent.parent
    fake_llm = FakeLLM(script=[])

    ret_no_confirm = run_experiment(
        split="test",
        confirm_test_run=False,
        repeats=1,
        configs=["agent"],
        bugs=["B001"],
        run_id="test_guard",
        output_base_dir=tmp_path / "exp_guard",
        ledger_path=tmp_path / "ledger_guard.json",
        llm_client=fake_llm,
        project_root=project_root,
        skip_preflight=True,
    )
    assert ret_no_confirm != 0
    assert len(fake_llm.calls) == 0


# ---------------------------------------------------------------------------
# 3. Resumption: Skips Completed Runs
# ---------------------------------------------------------------------------
def test_resumption_skips_completed_runs(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parent.parent
    out_dir = tmp_path / "exp_resume"

    # Pre-create completed trace for baseline/B012/1
    pre_trace = out_dir / "baseline" / "B012" / "1" / "trace.jsonl"
    pre_trace.parent.mkdir(parents=True, exist_ok=True)
    with open(pre_trace, "w", encoding="utf-8") as f:
        f.write(json.dumps({"event_type": "run_start", "run_id": "pre", "config": "baseline"}) + "\n")
        f.write(json.dumps({"event_type": "run_end", "run_id": "pre", "status": "COMPLETED", "llm_turns": 1}) + "\n")

    valid_answer = json.dumps({
        "file": "counter.v",
        "line": 26,
        "bug_type": "missing_branch",
        "root_cause": "Test",
        "patch": "",
        "verified": False,
    })
    # We run 2 bugs: B012 (already completed) and B015 (not completed).
    # FakeLLM should only be called once for B015!
    fake_llm = FakeLLM(script=[{"text": valid_answer, "input_tokens": 100, "output_tokens": 50}])

    ret = run_experiment(
        split="dev",
        confirm_test_run=False,
        repeats=1,
        configs=["baseline"],
        bugs=["B012", "B015"],
        run_id="exp_resume",
        output_base_dir=out_dir,
        ledger_path=tmp_path / "ledger_resume.json",
        llm_client=fake_llm,
        project_root=project_root,
        skip_preflight=True,
    )

    assert ret == 0
    assert len(fake_llm.calls) == 1  # Only B015 was called

    # Check that pre_trace content was NOT overwritten
    with open(pre_trace, "r", encoding="utf-8") as f:
        content = f.read()
    assert "pre" in content


# ---------------------------------------------------------------------------
# 4. Gate 4 Hash Verification
# ---------------------------------------------------------------------------
def test_gate4_hashes_integrity_and_tamper_detection(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parent.parent

    # Verify current repo gate4 hashes
    ok, errs = verify_gate4_hashes(project_root)
    assert ok is True, f"Gate 4 verification failed: {errs}"

    # Create dummy environment to test tamper detection
    dummy_repo = tmp_path / "dummy_repo"
    (dummy_repo / "agent").mkdir(parents=True)
    (dummy_repo / "eval").mkdir(parents=True)
    (dummy_repo / "docs").mkdir(parents=True)

    dummy_file = dummy_repo / "eval" / "test_file.py"
    dummy_file.write_text("# original content\n", encoding="utf-8")

    generate_gate4_hashes(dummy_repo, dummy_repo / "docs" / "gate4_hashes.sha256")
    ok_before, _ = verify_gate4_hashes(dummy_repo)
    assert ok_before is True

    # Tamper with file
    dummy_file.write_text("# modified content\n", encoding="utf-8")
    ok_after, errs_after = verify_gate4_hashes(dummy_repo)
    assert ok_after is False
    assert any("HASH MISMATCH" in e for e in errs_after)


# ---------------------------------------------------------------------------
# 5. Budget Hard Stop & Incomplete Decision Rule
# ---------------------------------------------------------------------------
def test_budget_hard_stop_and_incomplete_decision_rule(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parent.parent
    out_dir = tmp_path / "exp_hardstop"
    ledger_file = tmp_path / "ledger_hardstop.json"

    # Pre-record high usage into ledger to trigger hard stop immediately
    # At Haiku rate ($1.00/1M in): 50,000 * 1.00 / 1M = $0.05 > $0.01 hard stop
    b = BudgetManager(ledger_path=ledger_file, hard_stop_usd=0.01)
    b.record_usage(input_tokens=50_000, output_tokens=0, run_id="prior", config="agent", split="dev", turn=1)

    fake_llm = FakeLLM(script=[{"text": "{}", "input_tokens": 100, "output_tokens": 50}])

    ret = run_experiment(
        split="dev",
        confirm_test_run=False,
        repeats=1,
        configs=["agent"],
        bugs=["B012"],
        run_id="exp_hardstop",
        output_base_dir=out_dir,
        ledger_path=ledger_file,
        hard_stop_usd=0.01,
        llm_client=fake_llm,
        project_root=project_root,
        skip_preflight=True,
    )
    assert ret != 0  # Aborted on hard stop

    # Verify Decision Rule when repeats are incomplete
    summary = aggregate_scores([])
    dr = summary["decision_rule"]
    assert dr["status"] == "INCOMPLETE"
    assert "Thực nghiệm chưa hoàn tất" in dr["message"]


# ---------------------------------------------------------------------------
# 6. Score Distinction: Correct vs Overfit vs FAILED
# ---------------------------------------------------------------------------
def test_score_distinction_correct_vs_overfit_and_failed() -> None:
    project_root = Path(__file__).resolve().parent.parent

    # 1. Correct patch for B012: alu zero flag width
    correct_patch_b012 = (
        "--- a/alu.v\n"
        "+++ b/alu.v\n"
        "@@ -19,1 +19,1 @@\n"
        "-    assign flag_zero = (alu_res[3:0] == 4'b0000);\n"
        "+    assign flag_zero = (alu_res == {DATA_WIDTH{1'b0}});\n"
    )
    b012_variant = (project_root / "bugs" / "variants" / "B012" / "alu.v").read_text(encoding="utf-8")
    app1, vis1, held1 = evaluate_patch_on_testbenches(
        project_root=project_root,
        module="alu",
        variant_rtl_content=b012_variant,
        patch=correct_patch_b012,
    )
    assert app1 is True
    assert vis1 is True
    assert held1 is True

    # 2. Overfit patch for B005 (from fixture): passes visible, fails heldout
    fixture_diff = project_root / "tests" / "fixtures" / "agent_B005_haiku_patch.diff"
    assert fixture_diff.is_file(), f"Missing fixture diff: {fixture_diff}"
    overfit_patch_b005 = fixture_diff.read_text(encoding="utf-8")

    b005_variant = (project_root / "bugs" / "variants" / "B005" / "fsm_controller.v").read_text(encoding="utf-8")
    app2, vis2, held2 = evaluate_patch_on_testbenches(
        project_root=project_root,
        module="fsm_controller",
        variant_rtl_content=b005_variant,
        patch=overfit_patch_b005,
    )
    assert app2 is True
    assert vis2 is True
    assert held2 is False  # Heldout fails due to overfitting!

    # 3. FAILED run receives 0 points across all patch and diagnostic metrics
    failed_trace = Path("/tmp") / "test_failed_trace_r1.jsonl"
    with open(failed_trace, "w", encoding="utf-8") as f:
        f.write(json.dumps({"event_type": "run_start", "run_id": "r_fail", "config": "agent", "bug_id": "B012", "repeat": 1}) + "\n")
        f.write(json.dumps({
            "event_type": "run_end",
            "run_id": "r_fail",
            "config": "agent",
            "bug_id": "B012",
            "repeat": 1,
            "status": "FAILED",
            "stop_reason": "parse_failed",
            "final_patch": correct_patch_b012,
        }) + "\n")

    labels_map = load_labels_map(project_root)
    score_failed = score_single_trace(failed_trace, labels_map, project_root)
    assert score_failed.status == "FAILED"
    assert score_failed.pass_vis == 0
    assert score_failed.pass_heldout == 0
    assert score_failed.pass_first == 0
    assert score_failed.is_correct_loc == 0
    assert score_failed.is_correct_cls == 0
    if failed_trace.is_file():
        failed_trace.unlink()


# ---------------------------------------------------------------------------
# 7. Real Trace Evaluation: agent_B005 Visible PASS, Held-out FAIL (Fixture)
# ---------------------------------------------------------------------------
def test_real_agent_b005_trace_scoring() -> None:
    project_root = Path(__file__).resolve().parent.parent
    trace_b005 = project_root / "tests" / "fixtures" / "agent_B005_haiku_trace.jsonl"
    if not trace_b005.is_file():
        # Fallback to dev_runs if fixture missing
        trace_b005 = project_root / "results" / "dev_runs" / "agent_B005_trace.jsonl"
    assert trace_b005.is_file(), f"Trace file not found: {trace_b005}"

    labels_map = load_labels_map(project_root)
    score = score_single_trace(
        trace_path=trace_b005,
        labels_map=labels_map,
        project_root=project_root,
        override_repeat=1,
    )

    assert score.bug_id == "B005"
    assert score.config == "agent"
    assert score.status == "COMPLETED"
    assert score.is_correct_loc == 1  # Line 45 is within [45, 49]
    assert score.is_correct_cls == 1  # fsm_invalid_transition
    assert score.pass_vis == 1        # Visible testbench passes
    assert score.pass_heldout == 0    # Held-out testbench FAILS (overfitting)
    assert score.pass_first == 0


# ---------------------------------------------------------------------------
# 8. Budget Manager Mixed-Rate Ledger (Haiku Legacy vs Nova Pricing)
# ---------------------------------------------------------------------------
def test_budget_manager_mixed_pricing_ledger(tmp_path: Path) -> None:
    ledger_path = tmp_path / "mixed_ledger.json"

    # Step 1: Initial ledger with a Haiku call (legacy rate $1.00 / $5.00)
    bm = BudgetManager(ledger_path=ledger_path, hard_stop_usd=15.0)
    haiku_cost = bm.record_usage(
        input_tokens=1000,
        output_tokens=100,
        run_id="run_haiku",
        config="agent",
        split="dev",
        turn=1,
        model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
    )
    # Haiku: (1000 * 1.0 + 100 * 5.0) / 1,000,000 = (1000 + 500) / 1M = $0.0015
    assert haiku_cost == 0.0015
    assert bm.total_cost_usd == 0.0015
    assert bm.total_calls == 1

    # Step 2: Record a Nova call (new rate $0.30 / $2.50)
    nova_cost = bm.record_usage(
        input_tokens=1000,
        output_tokens=100,
        run_id="run_nova",
        config="agent",
        split="dev",
        turn=1,
        model_id="global.amazon.nova-2-lite-v1:0",
    )
    # Nova: (1000 * 0.30 + 100 * 2.50) / 1,000,000 = (300 + 250) / 1M = $0.00055
    assert nova_cost == 0.00055
    # Total cumulative: 0.0015 + 0.00055 = 0.00205
    assert bm.total_cost_usd == 0.00205
    assert bm.total_calls == 2

    # Step 3: Reload from disk and verify neither entry is altered or recomputed
    bm_reloaded = BudgetManager(ledger_path=ledger_path, hard_stop_usd=15.0)
    assert bm_reloaded.total_cost_usd == 0.00205
    assert bm_reloaded.total_calls == 2
    assert len(bm_reloaded.entries) == 2
    assert bm_reloaded.entries[0]["cost_usd"] == 0.0015
    assert bm_reloaded.entries[1]["cost_usd"] == 0.00055

    # Check payload header contains pricing status note
    raw_json = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert raw_json["pricing_status_note"] == PRICING_STATUS_NOTE
    assert raw_json["input_price_per_million_usd"] == 1.00
    assert raw_json["output_price_per_million_usd"] == 5.00


# ---------------------------------------------------------------------------
# 9. Safe Response Handling: Non-text/toolUse Content Blocks Filtered Out
# ---------------------------------------------------------------------------
def test_safe_response_handling_ignores_non_text_tooluse_blocks(tmp_path: Path) -> None:
    """Verifies that unexpected blocks (e.g. reasoningContent) never enter conversation history."""
    from agent.loop import run_triage_agent
    from agent.workspace import load_bug_metadata_for_run

    project_root = Path(__file__).resolve().parent.parent
    bug_meta = load_bug_metadata_for_run("B012", project_root=project_root, allow_test_split=False)

    # 1. Test BedrockClient safe block filtering on raw API responses
    class MockBedrockRuntimeClient:
        def converse(self, **kwargs):
            return {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"reasoningContent": {"reasoningText": {"text": "internal thought"}}},
                            {"text": "safe response"},
                            {"toolUse": {"toolUseId": "tu_mock", "name": "read_file", "input": {"path": "rtl/alu.v"}}},
                        ],
                    }
                },
                "usage": {"inputTokens": 100, "outputTokens": 50},
                "stopReason": "tool_use",
            }

    bedrock_client = BedrockClient(boto_client=MockBedrockRuntimeClient())
    llm_resp = bedrock_client.converse(system_prompt="sys", messages=[])
    assert llm_resp.content_text == "safe response"
    assert len(llm_resp.tool_calls) == 1
    raw_keys = [list(b.keys())[0] for b in llm_resp.raw_assistant_message["content"]]
    assert "reasoningContent" not in raw_keys
    assert raw_keys == ["text", "toolUse"]

    # 2. Test multi-turn loop with FakeLLM emitting unexpected blocks
    turn1_raw_blocks = [
        {"reasoningContent": {"reasoningText": {"text": "Internal hidden reasoning chain"}}},
        {"text": "Investigating ALU carry calculation."},
        {
            "toolUse": {
                "toolUseId": "tu_safe_1",
                "name": "read_file",
                "input": {"path": "rtl/alu.v", "start_line": 1, "end_line": 15},
            }
        },
    ]

    turn2_text = json.dumps({
        "file": "alu.v",
        "line": 19,
        "bug_type": "bit_width_mismatch",
        "root_cause": "Fixed carry",
        "patch": "--- a/alu.v\n+++ b/alu.v\n@@ -19,1 +19,1 @@\n",
        "verified": True,
    })

    script = [
        {"raw_content_blocks": turn1_raw_blocks, "stop_reason": "tool_use", "input_tokens": 100, "output_tokens": 20},
        {"text": turn2_text, "stop_reason": "end_turn", "input_tokens": 150, "output_tokens": 50},
    ]
    fake_llm = FakeLLM(script=script)
    bm = BudgetManager(ledger_path=tmp_path / "test_ledger.json")

    result = run_triage_agent(
        metadata=bug_meta,
        llm_client=fake_llm,
        budget_manager=bm,
        project_root=project_root,
        workspace_parent=tmp_path / "ws_safe",
        trace_path=tmp_path / "safe_trace.jsonl",
    )

    assert result.status == "COMPLETED"
    assert len(fake_llm.calls) == 2

    # Check the second call's messages payload: the assistant message from turn 1 MUST NOT
    # have any "reasoningContent" block!
    call2_messages = fake_llm.calls[1]["messages"]
    assistant_msg = next(m for m in call2_messages if m["role"] == "assistant")
    block_keys = [list(b.keys())[0] for b in assistant_msg["content"]]
    assert "reasoningContent" not in block_keys
    assert all(k in ("text", "toolUse") for k in block_keys)


# ---------------------------------------------------------------------------
# 10. Malformed Model Output Retry Rule: Agent Recovers on Retry
# ---------------------------------------------------------------------------
def test_malformed_model_output_retry_then_completed_agent(tmp_path: Path) -> None:
    """Verifies that 1 malformed output triggers retry with identical request and completes."""
    from agent.loop import run_triage_agent
    from agent.workspace import load_bug_metadata_for_run

    project_root = Path(__file__).resolve().parent.parent
    bug_meta = load_bug_metadata_for_run("B012", project_root=project_root, allow_test_split=False)

    valid_json = json.dumps({
        "file": "alu.v",
        "line": 19,
        "bug_type": "bit_width_mismatch",
        "root_cause": "Fixed carry computation",
        "patch": "--- a/alu.v\n+++ b/alu.v\n@@ -19,1 +19,1 @@\n",
        "verified": True,
    })

    # Call 1: malformed_model_output (usage 0/0, no text, no tool call)
    # Call 2: valid JSON answer
    script = [
        {"text": "", "stop_reason": "malformed_model_output", "input_tokens": 0, "output_tokens": 0},
        {"text": valid_json, "stop_reason": "end_turn", "input_tokens": 100, "output_tokens": 50},
    ]
    fake_llm = FakeLLM(script=script)
    bm = BudgetManager(ledger_path=tmp_path / "test_ledger_malformed1.json")

    result = run_triage_agent(
        metadata=bug_meta,
        llm_client=fake_llm,
        budget_manager=bm,
        project_root=project_root,
        workspace_parent=tmp_path / "ws_malformed1",
        trace_path=tmp_path / "trace_malformed1.jsonl",
    )

    assert result.status == "COMPLETED"
    assert result.stop_reason == "completed"
    assert result.llm_turns == 2
    assert result.malformed_output_retries == 1
    assert len(fake_llm.calls) == 2

    # Verify both calls sent the exact same messages without appending anything
    call1_messages = fake_llm.calls[0]["messages"]
    call2_messages = fake_llm.calls[1]["messages"]
    assert call1_messages == call2_messages

    # Verify trace contains 2 llm_turn events and 1 run_end with malformed_output_retries=1
    llm_events = [e for e in result.trace_events if e.get("event_type") == "llm_turn"]
    assert len(llm_events) == 2
    assert llm_events[0]["stop_reason"] == "malformed_model_output"
    assert llm_events[1]["stop_reason"] == "end_turn"
    end_events = [e for e in result.trace_events if e.get("event_type") == "run_end"]
    assert end_events[0]["malformed_output_retries"] == 1


# ---------------------------------------------------------------------------
# 11. Malformed Model Output Retry Rule: Consecutive 3 Fails Agent
# ---------------------------------------------------------------------------
def test_malformed_model_output_consecutive_3_fails_agent(tmp_path: Path) -> None:
    """Verifies that 3 consecutive malformed outputs result in FAILED with stop_reason malformed_model_output."""
    from agent.loop import run_triage_agent
    from agent.workspace import load_bug_metadata_for_run

    project_root = Path(__file__).resolve().parent.parent
    bug_meta = load_bug_metadata_for_run("B012", project_root=project_root, allow_test_split=False)

    # 3 consecutive malformed calls
    script = [
        {"text": "", "stop_reason": "malformed_model_output", "input_tokens": 0, "output_tokens": 0},
        {"text": "", "stop_reason": "malformed_model_output", "input_tokens": 0, "output_tokens": 0},
        {"text": "", "stop_reason": "malformed_model_output", "input_tokens": 0, "output_tokens": 0},
    ]
    fake_llm = FakeLLM(script=script)
    bm = BudgetManager(ledger_path=tmp_path / "test_ledger_malformed3.json")

    result = run_triage_agent(
        metadata=bug_meta,
        llm_client=fake_llm,
        budget_manager=bm,
        project_root=project_root,
        workspace_parent=tmp_path / "ws_malformed3",
        trace_path=tmp_path / "trace_malformed3.jsonl",
    )

    assert result.status == "FAILED"
    assert result.stop_reason == "malformed_model_output"
    assert result.llm_turns == 3
    assert result.malformed_output_retries == 2
    assert len(fake_llm.calls) == 3

    # Verify all 3 calls sent the exact same messages
    assert fake_llm.calls[0]["messages"] == fake_llm.calls[1]["messages"]
    assert fake_llm.calls[1]["messages"] == fake_llm.calls[2]["messages"]


# ---------------------------------------------------------------------------
# 12. Malformed Model Output Retry Rule: Baseline Configuration
# ---------------------------------------------------------------------------
def test_malformed_model_output_baseline(tmp_path: Path) -> None:
    """Verifies baseline follows the exact same malformed_model_output retry rule."""
    from eval.run_baseline import run_baseline_single
    from agent.workspace import load_bug_metadata_for_run

    project_root = Path(__file__).resolve().parent.parent
    bug_meta = load_bug_metadata_for_run("B012", project_root=project_root, allow_test_split=False)

    valid_json = json.dumps({
        "file": "alu.v",
        "line": 19,
        "bug_type": "bit_width_mismatch",
        "root_cause": "Fixed carry",
        "patch": "--- a/alu.v\n+++ b/alu.v\n@@ -19,1 +19,1 @@\n",
    })

    # Case 1: 1 malformed then valid JSON -> COMPLETED
    script_recover = [
        {"text": "", "stop_reason": "malformed_model_output", "input_tokens": 0, "output_tokens": 0},
        {"text": valid_json, "stop_reason": "end_turn", "input_tokens": 100, "output_tokens": 50},
    ]
    llm1 = FakeLLM(script=script_recover)
    bm1 = BudgetManager(ledger_path=tmp_path / "bm_base1.json")
    res1 = run_baseline_single(
        metadata=bug_meta,
        llm_client=llm1,
        budget_manager=bm1,
        project_root=project_root,
        workspace_parent=tmp_path / "ws_base1",
        trace_path=tmp_path / "trace_base1.jsonl",
    )
    assert res1.status == "COMPLETED"
    assert res1.llm_turns == 2
    assert res1.malformed_output_retries == 1
    assert len(llm1.calls) == 2
    assert llm1.calls[0]["messages"] == llm1.calls[1]["messages"]

    # Case 2: 3 consecutive malformed -> FAILED with stop_reason malformed_model_output
    script_fail = [
        {"text": "", "stop_reason": "malformed_model_output", "input_tokens": 0, "output_tokens": 0},
        {"text": "", "stop_reason": "malformed_model_output", "input_tokens": 0, "output_tokens": 0},
        {"text": "", "stop_reason": "malformed_model_output", "input_tokens": 0, "output_tokens": 0},
    ]
    llm2 = FakeLLM(script=script_fail)
    bm2 = BudgetManager(ledger_path=tmp_path / "bm_base2.json")
    res2 = run_baseline_single(
        metadata=bug_meta,
        llm_client=llm2,
        budget_manager=bm2,
        project_root=project_root,
        workspace_parent=tmp_path / "ws_base2",
        trace_path=tmp_path / "trace_base2.jsonl",
    )
    assert res2.status == "FAILED"
    assert res2.stop_reason == "malformed_model_output"
    assert res2.llm_turns == 3
    assert res2.malformed_output_retries == 2
    assert len(llm2.calls) == 3
    assert llm2.calls[0]["messages"] == llm2.calls[1]["messages"]
    assert llm2.calls[1]["messages"] == llm2.calls[2]["messages"]


# ---------------------------------------------------------------------------
# 13. Score: Missing Bug Incomplete & Missing Runs List
# ---------------------------------------------------------------------------
def test_score_completeness_and_missing_bug_incomplete() -> None:
    """Verifies that missing an entire bug across repeats marks Decision Rule as INCOMPLETE and lists missing runs."""
    project_root = Path(__file__).resolve().parent.parent
    labels_map = load_labels_map(project_root)
    # dev bugs are B005, B012, B015
    # Build complete runs for B005 and B012 only (missing B015)
    scores: list[SingleRunScore] = []
    for cfg in ("baseline", "agent"):
        for bug in ("B005", "B012"):
            for rep in (1, 2, 3):
                scores.append(SingleRunScore(
                    run_id=f"exp_{cfg}_{bug}_r{rep}",
                    config=cfg,
                    bug_id=bug,
                    repeat=rep,
                    module="alu" if bug != "B005" else "traffic_light",
                    status="COMPLETED",
                    stop_reason="completed",
                    is_valid_run=True,
                    is_infra_error=False,
                    is_correct_loc=1,
                    is_correct_cls=1,
                    pass_vis=1,
                    pass_heldout=1,
                    pass_first=1,
                    patch_apply_failed=0,
                    llm_turns=1,
                    sim_calls=1,
                    input_tokens=100,
                    output_tokens=50,
                    cost_usd=0.001,
                ))

    summary = aggregate_scores(scores, labels_map=labels_map)
    dr = summary["decision_rule"]
    assert dr["status"] == "INCOMPLETE"
    assert dr["passed_all_repeats"] is False
    assert dr["deltas"] == {}
    # B015 is missing for both baseline and agent across r1, r2, r3 = 6 missing runs
    missing = dr["missing_runs"]
    assert len(missing) == 6
    for cfg in ("baseline", "agent"):
        for rep in (1, 2, 3):
            assert f"{cfg}/B015/r{rep}" in missing


# ---------------------------------------------------------------------------
# 14. Score: Duplicate Trace Rejection (raises ValueError and exits != 0)
# ---------------------------------------------------------------------------
def test_score_duplicate_trace_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that duplicate traces for the same (config, bug, repeat) raise ValueError and cause CLI exit != 0."""
    project_root = Path(__file__).resolve().parent.parent
    labels_map = load_labels_map(project_root)

    s1 = SingleRunScore(
        run_id="run_1",
        config="agent",
        bug_id="B005",
        repeat=1,
        module="traffic_light",
        status="COMPLETED",
        stop_reason="completed",
        is_valid_run=True,
        is_infra_error=False,
        is_correct_loc=1,
        is_correct_cls=1,
        pass_vis=1,
        pass_heldout=1,
        pass_first=1,
        patch_apply_failed=0,
        llm_turns=1,
        sim_calls=1,
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
    )
    s2 = SingleRunScore(
        run_id="run_2_dup",
        config="agent",
        bug_id="B005",
        repeat=1,
        module="traffic_light",
        status="COMPLETED",
        stop_reason="completed",
        is_valid_run=True,
        is_infra_error=False,
        is_correct_loc=1,
        is_correct_cls=1,
        pass_vis=1,
        pass_heldout=1,
        pass_first=1,
        patch_apply_failed=0,
        llm_turns=1,
        sim_calls=1,
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
    )

    with pytest.raises(ValueError, match="Duplicate traces detected"):
        aggregate_scores([s1, s2], labels_map=labels_map)

    # CLI test: write two traces in separate directories that resolve to the same run
    d1 = tmp_path / "dir1"
    d2 = tmp_path / "dir2"
    d1.mkdir()
    d2.mkdir()
    t1 = d1 / "trace.jsonl"
    t2 = d2 / "trace.jsonl"
    line_start = json.dumps({"event_type": "run_start", "config": "agent", "bug_id": "B005", "repeat": 1, "run_id": "r1"}) + "\n"
    line_end = json.dumps({"event_type": "run_end", "config": "agent", "bug_id": "B005", "repeat": 1, "run_id": "r1", "status": "COMPLETED"}) + "\n"
    t1.write_text(line_start + line_end, encoding="utf-8")
    t2.write_text(line_start + line_end, encoding="utf-8")

    monkeypatch.setattr("sys.argv", ["score.py", "-d", str(tmp_path)])
    ret = score_main()
    assert ret != 0


# ---------------------------------------------------------------------------
# 15. Score: Mixed Splits Detection Rejection
# ---------------------------------------------------------------------------
def test_score_mixed_splits_rejected() -> None:
    """Verifies that traces mixing dev and test bugs raise ValueError."""
    project_root = Path(__file__).resolve().parent.parent
    labels_map = load_labels_map(project_root)

    s_dev = SingleRunScore(
        run_id="run_dev",
        config="agent",
        bug_id="B005",  # dev
        repeat=1,
        module="traffic_light",
        status="COMPLETED",
        stop_reason="completed",
        is_valid_run=True,
        is_infra_error=False,
        is_correct_loc=1,
        is_correct_cls=1,
        pass_vis=1,
        pass_heldout=1,
        pass_first=1,
        patch_apply_failed=0,
        llm_turns=1,
        sim_calls=1,
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
    )
    s_test = SingleRunScore(
        run_id="run_test",
        config="agent",
        bug_id="B001",  # test
        repeat=1,
        module="alu",
        status="COMPLETED",
        stop_reason="completed",
        is_valid_run=True,
        is_infra_error=False,
        is_correct_loc=1,
        is_correct_cls=1,
        pass_vis=1,
        pass_heldout=1,
        pass_first=1,
        patch_apply_failed=0,
        llm_turns=1,
        sim_calls=1,
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
    )

    with pytest.raises(ValueError, match="Mixed splits detected"):
        aggregate_scores([s_dev, s_test], labels_map=labels_map)


# ---------------------------------------------------------------------------
# 16. Score: No Implicit Defaults for bug_id or repeat
# ---------------------------------------------------------------------------
def test_score_no_implicit_defaults(tmp_path: Path) -> None:
    """Verifies that missing bug_id or missing/invalid repeat raises ValueError without defaulting."""
    project_root = Path(__file__).resolve().parent.parent
    labels_map = load_labels_map(project_root)

    # 1. Missing bug_id
    t1 = tmp_path / "unknown_trace.jsonl"
    t1.write_text(json.dumps({"event_type": "run_end", "config": "agent", "repeat": 1, "status": "COMPLETED"}) + "\n")
    with pytest.raises(ValueError, match="Cannot determine bug_id"):
        score_single_trace(t1, labels_map=labels_map, project_root=project_root)

    # 2. Missing repeat
    t2 = tmp_path / "agent_B005_norepeat.jsonl"
    t2.write_text(json.dumps({"event_type": "run_end", "config": "agent", "bug_id": "B005", "status": "COMPLETED"}) + "\n")
    with pytest.raises(ValueError, match="Cannot determine valid repeat"):
        score_single_trace(t2, labels_map=labels_map, project_root=project_root)

    # 3. Invalid repeat (e.g. repeat=4)
    t3 = tmp_path / "agent_B005_rep4.jsonl"
    t3.write_text(json.dumps({"event_type": "run_end", "config": "agent", "bug_id": "B005", "repeat": 4, "status": "COMPLETED"}) + "\n")
    with pytest.raises(ValueError, match="Cannot determine valid repeat"):
        score_single_trace(t3, labels_map=labels_map, project_root=project_root)


# ---------------------------------------------------------------------------
# 17. Score: Empty Group Keys Match Non-Empty Group Keys
# ---------------------------------------------------------------------------
def test_score_empty_group_keys_match_non_empty() -> None:
    """Verifies summarize_group for empty runs returns identical keys to non-empty runs."""
    project_root = Path(__file__).resolve().parent.parent
    labels_map = load_labels_map(project_root)

    s = SingleRunScore(
        run_id="run_1",
        config="agent",
        bug_id="B005",
        repeat=1,
        module="traffic_light",
        status="COMPLETED",
        stop_reason="completed",
        is_valid_run=True,
        is_infra_error=False,
        is_correct_loc=1,
        is_correct_cls=1,
        pass_vis=1,
        pass_heldout=1,
        pass_first=1,
        patch_apply_failed=0,
        llm_turns=1,
        sim_calls=1,
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
    )
    summary = aggregate_scores([s], labels_map=labels_map)

    # agent repeat_1 is non-empty; agent repeat_2 is empty
    r1_keys = set(summary["by_config"]["agent"]["repeat_1"].keys())
    r2_keys = set(summary["by_config"]["agent"]["repeat_2"].keys())
    assert r1_keys == r2_keys
    empty_r2 = summary["by_config"]["agent"]["repeat_2"]
    assert empty_r2["n_valid"] == 0
    assert empty_r2["acc_loc_count"] == 0
    assert empty_r2["acc_loc_rate"] == 0.0
    assert empty_r2["pass_heldout_count"] == 0
    assert empty_r2["pass_heldout_rate"] == 0.0


# ---------------------------------------------------------------------------
# 18. run_experiment: --skip-preflight with --split test Rejected
# ---------------------------------------------------------------------------
def test_run_experiment_skip_preflight_on_test_rejected() -> None:
    """Verifies that running test split with skip_preflight is strictly rejected."""
    ret = run_experiment(
        split="test",
        skip_preflight=True,
        confirm_test_run=True,
    )
    assert ret == 1


# ---------------------------------------------------------------------------
# 19. run_experiment: --hard-stop > 15.0 USD Rejected
# ---------------------------------------------------------------------------
def test_run_experiment_hard_stop_exceeds_15_rejected() -> None:
    """Verifies that hard_stop_usd > 15.0 is rejected."""
    ret = run_experiment(
        split="dev",
        hard_stop_usd=15.01,
    )
    assert ret == 1


# ---------------------------------------------------------------------------
# 20. run_experiment: Resume INFRA_ERROR and Incomplete Trace Renaming
# ---------------------------------------------------------------------------
def test_run_experiment_resume_infra_error_and_incomplete_trace(tmp_path: Path) -> None:
    """Verifies INFRA_ERROR is treated as completed, and incomplete trace is renamed without deleting."""
    run_dir = tmp_path / "agent" / "B005" / "1"
    run_dir.mkdir(parents=True)
    trace_file = run_dir / "trace.jsonl"

    # Case A: INFRA_ERROR trace is completed
    trace_file.write_text(
        json.dumps({"event_type": "run_start", "status": "running"}) + "\n" +
        json.dumps({"event_type": "run_end", "status": "INFRA_ERROR"}) + "\n",
        encoding="utf-8"
    )
    assert is_run_already_completed(run_dir) is True

    # Case B: Incomplete trace (no run_end)
    trace_file.write_text(
        json.dumps({"event_type": "run_start", "status": "running"}) + "\n" +
        json.dumps({"event_type": "llm_turn", "text": "thinking"}) + "\n",
        encoding="utf-8"
    )
    assert is_run_already_completed(run_dir) is False

    # Run experiment with FakeLLM; incomplete trace should be renamed, not deleted
    fake_llm = FakeLLM(script=[{
        "text": json.dumps({"file": "traffic_light.v", "line": 20, "bug_type": "fsm_invalid_transition", "root_cause": "fixed", "patch": "--- a\n+++\n"}),
        "input_tokens": 10,
        "output_tokens": 10
    }])
    ledger = tmp_path / "ledger.json"
    project_root = Path(__file__).resolve().parent.parent

    ret = run_experiment(
        split="dev",
        repeats=1,
        configs=["agent"],
        bugs=["B005"],
        run_id="test_incomplete_retry",
        output_base_dir=tmp_path,
        ledger_path=ledger,
        llm_client=fake_llm,
        project_root=project_root,
        skip_preflight=True,
    )
    assert ret == 0

    # Verify original trace was renamed to trace.incomplete.<timestamp>.jsonl
    incomplete_files = list(run_dir.glob("trace.incomplete.*.jsonl"))
    assert len(incomplete_files) == 1
    # Verify new trace.jsonl exists and is completed
    assert trace_file.is_file()
    assert is_run_already_completed(run_dir) is True


# ---------------------------------------------------------------------------
# 21. run_experiment: Manifest Resume Preserves Start Time & Tamper Detection
# ---------------------------------------------------------------------------
def test_run_experiment_manifest_resume_and_tamper_detection(tmp_path: Path) -> None:
    """Verifies manifest resume preserves start_time_utc and detects table alterations."""
    project_root = Path(__file__).resolve().parent.parent
    exp_dir = tmp_path / "manifest_test"
    exp_dir.mkdir(parents=True)
    manifest_path = exp_dir / "manifest.json"

    # Step 1: Run 1 repeat
    fake_llm = FakeLLM(script=[{
        "text": json.dumps({"file": "traffic_light.v", "line": 20, "bug_type": "fsm_invalid_transition", "root_cause": "fixed", "patch": "--- a\n+++\n"}),
        "input_tokens": 10,
        "output_tokens": 10
    }])
    ledger = tmp_path / "ledger_manifest.json"

    ret1 = run_experiment(
        split="dev",
        repeats=1,
        configs=["baseline"],
        bugs=["B005"],
        run_id="exp_manifest",
        output_base_dir=exp_dir,
        ledger_path=ledger,
        llm_client=fake_llm,
        project_root=project_root,
        skip_preflight=True,
    )
    assert ret1 == 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        m1 = json.load(f)
    initial_start_time = m1["start_time_utc"]
    assert m1["resumes"] == []

    # Step 2: Resume (run repeat 1 again, skipped)
    fake_llm2 = FakeLLM(script=[])
    ret2 = run_experiment(
        split="dev",
        repeats=1,
        configs=["baseline"],
        bugs=["B005"],
        run_id="exp_manifest",
        output_base_dir=exp_dir,
        ledger_path=ledger,
        llm_client=fake_llm2,
        project_root=project_root,
        skip_preflight=True,
    )
    assert ret2 == 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        m2 = json.load(f)
    assert m2["start_time_utc"] == initial_start_time
    assert len(m2["resumes"]) == 1

    # Step 3: Tamper with sha256_manifest_table in manifest
    m2["sha256_manifest_table"]["agent/budget.py"] = "tampered_hash_000000000"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(m2, f, indent=2)

    ret3 = run_experiment(
        split="dev",
        repeats=1,
        configs=["baseline"],
        bugs=["B005"],
        run_id="exp_manifest",
        output_base_dir=exp_dir,
        ledger_path=ledger,
        llm_client=fake_llm2,
        project_root=project_root,
        skip_preflight=True,
    )
    assert ret3 == 1


# ---------------------------------------------------------------------------
# 22. Gate 4 Hashes: Detects Unexpected Extra Files in agent/ or eval/
# ---------------------------------------------------------------------------
def test_verify_gate4_hashes_detects_unexpected_extra_files(tmp_path: Path) -> None:
    """Verifies that unexpected untracked files in agent/ or eval/ fail Gate 4 hash verification."""
    project_root = Path(__file__).resolve().parent.parent
    extra_file = project_root / "agent" / "untracked_test_file.tmp"
    try:
        extra_file.write_text("# untracked file\n", encoding="utf-8")
        ok, errs = verify_gate4_hashes(project_root)
        assert ok is False
        assert any("UNEXPECTED EXTRA FILE: agent/untracked_test_file.tmp" in e for e in errs)
    finally:
        if extra_file.exists():
            extra_file.unlink()


# ---------------------------------------------------------------------------
# 23. Budget: Unknown Model Raises ValueError Without Fallback
# ---------------------------------------------------------------------------
def test_budget_unknown_model_raises_value_error(tmp_path: Path) -> None:
    """Verifies that compute_call_cost_usd and record_usage raise ValueError for unknown models."""
    with pytest.raises(ValueError, match="not found in MODEL_PRICING"):
        compute_call_cost_usd(100, 50, model_id="unknown.vendor.model-v1:0")

    bm = BudgetManager(ledger_path=tmp_path / "ledger_unknown_model.json")
    with pytest.raises(ValueError, match="not found in MODEL_PRICING"):
        bm.record_usage(100, 50, model_id="unknown.vendor.model-v1:0")

