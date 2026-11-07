#!/usr/bin/env python3
"""
RTL Failure Triage Agent - Phase 6 Report & Evaluation Tests
Validates:
1. Generation of scores.csv, summary.md, breakdown.md, review_queue.csv, and failure_analysis.md.
2. Self-check: Mismatch in summary counts triggers exit code 1.
3. Idempotency: Consecutive runs produce byte-for-byte identical output.
4. Edge-case detection: loc_near_threshold (2, 3, 4 lines), vis_pass_heldout_fail, pass_but_differs.
5. Accuracy of score.py localization tolerance (+-3 lines correct, +-4 lines incorrect) by calling
   eval.score.score_single_trace without modifying eval/score.py.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest
import sys

# Ensure workspace root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.make_report import make_report, verify_counts_consistency
from eval.score import load_labels_map, score_single_trace


# ---------------------------------------------------------------------------
# 1. Direct Test of eval.score.score_single_trace Localization Threshold (+-3 vs +-4)
# ---------------------------------------------------------------------------
def test_score_py_localization_tolerance_threshold_exact_3_and_4(tmp_path: Path) -> None:
    """
    Directly tests that eval.score awards is_correct_loc = 1 for offsets of exactly +-3 lines,
    and is_correct_loc = 0 for offsets of +-4 lines, without modifying eval/score.py.
    """
    labels_map = load_labels_map(_REPO_ROOT)

    # B015 in counter.v: ground-truth range is [25, 25]
    b015_meta = labels_map["B015"]
    assert b015_meta["line_start"] == 25
    assert b015_meta["line_end"] == 25
    assert b015_meta["file"] == "counter.v"

    test_cases = [
        # (pred_line, expected_loc, description)
        (28, 1, "offset +3 lines from line_end (25 + 3 = 28) -> correct"),
        (22, 1, "offset -3 lines from line_start (25 - 3 = 22) -> correct"),
        (25, 1, "exact match on line 25 -> correct"),
        (29, 0, "offset +4 lines from line_end (25 + 4 = 29) -> incorrect"),
        (21, 0, "offset -4 lines from line_start (25 - 4 = 21) -> incorrect"),
    ]

    for pred_line, expected_loc, desc in test_cases:
        trace_file = tmp_path / f"trace_b015_line_{pred_line}.jsonl"
        with open(trace_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "event_type": "run_start",
                "run_id": f"test_b015_line_{pred_line}",
                "config": "agent",
                "bug_id": "B015",
                "repeat": 1,
            }) + "\n")
            f.write(json.dumps({
                "event_type": "run_end",
                "run_id": f"test_b015_line_{pred_line}",
                "config": "agent",
                "bug_id": "B015",
                "repeat": 1,
                "status": "COMPLETED",
                "stop_reason": "completed",
                "parsed_output": {
                    "file": "counter.v",
                    "line": pred_line,
                    "bug_type": "missing_branch",
                    "patch": "",
                },
            }) + "\n")

        scored = score_single_trace(
            trace_path=trace_file,
            labels_map=labels_map,
            project_root=_REPO_ROOT,
        )
        assert scored.is_correct_loc == expected_loc, f"Failed on {desc}: got {scored.is_correct_loc}"

    # Also test multi-line ground-truth: B005 in fsm_controller.v: range [45, 49]
    b005_meta = labels_map["B005"]
    assert b005_meta["line_start"] == 45
    assert b005_meta["line_end"] == 49
    assert b005_meta["file"] == "fsm_controller.v"

    multi_line_cases = [
        (52, 1, "offset +3 from end 49 (49 + 3 = 52) -> correct"),
        (42, 1, "offset -3 from start 45 (45 - 3 = 42) -> correct"),
        (47, 1, "inside interval [45, 49] -> correct"),
        (53, 0, "offset +4 from end 49 (49 + 4 = 53) -> incorrect"),
        (41, 0, "offset -4 from start 45 (45 - 4 = 41) -> incorrect"),
    ]

    for pred_line, expected_loc, desc in multi_line_cases:
        trace_file = tmp_path / f"trace_b005_line_{pred_line}.jsonl"
        with open(trace_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "event_type": "run_start",
                "run_id": f"test_b005_line_{pred_line}",
                "config": "agent",
                "bug_id": "B005",
                "repeat": 1,
            }) + "\n")
            f.write(json.dumps({
                "event_type": "run_end",
                "run_id": f"test_b005_line_{pred_line}",
                "config": "agent",
                "bug_id": "B005",
                "repeat": 1,
                "status": "COMPLETED",
                "stop_reason": "completed",
                "parsed_output": {
                    "file": "fsm_controller.v",
                    "line": pred_line,
                    "bug_type": "fsm_invalid_transition",
                    "patch": "",
                },
            }) + "\n")

        scored = score_single_trace(
            trace_path=trace_file,
            labels_map=labels_map,
            project_root=_REPO_ROOT,
        )
        assert scored.is_correct_loc == expected_loc, f"Failed on {desc}: got {scored.is_correct_loc}"


# ---------------------------------------------------------------------------
# 2. Comprehensive Test of make_report with Synthetic scores.json and Traces
# ---------------------------------------------------------------------------
def test_make_report_e2e_with_synthetic_known_answers(tmp_path: Path) -> None:
    """
    Tests make_report end-to-end on synthetic scores.json containing:
    - Offset exactly 3 lines (loc_near_threshold & correct_loc = 1)
    - Offset exactly 4 lines (loc_near_threshold & correct_loc = 0)
    - Offset exactly 2 lines (loc_near_threshold & correct_loc = 1)
    - One INFRA_ERROR run (excluded from N_valid)
    - One FAILED run (included in N_valid, 0 points)
    - One vis_pass_heldout_fail run
    - One pass_but_differs run (pass_heldout = 1, patched RTL != clean golden RTL)
    """
    run_dir = tmp_path / "exp_synthetic"
    run_dir.mkdir(parents=True)

    # Create dummy trace files for runs
    def _create_trace(config: str, bug_id: str, repeat: int, final_patch: str = "") -> Path:
        t_dir = run_dir / config / bug_id / str(repeat)
        t_dir.mkdir(parents=True, exist_ok=True)
        t_path = t_dir / "trace.jsonl"
        rid = f"synthetic_{config}_{bug_id}_r{repeat}"
        with open(t_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "event_type": "run_start",
                "run_id": rid,
                "config": config,
                "bug_id": bug_id,
                "repeat": repeat,
            }) + "\n")
            f.write(json.dumps({
                "event_type": "run_end",
                "run_id": rid,
                "config": config,
                "bug_id": bug_id,
                "repeat": repeat,
                "status": "COMPLETED",
                "final_patch": final_patch,
            }) + "\n")
        return t_path

    # Trace for pass_but_differs: valid patch that passes but differs from clean alu.v
    diff_patch = (
        "--- a/alu.v\n"
        "+++ b/alu.v\n"
        "@@ -16,7 +16,7 @@\n"
        " );\n"
        " \n"
        "     // Combinational zero flag\n"
        "-    assign flag_zero = (alu_res[3:0] == 4'b0000);\n"
        "+    assign flag_zero = (alu_res == 8'b00000000); // alternative comment\n"
        " \n"
        "     // Arithmetic and logic evaluation\n"
        "     always @(*) begin\n"
    )

    t_b012_diff = _create_trace("agent", "B012", 1, final_patch=diff_patch)
    _create_trace("agent", "B005", 1)
    _create_trace("agent", "B015", 1)
    _create_trace("baseline", "B005", 1)
    _create_trace("baseline", "B012", 1)
    _create_trace("baseline", "B015", 1)

    # Build individual_runs
    individual_runs = [
        # 1. Agent B005 (Repeat 1): vis_pass_heldout_fail, loc_near_threshold (line 42 is 45 - 3 = dist 3)
        {
            "run_id": "synthetic_agent_B005_r1",
            "config": "agent",
            "bug_id": "B005",
            "repeat": 1,
            "module": "fsm_controller",
            "status": "COMPLETED",
            "stop_reason": "completed",
            "is_valid_run": True,
            "is_infra_error": False,
            "is_correct_loc": 1,
            "is_correct_cls": 1,
            "pass_vis": 1,
            "pass_heldout": 0,
            "pass_first": 0,
            "patch_apply_failed": 0,
            "llm_turns": 5,
            "sim_calls": 2,
            "input_tokens": 10000,
            "output_tokens": 1000,
            "cost_usd": 0.015,
            "pred_file": "fsm_controller.v",
            "pred_line": 42,  # range [45, 49], distance 3 -> loc_near_threshold, is_correct_loc = 1
            "pred_bug_type": "fsm_invalid_transition",
            "first_patch_sim_vis": "FAIL",
            "first_patch_sim_held": "FAIL",
            "final_patch_sim_vis": "PASS",
            "final_patch_sim_held": "FAIL",
            "notes": "Evaluated normally.",
        },
        # 2. Agent B012 (Repeat 1): pass_but_differs (pass_heldout = 1, differs from golden)
        {
            "run_id": "synthetic_agent_B012_r1",
            "config": "agent",
            "bug_id": "B012",
            "repeat": 1,
            "module": "alu",
            "status": "COMPLETED",
            "stop_reason": "completed",
            "is_valid_run": True,
            "is_infra_error": False,
            "is_correct_loc": 1,
            "is_correct_cls": 1,
            "pass_vis": 1,
            "pass_heldout": 1,
            "pass_first": 1,
            "patch_apply_failed": 0,
            "llm_turns": 4,
            "sim_calls": 2,
            "input_tokens": 8000,
            "output_tokens": 800,
            "cost_usd": 0.012,
            "pred_file": "alu.v",
            "pred_line": 19,  # exact match on line 19
            "pred_bug_type": "bit_width_mismatch",
            "first_patch_sim_vis": "PASS",
            "first_patch_sim_held": "PASS",
            "final_patch_sim_vis": "PASS",
            "final_patch_sim_held": "PASS",
            "notes": "Evaluated normally.",
        },
        # 3. Agent B015 (Repeat 1): loc_near_threshold with offset 4 lines (25 + 4 = 29 -> loc 0, dist 4)
        {
            "run_id": "synthetic_agent_B015_r1",
            "config": "agent",
            "bug_id": "B015",
            "repeat": 1,
            "module": "counter",
            "status": "COMPLETED",
            "stop_reason": "completed",
            "is_valid_run": True,
            "is_infra_error": False,
            "is_correct_loc": 0,
            "is_correct_cls": 1,
            "pass_vis": 0,
            "pass_heldout": 0,
            "pass_first": 0,
            "patch_apply_failed": 0,
            "llm_turns": 3,
            "sim_calls": 1,
            "input_tokens": 6000,
            "output_tokens": 500,
            "cost_usd": 0.0085,
            "pred_file": "counter.v",
            "pred_line": 29,  # range [25, 25], distance 4 -> loc_near_threshold, is_correct_loc = 0
            "pred_bug_type": "missing_branch",
            "first_patch_sim_vis": "FAIL",
            "first_patch_sim_held": "FAIL",
            "final_patch_sim_vis": "FAIL",
            "final_patch_sim_held": "FAIL",
            "notes": "Evaluated normally.",
        },
        # 4. Baseline B005 (Repeat 1): INFRA_ERROR run (excluded from N_valid)
        {
            "run_id": "synthetic_baseline_B005_r1",
            "config": "baseline",
            "bug_id": "B005",
            "repeat": 1,
            "module": "fsm_controller",
            "status": "INFRA_ERROR",
            "stop_reason": "bedrock_error",
            "is_valid_run": False,
            "is_infra_error": True,
            "is_correct_loc": 0,
            "is_correct_cls": 0,
            "pass_vis": 0,
            "pass_heldout": 0,
            "pass_first": 0,
            "patch_apply_failed": 0,
            "llm_turns": 0,
            "sim_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "pred_file": None,
            "pred_line": None,
            "pred_bug_type": None,
            "first_patch_sim_vis": "SKIP",
            "first_patch_sim_held": "SKIP",
            "final_patch_sim_vis": "SKIP",
            "final_patch_sim_held": "SKIP",
            "notes": "Unrecoverable Bedrock timeout after retries.",
        },
        # 5. Baseline B012 (Repeat 1): FAILED run (parse_failed: Invalid JSON, valid run in denominator)
        {
            "run_id": "synthetic_baseline_B012_r1",
            "config": "baseline",
            "bug_id": "B012",
            "repeat": 1,
            "module": "alu",
            "status": "FAILED",
            "stop_reason": "parse_failed: Invalid JSON",
            "is_valid_run": True,
            "is_infra_error": False,
            "is_correct_loc": 0,
            "is_correct_cls": 0,
            "pass_vis": 0,
            "pass_heldout": 0,
            "pass_first": 0,
            "patch_apply_failed": 0,
            "llm_turns": 1,
            "sim_calls": 0,
            "input_tokens": 3000,
            "output_tokens": 200,
            "cost_usd": 0.004,
            "pred_file": None,
            "pred_line": None,
            "pred_bug_type": None,
            "first_patch_sim_vis": "FAIL",
            "first_patch_sim_held": "FAIL",
            "final_patch_sim_vis": "FAIL",
            "final_patch_sim_held": "FAIL",
            "notes": "Run ended as FAILED; 0 points awarded.",
        },
        # 6. Baseline B015 (Repeat 1): loc_near_threshold with offset 2 lines (25 - 2 = 23 -> dist 2)
        {
            "run_id": "synthetic_baseline_B015_r1",
            "config": "baseline",
            "bug_id": "B015",
            "repeat": 1,
            "module": "counter",
            "status": "COMPLETED",
            "stop_reason": "completed",
            "is_valid_run": True,
            "is_infra_error": False,
            "is_correct_loc": 1,
            "is_correct_cls": 1,
            "pass_vis": 1,
            "pass_heldout": 1,
            "pass_first": 1,
            "patch_apply_failed": 0,
            "llm_turns": 1,
            "sim_calls": 0,
            "input_tokens": 2500,
            "output_tokens": 300,
            "cost_usd": 0.004,
            "pred_file": "counter.v",
            "pred_line": 23,  # range [25, 25], distance 2 -> loc_near_threshold
            "pred_bug_type": "missing_branch",
            "first_patch_sim_vis": "PASS",
            "first_patch_sim_held": "PASS",
            "final_patch_sim_vis": "PASS",
            "final_patch_sim_held": "PASS",
            "notes": "Evaluated normally.",
        },
    ]

    # Compute expected summary counts
    # Agent: 3 total runs, 0 infra, 3 valid.
    # loc = 2 (B005, B012), cls = 3, vis = 2 (B005, B012), held = 1 (B012), first = 1 (B012)
    # Baseline: 3 total runs, 1 infra (B005), 2 valid (B012, B015).
    # loc = 1 (B015), cls = 1 (B015), vis = 1 (B015), held = 1 (B015), first = 1 (B015)
    summary_data = {
        "by_config": {
            "agent": {
                "repeat_1": {
                    "n_total": 3,
                    "n_valid": 3,
                    "n_infra": 0,
                    "acc_loc_count": 2,
                    "acc_loc_rate": 2 / 3,
                    "acc_cls_count": 3,
                    "acc_cls_rate": 1.0,
                    "pass_vis_count": 2,
                    "pass_vis_rate": 2 / 3,
                    "pass_heldout_count": 1,
                    "pass_heldout_rate": 1 / 3,
                    "pass_first_count": 1,
                    "pass_first_rate": 1 / 3,
                    "mean_turns": 4.0,
                    "mean_sims": 1.6666666666666667,
                    "mean_cost_usd": 0.011833333333333333,
                    "total_cost_usd": 0.0355,
                },
                "repeat_2": {
                    "n_total": 0, "n_valid": 0, "n_infra": 0,
                    "acc_loc_count": 0, "acc_loc_rate": 0.0,
                    "acc_cls_count": 0, "acc_cls_rate": 0.0,
                    "pass_vis_count": 0, "pass_vis_rate": 0.0,
                    "pass_heldout_count": 0, "pass_heldout_rate": 0.0,
                    "pass_first_count": 0, "pass_first_rate": 0.0,
                    "mean_turns": 0.0, "mean_sims": 0.0, "mean_cost_usd": 0.0, "total_cost_usd": 0.0,
                },
                "repeat_3": {
                    "n_total": 0, "n_valid": 0, "n_infra": 0,
                    "acc_loc_count": 0, "acc_loc_rate": 0.0,
                    "acc_cls_count": 0, "acc_cls_rate": 0.0,
                    "pass_vis_count": 0, "pass_vis_rate": 0.0,
                    "pass_heldout_count": 0, "pass_heldout_rate": 0.0,
                    "pass_first_count": 0, "pass_first_rate": 0.0,
                    "mean_turns": 0.0, "mean_sims": 0.0, "mean_cost_usd": 0.0, "total_cost_usd": 0.0,
                },
                "overall": {
                    "n_total": 3,
                    "n_valid": 3,
                    "n_infra": 0,
                    "acc_loc_count": 2,
                    "acc_loc_rate": 2 / 3,
                    "acc_cls_count": 3,
                    "acc_cls_rate": 1.0,
                    "pass_vis_count": 2,
                    "pass_vis_rate": 2 / 3,
                    "pass_heldout_count": 1,
                    "pass_heldout_rate": 1 / 3,
                    "pass_first_count": 1,
                    "pass_first_rate": 1 / 3,
                    "mean_turns": 4.0,
                    "mean_sims": 1.6666666666666667,
                    "mean_cost_usd": 0.011833333333333333,
                    "total_cost_usd": 0.0355,
                },
            },
            "baseline": {
                "repeat_1": {
                    "n_total": 3,
                    "n_valid": 2,
                    "n_infra": 1,
                    "acc_loc_count": 1,
                    "acc_loc_rate": 0.5,
                    "acc_cls_count": 1,
                    "acc_cls_rate": 0.5,
                    "pass_vis_count": 1,
                    "pass_vis_rate": 0.5,
                    "pass_heldout_count": 1,
                    "pass_heldout_rate": 0.5,
                    "pass_first_count": 1,
                    "pass_first_rate": 0.5,
                    "mean_turns": 0.6666666666666666,
                    "mean_sims": 0.0,
                    "mean_cost_usd": 0.0026666666666666666,
                    "total_cost_usd": 0.008,
                },
                "repeat_2": {
                    "n_total": 0, "n_valid": 0, "n_infra": 0,
                    "acc_loc_count": 0, "acc_loc_rate": 0.0,
                    "acc_cls_count": 0, "acc_cls_rate": 0.0,
                    "pass_vis_count": 0, "pass_vis_rate": 0.0,
                    "pass_heldout_count": 0, "pass_heldout_rate": 0.0,
                    "pass_first_count": 0, "pass_first_rate": 0.0,
                    "mean_turns": 0.0, "mean_sims": 0.0, "mean_cost_usd": 0.0, "total_cost_usd": 0.0,
                },
                "repeat_3": {
                    "n_total": 0, "n_valid": 0, "n_infra": 0,
                    "acc_loc_count": 0, "acc_loc_rate": 0.0,
                    "acc_cls_count": 0, "acc_cls_rate": 0.0,
                    "pass_vis_count": 0, "pass_vis_rate": 0.0,
                    "pass_heldout_count": 0, "pass_heldout_rate": 0.0,
                    "pass_first_count": 0, "pass_first_rate": 0.0,
                    "mean_turns": 0.0, "mean_sims": 0.0, "mean_cost_usd": 0.0, "total_cost_usd": 0.0,
                },
                "overall": {
                    "n_total": 3,
                    "n_valid": 2,
                    "n_infra": 1,
                    "acc_loc_count": 1,
                    "acc_loc_rate": 0.5,
                    "acc_cls_count": 1,
                    "acc_cls_rate": 0.5,
                    "pass_vis_count": 1,
                    "pass_vis_rate": 0.5,
                    "pass_heldout_count": 1,
                    "pass_heldout_rate": 0.5,
                    "pass_first_count": 1,
                    "pass_first_rate": 0.5,
                    "mean_turns": 0.6666666666666666,
                    "mean_sims": 0.0,
                    "mean_cost_usd": 0.0026666666666666666,
                    "total_cost_usd": 0.008,
                },
            },
        },
        "decision_rule": {
            "status": "INCOMPLETE",
            "message": "Thực nghiệm chưa hoàn tất (thiếu repeats 2 và 3).",
            "missing_runs": ["baseline/B005/r2", "agent/B005/r2"],
            "deltas": {},
            "passed_all_repeats": False,
        },
    }

    # Write scores.json
    scores_file = run_dir / "scores.json"
    with open(scores_file, "w", encoding="utf-8") as f:
        json.dump({"individual_runs": individual_runs, "summary": summary_data}, f, indent=2)

    fa_custom_path = tmp_path / "custom_failure_analysis.md"

    # Execute make_report
    outputs = make_report(
        run_dir=run_dir,
        labels_path=_REPO_ROOT / "labels" / "labels.json",
        project_root=_REPO_ROOT,
        failure_analysis_path=fa_custom_path,
    )

    # 1. Check scores.csv
    scores_csv = outputs["scores_csv"]
    assert scores_csv.is_file()
    with open(scores_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 6
    # Check ordering: agent runs first, then baseline; sorted by bug_id, repeat
    assert reader[0]["run_id"] == "synthetic_agent_B005_r1"
    assert reader[1]["run_id"] == "synthetic_agent_B012_r1"
    assert reader[2]["run_id"] == "synthetic_agent_B015_r1"
    assert reader[3]["run_id"] == "synthetic_baseline_B005_r1"
    assert reader[4]["run_id"] == "synthetic_baseline_B012_r1"
    assert reader[5]["run_id"] == "synthetic_baseline_B015_r1"

    # 2. Check summary.md
    summary_md = outputs["summary_md"]
    assert summary_md.is_file()
    sum_text = summary_md.read_text(encoding="utf-8")
    assert "2/3 (66.7%)" in sum_text  # Agent Acc_loc
    assert "1/2 (50.0%)" in sum_text  # Baseline Acc_loc with N_valid = 2
    assert "synthetic_baseline_B005_r1" in sum_text  # In INFRA_ERROR list
    assert "synthetic_baseline_B012_r1" in sum_text  # In FAILED list
    assert "Thực nghiệm chưa hoàn tất (thiếu repeats 2 và 3)." in sum_text  # Decision rule verbatim

    # 3. Check breakdown.md
    breakdown_md = outputs["breakdown_md"]
    assert breakdown_md.is_file()
    bd_text = breakdown_md.read_text(encoding="utf-8")
    assert "Mỗi loại chỉ có 1–4 bug, chỉ để minh họa, không kết luận theo loại" in bd_text
    assert "`fsm_invalid_transition`" in bd_text
    assert "`bit_width_mismatch`" in bd_text
    assert "`missing_branch`" in bd_text

    # 4. Check review_queue.csv
    review_queue_csv = outputs["review_queue_csv"]
    assert review_queue_csv.is_file()
    with open(review_queue_csv, "r", encoding="utf-8") as f:
        rq_rows = list(csv.DictReader(f))

    # Expect:
    # - synthetic_agent_B005_r1: loc_near_threshold (dist 3) AND vis_pass_heldout_fail
    # - synthetic_agent_B012_r1: pass_but_differs
    # - synthetic_agent_B015_r1: loc_near_threshold (dist 4)
    # - synthetic_baseline_B015_r1: loc_near_threshold (dist 2)
    rq_map = {r["run_id"]: r for r in rq_rows}
    assert "synthetic_agent_B005_r1" in rq_map
    b005_reasons = rq_map["synthetic_agent_B005_r1"]["reason"]
    assert "loc_near_threshold" in b005_reasons
    assert "vis_pass_heldout_fail" in b005_reasons

    assert "synthetic_agent_B012_r1" in rq_map
    assert "pass_but_differs" in rq_map["synthetic_agent_B012_r1"]["reason"]

    assert "synthetic_agent_B015_r1" in rq_map
    assert "loc_near_threshold" in rq_map["synthetic_agent_B015_r1"]["reason"]
    assert rq_map["synthetic_agent_B015_r1"]["pred_line"] == "29"

    assert "synthetic_baseline_B015_r1" in rq_map
    assert "loc_near_threshold" in rq_map["synthetic_baseline_B015_r1"]["reason"]
    assert rq_map["synthetic_baseline_B015_r1"]["pred_line"] == "23"

    # Verify blank human review columns
    for row in rq_rows:
        assert row["human_verdict"] == ""
        assert row["human_reason"] == ""

    # 5. Check failure_analysis.md template
    fa_file = outputs["failure_analysis_md"]
    assert fa_file.is_file()
    fa_text = fa_file.read_text(encoding="utf-8")
    assert "Phân Tích Lỗi Thất Bại (Failure Analysis Template)" in fa_text
    # Runs with pass_heldout == 0:
    # synthetic_agent_B005_r1, synthetic_agent_B015_r1, synthetic_baseline_B005_r1, synthetic_baseline_B012_r1
    assert "synthetic_agent_B005_r1" in fa_text
    assert "synthetic_agent_B015_r1" in fa_text
    assert "synthetic_baseline_B005_r1" in fa_text
    assert "synthetic_baseline_B012_r1" in fa_text
    # Should have case headers
    assert "### Case 1:" in fa_text
    assert "### Case 2:" in fa_text
    assert "- **Triệu chứng:**" in fa_text
    assert "- **Chỗ lập luận sai:**" in fa_text
    assert "- **Cách cải thiện:**" in fa_text


# ---------------------------------------------------------------------------
# 3. Test Count Mismatch Exit Code 1 Self-Check
# ---------------------------------------------------------------------------
def test_make_report_count_mismatch_exits_code_1(tmp_path: Path) -> None:
    """Verifies that make_report exits with code 1 if summary counts disagree with individual_runs."""
    run_dir = tmp_path / "exp_mismatch"
    run_dir.mkdir()

    individual_runs = [
        {
            "run_id": "r1", "config": "agent", "bug_id": "B005", "repeat": 1,
            "status": "COMPLETED", "is_valid_run": True, "is_infra_error": False,
            "is_correct_loc": 1, "is_correct_cls": 1, "pass_vis": 1, "pass_heldout": 1, "pass_first": 1,
        }
    ]
    # Tamper summary: report acc_loc_count = 0 instead of 1
    tampered_summary = {
        "by_config": {
            "agent": {
                "repeat_1": {
                    "n_total": 1, "n_valid": 1, "n_infra": 0,
                    "acc_loc_count": 0,  # MISMATCH (actual is 1)
                    "acc_cls_count": 1, "pass_vis_count": 1,
                    "pass_heldout_count": 1, "pass_first_count": 1,
                },
                "repeat_2": {"n_total": 0, "n_valid": 0, "n_infra": 0, "acc_loc_count": 0, "acc_cls_count": 0, "pass_vis_count": 0, "pass_heldout_count": 0, "pass_first_count": 0},
                "repeat_3": {"n_total": 0, "n_valid": 0, "n_infra": 0, "acc_loc_count": 0, "acc_cls_count": 0, "pass_vis_count": 0, "pass_heldout_count": 0, "pass_first_count": 0},
                "overall": {
                    "n_total": 1, "n_valid": 1, "n_infra": 0,
                    "acc_loc_count": 0,  # MISMATCH
                    "acc_cls_count": 1, "pass_vis_count": 1,
                    "pass_heldout_count": 1, "pass_first_count": 1,
                },
            },
            "baseline": {
                "repeat_1": {"n_total": 0, "n_valid": 0, "n_infra": 0, "acc_loc_count": 0, "acc_cls_count": 0, "pass_vis_count": 0, "pass_heldout_count": 0, "pass_first_count": 0},
                "repeat_2": {"n_total": 0, "n_valid": 0, "n_infra": 0, "acc_loc_count": 0, "acc_cls_count": 0, "pass_vis_count": 0, "pass_heldout_count": 0, "pass_first_count": 0},
                "repeat_3": {"n_total": 0, "n_valid": 0, "n_infra": 0, "acc_loc_count": 0, "acc_cls_count": 0, "pass_vis_count": 0, "pass_heldout_count": 0, "pass_first_count": 0},
                "overall": {"n_total": 0, "n_valid": 0, "n_infra": 0, "acc_loc_count": 0, "acc_cls_count": 0, "pass_vis_count": 0, "pass_heldout_count": 0, "pass_first_count": 0},
            },
        },
        "decision_rule": {"status": "INCOMPLETE", "message": "", "deltas": {}, "passed_all_repeats": False},
    }

    scores_file = run_dir / "scores.json"
    scores_file.write_text(json.dumps({"individual_runs": individual_runs, "summary": tampered_summary}))

    with pytest.raises(SystemExit) as exc_info:
        make_report(run_dir=run_dir, project_root=_REPO_ROOT)
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# 4. Idempotency Test: Successive Runs Produce 100% Identical Output
# ---------------------------------------------------------------------------
def test_make_report_idempotency_and_stability(tmp_path: Path) -> None:
    """Verifies that running make_report multiple times produces byte-for-byte identical output."""
    import hashlib

    run_dir = tmp_path / "exp_idem"
    run_dir.mkdir()

    individual_runs = [
        {
            "run_id": "r1", "config": "baseline", "bug_id": "B015", "repeat": 1,
            "status": "COMPLETED", "stop_reason": "completed",
            "is_valid_run": True, "is_infra_error": False,
            "is_correct_loc": 1, "is_correct_cls": 1, "pass_vis": 1, "pass_heldout": 1, "pass_first": 1,
            "patch_apply_failed": 0, "llm_turns": 1, "sim_calls": 0, "input_tokens": 100, "output_tokens": 50,
            "cost_usd": 0.00035, "pred_file": "counter.v", "pred_line": 25, "pred_bug_type": "missing_branch",
            "first_patch_sim_vis": "PASS", "first_patch_sim_held": "PASS",
            "final_patch_sim_vis": "PASS", "final_patch_sim_held": "PASS",
            "notes": "normal",
        }
    ]
    summary_data = {
        "by_config": {
            "baseline": {
                "repeat_1": {"n_total": 1, "n_valid": 1, "n_infra": 0, "acc_loc_count": 1, "acc_loc_rate": 1.0, "acc_cls_count": 1, "acc_cls_rate": 1.0, "pass_vis_count": 1, "pass_vis_rate": 1.0, "pass_heldout_count": 1, "pass_heldout_rate": 1.0, "pass_first_count": 1, "pass_first_rate": 1.0, "mean_turns": 1.0, "mean_sims": 0.0, "mean_cost_usd": 0.00035, "total_cost_usd": 0.00035},
                "repeat_2": {"n_total": 0, "n_valid": 0, "n_infra": 0, "acc_loc_count": 0, "acc_loc_rate": 0.0, "acc_cls_count": 0, "acc_cls_rate": 0.0, "pass_vis_count": 0, "pass_vis_rate": 0.0, "pass_heldout_count": 0, "pass_heldout_rate": 0.0, "pass_first_count": 0, "pass_first_rate": 0.0, "mean_turns": 0.0, "mean_sims": 0.0, "mean_cost_usd": 0.0, "total_cost_usd": 0.0},
                "repeat_3": {"n_total": 0, "n_valid": 0, "n_infra": 0, "acc_loc_count": 0, "acc_loc_rate": 0.0, "acc_cls_count": 0, "acc_cls_rate": 0.0, "pass_vis_count": 0, "pass_vis_rate": 0.0, "pass_heldout_count": 0, "pass_heldout_rate": 0.0, "pass_first_count": 0, "pass_first_rate": 0.0, "mean_turns": 0.0, "mean_sims": 0.0, "mean_cost_usd": 0.0, "total_cost_usd": 0.0},
                "overall": {"n_total": 1, "n_valid": 1, "n_infra": 0, "acc_loc_count": 1, "acc_loc_rate": 1.0, "acc_cls_count": 1, "acc_cls_rate": 1.0, "pass_vis_count": 1, "pass_vis_rate": 1.0, "pass_heldout_count": 1, "pass_heldout_rate": 1.0, "pass_first_count": 1, "pass_first_rate": 1.0, "mean_turns": 1.0, "mean_sims": 0.0, "mean_cost_usd": 0.00035, "total_cost_usd": 0.00035},
            },
            "agent": {
                "repeat_1": {"n_total": 0, "n_valid": 0, "n_infra": 0, "acc_loc_count": 0, "acc_loc_rate": 0.0, "acc_cls_count": 0, "acc_cls_rate": 0.0, "pass_vis_count": 0, "pass_vis_rate": 0.0, "pass_heldout_count": 0, "pass_heldout_rate": 0.0, "pass_first_count": 0, "pass_first_rate": 0.0, "mean_turns": 0.0, "mean_sims": 0.0, "mean_cost_usd": 0.0, "total_cost_usd": 0.0},
                "repeat_2": {"n_total": 0, "n_valid": 0, "n_infra": 0, "acc_loc_count": 0, "acc_loc_rate": 0.0, "acc_cls_count": 0, "acc_cls_rate": 0.0, "pass_vis_count": 0, "pass_vis_rate": 0.0, "pass_heldout_count": 0, "pass_heldout_rate": 0.0, "pass_first_count": 0, "pass_first_rate": 0.0, "mean_turns": 0.0, "mean_sims": 0.0, "mean_cost_usd": 0.0, "total_cost_usd": 0.0},
                "repeat_3": {"n_total": 0, "n_valid": 0, "n_infra": 0, "acc_loc_count": 0, "acc_loc_rate": 0.0, "acc_cls_count": 0, "acc_cls_rate": 0.0, "pass_vis_count": 0, "pass_vis_rate": 0.0, "pass_heldout_count": 0, "pass_heldout_rate": 0.0, "pass_first_count": 0, "pass_first_rate": 0.0, "mean_turns": 0.0, "mean_sims": 0.0, "mean_cost_usd": 0.0, "total_cost_usd": 0.0},
                "overall": {"n_total": 0, "n_valid": 0, "n_infra": 0, "acc_loc_count": 0, "acc_loc_rate": 0.0, "acc_cls_count": 0, "acc_cls_rate": 0.0, "pass_vis_count": 0, "pass_vis_rate": 0.0, "pass_heldout_count": 0, "pass_heldout_rate": 0.0, "pass_first_count": 0, "pass_first_rate": 0.0, "mean_turns": 0.0, "mean_sims": 0.0, "mean_cost_usd": 0.0, "total_cost_usd": 0.0},
            },
        },
        "decision_rule": {"status": "INCOMPLETE", "message": "msg", "deltas": {}, "passed_all_repeats": False},
    }

    scores_file = run_dir / "scores.json"
    scores_file.write_text(json.dumps({"individual_runs": individual_runs, "summary": summary_data}))

    fa_path = tmp_path / "fa.md"

    # Run 1
    outputs1 = make_report(run_dir=run_dir, project_root=_REPO_ROOT, failure_analysis_path=fa_path)
    hashes1 = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in outputs1.items()}

    # Run 2
    outputs2 = make_report(run_dir=run_dir, project_root=_REPO_ROOT, failure_analysis_path=fa_path)
    hashes2 = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in outputs2.items()}

    assert hashes1 == hashes2
