#!/usr/bin/env python3
"""
RTL Failure Triage Agent - Phase 6 Report Generator
Generates reproducible evaluation reports, breakdown tables, review queues,
and failure analysis templates strictly following preregistration rules.

Enforced constraints:
- Input: --run-dir results/<run_id> (scores.json) and labels/labels.json.
- Imports eval.patcher to test diff matching, but NEVER imports or re-implements
  the scoring logic of eval.score.
- Outputs into <run_dir>/:
    1. scores.csv
    2. summary.md
    3. breakdown.md
    4. review_queue.csv
- Outputs docs/failure_analysis.md template (with relative trace links).
- Self-check: All counts in summary.md MUST equal recomputed counts from individual_runs
  AND counts in summary of scores.json; any mismatch exits with code 1.
- Deterministic and idempotent: Repeated runs produce byte-for-byte identical files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

# Ensure workspace root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.patcher import apply_unified_diff

BUG_TYPES_ORDER = [
    "fifo_flag",
    "fsm_invalid_transition",
    "reset_polarity",
    "blocking_nonblocking",
    "bit_width_mismatch",
    "missing_branch",
    "off_by_one",
]

MODULES_ORDER = ["counter", "alu", "fifo", "fsm_controller"]


def index_traces_by_run_id(run_dir: Path) -> dict[str, Path]:
    """Scans run_dir for JSONL trace files and indexes them by run_id."""
    trace_map: dict[str, Path] = {}
    for p in sorted(run_dir.rglob("*.jsonl")):
        if p.name.startswith("trace.incomplete."):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                first_line = f.readline()
                if first_line:
                    data = json.loads(first_line)
                    rid = data.get("run_id")
                    if rid and rid not in trace_map:
                        trace_map[rid] = p.resolve()
        except Exception:
            pass
    return trace_map


def extract_final_patch_from_trace(trace_path: Path) -> str | None:
    """Extracts final_patch from the run_end event in a trace file."""
    if not trace_path or not trace_path.is_file():
        return None
    try:
        with open(trace_path, "r", encoding="utf-8") as f:
            for line in reversed(f.readlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                    if ev.get("event_type") == "run_end":
                        patch = ev.get("final_patch")
                        if not patch:
                            patch = ev.get("parsed_output", {}).get("patch")
                        return patch
                except Exception:
                    pass
    except Exception:
        pass
    return None


def verify_counts_consistency(
    individual_runs: list[dict[str, Any]],
    summary_data: dict[str, Any],
) -> None:
    """
    Self-check: Verifies that counts in summary match counts computed from individual_runs.
    Exits with code 1 if any count diverges.
    """
    by_config = summary_data.get("by_config", {})
    for cfg in ("baseline", "agent"):
        cfg_summary = by_config.get(cfg, {})
        for rep_key in ("repeat_1", "repeat_2", "repeat_3", "overall"):
            expected = cfg_summary.get(rep_key, {})
            if rep_key.startswith("repeat_"):
                rep_num = int(rep_key.split("_")[1])
                matching = [r for r in individual_runs if r.get("config") == cfg and r.get("repeat") == rep_num]
            else:
                matching = [r for r in individual_runs if r.get("config") == cfg]

            n_total = len(matching)
            n_infra = sum(1 for r in matching if r.get("is_infra_error") or r.get("status") == "INFRA_ERROR")
            valid = [r for r in matching if not (r.get("is_infra_error") or r.get("status") == "INFRA_ERROR")]
            n_valid = len(valid)

            acc_loc = sum(1 for r in valid if r.get("is_correct_loc") == 1)
            acc_cls = sum(1 for r in valid if r.get("is_correct_cls") == 1)
            pass_vis = sum(1 for r in valid if r.get("pass_vis") == 1)
            pass_held = sum(1 for r in valid if r.get("pass_heldout") == 1)
            pass_first = sum(1 for r in valid if r.get("pass_first") == 1)

            checks = [
                ("n_total", n_total, expected.get("n_total")),
                ("n_valid", n_valid, expected.get("n_valid")),
                ("n_infra", n_infra, expected.get("n_infra")),
                ("acc_loc_count", acc_loc, expected.get("acc_loc_count")),
                ("acc_cls_count", acc_cls, expected.get("acc_cls_count")),
                ("pass_vis_count", pass_vis, expected.get("pass_vis_count")),
                ("pass_heldout_count", pass_held, expected.get("pass_heldout_count")),
                ("pass_first_count", pass_first, expected.get("pass_first_count")),
            ]
            for metric_name, computed_val, expected_val in checks:
                if computed_val != expected_val:
                    sys.stderr.write(
                        f"[FATAL ERROR] Count mismatch for config '{cfg}' {rep_key} metric '{metric_name}': "
                        f"computed {computed_val} != summary {expected_val}\n"
                    )
                    sys.exit(1)


def generate_scores_csv(
    individual_runs: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Generates scores.csv sorted by config, bug_id, repeat."""
    sorted_runs = sorted(
        individual_runs,
        key=lambda r: (str(r.get("config", "")), str(r.get("bug_id", "")), int(r.get("repeat", 0))),
    )

    if sorted_runs:
        fieldnames = list(sorted_runs[0].keys())
        for r in sorted_runs[1:]:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
    else:
        fieldnames = []

    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for r in sorted_runs:
            writer.writerow(r)


def generate_summary_md(
    individual_runs: list[dict[str, Any]],
    summary_data: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Generates summary.md conforming strictly to Plan Phase 6 template.
    Copies decision_rule status, message, and deltas verbatim.
    Separately lists INFRA_ERROR and FAILED/EXCEEDED_LIMIT runs.
    """
    by_config = summary_data.get("by_config", {})
    decision_rule = summary_data.get("decision_rule", {})

    lines: list[str] = []
    lines.append("# Evaluation Summary Report")
    lines.append("")
    lines.append("## 1. Overall Summary Table (Phase 6)")
    lines.append("")
    lines.append("| Metric | Baseline | Triage agent |")
    lines.append("| :--- | :--- | :--- |")

    b_ov = by_config.get("baseline", {}).get("overall", {})
    a_ov = by_config.get("agent", {}).get("overall", {})

    b_n = b_ov.get("n_valid", 0)
    a_n = a_ov.get("n_valid", 0)

    def _fmt_rate(count: int, n: int, rate: float) -> str:
        pct = rate * 100.0
        return f"{count}/{n} ({pct:.1f}%)" if n > 0 else f"{count}/{n} (0.0%)"

    b_loc = _fmt_rate(b_ov.get("acc_loc_count", 0), b_n, b_ov.get("acc_loc_rate", 0.0))
    a_loc = _fmt_rate(a_ov.get("acc_loc_count", 0), a_n, a_ov.get("acc_loc_rate", 0.0))
    lines.append(f"| Correct file and line localization | {b_loc} | {a_loc} |")

    b_cls = _fmt_rate(b_ov.get("acc_cls_count", 0), b_n, b_ov.get("acc_cls_rate", 0.0))
    a_cls = _fmt_rate(a_ov.get("acc_cls_count", 0), a_n, a_ov.get("acc_cls_rate", 0.0))
    lines.append(f"| Correct classification | {b_cls} | {a_cls} |")

    b_vis = _fmt_rate(b_ov.get("pass_vis_count", 0), b_n, b_ov.get("pass_vis_rate", 0.0))
    a_vis = _fmt_rate(a_ov.get("pass_vis_count", 0), a_n, a_ov.get("pass_vis_rate", 0.0))
    lines.append(f"| Patch passes `tb/visible` | {b_vis} | {a_vis} |")

    b_held = _fmt_rate(b_ov.get("pass_heldout_count", 0), b_n, b_ov.get("pass_heldout_rate", 0.0))
    a_held = _fmt_rate(a_ov.get("pass_heldout_count", 0), a_n, a_ov.get("pass_heldout_rate", 0.0))
    lines.append(f"| Patch passes `tb/heldout` | {b_held} | {a_held} |")

    b_first = _fmt_rate(b_ov.get("pass_first_count", 0), b_n, b_ov.get("pass_first_rate", 0.0))
    a_first = _fmt_rate(a_ov.get("pass_first_count", 0), a_n, a_ov.get("pass_first_rate", 0.0))
    lines.append(f"| First patch passes `tb/heldout` | {b_first} | {a_first} |")

    b_turns = f"{b_ov.get('mean_turns', 0.0):.1f}"
    a_turns = f"{a_ov.get('mean_turns', 0.0):.1f}"
    lines.append(f"| Average LLM calls | {b_turns} | {a_turns} |")

    b_cost = f"${b_ov.get('mean_cost_usd', 0.0):.4f}"
    a_cost = f"${a_ov.get('mean_cost_usd', 0.0):.4f}"
    lines.append(f"| Cost per bug (USD) | {b_cost} | {a_cost} |")

    lines.append("")
    lines.append("## 2. Detailed Table by Repeat and Total")
    lines.append("")
    lines.append(
        "| Metric | Baseline R1 | Baseline R2 | Baseline R3 | Baseline Total | "
        "Triage Agent R1 | Triage Agent R2 | Triage Agent R3 | Triage Agent Total |"
    )
    lines.append(
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    )

    metrics_keys = [
        ("Correct file and line localization", "acc_loc_count"),
        ("Correct classification", "acc_cls_count"),
        ("Patch passes `tb/visible`", "pass_vis_count"),
        ("Patch passes `tb/heldout`", "pass_heldout_count"),
        ("First patch passes `tb/heldout`", "pass_first_count"),
    ]

    for label, count_key in metrics_keys:
        row_vals = [label]
        for cfg in ("baseline", "agent"):
            for rep in (1, 2, 3):
                cell = by_config.get(cfg, {}).get(f"repeat_{rep}", {})
                n = cell.get("n_valid", 0)
                c = cell.get(count_key, 0)
                row_vals.append(f"{c}/{n}")
            cell_ov = by_config.get(cfg, {}).get("overall", {})
            n_ov = cell_ov.get("n_valid", 0)
            c_ov = cell_ov.get(count_key, 0)
            row_vals.append(f"{c_ov}/{n_ov}")
        lines.append("| " + " | ".join(row_vals) + " |")

    # Resource rows
    row_turns = ["Average LLM calls"]
    for cfg in ("baseline", "agent"):
        for rep in (1, 2, 3):
            val = by_config.get(cfg, {}).get(f"repeat_{rep}", {}).get("mean_turns", 0.0)
            row_turns.append(f"{val:.1f}")
        row_turns.append(f"{by_config.get(cfg, {}).get('overall', {}).get('mean_turns', 0.0):.1f}")
    lines.append("| " + " | ".join(row_turns) + " |")

    row_costs = ["Cost per bug (USD)"]
    for cfg in ("baseline", "agent"):
        for rep in (1, 2, 3):
            val = by_config.get(cfg, {}).get(f"repeat_{rep}", {}).get("mean_cost_usd", 0.0)
            row_costs.append(f"${val:.4f}")
        row_costs.append(f"${by_config.get(cfg, {}).get('overall', {}).get('mean_cost_usd', 0.0):.4f}")
    lines.append("| " + " | ".join(row_costs) + " |")

    # Section 3: Decision Rule
    lines.append("")
    lines.append("## 3. Decision Rule")
    lines.append("")
    lines.append(f"- **Status:** `{decision_rule.get('status', 'UNKNOWN')}`")
    lines.append(f"- **Message:** {decision_rule.get('message', '')}")
    lines.append(f"- **Passed all repeats:** `{decision_rule.get('passed_all_repeats', False)}`")

    deltas = decision_rule.get("deltas", {})
    lines.append("- **Deltas:**")
    if deltas:
        lines.append("")
        lines.append("| Repeat | Agent Pass Held-out | Baseline Pass Held-out | Delta (Agent - Baseline) | Threshold >= 3 Met? |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for rep in (1, 2, 3):
            d_rep = deltas.get(f"repeat_{rep}", {})
            ag_p = d_rep.get("agent_pass_heldout", 0)
            bl_p = d_rep.get("baseline_pass_heldout", 0)
            d_val = d_rep.get("delta", 0)
            t_met = d_rep.get("threshold_met", False)
            lines.append(f"| Repeat {rep} | {ag_p} | {bl_p} | {d_val:+d} | {t_met} |")
    else:
        lines.append("  - No deltas (insufficient data across all 3 repeats).")

    # Section 4: INFRA_ERROR runs
    lines.append("")
    lines.append("## 4. Runs Excluded from Denominator (INFRA_ERROR)")
    lines.append("")
    infra_runs = [
        r for r in individual_runs
        if r.get("is_infra_error") or r.get("status") == "INFRA_ERROR"
    ]
    if infra_runs:
        lines.append("| Run ID | Config | Bug ID | Repeat | Status | Stop Reason | Notes |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for r in sorted(infra_runs, key=lambda x: (x.get("config", ""), x.get("bug_id", ""), x.get("repeat", 0))):
            lines.append(
                f"| `{r.get('run_id')}` | {r.get('config')} | {r.get('bug_id')} | {r.get('repeat')} | "
                f"`{r.get('status')}` | `{r.get('stop_reason')}` | {r.get('notes', '')} |"
            )
    else:
        lines.append("No runs encountered infrastructure errors (0 runs excluded from N_valid).")

    # Section 5: FAILED and EXCEEDED_LIMIT runs
    lines.append("")
    lines.append("## 5. FAILED and EXCEEDED_LIMIT Runs")
    lines.append("")
    failed_runs = [
        r for r in individual_runs
        if r.get("status") in ("FAILED", "EXCEEDED_LIMIT")
    ]
    if failed_runs:
        lines.append("| Run ID | Config | Bug ID | Repeat | Status | Stop Reason | Notes |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for r in sorted(failed_runs, key=lambda x: (x.get("config", ""), x.get("bug_id", ""), x.get("repeat", 0))):
            lines.append(
                f"| `{r.get('run_id')}` | {r.get('config')} | {r.get('bug_id')} | {r.get('repeat')} | "
                f"`{r.get('status')}` | `{r.get('stop_reason')}` | {r.get('notes', '')} |"
            )
    else:
        lines.append("No runs in FAILED or EXCEEDED_LIMIT status.")

    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_breakdown_md(
    individual_runs: list[dict[str, Any]],
    labels_map: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Generates breakdown.md detailing pass_heldout across bug_types and modules.
    Explicitly includes mandatory notice:
    'Each category only has 1–4 bugs, for illustration only, no category-level conclusions'.
    """
    lines: list[str] = []
    lines.append("# Pass Held-out Breakdown")
    lines.append("")
    lines.append("> **Note:** Each category only has 1–4 bugs, for illustration only, no category-level conclusions.")
    lines.append("")

    # 1. Breakdown by Bug Type
    lines.append("## 1. Breakdown by Bug Type")
    lines.append("")
    lines.append(
        "| Bug Type | Bug count | Baseline R1 | Baseline R2 | Baseline R3 | Baseline Total | Baseline Min–Max | "
        "Agent R1 | Agent R2 | Agent R3 | Agent Total | Agent Min–Max |"
    )
    lines.append(
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    )

    # Determine unique bugs present
    present_bugs = sorted(list(set(str(r.get("bug_id")) for r in individual_runs)))

    for bt in BUG_TYPES_ORDER:
        matching_bugs = [b for b in present_bugs if labels_map.get(b, {}).get("bug_type") == bt]
        n_bugs = len(matching_bugs)
        row = [f"`{bt}`", str(n_bugs)]

        for cfg in ("baseline", "agent"):
            rep_passes: list[int] = []
            for rep in (1, 2, 3):
                runs_r = [
                    r for r in individual_runs
                    if r.get("config") == cfg and str(r.get("bug_id")) in matching_bugs
                    and r.get("repeat") == rep and not r.get("is_infra_error")
                ]
                p_count = sum(1 for r in runs_r if r.get("pass_heldout") == 1)
                rep_passes.append(p_count)
                row.append(f"{p_count}/{len(runs_r)}")

            all_runs = [
                r for r in individual_runs
                if r.get("config") == cfg and str(r.get("bug_id")) in matching_bugs
                and not r.get("is_infra_error")
            ]
            tot_p = sum(1 for r in all_runs if r.get("pass_heldout") == 1)
            row.append(f"{tot_p}/{len(all_runs)}")

            min_val = min(rep_passes) if rep_passes else 0
            max_val = max(rep_passes) if rep_passes else 0
            row.append(f"{min_val}–{max_val}")

        lines.append("| " + " | ".join(row) + " |")

    # 2. Breakdown by Module (Design Module)
    lines.append("")
    lines.append("## 2. Breakdown by RTL Module (Design Module)")
    lines.append("")
    lines.append(
        "| Module | Bug count | Baseline R1 | Baseline R2 | Baseline R3 | Baseline Total | Baseline Min–Max | "
        "Agent R1 | Agent R2 | Agent R3 | Agent Total | Agent Min–Max |"
    )
    lines.append(
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    )

    for mod in MODULES_ORDER:
        matching_bugs = [b for b in present_bugs if labels_map.get(b, {}).get("module") == mod]
        n_bugs = len(matching_bugs)
        row = [f"`{mod}`", str(n_bugs)]

        for cfg in ("baseline", "agent"):
            rep_passes = []
            for rep in (1, 2, 3):
                runs_r = [
                    r for r in individual_runs
                    if r.get("config") == cfg and str(r.get("bug_id")) in matching_bugs
                    and r.get("repeat") == rep and not r.get("is_infra_error")
                ]
                p_count = sum(1 for r in runs_r if r.get("pass_heldout") == 1)
                rep_passes.append(p_count)
                row.append(f"{p_count}/{len(runs_r)}")

            all_runs = [
                r for r in individual_runs
                if r.get("config") == cfg and str(r.get("bug_id")) in matching_bugs
                and not r.get("is_infra_error")
            ]
            tot_p = sum(1 for r in all_runs if r.get("pass_heldout") == 1)
            row.append(f"{tot_p}/{len(all_runs)}")

            min_val = min(rep_passes) if rep_passes else 0
            max_val = max(rep_passes) if rep_passes else 0
            row.append(f"{min_val}–{max_val}")

        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def check_loc_near_threshold(
    run: dict[str, Any],
    labels_map: dict[str, dict[str, Any]],
) -> tuple[bool, int | None, str]:
    """
    Checks if predicted line is 2, 3, or 4 lines away from the ground-truth interval.
    Returns (is_near, pred_line, label_range_str).
    """
    bug_id = str(run.get("bug_id", ""))
    lbl = labels_map.get(bug_id, {})
    l_start = lbl.get("line_start")
    l_end = lbl.get("line_end")
    label_range_str = f"[{l_start}, {l_end}]" if l_start is not None else "[]"

    pred_line = run.get("pred_line")
    pred_file = run.get("pred_file")
    gt_file = lbl.get("file")

    if not gt_file or not pred_file:
        return False, None, label_range_str
    if Path(str(pred_file)).name != Path(str(gt_file)).name:
        return False, None, label_range_str

    if not isinstance(pred_line, int) or l_start is None or l_end is None:
        return False, None, label_range_str

    if pred_line < l_start:
        dist = l_start - pred_line
    elif pred_line > l_end:
        dist = pred_line - l_end
    else:
        dist = 0

    return dist in (2, 3, 4), pred_line, label_range_str


def check_pass_but_differs(
    run: dict[str, Any],
    trace_path: Path | None,
    labels_map: dict[str, dict[str, Any]],
    project_root: Path,
) -> bool:
    """
    Checks if pass_heldout == 1 but the patched RTL differs from golden rtl/<file>
    when comparing lines after stripping trailing whitespace.
    """
    if run.get("pass_heldout") != 1:
        return False

    bug_id = str(run.get("bug_id", ""))
    lbl = labels_map.get(bug_id, {})
    rel_file = lbl.get("file")
    if not rel_file:
        return False

    final_patch = None
    if trace_path:
        final_patch = extract_final_patch_from_trace(trace_path)
    if not final_patch:
        return False

    variant_path = project_root / "bugs" / "variants" / bug_id / rel_file
    golden_path = project_root / "rtl" / rel_file
    if not variant_path.is_file() or not golden_path.is_file():
        return False

    variant_text = variant_path.read_text(encoding="utf-8")
    golden_text = golden_path.read_text(encoding="utf-8")

    patch_res = apply_unified_diff(variant_text, final_patch)
    if not patch_res.success:
        return False

    patched_lines = [line.rstrip() for line in patch_res.patched_text.splitlines()]
    golden_lines = [line.rstrip() for line in golden_text.splitlines()]

    return patched_lines != golden_lines


def generate_review_queue_csv(
    individual_runs: list[dict[str, Any]],
    trace_map: dict[str, Path],
    labels_map: dict[str, dict[str, Any]],
    project_root: Path,
    output_path: Path,
) -> None:
    """
    Generates review_queue.csv containing edge cases:
    1. loc_near_threshold: distance to [line_start, line_end] is 2, 3, or 4 lines.
    2. vis_pass_heldout_fail: pass_vis == 1 and pass_heldout == 0.
    3. pass_but_differs: pass_heldout == 1 and patched RTL differs from original rtl/<file>.
    Columns: run_id, config, bug_id, repeat, reason, pred_line, label_range, trace_path,
             human_verdict, human_reason.
    """
    fieldnames = [
        "run_id",
        "config",
        "bug_id",
        "repeat",
        "reason",
        "pred_line",
        "label_range",
        "trace_path",
        "human_verdict",
        "human_reason",
    ]

    rows: list[dict[str, Any]] = []

    for r in individual_runs:
        run_id = str(r.get("run_id", ""))
        trace_path = trace_map.get(run_id)

        reasons: list[str] = []
        is_near, pred_line, label_range_str = check_loc_near_threshold(r, labels_map)
        if is_near:
            reasons.append("loc_near_threshold")

        if r.get("pass_vis") == 1 and r.get("pass_heldout") == 0:
            reasons.append("vis_pass_heldout_fail")

        if check_pass_but_differs(r, trace_path, labels_map, project_root):
            reasons.append("pass_but_differs")

        if reasons:
            if trace_path:
                try:
                    rel_trace = Path(os.path.relpath(trace_path, project_root)).as_posix()
                except Exception:
                    rel_trace = str(trace_path)
            else:
                rel_trace = ""

            rows.append({
                "run_id": run_id,
                "config": r.get("config", ""),
                "bug_id": r.get("bug_id", ""),
                "repeat": r.get("repeat", 0),
                "reason": "; ".join(reasons),
                "pred_line": pred_line if pred_line is not None else (r.get("pred_line") or ""),
                "label_range": label_range_str,
                "trace_path": rel_trace,
                "human_verdict": "",
                "human_reason": "",
            })

    sorted_rows = sorted(
        rows,
        key=lambda x: (str(x["config"]), str(x["bug_id"]), int(x["repeat"])),
    )

    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow(row)


def generate_failure_analysis_md(
    individual_runs: list[dict[str, Any]],
    trace_map: dict[str, Path],
    output_path: Path,
) -> None:
    """
    Generates failure_analysis.md TEMPLATE containing:
    1. Table of ALL runs where pass_heldout == 0 (with relative links to trace).
    2. Template frames for 2-3 detailed cases with blank sections:
       Symptom / Faulty reasoning / Improvement plan.
    """
    failed_runs = [r for r in individual_runs if r.get("pass_heldout") == 0]
    sorted_failed = sorted(
        failed_runs,
        key=lambda x: (str(x.get("config", "")), str(x.get("bug_id", "")), int(x.get("repeat", 0))),
    )

    lines: list[str] = []
    lines.append("# Failure Analysis Template")
    lines.append("")
    lines.append("> **Note:** This is a TEMPLATE file. The summary table lists all runs")
    lines.append("> that did not pass the held-out test (`pass_heldout = 0`). The in-depth analysis")
    lines.append("> section below is intended for manual human inspection and notes.")
    lines.append("")
    lines.append("## 1. Complete List of Failed Runs (`pass_heldout = 0`)")
    lines.append("")

    if sorted_failed:
        lines.append("| Run ID | Config | Bug ID | Repeat | Status | Stop Reason | Trace File |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for r in sorted_failed:
            rid = str(r.get("run_id", ""))
            trace_path = trace_map.get(rid)
            if trace_path:
                try:
                    rel_trace = Path(os.path.relpath(trace_path, output_path.parent)).as_posix()
                    trace_cell = f"[`{trace_path.name}`]({rel_trace})"
                except Exception:
                    trace_cell = f"`{trace_path.name}`"
            else:
                trace_cell = "N/A"

            lines.append(
                f"| `{rid}` | {r.get('config')} | `{r.get('bug_id')}` | {r.get('repeat')} | "
                f"`{r.get('status')}` | `{r.get('stop_reason')}` | {trace_cell} |"
            )
    else:
        lines.append("No runs failed (100% runs passed held-out).")

    lines.append("")
    lines.append("## 2. Detailed Case Analysis Framework (Human Analysis)")
    lines.append("")
    lines.append("<!-- Users select at least 2–3 failed cases above to fill in analysis -->")
    lines.append("")

    # Pick up to 3 failed runs as exemplar headers, or generic if none
    cases_to_show = sorted_failed[:3] if sorted_failed else []
    if not cases_to_show:
        cases_to_show = [
            {"run_id": "EXAMPLE_RUN_1", "bug_id": "B005", "config": "agent"},
            {"run_id": "EXAMPLE_RUN_2", "bug_id": "B012", "config": "baseline"},
        ]

    for idx, c in enumerate(cases_to_show, start=1):
        lines.append(f"### Case {idx}: `{c.get('run_id')}` (Bug {c.get('bug_id')}, {c.get('config')})")
        lines.append(f"- **Run ID:** `{c.get('run_id')}`")
        lines.append(f"- **Bug ID:** `{c.get('bug_id')}`")
        lines.append(f"- **Config:** `{c.get('config')}`")
        lines.append("- **Symptom:**")
        lines.append("  <!-- Describe observed error symptoms from simulation logs or tool calls -->")
        lines.append("- **Faulty reasoning:**")
        lines.append("  <!-- Analyze why the model made an incorrect diagnosis or patch -->")
        lines.append("- **Improvement plan:**")
        lines.append("  <!-- Propose improvements (prompts, tools, testing process) -->")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def make_report(
    run_dir: Path | str,
    labels_path: Path | str | None = None,
    project_root: Path | str | None = None,
    failure_analysis_path: Path | str | None = None,
) -> dict[str, Path]:
    """
    Main programmatic entry point to generate all Phase 6 reports.
    Returns mapping of artifact name to output Path.
    """
    run_dir = Path(run_dir).resolve()
    if project_root is None:
        project_root = _REPO_ROOT
    else:
        project_root = Path(project_root).resolve()

    if labels_path is None:
        labels_path = project_root / "labels" / "labels.json"
    else:
        labels_path = Path(labels_path).resolve()

    scores_json_path = run_dir / "scores.json"
    if not scores_json_path.is_file():
        raise FileNotFoundError(f"Scores file not found: {scores_json_path}")

    with open(scores_json_path, "r", encoding="utf-8") as f:
        scores_data = json.load(f)

    with open(labels_path, "r", encoding="utf-8") as f:
        labels_list = json.load(f)
    labels_map = {b["bug_id"]: b for b in labels_list}

    individual_runs: list[dict[str, Any]] = scores_data.get("individual_runs", [])
    summary_data: dict[str, Any] = scores_data.get("summary", {})

    # Self-check consistency requirement
    verify_counts_consistency(individual_runs, summary_data)

    trace_map = index_traces_by_run_id(run_dir)

    # 1. Output scores.csv
    scores_csv_path = run_dir / "scores.csv"
    generate_scores_csv(individual_runs, scores_csv_path)

    # 2. Output summary.md
    summary_md_path = run_dir / "summary.md"
    generate_summary_md(individual_runs, summary_data, summary_md_path)

    # 3. Output breakdown.md
    breakdown_md_path = run_dir / "breakdown.md"
    generate_breakdown_md(individual_runs, labels_map, breakdown_md_path)

    # 4. Output review_queue.csv
    review_queue_path = run_dir / "review_queue.csv"
    generate_review_queue_csv(individual_runs, trace_map, labels_map, project_root, review_queue_path)

    # 5. Output failure_analysis.md template
    if failure_analysis_path is not None:
        target_fa_path = Path(failure_analysis_path).resolve()
    else:
        target_fa_path = project_root / "docs" / "failure_analysis.md"

    generate_failure_analysis_md(individual_runs, trace_map, target_fa_path)

    # Also output failure_analysis.md into run_dir for direct run-dir self-containment
    run_dir_fa_path = run_dir / "failure_analysis.md"
    if run_dir_fa_path != target_fa_path:
        generate_failure_analysis_md(individual_runs, trace_map, run_dir_fa_path)

    return {
        "scores_csv": scores_csv_path,
        "summary_md": summary_md_path,
        "breakdown_md": breakdown_md_path,
        "review_queue_csv": review_queue_path,
        "failure_analysis_md": target_fa_path,
        "run_dir_failure_analysis_md": run_dir_fa_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 6 Report Generator: Produces scores.csv, summary.md, breakdown.md, review_queue.csv, and failure_analysis.md."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to results directory containing scores.json (e.g. results/test_main_1).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Path to labels.json (defaults to labels/labels.json).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Root directory of the project.",
    )
    parser.add_argument(
        "--failure-analysis-path",
        type=Path,
        default=None,
        help="Custom destination for failure_analysis.md (defaults to docs/failure_analysis.md).",
    )

    args = parser.parse_args()

    try:
        results = make_report(
            run_dir=args.run_dir,
            labels_path=args.labels,
            project_root=args.project_root,
            failure_analysis_path=args.failure_analysis_path,
        )
        print("[SUCCESS] Phase 6 reports generated successfully:")
        for name, p in results.items():
            print(f"  - {name}: {p}")
        return 0
    except Exception as e:
        sys.stderr.write(f"[ERROR] Failed to generate reports: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
