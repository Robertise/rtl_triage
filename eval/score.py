#!/usr/bin/env python3
"""
RTL Failure Triage Agent - Independent Automated Grader (score.py)
Scores experiment results strictly according to docs/preregistration.md.

Key Rules:
1. Pure post-hoc scoring: Only reads raw traces, labels, and executes simulation.
   NEVER calls LLMs or Bedrock.
2. Applies final_patch to original bug variant and runs tb/visible and tb/heldout
   via scripts/run_sim.py in a clean temporary directory.
3. A testbench PASS requires cocotb results.xml: tests > 0, failures == 0, errors == 0.
4. Pass_heldout requires passing BOTH tb/visible AND tb/heldout.
5. Pass_first evaluates first_accepted_patch passing BOTH tb/visible AND tb/heldout.
6. Localization (Acc_loc): Predicted line L within [L_start - 3, L_end + 3] on the correct file.
7. Classification (Acc_cls): Exact match on bug_type string.
8. Runs with status FAILED or EXCEEDED_LIMIT receive 0 points for final metrics.
   For EXCEEDED_LIMIT, Pass_first is still evaluated if a first accepted patch exists.
   For FAILED, all patch metrics and Acc_loc/Acc_cls receive 0.
9. INFRA_ERROR runs are reported separately and excluded from N_valid.
10. Decision Rule:
    Delta_k = Pass_heldout(Agent, k) - Pass_heldout(Baseline, k) >= 3 for all k in {1, 2, 3}.
    If repeats are incomplete, reports 'experiment incomplete' without evaluating rule.
"""

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any
import xml.etree.ElementTree as ET

# Ensure workspace root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.patcher import apply_unified_diff

BUG_ID_FROM_NAME_RE = re.compile(r"(B0(?:0[1-9]|1[0-8]))")


@dataclass
class SingleRunScore:
    run_id: str
    config: str  # "agent" or "baseline"
    bug_id: str
    repeat: int
    module: str
    status: str  # "COMPLETED", "FAILED", "EXCEEDED_LIMIT", "INFRA_ERROR"
    stop_reason: str
    is_valid_run: bool  # False if unrecoverable INFRA_ERROR
    is_infra_error: bool
    is_correct_loc: int  # 0 or 1
    is_correct_cls: int  # 0 or 1
    pass_vis: int  # 0 or 1
    pass_heldout: int  # 0 or 1
    pass_first: int  # 0 or 1
    patch_apply_failed: int  # 0 or 1
    llm_turns: int
    sim_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    pred_file: str | None = None
    pred_line: int | None = None
    pred_bug_type: str | None = None
    first_patch_sim_vis: str = "SKIP"
    first_patch_sim_held: str = "SKIP"
    final_patch_sim_vis: str = "SKIP"
    final_patch_sim_held: str = "SKIP"
    notes: str = ""


def load_labels_map(project_root: Path) -> dict[str, dict[str, Any]]:
    """Loads ground-truth labels from labels/labels.json indexed by bug_id."""
    labels_file = project_root / "labels" / "labels.json"
    if not labels_file.is_file():
        raise FileNotFoundError(f"labels/labels.json not found at {labels_file}")
    with open(labels_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["bug_id"]: item for item in data}


def check_xml_results(results_xml_path: Path) -> bool:
    """Verifies that cocotb results.xml has tests > 0, failures == 0, errors == 0."""
    if not results_xml_path.is_file():
        return False
    try:
        tree = ET.parse(results_xml_path)
        root = tree.getroot()
        testsuites = [root] if root.tag == "testsuite" else root.findall("testsuite")
        if not testsuites:
            testsuites = [root]

        total_tests = 0
        total_failures = 0
        total_errors = 0

        for ts in testsuites:
            total_tests += int(ts.attrib.get("tests", 0))
            total_failures += int(ts.attrib.get("failures", 0))
            total_errors += int(ts.attrib.get("errors", 0))

        return total_tests > 0 and total_failures == 0 and total_errors == 0
    except Exception:
        return False


def run_single_simulation(
    project_root: Path,
    module: str,
    patched_rtl_path: Path,
    test_type: str,  # "visible" or "heldout"
    temp_dir: Path,
    timeout: int = 60,
) -> bool:
    """Executes scripts/run_sim.py against visible or heldout testbench."""
    test_dir = project_root / "tb" / test_type
    build_dir = temp_dir / f"sim_build_{test_type}"
    results_xml = temp_dir / f"results_{test_type}.xml"

    cmd = [
        sys.executable,
        str(project_root / "scripts" / "run_sim.py"),
        "--module", module,
        "--src-file", str(patched_rtl_path),
        "--test-dir", str(test_dir),
        "--test-module", f"test_{module}",
        "--build-dir", str(build_dir),
        "--results-xml", str(results_xml),
        "--timeout", str(timeout),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 10)
    except subprocess.TimeoutExpired:
        return False

    return check_xml_results(results_xml)


def evaluate_patch_on_testbenches(
    project_root: Path,
    module: str,
    variant_rtl_content: str,
    patch: str | None,
) -> tuple[bool, bool, bool]:
    """
    Applies patch to variant RTL content and executes visible and heldout simulations.
    Returns (apply_success, visible_passed, heldout_passed).
    """
    if not patch or not isinstance(patch, str) or not patch.strip():
        return False, False, False

    patch_res = apply_unified_diff(variant_rtl_content, patch)
    if not patch_res.success:
        return False, False, False

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        rtl_file = tdp / f"{module}.v"
        rtl_file.write_text(patch_res.patched_text, encoding="utf-8")

        vis_pass = run_single_simulation(
            project_root=project_root,
            module=module,
            patched_rtl_path=rtl_file,
            test_type="visible",
            temp_dir=tdp,
        )
        if not vis_pass:
            return True, False, False

        held_pass = run_single_simulation(
            project_root=project_root,
            module=module,
            patched_rtl_path=rtl_file,
            test_type="heldout",
            temp_dir=tdp,
        )
        return True, vis_pass, held_pass


def score_single_trace(
    trace_path: Path,
    labels_map: dict[str, dict[str, Any]],
    project_root: Path,
    override_config: str | None = None,
    override_bug_id: str | None = None,
    override_repeat: int | None = None,
) -> SingleRunScore:
    """Scores a single trace JSONL file strictly following preregistration rules."""
    trace_events: list[dict[str, Any]] = []
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    trace_events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    run_start = next((e for e in trace_events if e.get("event_type") == "run_start"), {})
    run_end = next((e for e in trace_events if e.get("event_type") == "run_end"), {})

    # Infer metadata strictly (no implicit defaults)
    config = override_config or run_end.get("config") or run_start.get("config")
    if not config:
        path_str = str(trace_path).lower()
        if "baseline" in path_str and "agent" not in path_str:
            config = "baseline"
        elif "agent" in path_str and "baseline" not in path_str:
            config = "agent"
        else:
            raise ValueError(f"Cannot determine config for trace {trace_path}")
    if config not in ("baseline", "agent"):
        raise ValueError(f"Invalid config '{config}' for trace {trace_path} (must be 'baseline' or 'agent')")

    bug_id = override_bug_id or run_end.get("bug_id") or run_start.get("bug_id")
    if not bug_id:
        m = BUG_ID_FROM_NAME_RE.search(str(trace_path))
        if m:
            bug_id = m.group(1)
    if not bug_id:
        raise ValueError(f"Cannot determine bug_id for trace {trace_path}")

    repeat = override_repeat
    if repeat is None:
        repeat = run_end.get("repeat") or run_start.get("repeat")
        if repeat is None:
            # Check parent directory names for integer repeat
            for part in trace_path.parts:
                if part.isdigit() and int(part) in (1, 2, 3):
                    repeat = int(part)
                    break
        if repeat is None:
            # Check for _r1, _r2, _r3 in path or run_id
            m_rep = re.search(r"[_\./]r([1-3])(?:\b|_|\.)", str(trace_path))
            if m_rep:
                repeat = int(m_rep.group(1))
            else:
                rid = str(run_end.get("run_id") or run_start.get("run_id") or "")
                m_rid = re.search(r"_r([1-3])\b", rid)
                if m_rid:
                    repeat = int(m_rid.group(1))
    if repeat is None or repeat not in (1, 2, 3):
        raise ValueError(
            f"Cannot determine valid repeat (1, 2, or 3) for trace {trace_path} (got {repeat})"
        )

    label = labels_map.get(bug_id)
    if not label:
        raise ValueError(f"Unknown bug_id: {bug_id} (not found in labels.json)")

    module = label["module"]
    gt_file_base = Path(label["file"]).name
    gt_line_start = int(label["line_start"])
    gt_line_end = int(label["line_end"])
    gt_bug_type = label["bug_type"]

    run_id = run_end.get("run_id") or run_start.get("run_id") or trace_path.stem
    status = run_end.get("status") or "FAILED"
    stop_reason = run_end.get("stop_reason") or ""
    llm_turns = int(run_end.get("llm_turns", 0))
    sim_calls = int(run_end.get("sim_calls", 0))
    input_tokens = int(run_end.get("total_input_tokens", 0))
    output_tokens = int(run_end.get("total_output_tokens", 0))
    cost_usd = float(run_end.get("total_cost_usd", 0.0))

    first_accepted_patch = run_end.get("first_accepted_patch")
    final_patch = run_end.get("final_patch")
    parsed_output = run_end.get("parsed_output")

    variant_file = project_root / "bugs" / "variants" / bug_id / gt_file_base
    variant_content = variant_file.read_text(encoding="utf-8") if variant_file.is_file() else ""

    # Check INFRA_ERROR
    if status == "INFRA_ERROR":
        return SingleRunScore(
            run_id=run_id,
            config=config,
            bug_id=bug_id,
            repeat=repeat,
            module=module,
            status=status,
            stop_reason=stop_reason,
            is_valid_run=False,
            is_infra_error=True,
            is_correct_loc=0,
            is_correct_cls=0,
            pass_vis=0,
            pass_heldout=0,
            pass_first=0,
            patch_apply_failed=0,
            llm_turns=llm_turns,
            sim_calls=sim_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            notes="Infrastructure failure after max retries; excluded from N_valid.",
        )

    # Localization and Classification checks
    is_correct_loc = 0
    is_correct_cls = 0
    pred_file = None
    pred_line = None
    pred_bug_type = None

    if isinstance(parsed_output, dict):
        pred_file = parsed_output.get("file")
        pred_line = parsed_output.get("line")
        pred_bug_type = parsed_output.get("bug_type")

        # Metric 1: Acc_loc
        norm_file = Path(str(pred_file)).name if pred_file else ""
        if norm_file == gt_file_base and isinstance(pred_line, int):
            if (gt_line_start - 3) <= pred_line <= (gt_line_end + 3):
                is_correct_loc = 1

        # Metric 2: Acc_cls
        if pred_bug_type == gt_bug_type:
            is_correct_cls = 1

    # Preregistration rule: Runs ending in FAILED get 0 points for ALL patch and output metrics
    if status == "FAILED":
        return SingleRunScore(
            run_id=run_id,
            config=config,
            bug_id=bug_id,
            repeat=repeat,
            module=module,
            status=status,
            stop_reason=stop_reason,
            is_valid_run=True,
            is_infra_error=False,
            is_correct_loc=0,
            is_correct_cls=0,
            pass_vis=0,
            pass_heldout=0,
            pass_first=0,
            patch_apply_failed=1 if run_end.get("patch_apply_failed") else 0,
            llm_turns=llm_turns,
            sim_calls=sim_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            pred_file=pred_file,
            pred_line=pred_line,
            pred_bug_type=pred_bug_type,
            notes="Run ended as FAILED; 0 points awarded.",
        )

    # Preregistration rule: EXCEEDED_LIMIT gets 0 for final patch metrics, but can score Pass_first
    pass_first = 0
    first_patch_vis = "SKIP"
    first_patch_held = "SKIP"

    if first_accepted_patch:
        f_applied, f_vis, f_held = evaluate_patch_on_testbenches(
            project_root=project_root,
            module=module,
            variant_rtl_content=variant_content,
            patch=first_accepted_patch,
        )
        first_patch_vis = "PASS" if f_vis else ("FAIL" if f_applied else "APPLY_FAIL")
        first_patch_held = "PASS" if f_held else ("FAIL" if f_applied else "APPLY_FAIL")
        if f_vis and f_held:
            pass_first = 1

    if status == "EXCEEDED_LIMIT":
        return SingleRunScore(
            run_id=run_id,
            config=config,
            bug_id=bug_id,
            repeat=repeat,
            module=module,
            status=status,
            stop_reason=stop_reason,
            is_valid_run=True,
            is_infra_error=False,
            is_correct_loc=is_correct_loc,
            is_correct_cls=is_correct_cls,
            pass_vis=0,
            pass_heldout=0,
            pass_first=pass_first,
            patch_apply_failed=0,
            llm_turns=llm_turns,
            sim_calls=sim_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            pred_file=pred_file,
            pred_line=pred_line,
            pred_bug_type=pred_bug_type,
            first_patch_sim_vis=first_patch_vis,
            first_patch_sim_held=first_patch_held,
            notes="Run ended as EXCEEDED_LIMIT; final metrics scored 0.",
        )

    # COMPLETED run: evaluate final_patch
    pass_vis = 0
    pass_heldout = 0
    patch_apply_failed = 0
    final_patch_vis = "SKIP"
    final_patch_held = "SKIP"

    if not final_patch:
        patch_apply_failed = 1
        final_patch_vis = "NO_PATCH"
        final_patch_held = "NO_PATCH"
    else:
        # Check if final_patch is identical to first_accepted_patch to save simulation time
        if final_patch == first_accepted_patch and first_accepted_patch is not None:
            final_patch_vis = first_patch_vis
            final_patch_held = first_patch_held
            pass_vis = 1 if first_patch_vis == "PASS" else 0
            pass_heldout = 1 if (first_patch_vis == "PASS" and first_patch_held == "PASS") else 0
        else:
            p_applied, p_vis, p_held = evaluate_patch_on_testbenches(
                project_root=project_root,
                module=module,
                variant_rtl_content=variant_content,
                patch=final_patch,
            )
            if not p_applied:
                patch_apply_failed = 1
                final_patch_vis = "APPLY_FAIL"
                final_patch_held = "APPLY_FAIL"
            else:
                final_patch_vis = "PASS" if p_vis else "FAIL"
                final_patch_held = "PASS" if p_held else "FAIL"
                if p_vis:
                    pass_vis = 1
                    if p_held:
                        pass_heldout = 1

    # For baseline, pass_first equals final pass_heldout (single-shot patch)
    if config == "baseline":
        pass_first = pass_heldout
        first_patch_vis = final_patch_vis
        first_patch_held = final_patch_held

    return SingleRunScore(
        run_id=run_id,
        config=config,
        bug_id=bug_id,
        repeat=repeat,
        module=module,
        status=status,
        stop_reason=stop_reason,
        is_valid_run=True,
        is_infra_error=False,
        is_correct_loc=is_correct_loc,
        is_correct_cls=is_correct_cls,
        pass_vis=pass_vis,
        pass_heldout=pass_heldout,
        pass_first=pass_first,
        patch_apply_failed=patch_apply_failed,
        llm_turns=llm_turns,
        sim_calls=sim_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        pred_file=pred_file,
        pred_line=pred_line,
        pred_bug_type=pred_bug_type,
        first_patch_sim_vis=first_patch_vis,
        first_patch_sim_held=first_patch_held,
        final_patch_sim_vis=final_patch_vis,
        final_patch_sim_held=final_patch_held,
        notes="Evaluated normally.",
    )


def aggregate_scores(
    scores: list[SingleRunScore],
    labels_map: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Computes preregistered metric summaries grouped by config and repeat."""
    if labels_map is None:
        labels_map = load_labels_map(_REPO_ROOT)

    # 1. Check for duplicate traces across (config, bug_id, repeat)
    seen_runs: dict[tuple[str, str, int], list[str]] = {}
    for s in scores:
        key = (s.config, s.bug_id, s.repeat)
        if key not in seen_runs:
            seen_runs[key] = []
        seen_runs[key].append(s.run_id)

    duplicates = [k for k, runs in seen_runs.items() if len(runs) > 1]
    if duplicates:
        dup_str = ", ".join(
            f"(config={k[0]}, bug={k[1]}, repeat={k[2]}, count={len(seen_runs[k])})"
            for k in sorted(duplicates)
        )
        raise ValueError(
            f"Duplicate traces detected: {dup_str}. "
            "Each (config, bug, repeat) must have exactly 1 trace."
        )

    # 2. Infer split from scored traces and verify no mixed splits
    splits_found: set[str] = set()
    for s in scores:
        if s.bug_id not in labels_map:
            raise ValueError(f"Unknown bug_id '{s.bug_id}' not found in labels.json")
        splits_found.add(labels_map[s.bug_id]["split"])

    if len(splits_found) > 1:
        raise ValueError(
            f"Mixed splits detected in traces: {sorted(splits_found)}. "
            "All traces must belong to the exact same dataset split."
        )

    inferred_split = next(iter(splits_found)) if splits_found else None
    expected_bugs: list[str] = []
    if inferred_split is not None:
        expected_bugs = sorted([b for b, info in labels_map.items() if info["split"] == inferred_split])

    by_config: dict[str, dict[int, list[SingleRunScore]]] = {
        "baseline": {1: [], 2: [], 3: []},
        "agent": {1: [], 2: [], 3: []},
    }

    for s in scores:
        if s.config in by_config and s.repeat in (1, 2, 3):
            by_config[s.config][s.repeat].append(s)

    def summarize_group(group: list[SingleRunScore]) -> dict[str, Any]:
        n_total = len(group)
        valid = [s for s in group if s.is_valid_run]
        n_valid = len(valid)
        n_infra = n_total - n_valid

        if n_valid == 0:
            return {
                "n_total": n_total,
                "n_valid": 0,
                "n_infra": n_infra,
                "acc_loc_count": 0,
                "acc_loc_rate": 0.0,
                "acc_cls_count": 0,
                "acc_cls_rate": 0.0,
                "pass_vis_count": 0,
                "pass_vis_rate": 0.0,
                "pass_heldout_count": 0,
                "pass_heldout_rate": 0.0,
                "pass_first_count": 0,
                "pass_first_rate": 0.0,
                "mean_turns": 0.0,
                "mean_sims": 0.0,
                "mean_cost_usd": 0.0,
                "total_cost_usd": 0.0,
            }

        acc_loc_count = sum(s.is_correct_loc for s in valid)
        acc_cls_count = sum(s.is_correct_cls for s in valid)
        pass_vis_count = sum(s.pass_vis for s in valid)
        pass_heldout_count = sum(s.pass_heldout for s in valid)
        pass_first_count = sum(s.pass_first for s in valid)

        return {
            "n_total": n_total,
            "n_valid": n_valid,
            "n_infra": n_infra,
            "acc_loc_count": acc_loc_count,
            "acc_loc_rate": acc_loc_count / n_valid,
            "acc_cls_count": acc_cls_count,
            "acc_cls_rate": acc_cls_count / n_valid,
            "pass_vis_count": pass_vis_count,
            "pass_vis_rate": pass_vis_count / n_valid,
            "pass_heldout_count": pass_heldout_count,
            "pass_heldout_rate": pass_heldout_count / n_valid,
            "pass_first_count": pass_first_count,
            "pass_first_rate": pass_first_count / n_valid,
            "mean_turns": sum(s.llm_turns for s in valid) / n_valid,
            "mean_sims": sum(s.sim_calls for s in valid) / n_valid,
            "mean_cost_usd": sum(s.cost_usd for s in valid) / n_valid,
            "total_cost_usd": sum(s.cost_usd for s in valid),
        }

    summary: dict[str, Any] = {"by_config": {}}
    for cfg in ("baseline", "agent"):
        summary["by_config"][cfg] = {}
        all_cfg_scores: list[SingleRunScore] = []
        for rep in (1, 2, 3):
            rep_scores = by_config[cfg][rep]
            all_cfg_scores.extend(rep_scores)
            summary["by_config"][cfg][f"repeat_{rep}"] = summarize_group(rep_scores)
        summary["by_config"][cfg]["overall"] = summarize_group(all_cfg_scores)

    # 3. Check completeness for Decision Rule
    # Decision rule is applied ONLY when each (config, bug, repeat 1..3) has EXACTLY 1 trace
    missing_runs: list[str] = []
    if inferred_split is None or not expected_bugs:
        missing_runs = ["all"]
    else:
        for cfg in ("baseline", "agent"):
            for b in expected_bugs:
                for rep in (1, 2, 3):
                    if (cfg, b, rep) not in seen_runs:
                        missing_runs.append(f"{cfg}/{b}/r{rep}")

    decision_rule_result: dict[str, Any] = {
        "status": "INCOMPLETE",
        "message": (
            f"Experiment incomplete (missing {len(missing_runs)} run(s): "
            f"{', '.join(missing_runs[:10])}{'...' if len(missing_runs) > 10 else ''})."
            if missing_runs
            else "Experiment incomplete (all 3 complete repeats required to apply Decision Rule)."
        ),
        "missing_runs": missing_runs,
        "deltas": {},
        "passed_all_repeats": False,
    }

    if not missing_runs and expected_bugs:
        deltas = {}
        all_pass = True
        for rep in (1, 2, 3):
            held_agent = summary["by_config"]["agent"][f"repeat_{rep}"]["pass_heldout_count"]
            held_base = summary["by_config"]["baseline"][f"repeat_{rep}"]["pass_heldout_count"]
            delta = held_agent - held_base
            deltas[f"repeat_{rep}"] = {
                "agent_pass_heldout": held_agent,
                "baseline_pass_heldout": held_base,
                "delta": delta,
                "threshold_met": delta >= 3,
            }
            if delta < 3:
                all_pass = False

        decision_rule_result["deltas"] = deltas
        decision_rule_result["passed_all_repeats"] = all_pass
        if all_pass:
            decision_rule_result["status"] = "SUPERIOR"
            decision_rule_result["message"] = (
                "Met preregistered superiority threshold: Triage Agent outperforms Baseline "
                "(Delta >= 3 bugs pass held-out across all 3 repeats)."
            )
        else:
            decision_rule_result["status"] = "INSUFFICIENT_EVIDENCE"
            decision_rule_result["message"] = (
                "Insufficient evidence according to the preregistered rule to claim that "
                "Triage Agent outperforms Baseline under the evaluated conditions."
            )

    summary["decision_rule"] = decision_rule_result
    return summary


def find_all_trace_files(target_dir: Path) -> list[Path]:
    """Finds all trace jsonl files within a results directory, excluding incomplete traces."""
    traces: list[Path] = []
    if target_dir.is_file() and target_dir.name.endswith(".jsonl"):
        return [target_dir]
    for p in target_dir.rglob("*.jsonl"):
        if p.name.startswith("trace.incomplete."):
            continue
        if p.name.endswith("_trace.jsonl") or p.name == "trace.jsonl":
            traces.append(p)
    return sorted(traces)
    if target_dir.is_file() and target_dir.name.endswith(".jsonl"):
        return [target_dir]
    for p in target_dir.rglob("*.jsonl"):
        if p.name.endswith("_trace.jsonl") or p.name == "trace.jsonl":
            traces.append(p)
    return sorted(traces)


def print_score_report(scores: list[SingleRunScore], summary: dict[str, Any]) -> None:
    """Prints a structured text report to stdout."""
    print("=" * 80)
    print("RTL Failure Triage Agent - Independent Automated Grader Report")
    print("=" * 80)
    print(f"Total runs evaluated: {len(scores)}\n")

    print(f"{'Run ID':28s} | {'Cfg':8s} | {'Bug':5s} | {'Rep':3s} | {'Status':9s} | {'Loc':3s} | {'Cls':3s} | {'Vis':4s} | {'Held':4s} | {'First':5s} | {'Cost':8s}")
    print("-" * 95)
    for s in scores:
        print(
            f"{s.run_id:28s} | {s.config:8s} | {s.bug_id:5s} | {s.repeat:3d} | {s.status:9s} | "
            f"{s.is_correct_loc:3d} | {s.is_correct_cls:3d} | {s.pass_vis:4d} | {s.pass_heldout:4d} | {s.pass_first:5d} | ${s.cost_usd:.4f}"
        )
    print("-" * 95)

    print("\n--- SUMMARY BY CONFIGURATION ---")
    for cfg in ("baseline", "agent"):
        cfg_data = summary["by_config"].get(cfg, {})
        overall = cfg_data.get("overall", {})
        print(f"\n[{cfg.upper()}] (Overall {overall.get('n_valid', 0)} valid runs):")
        print(f"  Acc_loc:      {overall.get('acc_loc_count', 0)} / {overall.get('n_valid', 0)} ({overall.get('acc_loc_rate', 0.0) * 100:.1f}%)")
        print(f"  Acc_cls:      {overall.get('acc_cls_count', 0)} / {overall.get('n_valid', 0)} ({overall.get('acc_cls_rate', 0.0) * 100:.1f}%)")
        print(f"  Pass_vis:     {overall.get('pass_vis_count', 0)} / {overall.get('n_valid', 0)} ({overall.get('pass_vis_rate', 0.0) * 100:.1f}%)")
        print(f"  Pass_heldout: {overall.get('pass_heldout_count', 0)} / {overall.get('n_valid', 0)} ({overall.get('pass_heldout_rate', 0.0) * 100:.1f}%)")
        print(f"  Pass_first:   {overall.get('pass_first_count', 0)} / {overall.get('n_valid', 0)} ({overall.get('pass_first_rate', 0.0) * 100:.1f}%)")
        print(f"  Mean Turns:   {overall.get('mean_turns', 0.0):.2f}")
        print(f"  Mean Sims:    {overall.get('mean_sims', 0.0):.2f}")
        print(f"  Total Cost:   ${overall.get('total_cost_usd', 0.0):.4f}")

    print("\n--- DECISION RULE ---")
    dr = summary.get("decision_rule", {})
    print(f"Status:  {dr.get('status')}")
    print(f"Message: {dr.get('message')}")
    if dr.get("deltas"):
        for rep, d in dr["deltas"].items():
            print(f"  {rep}: Agent={d['agent_pass_heldout']} vs Base={d['baseline_pass_heldout']} -> Delta={d['delta']} (Threshold >= 3: {d['threshold_met']})")
    print("=" * 80)


def main() -> int:
    parser = argparse.ArgumentParser(description="Independent Automated Grader for RTL Triage Traces")
    parser.add_argument("--results-dir", "-d", type=str, default="results", help="Directory containing trace files")
    parser.add_argument("--trace", "-t", type=str, default=None, help="Path to single trace file to score")
    parser.add_argument("--out-json", "-o", type=str, default=None, help="Path to save output JSON")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    labels_map = load_labels_map(project_root)

    trace_files: list[Path] = []
    if args.trace:
        tp = Path(args.trace).resolve()
        if not tp.is_file():
            sys.stderr.write(f"[ERROR] Trace file not found: {tp}\n")
            return 1
        trace_files = [tp]
    else:
        rd = Path(args.results_dir).resolve()
        if not rd.exists():
            sys.stderr.write(f"[ERROR] Results directory not found: {rd}\n")
            return 1
        trace_files = find_all_trace_files(rd)

    if not trace_files:
        sys.stderr.write(f"[ERROR] No trace files found in {args.results_dir}\n")
        return 1

    scores: list[SingleRunScore] = []
    for tf in trace_files:
        try:
            score = score_single_trace(
                trace_path=tf,
                labels_map=labels_map,
                project_root=project_root,
            )
            scores.append(score)
        except Exception as exc:
            sys.stderr.write(f"[ERROR] Error scoring {tf}: {exc}\n")
            return 1

    try:
        summary = aggregate_scores(scores, labels_map=labels_map)
    except Exception as exc:
        sys.stderr.write(f"[ERROR] Aggregation error: {exc}\n")
        return 1

    print_score_report(scores, summary)

    if args.out_json:
        out_path = Path(args.out_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dump_data = {
            "individual_runs": [asdict(s) for s in scores],
            "summary": summary,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dump_data, f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] Wrote detailed score results to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
