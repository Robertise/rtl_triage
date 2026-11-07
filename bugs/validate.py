#!/usr/bin/env python3
"""
RTL Failure Triage Agent - Bug Validator Script
Performs strict, automated validation across all 18 bugs in Docker:
1. Verifies dataset split (15 test, 3 dev; test set covers 7 bug types and 4 modules).
2. Verifies variants cleanliness (no bug_id, bug_type, or clue words like bug/mutant/fix).
3. Verifies unique diffs and confirms diff touches labeled [line_start, line_end].
4. Verifies compilation and ensures tb/visible FAILS on mutated RTL.
5. Verifies applying correct_fix restores 100% byte-identical RTL and PASSES both tb/visible and tb/heldout.
6. Verifies tb/heldout behavior and synchronizes with labels/labels.json.
Exits with non-zero code if any validation check fails. Never silently drops bugs.
"""

import argparse
import difflib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
import yaml

FORBIDDEN_WORD_PATTERN = re.compile(r"\b(bug|bugs|mutant|mutants|fix|fixes)\b", re.IGNORECASE)
EXPECTED_BUG_TYPES = {
    "fifo_flag",
    "fsm_invalid_transition",
    "reset_polarity",
    "blocking_nonblocking",
    "bit_width_mismatch",
    "missing_branch",
    "off_by_one",
}
EXPECTED_MODULES = {"counter", "alu", "fifo", "fsm_controller"}


def run_simulation(module: str, src_file: Path, test_dir: str, timeout: int = 30) -> dict:
    """Executes scripts/run_sim.py and returns parsed output dictionary."""
    cmd = [
        sys.executable,
        "scripts/run_sim.py",
        "--module", module,
        "--src-file", str(src_file),
        "--test-dir", test_dir,
        "--test-module", f"test_{module}",
        "--timeout", str(timeout),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            "status": "FAIL",
            "tests": 0,
            "failures": 1,
            "errors": 1,
            "message": f"Execution failed: {proc.stderr.strip() or proc.stdout.strip()}",
        }


def validate_catalog_and_variants(
    catalog_path: Path,
    variants_dir: Path,
    labels_path: Path,
) -> bool:
    print("=" * 80)
    print("RTL Failure Triage Agent - Phase 3 Automated Validation")
    print("=" * 80)

    errors = []

    # 1. Load catalog
    if not catalog_path.exists():
        print(f"[FAIL] Catalog not found: {catalog_path}")
        return False

    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = yaml.safe_load(f)

    bugs = catalog.get("bugs", [])
    print(f"[CHECK 1] Catalog loaded: {len(bugs)} bugs defined.")

    # 2. Check split distribution
    test_bugs = [b for b in bugs if b["split"] == "test"]
    dev_bugs = [b for b in bugs if b["split"] == "dev"]

    if len(test_bugs) != 15 or len(dev_bugs) != 3 or len(bugs) != 18:
        msg = f"Invalid bug split: test={len(test_bugs)} (expected 15), dev={len(dev_bugs)} (expected 3), total={len(bugs)} (expected 18)"
        print(f"[FAIL] {msg}")
        errors.append(msg)
    else:
        print("[PASS] Split count: exactly 15 test bugs and 3 dev bugs (total 18).")

    test_types = set(b["bug_type"] for b in test_bugs)
    if test_types != EXPECTED_BUG_TYPES:
        missing = EXPECTED_BUG_TYPES - test_types
        msg = f"Test set missing bug type(s): {missing}"
        print(f"[FAIL] {msg}")
        errors.append(msg)
    else:
        print("[PASS] Test set bug types: all 7 distinct bug types represented.")

    test_modules = set(b["module"] for b in test_bugs)
    if test_modules != EXPECTED_MODULES:
        missing = EXPECTED_MODULES - test_modules
        msg = f"Test set missing module(s): {missing}"
        print(f"[FAIL] {msg}")
        errors.append(msg)
    else:
        print("[PASS] Test set modules: all 4 modules represented.")

    # 3. Check labels.json consistency
    labels_map = {}
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            labels_list = json.load(f)
        labels_map = {l["bug_id"]: l for l in labels_list}
        print(f"[PASS] labels/labels.json loaded ({len(labels_map)} entries).")
    else:
        msg = f"labels file missing: {labels_path}"
        print(f"[FAIL] {msg}")
        errors.append(msg)

    # 4. Check uniqueness of diffs and cleanliness of variants
    diff_registry = {}
    validation_records = []

    temp_patch_dir = Path("sim_build/tmp_validate_patches")
    temp_patch_dir.mkdir(parents=True, exist_ok=True)

    for bug in bugs:
        bug_id = bug["bug_id"]
        split = bug["split"]
        module = bug["module"]
        src_rel = bug["file"]
        file_name = Path(src_rel).name
        line_start = bug["line_start"]
        line_end = bug["line_end"]
        bug_type = bug["bug_type"]

        variant_file = variants_dir / bug_id / file_name

        if not variant_file.exists():
            msg = f"{bug_id}: Variant file does not exist: {variant_file}"
            print(f"[FAIL] {msg}")
            errors.append(msg)
            continue

        with open(src_rel, "r", encoding="utf-8") as f:
            orig_content = f.read()

        with open(variant_file, "r", encoding="utf-8") as f:
            mut_content = f.read()

        # Cleanliness check: no leaked bug_id, bug_type, or forbidden words in mutated snippet
        if bug_id.lower() in mut_content.lower():
            msg = f"{bug_id}: Bug ID leaked into {variant_file}"
            print(f"[FAIL] {msg}")
            errors.append(msg)

        if bug_type.lower() in mut_content.lower():
            msg = f"{bug_id}: Bug type '{bug_type}' leaked into {variant_file}"
            print(f"[FAIL] {msg}")
            errors.append(msg)

        forbidden_in_mut = FORBIDDEN_WORD_PATTERN.findall(bug["mutated"])
        if forbidden_in_mut:
            msg = f"{bug_id}: Forbidden word(s) {forbidden_in_mut} in mutation snippet"
            print(f"[FAIL] {msg}")
            errors.append(msg)

        # Diff check
        orig_lines = orig_content.splitlines(keepends=True)
        mut_lines = mut_content.splitlines(keepends=True)

        diff_str = "".join(
            difflib.unified_diff(
                orig_lines,
                mut_lines,
                fromfile=f"a/{file_name}",
                tofile=f"b/{file_name}",
                n=0,
            )
        )

        if not diff_str.strip():
            msg = f"{bug_id}: Variant is identical to original RTL (empty diff)"
            print(f"[FAIL] {msg}")
            errors.append(msg)
            continue

        # Check duplicate diffs
        if diff_str in diff_registry:
            msg = f"{bug_id}: Duplicate diff with {diff_registry[diff_str]}"
            print(f"[FAIL] {msg}")
            errors.append(msg)
        diff_registry[diff_str] = bug_id

        # Check diff touches [line_start, line_end]
        matcher = difflib.SequenceMatcher(None, orig_lines, mut_lines)
        changed_orig_lines = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                for l in range(i1 + 1, i2 + 1):
                    changed_orig_lines.append(l)

        # Diff must touch or overlap [line_start, line_end]
        # Allow +/- 3 lines tolerance in accordance with preregistered localization boundary
        touches_label = any(
            (line_start - 3 <= l <= line_end + 3) for l in changed_orig_lines
        )
        if not touches_label and changed_orig_lines:
            msg = f"{bug_id}: Diff touches lines {changed_orig_lines}, which does not overlap label [{line_start}, {line_end}]"
            print(f"[FAIL] {msg}")
            errors.append(msg)

        # 5. Simulation checks
        # 5a. Mutated variant MUST compile
        # 5b. tb/visible MUST fail
        print(f"Testing {bug_id} ({module}, {split})...", flush=True)
        vis_res = run_simulation(module, variant_file, "tb/visible")
        vis_status = vis_res.get("status")
        vis_msg = vis_res.get("message", "")

        if "Compilation / elaboration failed" in vis_msg:
            msg = f"{bug_id}: Mutated RTL failed to compile/elaborate: {vis_msg}"
            print(f"  [FAIL] {msg}")
            errors.append(msg)

        if vis_status != "FAIL":
            msg = f"{bug_id}: Expected tb/visible to FAIL, but got {vis_status}"
            print(f"  [FAIL] {msg}")
            errors.append(msg)
        else:
            print(f"  [PASS] tb/visible FAIL confirmed: {vis_msg[:50]}")

        # 5c. Mutated variant against tb/heldout
        hld_res = run_simulation(module, variant_file, "tb/heldout")
        hld_status = hld_res.get("status")
        hld_msg = hld_res.get("message", "")
        print(f"  [INFO] tb/heldout status: {hld_status}")

        # 5d. Apply correct_fix and verify exact identity and PASS on both suites
        patched_content = mut_content.replace(bug["mutated"], bug["correct_fix"], 1)
        if patched_content != orig_content:
            msg = f"{bug_id}: Applying correct_fix did not restore original RTL exactly."
            print(f"  [FAIL] {msg}")
            errors.append(msg)
        else:
            # Test patched file
            patched_file = temp_patch_dir / f"patched_{bug_id}_{file_name}"
            with open(patched_file, "w", encoding="utf-8", newline="\n") as pf:
                pf.write(patched_content)

            p_vis = run_simulation(module, patched_file, "tb/visible")
            p_hld = run_simulation(module, patched_file, "tb/heldout")

            if p_vis.get("status") != "PASS":
                msg = f"{bug_id}: Patched file failed tb/visible: {p_vis.get('message')}"
                print(f"  [FAIL] {msg}")
                errors.append(msg)

            if p_hld.get("status") != "PASS":
                msg = f"{bug_id}: Patched file failed tb/heldout: {p_hld.get('message')}"
                print(f"  [FAIL] {msg}")
                errors.append(msg)

            if p_vis.get("status") == "PASS" and p_hld.get("status") == "PASS":
                print(f"  [PASS] Patched file passed both visible and held-out tests.")

            # Clean temporary patched file
            try:
                patched_file.unlink()
            except OSError:
                pass

        # 5e. Cross-check labels.json entry
        if bug_id in labels_map:
            lbl = labels_map[bug_id]
            if lbl.get("visible_fails") is not (vis_status != "PASS"):
                msg = f"{bug_id}: labels.json visible_fails mismatch with simulation result"
                print(f"  [FAIL] {msg}")
                errors.append(msg)

        validation_records.append({
            "bug_id": bug_id,
            "split": split,
            "module": module,
            "bug_type": bug_type,
            "lines": f"[{line_start}, {line_end}]",
            "vis_status": vis_status,
            "vis_msg": vis_msg,
            "hld_status": hld_status,
            "hld_msg": hld_msg,
        })

    # Cleanup temp patch directory
    shutil.rmtree(temp_patch_dir, ignore_errors=True)

    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY TABLE")
    print("=" * 80)
    print(f"{'ID':<6} {'Split':<6} {'Module':<15} {'Bug Type':<24} {'Lines':<10} {'Visible':<8} {'Heldout':<8}")
    print("-" * 80)
    for rec in validation_records:
        print(
            f"{rec['bug_id']:<6} {rec['split']:<6} {rec['module']:<15} "
            f"{rec['bug_type']:<24} {rec['lines']:<10} {rec['vis_status']:<8} {rec['hld_status']:<8}"
        )
    print("=" * 80)

    if errors:
        print(f"\n[FAIL] Validation finished with {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print(f"\n[ALL PASS] 100% of {len(bugs)} bugs passed all automated criteria perfectly.")
        return True


def main():
    parser = argparse.ArgumentParser(description="Validate all bug variants and labels.")
    parser.add_argument("--catalog", default="bugs/catalog.yaml", help="Path to bugs/catalog.yaml")
    parser.add_argument("--variants-dir", default="bugs/variants", help="Variants directory")
    parser.add_argument("--labels", default="labels/labels.json", help="Path to labels/labels.json")

    args = parser.parse_args()
    success = validate_catalog_and_variants(
        catalog_path=Path(args.catalog),
        variants_dir=Path(args.variants_dir),
        labels_path=Path(args.labels),
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
