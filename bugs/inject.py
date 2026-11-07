#!/usr/bin/env python3
"""
RTL Failure Triage Agent - Bug Injection Script
Reads bugs/catalog.yaml, creates isolated bug variants under bugs/variants/<bug_id>/<file>,
verifies single match substitution, ensures clean output free of metadata/clues,
runs simulations to determine actual visible_fails and heldout_fails,
and generates labels/labels.json.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
import yaml

FORBIDDEN_WORD_PATTERN = re.compile(r"\b(bug|bugs|mutant|mutants|fix|fixes)\b", re.IGNORECASE)


def run_sim_check(module: str, src_file: Path, test_dir: str, timeout: int = 30) -> tuple[bool, str]:
    """
    Runs simulation via scripts/run_sim.py and returns (fails: bool, first_error_msg: str).
    Returns (True, message) if simulation fails (as expected for a bug).
    """
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
        data = json.loads(proc.stdout)
        status = data.get("status", "FAIL")
        msg = data.get("message", "")
    except json.JSONDecodeError:
        status = "FAIL"
        msg = proc.stderr.strip() or proc.stdout.strip()

    fails = (status != "PASS")
    return fails, msg


def inject_bugs(catalog_path: Path, variants_dir: Path, labels_out: Path, run_sim: bool = True):
    if not catalog_path.exists():
        sys.stderr.write(f"Error: Catalog file not found: {catalog_path}\n")
        sys.exit(1)

    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = yaml.safe_load(f)

    bugs = catalog.get("bugs", [])
    if not bugs:
        sys.stderr.write("Error: No bugs found in catalog.\n")
        sys.exit(1)

    print(f"Loaded {len(bugs)} bugs from {catalog_path}")
    variants_dir.mkdir(parents=True, exist_ok=True)
    labels_out.parent.mkdir(parents=True, exist_ok=True)

    labels_data = []

    for bug in bugs:
        bug_id = bug["bug_id"]
        split = bug["split"]
        module = bug["module"]
        src_rel = bug["file"]
        file_name = Path(src_rel).name
        line_start = bug["line_start"]
        line_end = bug["line_end"]
        bug_type = bug["bug_type"]
        description = bug.get("description") or bug.get("expected_symptom", "")
        correct_fix = bug["correct_fix"]
        alt_fixes = bug.get("alternative_fixes", [])

        orig_snippet = bug["original"]
        mut_snippet = bug["mutated"]

        # Read original RTL
        src_path = Path(src_rel)
        if not src_path.exists():
            sys.stderr.write(f"Error: Original source file not found: {src_path}\n")
            sys.exit(1)

        with open(src_path, "r", encoding="utf-8") as f:
            orig_content = f.read()

        # Enforce exact single match requirement
        match_count = orig_content.count(orig_snippet)
        if match_count != 1:
            sys.stderr.write(
                f"Error in {bug_id}: 'original' snippet matched {match_count} times in {src_path}. Must match exactly once.\n"
            )
            sys.exit(1)

        # Mutate
        mut_content = orig_content.replace(orig_snippet, mut_snippet, 1)

        if mut_content == orig_content:
            sys.stderr.write(f"Error in {bug_id}: Mutated content is identical to original.\n")
            sys.exit(1)

        # Cleanliness verification: No bug_id, no bug_type, no standalone forbidden words
        if bug_id.lower() in mut_content.lower():
            sys.stderr.write(f"Error in {bug_id}: Bug ID '{bug_id}' leaked into variant content.\n")
            sys.exit(1)

        if bug_type.lower() in mut_content.lower():
            sys.stderr.write(f"Error in {bug_id}: Bug type '{bug_type}' leaked into variant content.\n")
            sys.exit(1)

        forbidden_matches = FORBIDDEN_WORD_PATTERN.findall(mut_snippet)
        if forbidden_matches:
            sys.stderr.write(
                f"Error in {bug_id}: Forbidden word(s) {forbidden_matches} present in mutation snippet.\n"
            )
            sys.exit(1)

        # Write mutated variant file
        bug_dir = variants_dir / bug_id
        bug_dir.mkdir(parents=True, exist_ok=True)
        dest_file = bug_dir / file_name

        with open(dest_file, "w", encoding="utf-8", newline="\n") as f:
            f.write(mut_content)

        # Run simulations to determine real visible_fails and heldout_fails
        visible_fails = True
        heldout_fails = True
        vis_msg = ""
        hld_msg = ""

        if run_sim:
            print(f"  Simulating {bug_id} ({module}, {split})...", flush=True)
            vis_fails, vis_msg = run_sim_check(module, dest_file, "tb/visible")
            hld_fails, hld_msg = run_sim_check(module, dest_file, "tb/heldout")

            visible_fails = vis_fails
            heldout_fails = hld_fails
            print(
                f"    -> visible_fails={visible_fails} ({vis_msg[:45]}), heldout_fails={heldout_fails}",
                flush=True,
            )

        label_entry = {
            "bug_id": bug_id,
            "split": split,
            "module": module,
            "file": file_name,
            "line_start": line_start,
            "line_end": line_end,
            "bug_type": bug_type,
            "description": description,
            "correct_fix": correct_fix,
            "alternative_fixes": alt_fixes,
            "visible_fails": visible_fails,
            "heldout_fails": heldout_fails,
        }
        labels_data.append(label_entry)

    # Save labels.json deterministically
    with open(labels_out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(labels_data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\nSuccessfully generated {len(bugs)} bug variants in {variants_dir}/")
    print(f"Generated {labels_out} with {len(labels_data)} labels.")


def main():
    parser = argparse.ArgumentParser(description="Inject bugs into RTL files and generate labels.json.")
    parser.add_argument("--catalog", default="bugs/catalog.yaml", help="Path to bugs/catalog.yaml")
    parser.add_argument("--variants-dir", default="bugs/variants", help="Target variants directory")
    parser.add_argument("--labels-out", default="labels/labels.json", help="Target labels.json path")
    parser.add_argument("--no-sim", action="store_true", help="Skip simulation and assume failure flags")

    args = parser.parse_args()
    inject_bugs(
        catalog_path=Path(args.catalog),
        variants_dir=Path(args.variants_dir),
        labels_out=Path(args.labels_out),
        run_sim=not args.no_sim,
    )


if __name__ == "__main__":
    main()
