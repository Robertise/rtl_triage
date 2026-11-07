#!/usr/bin/env python3
"""
Verification script for Phase 2: Runs visible and held-out testbenches 3 times
each for all 4 modules, records raw JSON results, and confirms 100% pass consistency.
"""

import json
import subprocess
import sys

MODULES = ["counter", "alu", "fifo", "fsm_controller"]
TEST_TYPES = ["visible", "heldout"]

def run_test(module, test_type):
    cmd = [
        "python3", "scripts/run_sim.py",
        "--module", module,
        "--test-dir", f"tb/{test_type}",
        "--test-module", f"test_{module}",
        "--timeout", "30",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        data = {"status": "FAIL", "raw": proc.stdout, "stderr": proc.stderr}
    return proc.returncode, data

def main():
    all_results = {}
    mismatches = 0

    for module in MODULES:
        for ttype in TEST_TYPES:
            key = f"{module} ({ttype})"
            print(f"Testing {key} (3 repeats)...", flush=True)
            runs = []
            for r in range(1, 4):
                ret, data = run_test(module, ttype)
                runs.append((ret, data))
                print(f"  Run {r}: status={data.get('status')}, tests={data.get('tests')}, exit={ret}", flush=True)

            # Verify identical results
            first_ret, first_data = runs[0]
            for r, (ret, data) in enumerate(runs[1:], 2):
                if ret != first_ret or data.get("status") != first_data.get("status") or data.get("tests") != first_data.get("tests") or data.get("failures") != first_data.get("failures"):
                    print(f"  MISMATCH on run {r}!", flush=True)
                    mismatches += 1

            all_results[key] = [r[1] for r in runs]

    print("\n=== SUMMARY OF 3 RUNS ===")
    print(f"Total Suites Tested: {len(all_results)}")
    print(f"Inconsistencies: {mismatches}")
    if mismatches > 0:
        sys.exit(1)

    with open("sim_build/phase2_3runs.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print("Results saved to sim_build/phase2_3runs.json")

if __name__ == "__main__":
    main()
