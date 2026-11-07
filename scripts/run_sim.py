#!/usr/bin/env python3
"""
RTL Failure Triage Agent - Simulation Runner Script
Executes Verilog simulation using Icarus Verilog and cocotb.
Enforces process-group timeout, parses results.xml, and returns structured JSON output.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def run_worker_main(args):
    """Internal simulation runner invoked as a worker process."""
    if shutil.which("iverilog") is None:
        sys.stderr.write("Error: 'iverilog' executable not found on PATH.\n")
        sys.exit(1)

    try:
        from cocotb_tools.runner import get_runner
    except ImportError:
        try:
            from cocotb.runner import get_runner
        except ImportError:
            sys.stderr.write("Error: cocotb is not installed in the current environment.\n")
            sys.exit(1)

    src_file = Path(args.src_file).resolve()
    test_dir = Path(args.test_dir).resolve()
    build_dir = Path(args.build_dir).resolve()
    results_xml = Path(args.results_xml).resolve()

    build_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous results.xml if any
    if results_xml.exists():
        try:
            results_xml.unlink()
        except OSError:
            pass

    stage = "init"
    try:
        runner = get_runner("icarus")

        if getattr(args, "waves", False):
            orig_test_command = runner._test_command

            def vcd_create_dump():
                vcd_path = str(build_dir / f"{args.module}.vcd")
                with open(runner.iverilog_dump_file, "w") as f:
                    f.write("module cocotb_iverilog_dump();\n")
                    f.write("initial begin\n")
                    f.write(f'    $dumpfile("{vcd_path}");\n')
                    f.write(f"    $dumpvars(0, {args.module});\n")
                    f.write("end\n")
                    f.write("endmodule\n")

            def vcd_test_command():
                cmds = orig_test_command()
                for c in cmds:
                    for i, a in enumerate(c):
                        if a == "-fst":
                            c[i] = "-vcd"
                return cmds

            runner._create_iverilog_dump_file = vcd_create_dump
            runner._test_command = vcd_test_command

        # Step 1: Build HDL
        stage = "hdl_build"
        sys.stderr.write("[RUN_SIM_STAGE:HDL_BUILD]\n")
        sys.stderr.flush()
        runner.build(
            sources=[src_file],
            hdl_toplevel=args.module,
            build_dir=build_dir,
            always=True,
            waves=getattr(args, "waves", False),
        )

        # Step 2: Run cocotb testbench
        stage = "sim_test"
        sys.stderr.write("[RUN_SIM_STAGE:SIM_TEST]\n")
        sys.stderr.flush()
        runner.test(
            hdl_toplevel=args.module,
            test_module=args.test_module,
            test_dir=test_dir,
            build_dir=build_dir,
            results_xml=str(results_xml),
            waves=getattr(args, "waves", False),
        )
    except (Exception, SystemExit) as exc:
        sys.stderr.write(f"\n[RUN_SIM_FAILED_STAGE:{stage}] {type(exc).__name__}: {exc}\n")
        sys.stderr.flush()
        sys.exit(1)


def parse_results_xml(xml_path: Path):
    """
    Parses cocotb JUnit-compatible results.xml.
    Returns dict with tests, failures, errors, and failure messages.
    """
    if not xml_path.exists():
        return None

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return None

    total_tests = 0
    total_failures = 0
    total_errors = 0
    failure_messages = []

    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")

    for suite in suites:
        total_tests += int(suite.attrib.get("tests", 0))
        total_failures += int(suite.attrib.get("failures", 0))
        total_errors += int(suite.attrib.get("errors", 0))

        for failure in suite.iter("failure"):
            msg = failure.attrib.get("message") or failure.text or "Assertion failed"
            failure_messages.append(msg.strip())

        for error in suite.iter("error"):
            msg = error.attrib.get("message") or error.text or "Simulation error"
            failure_messages.append(msg.strip())

    return {
        "tests": total_tests,
        "failures": total_failures,
        "errors": total_errors,
        "messages": failure_messages,
    }


def main():
    parser = argparse.ArgumentParser(description="Run RTL simulation with timeout and XML evaluation.")
    parser.add_argument("--module", required=True, help="Toplevel Verilog module name")
    parser.add_argument("--src-file", help="Path to Verilog source file (defaults to rtl/<module>.v or smoke/<module>.v)")
    parser.add_argument("--test-dir", help="Directory containing cocotb testbench (defaults to tb/visible or smoke)")
    parser.add_argument("--test-module", help="Python test module name (defaults to test_<module>)")
    parser.add_argument("--timeout", type=int, default=60, help="Simulation timeout in seconds (default: 60)")
    parser.add_argument("--build-dir", default="sim_build", help="Directory for build artifacts and logs")
    parser.add_argument("--results-xml", help="Target path for results.xml (defaults to <build_dir>/results.xml)")
    parser.add_argument("--out-json", help="Path to write output JSON to")
    parser.add_argument("--waves", action="store_true", help="Enable VCD waveform generation")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    # If called as worker subprocess, execute the simulation directly
    if args.worker:
        run_worker_main(args)
        return

    # Check for iverilog on PATH
    if shutil.which("iverilog") is None:
        result_json = {
            "status": "FAIL",
            "tests": 0,
            "failures": 0,
            "errors": 0,
            "log_path": "",
            "message": "Icarus Verilog ('iverilog') executable not found on PATH. Simulation environment required."
        }
        json_output = json.dumps(result_json, indent=2)
        print(json_output)
        if args.out_json:
            with open(args.out_json, "w", encoding="utf-8") as f:
                f.write(json_output)
        sys.exit(1)

    # Parent runner logic
    project_root = Path(__file__).resolve().parent.parent

    # Resolve default paths
    if not args.src_file:
        candidate_paths = [
            project_root / "rtl" / f"{args.module}.v",
            project_root / "smoke" / f"{args.module}.v",
        ]
        for p in candidate_paths:
            if p.exists():
                args.src_file = str(p)
                break
        if not args.src_file:
            args.src_file = str(candidate_paths[0])

    if not args.test_dir:
        candidate_dirs = [
            project_root / "tb" / "visible",
            project_root / "smoke",
        ]
        for d in candidate_dirs:
            if (d / f"test_{args.module}.py").exists():
                args.test_dir = str(d)
                break
        if not args.test_dir:
            args.test_dir = str(candidate_dirs[0])

    if not args.test_module:
        args.test_module = f"test_{args.module}"

    build_dir = Path(args.build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    log_path = build_dir / "sim.log"

    if not args.results_xml:
        args.results_xml = str(build_dir / "results.xml")
    results_xml = Path(args.results_xml).resolve()

    # Clean existing results.xml
    if results_xml.exists():
        try:
            results_xml.unlink()
        except OSError:
            pass

    # Construct worker command
    worker_cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--module", args.module,
        "--src-file", args.src_file,
        "--test-dir", args.test_dir,
        "--test-module", args.test_module,
        "--build-dir", str(build_dir),
        "--results-xml", str(results_xml),
    ]
    if args.waves:
        worker_cmd.append("--waves")

    timed_out = False
    return_code = 0

    # Process group setup: os.setsid on Linux ensures all children (including vvp) share a process group
    preexec = os.setsid if hasattr(os, "setsid") else None
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0

    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            worker_cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=preexec,
            creationflags=creationflags,
            text=True,
        )
        try:
            proc.wait(timeout=args.timeout)
            return_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            log_file.write(f"\n[ERROR] Simulation timed out after {args.timeout} seconds.\n")
            log_file.flush()

            # Kill the entire process group to ensure worker and any spawned simulator (vvp) are terminated
            if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except OSError:
                    pass
            elif sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
            else:
                proc.kill()

            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    # Evaluate outcome strictly according to preregistration criteria
    status = "FAIL"
    tests = 0
    failures = 0
    errors = 0
    message = ""

    if timed_out:
        status = "FAIL"
        message = f"Simulation timed out after {args.timeout} seconds"
    else:
        xml_summary = parse_results_xml(results_xml)
        if xml_summary is None:
            status = "FAIL"
            failed_stage = None
            log_lines = []
            if log_path.exists():
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    log_lines = [line.strip() for line in f]

            # Detect failed stage from worker markers
            for line in log_lines:
                if "[RUN_SIM_FAILED_STAGE:" in line:
                    failed_stage = line.split("[RUN_SIM_FAILED_STAGE:")[1].split("]")[0].strip()
                    break

            # Distinguish HDL compilation/elaboration errors from Python/test errors
            is_hdl_error = (failed_stage == "hdl_build")
            hdl_err_line = None
            for line in log_lines:
                # Common iverilog failure signatures
                if any(v_ext in line for v_ext in [".v:", ".sv:"]):
                    hdl_err_line = line
                    is_hdl_error = True
                    break
                if any(k in line.lower() for k in ["syntax error", "errors during elaboration", "error during elaboration"]):
                    hdl_err_line = line
                    is_hdl_error = True
                    break

            if is_hdl_error:
                if not hdl_err_line:
                    for line in log_lines:
                        if line.lower().startswith("error:"):
                            hdl_err_line = line
                            break
                detail = hdl_err_line if hdl_err_line else "HDL build failed"
                message = f"Compilation / elaboration failed: {detail}"
            else:
                # Python / test error
                py_err_line = None
                for line in reversed(log_lines):
                    if any(line.startswith(err_type) for err_type in [
                        "ImportError", "ModuleNotFoundError", "SyntaxError",
                        "NameError", "AttributeError", "TypeError", "ValueError",
                        "AssertionError", "RuntimeError", "Exception", "Error"
                    ]):
                        py_err_line = line
                        break
                if not py_err_line:
                    for line in log_lines:
                        if "traceback (most recent call last)" in line.lower():
                            py_err_line = line
                            break
                if py_err_line:
                    message = f"Python / test execution failed: {py_err_line}"
                elif failed_stage:
                    message = f"Python / test execution failed during stage: {failed_stage}"
                else:
                    message = f"Missing or unparseable results.xml (Exit code: {return_code})"
        else:
            tests = xml_summary["tests"]
            failures = xml_summary["failures"]
            errors = xml_summary["errors"]

            # Preregistration rule: tests > 0 and failures == 0 and errors == 0
            if tests > 0 and failures == 0 and errors == 0:
                status = "PASS"
                message = f"All {tests} test(s) passed successfully."
            else:
                status = "FAIL"
                first_msg = xml_summary["messages"][0] if xml_summary["messages"] else "Assertion or simulation error"
                first_msg_clean = first_msg.splitlines()[0] if first_msg else ""
                message = f"Simulation failed ({failures} failure(s), {errors} error(s)). {first_msg_clean}"

    result_json = {
        "status": status,
        "tests": tests,
        "failures": failures,
        "errors": errors,
        "log_path": str(log_path),
        "message": message,
    }

    # Print JSON output to stdout
    json_output = json.dumps(result_json, indent=2)
    print(json_output)

    # Save to file if requested
    if args.out_json:
        out_json_path = Path(args.out_json).resolve()
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            f.write(json_output)

    # Exit code: 0 if PASS, 1 if FAIL
    sys.exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
