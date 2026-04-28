#!/usr/bin/env python3

import os
import sys
import subprocess
from typing import List, Tuple


def _print_header() -> None:
    title = "HYBRID BLOCKCHAIN SECURITY SIMULATOR"
    line = "=" * len(title)
    print("\n" + line)
    print(title)
    print(line + "\n")


def _run_step(script_name: str, args: List[str]) -> Tuple[bool, int]:
    cmd = [sys.executable, script_name, *args]
    print(f"--- In Progress: {script_name} ---")
    print("Command:")
    print(" ".join(cmd))
    print("")

    proc = subprocess.run(cmd, text=True)

    if proc.returncode != 0:
        print("")
        print("""ERROR: A pipeline step failed.

Step: {step}
Exit code: {code}

Stopping execution.
""".format(step=script_name, code=proc.returncode))
        return False, proc.returncode

    print(f"--- Completed: {script_name} ---\n")
    return True, 0


def _print_post_run_map(project_root: str) -> None:
    results_dir = os.path.join(project_root, "results_output")

    print("\n" + "=" * 60)
    print("POST-RUN SUMMARY")
    print("=" * 60)
    print("")

    print("Project root:")
    print(f"  {project_root}")
    print("")

    print("Key outputs:")
    print(f"  - Master dataset: {os.path.join(project_root, 'blockchain_temporal_master_dataset.csv')}")
    print(f"  - Sleeper evidence: {os.path.join(project_root, 'sleeper_node_evidence.csv')}")
    print("")

    print("Check /results_output for charts and reports:")
    print(f"  - Killer Chart (RF vs Hybrid): {os.path.join(results_dir, 'robustness_plot.png')}")
    print(f"  - Sleeper Signature: {os.path.join(results_dir, 'sleeper_node3_reputation.png')}")
    print(f"  - Benchmark table: {os.path.join(results_dir, 'benchmark_comparison.csv')}")
    print(f"  - Heatmap: {os.path.join(results_dir, 'research_feature_heatmap.png')}")
    print(f"  - Stats: {os.path.join(results_dir, 'project_final_stats.txt')}")
    print("")

    if os.path.isdir(results_dir):
        try:
            items = sorted(os.listdir(results_dir))
        except OSError:
            items = []
        print("Results directory contents:")
        for name in items:
            print(f"  - {name}")
    else:
        print("NOTE: results_output folder was not found. If this is unexpected, re-run this script.")

    print("")
    print("Done.")


def main() -> int:
    _print_header()

    steps = [
        ("save_final_dataset.py", []),
        ("stress_test_analysis.py", []),
        ("final_analysis.py", []),
    ]

    for script, args in steps:
        ok, code = _run_step(script, args)
        if not ok:
            return code

    _print_post_run_map(os.path.abspath(os.getcwd()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
