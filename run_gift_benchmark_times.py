#!/usr/bin/env python3
"""
Run DREAMPlace on a given benchmark multiple times and report runtime per run.

Usage (example):
    python run_gift_benchmark_times.py \
        --runs 10 \
        --benchmark test/ispd2005/adaptec1.json
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_once(cmd):
    """Run command once and return elapsed time in seconds."""
    start = time.perf_counter()
    completed = subprocess.run(cmd, check=False)
    elapsed = time.perf_counter() - start
    return completed.returncode, elapsed


def main():
    parser = argparse.ArgumentParser(description="Batch-run DREAMPlace placement.")
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of times to run placement (default: 10)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="test/ispd2005/adaptec1.json",
        help="Path to the benchmark json passed to dreamplace/Placer.py",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help="Optional python executable to invoke (default: current interpreter)",
    )

    args = parser.parse_args()

    if args.runs <= 0:
        parser.error("--runs must be positive")

    python_exec = args.python or sys.executable
    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        parser.error(f"Benchmark file not found: {benchmark_path}")

    cmd = [python_exec, "dreamplace/Placer.py", str(benchmark_path)]

    print(f"Running command: {' '.join(cmd)}")
    print(f"Total runs: {args.runs}")

    results = []

    for i in range(1, args.runs + 1):
        print(f"\n=== Run {i} / {args.runs} ===")
        returncode, elapsed = run_once(cmd)
        success = returncode == 0
        results.append((elapsed, success, returncode))
        if not success:
            print(f"Run {i} failed with return code {returncode}")
        print(f"Elapsed time: {elapsed:.2f} seconds")

    print("\n=== Summary ===")
    for idx, (dur, success, returncode) in enumerate(results, 1):
        status = "OK" if success else f"FAIL (rc={returncode})"
        print(f"Run {idx:02d}: {dur:8.2f} s  {status}")

    successful = sum(1 for _, success, _ in results if success)
    if successful > 0:
        avg = sum(dur for dur, success, _ in results if success) / successful
        print(f"\nSuccessful runs: {successful}/{args.runs}")
        print(f"Average time over successful runs: {avg:.2f} s")
    else:
        print("\nNo successful runs.")


if __name__ == "__main__":
    main()
