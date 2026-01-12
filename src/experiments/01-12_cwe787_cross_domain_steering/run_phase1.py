#!/usr/bin/env python3
"""
Run Phase 1: L31 + Alpha Sweep

This script orchestrates the full Phase 1 experiment:
1. Collect activations from 210 prompts
2. Compute steering directions
3. Generate baseline (no steering)
4. Run alpha sweep at L31
5. Analyze results and make go/no-go decision for Phase 2
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json


def run_script(script_name: str, args: list = None) -> tuple:
    """Run a Python script and return success status and output path."""
    script_path = Path(__file__).parent / script_name
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    print(f"\n{'='*60}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"ERROR: {script_name} failed with return code {result.returncode}")
        return False, None

    return True, None


def find_latest_file(data_dir: Path, pattern: str) -> Path:
    """Find the most recent file matching pattern."""
    files = sorted(data_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def main():
    print("="*60)
    print("PHASE 1: Cross-Domain Steering Experiment")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    results_dir = script_dir / "results"
    results_dir.mkdir(exist_ok=True)

    dataset_path = "../01-12_cwe787_dataset_expansion/data/cwe787_expanded_20260112_143316.jsonl"

    # Step 1: Collect activations
    print("\n" + "#"*60)
    print("# STEP 1: Collect Activations")
    print("#"*60)
    success, _ = run_script("01_collect_activations.py", [
        "--dataset", dataset_path
    ])
    if not success:
        print("FAILED at Step 1")
        return 1

    act_path = find_latest_file(data_dir, "activations_*.npz")
    print(f"Activations: {act_path}")

    # Step 2: Compute directions
    print("\n" + "#"*60)
    print("# STEP 2: Compute Directions")
    print("#"*60)
    success, _ = run_script("02_compute_directions.py", [
        "--activations", str(act_path)
    ])
    if not success:
        print("FAILED at Step 2")
        return 1

    dir_path = find_latest_file(data_dir, "directions_*.npz")
    print(f"Directions: {dir_path}")

    # Step 3: Generate baseline
    print("\n" + "#"*60)
    print("# STEP 3: Generate Baseline")
    print("#"*60)
    success, _ = run_script("03_baseline_generation.py", [
        "--dataset", dataset_path,
        "--temperature", "0.6",
        "--max-tokens", "300"
    ])
    if not success:
        print("FAILED at Step 3")
        return 1

    baseline_path = find_latest_file(data_dir, "baseline_*.json")
    print(f"Baseline: {baseline_path}")

    # Step 4: Alpha sweep at L31
    print("\n" + "#"*60)
    print("# STEP 4: Alpha Sweep at L31")
    print("#"*60)
    success, _ = run_script("04_steered_generation.py", [
        "--dataset", dataset_path,
        "--directions", str(dir_path),
        "--layer", "31",
        "--alphas", "0.5,1.0,1.5,2.0,3.0",
        "--temperature", "0.6",
        "--max-tokens", "300",
        "--mode", "alpha_sweep"
    ])
    if not success:
        print("FAILED at Step 4")
        return 1

    steered_path = find_latest_file(data_dir, "steered_L31_*.json")
    print(f"Steered: {steered_path}")

    # Step 5: Analysis
    print("\n" + "#"*60)
    print("# STEP 5: Analysis")
    print("#"*60)
    success, _ = run_script("05_analysis.py", [
        "--baseline", str(baseline_path),
        "--steered", str(steered_path)
    ])
    if not success:
        print("FAILED at Step 5")
        return 1

    # Final summary
    print("\n" + "="*60)
    print("PHASE 1 COMPLETE")
    print("="*60)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutputs:")
    print(f"  Activations: {act_path}")
    print(f"  Directions: {dir_path}")
    print(f"  Baseline: {baseline_path}")
    print(f"  Steered: {steered_path}")
    print(f"  Results: {results_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
