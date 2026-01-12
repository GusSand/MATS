#!/usr/bin/env python3
"""
Run All Experiments: SR vs SCG Separation

This script orchestrates the full experiment pipeline:
1. Validate prompts
2. Collect activations for SR and SCG
3. Train separate probes and compute similarity
4. Run differential steering test
5. Run jailbreak test
6. Evaluate Latent Security Guard
7. Generate synthesis

Usage:
    python run_all.py              # Run full pipeline with core pairs
    python run_all.py --all-pairs  # Run with all 14 pairs
    python run_all.py --skip-collection  # Skip data collection (use existing)
"""

import sys
import subprocess
from pathlib import Path
import argparse
from datetime import datetime


def run_script(script_name: str, args: list = None) -> bool:
    """Run a Python script and return success status."""
    script_path = Path(__file__).parent / script_name

    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    print(f"\n{'='*70}")
    print(f"Running: {script_name}")
    print(f"{'='*70}")

    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run SR/SCG separation experiments")
    parser.add_argument("--all-pairs", action="store_true",
                        help="Use all 14 pairs instead of core 5")
    parser.add_argument("--skip-collection", action="store_true",
                        help="Skip data collection step")
    parser.add_argument("--only", type=str,
                        help="Run only specific step (validate, collect, probes, steering, jailbreak, guard, synthesis)")
    args = parser.parse_args()

    start_time = datetime.now()
    print(f"\n{'#'*70}")
    print(f"# SR vs SCG SEPARATION EXPERIMENT")
    print(f"# Started: {start_time.isoformat()}")
    print(f"{'#'*70}")

    steps = {
        'validate': ('01_generate_prompts.py', []),
        'collect': ('02_collect_activations.py', ['--pairs', 'all' if args.all_pairs else 'core']),
        'probes': ('03_train_separate_probes.py', []),
        'steering': ('04_differential_steering.py', []),
        'jailbreak': ('05_jailbreak_test.py', []),
        'guard': ('06_latent_security_guard.py', []),
        'synthesis': ('07_synthesis.py', [])
    }

    if args.only:
        if args.only not in steps:
            print(f"Unknown step: {args.only}")
            print(f"Available: {list(steps.keys())}")
            return 1
        steps = {args.only: steps[args.only]}

    if args.skip_collection and 'collect' in steps:
        del steps['collect']
        print("Skipping data collection step")

    results = {}
    for step_name, (script, script_args) in steps.items():
        success = run_script(script, script_args)
        results[step_name] = success
        if not success:
            print(f"\nWARNING: {step_name} failed!")
            # Continue with remaining steps where possible

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'#'*70}")
    print(f"# EXPERIMENT COMPLETE")
    print(f"# Duration: {duration}")
    print(f"{'#'*70}")

    print("\nResults:")
    for step_name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {step_name}: {status}")

    if all(results.values()):
        print("\nAll steps completed successfully!")
        print("\nNext steps:")
        print("  1. Review results in: src/experiments/01-08_llama8b_sr_scg_separation/results/")
        print("  2. Check synthesis figure: synthesis_*.png")
        print("  3. Update research_journal.md with findings")
        return 0
    else:
        print("\nSome steps failed. Check output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
