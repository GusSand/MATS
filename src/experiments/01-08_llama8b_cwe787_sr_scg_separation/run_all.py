#!/usr/bin/env python3
"""
Run All Experiments: CWE-787 SR vs SCG Separation

Orchestrates the full experiment pipeline:
1. Collect activations with SR/SCG labels
2. Train separate probes and compute similarity
3. Run differential steering test
4. Run jailbreak test
5. Evaluate Latent Security Guard
6. Generate synthesis

Usage:
    python run_all.py              # Full pipeline
    python run_all.py --skip-collection  # Skip data collection (use existing)
    python run_all.py --only probes      # Run only specific step
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
    parser = argparse.ArgumentParser(description="Run CWE-787 SR/SCG separation experiments")
    parser.add_argument("--skip-collection", action="store_true",
                        help="Skip data collection step")
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Samples per prompt for collection (default: 50)")
    parser.add_argument("--only", type=str,
                        help="Run only specific step (collect, probes, steering, jailbreak, guard, synthesis)")
    args = parser.parse_args()

    start_time = datetime.now()
    print(f"\n{'#'*70}")
    print(f"# CWE-787 SR vs SCG SEPARATION EXPERIMENT")
    print(f"# Started: {start_time.isoformat()}")
    print(f"{'#'*70}")

    steps = {
        'collect': ('01_collect_activations.py', ['--n-samples', str(args.n_samples)]),
        'probes': ('02_train_probes.py', []),
        'steering': ('03_differential_steering.py', []),
        'jailbreak': ('04_jailbreak_test.py', []),
        'guard': ('05_latent_guard.py', []),
        'synthesis': ('06_synthesis.py', [])
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
        print("  1. Review results in: src/experiments/01-08_llama8b_cwe787_sr_scg_separation/results/")
        print("  2. Check synthesis figure: synthesis_*.png")
        print("  3. Update docs/research_journal.md with findings")
        print("  4. Create detailed report in docs/experiments/")
        return 0
    else:
        print("\nSome steps failed. Check output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
