#!/usr/bin/env python3
"""
Steering Mechanism Verification - Main Orchestrator

This experiment verifies that activation steering works through the mechanism
predicted by prior analysis (probes, logit lens, SAE features).

Research Question:
Does steering at Layer 31 shift the model's internal representations toward
the "secure" direction identified by our probes and SAE features?

Three Conditions:
A: Vulnerable prompts, alpha=0.0 (baseline)
B: Vulnerable prompts, alpha=3.5 (steered)
C: Secure prompts, alpha=0.0 (natural reference)

Core Hypothesis:
If steering works through the mechanism we identified:
- Condition B activations should be between A and C (or close to C)
- Security-promoting SAE features should increase: A < B <= C
- Security-suppressing SAE features should decrease: A > B >= C

Run with: python run_experiment.py

Expected runtime: ~2-3 hours total
- Activation collection: ~1-2 hours (150 generations x 512 tokens)
- Metric computation: ~10-20 minutes
- Statistical analysis: ~1 minute
- Visualizations: ~1 minute
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

EXPERIMENT_DIR = Path(__file__).parent


def run_script(script_name):
    """Run a Python script and check for errors."""
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*60 + "\n")

    script_path = EXPERIMENT_DIR / script_name

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(EXPERIMENT_DIR),
        capture_output=False
    )

    if result.returncode != 0:
        print(f"\n{'!'*60}")
        print(f"ERROR in {script_name}")
        print(f"Return code: {result.returncode}")
        print('!'*60)
        sys.exit(1)

    print(f"\nCompleted: {script_name}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    start_time = datetime.now()

    print("="*60)
    print("STEERING MECHANISM VERIFICATION EXPERIMENT")
    print("="*60)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Experiment directory: {EXPERIMENT_DIR}")
    print("="*60)

    print("""
Research Question:
Does steering at Layer 31 shift internal representations toward the
"secure" direction identified by our probes and SAE features?

Conditions:
  A: Vulnerable prompts, alpha=0.0 (baseline)
  B: Vulnerable prompts, alpha=3.5 (steered)
  C: Secure prompts, alpha=0.0 (natural reference)

Hypothesis:
  If steering works through the predicted mechanism:
  - B activations should be between A and C
  - Security-promoting features should increase A -> B
  - Security-suppressing features should decrease A -> B
""")

    scripts = [
        "01_collect_activations.py",
        "02_compute_metrics.py",
        "03_statistical_analysis.py",
        "04_visualizations.py",
    ]

    for script in scripts:
        run_script(script)

    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print("="*60)

    print("""
Next Steps:
1. Review figures in results/figures_*/
2. Check statistics in results/statistics_*.json
3. Update research_journal.md with findings
4. Add results to dissertation chapter

Key files to review:
- results/statistics_*.json  -> Hypothesis test results
- results/figures_*/fig1_*.png -> Probe projections by layer
- results/figures_*/fig2_*.png -> L31 comparison with significance
- results/figures_*/fig3_*.png -> PCA activation space
- results/figures_*/fig5_*.png -> Gap closure summary
""")


if __name__ == "__main__":
    main()
