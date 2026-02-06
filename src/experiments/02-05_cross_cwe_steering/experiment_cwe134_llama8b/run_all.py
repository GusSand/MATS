#!/usr/bin/env python3
"""
CWE-134 Steering Experiment: Llama-3.1-8B-Instruct

Run the full pipeline: baseline -> activations -> layer sweep -> pilot LOBO -> full LOBO

Usage:
    python run_all.py                  # Run all steps
    python run_all.py baseline         # Run step 1 only
    python run_all.py activations      # Run step 2 only
    python run_all.py layer_sweep      # Run step 3 only
    python run_all.py pilot            # Run step 4 only
    python run_all.py lobo             # Run step 5 only
"""

import sys
from pathlib import Path

# Add shared to path
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

import experiment_config as config
from run_pipeline import run_baseline, run_collect_activations, run_layer_sweep, run_lobo


def main():
    step = sys.argv[1] if len(sys.argv) > 1 else "all"

    if step in ("all", "baseline"):
        run_baseline(config)
        if step != "all":
            return

    if step in ("all", "activations"):
        run_collect_activations(config)
        if step != "all":
            return

    if step in ("all", "layer_sweep"):
        run_layer_sweep(config)
        if step != "all":
            return

    if step in ("all", "pilot"):
        result = run_lobo(config, n_folds=config.PILOT_FOLDS, is_pilot=True)
        if step != "all":
            return
        # Check gate
        if result['improvement_pp'] <= 10:
            print("Pilot gate failed. Stopping.")
            return

    if step in ("all", "lobo"):
        run_lobo(config, is_pilot=False)


if __name__ == "__main__":
    main()
