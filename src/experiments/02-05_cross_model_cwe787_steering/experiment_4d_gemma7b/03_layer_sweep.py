#!/usr/bin/env python3
"""
Experiment 4D Step 3: Gemma-7B Layer Sweep

Linear probes at all 48 layers.
Select top-5 layers by CV accuracy.
"""

import sys
import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from experiment_config import DATA_DIR
from layer_sweep import run_layer_sweep, select_top_layers, save_sweep_results


def find_latest_npz():
    """Find most recent activations file."""
    npz_files = sorted(glob.glob(str(DATA_DIR / "activations_*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No activation files found in {DATA_DIR}")
    return Path(npz_files[-1])


def main():
    print("=" * 60)
    print("EXPERIMENT 4C STEP 3: Gemma-7B Layer Sweep")
    print("=" * 60)

    npz_path = find_latest_npz()
    print(f"Using activations: {npz_path}")

    print("\nRunning layer sweep...")
    results = run_layer_sweep(npz_path)

    output_path = save_sweep_results(results, DATA_DIR)

    # Print top-5
    print("\nTop-5 layers by probe accuracy:")
    for r in results[:5]:
        print(f"  Layer {r['layer']:3d}: acc={r['probe_accuracy']:.4f}, "
              f"norm={r['direction_norm']:.2f}, sep={r['cluster_separation']:.4f}")

    best_layer = results[0]['layer']
    print(f"\nBest layer: {best_layer} (accuracy: {results[0]['probe_accuracy']:.4f})")

    return best_layer


if __name__ == "__main__":
    main()
