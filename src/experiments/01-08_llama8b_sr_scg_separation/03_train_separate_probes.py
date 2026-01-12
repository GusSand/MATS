#!/usr/bin/env python3
"""
Train separate SR and SCG probes and compute direction similarity.

Key question: Are SR (Security Recognition) and SCG (Secure Code Generation)
encoded as separate directions (like harmfulness vs refusal)?

Metrics:
- Probe accuracy at each layer for both SR and SCG
- Cosine similarity between SR and SCG directions at each layer
- If cosine similarity < 0.5: Evidence for separate encoding
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import argparse

from utils.probe_trainer import ProbeTrainer, extract_directions_from_data, compute_cosine_similarity


def load_merged_data(data_dir: Path) -> tuple:
    """Load the most recent merged SR and SCG data."""
    sr_files = sorted(data_dir.glob("sr_merged_*.npz"))
    scg_files = sorted(data_dir.glob("scg_merged_*.npz"))

    if not sr_files or not scg_files:
        raise FileNotFoundError("Merged data not found. Run 02_collect_activations.py first.")

    sr_file = sr_files[-1]
    scg_file = scg_files[-1]

    print(f"Loading SR data: {sr_file.name}")
    print(f"Loading SCG data: {scg_file.name}")

    sr_data = np.load(sr_file)
    scg_data = np.load(scg_file)

    return sr_data, scg_data


def npz_to_dict(npz_data, n_layers: int) -> dict:
    """Convert NPZ file to layer dict format."""
    data = {}
    for layer in range(n_layers):
        X_key = f"X_layer_{layer}"
        y_key = f"y_layer_{layer}"
        if X_key in npz_data and y_key in npz_data:
            data[layer] = {
                'X': npz_data[X_key],
                'y': npz_data[y_key]
            }
    return data


def train_and_compare(sr_data: dict, scg_data: dict, n_layers: int) -> dict:
    """Train SR and SCG probes and compare directions."""
    trainer = ProbeTrainer()

    print("\n" + "=" * 60)
    print("TRAINING SR PROBES (Security Recognition)")
    print("=" * 60)
    sr_results = trainer.train_all_layers(sr_data, "sr")

    print("\n| Layer | Accuracy | AUC | CV Mean |")
    print("|-------|----------|-----|---------|")
    for r in sr_results:
        if r['accuracy'] is not None:
            print(f"| {r['layer']:5d} | {r['accuracy']*100:6.1f}% | {r['auc']:.3f} | {r['cv_accuracy_mean']*100:.1f}% |")

    print("\n" + "=" * 60)
    print("TRAINING SCG PROBES (Secure Code Generation)")
    print("=" * 60)
    scg_results = trainer.train_all_layers(scg_data, "scg")

    print("\n| Layer | Accuracy | AUC | CV Mean |")
    print("|-------|----------|-----|---------|")
    for r in scg_results:
        if r['accuracy'] is not None:
            print(f"| {r['layer']:5d} | {r['accuracy']*100:6.1f}% | {r['auc']:.3f} | {r['cv_accuracy_mean']*100:.1f}% |")

    # Compute direction similarities
    print("\n" + "=" * 60)
    print("DIRECTION SIMILARITY (SR vs SCG)")
    print("=" * 60)

    similarities = trainer.compute_all_similarities("sr", "scg", n_layers)

    print("\n| Layer | Cosine Sim | Interpretation |")
    print("|-------|------------|----------------|")
    for s in similarities:
        if s['cosine_similarity'] is not None:
            sim = s['cosine_similarity']
            if sim < 0.3:
                interp = "SEPARATE"
            elif sim < 0.5:
                interp = "somewhat separate"
            elif sim < 0.7:
                interp = "somewhat aligned"
            else:
                interp = "ALIGNED"
            print(f"| {s['layer']:5d} | {sim:10.3f} | {interp} |")

    # Also compute mean-difference directions for comparison
    print("\n" + "=" * 60)
    print("MEAN-DIFFERENCE DIRECTIONS (Alternative method)")
    print("=" * 60)

    sr_directions = extract_directions_from_data(sr_data, n_layers)
    scg_directions = extract_directions_from_data(scg_data, n_layers)

    mean_similarities = []
    for layer in range(n_layers):
        if sr_directions[layer] is not None and scg_directions[layer] is not None:
            sim = compute_cosine_similarity(
                sr_directions[layer]['normalized'],
                scg_directions[layer]['normalized']
            )
            mean_similarities.append({'layer': layer, 'cosine_similarity': sim})
        else:
            mean_similarities.append({'layer': layer, 'cosine_similarity': None})

    return {
        'sr_results': sr_results,
        'scg_results': scg_results,
        'probe_similarities': similarities,
        'mean_similarities': mean_similarities,
        'trainer': trainer,
        'sr_directions': sr_directions,
        'scg_directions': scg_directions
    }


def plot_results(results: dict, output_path: Path):
    """Create visualization of probe results and similarities."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    n_layers = len(results['sr_results'])
    layers = list(range(n_layers))

    # Plot 1: SR Probe Accuracy
    ax1 = axes[0, 0]
    sr_acc = [r['accuracy'] if r['accuracy'] else 0.5 for r in results['sr_results']]
    ax1.plot(layers, sr_acc, 'b-o', markersize=4, label='SR Accuracy')
    ax1.axhline(y=0.5, color='gray', linestyle=':', label='Chance')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('SR Probe: Security Recognition\n(Does context have security warning?)')
    ax1.legend()
    ax1.set_ylim(0.4, 1.05)
    ax1.grid(True, alpha=0.3)

    # Plot 2: SCG Probe Accuracy
    ax2 = axes[0, 1]
    scg_acc = [r['accuracy'] if r['accuracy'] else 0.5 for r in results['scg_results']]
    ax2.plot(layers, scg_acc, 'r-o', markersize=4, label='SCG Accuracy')
    ax2.axhline(y=0.5, color='gray', linestyle=':', label='Chance')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('SCG Probe: Secure Code Generation\n(Will model output secure function?)')
    ax2.legend()
    ax2.set_ylim(0.4, 1.05)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Both accuracies overlaid
    ax3 = axes[1, 0]
    ax3.plot(layers, sr_acc, 'b-o', markersize=4, label='SR (Recognition)')
    ax3.plot(layers, scg_acc, 'r-s', markersize=4, label='SCG (Generation)')
    ax3.axhline(y=0.5, color='gray', linestyle=':', label='Chance')
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('SR vs SCG Probe Accuracy')
    ax3.legend()
    ax3.set_ylim(0.4, 1.05)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Direction Similarity
    ax4 = axes[1, 1]
    probe_sim = [s['cosine_similarity'] if s['cosine_similarity'] else 0 for s in results['probe_similarities']]
    mean_sim = [s['cosine_similarity'] if s['cosine_similarity'] else 0 for s in results['mean_similarities']]

    ax4.plot(layers, probe_sim, 'g-o', markersize=4, label='Probe directions')
    ax4.plot(layers, mean_sim, 'm-s', markersize=4, label='Mean-diff directions')
    ax4.axhline(y=0.5, color='orange', linestyle='--', label='Threshold (0.5)')
    ax4.axhline(y=0.0, color='gray', linestyle=':')
    ax4.fill_between(layers, 0, 0.5, alpha=0.1, color='green', label='Separate encoding')
    ax4.fill_between(layers, 0.5, 1.0, alpha=0.1, color='red', label='Same encoding')
    ax4.set_xlabel('Layer')
    ax4.set_ylabel('Cosine Similarity')
    ax4.set_title('SR vs SCG Direction Similarity\n(Low = Separate encoding like harmfulness/refusal)')
    ax4.legend(loc='upper left', fontsize=8)
    ax4.set_ylim(-0.2, 1.0)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {output_path}")


def analyze_results(results: dict) -> dict:
    """Analyze results and draw conclusions."""
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    # Find layers where both probes work well
    good_sr_layers = [r['layer'] for r in results['sr_results']
                      if r['accuracy'] and r['accuracy'] > 0.7]
    good_scg_layers = [r['layer'] for r in results['scg_results']
                       if r['accuracy'] and r['accuracy'] > 0.7]

    print(f"\nLayers with >70% SR accuracy: {good_sr_layers}")
    print(f"Layers with >70% SCG accuracy: {good_scg_layers}")

    # Analyze similarity at good layers
    low_sim_layers = []
    high_sim_layers = []

    for s in results['probe_similarities']:
        if s['cosine_similarity'] is not None:
            if s['cosine_similarity'] < 0.5:
                low_sim_layers.append((s['layer'], s['cosine_similarity']))
            else:
                high_sim_layers.append((s['layer'], s['cosine_similarity']))

    print(f"\nLayers with LOW similarity (<0.5): {len(low_sim_layers)}")
    if low_sim_layers:
        for layer, sim in sorted(low_sim_layers, key=lambda x: x[1])[:5]:
            print(f"  L{layer}: {sim:.3f}")

    print(f"\nLayers with HIGH similarity (>=0.5): {len(high_sim_layers)}")
    if high_sim_layers:
        for layer, sim in sorted(high_sim_layers, key=lambda x: -x[1])[:5]:
            print(f"  L{layer}: {sim:.3f}")

    # Compute average similarity
    valid_sims = [s['cosine_similarity'] for s in results['probe_similarities']
                  if s['cosine_similarity'] is not None]
    avg_sim = np.mean(valid_sims) if valid_sims else 0

    print(f"\nAverage similarity across layers: {avg_sim:.3f}")

    # Conclusion
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    if avg_sim < 0.3:
        conclusion = "STRONG EVIDENCE for separate encoding (like harmfulness vs refusal)"
    elif avg_sim < 0.5:
        conclusion = "MODERATE EVIDENCE for separate encoding"
    elif avg_sim < 0.7:
        conclusion = "WEAK EVIDENCE - directions are somewhat aligned"
    else:
        conclusion = "NO EVIDENCE for separate encoding - SR and SCG use same direction"

    print(f"\n{conclusion}")
    print(f"Average cosine similarity: {avg_sim:.3f}")

    return {
        'good_sr_layers': good_sr_layers,
        'good_scg_layers': good_scg_layers,
        'low_sim_layers': low_sim_layers,
        'high_sim_layers': high_sim_layers,
        'avg_similarity': avg_sim,
        'conclusion': conclusion
    }


def main():
    parser = argparse.ArgumentParser(description="Train SR and SCG probes")
    parser.add_argument("--n-layers", type=int, default=32,
                        help="Number of layers (default: 32 for LLaMA-8B)")
    args = parser.parse_args()

    data_dir = Path(__file__).parent / "data"
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Load data
    sr_npz, scg_npz = load_merged_data(data_dir)

    # Convert to dict format
    sr_data = npz_to_dict(sr_npz, args.n_layers)
    scg_data = npz_to_dict(scg_npz, args.n_layers)

    print(f"\nLoaded data:")
    print(f"  SR samples: {len(sr_data[0]['X'])}")
    print(f"  SCG samples: {len(scg_data[0]['X'])}")

    # Train probes and compare
    results = train_and_compare(sr_data, scg_data, args.n_layers)

    # Analyze
    analysis = analyze_results(results)

    # Plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = results_dir / f"sr_scg_comparison_{timestamp}.png"
    plot_results(results, plot_path)

    # Save results
    save_results = {
        'timestamp': timestamp,
        'sr_probe_results': results['sr_results'],
        'scg_probe_results': results['scg_results'],
        'probe_similarities': results['probe_similarities'],
        'mean_similarities': results['mean_similarities'],
        'analysis': analysis
    }

    with open(results_dir / f"sr_scg_probes_{timestamp}.json", 'w') as f:
        json.dump(save_results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_dir}")

    return results, analysis


if __name__ == "__main__":
    results, analysis = main()
