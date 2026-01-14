#!/usr/bin/env python3
"""
Steering Mechanism Verification - Visualizations

Creates publication-ready figures:
1. Probe projection by layer (line plot with confidence bands)
2. Layer 31 bar comparison with significance annotations
3. Activation space trajectory (PCA)
4. Steering alignment distribution
5. Gap closure summary
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.decomposition import PCA
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_config import (
    DATA_DIR, RESULTS_DIR, LAYERS_TO_EXTRACT, STEERING_LAYER,
)


# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

# Color scheme
COLORS = {
    'A': '#e74c3c',  # Red - baseline vulnerable
    'B': '#3498db',  # Blue - steered
    'C': '#2ecc71',  # Green - natural secure
}

LABELS = {
    'A': 'Vulnerable (alpha=0)',
    'B': 'Vulnerable (alpha=3.5)',
    'C': 'Secure (alpha=0)',
}

SHORT_LABELS = {
    'A': 'Baseline',
    'B': 'Steered',
    'C': 'Natural Secure',
}


# =============================================================================
# FIGURE 1: Probe Projection by Layer
# =============================================================================

def plot_probe_projections(metrics, stats, output_dir):
    """
    Figure 1: Probe projection by layer for all three conditions.
    Shows how steering shifts internal representations toward secure direction.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    layers = sorted([int(l) for l in metrics["probe_projections"]["A"].keys()])

    for cond in ["A", "B", "C"]:
        means = []
        sems = []  # Standard error of mean
        for layer in layers:
            projs = metrics["probe_projections"][cond].get(str(layer), [])
            if projs:
                means.append(np.mean(projs))
                sems.append(np.std(projs) / np.sqrt(len(projs)))
            else:
                means.append(np.nan)
                sems.append(np.nan)

        means = np.array(means)
        sems = np.array(sems)

        ax.plot(layers, means, 'o-', color=COLORS[cond], label=LABELS[cond],
                linewidth=2, markersize=8)
        ax.fill_between(layers, means - 1.96*sems, means + 1.96*sems,
                       color=COLORS[cond], alpha=0.2)

    # Mark steering layer
    ax.axvline(x=STEERING_LAYER, color='gray', linestyle='--', alpha=0.5,
               linewidth=1, label=f'Steering layer (L{STEERING_LAYER})')

    ax.set_xlabel("Layer")
    ax.set_ylabel("Projection onto Probe Direction\n(higher = more secure)")
    ax.set_title("Steering Shifts Representations Toward Secure Direction")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers)

    output_path = output_dir / "fig1_probe_projections.pdf"
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix('.png'))
    print(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# FIGURE 2: Layer 31 Bar Comparison
# =============================================================================

def plot_layer31_comparison(metrics, stats, output_dir):
    """
    Figure 2: Bar chart comparing L31 projections with statistical annotations.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    layer_key = str(STEERING_LAYER)
    conditions = ["A", "B", "C"]

    means = []
    sems = []
    for cond in conditions:
        projs = metrics["probe_projections"][cond].get(layer_key, [])
        means.append(np.mean(projs))
        sems.append(np.std(projs) / np.sqrt(len(projs)))

    x = np.arange(len(conditions))
    bars = ax.bar(x, means, yerr=[1.96*s for s in sems], capsize=5,
                  color=[COLORS[c] for c in conditions], alpha=0.8,
                  edgecolor='black', linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[c] for c in conditions])
    ax.set_ylabel("Projection onto Probe Direction")
    ax.set_title(f"Layer {STEERING_LAYER} Representations")

    # Add significance annotations
    if layer_key in stats.get("probe_projection_comparisons", {}):
        comp_ab = stats["probe_projection_comparisons"][layer_key]["A_vs_B"]
        p_val = comp_ab["significance"]["t_test_p"]

        # Draw significance bracket A vs B
        y_max = max(means) + max(sems) * 2.5
        bracket_height = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])

        ax.plot([0, 0, 1, 1], [y_max, y_max + bracket_height, y_max + bracket_height, y_max],
                'k-', linewidth=1)

        if p_val < 0.001:
            sig_text = "***"
        elif p_val < 0.01:
            sig_text = "**"
        elif p_val < 0.05:
            sig_text = "*"
        else:
            sig_text = "n.s."

        ax.text(0.5, y_max + bracket_height * 1.5, sig_text, ha='center', fontsize=14)

        # Add effect size annotation
        d = comp_ab["effect_size"]["cohens_d"]
        ax.text(0.5, y_max + bracket_height * 3.5, f"d={d:.2f}", ha='center', fontsize=10)

    # Add gap closure annotation
    if layer_key in stats.get("gap_closure", {}):
        closure = stats["gap_closure"][layer_key]["closure_percent"]
        ax.text(0.02, 0.98, f"Gap closure: {closure:.1f}%",
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    output_path = output_dir / "fig2_layer31_comparison.pdf"
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix('.png'))
    print(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# FIGURE 3: Activation Space (PCA)
# =============================================================================

def plot_activation_space(act_data, output_dir):
    """
    Figure 3: PCA visualization of activation space at L31.
    Shows that steering moves vulnerable activations toward secure region.
    """
    # Load L31 activations
    layer_key = f"L{STEERING_LAYER}"

    acts_a = act_data.get(f"condition_A_{layer_key}")
    acts_b = act_data.get(f"condition_B_{layer_key}")
    acts_c = act_data.get(f"condition_C_{layer_key}")

    if acts_a is None or acts_b is None or acts_c is None:
        print("Warning: Missing activation data for PCA plot")
        return

    # Combine all activations
    all_acts = np.vstack([acts_a, acts_b, acts_c])
    labels = ['A'] * len(acts_a) + ['B'] * len(acts_b) + ['C'] * len(acts_c)

    # PCA
    pca = PCA(n_components=2)
    projected = pca.fit_transform(all_acts)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each condition
    for cond in ["A", "B", "C"]:
        mask = [l == cond for l in labels]
        ax.scatter(projected[mask, 0], projected[mask, 1],
                  c=COLORS[cond], label=LABELS[cond], alpha=0.6, s=50)

    # Plot centroids
    centroids = {}
    for cond in ["A", "B", "C"]:
        mask = [l == cond for l in labels]
        centroid = projected[mask].mean(axis=0)
        centroids[cond] = centroid
        ax.scatter(centroid[0], centroid[1], c=COLORS[cond],
                  marker='X', s=200, edgecolors='black', linewidth=2,
                  zorder=10)

    # Draw arrow from A centroid to B centroid (steering effect)
    ax.annotate('', xy=centroids['B'], xytext=centroids['A'],
               arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Draw dashed line from A to C (full gap)
    ax.plot([centroids['A'][0], centroids['C'][0]],
           [centroids['A'][1], centroids['C'][1]],
           'k--', alpha=0.3, linewidth=1)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title(f"Activation Space at Layer {STEERING_LAYER}\n(Arrow: steering effect)")
    ax.legend(loc='best')

    output_path = output_dir / "fig3_activation_space_pca.pdf"
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix('.png'))
    print(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# FIGURE 4: Steering Alignment Distribution
# =============================================================================

def plot_steering_alignment(metrics, output_dir):
    """
    Figure 4: Distribution of steering alignment values.
    """
    if not metrics.get("steering_alignment"):
        print("Warning: No steering alignment data for plot")
        return

    alignments = [a["alignment"] for a in metrics["steering_alignment"]]
    ratios = [a["alignment_ratio"] for a in metrics["steering_alignment"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Alignment distribution
    ax1 = axes[0]
    ax1.hist(alignments, bins=20, color=COLORS['B'], alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero alignment')
    ax1.axvline(x=np.mean(alignments), color='black', linestyle='-', linewidth=2,
               label=f'Mean: {np.mean(alignments):.3f}')
    ax1.set_xlabel("Steering Alignment")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Steering Alignment\n(positive = moves toward secure)")
    ax1.legend()

    # Ratio distribution
    ax2 = axes[1]
    ax2.hist(ratios, bins=20, color=COLORS['B'], alpha=0.7, edgecolor='black')
    ax2.axvline(x=1, color='red', linestyle='--', linewidth=2, label='Ratio = 1')
    ax2.axvline(x=np.mean(ratios), color='black', linestyle='-', linewidth=2,
               label=f'Mean: {np.mean(ratios):.2f}')
    ax2.set_xlabel("Alignment Ratio (Parallel / Orthogonal)")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of Alignment Ratio\n(>1 = mostly in steering direction)")
    ax2.legend()

    plt.tight_layout()

    output_path = output_dir / "fig4_steering_alignment.pdf"
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix('.png'))
    print(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# FIGURE 5: Gap Closure Summary
# =============================================================================

def plot_gap_closure(stats, output_dir):
    """
    Figure 5: Gap closure across layers.
    """
    if not stats.get("gap_closure"):
        print("Warning: No gap closure data for plot")
        return

    layers = sorted([int(k) for k in stats["gap_closure"].keys()])
    closures = [stats["gap_closure"][str(l)]["closure_percent"] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(range(len(layers)), closures, color=COLORS['B'], alpha=0.8,
                  edgecolor='black', linewidth=1)

    # Color the steering layer bar differently
    for i, layer in enumerate(layers):
        if layer == STEERING_LAYER:
            bars[i].set_color('#8e44ad')  # Purple for steering layer

    ax.axhline(y=30, color='red', linestyle='--', linewidth=2,
              label='30% threshold')
    ax.axhline(y=100, color='green', linestyle=':', linewidth=1,
              label='100% (full gap)')

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Gap Closure (%)")
    ax.set_title("How Much of the Baseline-to-Secure Gap Does Steering Close?")
    ax.legend()

    # Annotate steering layer
    steer_idx = layers.index(STEERING_LAYER) if STEERING_LAYER in layers else -1
    if steer_idx >= 0:
        ax.annotate(f'Steering\nLayer',
                   xy=(steer_idx, closures[steer_idx]),
                   xytext=(steer_idx, closures[steer_idx] + 15),
                   ha='center', fontsize=10,
                   arrowprops=dict(arrowstyle='->', color='black'))

    output_path = output_dir / "fig5_gap_closure.pdf"
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix('.png'))
    print(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# FIGURE 6: Classification Results
# =============================================================================

def plot_classification_results(metrics, output_dir):
    """
    Figure 6: Classification rates across conditions.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    conditions = ["A", "B", "C"]
    categories = ["secure", "insecure", "other"]
    category_colors = {'secure': '#2ecc71', 'insecure': '#e74c3c', 'other': '#95a5a6'}

    x = np.arange(len(conditions))
    width = 0.25

    for i, cat in enumerate(categories):
        rates = []
        for cond in conditions:
            classes = metrics["classifications"][cond]
            rate = sum(1 for c in classes if c == cat) / len(classes) * 100
            rates.append(rate)
        ax.bar(x + i*width, rates, width, label=cat.capitalize(),
              color=category_colors[cat], edgecolor='black', linewidth=1)

    ax.set_xticks(x + width)
    ax.set_xticklabels([SHORT_LABELS[c] for c in conditions])
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Classification Rates by Condition")
    ax.legend()
    ax.set_ylim(0, 100)

    output_path = output_dir / "fig6_classification_rates.pdf"
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix('.png'))
    print(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def find_latest_file(directory, pattern):
    """Find the most recent file matching pattern."""
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")
    return files[-1]


def main():
    print("=" * 60)
    print("Steering Mechanism Verification - Visualizations")
    print("=" * 60)

    # Find most recent data files
    try:
        metrics_file = find_latest_file(RESULTS_DIR, "metrics_*.json")
        stats_file = find_latest_file(RESULTS_DIR, "statistics_*.json")
        act_file = find_latest_file(DATA_DIR, "activations_*.npz")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run previous steps first.")
        return

    # Load data
    print(f"\nLoading metrics from: {metrics_file}")
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    print(f"Loading statistics from: {stats_file}")
    with open(stats_file, 'r') as f:
        stats = json.load(f)

    print(f"Loading activations from: {act_file}")
    act_data = dict(np.load(act_file))

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_dir = RESULTS_DIR / f"figures_{timestamp}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving figures to: {fig_dir}")

    # Generate figures
    print("\n" + "-" * 40)
    print("Generating figures...")
    print("-" * 40)

    plot_probe_projections(metrics, stats, fig_dir)
    plot_layer31_comparison(metrics, stats, fig_dir)
    plot_activation_space(act_data, fig_dir)
    plot_steering_alignment(metrics, fig_dir)
    plot_gap_closure(stats, fig_dir)
    plot_classification_results(metrics, fig_dir)

    print(f"\nAll figures saved to: {fig_dir}")

    return fig_dir


if __name__ == "__main__":
    main()
