#!/usr/bin/env python3
"""
Plotting Module for LOBO Steering Experiment

Generates publication-quality figures:
1. Main α-sweep curve (aggregated secure/insecure rates)
2. Per-fold curves showing generalization
3. Error bars with fold variance

Output formats: PDF (vector) + PNG (raster)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

from experiment_config import DATA_DIR, FIGURES_DIR, ALPHA_GRID, BASE_IDS

# Publication-quality settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_results(results_path: Path) -> Dict:
    """Load experiment results from JSON."""
    with open(results_path) as f:
        return json.load(f)


def extract_alpha_curves(results: Dict, scoring_mode: str = 'strict') -> Dict:
    """
    Extract secure/insecure rate curves from aggregated results.

    Args:
        results: Full results dict
        scoring_mode: 'strict' or 'expanded'

    Returns:
        Dict with alphas, secure_rates, insecure_rates
    """
    aggregated = results['aggregated']
    alphas = sorted([float(a) for a in aggregated.keys()])

    secure_rates = []
    insecure_rates = []
    refusal_rates = []

    for alpha in alphas:
        alpha_key = str(alpha)
        r = aggregated[alpha_key]
        secure_rates.append(r[f'{scoring_mode}_secure_rate'])
        insecure_rates.append(r[f'{scoring_mode}_insecure_rate'])
        refusal_rates.append(r['refusal_rate'])

    return {
        'alphas': alphas,
        'secure_rates': secure_rates,
        'insecure_rates': insecure_rates,
        'refusal_rates': refusal_rates,
    }


def extract_per_fold_curves(results: Dict, scoring_mode: str = 'strict') -> Dict:
    """
    Extract per-fold curves for variance analysis.

    Args:
        results: Full results dict
        scoring_mode: 'strict' or 'expanded'

    Returns:
        Dict with fold_id -> {alphas, secure_rates, insecure_rates}
    """
    fold_summaries = results['fold_summaries']
    fold_curves = {}

    for fold_data in fold_summaries:
        fold_id = fold_data['fold_id']
        summary = fold_data['summary']

        alphas = sorted([float(a) for a in summary.keys()])
        secure_rates = []
        insecure_rates = []

        for alpha in alphas:
            alpha_key = str(alpha)
            r = summary[alpha_key]
            secure_rates.append(r[scoring_mode]['secure_rate'])
            insecure_rates.append(r[scoring_mode]['insecure_rate'])

        fold_curves[fold_id] = {
            'alphas': alphas,
            'secure_rates': secure_rates,
            'insecure_rates': insecure_rates,
        }

    return fold_curves


def compute_fold_stats(fold_curves: Dict) -> Dict:
    """
    Compute mean and std across folds for error bars.

    Returns:
        Dict with alphas, mean_secure, std_secure, mean_insecure, std_insecure
    """
    fold_ids = list(fold_curves.keys())
    alphas = fold_curves[fold_ids[0]]['alphas']

    secure_matrix = np.array([fold_curves[f]['secure_rates'] for f in fold_ids])
    insecure_matrix = np.array([fold_curves[f]['insecure_rates'] for f in fold_ids])

    return {
        'alphas': alphas,
        'mean_secure': secure_matrix.mean(axis=0),
        'std_secure': secure_matrix.std(axis=0),
        'mean_insecure': insecure_matrix.mean(axis=0),
        'std_insecure': insecure_matrix.std(axis=0),
    }


def plot_main_alpha_sweep(
    results: Dict,
    scoring_mode: str = 'strict',
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot main α-sweep curve showing secure/insecure rates.

    Args:
        results: Full results dict
        scoring_mode: 'strict' or 'expanded'
        output_path: Optional path to save figure (without extension)

    Returns:
        matplotlib Figure
    """
    curves = extract_alpha_curves(results, scoring_mode)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        curves['alphas'], [r * 100 for r in curves['secure_rates']],
        'g-o', linewidth=2, markersize=8, label='Secure'
    )
    ax.plot(
        curves['alphas'], [r * 100 for r in curves['insecure_rates']],
        'r-s', linewidth=2, markersize=8, label='Insecure'
    )
    ax.plot(
        curves['alphas'], [r * 100 for r in curves['refusal_rates']],
        'b--^', linewidth=1.5, markersize=6, alpha=0.7, label='Refusal'
    )

    ax.set_xlabel('Steering Strength (α)')
    ax.set_ylabel('Rate (%)')
    ax.set_title(f'LOBO α-Sweep: {scoring_mode.upper()} Scoring\n(Aggregated across 7 held-out scenario families)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    ax.set_xlim(min(curves['alphas']) - 0.1, max(curves['alphas']) + 0.1)

    plt.tight_layout()

    if output_path:
        fig.savefig(f"{output_path}.pdf", format='pdf')
        fig.savefig(f"{output_path}.png", format='png')
        print(f"Saved: {output_path}.pdf, {output_path}.png")

    return fig


def plot_per_fold_curves(
    results: Dict,
    scoring_mode: str = 'strict',
    metric: str = 'secure',
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot per-fold curves with aggregated mean and error bars.

    Args:
        results: Full results dict
        scoring_mode: 'strict' or 'expanded'
        metric: 'secure' or 'insecure'
        output_path: Optional path to save figure (without extension)

    Returns:
        matplotlib Figure
    """
    fold_curves = extract_per_fold_curves(results, scoring_mode)
    fold_stats = compute_fold_stats(fold_curves)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot individual folds with lighter colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(fold_curves)))

    for i, (fold_id, curves) in enumerate(fold_curves.items()):
        rates = [r * 100 for r in curves[f'{metric}_rates']]
        # Shorten fold_id for legend
        short_id = fold_id.replace('pair_', '').replace('_', ' ')
        ax.plot(
            curves['alphas'], rates,
            'o-', color=colors[i], alpha=0.5, linewidth=1,
            markersize=4, label=short_id
        )

    # Plot mean with error bars (thicker)
    mean_rates = fold_stats[f'mean_{metric}'] * 100
    std_rates = fold_stats[f'std_{metric}'] * 100

    ax.errorbar(
        fold_stats['alphas'], mean_rates,
        yerr=std_rates,
        fmt='ko-', linewidth=3, markersize=10,
        capsize=5, capthick=2,
        label='Mean ± std', zorder=10
    )

    ax.set_xlabel('Steering Strength (α)')
    ax.set_ylabel(f'{metric.capitalize()} Rate (%)')
    ax.set_title(f'LOBO α-Sweep: Per-Fold {metric.capitalize()} Rates ({scoring_mode.upper()})\nEach line = model trained on 6 families, tested on held-out family')
    ax.legend(loc='best', ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    plt.tight_layout()

    if output_path:
        fig.savefig(f"{output_path}.pdf", format='pdf')
        fig.savefig(f"{output_path}.png", format='png')
        print(f"Saved: {output_path}.pdf, {output_path}.png")

    return fig


def plot_dual_panel(
    results: Dict,
    scoring_mode: str = 'strict',
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create dual-panel figure: left=main curve, right=per-fold with variance.

    Args:
        results: Full results dict
        scoring_mode: 'strict' or 'expanded'
        output_path: Optional path to save figure (without extension)

    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: Main aggregated curve
    curves = extract_alpha_curves(results, scoring_mode)

    ax1.plot(
        curves['alphas'], [r * 100 for r in curves['secure_rates']],
        'g-o', linewidth=2.5, markersize=10, label='Secure'
    )
    ax1.plot(
        curves['alphas'], [r * 100 for r in curves['insecure_rates']],
        'r-s', linewidth=2.5, markersize=10, label='Insecure'
    )
    ax1.plot(
        curves['alphas'], [r * 100 for r in curves['refusal_rates']],
        'b--^', linewidth=1.5, markersize=6, alpha=0.7, label='Refusal'
    )

    ax1.set_xlabel('Steering Strength (α)')
    ax1.set_ylabel('Rate (%)')
    ax1.set_title(f'(a) Aggregated LOBO Results ({scoring_mode.upper()})')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-5, 105)

    # Right panel: Per-fold secure rates with error bars
    fold_curves = extract_per_fold_curves(results, scoring_mode)
    fold_stats = compute_fold_stats(fold_curves)

    # Plot individual folds
    colors = plt.cm.tab10(np.linspace(0, 1, len(fold_curves)))
    for i, (fold_id, fc) in enumerate(fold_curves.items()):
        rates = [r * 100 for r in fc['secure_rates']]
        short_id = fold_id.replace('pair_', '').replace('_', ' ')
        ax2.plot(
            fc['alphas'], rates,
            'o-', color=colors[i], alpha=0.4, linewidth=1,
            markersize=3, label=short_id
        )

    # Mean with error bars
    mean_rates = fold_stats['mean_secure'] * 100
    std_rates = fold_stats['std_secure'] * 100

    ax2.errorbar(
        fold_stats['alphas'], mean_rates,
        yerr=std_rates,
        fmt='ko-', linewidth=3, markersize=8,
        capsize=4, capthick=2,
        label='Mean ± std', zorder=10
    )

    ax2.set_xlabel('Steering Strength (α)')
    ax2.set_ylabel('Secure Rate (%)')
    ax2.set_title('(b) Per-Fold Generalization')
    ax2.legend(loc='best', ncol=2, fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-5, 105)

    plt.tight_layout()

    if output_path:
        fig.savefig(f"{output_path}.pdf", format='pdf')
        fig.savefig(f"{output_path}.png", format='png')
        print(f"Saved: {output_path}.pdf, {output_path}.png")

    return fig


def generate_all_figures(results_path: Path, output_dir: Optional[Path] = None):
    """
    Generate all figures from results file.

    Args:
        results_path: Path to lobo_results JSON
        output_dir: Directory to save figures (default: FIGURES_DIR)
    """
    if output_dir is None:
        output_dir = FIGURES_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_path}")
    results = load_results(results_path)

    timestamp = results['timestamp']

    # Generate figures for both scoring modes
    for scoring_mode in ['strict', 'expanded']:
        print(f"\n--- Generating {scoring_mode.upper()} figures ---")

        # Main α-sweep
        plot_main_alpha_sweep(
            results, scoring_mode,
            output_dir / f"lobo_alpha_sweep_{scoring_mode}_{timestamp}"
        )

        # Per-fold secure rates
        plot_per_fold_curves(
            results, scoring_mode, 'secure',
            output_dir / f"lobo_per_fold_secure_{scoring_mode}_{timestamp}"
        )

        # Per-fold insecure rates
        plot_per_fold_curves(
            results, scoring_mode, 'insecure',
            output_dir / f"lobo_per_fold_insecure_{scoring_mode}_{timestamp}"
        )

        # Dual panel
        plot_dual_panel(
            results, scoring_mode,
            output_dir / f"lobo_dual_panel_{scoring_mode}_{timestamp}"
        )

    print(f"\nAll figures saved to: {output_dir}")


def main():
    """Generate figures from most recent results file."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate LOBO experiment figures")
    parser.add_argument(
        "--results", type=str, default=None,
        help="Path to results JSON (default: most recent in data/)"
    )
    args = parser.parse_args()

    if args.results:
        results_path = Path(args.results)
    else:
        # Find most recent results file
        result_files = list(DATA_DIR.glob("lobo_results_*.json"))
        if not result_files:
            print("No results files found. Run experiment first.")
            return
        results_path = sorted(result_files)[-1]

    generate_all_figures(results_path)


if __name__ == "__main__":
    main()
