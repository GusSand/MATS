#!/usr/bin/env python3
"""
Experiment 4 Phase 4: Cross-Model Analysis

Compare steering results across Llama-8B, Mistral-7B, and Llama-70B.
Produces:
  - Table 1: Model comparison
  - Figure 1: Probe accuracy vs relative layer depth
  - Figure 2: STRICT secure % vs alpha
  - Figure 3: Per-fold secure rates heatmap
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Paths
EXPERIMENT_DIR = Path(__file__).parent
EXPERIMENTS_ROOT = EXPERIMENT_DIR.parent

MISTRAL_DATA = EXPERIMENT_DIR / "experiment_4a_mistral7b" / "data"
LLAMA70B_DATA = EXPERIMENT_DIR / "experiment_4b_llama70b" / "data"
LLAMA8B_LOBO = EXPERIMENTS_ROOT / "01-12_llama8b_cwe787_lobo_steering" / "data"

OUTPUT_DIR = EXPERIMENT_DIR / "cross_model_analysis"


def load_latest_json(data_dir, pattern):
    files = sorted(glob.glob(str(data_dir / pattern)))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def load_layer_sweep(data_dir):
    path = data_dir / "layer_sweep_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_llama8b_results():
    """Load Llama-8B results from existing Experiment 2."""
    results_files = sorted(glob.glob(str(LLAMA8B_LOBO / "lobo_results_*.json")))
    if not results_files:
        return None
    with open(results_files[-1]) as f:
        return json.load(f)


def make_table1(models_data):
    """Print Table 1: Model comparison."""
    print("\n" + "=" * 80)
    print("TABLE 1: Cross-Model CWE-787 Steering Comparison")
    print("=" * 80)
    print(f"{'Model':<30} {'Baseline%':<12} {'Best Steered%':<14} {'Layer':<8} "
          f"{'Alpha':<8} {'Dir Norm':<10}")
    print("-" * 80)

    for m in models_data:
        print(f"{m['name']:<30} {m['baseline']*100:>9.1f}% {m['best_steered']*100:>11.1f}% "
              f"{m['layer']:>5} {m['best_alpha']:>6.1f} {m['direction_norm']:>9.2f}")

    print("-" * 80)


def figure1_probe_accuracy(models_sweep_data, output_dir):
    """Figure 1: Probe accuracy vs relative layer depth (all models overlaid)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {'Llama-3.1-8B': '#2196F3', 'Mistral-7B': '#FF9800', 'Llama-3.1-70B': '#4CAF50'}

    for name, sweep in models_sweep_data.items():
        if sweep is None:
            continue
        results = sweep['results']
        # Sort by layer for plotting
        results_sorted = sorted(results, key=lambda r: r['layer'])
        layers = [r['layer'] for r in results_sorted]
        n_layers = max(layers) + 1
        rel_depth = [l / n_layers for l in layers]
        accs = [r['probe_accuracy'] for r in results_sorted]

        ax.plot(rel_depth, accs, 'o-', label=name, color=colors.get(name, '#666'),
                markersize=3, alpha=0.8)

    ax.set_xlabel('Relative Layer Depth', fontsize=12)
    ax.set_ylabel('5-Fold CV Probe Accuracy', fontsize=12)
    ax.set_title('Linear Probe Accuracy: Secure vs Vulnerable Activation Classification', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.45, 1.02)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "figure1_probe_accuracy_by_depth.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


def figure2_secure_vs_alpha(models_lobo_data, output_dir):
    """Figure 2: STRICT secure % vs alpha (all models at optimal layers)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'Llama-3.1-8B': '#2196F3', 'Mistral-7B': '#FF9800', 'Llama-3.1-70B': '#4CAF50'}

    for name, lobo in models_lobo_data.items():
        if lobo is None:
            continue
        agg = lobo['aggregated']
        alphas = []
        rates = []
        for alpha_key, data in sorted(agg.items(), key=lambda x: float(x[0])):
            alphas.append(float(alpha_key))
            rates.append(data['strict_secure_rate'] * 100)

        ax.plot(alphas, rates, 'o-', label=name, color=colors.get(name, '#666'),
                markersize=6, linewidth=2)

    ax.set_xlabel('Steering Strength (alpha)', fontsize=12)
    ax.set_ylabel('STRICT Secure Rate (%)', fontsize=12)
    ax.set_title('CWE-787 Secure Code Rate vs Steering Strength', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-2, 102)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "figure2_secure_vs_alpha.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


def figure3_fold_heatmap(models_lobo_data, output_dir):
    """Figure 3: Per-fold secure rates heatmap at best alpha."""
    # Collect per-fold rates at each model's best alpha
    model_names = []
    fold_names = []
    data_matrix = []

    for name, lobo in models_lobo_data.items():
        if lobo is None:
            continue

        # Find best alpha
        agg = lobo['aggregated']
        best_alpha = max(agg.items(), key=lambda x: x[1]['strict_secure_rate'])[0]

        fold_summaries = lobo.get('fold_summaries', [])
        if not fold_summaries:
            continue

        model_names.append(f"{name}\n(alpha={best_alpha})")
        fold_rates = []
        fold_ids = []

        for fs in fold_summaries:
            fold_id = fs['fold_id']
            fold_ids.append(fold_id.replace('pair_', '').replace('_', '\n'))
            summary = fs['summary']
            if best_alpha in summary:
                rate = summary[best_alpha].get('strict_secure_rate',
                       summary[best_alpha].get('strict', {}).get('secure_rate', 0))
            else:
                rate = 0
            fold_rates.append(rate * 100)

        data_matrix.append(fold_rates)
        if not fold_names:
            fold_names = fold_ids

    if not data_matrix:
        print("No fold data available for heatmap")
        return

    data_array = np.array(data_matrix)

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(data_array, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=fold_names, yticklabels=model_names,
                vmin=0, vmax=100, ax=ax, cbar_kws={'label': 'STRICT Secure %'})
    ax.set_title('Per-Fold Secure Rates at Optimal Alpha', fontsize=13)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "figure3_fold_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("EXPERIMENT 4 PHASE 4: Cross-Model Analysis")
    print(f"Timestamp: {timestamp}")
    print("=" * 60)

    # Load data
    print("\nLoading results...")

    # Llama 8B (existing)
    llama8b_lobo = load_llama8b_results()
    if llama8b_lobo:
        print(f"  Llama-8B LOBO: loaded ({len(llama8b_lobo.get('fold_summaries', []))} folds)")
    else:
        print("  Llama-8B LOBO: NOT FOUND")

    # Mistral 7B
    mistral_baseline = load_latest_json(MISTRAL_DATA, "baseline_*.json")
    mistral_lobo = load_latest_json(MISTRAL_DATA, "lobo_results_*.json")
    mistral_sweep = load_layer_sweep(MISTRAL_DATA)
    print(f"  Mistral-7B: baseline={'yes' if mistral_baseline else 'no'}, "
          f"lobo={'yes' if mistral_lobo else 'no'}, sweep={'yes' if mistral_sweep else 'no'}")

    # Llama 70B
    llama70b_baseline = load_latest_json(LLAMA70B_DATA, "baseline_*.json")
    llama70b_lobo = load_latest_json(LLAMA70B_DATA, "lobo_results_*.json")
    llama70b_sweep = load_layer_sweep(LLAMA70B_DATA)
    print(f"  Llama-70B: baseline={'yes' if llama70b_baseline else 'no'}, "
          f"lobo={'yes' if llama70b_lobo else 'no'}, sweep={'yes' if llama70b_sweep else 'no'}")

    # Build comparison table
    models_data = []

    if llama8b_lobo:
        agg = llama8b_lobo['aggregated']
        best_alpha = max(agg.items(), key=lambda x: x[1]['strict_secure_rate'])
        baseline_rate = agg.get('0.0', {}).get('strict_secure_rate', 0)
        models_data.append({
            'name': 'Llama-3.1-8B-Instruct',
            'baseline': baseline_rate,
            'best_steered': best_alpha[1]['strict_secure_rate'],
            'best_alpha': float(best_alpha[0]),
            'layer': llama8b_lobo.get('config', {}).get('layer', 31),
            'direction_norm': np.mean([fs.get('direction_norm', 0)
                                       for fs in llama8b_lobo.get('fold_summaries', [])
                                       if 'direction_norm' in fs]),
        })

    if mistral_lobo:
        agg = mistral_lobo['aggregated']
        best_alpha = max(agg.items(), key=lambda x: x[1]['strict_secure_rate'])
        baseline_rate = agg.get('0.0', {}).get('strict_secure_rate', 0)
        if mistral_baseline:
            baseline_rate = mistral_baseline['summary']['strict']['secure_rate']
        models_data.append({
            'name': 'Mistral-7B-Instruct-v0.3',
            'baseline': baseline_rate,
            'best_steered': best_alpha[1]['strict_secure_rate'],
            'best_alpha': float(best_alpha[0]),
            'layer': mistral_lobo.get('config', {}).get('layer', -1),
            'direction_norm': np.mean([fs.get('direction_norm', 0)
                                       for fs in mistral_lobo.get('fold_summaries', [])
                                       if 'direction_norm' in fs]),
        })
    elif mistral_baseline:
        models_data.append({
            'name': 'Mistral-7B-Instruct-v0.3',
            'baseline': mistral_baseline['summary']['strict']['secure_rate'],
            'best_steered': 0, 'best_alpha': 0, 'layer': -1, 'direction_norm': 0,
        })

    if llama70b_lobo:
        agg = llama70b_lobo['aggregated']
        best_alpha = max(agg.items(), key=lambda x: x[1]['strict_secure_rate'])
        baseline_rate = agg.get('0.0', {}).get('strict_secure_rate', 0)
        if llama70b_baseline:
            baseline_rate = llama70b_baseline['summary']['strict']['secure_rate']
        models_data.append({
            'name': 'Llama-3.1-70B-Instruct (8-bit)',
            'baseline': baseline_rate,
            'best_steered': best_alpha[1]['strict_secure_rate'],
            'best_alpha': float(best_alpha[0]),
            'layer': llama70b_lobo.get('config', {}).get('layer', -1),
            'direction_norm': np.mean([fs.get('direction_norm', 0)
                                       for fs in llama70b_lobo.get('fold_summaries', [])
                                       if 'direction_norm' in fs]),
        })
    elif llama70b_baseline:
        models_data.append({
            'name': 'Llama-3.1-70B-Instruct (8-bit)',
            'baseline': llama70b_baseline['summary']['strict']['secure_rate'],
            'best_steered': 0, 'best_alpha': 0, 'layer': -1, 'direction_norm': 0,
        })

    # Table 1
    if models_data:
        make_table1(models_data)

    # Figures
    models_sweep = {}
    if mistral_sweep:
        models_sweep['Mistral-7B'] = mistral_sweep
    if llama70b_sweep:
        models_sweep['Llama-3.1-70B'] = llama70b_sweep

    models_lobo = {}
    if llama8b_lobo:
        models_lobo['Llama-3.1-8B'] = llama8b_lobo
    if mistral_lobo:
        models_lobo['Mistral-7B'] = mistral_lobo
    if llama70b_lobo:
        models_lobo['Llama-3.1-70B'] = llama70b_lobo

    if models_sweep:
        figure1_probe_accuracy(models_sweep, OUTPUT_DIR)

    if models_lobo:
        figure2_secure_vs_alpha(models_lobo, OUTPUT_DIR)
        figure3_fold_heatmap(models_lobo, OUTPUT_DIR)

    # Save summary JSON
    summary = {
        'timestamp': timestamp,
        'models': models_data,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / f"cross_model_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {summary_path}")

    print("\n" + "=" * 60)
    print("Cross-model analysis complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
