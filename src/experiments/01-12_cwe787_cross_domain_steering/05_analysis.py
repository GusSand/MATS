#!/usr/bin/env python3
"""
Step 5: Analyze steering experiment results.

Compares baseline vs steered outputs and computes conversion metrics.
"""

import json
from pathlib import Path
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def analyze_alpha_sweep(baseline: dict, steered: dict) -> dict:
    """Analyze alpha sweep results against baseline."""
    baseline_summary = baseline['summary']
    steered_summaries = steered['summaries']

    analysis = {
        'baseline': {
            'secure_rate': baseline_summary['secure_rate'],
            'insecure_rate': baseline_summary['insecure_rate'],
            'incomplete_rate': baseline_summary['incomplete_rate']
        },
        'alpha_results': {},
        'best_alpha': None,
        'best_conversion': 0
    }

    for alpha, summary in steered_summaries.items():
        conversion = summary['secure_rate'] - baseline_summary['secure_rate']
        degradation = summary['incomplete_rate'] - baseline_summary['incomplete_rate']

        analysis['alpha_results'][alpha] = {
            'secure_rate': summary['secure_rate'],
            'insecure_rate': summary['insecure_rate'],
            'incomplete_rate': summary['incomplete_rate'],
            'conversion_rate': conversion,
            'degradation': degradation
        }

        if conversion > analysis['best_conversion']:
            analysis['best_conversion'] = conversion
            analysis['best_alpha'] = alpha

    return analysis


def analyze_layer_sweep(baseline: dict, steered: dict) -> dict:
    """Analyze layer sweep results against baseline."""
    baseline_summary = baseline['summary']
    steered_summaries = steered['summaries']

    analysis = {
        'baseline': {
            'secure_rate': baseline_summary['secure_rate'],
            'insecure_rate': baseline_summary['insecure_rate'],
            'incomplete_rate': baseline_summary['incomplete_rate']
        },
        'layer_results': {},
        'best_layer': None,
        'best_conversion': 0
    }

    for layer, summary in steered_summaries.items():
        conversion = summary['secure_rate'] - baseline_summary['secure_rate']
        degradation = summary['incomplete_rate'] - baseline_summary['incomplete_rate']

        analysis['layer_results'][layer] = {
            'secure_rate': summary['secure_rate'],
            'insecure_rate': summary['insecure_rate'],
            'incomplete_rate': summary['incomplete_rate'],
            'conversion_rate': conversion,
            'degradation': degradation
        }

        if conversion > analysis['best_conversion']:
            analysis['best_conversion'] = conversion
            analysis['best_layer'] = layer

    return analysis


def plot_alpha_sweep(analysis: dict, output_path: Path, layer: int):
    """Create visualization for alpha sweep."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    alphas = sorted([float(a) for a in analysis['alpha_results'].keys()])
    secure_rates = [analysis['alpha_results'][str(a)]['secure_rate'] * 100 for a in alphas]
    insecure_rates = [analysis['alpha_results'][str(a)]['insecure_rate'] * 100 for a in alphas]
    incomplete_rates = [analysis['alpha_results'][str(a)]['incomplete_rate'] * 100 for a in alphas]
    conversion_rates = [analysis['alpha_results'][str(a)]['conversion_rate'] * 100 for a in alphas]

    baseline_secure = analysis['baseline']['secure_rate'] * 100

    # Plot 1: Classification rates
    ax1 = axes[0]
    ax1.plot(alphas, secure_rates, 'g-o', label='Secure', linewidth=2, markersize=8)
    ax1.plot(alphas, insecure_rates, 'r-s', label='Insecure', linewidth=2, markersize=8)
    ax1.plot(alphas, incomplete_rates, 'gray', linestyle='--', marker='^', label='Incomplete')
    ax1.axhline(y=baseline_secure, color='g', linestyle=':', alpha=0.7, label=f'Baseline secure ({baseline_secure:.1f}%)')
    ax1.set_xlabel('Alpha (Steering Strength)', fontsize=12)
    ax1.set_ylabel('Rate (%)', fontsize=12)
    ax1.set_title(f'Classification Rates vs Alpha (L{layer})', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    # Plot 2: Conversion rate
    ax2 = axes[1]
    colors = ['green' if c > 0 else 'red' for c in conversion_rates]
    bars = ax2.bar(alphas, conversion_rates, color=colors, alpha=0.7, width=0.3)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xlabel('Alpha (Steering Strength)', fontsize=12)
    ax2.set_ylabel('Conversion Rate (pp)', fontsize=12)
    ax2.set_title(f'Conversion Rate: Steered - Baseline (L{layer})', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, conversion_rates):
        height = bar.get_height()
        ax2.annotate(f'{val:+.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")


def plot_layer_sweep(analysis: dict, output_path: Path, alpha: float):
    """Create visualization for layer sweep."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    layers = sorted([int(l) for l in analysis['layer_results'].keys()])
    secure_rates = [analysis['layer_results'][str(l)]['secure_rate'] * 100 for l in layers]
    conversion_rates = [analysis['layer_results'][str(l)]['conversion_rate'] * 100 for l in layers]

    baseline_secure = analysis['baseline']['secure_rate'] * 100

    # Plot 1: Secure rate by layer
    ax1 = axes[0]
    ax1.plot(layers, secure_rates, 'g-o', linewidth=2, markersize=4)
    ax1.axhline(y=baseline_secure, color='r', linestyle='--', label=f'Baseline ({baseline_secure:.1f}%)')
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Secure Rate (%)', fontsize=12)
    ax1.set_title(f'Secure Rate by Layer (α={alpha})', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Conversion rate by layer
    ax2 = axes[1]
    colors = ['green' if c > 0 else 'red' for c in conversion_rates]
    ax2.bar(layers, conversion_rates, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Conversion Rate (pp)', fontsize=12)
    ax2.set_title(f'Conversion Rate by Layer (α={alpha})', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")


def print_phase1_summary(analysis: dict, layer: int):
    """Print Phase 1 summary and decision gate."""
    print("\n" + "="*60)
    print("PHASE 1 ANALYSIS: L31 Alpha Sweep")
    print("="*60)

    print(f"\nBaseline (no steering):")
    print(f"  Secure: {analysis['baseline']['secure_rate']*100:.1f}%")
    print(f"  Insecure: {analysis['baseline']['insecure_rate']*100:.1f}%")
    print(f"  Incomplete: {analysis['baseline']['incomplete_rate']*100:.1f}%")

    print(f"\nAlpha Sweep Results (L{layer}):")
    print(f"{'Alpha':<8} {'Secure%':<10} {'Δ Secure':<12} {'Incomplete%':<12}")
    print("-" * 42)

    for alpha in sorted(analysis['alpha_results'].keys(), key=float):
        r = analysis['alpha_results'][alpha]
        print(f"{alpha:<8} {r['secure_rate']*100:<10.1f} {r['conversion_rate']*100:+.1f} pp      {r['incomplete_rate']*100:<12.1f}")

    print(f"\nBest Alpha: {analysis['best_alpha']}")
    print(f"Best Conversion: {analysis['best_conversion']*100:+.1f} percentage points")

    # Decision gate
    print("\n" + "="*60)
    print("DECISION GATE")
    print("="*60)

    if analysis['best_conversion'] > 0.10:
        print(f"✅ PASS: Conversion rate {analysis['best_conversion']*100:.1f}% > 10% threshold")
        print("   → PROCEED TO PHASE 2 (Layer Sweep)")
    elif analysis['best_conversion'] > 0:
        print(f"⚠️  MARGINAL: Conversion rate {analysis['best_conversion']*100:.1f}% is positive but < 10%")
        print("   → Consider Phase 2 with caution")
    else:
        print(f"❌ FAIL: No positive conversion rate achieved")
        print("   → Steering at L{layer} does not improve security")


def main():
    parser = argparse.ArgumentParser(description="Analyze steering results")
    parser.add_argument("--baseline", type=str, required=True,
                        help="Path to baseline results JSON")
    parser.add_argument("--steered", type=str, required=True,
                        help="Path to steered results JSON")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    results_dir = script_dir / "results"
    results_dir.mkdir(exist_ok=True)

    baseline_path = Path(args.baseline)
    if not baseline_path.is_absolute():
        baseline_path = data_dir / args.baseline

    steered_path = Path(args.steered)
    if not steered_path.is_absolute():
        steered_path = data_dir / args.steered

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load data
    print(f"Loading baseline: {baseline_path}")
    baseline = load_json(baseline_path)

    print(f"Loading steered: {steered_path}")
    steered = load_json(steered_path)

    # Analyze based on mode
    if steered['mode'] == 'alpha_sweep':
        layer = steered['config']['layer']
        analysis = analyze_alpha_sweep(baseline, steered)

        # Plot
        plot_path = results_dir / f"phase1_L{layer}_alpha_sweep_{timestamp}.png"
        plot_alpha_sweep(analysis, plot_path, layer)

        # Print summary
        print_phase1_summary(analysis, layer)

        # Save analysis
        output = {
            'timestamp': timestamp,
            'mode': 'alpha_sweep',
            'layer': layer,
            'analysis': analysis
        }

    else:  # layer_sweep
        alpha = steered['config']['alpha']
        analysis = analyze_layer_sweep(baseline, steered)

        # Plot
        plot_path = results_dir / f"phase2_layer_sweep_a{alpha}_{timestamp}.png"
        plot_layer_sweep(analysis, plot_path, alpha)

        # Print summary
        print("\n" + "="*60)
        print(f"PHASE 2 ANALYSIS: Layer Sweep (α={alpha})")
        print("="*60)
        print(f"\nBest Layer: L{analysis['best_layer']}")
        print(f"Best Conversion: {analysis['best_conversion']*100:+.1f} pp")

        output = {
            'timestamp': timestamp,
            'mode': 'layer_sweep',
            'alpha': alpha,
            'analysis': analysis
        }

    # Save
    analysis_path = results_dir / f"analysis_{timestamp}.json"
    with open(analysis_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nAnalysis saved to: {analysis_path}")

    return analysis


if __name__ == "__main__":
    main()
