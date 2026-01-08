#!/usr/bin/env python3
"""
Differential Steering Test for CWE-787 Prompt Pairs

Key question: Can we steer SR (Security Recognition) without affecting SCG
(Secure Code Generation), and vice versa?

If SR and SCG are separate directions:
- Steering along SR should change "does model recognize security context"
  but NOT "will model output secure code"
- Steering along SCG should change "will model output secure code"
  but NOT "does model recognize security context"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt

from validated_pairs import get_pair
from utils.steering import ActivationSteering


def load_directions(data_dir: Path, n_layers: int = 32) -> tuple:
    """Load data and compute mean-difference directions."""
    sr_files = sorted(data_dir.glob("sr_data_*.npz"))
    scg_files = sorted(data_dir.glob("scg_data_*.npz"))

    sr_npz = np.load(sr_files[-1])
    scg_npz = np.load(scg_files[-1])

    sr_directions = {}
    scg_directions = {}

    for layer in range(n_layers):
        # SR direction: mean(secure prompts) - mean(vulnerable prompts)
        X_sr = sr_npz[f'X_layer_{layer}']
        y_sr = sr_npz[f'y_layer_{layer}']

        if len(X_sr) > 0:
            secure_mean = X_sr[y_sr == 1].mean(axis=0)
            vuln_mean = X_sr[y_sr == 0].mean(axis=0)
            sr_dir = secure_mean - vuln_mean
            sr_directions[layer] = {
                'raw': sr_dir,
                'normalized': sr_dir / (np.linalg.norm(sr_dir) + 1e-10)
            }
        else:
            sr_directions[layer] = None

        # SCG direction: mean(secure outputs) - mean(insecure outputs)
        X_scg = scg_npz[f'X_layer_{layer}']
        y_scg = scg_npz[f'y_layer_{layer}']

        if len(X_scg) > 0:
            secure_mean = X_scg[y_scg == 1].mean(axis=0)
            insecure_mean = X_scg[y_scg == 0].mean(axis=0)
            scg_dir = secure_mean - insecure_mean
            scg_directions[layer] = {
                'raw': scg_dir,
                'normalized': scg_dir / (np.linalg.norm(scg_dir) + 1e-10)
            }
        else:
            scg_directions[layer] = None

    return sr_directions, scg_directions


def run_steering_test(
    steering: ActivationSteering,
    sr_directions: dict,
    scg_directions: dict,
    pair_name: str = 'time_pressure',
    target_layers: list = None,
    alphas: list = None
) -> dict:
    """Test differential steering effects."""

    if target_layers is None:
        target_layers = [16, 20, 24, 28, 31]

    if alphas is None:
        alphas = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

    pair = get_pair(pair_name)
    # Use vulnerable prompt as baseline (should produce insecure code)
    test_prompt = pair['vulnerable']

    # Token mappings for this pair
    if pair['vulnerability_type'] == 'sprintf':
        tokens = {'secure': ' snprintf', 'insecure': ' sprintf'}
    else:  # strcat
        tokens = {'secure': ' strncat', 'insecure': ' strcat'}

    print(f"\nTesting pair: {pair_name}")
    print(f"  Vulnerability: {pair['vulnerability_type']}")
    print(f"  Tokens: secure='{tokens['secure']}', insecure='{tokens['insecure']}'")
    print(f"  Using vulnerable prompt (baseline should prefer insecure)")

    results = {
        'pair': pair_name,
        'tokens': tokens,
        'baseline': None,
        'sr_steering': {},
        'scg_steering': {}
    }

    # Baseline (no steering)
    baseline_probs = steering.get_token_probs_with_steering(
        test_prompt, np.zeros(steering.hidden_size), target_layers[0], 0.0, tokens
    )
    results['baseline'] = baseline_probs
    print(f"\nBaseline: P(secure)={baseline_probs['secure_prob']:.4f}, P(insecure)={baseline_probs['insecure_prob']:.4f}")

    # Test SR steering
    print("\n--- SR STEERING (Security Recognition direction) ---")
    for layer in target_layers:
        if sr_directions[layer] is None:
            continue

        sr_dir = sr_directions[layer]['raw']
        layer_results = []

        print(f"\nLayer {layer}:")
        for alpha in alphas:
            probs = steering.get_token_probs_with_steering(
                test_prompt, sr_dir, layer, alpha, tokens
            )
            layer_results.append({
                'alpha': alpha,
                'secure_prob': probs['secure_prob'],
                'insecure_prob': probs['insecure_prob'],
                'secure_delta': probs['secure_prob'] - baseline_probs['secure_prob']
            })
            print(f"  alpha={alpha:+.1f}: P(secure)={probs['secure_prob']:.4f} ({probs['secure_prob']-baseline_probs['secure_prob']:+.4f})")

        results['sr_steering'][layer] = layer_results

    # Test SCG steering
    print("\n--- SCG STEERING (Secure Code Generation direction) ---")
    for layer in target_layers:
        if scg_directions[layer] is None:
            continue

        scg_dir = scg_directions[layer]['raw']
        layer_results = []

        print(f"\nLayer {layer}:")
        for alpha in alphas:
            probs = steering.get_token_probs_with_steering(
                test_prompt, scg_dir, layer, alpha, tokens
            )
            layer_results.append({
                'alpha': alpha,
                'secure_prob': probs['secure_prob'],
                'insecure_prob': probs['insecure_prob'],
                'secure_delta': probs['secure_prob'] - baseline_probs['secure_prob']
            })
            print(f"  alpha={alpha:+.1f}: P(secure)={probs['secure_prob']:.4f} ({probs['secure_prob']-baseline_probs['secure_prob']:+.4f})")

        results['scg_steering'][layer] = layer_results

    return results


def analyze_differential_effects(results: dict) -> dict:
    """Analyze whether SR and SCG steering have different effects."""
    print("\n" + "=" * 60)
    print("DIFFERENTIAL EFFECT ANALYSIS")
    print("=" * 60)

    analysis = {}

    for layer in results['sr_steering']:
        if layer not in results['scg_steering']:
            continue

        sr_data = results['sr_steering'][layer]
        scg_data = results['scg_steering'][layer]

        sr_max_effect = max(abs(d['secure_delta']) for d in sr_data)
        scg_max_effect = max(abs(d['secure_delta']) for d in scg_data)

        if sr_max_effect > 0.001:
            ratio = scg_max_effect / sr_max_effect
        else:
            ratio = float('inf') if scg_max_effect > 0.001 else 1.0

        analysis[layer] = {
            'sr_max_effect': sr_max_effect,
            'scg_max_effect': scg_max_effect,
            'scg_to_sr_ratio': ratio
        }

        print(f"\nLayer {layer}:")
        print(f"  SR steering max effect: {sr_max_effect:.4f}")
        print(f"  SCG steering max effect: {scg_max_effect:.4f}")
        print(f"  Ratio (SCG/SR): {ratio:.2f}x")

        if ratio > 2:
            print("  -> SCG has STRONGER effect (supports separation)")
        elif ratio < 0.5:
            print("  -> SR has STRONGER effect (contradicts separation)")
        else:
            print("  -> Similar effects")

    return analysis


def plot_results(results: dict, analysis: dict, output_path: Path):
    """Plot steering results."""
    layers = sorted(results['sr_steering'].keys())
    if not layers:
        print("No results to plot")
        return

    n_layers = len(layers)
    fig, axes = plt.subplots(n_layers, 2, figsize=(12, 4*n_layers))

    if n_layers == 1:
        axes = axes.reshape(1, -1)

    for i, layer in enumerate(layers):
        sr_data = results['sr_steering'][layer]
        scg_data = results['scg_steering'][layer]

        alphas = [d['alpha'] for d in sr_data]

        # SR steering
        ax1 = axes[i, 0]
        sr_secure = [d['secure_prob'] for d in sr_data]
        sr_insecure = [d['insecure_prob'] for d in sr_data]

        ax1.plot(alphas, sr_secure, 'g-o', label='P(secure)')
        ax1.plot(alphas, sr_insecure, 'r-s', label='P(insecure)')
        ax1.axhline(y=results['baseline']['secure_prob'], color='g', linestyle=':', alpha=0.5)
        ax1.axhline(y=results['baseline']['insecure_prob'], color='r', linestyle=':', alpha=0.5)
        ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        ax1.set_xlabel('Alpha')
        ax1.set_ylabel('Probability')
        ax1.set_title(f'Layer {layer}: SR Steering')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # SCG steering
        ax2 = axes[i, 1]
        scg_secure = [d['secure_prob'] for d in scg_data]
        scg_insecure = [d['insecure_prob'] for d in scg_data]

        ax2.plot(alphas, scg_secure, 'g-o', label='P(secure)')
        ax2.plot(alphas, scg_insecure, 'r-s', label='P(insecure)')
        ax2.axhline(y=results['baseline']['secure_prob'], color='g', linestyle=':', alpha=0.5)
        ax2.axhline(y=results['baseline']['insecure_prob'], color='r', linestyle=':', alpha=0.5)
        ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Alpha')
        ax2.set_ylabel('Probability')
        ax2.set_title(f'Layer {layer}: SCG Steering')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {output_path}")


def main():
    data_dir = Path(__file__).parent / "data"
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("DIFFERENTIAL STEERING TEST: CWE-787 PROMPT PAIRS")
    print("=" * 70)

    # Load directions
    print("\nLoading directions from data...")
    sr_directions, scg_directions = load_directions(data_dir)

    # Initialize steering
    print("\nInitializing steering...")
    steering = ActivationSteering()

    # Run test
    results = run_steering_test(
        steering, sr_directions, scg_directions,
        pair_name='time_pressure',
        target_layers=[16, 20, 24, 28, 31],
        alphas=[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    )

    # Analyze
    analysis = analyze_differential_effects(results)

    # Compute summary
    valid_ratios = [a['scg_to_sr_ratio'] for a in analysis.values()
                    if a['scg_to_sr_ratio'] != float('inf')]
    avg_ratio = np.mean(valid_ratios) if valid_ratios else 1.0

    if avg_ratio > 2:
        conclusion = "SUPPORTS SEPARATION: SCG steering has stronger effect"
    elif avg_ratio < 0.5:
        conclusion = "CONTRADICTS SEPARATION: SR steering has stronger effect"
    else:
        conclusion = "INCONCLUSIVE: Similar effects from both directions"

    print(f"\nAverage SCG/SR ratio: {avg_ratio:.2f}")
    print(f"Conclusion: {conclusion}")

    # Plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = results_dir / f"differential_steering_{timestamp}.png"
    plot_results(results, analysis, plot_path)

    # Save
    save_data = {
        'timestamp': timestamp,
        'test_pair': results['pair'],
        'baseline': results['baseline'],
        'sr_steering': {str(k): v for k, v in results['sr_steering'].items()},
        'scg_steering': {str(k): v for k, v in results['scg_steering'].items()},
        'analysis': {str(k): v for k, v in analysis.items()},
        'avg_ratio': avg_ratio,
        'conclusion': conclusion
    }

    with open(results_dir / f"differential_steering_{timestamp}.json", 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to: {results_dir}")
    print(f"\nNext step: python 04_jailbreak_test.py")

    return results, analysis


if __name__ == "__main__":
    results, analysis = main()
