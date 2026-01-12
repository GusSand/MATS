#!/usr/bin/env python3
"""
Differential Steering Test

Key question: Can we steer SR (Security Recognition) without affecting SCG
(Secure Code Generation), and vice versa?

If SR and SCG are separate directions (like harmfulness vs refusal):
- Steering along SR should change "does model recognize security context"
  but NOT "will model output secure code"
- Steering along SCG should change "will model output secure code"
  but NOT "does model recognize security context"

This is the causal test of the separation hypothesis.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import pickle

from config.security_pairs import SECURITY_PAIRS, CORE_PAIRS
from utils.steering import ActivationSteering
from utils.probe_trainer import ProbeTrainer, extract_directions_from_data


def load_probes_and_directions(results_dir: Path, data_dir: Path, n_layers: int = 32):
    """Load trained probes and extract directions from data."""
    # Load most recent probe results
    probe_files = sorted(results_dir.glob("sr_scg_probes_*.json"))
    if not probe_files:
        raise FileNotFoundError("No probe results found. Run 03_train_separate_probes.py first.")

    with open(probe_files[-1]) as f:
        probe_results = json.load(f)

    # Load merged data to extract directions
    sr_files = sorted(data_dir.glob("sr_merged_*.npz"))
    scg_files = sorted(data_dir.glob("scg_merged_*.npz"))

    sr_npz = np.load(sr_files[-1])
    scg_npz = np.load(scg_files[-1])

    # Convert to dict format
    sr_data = {}
    scg_data = {}
    for layer in range(n_layers):
        sr_data[layer] = {
            'X': sr_npz[f'X_layer_{layer}'],
            'y': sr_npz[f'y_layer_{layer}']
        }
        scg_data[layer] = {
            'X': scg_npz[f'X_layer_{layer}'],
            'y': scg_npz[f'y_layer_{layer}']
        }

    # Extract directions using mean difference
    sr_directions = extract_directions_from_data(sr_data, n_layers)
    scg_directions = extract_directions_from_data(scg_data, n_layers)

    return probe_results, sr_directions, scg_directions, sr_data, scg_data


def run_differential_steering_test(
    steering: ActivationSteering,
    sr_directions: dict,
    scg_directions: dict,
    test_pair: str,
    target_layers: list = None,
    alphas: list = None
) -> dict:
    """
    Test differential steering: does steering one direction affect the other?

    We measure:
    1. Steer SR direction -> measure change in P(secure_token) and P(insecure_token)
    2. Steer SCG direction -> measure change in P(secure_token) and P(insecure_token)

    If separate: SR steering should have less effect on output than SCG steering
    """
    if target_layers is None:
        target_layers = [16, 20, 24, 28, 31]  # Late layers where behavior emerges

    if alphas is None:
        alphas = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

    config = SECURITY_PAIRS[test_pair]
    neutral_prompt = config['neutral_templates'][0]

    # Get token names for this pair
    secure_token = f" {config['secure'].split('(')[0]}"  # e.g., " snprintf"
    insecure_token = f" {config['insecure'].split('(')[0]}"  # e.g., " sprintf"

    tokens = {'secure': secure_token, 'insecure': insecure_token}

    print(f"\nTesting pair: {test_pair}")
    print(f"  Secure token: '{secure_token}'")
    print(f"  Insecure token: '{insecure_token}'")
    print(f"  Prompt: {neutral_prompt[:60]}...")

    results = {
        'pair': test_pair,
        'tokens': tokens,
        'sr_steering': {},
        'scg_steering': {}
    }

    # Baseline (no steering)
    baseline_acts = steering.collect_activations(neutral_prompt)
    baseline_probs = steering.get_token_probs_with_steering(
        neutral_prompt,
        np.zeros(steering.hidden_size),  # Zero steering
        target_layers[0],
        alpha=0.0,
        tokens=tokens
    )
    print(f"\nBaseline: P(secure)={baseline_probs['secure_prob']:.4f}, P(insecure)={baseline_probs['insecure_prob']:.4f}")

    results['baseline'] = baseline_probs

    # Test SR steering at each layer
    print("\n--- SR STEERING (Security Recognition direction) ---")
    for layer in target_layers:
        if sr_directions[layer] is None:
            continue

        sr_dir = sr_directions[layer]['raw']  # Use raw direction (not normalized) for steering
        layer_results = []

        print(f"\nLayer {layer}:")
        for alpha in alphas:
            probs = steering.get_token_probs_with_steering(
                neutral_prompt, sr_dir, layer, alpha, tokens
            )
            layer_results.append({
                'alpha': alpha,
                'secure_prob': probs['secure_prob'],
                'insecure_prob': probs['insecure_prob'],
                'secure_delta': probs['secure_prob'] - baseline_probs['secure_prob'],
                'insecure_delta': probs['insecure_prob'] - baseline_probs['insecure_prob']
            })
            print(f"  alpha={alpha:+.1f}: P(secure)={probs['secure_prob']:.4f} ({probs['secure_prob']-baseline_probs['secure_prob']:+.4f})")

        results['sr_steering'][layer] = layer_results

    # Test SCG steering at each layer
    print("\n--- SCG STEERING (Secure Code Generation direction) ---")
    for layer in target_layers:
        if scg_directions[layer] is None:
            continue

        scg_dir = scg_directions[layer]['raw']
        layer_results = []

        print(f"\nLayer {layer}:")
        for alpha in alphas:
            probs = steering.get_token_probs_with_steering(
                neutral_prompt, scg_dir, layer, alpha, tokens
            )
            layer_results.append({
                'alpha': alpha,
                'secure_prob': probs['secure_prob'],
                'insecure_prob': probs['insecure_prob'],
                'secure_delta': probs['secure_prob'] - baseline_probs['secure_prob'],
                'insecure_delta': probs['insecure_prob'] - baseline_probs['insecure_prob']
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

        # Compute max effect (at alpha=2.0 or highest available)
        sr_max_effect = max(abs(d['secure_delta']) for d in sr_data)
        scg_max_effect = max(abs(d['secure_delta']) for d in scg_data)

        # Compute effect ratio
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
        print(f"  SR steering max effect on P(secure): {sr_max_effect:.4f}")
        print(f"  SCG steering max effect on P(secure): {scg_max_effect:.4f}")
        print(f"  Ratio (SCG/SR): {ratio:.2f}x")

        if ratio > 2:
            print(f"  -> SCG has STRONGER effect (supports separation)")
        elif ratio < 0.5:
            print(f"  -> SR has STRONGER effect (contradicts separation)")
        else:
            print(f"  -> Similar effects (no clear separation)")

    return analysis


def plot_differential_steering(results: dict, output_path: Path):
    """Visualize differential steering effects."""
    layers = sorted(results['sr_steering'].keys())
    if not layers:
        print("No steering results to plot")
        return

    n_layers = len(layers)
    fig, axes = plt.subplots(n_layers, 2, figsize=(12, 4*n_layers))

    if n_layers == 1:
        axes = axes.reshape(1, -1)

    for i, layer in enumerate(layers):
        sr_data = results['sr_steering'][layer]
        scg_data = results['scg_steering'][layer]

        alphas = [d['alpha'] for d in sr_data]

        # Plot SR steering
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
        ax1.set_title(f'Layer {layer}: SR Steering\n(Security Recognition direction)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot SCG steering
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
        ax2.set_title(f'Layer {layer}: SCG Steering\n(Secure Code Generation direction)')
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

    # Load probes and directions
    print("Loading probes and directions...")
    probe_results, sr_directions, scg_directions, sr_data, scg_data = load_probes_and_directions(
        results_dir, data_dir
    )

    # Initialize steering
    steering = ActivationSteering()

    # Run differential steering test
    test_pair = "sprintf_snprintf"  # Use the original pair first
    results = run_differential_steering_test(
        steering, sr_directions, scg_directions, test_pair,
        target_layers=[16, 20, 24, 28, 31],
        alphas=[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    )

    # Analyze differential effects
    analysis = analyze_differential_effects(results)

    # Plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = results_dir / f"differential_steering_{timestamp}.png"
    plot_differential_steering(results, plot_path)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    avg_ratio = np.mean([a['scg_to_sr_ratio'] for a in analysis.values()
                         if a['scg_to_sr_ratio'] != float('inf')])

    if avg_ratio > 2:
        conclusion = "SUPPORTS SEPARATION: SCG steering has stronger effect on output"
    elif avg_ratio < 0.5:
        conclusion = "CONTRADICTS SEPARATION: SR steering has stronger effect"
    else:
        conclusion = "INCONCLUSIVE: Similar effects from both directions"

    print(f"\nAverage SCG/SR effect ratio: {avg_ratio:.2f}")
    print(f"Conclusion: {conclusion}")

    # Save results
    save_data = {
        'timestamp': timestamp,
        'test_pair': test_pair,
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

    return results, analysis


if __name__ == "__main__":
    results, analysis = main()
