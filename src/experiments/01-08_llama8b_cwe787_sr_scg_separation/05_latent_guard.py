#!/usr/bin/env python3
"""
Latent Security Guard for CWE-787 Prompt Pairs

Use the model's internal SR representation to detect security-relevant contexts,
REGARDLESS of actual output.

Key insight: If SR is robust and separate from SCG, we can detect when:
1. The model "knows" code should be secure but outputs insecure code
2. Adversarial prompts try to bypass security

This is a defense mechanism using the model's own representations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import json
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from validated_pairs import get_all_pairs
from utils.activation_collector import ActivationCollector


class LatentSecurityGuard:
    """Detects security-relevant contexts using internal SR representation."""

    def __init__(self, probes: dict, scalers: dict, layers: list, threshold: float = 0.5):
        self.probes = probes
        self.scalers = scalers
        self.layers = layers
        self.threshold = threshold

    def get_security_score(self, activations: dict) -> dict:
        """Compute security relevance score from activations."""
        scores = {}

        for layer in self.layers:
            if layer not in self.probes:
                continue

            act = activations[layer]
            if act.ndim == 1:
                act = act.reshape(1, -1)

            act_scaled = self.scalers[layer].transform(act)
            prob = self.probes[layer].predict_proba(act_scaled)[0, 1]
            scores[layer] = prob

        # Aggregate using late layers
        late_layers = [l for l in self.layers if l >= 16]
        if late_layers:
            agg_score = np.mean([scores[l] for l in late_layers if l in scores])
        else:
            agg_score = np.mean(list(scores.values()))

        return {
            'layer_scores': scores,
            'aggregate_score': agg_score,
            'is_security_relevant': agg_score > self.threshold
        }

    def detect(self, activations: dict) -> bool:
        """Simple binary detection."""
        return self.get_security_score(activations)['is_security_relevant']


def train_latent_guard(data_dir: Path, layers_to_use: list = None) -> tuple:
    """Train the Latent Security Guard probes."""
    if layers_to_use is None:
        layers_to_use = [0, 4, 8, 12, 16, 20, 24, 28, 31]

    sr_files = sorted(data_dir.glob("sr_data_*.npz"))
    sr_npz = np.load(sr_files[-1])

    probes = {}
    scalers = {}
    results = {}

    print("Training Latent Security Guard probes...")
    print(f"Using layers: {layers_to_use}")

    for layer in layers_to_use:
        X = sr_npz[f'X_layer_{layer}']
        y = sr_npz[f'y_layer_{layer}']

        # Split for evaluation
        n = len(X)
        idx = np.random.permutation(n)
        train_idx = idx[:int(0.8*n)]
        test_idx = idx[int(0.8*n):]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_scaled, y_train)

        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)

        probes[layer] = clf
        scalers[layer] = scaler
        results[layer] = {
            'accuracy': float(acc),
            'n_train': len(X_train),
            'n_test': len(X_test)
        }

        print(f"  Layer {layer}: {acc*100:.1f}% accuracy")

    guard = LatentSecurityGuard(probes, scalers, layers_to_use)
    return guard, results


def evaluate_guard(guard: LatentSecurityGuard, collector: ActivationCollector,
                   n_samples_per_pair: int = 10) -> dict:
    """Evaluate guard on held-out samples."""
    pairs = get_all_pairs()

    results = {
        'true_positives': 0,  # Secure prompt, guard flags
        'false_positives': 0,  # Vulnerable prompt, guard flags
        'true_negatives': 0,  # Vulnerable prompt, guard doesn't flag
        'false_negatives': 0,  # Secure prompt, guard doesn't flag
        'samples': []
    }

    print("\nEvaluating Latent Security Guard...")

    for pair in pairs:
        pair_id = pair['id']

        # Test secure prompts (should be flagged - label=1)
        for _ in range(n_samples_per_pair):
            acts = collector.get_activations(pair['secure'])
            guard_result = guard.get_security_score(acts)

            sample = {
                'pair_id': pair_id,
                'prompt_type': 'secure',
                'expected': True,
                'guard_score': guard_result['aggregate_score'],
                'guard_flags': guard_result['is_security_relevant']
            }
            results['samples'].append(sample)

            if guard_result['is_security_relevant']:
                results['true_positives'] += 1
            else:
                results['false_negatives'] += 1

        # Test vulnerable prompts (should NOT be flagged - label=0)
        for _ in range(n_samples_per_pair):
            acts = collector.get_activations(pair['vulnerable'])
            guard_result = guard.get_security_score(acts)

            sample = {
                'pair_id': pair_id,
                'prompt_type': 'vulnerable',
                'expected': False,
                'guard_score': guard_result['aggregate_score'],
                'guard_flags': guard_result['is_security_relevant']
            }
            results['samples'].append(sample)

            if not guard_result['is_security_relevant']:
                results['true_negatives'] += 1
            else:
                results['false_positives'] += 1

    # Compute metrics
    tp, fp, tn, fn = (results['true_positives'], results['false_positives'],
                      results['true_negatives'], results['false_negatives'])

    total = tp + fp + tn + fn
    results['metrics'] = {
        'accuracy': (tp + tn) / total if total > 0 else 0,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'f1': 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0
    }

    return results


def plot_guard_results(results: dict, output_path: Path):
    """Plot guard evaluation results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Confusion matrix
    ax1 = axes[0]
    cm = np.array([
        [results['true_negatives'], results['false_positives']],
        [results['false_negatives'], results['true_positives']]
    ])
    im = ax1.imshow(cm, cmap='Blues')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['No Flag', 'Flag'])
    ax1.set_yticklabels(['Vulnerable', 'Secure'])
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual Prompt Type')
    ax1.set_title('Latent Guard Confusion Matrix')

    for i in range(2):
        for j in range(2):
            ax1.text(j, i, cm[i, j], ha='center', va='center', fontsize=14)

    # Score distribution
    ax2 = axes[1]
    secure_scores = [s['guard_score'] for s in results['samples'] if s['prompt_type'] == 'secure']
    vuln_scores = [s['guard_score'] for s in results['samples'] if s['prompt_type'] == 'vulnerable']

    ax2.hist(secure_scores, bins=20, alpha=0.7, label='Secure prompts', color='green')
    ax2.hist(vuln_scores, bins=20, alpha=0.7, label='Vulnerable prompts', color='red')
    ax2.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    ax2.set_xlabel('Guard Score')
    ax2.set_ylabel('Count')
    ax2.set_title('Guard Score Distribution')
    ax2.legend()

    # Metrics
    ax3 = axes[2]
    metrics = results['metrics']
    bars = ax3.bar(
        ['Accuracy', 'Precision', 'Recall', 'F1'],
        [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']],
        color=['blue', 'green', 'orange', 'purple']
    )
    ax3.set_ylim(0, 1.1)
    ax3.set_ylabel('Score')
    ax3.set_title('Latent Guard Performance')

    for bar, val in zip(bars, [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.2f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {output_path}")


def main():
    data_dir = Path(__file__).parent / "data"
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("LATENT SECURITY GUARD: CWE-787 PROMPT PAIRS")
    print("=" * 70)

    # Train guard
    guard, train_results = train_latent_guard(
        data_dir, layers_to_use=[0, 4, 8, 12, 16, 20, 24, 28, 31]
    )

    # Initialize collector
    print("\nInitializing collector for evaluation...")
    collector = ActivationCollector()

    # Evaluate
    eval_results = evaluate_guard(guard, collector, n_samples_per_pair=5)

    # Print summary
    print("\n" + "=" * 60)
    print("LATENT SECURITY GUARD EVALUATION")
    print("=" * 60)

    metrics = eval_results['metrics']
    print(f"\nGuard Performance:")
    print(f"  Accuracy:  {metrics['accuracy']*100:.1f}%")
    print(f"  Precision: {metrics['precision']*100:.1f}%")
    print(f"  Recall:    {metrics['recall']*100:.1f}%")
    print(f"  F1 Score:  {metrics['f1']*100:.1f}%")

    print(f"\nConfusion Matrix:")
    print(f"  True Positives (secure, flagged): {eval_results['true_positives']}")
    print(f"  True Negatives (vuln, not flagged): {eval_results['true_negatives']}")
    print(f"  False Positives (vuln, flagged): {eval_results['false_positives']}")
    print(f"  False Negatives (secure, not flagged): {eval_results['false_negatives']}")

    # Plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = results_dir / f"latent_guard_{timestamp}.png"
    plot_guard_results(eval_results, plot_path)

    # Save
    save_data = {
        'timestamp': timestamp,
        'train_results': train_results,
        'eval_metrics': metrics,
        'confusion_matrix': {
            'tp': eval_results['true_positives'],
            'fp': eval_results['false_positives'],
            'tn': eval_results['true_negatives'],
            'fn': eval_results['false_negatives']
        },
        'n_samples': len(eval_results['samples'])
    }

    with open(results_dir / f"latent_guard_{timestamp}.json", 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to: {results_dir}")
    print(f"\nNext step: python 06_synthesis.py")

    return guard, eval_results


if __name__ == "__main__":
    guard, eval_results = main()
