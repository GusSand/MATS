#!/usr/bin/env python3
"""
Latent Security Guard

Inspired by "Latent Guard" from the arxiv paper: use the model's internal
security recognition (SR) representation to detect potentially insecure code,
REGARDLESS of what the model actually outputs.

Key insight: If SR is robust and separate from SCG, we can detect when:
1. The model "knows" code should be secure but outputs insecure code
2. Adversarial prompts try to bypass security without changing internal recognition

This is a defense mechanism using the model's own representations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import json
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

from config.security_pairs import SECURITY_PAIRS, CORE_PAIRS
from utils.activation_collector import ActivationCollector
from utils.probe_trainer import extract_directions_from_data


class LatentSecurityGuard:
    """
    A security detector that uses the model's internal SR representation
    to detect security-relevant contexts, independent of output.
    """

    def __init__(self, probes: dict, scalers: dict, layers: list, threshold: float = 0.5):
        """
        Args:
            probes: Dict of {layer: trained_probe}
            scalers: Dict of {layer: scaler}
            layers: List of layers to use for detection
            threshold: Probability threshold for flagging as security-relevant
        """
        self.probes = probes
        self.scalers = scalers
        self.layers = layers
        self.threshold = threshold

    def get_security_score(self, activations: dict) -> dict:
        """
        Compute security relevance score from activations.

        Returns probability that context is security-relevant at each layer
        and an aggregated score.
        """
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

        # Aggregate using mean of late layers (more reliable)
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
        """Simple binary detection: is this security-relevant?"""
        result = self.get_security_score(activations)
        return result['is_security_relevant']


def train_latent_guard(data_dir: Path, n_layers: int = 32, layers_to_use: list = None):
    """Train the Latent Security Guard probes."""
    if layers_to_use is None:
        layers_to_use = [0, 4, 8, 12, 16, 20, 24, 28, 31]

    # Load SR data
    sr_files = sorted(data_dir.glob("sr_merged_*.npz"))
    if not sr_files:
        raise FileNotFoundError("SR data not found")

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
            'accuracy': acc,
            'n_train': len(X_train),
            'n_test': len(X_test)
        }

        print(f"  Layer {layer}: {acc*100:.1f}% accuracy")

    guard = LatentSecurityGuard(probes, scalers, layers_to_use)

    return guard, results


def evaluate_guard_vs_output(
    guard: LatentSecurityGuard,
    collector: ActivationCollector,
    test_pairs: list = None,
    n_samples: int = 20
) -> dict:
    """
    Evaluate the guard's ability to detect security contexts vs actual model output.

    Compare:
    1. Guard's detection (based on SR)
    2. Model's actual output (secure vs insecure)

    We want to see cases where guard detects security-relevant context
    even when model outputs insecure code.
    """
    if test_pairs is None:
        test_pairs = CORE_PAIRS

    results = {
        'true_positives': 0,  # Guard flags, context IS security-relevant
        'false_positives': 0,  # Guard flags, context is NOT security-relevant
        'true_negatives': 0,  # Guard doesn't flag, context is NOT security-relevant
        'false_negatives': 0,  # Guard doesn't flag, context IS security-relevant
        'guard_vs_output_mismatch': [],  # Cases where guard sees security but output is insecure
        'samples': []
    }

    print("\nEvaluating Latent Security Guard...")

    for pair_name in test_pairs:
        config = SECURITY_PAIRS[pair_name]
        detection_patterns = config['detection_patterns']

        # Test on secure contexts (guard should flag, output should be secure)
        for template in config['secure_templates'][:2]:
            for _ in range(n_samples // 2):
                acts = collector.get_activations(template)
                guard_result = guard.get_security_score(acts)

                # Generate output
                gen_result = collector.generate_and_classify(template, detection_patterns)

                sample = {
                    'pair': pair_name,
                    'context_type': 'secure',
                    'guard_score': guard_result['aggregate_score'],
                    'guard_flags': guard_result['is_security_relevant'],
                    'output_label': gen_result['label']
                }
                results['samples'].append(sample)

                # Guard should flag (context is security-relevant)
                if guard_result['is_security_relevant']:
                    results['true_positives'] += 1
                else:
                    results['false_negatives'] += 1

                # Check for mismatch: guard flags but output is insecure
                if guard_result['is_security_relevant'] and gen_result['label'] == 'insecure':
                    results['guard_vs_output_mismatch'].append(sample)

        # Test on neutral contexts (guard should NOT flag, output depends)
        for template in config['neutral_templates'][:2]:
            for _ in range(n_samples // 2):
                acts = collector.get_activations(template)
                guard_result = guard.get_security_score(acts)

                gen_result = collector.generate_and_classify(template, detection_patterns)

                sample = {
                    'pair': pair_name,
                    'context_type': 'neutral',
                    'guard_score': guard_result['aggregate_score'],
                    'guard_flags': guard_result['is_security_relevant'],
                    'output_label': gen_result['label']
                }
                results['samples'].append(sample)

                # Guard should NOT flag (context is not security-relevant)
                if not guard_result['is_security_relevant']:
                    results['true_negatives'] += 1
                else:
                    results['false_positives'] += 1

    # Compute metrics
    tp, fp, tn, fn = (results['true_positives'], results['false_positives'],
                      results['true_negatives'], results['false_negatives'])

    results['metrics'] = {
        'accuracy': (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'f1': 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0,
        'n_mismatches': len(results['guard_vs_output_mismatch'])
    }

    return results


def plot_guard_evaluation(results: dict, output_path: Path):
    """Plot guard evaluation results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Confusion matrix
    ax1 = axes[0]
    cm = np.array([
        [results['true_negatives'], results['false_positives']],
        [results['false_negatives'], results['true_positives']]
    ])
    im = ax1.imshow(cm, cmap='Blues')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Pred: No Flag', 'Pred: Flag'])
    ax1.set_yticklabels(['Actual: Neutral', 'Actual: Secure'])
    ax1.set_title('Latent Guard Confusion Matrix')

    for i in range(2):
        for j in range(2):
            ax1.text(j, i, cm[i, j], ha='center', va='center', fontsize=14)

    # Plot 2: Score distribution by context type
    ax2 = axes[1]
    secure_scores = [s['guard_score'] for s in results['samples'] if s['context_type'] == 'secure']
    neutral_scores = [s['guard_score'] for s in results['samples'] if s['context_type'] == 'neutral']

    ax2.hist(secure_scores, bins=20, alpha=0.7, label='Secure context', color='green')
    ax2.hist(neutral_scores, bins=20, alpha=0.7, label='Neutral context', color='red')
    ax2.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    ax2.set_xlabel('Guard Score')
    ax2.set_ylabel('Count')
    ax2.set_title('Guard Score Distribution')
    ax2.legend()

    # Plot 3: Metrics
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

    # Train guard
    guard, train_results = train_latent_guard(
        data_dir,
        layers_to_use=[0, 4, 8, 12, 16, 20, 24, 28, 31]
    )

    # Initialize collector
    collector = ActivationCollector()

    # Evaluate guard
    eval_results = evaluate_guard_vs_output(guard, collector, CORE_PAIRS, n_samples=15)

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

    print(f"\nGuard vs Output Mismatches: {metrics['n_mismatches']}")
    print("(Cases where guard flags security context but model outputs insecure)")

    if eval_results['guard_vs_output_mismatch']:
        print("\nMismatch examples:")
        for m in eval_results['guard_vs_output_mismatch'][:3]:
            print(f"  - {m['pair']}: guard={m['guard_score']:.2f}, output={m['output_label']}")

    # Plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = results_dir / f"latent_guard_{timestamp}.png"
    plot_guard_evaluation(eval_results, plot_path)

    # Save results
    save_data = {
        'timestamp': timestamp,
        'train_results': train_results,
        'eval_metrics': metrics,
        'n_samples': len(eval_results['samples']),
        'confusion_matrix': {
            'tp': eval_results['true_positives'],
            'fp': eval_results['false_positives'],
            'tn': eval_results['true_negatives'],
            'fn': eval_results['false_negatives']
        },
        'n_mismatches': metrics['n_mismatches']
    }

    with open(results_dir / f"latent_guard_{timestamp}.json", 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to: {results_dir}")

    return guard, eval_results


if __name__ == "__main__":
    guard, eval_results = main()
