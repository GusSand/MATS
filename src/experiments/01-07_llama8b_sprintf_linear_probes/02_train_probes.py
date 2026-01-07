#!/usr/bin/env python3
"""
Train linear probes on collected activations.

Trains two types of probes:
1. Context probe: Detect secure vs neutral context from activations
2. Behavior probe: Predict snprintf vs sprintf output from activations
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt


def load_data(data_dir: Path) -> tuple:
    """Load most recent activation files."""
    # Find most recent files
    context_files = sorted(data_dir.glob("context_activations_*.npz"))
    behavior_files = sorted(data_dir.glob("behavior_activations_*.npz"))

    if not context_files or not behavior_files:
        raise FileNotFoundError("No activation files found. Run 01_collect_activations.py first.")

    context_file = context_files[-1]
    behavior_file = behavior_files[-1]

    print(f"Loading context data: {context_file.name}")
    print(f"Loading behavior data: {behavior_file.name}")

    context_data = np.load(context_file)
    behavior_data = np.load(behavior_file)

    return context_data, behavior_data


def train_probe_at_layer(X: np.ndarray, y: np.ndarray, layer_idx: int) -> dict:
    """Train a logistic regression probe at one layer."""
    if len(X) < 10:
        return {
            'layer': layer_idx,
            'n_samples': len(X),
            'accuracy': None,
            'cv_accuracy': None,
            'auc': None,
            'error': 'Not enough samples'
        }

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train logistic regression
    clf = LogisticRegression(max_iter=1000, random_state=42)

    try:
        # Cross-validation on training set
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5)

        # Final fit and test
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        return {
            'layer': layer_idx,
            'n_samples': len(X),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'accuracy': float(accuracy),
            'cv_accuracy_mean': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
            'auc': float(auc),
            'class_balance': float(y.mean()),
            'coefficients_norm': float(np.linalg.norm(clf.coef_))
        }
    except Exception as e:
        return {
            'layer': layer_idx,
            'n_samples': len(X),
            'accuracy': None,
            'error': str(e)
        }


def train_all_probes(data: np.lib.npyio.NpzFile, probe_name: str) -> list:
    """Train probes at all layers."""
    print(f"\n{'='*60}")
    print(f"TRAINING {probe_name.upper()} PROBES")
    print(f"{'='*60}")

    results = []

    # Find number of layers
    layer_keys = [k for k in data.files if k.startswith('X_layer_')]
    n_layers = len(layer_keys)

    for layer_idx in range(n_layers):
        X = data[f'X_layer_{layer_idx}']
        y = data[f'y_layer_{layer_idx}']

        result = train_probe_at_layer(X, y, layer_idx)
        results.append(result)

        if result['accuracy'] is not None:
            bar = "█" * int(result['accuracy'] * 20)
            symbol = "✅" if result['accuracy'] > 0.8 else "🔶" if result['accuracy'] > 0.6 else "❌"
            print(f"Layer {layer_idx:2d}: {bar:20s} {result['accuracy']*100:5.1f}% (AUC={result['auc']:.3f}) {symbol}")
        else:
            print(f"Layer {layer_idx:2d}: ERROR - {result.get('error', 'Unknown')}")

    return results


def plot_results(context_results: list, behavior_results: list, output_path: Path):
    """Plot probe accuracy across layers for both probe types."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Extract accuracies
    context_acc = [r['accuracy'] if r['accuracy'] else 0.5 for r in context_results]
    behavior_acc = [r['accuracy'] if r['accuracy'] else 0.5 for r in behavior_results]

    context_auc = [r['auc'] if r.get('auc') else 0.5 for r in context_results]
    behavior_auc = [r['auc'] if r.get('auc') else 0.5 for r in behavior_results]

    layers = range(len(context_results))

    # Plot 1: Context probe
    ax1 = axes[0]
    ax1.plot(layers, context_acc, 'b-o', label='Accuracy', markersize=4)
    ax1.plot(layers, context_auc, 'b--', label='AUC', alpha=0.7)
    ax1.axhline(y=0.5, color='gray', linestyle=':', label='Chance')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Score')
    ax1.set_title('Probe A: Context Detection\n(Secure vs Neutral)')
    ax1.legend()
    ax1.set_ylim(0.4, 1.05)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Behavior probe
    ax2 = axes[1]
    ax2.plot(layers, behavior_acc, 'r-o', label='Accuracy', markersize=4)
    ax2.plot(layers, behavior_auc, 'r--', label='AUC', alpha=0.7)
    ax2.axhline(y=0.5, color='gray', linestyle=':', label='Chance')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Score')
    ax2.set_title('Probe B: Behavior Prediction\n(snprintf vs sprintf)')
    ax2.legend()
    ax2.set_ylim(0.4, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Plot saved to: {output_path}")


def analyze_results(context_results: list, behavior_results: list) -> dict:
    """Analyze and compare results from both probes."""
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    # Find best layers for each probe
    context_best = max(context_results, key=lambda x: x['accuracy'] or 0)
    behavior_best = max(behavior_results, key=lambda x: x['accuracy'] or 0)

    # Find when accuracy crosses thresholds
    def find_threshold_layer(results, threshold):
        for r in results:
            if r['accuracy'] and r['accuracy'] >= threshold:
                return r['layer']
        return None

    context_70 = find_threshold_layer(context_results, 0.7)
    context_80 = find_threshold_layer(context_results, 0.8)
    context_90 = find_threshold_layer(context_results, 0.9)

    behavior_70 = find_threshold_layer(behavior_results, 0.7)
    behavior_80 = find_threshold_layer(behavior_results, 0.8)
    behavior_90 = find_threshold_layer(behavior_results, 0.9)

    print("\n📈 Context Probe (Secure vs Neutral):")
    print(f"  Best layer: {context_best['layer']} with {context_best['accuracy']*100:.1f}% accuracy")
    print(f"  70% accuracy at layer: {context_70}")
    print(f"  80% accuracy at layer: {context_80}")
    print(f"  90% accuracy at layer: {context_90}")

    print("\n📈 Behavior Probe (snprintf vs sprintf):")
    print(f"  Best layer: {behavior_best['layer']} with {behavior_best['accuracy']*100:.1f}% accuracy")
    print(f"  70% accuracy at layer: {behavior_70}")
    print(f"  80% accuracy at layer: {behavior_80}")
    print(f"  90% accuracy at layer: {behavior_90}")

    # Compare timing
    print("\n🔍 Comparison:")
    if context_80 and behavior_80:
        if context_80 < behavior_80:
            print(f"  Context becomes readable (80%) at layer {context_80}")
            print(f"  Behavior becomes predictable (80%) at layer {behavior_80}")
            print(f"  → Context precedes behavior by {behavior_80 - context_80} layers")
        elif behavior_80 < context_80:
            print(f"  Behavior becomes predictable (80%) at layer {behavior_80}")
            print(f"  Context becomes readable (80%) at layer {context_80}")
            print(f"  → Behavior precedes context by {context_80 - behavior_80} layers (surprising!)")
        else:
            print(f"  Both reach 80% at layer {context_80}")

    return {
        'context': {
            'best_layer': context_best['layer'],
            'best_accuracy': context_best['accuracy'],
            'threshold_70': context_70,
            'threshold_80': context_80,
            'threshold_90': context_90
        },
        'behavior': {
            'best_layer': behavior_best['layer'],
            'best_accuracy': behavior_best['accuracy'],
            'threshold_70': behavior_70,
            'threshold_80': behavior_80,
            'threshold_90': behavior_90
        }
    }


def main():
    data_dir = Path(__file__).parent / "data"
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Load data
    context_data, behavior_data = load_data(data_dir)

    # Train probes
    context_results = train_all_probes(context_data, "context")
    behavior_results = train_all_probes(behavior_data, "behavior")

    # Analyze
    analysis = analyze_results(context_results, behavior_results)

    # Plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = results_dir / f"probe_accuracy_{timestamp}.png"
    plot_results(context_results, behavior_results, plot_path)

    # Save results
    full_results = {
        'timestamp': timestamp,
        'context_probe': context_results,
        'behavior_probe': behavior_results,
        'analysis': analysis
    }

    with open(results_dir / f"probe_results_{timestamp}.json", 'w') as f:
        json.dump(full_results, f, indent=2)

    print(f"\n💾 Results saved to: {results_dir}")

    return full_results


if __name__ == "__main__":
    results = main()
