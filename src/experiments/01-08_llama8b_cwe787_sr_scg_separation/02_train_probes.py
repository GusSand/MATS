#!/usr/bin/env python3
"""
Train separate SR and SCG probes and compute direction similarity.

Key question: Are SR (Security Recognition) and SCG (Secure Code Generation)
encoded as separate directions using validated CWE-787 prompt pairs?

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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score


def load_data(data_dir: Path, n_layers: int = 32) -> tuple:
    """Load the most recent SR and SCG data."""
    sr_files = sorted(data_dir.glob("sr_data_*.npz"))
    scg_files = sorted(data_dir.glob("scg_data_*.npz"))

    if not sr_files or not scg_files:
        raise FileNotFoundError("Data not found. Run 01_collect_activations.py first.")

    sr_file = sr_files[-1]
    scg_file = scg_files[-1]

    print(f"Loading SR data: {sr_file.name}")
    print(f"Loading SCG data: {scg_file.name}")

    sr_npz = np.load(sr_file)
    scg_npz = np.load(scg_file)

    # Convert to dict format
    sr_data = {}
    scg_data = {}
    for layer in range(n_layers):
        X_key = f"X_layer_{layer}"
        y_key = f"y_layer_{layer}"
        if X_key in sr_npz and y_key in sr_npz:
            sr_data[layer] = {'X': sr_npz[X_key], 'y': sr_npz[y_key]}
        if X_key in scg_npz and y_key in scg_npz:
            scg_data[layer] = {'X': scg_npz[X_key], 'y': scg_npz[y_key]}

    return sr_data, scg_data


def train_probe(X: np.ndarray, y: np.ndarray) -> dict:
    """Train a linear probe and return metrics."""
    if len(X) < 10:
        return {'accuracy': None, 'auc': None, 'cv_mean': None, 'direction': None}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_scaled, y)

    y_pred = clf.predict(X_scaled)
    y_prob = clf.predict_proba(X_scaled)[:, 1]

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0.5

    # Cross-validation
    cv_scores = cross_val_score(clf, X_scaled, y, cv=min(5, len(X) // 5))
    cv_mean = np.mean(cv_scores)

    # Extract direction (probe coefficients)
    direction = clf.coef_[0]
    direction_norm = direction / (np.linalg.norm(direction) + 1e-10)

    return {
        'accuracy': acc,
        'auc': auc,
        'cv_mean': cv_mean,
        'direction': direction,
        'direction_norm': direction_norm,
        'probe': clf,
        'scaler': scaler
    }


def compute_cosine_similarity(dir1: np.ndarray, dir2: np.ndarray) -> float:
    """Compute cosine similarity between two direction vectors."""
    norm1 = np.linalg.norm(dir1)
    norm2 = np.linalg.norm(dir2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return float(np.dot(dir1, dir2) / (norm1 * norm2))


def train_all_probes(sr_data: dict, scg_data: dict, n_layers: int) -> dict:
    """Train SR and SCG probes at all layers."""
    results = {
        'sr_probes': {},
        'scg_probes': {},
        'similarities': {}
    }

    print("\n" + "=" * 60)
    print("TRAINING PROBES")
    print("=" * 60)

    for layer in range(n_layers):
        # Train SR probe
        if layer in sr_data and len(sr_data[layer]['X']) > 0:
            sr_result = train_probe(sr_data[layer]['X'], sr_data[layer]['y'])
            results['sr_probes'][layer] = {
                'accuracy': sr_result['accuracy'],
                'auc': sr_result['auc'],
                'cv_mean': sr_result['cv_mean'],
                'direction': sr_result['direction'],
                'direction_norm': sr_result['direction_norm']
            }
        else:
            results['sr_probes'][layer] = {'accuracy': None}

        # Train SCG probe
        if layer in scg_data and len(scg_data[layer]['X']) > 0:
            scg_result = train_probe(scg_data[layer]['X'], scg_data[layer]['y'])
            results['scg_probes'][layer] = {
                'accuracy': scg_result['accuracy'],
                'auc': scg_result['auc'],
                'cv_mean': scg_result['cv_mean'],
                'direction': scg_result['direction'],
                'direction_norm': scg_result['direction_norm']
            }
        else:
            results['scg_probes'][layer] = {'accuracy': None}

        # Compute similarity
        sr_dir = results['sr_probes'][layer].get('direction_norm')
        scg_dir = results['scg_probes'][layer].get('direction_norm')

        if sr_dir is not None and scg_dir is not None:
            sim = compute_cosine_similarity(sr_dir, scg_dir)
            results['similarities'][layer] = sim
        else:
            results['similarities'][layer] = None

    # Print summary tables
    print("\nSR Probe Results:")
    print("| Layer | Accuracy | AUC   | CV Mean |")
    print("|-------|----------|-------|---------|")
    for layer in range(n_layers):
        r = results['sr_probes'][layer]
        if r['accuracy']:
            print(f"| {layer:5d} | {r['accuracy']*100:6.1f}% | {r['auc']:.3f} | {r['cv_mean']*100:5.1f}% |")

    print("\nSCG Probe Results:")
    print("| Layer | Accuracy | AUC   | CV Mean |")
    print("|-------|----------|-------|---------|")
    for layer in range(n_layers):
        r = results['scg_probes'][layer]
        if r['accuracy']:
            print(f"| {layer:5d} | {r['accuracy']*100:6.1f}% | {r['auc']:.3f} | {r['cv_mean']*100:5.1f}% |")

    print("\nDirection Similarities (SR vs SCG):")
    print("| Layer | Cosine Sim | Interpretation |")
    print("|-------|------------|----------------|")
    for layer in range(n_layers):
        sim = results['similarities'][layer]
        if sim is not None:
            if sim < 0.3:
                interp = "SEPARATE"
            elif sim < 0.5:
                interp = "somewhat separate"
            elif sim < 0.7:
                interp = "somewhat aligned"
            else:
                interp = "ALIGNED"
            print(f"| {layer:5d} | {sim:10.3f} | {interp} |")

    return results


def analyze_results(results: dict, n_layers: int) -> dict:
    """Analyze results and draw conclusions."""
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    # Compute average similarity
    valid_sims = [s for s in results['similarities'].values() if s is not None]
    avg_sim = np.mean(valid_sims) if valid_sims else 0

    # Find best probe layers
    sr_accs = [(l, r['accuracy']) for l, r in results['sr_probes'].items() if r['accuracy']]
    scg_accs = [(l, r['accuracy']) for l, r in results['scg_probes'].items() if r['accuracy']]

    best_sr = max(sr_accs, key=lambda x: x[1]) if sr_accs else (None, None)
    best_scg = max(scg_accs, key=lambda x: x[1]) if scg_accs else (None, None)

    # Count low similarity layers
    low_sim_layers = sum(1 for s in valid_sims if s < 0.5)

    analysis = {
        'avg_similarity': avg_sim,
        'best_sr_layer': best_sr[0],
        'best_sr_accuracy': best_sr[1],
        'best_scg_layer': best_scg[0],
        'best_scg_accuracy': best_scg[1],
        'n_low_similarity_layers': low_sim_layers,
        'n_total_layers': len(valid_sims)
    }

    print(f"\nAverage SR-SCG similarity: {avg_sim:.3f}")
    print(f"Best SR probe: Layer {best_sr[0]} ({best_sr[1]*100:.1f}%)")
    print(f"Best SCG probe: Layer {best_scg[0]} ({best_scg[1]*100:.1f}%)")
    print(f"Layers with low similarity (<0.5): {low_sim_layers}/{len(valid_sims)}")

    # Conclusion
    if avg_sim < 0.3:
        conclusion = "STRONG EVIDENCE for separate encoding (like harmfulness vs refusal)"
    elif avg_sim < 0.5:
        conclusion = "MODERATE EVIDENCE for separate encoding"
    elif avg_sim < 0.7:
        conclusion = "WEAK EVIDENCE - directions are somewhat aligned"
    else:
        conclusion = "NO EVIDENCE for separate encoding - SR and SCG use same direction"

    analysis['conclusion'] = conclusion
    print(f"\nConclusion: {conclusion}")

    return analysis


def plot_results(results: dict, analysis: dict, output_path: Path, n_layers: int):
    """Create visualization of probe results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    layers = list(range(n_layers))

    # Plot 1: SR Accuracy
    ax1 = axes[0, 0]
    sr_acc = [results['sr_probes'][l].get('accuracy', 0.5) or 0.5 for l in layers]
    ax1.plot(layers, sr_acc, 'b-o', markersize=4, label='SR Accuracy')
    ax1.axhline(y=0.5, color='gray', linestyle=':', label='Chance')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('SR Probe: Security Recognition\n(Is this a secure prompt?)')
    ax1.legend()
    ax1.set_ylim(0.4, 1.05)
    ax1.grid(True, alpha=0.3)

    # Plot 2: SCG Accuracy
    ax2 = axes[0, 1]
    scg_acc = [results['scg_probes'][l].get('accuracy', 0.5) or 0.5 for l in layers]
    ax2.plot(layers, scg_acc, 'r-o', markersize=4, label='SCG Accuracy')
    ax2.axhline(y=0.5, color='gray', linestyle=':', label='Chance')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('SCG Probe: Secure Code Generation\n(Will model output secure code?)')
    ax2.legend()
    ax2.set_ylim(0.4, 1.05)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Both overlaid
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

    # Plot 4: Similarity
    ax4 = axes[1, 1]
    sims = [results['similarities'].get(l, 0) or 0 for l in layers]
    ax4.bar(layers, sims, color='purple', alpha=0.7)
    ax4.axhline(y=0.5, color='orange', linestyle='--', label='Threshold (0.5)')
    ax4.axhline(y=analysis['avg_similarity'], color='red', linestyle='-',
                label=f"Mean ({analysis['avg_similarity']:.3f})")
    ax4.fill_between(layers, 0, 0.5, alpha=0.1, color='green')
    ax4.set_xlabel('Layer')
    ax4.set_ylabel('Cosine Similarity')
    ax4.set_title('SR vs SCG Direction Similarity\n(Low = Separate Encoding)')
    ax4.legend()
    ax4.set_ylim(-0.1, 1.0)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {output_path}")


def main():
    data_dir = Path(__file__).parent / "data"
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    n_layers = 32

    # Load data
    print("\n" + "=" * 70)
    print("SR vs SCG PROBE TRAINING: CWE-787 PROMPT PAIRS")
    print("=" * 70)

    sr_data, scg_data = load_data(data_dir, n_layers)

    print(f"\nLoaded data:")
    print(f"  SR samples: {len(sr_data[0]['X'])}")
    print(f"  SCG samples: {len(scg_data[0]['X'])}")

    # Train probes
    results = train_all_probes(sr_data, scg_data, n_layers)

    # Analyze
    analysis = analyze_results(results, n_layers)

    # Plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = results_dir / f"sr_scg_probes_{timestamp}.png"
    plot_results(results, analysis, plot_path, n_layers)

    # Save results (convert numpy arrays to lists for JSON)
    save_results = {
        'timestamp': timestamp,
        'sr_probe_results': [
            {'layer': l, 'accuracy': r['accuracy'], 'auc': r.get('auc'), 'cv_mean': r.get('cv_mean')}
            for l, r in results['sr_probes'].items()
        ],
        'scg_probe_results': [
            {'layer': l, 'accuracy': r['accuracy'], 'auc': r.get('auc'), 'cv_mean': r.get('cv_mean')}
            for l, r in results['scg_probes'].items()
        ],
        'similarities': [
            {'layer': l, 'cosine_similarity': s}
            for l, s in results['similarities'].items()
        ],
        'analysis': analysis
    }

    with open(results_dir / f"sr_scg_probes_{timestamp}.json", 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to: {results_dir}")
    print(f"\nNext step: python 03_differential_steering.py")

    return results, analysis


if __name__ == "__main__":
    results, analysis = main()
