"""
Layer Sweep: Probe-Based Optimal Layer Selection

Trains logistic regression at each layer to find best steering candidates.
Reports 5-fold CV accuracy, mean-diff direction norm, and cluster separation.
"""

import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist


def probe_accuracy_at_layer(X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> float:
    """
    Train logistic regression probe and return mean CV accuracy.

    Args:
        X: activations (n_samples, hidden_size)
        y: labels (n_samples,) - 0=vulnerable, 1=secure
        n_folds: number of CV folds

    Returns:
        Mean CV accuracy
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.astype(np.float32))

    clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    scores = cross_val_score(clf, X_scaled, y, cv=n_folds, scoring='accuracy')
    return float(scores.mean())


def direction_norm_at_layer(X: np.ndarray, y: np.ndarray) -> float:
    """Compute mean-difference direction norm."""
    secure_mean = X[y == 1].mean(axis=0)
    vulnerable_mean = X[y == 0].mean(axis=0)
    direction = secure_mean - vulnerable_mean
    return float(np.linalg.norm(direction))


def cluster_separation_at_layer(X: np.ndarray, y: np.ndarray) -> float:
    """
    Compute cluster separation (inter-cluster / intra-cluster distance ratio).
    Higher = better separation between secure and vulnerable.
    """
    X_sec = X[y == 1].astype(np.float32)
    X_vul = X[y == 0].astype(np.float32)

    sec_mean = X_sec.mean(axis=0, keepdims=True)
    vul_mean = X_vul.mean(axis=0, keepdims=True)

    inter_dist = float(np.linalg.norm(sec_mean - vul_mean))

    # Average intra-cluster distance (use subset for speed)
    n_sample = min(50, len(X_sec), len(X_vul))
    rng = np.random.RandomState(42)
    sec_sample = X_sec[rng.choice(len(X_sec), n_sample, replace=False)]
    vul_sample = X_vul[rng.choice(len(X_vul), n_sample, replace=False)]

    intra_sec = float(cdist(sec_sample, sec_sample).mean())
    intra_vul = float(cdist(vul_sample, vul_sample).mean())
    intra_avg = (intra_sec + intra_vul) / 2

    if intra_avg == 0:
        return 0.0
    return inter_dist / intra_avg


def run_layer_sweep(npz_path: Path, layers: list = None) -> list:
    """
    Run probe sweep across all layers.

    Args:
        npz_path: Path to activations NPZ file
        layers: Optional list of layers to sweep (default: all available)

    Returns:
        List of dicts with per-layer results, sorted by probe_accuracy descending
    """
    data = np.load(npz_path)

    if layers is None:
        layers = sorted([int(k.split('_')[-1]) for k in data.files if k.startswith('X_layer_')])

    results = []
    for layer in layers:
        X = data[f'X_layer_{layer}'].astype(np.float32)
        y = data[f'y_layer_{layer}']

        acc = probe_accuracy_at_layer(X, y)
        norm = direction_norm_at_layer(X, y)
        sep = cluster_separation_at_layer(X, y)

        results.append({
            'layer': layer,
            'probe_accuracy': acc,
            'direction_norm': norm,
            'cluster_separation': sep,
        })

        print(f"  Layer {layer:3d}: acc={acc:.4f}, norm={norm:.2f}, sep={sep:.4f}")

    results.sort(key=lambda r: r['probe_accuracy'], reverse=True)
    return results


def select_top_layers(results: list, n: int = 5) -> list:
    """Select top-N layers by probe accuracy."""
    return [r['layer'] for r in results[:n]]


def save_sweep_results(results: list, output_dir: Path) -> Path:
    """Save layer sweep results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "layer_sweep_results.json"

    output = {
        'results': results,
        'top5_layers': select_top_layers(results, 5),
        'best_layer': results[0]['layer'],
        'best_accuracy': results[0]['probe_accuracy'],
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved layer sweep results: {output_path}")
    return output_path
