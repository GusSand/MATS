#!/usr/bin/env python3
"""
Vector Correlation Analysis: CWE-787 vs CWE-119 vs CWE-134

Computes L31 mean-difference steering directions for each CWE type
on Llama-3.1-8B-Instruct and computes pairwise cosine similarity.

Reuses: shared/model_loader.py, shared/activation_collector.py
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Add shared modules to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / "shared"))

from model_loader import ModelLoader
from activation_collector import ActivationCollector


def load_dataset(path):
    dataset = []
    with open(path) as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def compute_direction(X, y):
    """Compute mean-difference direction: secure_mean - vulnerable_mean."""
    secure_mean = X[y == 1].mean(axis=0)
    vulnerable_mean = X[y == 0].mean(axis=0)
    return (secure_mean - vulnerable_mean).astype(np.float32)


def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def collect_l31_activations(collector, dataset, layer=31):
    """Collect activations at a single layer for a dataset. Returns X, y."""
    n_pairs = len(dataset)
    hidden_size = collector.hidden_size

    X = np.zeros((2 * n_pairs, hidden_size), dtype=np.float16)
    y = np.array([0] * n_pairs + [1] * n_pairs)

    collector.register_all_layer_hooks()

    print(f"  Collecting {n_pairs} vulnerable prompts...")
    for i, pair in enumerate(dataset):
        inputs = collector.tokenizer(pair['vulnerable'], return_tensors="pt").to(collector.device)
        with torch.no_grad():
            _ = collector.model(**inputs)
        X[i] = collector.activations[layer].numpy().astype(np.float16)

    print(f"  Collecting {n_pairs} secure prompts...")
    for i, pair in enumerate(dataset):
        idx = n_pairs + i
        inputs = collector.tokenizer(pair['secure'], return_tensors="pt").to(collector.device)
        with torch.no_grad():
            _ = collector.model(**inputs)
        X[idx] = collector.activations[layer].numpy().astype(np.float16)

    collector.clear_hooks()
    return X, y


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    layer = 31

    print("=" * 60)
    print("Vector Correlation Analysis: CWE-787 vs CWE-119 vs CWE-134")
    print(f"Layer: {layer}")
    print("=" * 60)

    # --- Load CWE-787 direction (already computed) ---
    print("\n[1/4] Loading CWE-787 L31 direction from existing .npz...")
    cwe787_npz = np.load(SCRIPT_DIR.parent / "01-12_cwe787_cross_domain_steering/data/directions_20260112_153536.npz")
    dir_787 = cwe787_npz['direction_layer_31'].astype(np.float32)
    print(f"  CWE-787 direction: shape={dir_787.shape}, norm={np.linalg.norm(dir_787):.4f}")

    # --- Load model once for CWE-119 and CWE-134 ---
    print("\n[2/4] Loading Llama-3.1-8B-Instruct...")
    loader = ModelLoader("meta-llama/Meta-Llama-3.1-8B-Instruct", quantization=None)
    collector = ActivationCollector(loader)

    # --- CWE-119 ---
    print("\n[3/4] Computing CWE-119 L31 direction...")
    ds_119 = load_dataset(SCRIPT_DIR / "datasets/cwe119/data/cwe119_expanded_20260205_151207.jsonl")
    print(f"  Dataset: {len(ds_119)} pairs")
    X_119, y_119 = collect_l31_activations(collector, ds_119, layer=layer)
    dir_119 = compute_direction(X_119, y_119)
    print(f"  CWE-119 direction: norm={np.linalg.norm(dir_119):.4f}")

    # --- CWE-134 ---
    print("\n[4/4] Computing CWE-134 L31 direction...")
    ds_134 = load_dataset(SCRIPT_DIR / "datasets/cwe134/data/cwe134_expanded_20260205_151207.jsonl")
    print(f"  Dataset: {len(ds_134)} pairs")
    X_134, y_134 = collect_l31_activations(collector, ds_134, layer=layer)
    dir_134 = compute_direction(X_134, y_134)
    print(f"  CWE-134 direction: norm={np.linalg.norm(dir_134):.4f}")

    loader.unload()

    # --- Compute cosine similarity matrix ---
    print("\n" + "=" * 60)
    print("COSINE SIMILARITY MATRIX (L31 Directions)")
    print("=" * 60)

    directions = {
        'CWE-787': dir_787,
        'CWE-119': dir_119,
        'CWE-134': dir_134,
    }

    cwes = list(directions.keys())
    matrix = np.zeros((3, 3))
    for i, cwe_i in enumerate(cwes):
        for j, cwe_j in enumerate(cwes):
            matrix[i, j] = cosine_similarity(directions[cwe_i], directions[cwe_j])

    # Print matrix
    header = f"{'':>10}" + "".join(f"{c:>12}" for c in cwes)
    print(header)
    print("-" * len(header))
    for i, cwe_i in enumerate(cwes):
        row = f"{cwe_i:>10}" + "".join(f"{matrix[i,j]:>12.4f}" for j in range(3))
        print(row)

    # --- Save direction vectors ---
    output_dir = SCRIPT_DIR / "cross_cwe_analysis" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f"direction_cwe787_L31_{timestamp}.npy", dir_787)
    np.save(output_dir / f"direction_cwe119_L31_{timestamp}.npy", dir_119)
    np.save(output_dir / f"direction_cwe134_L31_{timestamp}.npy", dir_134)
    print(f"\nDirection vectors saved to {output_dir}/")

    # --- Save results JSON ---
    results = {
        'timestamp': timestamp,
        'layer': layer,
        'model': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'cwe787_source': 'directions_20260112_153536.npz (full dataset, 105 pairs)',
        'cwe119_source': f'cwe119_expanded_20260205_151207.jsonl ({len(ds_119)} pairs)',
        'cwe134_source': f'cwe134_expanded_20260205_151207.jsonl ({len(ds_134)} pairs)',
        'direction_norms': {
            'CWE-787': float(np.linalg.norm(dir_787)),
            'CWE-119': float(np.linalg.norm(dir_119)),
            'CWE-134': float(np.linalg.norm(dir_134)),
        },
        'cosine_similarity_matrix': {
            'labels': cwes,
            'matrix': matrix.tolist(),
        },
        'pairwise': {
            'CWE-787_vs_CWE-119': float(matrix[0, 1]),
            'CWE-787_vs_CWE-134': float(matrix[0, 2]),
            'CWE-119_vs_CWE-134': float(matrix[1, 2]),
        },
    }

    results_path = output_dir / f"vector_correlation_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")


if __name__ == "__main__":
    main()
