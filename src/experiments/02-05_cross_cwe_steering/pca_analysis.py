#!/usr/bin/env python3
"""
Experiment 7A Step 1: PCA Analysis of Per-CWE Steering Vectors

Decomposes the 3 per-CWE mean-difference steering vectors via SVD to determine
the dimensionality of the "security subspace" at Layer 31 of Llama-3.1-8B-Instruct.

CPU only — no GPU needed.
"""

import numpy as np
from pathlib import Path
import json
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "cross_cwe_analysis" / "data"


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("Experiment 7A Step 1: PCA Analysis of Steering Vectors")
    print("=" * 60)

    # --- Load the 3 per-CWE L31 direction vectors ---
    vec_787 = np.load(DATA_DIR / "direction_cwe787_L31_20260206_031901.npy")
    vec_119 = np.load(DATA_DIR / "direction_cwe119_L31_20260206_031901.npy")
    vec_134 = np.load(DATA_DIR / "direction_cwe134_L31_20260206_031901.npy")

    print(f"\nLoaded direction vectors:")
    print(f"  CWE-787: shape={vec_787.shape}, norm={np.linalg.norm(vec_787):.4f}")
    print(f"  CWE-119: shape={vec_119.shape}, norm={np.linalg.norm(vec_119):.4f}")
    print(f"  CWE-134: shape={vec_134.shape}, norm={np.linalg.norm(vec_134):.4f}")

    # --- Stack and compute SVD ---
    V = np.stack([vec_787.astype(np.float64),
                  vec_119.astype(np.float64),
                  vec_134.astype(np.float64)])  # shape: (3, 4096)

    print(f"\nStacked matrix V: shape={V.shape}")

    U, S, Vt = np.linalg.svd(V, full_matrices=False)
    # U: (3, 3), S: (3,), Vt: (3, 4096)

    print(f"\nSVD results:")
    print(f"  U: {U.shape}, S: {S.shape}, Vt: {Vt.shape}")

    # --- Eigenvalue spectrum ---
    total_var = np.sum(S**2)
    cumulative = 0.0
    print(f"\n{'='*60}")
    print(f"EIGENVALUE SPECTRUM")
    print(f"{'='*60}")
    print(f"{'PC':>4} {'Singular Val':>14} {'Var Explained':>14} {'Cumulative':>12}")
    print(f"{'-'*4:>4} {'-'*14:>14} {'-'*14:>14} {'-'*12:>12}")

    variance_explained = []
    for i, s in enumerate(S):
        var_pct = s**2 / total_var * 100
        cumulative += var_pct
        variance_explained.append(var_pct)
        print(f"PC{i+1:>2} {s:>14.4f} {var_pct:>13.1f}% {cumulative:>11.1f}%")

    # --- Cosine similarity of each PC with original vectors ---
    print(f"\n{'='*60}")
    print(f"COSINE SIMILARITY: Original Vectors vs Principal Components")
    print(f"{'='*60}")

    vecs = [('CWE-787', vec_787), ('CWE-119', vec_119), ('CWE-134', vec_134)]
    pc_names = ['PC1', 'PC2', 'PC3']

    header = f"{'':>10}" + "".join(f"{pc:>10}" for pc in pc_names)
    print(header)
    print("-" * len(header))

    cos_sim_matrix = {}
    for name, vec in vecs:
        cos_sim_matrix[name] = {}
        row = f"{name:>10}"
        for j in range(3):
            pc = Vt[j]
            cos = float(np.dot(vec.astype(np.float64), pc) / (np.linalg.norm(vec) * np.linalg.norm(pc)))
            cos_sim_matrix[name][pc_names[j]] = cos
            row += f"{cos:>10.4f}"
        print(row)

    # --- Pairwise cosine similarity of original vectors (for reference) ---
    print(f"\n{'='*60}")
    print(f"PAIRWISE COSINE SIMILARITY (Original Vectors)")
    print(f"{'='*60}")

    header = f"{'':>10}" + "".join(f"{n:>10}" for n, _ in vecs)
    print(header)
    print("-" * len(header))
    pairwise = {}
    for i, (name_i, vec_i) in enumerate(vecs):
        row = f"{name_i:>10}"
        for j, (name_j, vec_j) in enumerate(vecs):
            cos = float(np.dot(vec_i.astype(np.float64), vec_j.astype(np.float64)) /
                        (np.linalg.norm(vec_i) * np.linalg.norm(vec_j)))
            row += f"{cos:>10.4f}"
            if i < j:
                pairwise[f"{name_i}_vs_{name_j}"] = cos
        print(row)

    # --- Interpretation ---
    print(f"\n{'='*60}")
    print(f"INTERPRETATION")
    print(f"{'='*60}")

    if variance_explained[0] > 70:
        print(f"PC1 explains {variance_explained[0]:.1f}% — a single direction may suffice.")
        print(f"The security subspace is approximately 1-dimensional.")
        recommended_dim = 1
    elif sum(variance_explained[:2]) > 90:
        print(f"PC1+PC2 explain {sum(variance_explained[:2]):.1f}% — a 2D subspace captures most variance.")
        print(f"The security subspace is approximately 2-dimensional.")
        recommended_dim = 2
    else:
        print(f"All 3 PCs needed ({sum(variance_explained[:3]):.1f}% total) — full 3D subspace required.")
        print(f"The security subspace is 3-dimensional (each CWE adds unique information).")
        recommended_dim = 3

    # --- Compute sv-weighted alphas for the steering experiment ---
    sv_weights = S / S[0]
    print(f"\nSingular-value weights for sv-weighted steering (alpha=3.0 base):")
    for i in range(3):
        print(f"  PC{i+1}: weight={sv_weights[i]:.4f}, alpha={3.0 * sv_weights[i]:.4f}")

    # --- Save PCs ---
    pc1 = Vt[0].astype(np.float32)
    pc2 = Vt[1].astype(np.float32)
    pc3 = Vt[2].astype(np.float32)

    np.save(DATA_DIR / f"pc1_security_L31_{timestamp}.npy", pc1)
    np.save(DATA_DIR / f"pc2_security_L31_{timestamp}.npy", pc2)
    np.save(DATA_DIR / f"pc3_security_L31_{timestamp}.npy", pc3)
    print(f"\nPrincipal components saved to {DATA_DIR}/")

    # --- Save results JSON ---
    results = {
        'timestamp': timestamp,
        'experiment': '7A_pca_analysis',
        'model': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'layer': 31,
        'source_vectors': {
            'cwe787': 'direction_cwe787_L31_20260206_031901.npy',
            'cwe119': 'direction_cwe119_L31_20260206_031901.npy',
            'cwe134': 'direction_cwe134_L31_20260206_031901.npy',
        },
        'singular_values': S.tolist(),
        'variance_explained_pct': variance_explained,
        'cumulative_variance_pct': [sum(variance_explained[:i+1]) for i in range(3)],
        'recommended_dimensionality': recommended_dim,
        'sv_weights': sv_weights.tolist(),
        'cosine_similarity_pcs_vs_originals': cos_sim_matrix,
        'pairwise_cosine_similarity': pairwise,
        'U_matrix': U.tolist(),
        'pc_files': {
            'pc1': f'pc1_security_L31_{timestamp}.npy',
            'pc2': f'pc2_security_L31_{timestamp}.npy',
            'pc3': f'pc3_security_L31_{timestamp}.npy',
        },
    }

    results_path = DATA_DIR / f"pca_analysis_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")


if __name__ == "__main__":
    main()
