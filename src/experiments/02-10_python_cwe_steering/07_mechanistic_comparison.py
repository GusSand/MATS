#!/usr/bin/env python3
"""
Experiment 10, Step 7: Mechanistic Comparison (Python vs C)

Analyzes whether Python and C CWEs share vulnerability representations:
  1. Cosine similarity matrix (6×6 from Step 2 — reload and display)
  2. PCA of all 6 directions — do they cluster by vulnerability type or language?
  3. Activation space analysis: project insecure/secure prompts from both languages
     onto shared PCA space
  4. Summary statistics and figures

This is the "interpretability" piece — what do the representations mean?

Reuses: Pre-computed directions from Step 2 and C experiments
"""
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.decomposition import PCA

# ─── Paths ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
EXPERIMENT_DIR = Path("/home/paperspace/MATS/src/experiments/02-05_cross_cwe_steering")
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"

# C directions
C_DIRECTION_DIR = EXPERIMENT_DIR / "cross_cwe_analysis" / "data"
C_DIRECTIONS = {
    "C-787": "direction_cwe787_L31_20260206_031901.npy",
    "C-119": "direction_cwe119_L31_20260206_031901.npy",
    "C-134": "direction_cwe134_L31_20260206_031901.npy",
}

LAYER = 31


# ─── Helpers ─────────────────────────────────────────────────────────────────

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Experiment 10, Step 7: Mechanistic Comparison (Python vs C)")
    print("=" * 70)

    # ─── Load all 6 directions ───────────────────────────────────────────
    print("\nLoading steering directions...")

    c_dirs = {}
    for label, fname in C_DIRECTIONS.items():
        c_dirs[label] = np.load(C_DIRECTION_DIR / fname).astype(np.float64)
        print(f"  {label}: norm={np.linalg.norm(c_dirs[label]):.4f}")

    py_dirs = {}
    for cwe_num in ["89", "78", "79"]:
        npy_files = sorted(DATA_DIR.glob(f"direction_cwe{cwe_num}_L{LAYER}_*.npy"))
        if not npy_files:
            print(f"  ERROR: No direction for CWE-{cwe_num}. Run Step 2 first.")
            return
        label = f"Py-{cwe_num}"
        py_dirs[label] = np.load(npy_files[-1]).astype(np.float64)
        print(f"  {label}: norm={np.linalg.norm(py_dirs[label]):.4f}")

    all_labels = list(c_dirs.keys()) + list(py_dirs.keys())
    all_vecs = [c_dirs[k] for k in c_dirs] + [py_dirs[k] for k in py_dirs]

    # ═══════════════════════════════════════════════════════════════════════
    # 1. COSINE SIMILARITY MATRIX
    # ═══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print("1. COSINE SIMILARITY MATRIX")
    print(f"{'='*70}")

    n = len(all_vecs)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = cosine_similarity(all_vecs[i], all_vecs[j])

    # Print
    header = all_labels
    print(f"\n{'':>10}", end="")
    for h in header:
        print(f"{h:>10}", end="")
    print()
    print("-" * (10 + 10 * n))
    for i, label in enumerate(header):
        print(f"{label:>10}", end="")
        for j in range(n):
            print(f"{sim_matrix[i,j]:>10.3f}", end="")
        print()

    # Key statistics
    c_c_sims = [sim_matrix[i, j] for i in range(3) for j in range(3) if i != j]
    py_py_sims = [sim_matrix[i, j] for i in range(3, 6) for j in range(3, 6) if i != j]
    cross_sims = [sim_matrix[i, j] for i in range(3) for j in range(3, 6)]
    cross_sims += [sim_matrix[i, j] for i in range(3, 6) for j in range(3)]

    print(f"\nWithin-language (C-C) avg sim: {np.mean(c_c_sims):.3f}")
    print(f"Within-language (Py-Py) avg sim: {np.mean(py_py_sims):.3f}")
    print(f"Cross-language avg sim: {np.mean(cross_sims):.3f}")

    # ═══════════════════════════════════════════════════════════════════════
    # 2. PCA OF STEERING DIRECTIONS
    # ═══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print("2. PCA OF 6 STEERING DIRECTIONS")
    print(f"{'='*70}")

    # Normalize vectors before PCA
    normed_vecs = np.array([v / np.linalg.norm(v) for v in all_vecs])

    pca = PCA(n_components=min(3, n))
    coords = pca.fit_transform(normed_vecs)

    print(f"\nExplained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Cumulative: {np.cumsum(pca.explained_variance_ratio_)}")

    print(f"\n{'Label':>10} {'PC1':>10} {'PC2':>10} {'PC3':>10}")
    print("-" * 42)
    for i, label in enumerate(all_labels):
        pc3_val = coords[i, 2] if coords.shape[1] > 2 else 0.0
        print(f"{label:>10} {coords[i,0]:>10.4f} {coords[i,1]:>10.4f} {pc3_val:>10.4f}")

    # Check if vectors cluster by vulnerability type or by language
    # Compute centroid distance for vulnerability clusters vs language clusters

    # Language clusters: C (0,1,2) vs Python (3,4,5)
    c_centroid = coords[:3].mean(axis=0)
    py_centroid = coords[3:].mean(axis=0)
    lang_distance = np.linalg.norm(c_centroid - py_centroid)

    print(f"\nLanguage cluster distance (C vs Python centroids): {lang_distance:.4f}")

    # Within-language spread
    c_spread = np.mean([np.linalg.norm(coords[i] - c_centroid) for i in range(3)])
    py_spread = np.mean([np.linalg.norm(coords[i] - py_centroid) for i in range(3, 6)])
    print(f"C spread: {c_spread:.4f}")
    print(f"Python spread: {py_spread:.4f}")

    if lang_distance < max(c_spread, py_spread):
        print("\n  FINDING: Directions cluster more by VULNERABILITY TYPE than by language")
        print("  → Shared vulnerability representations across languages!")
    else:
        print("\n  FINDING: Directions cluster more by LANGUAGE than by vulnerability type")
        print("  → Language-specific representations")

    # ═══════════════════════════════════════════════════════════════════════
    # 3. ACTIVATION SPACE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print("3. ACTIVATION SPACE ANALYSIS")
    print(f"{'='*70}")

    # Load Python activation data and project onto steering directions
    for cwe_num in ["89", "78", "79"]:
        act_files = sorted(DATA_DIR.glob(f"activations_cwe{cwe_num}_L{LAYER}_*.npz"))
        if not act_files:
            continue
        data = np.load(act_files[-1])
        X = data["X"].astype(np.float64)
        y = data["y"]
        n_pairs = len(y) // 2

        X_insecure = X[:n_pairs]
        X_secure = X[n_pairs:]

        # Project onto own direction
        own_dir = py_dirs[f"Py-{cwe_num}"]
        own_dir_normed = own_dir / np.linalg.norm(own_dir)

        insecure_proj = X_insecure @ own_dir_normed
        secure_proj = X_secure @ own_dir_normed

        sep = secure_proj.mean() - insecure_proj.mean()
        insecure_std = insecure_proj.std()
        secure_std = secure_proj.std()
        cohens_d = sep / np.sqrt((insecure_std**2 + secure_std**2) / 2)

        print(f"\n  Py-CWE-{cwe_num} projection onto own direction:")
        print(f"    Insecure mean: {insecure_proj.mean():.4f} ± {insecure_std:.4f}")
        print(f"    Secure mean:   {secure_proj.mean():.4f} ± {secure_std:.4f}")
        print(f"    Separation:    {sep:.4f}")
        print(f"    Cohen's d:     {cohens_d:.2f}")

        # Project onto other directions (cross-CWE)
        for other_label, other_dir in list(c_dirs.items()) + list(py_dirs.items()):
            if other_label == f"Py-{cwe_num}":
                continue
            other_normed = other_dir / np.linalg.norm(other_dir)
            ins_proj = X_insecure @ other_normed
            sec_proj = X_secure @ other_normed
            cross_sep = sec_proj.mean() - ins_proj.mean()
            cross_d = cross_sep / np.sqrt((ins_proj.std()**2 + sec_proj.std()**2) / 2)
            print(f"    → {other_label}: sep={cross_sep:.4f}, Cohen's d={cross_d:.2f}")

    # ═══════════════════════════════════════════════════════════════════════
    # 4. SUMMARY
    # ═══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print("MECHANISTIC COMPARISON SUMMARY")
    print(f"{'='*70}")

    # Are vulnerability representations language-specific or shared?
    print("\nKey Questions:")
    print(f"  1. Do same-CWE vectors correlate across languages?")
    # This would need same-vulnerability-type pairs (e.g., buffer overflow in C and Python)
    # Since Python CWEs are different from C CWEs, we check general patterns

    print(f"  2. Within-language similarity:")
    print(f"     C vectors (avg pairwise): {np.mean(c_c_sims):.3f}")
    print(f"     Py vectors (avg pairwise): {np.mean(py_py_sims):.3f}")

    print(f"  3. Cross-language similarity: {np.mean(cross_sims):.3f}")

    if np.mean(cross_sims) < 0.1:
        print(f"     → LOW cross-language similarity: vulnerability representations are language-specific")
    elif np.mean(cross_sims) > 0.3:
        print(f"     → HIGH cross-language similarity: shared vulnerability representations!")
    else:
        print(f"     → MODERATE cross-language similarity: partial overlap")

    # ─── Save ────────────────────────────────────────────────────────────
    output_data = {
        "timestamp": timestamp,
        "layer": LAYER,
        "similarity_matrix": sim_matrix.tolist(),
        "similarity_labels": all_labels,
        "within_c_avg": float(np.mean(c_c_sims)),
        "within_py_avg": float(np.mean(py_py_sims)),
        "cross_language_avg": float(np.mean(cross_sims)),
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "pca_coordinates": {label: coords[i].tolist() for i, label in enumerate(all_labels)},
        "language_cluster_distance": float(lang_distance),
        "c_spread": float(c_spread),
        "py_spread": float(py_spread),
        "direction_norms": {label: float(np.linalg.norm(v)) for label, v in zip(all_labels, all_vecs)},
    }

    results_path = RESULTS_DIR / f"mechanistic_comparison_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved: {results_path}")
    print("\nMechanistic comparison complete.")


if __name__ == "__main__":
    main()
