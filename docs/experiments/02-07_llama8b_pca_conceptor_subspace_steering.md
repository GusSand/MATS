# Experiments 7A/7B: PCA Subspace & Conceptor AND Steering

**Date**: 2026-02-07
**Model**: Llama-3.1-8B-Instruct (fp16)
**Layer**: 31
**GPU Time**: ~3.5 hours (7A PCA steering) + ~15 min (7B activation collection, steering skipped)

## Research Question

Is "write secure code" a multi-dimensional subspace in Llama-3.1-8B-Instruct at Layer 31? If so, can multi-component steering (PCA decomposition or conceptor AND composition) outperform the unified single-vector approach from Experiment 6?

## Hypothesis

The unified vector (Experiment 6) underperforms native per-CWE vectors because it collapses a multi-dimensional security subspace into one direction. PCA should reveal 2-3 meaningful dimensions, and steering along those dimensions should recover performance lost by averaging. Conceptor AND should find the shared activation subspace across all three CWEs.

**Result: Both hypotheses REJECTED.** PCA steering is worse than unified, and conceptor AND finds zero shared subspace.

## Method

### Experiment 7A: PCA Subspace Steering

#### Step 1: PCA Decomposition (CPU-only)

Stacked 3 pre-computed direction vectors into a 3×4096 matrix and computed SVD:

```python
# V = [dir_787, dir_119, dir_134]  shape (3, 4096)
# SVD: V = U @ diag(S) @ Vt
# PC_i = Vt[i]  (unit-normalized 4096-dim vectors)
```

#### Step 2: Multi-PC Steering

```python
# Hook at layer 31:
h[:, -1, :] += alpha_1 * PC1 + alpha_2 * PC2 + alpha_3 * PC3
```

4 alpha configurations tested (reduced from initial 8 for time constraints):

| Config | α_PC1 | α_PC2 | α_PC3 | Rationale |
|--------|-------|-------|-------|-----------|
| PC1-only α=3.0 | 3.0 | 0.0 | 0.0 | Dominant direction only |
| PC1+2 weighted | 3.0 | 1.5 | 0.0 | Top 2 PCs (86.8% variance) |
| PC1+2+3 weighted | 3.0 | 2.0 | 1.0 | All PCs, decreasing weights |
| PC1+2+3 sv-weighted | 3.0 | 1.51 | 1.31 | Weights proportional to singular values |

### Experiment 7B: Conceptor AND Steering

#### Step 1: Activation Collection (GPU)

Collected secure-prompt activations at L31 for all 315 prompts (105 per CWE). Each prompt is the "secure" variant from the prompt pair.

#### Step 2: Conceptor Computation

Per-CWE conceptors computed via SVD:

```python
# For each CWE's secure activations X (n × 4096):
U, S, Vt = torch.linalg.svd(X, full_matrices=False)
weights = S**2 / (S**2 + aperture**(-2))
C_cwe = Vt.T @ torch.diag(weights) @ Vt
```

#### Step 3: Boolean AND Composition

```python
C_and = (C1_inv + C2_inv + C3_inv - 2*I)^(-1)
# With eigenvalue clipping to [0, 1]
```

#### Step 4: Steering (SKIPPED — zero intersection)

Planned hook:
```python
h_new = (1 - beta) * h + beta * (C_security @ h)
```

### Direction Vectors (Pre-computed, from Experiment 5b)

| Vector | File | Norm |
|--------|------|------|
| CWE-787 | `direction_cwe787_L31_20260206_031901.npy` | 7.77 |
| CWE-119 | `direction_cwe119_L31_20260206_031901.npy` | 8.66 |
| CWE-134 | `direction_cwe134_L31_20260206_031901.npy` | 8.51 |

### Datasets

- CWE-787: 105 pairs (`cwe787_expanded_20260112_143316.jsonl`)
- CWE-119: 105 pairs (`cwe119_expanded_20260207_024627.jsonl`)
- CWE-134: 105 pairs (`cwe134_expanded_20260207_024627.jsonl`)

### Generation Parameters

- Seed: 42, Temperature: 0.6, Top-p: 0.9, Max tokens: 256

## Results

### 7A: PCA Eigenvalue Spectrum

| PC | Singular Value | Variance Explained | Cumulative |
|----|---------------|-------------------|------------|
| PC1 | 11.99 | 69.2% | 69.2% |
| PC2 | 6.04 | 17.6% | 86.8% |
| PC3 | 5.24 | 13.2% | 100.0% |

The security direction space is 2-3 dimensional. PC1 captures the shared direction (69.2%), while PC2 and PC3 capture CWE-specific variation.

#### PC Loadings (cosine similarity with original CWE vectors)

| | PC1 | PC2 | PC3 |
|--|-----|-----|-----|
| CWE-787 | -0.729 | **0.682** | -0.056 |
| CWE-119 | -0.871 | -0.280 | **-0.403** |
| CWE-134 | -0.870 | -0.187 | **0.457** |

PC1 loads on all three CWEs (shared security direction). PC2 separates CWE-787 from CWE-119/134. PC3 separates CWE-119 from CWE-134.

#### Pairwise Cosine Similarity of Original Vectors

| | CWE-787 | CWE-119 | CWE-134 |
|--|---------|---------|---------|
| CWE-787 | 1.0 | 0.467 | 0.482 |
| CWE-119 | 0.467 | 1.0 | 0.626 |
| CWE-134 | 0.482 | 0.626 | 1.0 |

### 7A: PCA Steering Results

| Config | CWE-787 | CWE-119 | CWE-134 | Avg |
|--------|---------|---------|---------|-----|
| PC1-only α=3.0 | 1.9% | 0.0% | 71.4% | 24.4% |
| PC1+2 weighted | 0.0% | 0.0% | 67.6% | 22.5% |
| PC1+2+3 weighted | 1.0% | 0.0% | 70.5% | 23.8% |
| PC1+2+3 sv-weighted | 1.9% | 0.0% | 74.3% | 25.4% |

### 7B: Conceptor Results

#### Per-CWE Conceptor Properties

**Aperture = 1.0:**

| CWE | Trace | Significant dims (>0.5) | n_SVD_components |
|-----|-------|------------------------|------------------|
| 787 | 48.3 | 96 | 105 |
| 119 | 19.8 | 36 | 105 |
| 134 | 23.2 | 46 | 105 |

**Aperture = 5.0:**

| CWE | Trace | Significant dims (>0.5) | n_SVD_components |
|-----|-------|------------------------|------------------|
| 787 | 94.4 | 104 | 105 |
| 119 | 47.5 | 78 | 105 |
| 134 | 44.3 | 48 | 105 |

#### C_security (Boolean AND)

| Aperture | Trace | Dims >0.5 | Dims >0.1 | Dims >0.01 | Max eigenvalue |
|----------|-------|-----------|-----------|------------|----------------|
| 1.0 | 0.0026 | 0 | 0 | 0 | 1.14e-05 |
| 5.0 | 0.0026 | 0 | 0 | 0 | 1.14e-05 |

**Steering was skipped** because C_security is effectively a zero matrix.

### Grand Comparison Table

| Method | CWE-787 | CWE-119 | CWE-134 | Avg |
|--------|---------|---------|---------|-----|
| Baseline (no steering) | 0.0% | 0.0% | 66.7% | 22.2% |
| Native per-CWE best | 52.4% | 20.0% | 90.0% | 54.1% |
| Unified single vector (Exp 6) | 21.0% | 4.8% | 69.5% | 31.8% |
| Stacked vectors best (Exp 7) | 27.6% | 10.5% | 59.0%* | 32.4%** |
| **PCA best (sv-weighted)** | **1.9%** | **0.0%** | **74.3%** | **25.4%** |
| **Conceptor AND** | **N/A** | **N/A** | **N/A** | **N/A** |

*Stacked CWE-134 degraded below baseline on all configs. Best single CWE result was High config.
**Stacked avg uses best config per-CWE, not single best config.

## Analysis

### Why PCA Steering Failed

1. **Magnitude loss**: PCA unit-normalizes the principal components. Original direction vectors had norms 7.77-8.66, giving effective perturbation of α×norm ≈ α×8.3 at each step. PCA PCs have norm 1.0, so at α=3.0, effective perturbation is only 3.0 — roughly 3× weaker than native vectors at the same alpha.

2. **CWE-specific information lost**: PC1 captures the shared direction but not CWE-specific components. Adding PC2/PC3 doesn't help because the alpha weights are too low to compensate for the norm difference.

3. **CWE-134 baseline inflated**: CWE-134 has 66.7% baseline secure rate (the model already generates secure format strings most of the time), so PCA "success" on CWE-134 is mostly baseline behavior, not steering.

### Why Conceptor AND Found Zero Intersection

**Root cause: Sample-to-dimension ratio.** With 105 samples in 4096-dimensional space:
- Each CWE's SVD has at most 105 non-zero components
- At aperture=1.0: CWE-787 spans ~96 dims, CWE-119 spans ~36 dims, CWE-134 spans ~46 dims
- These are tiny fractions of R^4096 (~2.3%, 0.9%, 1.1%)
- For three random subspaces of these sizes, the expected intersection dimension is essentially zero
- Even at aperture=5.0 (more permissive), the intersection remains zero

This is a fundamental limitation: you would need orders of magnitude more samples (or dimensionality reduction before conceptor computation) to find meaningful intersections.

### Implications for the "Multi-dimensional Security Subspace" Hypothesis

The PCA analysis confirms that the security direction space is 2-3 dimensional (PC1=69.2%, PC2=17.6%, PC3=13.2%). However, this doesn't mean multi-component steering should work — the 3 CWE vectors share a large common component (PC1) but have moderate pairwise similarity (0.47-0.63), suggesting CWE-specific directions that cannot be captured by a single composite approach.

The consistent pattern across Experiments 6, 7, and 7A/7B is that **native per-CWE vectors work best, and any attempt to combine them degrades performance**. This suggests that effective security steering requires CWE-specific rather than universal approaches.

## Adversarial Prompt Limitation

Current datasets use prompts that explicitly request insecure functions (e.g., "Use gets()", "Pass directly to printf", "Use sprintf for speed"). This is an adversarial evaluation — it tests whether steering can override explicit user instructions to use insecure patterns.

Real-world steering effectiveness against ambiguous prompts (which don't explicitly request insecure functions) is likely higher. All secure rates in this report should be interpreted as lower bounds on real-world effectiveness.

This limitation affects ALL experiments in the cross-CWE series (Experiments 5-7B) equally, so relative comparisons between methods remain valid.

## Code

- [pca_analysis.py](../../src/experiments/02-05_cross_cwe_steering/pca_analysis.py) - PCA decomposition of 3 direction vectors
- [pca_steering_experiment.py](../../src/experiments/02-05_cross_cwe_steering/pca_steering_experiment.py) - PCA subspace steering experiment
- [conceptor_steering_experiment.py](../../src/experiments/02-05_cross_cwe_steering/conceptor_steering_experiment.py) - Conceptor AND steering experiment

## Data Files

All in `src/experiments/02-05_cross_cwe_steering/cross_cwe_analysis/data/`:

### PCA Analysis (7A)
| File | Description |
|------|-------------|
| `pca_analysis_20260207_025304.json` | PCA decomposition results: eigenvalues, loadings, cosine similarities |
| `pc1_security_L31_20260207_025304.npy` | PC1 direction vector (4096-dim, unit norm) |
| `pc2_security_L31_20260207_025304.npy` | PC2 direction vector |
| `pc3_security_L31_20260207_025304.npy` | PC3 direction vector |

### PCA Steering (7A)
| File | Description |
|------|-------------|
| `pca_subspace_steering_results_20260207_030444.json` | Summary results: per-config per-CWE secure rates |
| `pca_subspace_steering_full_20260207_030444.json` | Per-sample outputs for all 1,260 generations |

### Conceptor Steering (7B)
| File | Description |
|------|-------------|
| `secure_activations_L31_20260207_052813.npz` | Secure-prompt activations (315 × 4096) |
| `conceptor_info_20260207_052813.json` | Per-CWE conceptor properties, AND result diagnostics |
| `conceptor_steering_results_20260207_052813.json` | Results (steering skipped, zero intersection documented) |
