# Experiment 5b: Cross-CWE Parallel Analysis (Vector Correlation, Failure Analysis, Statistical Tables)

**Date**: 2026-02-06
**Model**: Llama-3.1-8B-Instruct (all CWEs), Mistral-7B, Llama-70B (CWE-787 cross-model)
**Datasets**: CWE-787 (105 pairs), CWE-119 (105 pairs), CWE-134 (105 pairs)

## Overview

Follow-up analyses on completed steering experiments. Three CPU-oriented tasks run while GPU was occupied with Qwen-14B LOBO folds:

1. **Vector Correlation**: Cosine similarity between L31 steering directions across CWE types
2. **CWE-119 Failure Analysis**: Manual categorization of α=4.0 steered outputs
3. **Statistical Tables**: Bootstrap 95% CIs for all main results across LOBO folds

## Research Questions

1. How similar are the steering directions for different CWE types? (Informs cross-CWE transfer feasibility)
2. Why does CWE-119 steering achieve only 20%? What do the failures look like?
3. What are the confidence intervals on our headline numbers?

## Task 1: Vector Correlation Analysis

### Method

- **CWE-787 direction**: Loaded from pre-computed `directions_20260112_153536.npz` (full dataset, 105 pairs)
- **CWE-119 and CWE-134 directions**: Computed fresh — loaded Llama-8B, collected L31 activations for all 210 prompts per CWE, computed mean-difference (secure_mean - vulnerable_mean)
- **Metric**: Cosine similarity between each pair of 4096-dim direction vectors

### Results

**Direction Norms (L31):**

| CWE | Direction Norm |
|-----|---------------|
| CWE-787 | 7.77 |
| CWE-119 | 8.66 |
| CWE-134 | 8.51 |

**Cosine Similarity Matrix:**

|           | CWE-787 | CWE-119 | CWE-134 |
|-----------|---------|---------|---------|
| **CWE-787** | 1.0000  | 0.4669  | 0.4819  |
| **CWE-119** | 0.4669  | 1.0000  | 0.6263  |
| **CWE-134** | 0.4819  | 0.6263  | 1.0000  |

### Observations (No Interpretation)

- CWE-119 and CWE-134 are most similar (0.63 cosine similarity)
- CWE-787 has moderate similarity to both (~0.47-0.48)
- No pair is near-identical — these are partially overlapping but distinct directions
- All direction norms are comparable (~7.8-8.7)

## Task 2: CWE-119 Failure Analysis (α=4.0)

### Method

- Loaded all 105 steered outputs from the 7 LOBO folds at α=4.0
- Categorized each output using regex-based heuristics into 6 categories
- Breakdown computed by vulnerability type (gets vs strcpy)

### Results

| Category | Count | % |
|---|---|---|
| Still uses gets/strcpy | 57 | 54.3% |
| Other (code, no recognized pattern) | 19 | 18.1% |
| Correct (fgets/strncpy) | 13 | 12.4% |
| Manual bounds check instead | 8 | 7.6% |
| Attempted but malformed syntax | 5 | 4.8% |
| Degenerate/garbage | 3 | 2.9% |

**By vulnerability type:**

| Category | gets (n=45) | strcpy (n=60) |
|---|---|---|
| Correct | 10 (22.2%) | 3 (5.0%) |
| Malformed | 4 (8.9%) | 1 (1.7%) |
| Bounds check | 2 (4.4%) | 6 (10.0%) |
| Still insecure | 27 (60.0%) | 30 (50.0%) |
| Degenerate | 0 (0.0%) | 3 (5.0%) |
| Other | 2 (4.4%) | 17 (28.3%) |

### Observations (No Interpretation)

- `gets` → `fgets` steering works better (22.2% correct) than `strcpy` → `strncpy` (5.0%)
- 5 samples attempted fgets but used `sizeof(pointer)` instead of buffer size — a common C bug
- 18.1% "Other" category = code structure present but no recognized secure/insecure API call matched
- Only 2.9% degenerate — model still produces coherent C code at α=4.0

## Task 3: Statistical Tables with Bootstrap CIs

### Method

- Loaded per-fold LOBO results for all completed experiments
- Computed per-fold strict secure rates at each alpha
- Bootstrapped (10,000 resamples, seed=42) across folds to get 95% CIs

### Results

| Experiment | Folds | Baseline | Steered | 95% CI (steered) | Improvement |
|---|---|---|---|---|---|
| Llama-8B CWE-787 | 7 | 0.0% | 52.4% | [39.0%, 65.7%] | +52.4pp |
| Mistral-7B CWE-787 | 7 | 26.7% | 92.4% | [84.8%, 100.0%] | +65.7pp |
| Llama-70B CWE-787 | 7 | 1.9% | 52.4% | [29.5%, 73.3%] | +50.5pp |
| Llama-8B CWE-119 | 7 | 0.0% | 20.0% | [10.5%, 30.5%] | +20.0pp |
| Llama-8B CWE-134 (pilot) | 2 | 66.7% | 90.0% | [86.7%, 93.3%] | +23.3pp |

**Per-fold steered rates:**

| Experiment | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 |
|---|---|---|---|---|---|---|---|
| Llama-8B CWE-787 (α=3.5) | 66.7% | 80.0% | 40.0% | 20.0% | 46.7% | 60.0% | 53.3% |
| Mistral-7B CWE-787 (α=3.5) | 100% | 100% | 73.3% | 73.3% | 100% | 100% | 100% |
| Llama-70B CWE-787 (α=4.0) | 40.0% | 73.3% | 53.3% | 0.0% | 86.7% | 26.7% | 86.7% |
| Llama-8B CWE-119 (α=4.0) | 40.0% | 40.0% | 13.3% | 20.0% | 13.3% | 6.7% | 6.7% |
| Llama-8B CWE-134 (α=1.5) | 86.7% | 93.3% | — | — | — | — | — |

### Observations (No Interpretation)

- Llama-70B has the widest CI [29.5%, 73.3%] — fold 4 scored 0%, fold 5 scored 87%
- CWE-134 CI is artificially tight (only 2 pilot folds)
- CWE-119 CI lower bound is 10.5% — the effect is real but weak
- Mistral-7B is the most consistent performer (lowest CI width: 15.2pp)

## Code / Files Generated

- [vector_correlation_analysis.py](../../src/experiments/02-05_cross_cwe_steering/vector_correlation_analysis.py) - Computes L31 direction vectors for CWE-119/134 and cosine similarity matrix
- [cwe119_failure_analysis.py](../../src/experiments/02-05_cross_cwe_steering/cwe119_failure_analysis.py) - Categorizes CWE-119 steered outputs at α=4.0
- [statistical_tables.py](../../src/experiments/02-05_cross_cwe_steering/statistical_tables.py) - Bootstrap 95% CIs across LOBO folds

## Data Generated

All output data in `src/experiments/02-05_cross_cwe_steering/cross_cwe_analysis/data/`:

| File | Description |
|---|---|
| `direction_cwe787_L31_20260206_031901.npy` | CWE-787 L31 direction vector (4096-dim, float32) |
| `direction_cwe119_L31_20260206_031901.npy` | CWE-119 L31 direction vector (4096-dim, float32) |
| `direction_cwe134_L31_20260206_031901.npy` | CWE-134 L31 direction vector (4096-dim, float32) |
| `vector_correlation_20260206_031901.json` | Cosine similarity matrix and metadata |
| `cwe119_failure_analysis.json` | Full categorization of 105 samples at α=4.0 |
| `statistical_tables.json` | Bootstrap CIs for all experiments |
