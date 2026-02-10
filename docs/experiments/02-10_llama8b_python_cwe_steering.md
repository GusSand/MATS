# Experiment 10: Python CWE Steering & Cross-Language Validation

**Date**: 2026-02-10
**Model**: meta-llama/Meta-Llama-3.1-8B-Instruct
**GPU**: A100-80GB
**Dataset**: 3 Python CWEs (CWE-89, CWE-78, CWE-79) — 105 prompt pairs each + 21 neutral prompts

## Research Question

Do mean-difference activation steering vectors generalize from C to Python vulnerabilities? Are vulnerability representations language-specific or vulnerability-specific?

## 7-Step Pipeline

### Step 1: Baseline (01_baseline.py)

Generated 3,150 insecure completions (105 prompts × 10 seeds × 3 CWEs) + 210 neutral completions.

**Re-scored results** (after scorer fixes):

| CWE | n | Secure | Insecure | Other | Secure% |
|-----|---|--------|----------|-------|---------|
| CWE-89 | 1050 | 586 | 450 | 14 | 55.8% |
| CWE-78 | 1050 | 150 | 898 | 2 | 14.3% |
| CWE-79 | 1050 | 0 | 1023 | 27 | 0.0% |

Neutral prompts: CWE-89 100%, CWE-78 75.7%, CWE-79 0%

### Step 2: Extract Vectors (02_extract_vectors.py)

Mean-difference vectors at Layer 31:

| Direction | Norm |
|-----------|------|
| Py-CWE-89 | 2.726 |
| Py-CWE-78 | 5.279 |
| Py-CWE-79 | 7.018 |

Cross-language cosine similarity (C vs Python): ~0.007 average (near zero).

### Step 3: LOBO Validation (03_lobo_validation.py)

Leave-One-Base-Out cross-validation with α ∈ {0, 1, 2, 3, 4, 5}, 10 seeds per prompt:

| CWE | Baseline | α=1.0 | α=2.0 | α=3.0 | α=4.0 | α=5.0 (best) | Improvement |
|-----|----------|-------|-------|-------|-------|---------------|-------------|
| CWE-89 | 57.0% | 61.0% | 64.5% | 68.1% | 68.9% | **70.3%** | +13.3pp |
| CWE-78 | 14.3% | 16.5% | 18.8% | 20.2% | 21.7% | **22.0%** | +7.7pp |
| CWE-79 | 0.2% | 0.5% | 1.7% | 10.7% | 19.3% | **30.5%** | +30.3pp |

All CWEs show monotonic improvement with increasing α; best α = 5.0 for all three.

### Step 4: Transfer Matrix (04_transfer_matrix.py) — KEY PAPER FIGURE

6×6 matrix: apply each steering vector to each prompt set (15 prompts × 10 seeds = 150 per cell).

| vec\prompts | C-787 | C-119 | C-134 | Py-89 | Py-78 | Py-79 |
|-------------|-------|-------|-------|-------|-------|-------|
| **C-787** (α=3.5) | **78.7%** | 4.7% | 0.0% | 85.3% | 1.3% | 0.0% |
| **C-119** (α=4.0) | 0.7% | **95.3%** | 0.0% | 10.7% | 0.0% | 0.0% |
| **C-134** (α=1.5) | 0.0% | 0.7% | **0.0%** | 69.3% | 4.0% | 0.0% |
| **Py-89** (α=5.0) | 0.0% | 0.0% | 0.0% | **82.7%** | 8.7% | 0.0% |
| **Py-78** (α=5.0) | 0.0% | 34.7% | 0.0% | 67.3% | **25.3%** | 0.0% |
| **Py-79** (α=5.0) | 0.0% | 0.0% | 0.0% | 93.3% | 13.3% | **17.3%** |

**Summary**:
- Diagonal average: 49.9%
- Off-diagonal average: 13.1%
- **Diagonal / Off-diagonal ratio: 3.8x** — clear vulnerability-specificity
- C diagonal: 58.0% (C-787: 78.7%, C-119: 95.3%, C-134: 0.0%)
- Python diagonal: 41.8% (Py-89: 82.7%, Py-78: 25.3%, Py-79: 17.3%)
- C→Python transfer: 19.0%
- Python→C transfer: 3.9%

**Notable**: C-134 diagonal is 0% (α=1.5 too weak), Py-89 column is uniformly high (SQL is easily steerable). Cross-language transfer is asymmetric (C→Py > Py→C).

### Step 5: Probe Routing (05_probe_routing.py)

3-class LogisticRegression on L31 activations:

| Metric | Value |
|--------|-------|
| Training accuracy | 100.0% |
| 5-fold CV accuracy | 100.0% ± 0.0% |
| Routing accuracy (21 neutral) | 21/21 (100.0%) |
| Min confidence | 0.999 |

### Step 6: E2E Pipeline (06_e2e_pipeline.py)

Full probe → route → steer pipeline on 21 neutral prompts × 10 seeds:

| Mode | CWE-89 | CWE-78 | CWE-79 | Overall |
|------|--------|--------|--------|---------|
| Baseline | 100.0% | 75.7% | 0.0% | 58.6% |
| Steered | 100.0% | 82.9% | 50.0% | **77.6%** |
| Δ | +0.0pp | +7.1pp | +50.0pp | **+19.0pp** |

Routing accuracy: 21/21 (100.0%)

### Step 7: Mechanistic Comparison (07_mechanistic_comparison.py)

- Cross-language cosine similarity: 0.007 (near zero)
- PCA visualization shows clear language-specific clustering
- Cohen's d on own direction: CWE-89=2.93, CWE-78=4.07, CWE-79=5.21

## Bug Fixes During Experiment

### Scorer Calibration Issues
- **CWE-89**: 42% "other" rate → 1.3% after 3 fixes (variable-passed queries, f-prefix, triple-quoted f-strings)
- **CWE-79**: 44% "other" rate → 2.6% after 1 fix (triple-quoted f-string HTML detection)

### SteeringGenerator Prompt Stripping Bug (CRITICAL)
- **Bug**: `generated[len(prompt):]` (character-based) fails with `skip_special_tokens=True` because special tokens removed from decoded output make it shorter than prompt string
- **Fix**: Token-based stripping: `new_tokens = outputs[0][input_len:]`
- **Impact**: First LOBO run (~6 hours) produced 77% "other" rate, had to re-run

## Code

| File | Description |
|------|-------------|
| [01_baseline.py](../../src/experiments/02-10_python_cwe_steering/01_baseline.py) | Baseline generation for 3 Python CWEs |
| [02_extract_vectors.py](../../src/experiments/02-10_python_cwe_steering/02_extract_vectors.py) | Extract L31 mean-difference vectors |
| [03_lobo_validation.py](../../src/experiments/02-10_python_cwe_steering/03_lobo_validation.py) | LOBO cross-validation |
| [04_transfer_matrix.py](../../src/experiments/02-10_python_cwe_steering/04_transfer_matrix.py) | 6×6 cross-language transfer matrix |
| [05_probe_routing.py](../../src/experiments/02-10_python_cwe_steering/05_probe_routing.py) | 3-class probe routing |
| [06_e2e_pipeline.py](../../src/experiments/02-10_python_cwe_steering/06_e2e_pipeline.py) | End-to-end steered generation pipeline |
| [07_mechanistic_comparison.py](../../src/experiments/02-10_python_cwe_steering/07_mechanistic_comparison.py) | Mechanistic analysis and PCA |

## Data Files

| File | Description |
|------|-------------|
| `data/direction_cwe{89,78,79}_L31_*.npy` | Steering direction vectors |
| `data/activations_cwe{89,78,79}_L31_*.npz` | Activation data |
| `data/vector_metadata_*.json` | Vector metadata and similarity matrix |
| `data/cwe_probe_weights_*.npz` | Probe weights for routing |
| `results/baseline_results_rescored_*.json` | Re-scored baseline results |
| `results/lobo_results_*.json` | LOBO summary |
| `results/transfer_matrix_*.json` | 6×6 transfer matrix |
| `results/probe_routing_*.json` | Probe routing results |
| `results/e2e_results_*.json` | E2E pipeline results |
| `results/mechanistic_comparison_*.json` | Mechanistic comparison data |
