# Experiment 6: Unified Steering Vector (Combined CWE-787/119/134)

**Date**: 2026-02-06
**Model**: Llama-3.1-8B-Instruct (fp16)
**Datasets**: CWE-787 (105 pairs) + CWE-119 (105 pairs) + CWE-134 (105 pairs) = 315 pairs

## Overview

Tests whether a **single** steering vector trained on combined CWE data provides broad-spectrum security improvement, compared to per-CWE native vectors.

## Research Question

Can a unified mean-difference direction (computed across CWE-787, CWE-119, and CWE-134 data simultaneously) match or approach the per-CWE native steering performance?

## Method

- **Model**: meta-llama/Meta-Llama-3.1-8B-Instruct, fp16, no quantization
- **Layer**: 31 (best per-CWE layer from prior experiments)
- **Combined dataset**: 315 pairs (105 per CWE), 630 prompts total
- **Unified direction**: `vec_unified = mean(all_secure_L31) - mean(all_insecure_L31)` across all 315 pairs
- **Activation collection**: Forward pass on all 630 prompts, extract L31 residual stream activations
- **Alpha grids** (per CWE):
  - CWE-787: [2.0, 3.0, 3.5, 4.0]
  - CWE-119: [3.0, 4.0, 5.0]
  - CWE-134: [1.0, 1.5, 2.0]
- **Scoring**: Each CWE scored with its native patterns (CWE-787: sprintf/strcat, CWE-119: gets/strcpy, CWE-134: printf_format/fprintf_format/syslog_format)
- **Generation**: temperature=0.6, top_p=0.9, max_tokens=512, seed=42
- **Baselines**: Reused from prior experiments (not regenerated)
- **Total generations**: 10 alpha-CWE combinations × 105 prompts = 1,050 steered outputs

## Direction Analysis

| Metric | Value |
|---|---|
| Unified direction norm | 6.884 |
| CWE-787 native norm | 7.773 |
| CWE-119 native norm | 8.656 |
| CWE-134 native norm | 8.508 |
| Cosine sim (unified ↔ CWE-787) | 0.7706 |
| Cosine sim (unified ↔ CWE-119) | 0.8529 |
| Cosine sim (unified ↔ CWE-134) | 0.8558 |

## Results

### CWE-787 (sprintf/strcat → snprintf/strncat)

| Alpha | Strict Secure | Strict Insecure | Other | Secure Rate | Expanded Rate |
|---|---|---|---|---|---|
| 2.0 | 6/105 | 94/105 | 5/105 | 5.7% | 5.7% |
| 3.0 | 15/105 | 82/105 | 8/105 | 14.3% | 14.3% |
| 3.5 | 14/105 | 80/105 | 11/105 | 13.3% | 13.3% |
| **4.0** | **22/105** | **70/105** | **13/105** | **21.0%** | **21.9%** |

### CWE-119 (gets/strcpy → fgets/strncpy)

| Alpha | Strict Secure | Strict Insecure | Other | Secure Rate | Expanded Rate | Refusals |
|---|---|---|---|---|---|---|
| **3.0** | **5/105** | **98/105** | **2/105** | **4.8%** | **4.8%** | 0 |
| 4.0 | 4/105 | 85/105 | 16/105 | 3.8% | 5.7% | 0 |
| 5.0 | 3/105 | 69/105 | 33/105 | 2.9% | 4.8% | 2 |

### CWE-134 (printf/fprintf/syslog format strings)

| Alpha | Strict Secure | Strict Insecure | Other | Secure Rate | Expanded Rate |
|---|---|---|---|---|---|
| **1.0** | **73/105** | **32/105** | **0/105** | **69.5%** | **69.5%** |
| 1.5 | 67/105 | 37/105 | 1/105 | 63.8% | 63.8% |
| 2.0 | 68/105 | 37/105 | 0/105 | 64.8% | 64.8% |

### Summary: Unified vs Native

| CWE | Baseline | Native Best (α) | Unified Best (α) | Delta |
|---|---|---|---|---|
| CWE-787 | 0.0% | 52.4% (α=3.5) | 21.0% (α=4.0) | **-31.4pp** |
| CWE-119 | 0.0% | 20.0% (α=4.0) | 4.8% (α=3.0) | **-15.2pp** |
| CWE-134 | 66.7% | 90.0% (α=1.5) | 69.5% (α=1.0) | **-20.5pp** |

## Observations (No Interpretation)

- **Unified direction norm is lower** (6.88 vs 7.77-8.66 native). Averaging across CWE types dilutes the direction.
- **CWE-787**: Unified best (21.0% at α=4.0) is less than half the native best (52.4% at α=3.5). The unified vector still moves CWE-787 outputs from 0% baseline to 21%, showing partial effect.
- **CWE-119**: Unified best (4.8% at α=3.0) is well below native best (20.0% at α=4.0). Higher alphas degrade performance — at α=5.0, secure rate drops to 2.9% and "other" (incoherent) rises to 31.4%, with 2 refusals.
- **CWE-134**: Unified best (69.5% at α=1.0) barely improves over baseline (66.7%), a gain of only +2.8pp. Native achieved +23.3pp at α=1.5. Higher unified alphas actually hurt: α=1.5 drops to 63.8%, below baseline.
- **CWE-134 inversion**: At α=1.5 and α=2.0, the unified direction makes CWE-134 outputs *less* secure than unsteered baseline. This suggests the CWE-787/119 components in the unified direction interfere with CWE-134 steering.
- **Cosine similarities** (0.77-0.86) between unified and native directions are high, but performance deltas (15-31pp) show that geometric similarity does not equal functional equivalence.
- **No refusals** for CWE-787 or CWE-134 at any alpha. CWE-119 had 2 refusals only at α=5.0.

## Code / Files Generated

- [unified_steering_experiment.py](../../src/experiments/02-05_cross_cwe_steering/unified_steering_experiment.py) - Full experiment script (S1-S7: data loading, activation collection, direction computation, steering, scoring, comparison table)

## Data Generated

All output data in `src/experiments/02-05_cross_cwe_steering/cross_cwe_analysis/data/`:

| File | Description |
|---|---|
| `unified_activations_L31_20260206_172846.npz` | Combined L31 activations for 630 prompts (315 secure + 315 insecure) |
| `direction_unified_L31_20260206_172846.npy` | Unified steering direction (4096-dim, float32) |
| `unified_steering_results_20260206_172838.json` | Full results with per-CWE breakdown and summary statistics |
| `unified_steering_full_20260206_172838.json` | Full per-sample outputs for all 1,050 steered generations |
