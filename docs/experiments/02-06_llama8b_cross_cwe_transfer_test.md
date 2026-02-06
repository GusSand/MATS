# Experiment 5c/5d: CodeQL Feasibility + Cross-CWE Transfer Test

**Date**: 2026-02-06
**Model**: Llama-3.1-8B-Instruct (fp16)
**Datasets**: CWE-787 (105 pairs), CWE-134 (105 pairs)

## Overview

Two experiments run as follow-ups to the cross-CWE parallel analysis:

1. **CodeQL Feasibility Check (5c)**: Can CWE-134 steered outputs be compiled and analyzed with CodeQL?
2. **Cross-CWE Transfer Test (5d)**: Do steering vectors transfer across CWE types?

## Experiment 5c: CodeQL Feasibility Check for CWE-134

### Research Question

Are CWE-134 steered outputs at α=1.5 compilable enough for CodeQL static analysis?

### Method

- Sampled 30 CWE-134 outputs at α=1.5 from pilot/LOBO fold results (random seed=42)
- Extracted C code using regex-based extraction (reused from `01-14_codeql_scoring_prototype/02_wrap_code.py`)
- Wrapped extracted code as standalone C files: added `#include` headers, `syslog.h` when needed, `main()` stub when absent
- Tested compilation with `gcc -fsyntax-only -w` (syntax check only, suppress warnings)

### Results

| Metric | Value |
|---|---|
| Total samples | 30 |
| Has extractable code | 30/30 (100%) |
| Compiles successfully | 30/30 (100%) |
| Compilation failures | 0 |
| No code extracted | 0 |

### Observations (No Interpretation)

- 100% compilation rate is the highest of any CWE tested
- CWE-134 outputs are cleaner C code than CWE-787 (format string patterns are simpler syntactically)
- CodeQL validation is feasible for CWE-134

## Experiment 5d: Cross-CWE Transfer Test (CWE-787 ↔ CWE-134)

### Research Question

Do activation steering vectors transfer across CWE types? Given cosine similarity of 0.48 between CWE-787 and CWE-134 directions, does applying a "foreign" direction produce proportional security improvements?

### Hypothesis

Transfer rate ≈ cosine_similarity (0.48) × native rate

### Method

- **Direction vectors**: L31 mean-difference vectors computed by `vector_correlation_analysis.py`
  - CWE-787 direction: norm=7.77 (4096-dim, float32)
  - CWE-134 direction: norm=8.51 (4096-dim, float32)
  - Cosine similarity: 0.4819
- **Transfer 1**: CWE-787 direction → CWE-134 prompts (105 pairs)
  - Scored with CWE-134 patterns: printf/fprintf/syslog format string checks
- **Transfer 2**: CWE-134 direction → CWE-787 prompts (105 pairs)
  - Scored with CWE-787 patterns: sprintf→snprintf, strcat→strncat checks
- **Alphas**: 0.0 (baseline rerun), 1.5, 3.5
- **Generation**: temperature=0.6, top_p=0.9, max_tokens=512, seed=42
- **Total**: 6 batches × 105 prompts = 630 generations

### Results

**Transfer Results Table:**

| Condition | α=0.0 | α=1.5 | α=3.5 |
|---|---|---|---|
| 787→134 (transfer) | 62.9% | 62.9% | 55.2% |
| 134→134 (native, pilot ref) | 66.7% | 90.0% | 90.0% |
| 134→787 (transfer) | 1.0% | 5.7% | 2.9% |
| 787→787 (native, LOBO ref) | 0.0% | 12.4% | 52.4% |

**Hypothesis Check:**

| Transfer | Predicted | Actual |
|---|---|---|
| 787→134 at α=1.5 | 43.4% | 62.9% |
| 134→787 at α=3.5 | 25.3% | 2.9% |

### Observations (No Interpretation)

- **787→134**: CWE-787 direction has NO effect on CWE-134 security. Rate stays at 62.9% (same as baseline), and actually drops to 55.2% at α=3.5 (7.7pp below baseline)
- **134→787**: CWE-134 direction has negligible effect on CWE-787 security. Goes from 1.0% baseline to 5.7% at α=1.5, then drops to 2.9% at α=3.5
- **Baseline discrepancy**: 787→134 baseline (62.9%) is slightly different from native 134 baseline (66.7%) because baseline runs differ in random sampling
- **Hypothesis rejected**: The linear relationship (transfer ≈ cosine_sim × native) does not hold. The 48% cosine overlap does not translate to proportional steering transfer
- Directions are CWE-specific: despite moderate geometric similarity in activation space, the "security" component is not shared across CWE types

## Code / Files Generated

- [cross_cwe_transfer_test.py](../../src/experiments/02-05_cross_cwe_steering/cross_cwe_transfer_test.py) - Bidirectional transfer test with scoring
- [cwe134_codeql_feasibility.py](../../src/experiments/02-05_cross_cwe_steering/cwe134_codeql_feasibility.py) - CodeQL feasibility check (gcc compilation test)

## Data Generated

All output data in `src/experiments/02-05_cross_cwe_steering/cross_cwe_analysis/data/`:

| File | Description |
|---|---|
| `cross_cwe_transfer_20260206_040528.json` | Transfer test summary (secure rates, hypothesis check) |
| `cross_cwe_transfer_full_20260206_040528.json` | Full per-sample outputs for all 630 generations |
| `cwe134_codeql_feasibility.json` | CodeQL feasibility results (30 samples, compilation status) |
