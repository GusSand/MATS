# Experiment 13: Mistral-7B CWE-89 LOBO (Cross-Architecture Replication)

**Date**: 2026-02-16
**Model**: mistralai/Mistral-7B-Instruct-v0.3 (fp16)
**Layer**: 31 (last hidden layer)
**Dataset**: CWE-89 Expanded (105 pairs, 7 base_ids) — Python SQL Injection
**Reference**: Llama-3.1-8B-Instruct (Experiment 10, 70.3% at alpha=5.0)

## Overview

This experiment replicates the CWE-89 (SQL injection) LOBO steering validation on a second architecture: Mistral-7B. The goal is to determine whether mean-difference activation steering generalizes beyond the Llama architecture for Python vulnerability types.

## Research Question

Does mean-difference activation steering for CWE-89 (SQL injection) generalize from Llama-8B to Mistral-7B? How do baseline security and steering effectiveness compare across architectures?

## Methodology

### Pipeline

1. **Activation Collection**: Collect activations at Layer 31 for all 210 prompts (105 insecure + 105 secure)
2. **Direction Extraction**: Mean-difference steering vector (secure - insecure)
3. **LOBO**: 7-fold leave-one-base-out cross-validation with alpha sweep
4. **Scoring**: CWE-89 regex scorer (parameterized queries = secure, string formatting = insecure)

### Configuration

| Parameter | Value |
|-----------|-------|
| Model | mistralai/Mistral-7B-Instruct-v0.3 |
| Quantization | None (fp16) |
| Hidden dim | 4096 |
| Best layer | 31 |
| Alpha grid | [0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0] |
| Seeds | 3 (42, 123, 456) |
| Temperature | 0.6 |
| Top-p | 0.9 |
| Max new tokens | 512 |
| Total generations | 2,835 |

## Results

### Aggregated Alpha Sweep (N=315 per alpha)

| Alpha | N | Secure | Insecure | Other | Secure% | Insecure% | Other% |
|-------|---|--------|----------|-------|---------|-----------|--------|
| 0.0 | 315 | 135 | 180 | 0 | 42.9% | 57.1% | 0.0% |
| 1.0 | 315 | 148 | 167 | 0 | 47.0% | 53.0% | 0.0% |
| 2.0 | 315 | 168 | 147 | 0 | 53.3% | 46.7% | 0.0% |
| 3.0 | 315 | 187 | 128 | 0 | 59.4% | 40.6% | 0.0% |
| 3.5 | 315 | 192 | 123 | 0 | 61.0% | 39.0% | 0.0% |
| 4.0 | 315 | 194 | 121 | 0 | 61.6% | 38.4% | 0.0% |
| 5.0 | 315 | 196 | 119 | 0 | 62.2% | 37.8% | 0.0% |
| **6.0** | **315** | **200** | **115** | **0** | **63.5%** | **36.5%** | **0.0%** |
| 7.0 | 315 | 197 | 118 | 0 | 62.5% | 37.5% | 0.0% |

**Best alpha**: 6.0 (63.5% secure, +20.6pp over baseline)

### Per-Fold Breakdown

| Fold (base_id) | Dir. Norm | Baseline (a=0) | Best Rate | Best Alpha | Delta |
|-----------------|-----------|----------------|-----------|------------|-------|
| admin_delete | 1.192 | 0.0% | 0.0% | — | +0.0pp |
| log_entry | 1.095 | 91.1% | 97.8% | 3.0 | +6.7pp |
| order_history | 1.127 | 24.4% | 80.0% | 6.0 | +55.6pp |
| product_search | 1.116 | 40.0% | 95.6% | 6.0 | +55.6pp |
| report_filter | 1.091 | 66.7% | 91.1% | 3.0 | +24.4pp |
| user_login | 1.071 | 77.8% | 84.4% | 2.0 | +6.7pp |
| user_profile_update | 1.213 | 0.0% | 0.0% | — | +0.0pp |

**Key observations**:
- Direction norms are very consistent across folds (~1.07-1.21), much smaller than Llama (~2.7)
- Two folds (admin_delete, user_profile_update) at 0% baseline never improve — completely resistant to steering
- order_history and product_search show dramatic improvement (+55.6pp each)
- log_entry starts very high (91.1%) and reaches near-ceiling (97.8%)

### Cross-Architecture Comparison (CWE-89)

| Metric | Llama-3.1-8B | Mistral-7B |
|--------|-------------|------------|
| Baseline | 57.0% | 42.9% |
| Best rate | 70.3% | 63.5% |
| Best alpha | 5.0 | 6.0 |
| Improvement | +13.3pp | +20.6pp |
| Direction norm | ~2.7 | ~1.1 |
| Other rate | Low | 0.0% (zero) |

- Mistral has a **lower baseline** (42.9% vs 57.0%) indicating less inherent SQL safety
- But a **stronger steering effect** (+20.6pp vs +13.3pp)
- Zero "other" rate across all alphas — Mistral tolerates CWE-89 steering without coherence collapse
- Smaller direction norms (~1.1 vs ~2.7) — the security direction is more compact in Mistral's hidden space

## Notable Findings

1. **Zero coherence collapse**: Unlike many other CWE/model combinations, CWE-89 on Mistral produces zero "other" outputs at any alpha tested (up to 7.0). This is remarkable.

2. **Monotonic improvement through alpha=6.0**: The secure rate increases monotonically from 42.9% to 63.5%, with only slight rollover at alpha=7.0 (62.5%).

3. **Two stubborn folds**: admin_delete and user_profile_update are completely resistant to steering. These prompt patterns likely describe SQL operations where the model has no parameterized query pattern to fall back to.

4. **Massive fold variance**: Secure rate ranges from 0% (admin_delete) to 97.8% (log_entry), indicating prompt-dependent vulnerability.

## Code

- [01_run_experiment.py](../../src/experiments/02-16_mistral_cwe89_lobo/01_run_experiment.py) — Full pipeline: activation collection, direction extraction, LOBO cross-validation
- Results directory: `src/experiments/02-16_mistral_cwe89_lobo/results/`
- Data directory: `src/experiments/02-16_mistral_cwe89_lobo/data/`
