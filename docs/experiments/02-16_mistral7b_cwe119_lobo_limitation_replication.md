# Experiment 14: Mistral-7B CWE-119 LOBO (Limitation Replication)

**Date**: 2026-02-17
**Model**: mistralai/Mistral-7B-Instruct-v0.3 (fp16)
**Layer**: 31 (last hidden layer)
**Dataset**: CWE-119 Expanded (105 pairs, 7 base_ids) — C Buffer Read Overflow
**Reference**: Mistral-7B CWE-787 (Experiment 4A, 92.4% at alpha=3.5)

## Overview

This experiment tests CWE-119 (buffer read overflow) steering on Mistral-7B, investigating whether the CWE-119 steering limitation discovered on Llama replicates on a second architecture. Additionally, we compare the CWE-787 vs CWE-119 representational similarity across architectures.

## Research Question

1. Does the CWE-119 steering failure replicate on Mistral-7B?
2. Are CWE-787 and CWE-119 "representationally inseparable" on Mistral as they were on Llama?

## Methodology

### Configuration

| Parameter | Value |
|-----------|-------|
| Model | mistralai/Mistral-7B-Instruct-v0.3 |
| Quantization | None (fp16) |
| Hidden dim | 4096 |
| Best layer | 31 |
| Alpha grid | [0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 5.0] |
| Seeds | 3 (42, 123, 456) |
| Temperature | 0.6 |
| Top-p | 0.9 |
| Max new tokens | 512 |
| Total generations | 2,205 |

### Scoring

CWE-119 uses both strict and expanded scoring:
- **Strict secure**: Uses bounds-checked functions (e.g., `memcpy` with size validation)
- **Expanded secure**: Includes additional safe patterns beyond strict criteria

## Results

### Aggregated Alpha Sweep (N=315 per alpha)

| Alpha | N | Strict Secure% | Strict Insecure% | Expanded Secure% | Refusal% |
|-------|---|----------------|-------------------|------------------|----------|
| 0.0 | 315 | 0.3% | 96.8% | 1.0% | 0.0% |
| 1.0 | 315 | 1.0% | 94.6% | 1.3% | 0.0% |
| 2.0 | 315 | 1.0% | 93.7% | 3.8% | 0.0% |
| **3.0** | **315** | **1.6%** | **94.3%** | **6.3%** | **0.0%** |
| 3.5 | 315 | 1.3% | 93.3% | 4.1% | 0.0% |
| 4.0 | 315 | 1.0% | 92.7% | 2.9% | 0.0% |
| 5.0 | 315 | 1.3% | 90.8% | 2.9% | 0.0% |

**Best alpha (strict)**: 3.0 (1.6% secure, +1.3pp over baseline)
**Best alpha (expanded)**: 3.0 (6.3% expanded secure)

### Per-Fold Breakdown

| Fold (base_id) | Dir. Norm | Strict Baseline | Strict Best | Strict Best Alpha | Expanded Baseline | Expanded Best | Expanded Best Alpha |
|-----------------|-----------|-----------------|-------------|-------------------|-------------------|---------------|---------------------|
| pair_01_user_input | 5.954 | 0.0% | 0.0% | — | 0.0% | 8.9% | 3.0 |
| pair_02_command_parser | 5.898 | 0.0% | 0.0% | — | 0.0% | 11.1% | 3.0 |
| pair_03_config_reader | 5.882 | 0.0% | 0.0% | — | 0.0% | 13.3% | 2.0 |
| pair_04_username_copy | 6.019 | 0.0% | 6.7% | 3.5 | 0.0% | 6.7% | 3.5 |
| pair_05_filepath_copy | 5.972 | 2.2% | 6.7% | 1.0 | 6.7% | 6.7% | 0.0 |
| pair_06_error_msg_copy | 6.016 | 0.0% | 2.2% | 5.0 | 0.0% | 2.2% | 5.0 |
| pair_07_hostname_copy | 6.000 | 0.0% | 0.0% | — | 0.0% | 0.0% | — |

**Key observations**:
- Direction norms are consistent across folds (~5.9-6.0), much larger than CWE-89 on Mistral (~1.1)
- Most folds show essentially zero strict secure rate even at best alpha
- pair_07_hostname_copy never improves under any scoring criterion
- Only pair_04 and pair_05 show any strict secure improvement

### CWE-787 vs CWE-119 Representational Comparison

| Metric | CWE-787 (Mistral) | CWE-119 (Mistral) |
|--------|-------------------|-------------------|
| Baseline secure | 26.7% | 0.3% |
| Best secure | 92.4% | 1.6% |
| Improvement | +65.7pp | +1.3pp |
| Best alpha | 3.5 | 3.0 |
| Direction norm | ~3.9 | ~6.0 |

**CWE-787 vs CWE-119 cosine similarity: 0.005 (near orthogonal)**

### Cross-Architecture Comparison of CWE-787/CWE-119 Relationship

| Property | Llama-8B | Mistral-7B |
|----------|----------|------------|
| CWE-787 best rate | 52.4% | 92.4% |
| CWE-119 best rate | ~0% | 1.6% |
| CWE-119 improvement | ~0pp | +1.3pp |
| 787/119 cosine similarity | High (representationally inseparable) | 0.005 (near orthogonal) |
| CWE-119 steering works? | No | No |

## Notable Findings

1. **CWE-119 failure replicates**: Steering for buffer read overflow produces near-zero improvement on both Llama and Mistral. This is a consistent limitation across architectures.

2. **Different failure mechanism**: On Llama, CWE-787 and CWE-119 directions were "representationally inseparable" (high cosine similarity), suggesting the model treated them as the same vulnerability. On Mistral, they are nearly orthogonal (cosine=0.005), meaning the model distinguishes them but STILL cannot steer CWE-119. The failure is not caused by representational overlap.

3. **Inherently harder vulnerability**: CWE-119 (buffer read overflow) may lack a clear "secure alternative" pattern in the model's learned code generation space. Unlike CWE-787 (where `snprintf` is an obvious safe replacement for `sprintf`), there is no single canonical fix for buffer read overflows — the fix depends heavily on context.

4. **Contrast with CWE-787**: The same model (Mistral) achieves 92.4% for CWE-787 but only 1.6% for CWE-119, despite both being buffer-related C vulnerabilities. This is the strongest evidence that steering effectiveness is vulnerability-specific, not language-specific.

## Code

- [01_run_experiment.py](../../src/experiments/02-17_mistral_cwe119_lobo/01_run_experiment.py) — Full pipeline: activation collection, direction extraction, LOBO cross-validation
- Results directory: `src/experiments/02-17_mistral_cwe119_lobo/results/`
- Data directory: `src/experiments/02-17_mistral_cwe119_lobo/data/`
