# Experiment 23: Mistral-Small-24B CWE-119 LOBO

**Date**: 2026-02-27
**Model**: mistralai/Mistral-Small-24B-Instruct-2501
**Task**: CWE-119 7-fold LOBO activation steering

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Mistral-Small-24B-Instruct-2501 |
| Parameters | ~24B |
| Layers | 40 |
| Hidden dim | 5120 |
| Quantization | None (fp16, ~47 GB VRAM) |
| Steering layer | 39 (last hidden layer) |
| Alpha grid | [0.0, 1.0, 1.5, 2.0] |
| Dataset | CWE-119 expanded (105 pairs) |
| Folds | 7 (LOBO by base_id) |
| Temperature | 0.6 |
| Top-p | 0.9 |
| Max new tokens | 512 |
| Generations/prompt | 1 |
| Seed | 42 |

## Base IDs (7 folds)

1. pair_01_user_input
2. pair_02_command_parser
3. pair_03_config_reader
4. pair_04_username_copy
5. pair_05_filepath_copy
6. pair_06_error_msg_copy
7. pair_07_hostname_copy

## Aggregated Results

| Alpha | Strict Secure% | Strict Insecure% | Expanded Secure% | Refusal% |
|-------|---------------|-------------------|-------------------|----------|
| 0.0   | 0.0%          | 100.0%            | 0.0%              | 0.0%     |
| 1.0   | 0.0%          | 100.0%            | 0.0%              | 0.0%     |
| 1.5   | 0.0%          | 100.0%            | 0.0%              | 0.0%     |
| 2.0   | 8.6%          | 89.5%             | 8.6%              | 0.0%     |

**Baseline**: 0.0% strict secure (100% insecure)
**Best**: 8.6% strict secure at α=2.0 (+8.6pp)
**Overall direction norm**: 13.36
**Refusals**: 0% across all alphas

## Per-Fold Results at α=2.0

| Fold | Dir Norm | Strict Secure% | Strict Insecure% |
|------|----------|----------------|-------------------|
| pair_01_user_input | 11.92 | 6.7% | 93.3% |
| pair_02_command_parser | 11.90 | 20.0% | 73.3% |
| pair_03_config_reader | 11.83 | 26.7% | 73.3% |
| pair_04_username_copy | 14.60 | 6.7% | 93.3% |
| pair_05_filepath_copy | 14.58 | 0.0% | 100.0% |
| pair_06_error_msg_copy | 14.61 | 0.0% | 93.3% |
| pair_07_hostname_copy | 14.61 | 0.0% | 93.3% |

### Observations
- Very weak steering effect compared to CWE-787 (8.6pp vs 39.0pp on same model)
- Direction norm is ~1.8x larger than CWE-787 (13.36 vs 7.53) — large norm does not correlate with better steering
- Clear two-cluster pattern in fold norms: gets-type folds (~11.9) vs strcpy-type folds (~14.6)
- strcpy-type folds (pairs 04-07) show almost no response to steering
- Only gets-type folds show modest improvement, and only at maximum alpha
- Alpha grid may be too narrow — higher alphas might help but risk coherence

## Cross-Model Comparison (CWE-119 LOBO)

| Model | Params | Best α | Strict Secure% | Improvement |
|-------|--------|--------|----------------|-------------|
| Llama-3.1-8B-Instruct | 8B | 3.0 | 24.8% | +24.8pp |
| Mistral-7B-Instruct | 7B | 2.0 | 6.7% | +6.7pp |
| **Mistral-Small-24B** | **24B** | **2.0** | **8.6%** | **+8.6pp** |

Note: CWE-119 steering is consistently weaker than CWE-787 across all models. Mistral family shows particularly weak CWE-119 response.

## Code

- [experiment_config.py](../../src/experiments/02-27_mistral24b_cwe787_lobo/experiment_config.py) - Shared configuration
- [02_cwe119_lobo.py](../../src/experiments/02-27_mistral24b_cwe787_lobo/02_cwe119_lobo.py) - CWE-119 LOBO script

## Results Files

- `src/experiments/02-27_mistral24b_cwe787_lobo/results/cwe119_lobo_results_20260227_165800.json` - Aggregated results
- `src/experiments/02-27_mistral24b_cwe787_lobo/results/cwe119_lobo_full_20260227_165800.json` - Full per-generation data
- `src/experiments/02-27_mistral24b_cwe787_lobo/results/cwe119_fold_pair_*_20260227_165800.json` - Per-fold results
- `src/experiments/02-27_mistral24b_cwe787_lobo/data/activations_mistral24b_cwe119_L39.npz` - Cached activations
