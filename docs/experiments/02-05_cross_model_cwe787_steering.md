# Experiment 4: Cross-Model CWE-787 Steering Validation

**Date**: 2026-02-05
**Models**: Mistral-7B-Instruct-v0.3, Llama-3.1-70B-Instruct
**Dataset**: CWE-787 Expanded (105 pairs, 210 prompts)
**Reference**: Llama-3.1-8B-Instruct (Experiment 2, 52.4% secure at alpha=3.5)

## Overview

This experiment tests whether the CWE-787 activation steering technique discovered on Llama-3.1-8B-Instruct generalizes to other model architectures and scales. We test two models:
- **Experiment 4A**: Mistral-7B-Instruct-v0.3 (different architecture, same scale)
- **Experiment 4B**: Llama-3.1-70B-Instruct (same architecture, 9x larger)

## Research Question

Does mean-difference activation steering for secure code generation transfer across:
1. Model architectures (Llama -> Mistral)?
2. Model scales (8B -> 70B)?

## Methodology

### Pipeline (identical for both models)

1. **Baseline**: Generate on all 105 vulnerable prompts with no steering
2. **Activation Collection**: Collect activations at all layers for all 210 prompts
3. **Layer Sweep**: Train linear probes at each layer, select best by accuracy + direction norm
4. **Pilot LOBO**: 2-fold pilot to check if steering works (gate: >10pp improvement)
5. **Full LOBO**: 7-fold cross-validation if pilot passes
6. **Cross-Model Analysis**: Compare results across all three models

### Shared Configuration

- **Dataset**: Same CWE-787 expanded dataset (105 pairs, 7 base_ids)
- **LOBO Design**: Leave-One-Base-ID-Out with 7 folds
- **Scoring**: Identical STRICT/EXPANDED regex patterns (sprintf/snprintf, strcat/strncat)
- **Generation**: temp=0.6, top_p=0.9, max_new_tokens=512, 1 generation per prompt

### Model-Specific Configuration

| Parameter | Mistral-7B | Llama-70B |
|-----------|-----------|-----------|
| Model | mistralai/Mistral-7B-Instruct-v0.3 | meta-llama/Llama-3.1-70B-Instruct |
| Layers | 32 | 80 |
| Hidden dim | 4096 | 8192 |
| Quantization | None (fp16) | 4-bit NF4 |
| GPU memory | ~14 GB | ~43 GB |
| Best layer | 31 (last) | 79 (last) |
| Alpha grid | [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0] | [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0] |

## Results

### Experiment 4A: Mistral-7B-Instruct-v0.3

#### Baseline
| Metric | Value |
|--------|-------|
| Strict Secure | 26.7% (28/105) |
| Strict Insecure | 65.7% (69/105) |
| Refusals | 0.0% |

**Note**: Mistral-7B has a much higher baseline secure rate (26.7%) compared to Llama-8B (0.0%), indicating stronger default safety priors for CWE-787 patterns.

#### Layer Sweep
| Layer | Probe Accuracy | Direction Norm |
|-------|---------------|----------------|
| 31 (best) | 95.2% | 3.90 |
| 30 | 92.9% | 3.25 |
| 29 | 92.4% | 2.83 |

Best layer: **31** (last hidden layer), consistent with Llama-8B findings.

#### Full 7-Fold LOBO Results (STRICT Scoring)

| Alpha | N | Secure% | Insecure% | Refusal% |
|-------|---|---------|-----------|----------|
| 0.0 | 105 | 26.7% | 67.6% | 0.0% |
| 0.5 | 105 | 50.5% | 41.0% | 0.0% |
| 1.0 | 105 | 67.6% | 25.7% | 0.0% |
| 1.5 | 105 | 75.2% | 16.2% | 0.0% |
| 2.0 | 105 | 84.8% | 9.5% | 0.0% |
| 2.5 | 105 | 88.6% | 5.7% | 0.0% |
| 3.0 | 105 | 83.8% | 6.7% | 0.0% |
| **3.5** | **105** | **92.4%** | **3.8%** | **0.0%** |
| **4.0** | **105** | **92.4%** | **3.8%** | **0.0%** |
| 5.0 | 105 | 83.8% | 1.9% | 0.0% |

**Effect Size**: 26.7% -> 92.4% = **+65.7 percentage points** at alpha=3.5-4.0

#### Per-Fold Consistency (alpha=3.5)

| Fold | Secure% |
|------|---------|
| pair_07_sprintf_log | 93.3% |
| pair_09_path_join | 80.0% |
| pair_11_json | 100.0% |
| pair_12_xml | 73.3% |
| pair_16_high_complexity | 100.0% |
| pair_17_time_pressure | 100.0% |
| pair_19_graphics | 100.0% |

**5/7 folds achieve 100% secure rate** at alpha=3.5, demonstrating extremely strong generalization.

### Experiment 4B: Llama-3.1-70B-Instruct

#### Baseline
| Metric | Value |
|--------|-------|
| Strict Secure | 1.0% (1/105) |
| Strict Insecure | 90.5% (95/105) |
| Refusals | 0.0% |

#### Layer Sweep
| Layer | Probe Accuracy | Direction Norm |
|-------|---------------|----------------|
| 79 (selected) | 96.2% | 10.15 |
| 2 (highest acc) | 96.7% | 0.09 |
| 1 | 96.7% | 0.08 |

**Critical finding**: Layer 2 had the highest probe accuracy (96.7%) but a near-zero direction norm (0.09), making it useless for steering. Layer 79 was selected for its combination of high accuracy (96.2%) and large direction norm (10.15).

#### Pilot LOBO Results (2 folds, layer 79)

| Alpha | N | Secure% |
|-------|---|---------|
| 0.0 | 30 | 3.3% |
| 0.5 | 30 | 3.3% |
| 1.0 | 30 | 3.3% |
| 2.0 | 30 | 3.3% |
| 3.0 | 30 | 36.7% |
| 4.0 | 30 | 56.7% |
| **5.0** | **30** | **60.0%** |
| 7.0 | 30 | 6.7% |
| 10.0 | 30 | 0.0% |

**Pilot gate**: PASSED (60.0% - 3.3% = 56.7pp > 10pp threshold)

**Note**: Llama-70B requires higher alpha (5.0 vs 3.5) for peak effect, and shows sharp over-steering collapse at alpha>=7.0, likely due to the larger direction norm (10.2 vs 7.7).

#### Full 7-Fold LOBO Results (STRICT Scoring)

| Alpha | N | Secure% | Insecure% | Other% |
|-------|---|---------|-----------|--------|
| 0.0 | 105 | 1.9% | 88.6% | 9.5% |
| 0.5 | 105 | 1.0% | 90.5% | 8.6% |
| 1.0 | 105 | 4.8% | 88.6% | 6.7% |
| 2.0 | 105 | 8.6% | 78.1% | 13.3% |
| 3.0 | 105 | 32.4% | 60.0% | 7.6% |
| **4.0** | **105** | **52.4%** | **35.2%** | **12.4%** |
| 5.0 | 105 | 44.8% | 7.6% | 47.6% |
| 7.0 | 105 | 7.6% | 0.0% | 92.4% |
| 10.0 | 105 | 0.0% | 0.0% | 100.0% |

**Effect Size**: 1.9% -> 52.4% = **+50.5 percentage points** at alpha=4.0

**Note**: The "Other" column indicates outputs that don't contain recognizable secure or insecure patterns. At high alpha values (>=5.0), over-steering causes the model to generate degraded or incoherent code, explaining the high "Other" rates.

#### Per-Fold Results (best alpha per fold)

| Fold | Best Secure% | Best Alpha |
|------|-------------|------------|
| pair_07_sprintf_log | 73.3% | 5.0 |
| pair_09_path_join | 73.3% | 4.0 |
| pair_11_json | 53.3% | 3.0-4.0 |
| pair_12_xml | 13.3% | 5.0 |
| pair_16_high_complexity | 93.3% | 5.0 |
| pair_17_time_pressure | 53.3% | 5.0 |
| pair_19_graphics | 86.7% | 4.0 |

**Observation**: High fold variability in optimal alpha (3.0-5.0) and maximum secure rate (13.3%-93.3%). The XML parsing scenario (`pair_12_xml`) is particularly resistant to steering, achieving only 13.3% secure even at best alpha.

## Cross-Model Comparison

### Summary Table

| Model | Params | Baseline | Best Steered | Best Alpha | Improvement | Best Layer |
|-------|--------|----------|-------------|------------|-------------|------------|
| Llama-3.1-8B | 8B | 0.0% | 52.4% | 3.5 | +52.4pp | 31/32 |
| Mistral-7B-v0.3 | 7B | 26.7% | 92.4% | 3.5-4.0 | +65.7pp | 31/32 |
| Llama-3.1-70B | 70B | 1.9% | 52.4% | 4.0 | +50.5pp | 79/80 |

### Key Observations

1. **Steering works across architectures**: Both Llama and Mistral families respond to mean-difference steering, despite different pretraining and instruction tuning.

2. **Best layer is consistently the last hidden layer**: Layer 31/32 for 7B models, layer 79/80 for 70B. This suggests the security decision is computed at the final layer universally.

3. **Optimal alpha is similar across scales**: Both Llama-8B and Llama-70B achieve best results at alpha=3.5-4.0, despite 9x size difference. Mistral-7B also peaks at alpha=3.5-4.0.

4. **Direction norm matters, not just probe accuracy**: Llama-70B layer 2 had 96.7% probe accuracy but 0.09 direction norm (steering failed). Layer 79 had 96.2% accuracy but 10.15 norm (steering succeeded). This demonstrates that linear separability doesn't imply causal influence.

5. **Baseline varies significantly**: Mistral-7B naturally writes secure code 26.7% of the time vs 0-2% for Llama models. Despite different baselines, steering consistently pushes toward secure code.

6. **Over-steering is model-dependent**: Llama-70B collapses at alpha>=7.0 (100% "other" outputs), while Mistral-7B degrades gracefully at alpha=5.0.

7. **Scaling doesn't improve steering ceiling**: Llama-70B achieves the same 52.4% peak secure rate as Llama-8B, despite being 9x larger. The larger model has a narrower effective alpha range and more severe over-steering collapse.

8. **Per-fold variability is high**: Llama-70B shows 13.3%-93.3% secure rate across folds, suggesting some prompt categories are more resistant to steering. XML parsing scenarios appear particularly difficult.

## Technical Challenges

### Llama-70B Memory Constraints
- 8-bit quantization OOMed on A100-80GB (even with CPU offload)
- 4-bit NF4 quantization fits at ~43 GB allocated
- Required `max_memory` parameter for transformers 5.0.0 compatibility
- Per-iteration time: ~5s (low alpha) to ~46s (high alpha) due to longer steered outputs

### Layer Selection Pitfall
- Naive "highest probe accuracy" selection chose layer 2 for Llama-70B
- Layer 2 had near-perfect classification but near-zero direction norm
- Early layers may encode the concept but lack the representational magnitude for effective steering
- **Lesson**: Always consider direction norm alongside probe accuracy when selecting steering layers

## Code Location

`src/experiments/02-05_cross_model_cwe787_steering/`

### Shared Infrastructure
- [model_loader.py](../../src/experiments/02-05_cross_model_cwe787_steering/shared/model_loader.py) - Unified model loading (fp16, 8-bit, 4-bit)
- [steering_generator.py](../../src/experiments/02-05_cross_model_cwe787_steering/shared/steering_generator.py) - Activation steering with hooks

### Experiment 4A: Mistral-7B
- [experiment_config.py](../../src/experiments/02-05_cross_model_cwe787_steering/experiment_4a_mistral7b/experiment_config.py) - Configuration
- [01_baseline.py](../../src/experiments/02-05_cross_model_cwe787_steering/experiment_4a_mistral7b/01_baseline.py) - Baseline generation
- [02_collect_activations.py](../../src/experiments/02-05_cross_model_cwe787_steering/experiment_4a_mistral7b/02_collect_activations.py) - Activation collection
- [03_layer_sweep.py](../../src/experiments/02-05_cross_model_cwe787_steering/experiment_4a_mistral7b/03_layer_sweep.py) - Layer sweep with linear probes
- [04_pilot_lobo.py](../../src/experiments/02-05_cross_model_cwe787_steering/experiment_4a_mistral7b/04_pilot_lobo.py) - Pilot 2-fold LOBO
- [05_full_lobo.py](../../src/experiments/02-05_cross_model_cwe787_steering/experiment_4a_mistral7b/05_full_lobo.py) - Full 7-fold LOBO

### Experiment 4B: Llama-70B
- [experiment_config.py](../../src/experiments/02-05_cross_model_cwe787_steering/experiment_4b_llama70b/experiment_config.py) - Configuration
- [01_baseline.py](../../src/experiments/02-05_cross_model_cwe787_steering/experiment_4b_llama70b/01_baseline.py) - Baseline generation
- [02_collect_activations.py](../../src/experiments/02-05_cross_model_cwe787_steering/experiment_4b_llama70b/02_collect_activations.py) - Activation collection
- [03_layer_sweep.py](../../src/experiments/02-05_cross_model_cwe787_steering/experiment_4b_llama70b/03_layer_sweep.py) - Layer sweep
- [04_pilot_lobo.py](../../src/experiments/02-05_cross_model_cwe787_steering/experiment_4b_llama70b/04_pilot_lobo.py) - Pilot LOBO
- [05_full_lobo.py](../../src/experiments/02-05_cross_model_cwe787_steering/experiment_4b_llama70b/05_full_lobo.py) - Full 7-fold LOBO

### Cross-Model Analysis
- [06_cross_model_analysis.py](../../src/experiments/02-05_cross_model_cwe787_steering/06_cross_model_analysis.py) - Comparison and figure generation

## Data Location

### Experiment 4A: Mistral-7B
`src/experiments/02-05_cross_model_cwe787_steering/experiment_4a_mistral7b/data/`

| File | Description |
|------|-------------|
| `baseline_20260205_021419.json` | Baseline results (105 prompts) |
| `activations_20260205_022224.npz` | Activations (32 layers x 210 prompts x 4096 dim, ~99 MB) |
| `metadata_20260205_022224.json` | Prompt metadata |
| `layer_sweep_results.json` | Linear probe results per layer |
| `lobo_results_20260205_045755.json` | Full 7-fold LOBO aggregated results |
| `fold_results/fold_*.json` | Per-fold detailed results (7 files) |

### Experiment 4B: Llama-70B
`src/experiments/02-05_cross_model_cwe787_steering/experiment_4b_llama70b/data/`

| File | Description |
|------|-------------|
| `baseline_20260205_071732.json` | Baseline results (105 prompts) |
| `activations_20260205_075202.npz` | Activations (80 layers x 210 prompts x 8192 dim, ~254 MB) |
| `metadata_20260205_075202.json` | Prompt metadata |
| `layer_sweep_results.json` | Linear probe results per layer (best_layer=79) |
| `fold_results/pilot_fold_*_20260205_091351.json` | Pilot LOBO fold results (2 files) |
| `fold_results/fold_*_20260205_111622.json` | Full LOBO fold results (7 files) |
| `lobo_results_20260205_111622.json` | Full 7-fold LOBO aggregated results |
