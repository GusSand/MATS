# Experiments 17–21: Llama-3.1-70B-Instruct Full Suite

**Date**: 2026-02-26 to 2026-02-27
**Model**: meta-llama/Meta-Llama-3.1-70B-Instruct
**Quantization**: 4-bit NF4 (BitsAndBytes)
**GPU**: A100-80GB (~42.7GB VRAM allocated)

## Overview

Complete activation steering experiment suite on a 70B-parameter model. Tests LOBO cross-validation for CWE-787 (buffer overflow), CWE-89 (SQL injection), CWE-119 (general buffer overflow), plus E2E pipeline and logit lens analysis.

## Configuration

### Model Setup
- **Quantization**: 4-bit NF4 via BitsAndBytesConfig (`load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"`)
- **Memory**: `max_memory={0: "60GiB", "cpu": "60GiB"}` for CPU offloading
- **VRAM**: 42.7GB allocated, 34.8GB headroom
- **Steering layer**: 79 (last hidden layer; model has 80 layers total)
- **Hidden dim**: 8192

### Generation Config
- temperature=0.6, top_p=0.9, max_new_tokens=512
- pad_token_id=tokenizer.eos_token_id
- Steering hook: `h[:, -1, :] += alpha * direction_tensor.to(h.dtype)`

## Experiment 17: CWE-787 (Buffer Overflow) LOBO

### Setup
- **Dataset**: CWE-787 expanded (105 pairs, 7 base_ids)
- **Alpha grid**: [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
- **Generations per prompt**: 1
- **Runtime**: ~8h

### Results

#### Aggregated (All 7 Folds)

| Alpha | Secure% | Insecure% | Refusal% |
|-------|---------|-----------|----------|
| 0.0   | 1.9%    | 88.6%     | 0.0%     |
| 0.5   | 1.0%    | 90.5%     | 0.0%     |
| 1.0   | 4.8%    | 88.6%     | 0.0%     |
| 2.0   | 8.6%    | 78.1%     | 0.0%     |
| 3.0   | 32.4%   | 60.0%     | 0.0%     |
| **4.0** | **52.4%** | **35.2%** | **0.0%** |
| 5.0   | 44.8%   | 7.6%      | 0.0%     |
| 7.0   | 7.6%    | 0.0%      | 0.0%     |
| 10.0  | 0.0%    | 0.0%      | 0.0%     |

**Best alpha: 4.0** → 52.4% strict secure rate (+50.5pp from baseline)

#### Direction Norms by Fold

| Fold | Direction Norm |
|------|---------------|
| pair_07_sprintf_log | ~11.5 |
| pair_09_path_join | ~10.8 |
| pair_11_json | ~11.5 |
| pair_12_xml | ~10.5 |
| pair_16_high_complexity | ~9.9 |
| pair_17_time_pressure | ~10.2 |
| pair_19_graphics | ~10.1 |

Direction norms (9.9–12.0) are larger than Llama-8B (7.3–8.1), consistent with the larger hidden dimension (8192 vs 4096).

#### Key Observations
- Very low baseline (1.9%) — 70B strongly defaults to sprintf over snprintf
- Sharp transition zone: alpha 3.0→4.0 jumps from 32.4% to 52.4%
- Rapid degeneration: alpha 5.0 drops to 44.8%, alpha 7.0+ produces gibberish
- Zero refusals at any alpha (unlike some smaller models)
- Effective magnitude at best alpha: ~11 × 4.0 = ~44 (vs ~30-35 sweet spot for smaller models)

## Experiment 18: CWE-89 (SQL Injection) LOBO

### Setup
- **Dataset**: CWE-89 expanded (105 pairs, 7 base_ids)
- **Alpha grid**: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
- **Seeds**: [42, 123, 456] (3 per prompt)
- **Generations per prompt**: 3 (one per seed)
- **Runtime**: ~5.75h

### Results

#### Aggregated (All 7 Folds, 315 total generations)

| Alpha | Secure% | Insecure% | Other% |
|-------|---------|-----------|--------|
| 0.0   | 52.1%   | 47.6%     | 0.3%   |
| 1.0   | 54.0%   | 46.0%     | 0.0%   |
| 2.0   | 54.9%   | 45.1%     | 0.0%   |
| 3.0   | 58.1%   | 41.6%     | 0.3%   |
| 4.0   | 60.3%   | 38.7%     | 1.0%   |
| **5.0** | **60.6%** | **37.8%** | **1.6%** |
| 6.0   | 59.7%   | 36.5%     | 3.8%   |
| 7.0   | 54.0%   | 36.2%     | 9.8%   |

**Best alpha: 5.0** → 60.6% secure rate (+8.6pp from baseline)

#### Per-Fold Breakdown

| Fold | Dir. Norm | Baseline | Best Rate | Best Alpha | Response |
|------|-----------|----------|-----------|------------|----------|
| admin_delete | 5.41 | 0.0% | 0.0% | — | Resistant |
| log_entry | 5.97 | 82.2% | 93.3% | 7.0 | Weak |
| order_history | 5.87 | 91.1% | 93.3% | 1.0 | Saturated |
| product_search | 5.96 | 71.1% | 97.8% | 5.0 | **Strong** |
| report_filter | 5.76 | 68.9% | 77.8% | 3.0 | Moderate |
| user_login | 5.80 | 48.9% | 84.4% | 7.0 | **Strong** |
| user_profile_update | 5.53 | 2.2% | 4.4% | 5.0 | Resistant |

#### Key Observations
- Higher baseline (52.1%) than smaller models — 70B already generates parameterized queries ~half the time
- Smaller improvement margin (+8.6pp) — less room to move
- Two folds completely resistant: admin_delete (0% at all alphas) and user_profile_update (~2%)
- Direction norms (5.4–6.0) significantly smaller than CWE-787 (9.9–12.0)
- Gradual improvement curve (no sharp transition like CWE-787)

## Cross-Model Comparison

### CWE-787 (Buffer Overflow)

| Model | Params | Quant | Baseline | Best Rate | Best α | Δpp |
|-------|--------|-------|----------|-----------|--------|-----|
| Llama-8B | 8B | fp16 | 6.7% | 73.3% | 4.0 | +66.6 |
| Mistral-7B | 7B | fp16 | 3.8% | 74.3% | 3.0 | +70.5 |
| Qwen-14B | 14B | fp16 | 3.8% | 54.3% | 5.0 | +50.5 |
| **Llama-70B** | **70B** | **4bit** | **1.9%** | **52.4%** | **4.0** | **+50.5** |

### CWE-89 (SQL Injection)

| Model | Params | Quant | Baseline | Best Rate | Best α | Δpp |
|-------|--------|-------|----------|-----------|--------|-----|
| Llama-8B | 8B | fp16 | 57.0% | 70.3% | 5.0 | +13.3 |
| Mistral-7B | 7B | fp16 | 42.9% | 63.5% | 6.0 | +20.6 |
| Qwen-14B | 14B | fp16 | 38.4% | 54.0% | 7.0 | +15.6 |
| **Llama-70B** | **70B** | **4bit** | **52.1%** | **60.6%** | **5.0** | **+8.6** |

## Timing & Infrastructure Notes

### Timing Probe Results (pre-experiment)
- Model load: ~4 min (4-bit NF4)
- Per-prompt generation: 38.5s ± 9.7s at ~10.8 tok/s
- Activation collection: ~0.27s per prompt

### Issues Encountered
1. **8-bit OOM**: `load_in_8bit=True` OOMs on A100-80GB. Switched to 4-bit NF4.
2. **NaN in steering**: Near-zero direction norm (from identical placeholder prompts) caused division by zero. Fixed with random unit vector guard.
3. **dtype mismatch**: `bfloat16` compute dtype + `float16` steering tensor caused CUDA assert. Fixed with `.to(h.dtype)` in hook.
4. **CPU offloading required**: Pure GPU loading OOMs. Need `max_memory={0: "60GiB", "cpu": "60GiB"}`.

## Experiment 19: CWE-119 (General Buffer Overflow) LOBO

### Setup
- **Dataset**: CWE-119 expanded (105 pairs, 7 base_ids)
- **Alpha grid**: Folds 1-2: [0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 5.0]; Folds 3-7: [0.0, 1.0, 1.5]
- **Seeds**: [42, 123, 456] (3 per prompt)
- **Runtime**: ~8h (original run crashed at fold 3, resumed with narrowed grid)
- **Infrastructure issue**: Silent OOM kill required zombie process cleanup and restart

### Results

#### Aggregated (All 7 Folds, 315 total generations on common alphas)

| Alpha | Secure% | Insecure% | Other% |
|-------|---------|-----------|--------|
| 0.0   | 0.0%    | 100%      | 0%     |
| **1.0** | **38.4%** | **13.0%** | **48.6%** |

**Best alpha: 1.0** → 38.4% strict secure rate (+38.4pp from baseline)

Alpha 1.5 (5/7 folds only): 16.9% secure, 0% insecure, 83.1% other

#### Per-Fold Breakdown

| Fold | Dir. Norm | α=0 | α=1.0 | α=1.5 | Response |
|------|-----------|-----|-------|-------|----------|
| pair_01_user_input | 24.7 | 0% | **91.1%** | — | **Strong** |
| pair_02_command_parser | 24.7 | 0% | **88.9%** | — | **Strong** |
| pair_03_config_reader | 24.6 | 0% | **82.2%** | 84.4% | **Strong** |
| pair_04_username_copy | 28.0 | 0% | 6.7% | 0% | Resistant |
| pair_05_filepath_copy | 28.1 | 0% | 0% | 0% | Resistant |
| pair_06_error_msg_copy | 28.2 | 0% | 0% | 0% | Resistant |
| pair_07_hostname_copy | 28.1 | 0% | 0% | 0% | Resistant |

#### Key Observations
- **Bimodal split**: `gets()→fgets()` folds (1-3, norms ~24.7) show strong response (82-91%); `strcpy()→strncpy()` folds (4-7, norms ~28) are completely resistant
- Direction norm ~24.7: effective magnitude at α=1.0 ≈ 25 (in the sweet spot)
- Direction norm ~28: effective magnitude at α=1.0 ≈ 28 (above sweet spot, causes degeneration before secure behavior emerges)
- Cosine similarity between CWE-787 and CWE-119 directions: 0.12 (near orthogonal)
- 70B achieves the best aggregated CWE-119 rate of any model: +38.4pp vs +20pp on Llama-8B

## Experiment 20: E2E Pipeline

### Setup
- **Steering**: CWE-787 direction at α=4.0 (best from LOBO), layer 79
- **Dataset**: 21 neutral evaluation prompts × 10 seeds × 2 conditions = 420 generations
- **CWEs evaluated**: CWE-787, CWE-119, CWE-134 (70 each)
- **Runtime**: ~1h

### Results

| CWE | Baseline | Steered (α=4.0) | Change |
|-----|----------|-----------------|--------|
| CWE-787 | 78.6% | **100.0%** | **+21.4pp** |
| CWE-119 | 81.4% | 61.4% | -20.0pp |
| CWE-134 | 100.0% | 100.0% | — |
| **Overall** | **86.7%** | **87.1%** | **+0.5pp** |

#### Key Observations
- CWE-787 steering achieves perfect 100% secure rate on targeted vulnerability
- CWE-119 hurt by cross-CWE interference (-20pp) — consistent with low cosine similarity (0.12)
- CWE-134 unaffected (already 100% at baseline)
- Llama-8B E2E: 88.6% overall; 70B: 87.1% (-1.5pp from 8B)
- Generation timing: mean 8.3s ± 9.1s per prompt

## Experiment 21: Logit Lens

### Setup
- **Layers analyzed**: All 80 layers (full sweep on static prompts, key layers on dataset)
- **Tokens**: sprintf (ID 12075), snprintf (ID 37546) — both single tokens
- **Prompts**: 5 secure/vulnerable pairs from CWE-787 dataset

### Results

#### Token Emergence Pattern
- Layers 0-74: Both sprintf and snprintf have near-zero probability (ranks >10K)
- Layers 75-79: Differentiation emerges in final 5 layers
- Final layer: P(sprintf|vulnerable) = 0.0133, P(snprintf|secure) = 0.0202

#### Average Probabilities (Secure vs Vulnerable Prompts)
| Layer | Sec P(sprintf) | Vul P(sprintf) | Diff | Sec P(snprintf) | Vul P(snprintf) | Diff |
|-------|------------|------------|------|-------------|-------------|------|
| 0-70 | ~0.000 | ~0.000 | ~0 | ~0.000 | ~0.000 | ~0 |
| 75 | 0.000002 | 0.000004 | -0.000003 | 0.000004 | 0.000001 | +0.000003 |
| 77 | 0.000003 | 0.000006 | -0.000004 | 0.000004 | 0.000002 | +0.000003 |
| 78 | 0.000004 | 0.000018 | -0.000013 | 0.000006 | 0.000002 | +0.000004 |
| 79 | 0.000003 | 0.000015 | -0.000012 | 0.000009 | 0.000004 | +0.000005 |

#### Key Observations
- Consistent with late-layer emergence pattern on Llama-8B (emergence at L31/32) and Mistral-7B
- 70B emergence occurs in final ~5 layers (layers 75-79), proportionally similar to 8B (final 1-2 of 32 layers)
- Both sprintf and snprintf are single tokens (unlike Mistral-7B where snprintf splits to ['sn', 'printf'])
- Probabilities are much smaller than Llama-8B (~37% P(snprintf) at final layer on 8B vs ~2% on 70B) — 4-bit quantization may reduce confidence

## Cross-Model Comparison (All CWEs)

### CWE-787 (Buffer Overflow)

| Model | Params | Quant | Baseline | Best Rate | Best α | Δpp |
|-------|--------|-------|----------|-----------|--------|-----|
| Llama-8B | 8B | fp16 | 6.7% | 73.3% | 4.0 | +66.6 |
| Mistral-7B | 7B | fp16 | 3.8% | 74.3% | 3.0 | +70.5 |
| Qwen-14B | 14B | fp16 | 3.8% | 54.3% | 5.0 | +50.5 |
| **Llama-70B** | **70B** | **4bit** | **1.9%** | **52.4%** | **4.0** | **+50.5** |

### CWE-89 (SQL Injection)

| Model | Params | Quant | Baseline | Best Rate | Best α | Δpp |
|-------|--------|-------|----------|-----------|--------|-----|
| Llama-8B | 8B | fp16 | 57.0% | 70.3% | 5.0 | +13.3 |
| Mistral-7B | 7B | fp16 | 42.9% | 63.5% | 6.0 | +20.6 |
| Qwen-14B | 14B | fp16 | 38.4% | 54.0% | 7.0 | +15.6 |
| **Llama-70B** | **70B** | **4bit** | **52.1%** | **60.6%** | **5.0** | **+8.6** |

### CWE-119 (General Buffer Overflow)

| Model | Params | Quant | Baseline | Best Rate | Best α | Δpp |
|-------|--------|-------|----------|-----------|--------|-----|
| Llama-8B | 8B | fp16 | 0.0% | 20.0% | 4.0 | +20.0 |
| Mistral-7B | 7B | fp16 | 0.3% | 1.6% | 3.0 | +1.3 |
| **Llama-70B** | **70B** | **4bit** | **0.0%** | **38.4%** | **1.0** | **+38.4** |

## Code

- [05_full_lobo.py](../../src/experiments/02-05_cross_model_cwe787_steering/experiment_4b_llama70b/05_full_lobo.py) - CWE-787 LOBO script
- [experiment_config.py](../../src/experiments/02-05_cross_model_cwe787_steering/experiment_4b_llama70b/experiment_config.py) - CWE-787 config
- [01_cwe89_lobo.py](../../src/experiments/02-26_llama70b_full_suite/01_cwe89_lobo.py) - CWE-89 LOBO script
- [02_cwe119_lobo.py](../../src/experiments/02-26_llama70b_full_suite/02_cwe119_lobo.py) - CWE-119 LOBO script (original)
- [02b_cwe119_lobo_resume.py](../../src/experiments/02-26_llama70b_full_suite/02b_cwe119_lobo_resume.py) - CWE-119 LOBO resume (folds 3-7, narrowed grid)
- [03_e2e_pipeline.py](../../src/experiments/02-26_llama70b_full_suite/03_e2e_pipeline.py) - E2E pipeline script
- [04_logit_lens.py](../../src/experiments/02-26_llama70b_full_suite/04_logit_lens.py) - Logit lens script
- [run_all.sh](../../src/experiments/02-26_llama70b_full_suite/run_all.sh) - Sequential runner
- [timing_probe_70b.py](../../src/experiments/timing_probe_70b.py) - Timing probe script
