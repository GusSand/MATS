# Experiment 9: Cross-Model Neutral Evaluation

**Date**: 2026-02-07
**Models**: Mistral-7B-Instruct-v0.3, Qwen2.5-14B-Instruct (+ Llama-3.1-8B-Instruct reference)
**Dataset**: 21 neutral prompts (7/CWE), 10 seeds each = 210 generations per model per condition

## Research Question

Does the "instruction resistance gap" (neutral_steered - adversarial_steered rates) found in Llama-8B generalize across model architectures? Are CWE-specific trends (CWE-134 easy, CWE-119 hard) universal?

## Configuration

### Models
| Model | Layer | Hidden Dim | CWE-787 α | CWE-119 α Grid | CWE-134 α Grid |
|-------|-------|-----------|-----------|----------------|----------------|
| Mistral-7B-Instruct-v0.3 | 31 | 4096 | 3.5 | [3.0, 3.5, 4.0] | [3.0, 3.5, 4.0] |
| Qwen2.5-14B-Instruct | 47 | 5120 | 4.0 | [3.0, 3.5, 4.0] | [3.0, 3.5, 4.0] |

### Generation Parameters
- temperature=0.6, top_p=0.9, max_new_tokens=512, do_sample=True
- Seeds: [42, 123, 456, 789, 1000, 1111, 2222, 3333, 4444, 5555]

### Steering Vectors
Direction = mean(secure_activations) - mean(vulnerable_activations) at last token position.

**Mistral-7B (Layer 31):**
| CWE | Vector Norm | Source |
|-----|------------|--------|
| CWE-787 | 3.90 | Stored NPZ from Exp 4a |
| CWE-119 | 5.38 | Forward passes on 40 pairs |
| CWE-134 | 3.72 | Forward passes on 40 pairs |

Cross-CWE cosine similarities: 787↔119=0.63, 787↔134=0.47, 119↔134=0.52

**Qwen-14B (Layer 47):**
| CWE | Vector Norm | Source |
|-----|------------|--------|
| CWE-787 | 88.86 | Stored NPZ from Exp 4c |
| CWE-119 | 235.09 | Forward passes on 40 pairs |
| CWE-134 | 148.09 | Forward passes on 40 pairs |

Cross-CWE cosine similarities: 787↔119=0.33, 787↔134=0.43, 119↔134=0.62

## Results

### Phase 2: Neutral Baseline (No Steering)

| Model | CWE-787 | CWE-119 | CWE-134 | Avg |
|-------|---------|---------|---------|-----|
| Llama-8B | 47.1% | 65.0% | 100.0% | 70.7% |
| Mistral-7B | 75.7% | 90.0% | 100.0% | 88.6% |
| Qwen-14B | 78.6% | 100.0% | 100.0% | 92.9% |

Observation: Larger/more capable models have higher neutral baselines. CWE-134 = 100% for all models.

### Phase 3: Neutral + Per-CWE Steering

**Best Alpha Per CWE:**

| Model | CWE-787 (α, rate) | CWE-119 (α, rate) | CWE-134 (α, rate) | Avg |
|-------|-------------------|-------------------|-------------------|-----|
| Llama-8B | 3.5, 100.0% | 4.0, 81.4% | 1.5, 100.0% | 93.8% |
| Mistral-7B | 3.5, 98.6% | 3.0, 75.7% | 3.0, 100.0% | 91.4% |
| Qwen-14B | 4.0, 100.0% | 3.0, 81.4% | 3.0, 100.0% | 93.8% |

**Qwen-14B CWE-119 Alpha Sweep (overshoot observed):**
| α | Secure Rate | Change from Baseline |
|---|------------|---------------------|
| 3.0 | 81.4% (57/70) | -18.6pp |
| 3.5 | 64.3% (45/70) | -35.7pp |
| 4.0 | 51.4% (36/70) | -48.6pp |

CWE-119 steering on Qwen-14B shows severe alpha overshoot — performance degrades monotonically with alpha.

**Mistral-7B CWE-119 Alpha Sweep:**
| α | Secure Rate | Change from Baseline |
|---|------------|---------------------|
| 3.0 | 75.7% (53/70) | -14.3pp |
| 3.5 | 51.4% (36/70) | -38.6pp |
| 4.0 | 42.9% (30/70) | -47.1pp |

Same overshoot pattern — both models degrade on CWE-119 with higher alpha.

### Phase 4: Cross-CWE Interference Check

**Mistral-7B:**
| Steering → Prompt CWE | Rate | Baseline | Delta |
|----------------------|------|----------|-------|
| CWE-787 → CWE-119 | 83.3% | 90.0% | **-6.7pp** |
| CWE-787 → CWE-134 | 100.0% | 100.0% | +0.0pp |
| CWE-119 → CWE-787 | 83.3% | 75.7% | +7.6pp |
| CWE-119 → CWE-134 | 100.0% | 100.0% | +0.0pp |
| CWE-134 → CWE-787 | 70.0% | 75.7% | **-5.7pp** |
| CWE-134 → CWE-119 | 100.0% | 90.0% | +10.0pp |

**Qwen-14B:**
| Steering → Prompt CWE | Rate | Baseline | Delta |
|----------------------|------|----------|-------|
| CWE-787 → CWE-119 | 100.0% | 100.0% | +0.0pp |
| CWE-787 → CWE-134 | 100.0% | 100.0% | +0.0pp |
| CWE-119 → CWE-787 | 80.0% | 78.6% | +1.4pp |
| CWE-119 → CWE-134 | 100.0% | 100.0% | +0.0pp |
| CWE-134 → CWE-787 | 70.0% | 78.6% | **-8.6pp** |
| CWE-134 → CWE-119 | 100.0% | 100.0% | +0.0pp |

Cross-model pattern: CWE-134→CWE-787 shows degradation on both models (-5.7pp Mistral, -8.6pp Qwen).

### Phase 5: Instruction Resistance Gap

| Model | Neutral Steered | Adversarial Steered | Gap |
|-------|----------------|--------------------|----|
| Llama-8B | 100.0% | 52.4% | **+47.6pp** |
| Qwen-14B | 100.0% | 77.1% | **+22.9pp** |
| Mistral-7B | 98.6% | 92.4% | **+6.2pp** |

(CWE-787 only — no adversarial data for CWE-119/134 on Mistral/Qwen)

## Hypothesis Evaluation

1. **H1: Instruction resistance gap is architecture-dependent** — **CONFIRMED**
   - Gap ranges from +6.2pp (Mistral) to +47.6pp (Llama), a 7.7x range
   - Mistral's small gap (+6.2pp) suggests its instruction-following circuits resist steering less

2. **H2: CWE-134 neutral baselines high across models** — **CONFIRMED**
   - 100.0% for all three models
   - printf format-string best practices are deeply encoded in all models tested

3. **H3: CWE-119 remains hardest to steer** — **CONFIRMED**
   - Lowest steered rate in all 3 models (81.4%, 75.7%, 81.4%)
   - Alpha overshoot on CWE-119 is a cross-model phenomenon

4. **H4: No cross-CWE interference** — **PARTIAL**
   - CWE-134→CWE-787 degrades on both Mistral (-5.7pp) and Qwen (-8.6pp)
   - CWE-787→CWE-119 degrades on Mistral only (-6.7pp)
   - Generally interference is mild (< 10pp)

## Key Observations

1. **Baseline security increases with model capability**: Llama avg=70.7%, Mistral avg=88.6%, Qwen avg=92.9%
2. **Steering ceiling is universal**: All models converge to ~91-94% average steered rate despite different baselines
3. **CWE-119 overshoot is universal**: Higher alpha hurts CWE-119 performance on all models. The gets→fgets pattern may require more nuanced intervention
4. **CWE-134→CWE-787 interference is consistent**: Both Mistral and Qwen show degradation when CWE-134 vector applied to CWE-787 prompts, suggesting shared representational space

## Code

- [01_extract_vectors.py](../../src/experiments/02-09_cross_model_neutral_eval/01_extract_vectors.py) - Phase 1: Extract per-CWE steering vectors for both models
- [02_neutral_eval.py](../../src/experiments/02-09_cross_model_neutral_eval/02_neutral_eval.py) - Phases 2-4: Baseline + steering + cross-CWE (per model)
- [03_analysis.py](../../src/experiments/02-09_cross_model_neutral_eval/03_analysis.py) - Phase 5: Cross-model comparison tables

## Files Generated

### Steering Vectors (timestamp: 20260207_202621)
- `mistral7b/data/direction_cwe787_L31_20260207_202621.npy`
- `mistral7b/data/direction_cwe119_L31_20260207_202621.npy`
- `mistral7b/data/direction_cwe134_L31_20260207_202621.npy`
- `mistral7b/data/vector_metadata_20260207_202621.json`
- `qwen14b/data/direction_cwe787_L47_20260207_202621.npy`
- `qwen14b/data/direction_cwe119_L47_20260207_202621.npy`
- `qwen14b/data/direction_cwe134_L47_20260207_202621.npy`
- `qwen14b/data/vector_metadata_20260207_202621.json`

### Mistral-7B Results (timestamp: 20260207_202836)
- `mistral7b/data/neutral_baseline_results_20260207_202836.json`
- `mistral7b/data/neutral_steered_results_20260207_202836.json`
- `mistral7b/data/neutral_steered_full_20260207_202836.json`
- `mistral7b/data/cross_cwe_sanity_check_20260207_202836.json`

### Qwen-14B Results (timestamp: 20260207_210947)
- `qwen14b/data/neutral_baseline_results_20260207_210947.json`
- `qwen14b/data/neutral_steered_results_20260207_210947.json`
- `qwen14b/data/neutral_steered_full_20260207_210947.json`
- `qwen14b/data/cross_cwe_sanity_check_20260207_210947.json`

### Analysis Output
- `cross_model_comparison.md` - Cross-model comparison tables
- `mistral7b/analysis/summary_table.md` - Mistral per-alpha detail
- `qwen14b/analysis/summary_table.md` - Qwen per-alpha detail
