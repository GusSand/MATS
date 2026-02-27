# Experiment 25: Functional Correctness of Steered Code

**Date**: 2026-02-27
**Models evaluated**: Mistral-7B-Instruct-v0.3, Llama-3.1-8B-Instruct
**Judge**: GPT-4o (openai/gpt-4o-2024-05-13) via OpenRouter
**CWE**: CWE-787 (Out-of-bounds Write)

## Research Question

Does activation steering degrade functional correctness of generated code? Is the security improvement coming at the cost of broken code?

## Methods

### Setup
- **Judge model**: GPT-4o via OpenRouter API (temperature=0.0, max_tokens=10)
- **Steering alpha**: α=3.5 (best alpha for both models from previous LOBO experiments)
- **Baseline alpha**: α=0.0 (no steering)
- **Samples**: 25 per condition × 4 conditions = 100 total evaluations
- **Random seed**: 42 (with offsets +1, +2, +3 per condition)
- **Source data**: Raw outputs from LOBO fold result files

### Data Sources
- **Mistral-7B**: `experiment_4a_mistral7b/data/fold_results/fold_pair_*_20260205_045755.json` (105 outputs per alpha)
- **Llama-8B**: `01-12_llama8b_cwe787_lobo_steering/data/fold_results/fold_pair_*_20260113_171820.json` (105 outputs per alpha)

### Rating Scale
- **CORRECT**: Code would compile and correctly implement the intended function
- **PARTIALLY_CORRECT**: Code has minor issues but core logic is sound
- **INCORRECT**: Code has significant logical errors or would not compile
- **INCOMPLETE**: Code is truncated or missing essential parts

### Known Confound
Llama-8B outputs were capped at 500 characters in stored results. This inflates INCOMPLETE ratings, particularly for steered outputs if steering produces longer code. Average output lengths:
- Mistral-7B steered: 419 chars
- Mistral-7B baseline: 467 chars
- Llama-8B steered: 500 chars (capped)
- Llama-8B baseline: 500 chars (capped)

## Results

### Per-Condition Breakdown

| Condition | N | CORRECT | PARTIAL | INCORRECT | INCOMPLETE | Functional% |
|-----------|---|---------|---------|-----------|------------|-------------|
| Mistral-7B steered (α=3.5) | 25 | 52.0% | 0.0% | 24.0% | 24.0% | **52.0%** |
| Mistral-7B baseline (α=0.0) | 25 | 36.0% | 8.0% | 16.0% | 40.0% | **44.0%** |
| Llama-8B steered (α=3.5) | 25 | 4.0% | 4.0% | 16.0% | 76.0% | **8.0%** |
| Llama-8B baseline (α=0.0) | 25 | 32.0% | 4.0% | 0.0% | 64.0% | **36.0%** |

### 2×2 Summary (Functional = CORRECT + PARTIALLY_CORRECT)

| Model | Baseline | Steered | Diff |
|-------|----------|---------|------|
| Mistral-7B | 44.0% | 52.0% | **+8.0pp** |
| Llama-8B | 36.0% | 8.0% | **-28.0pp** |

### Key Observations (No Interpretation)
1. Mistral-7B steered outputs have a *higher* functional correctness rate than baseline (+8pp)
2. Llama-8B steered outputs show a large drop (-28pp), but 76% of steered outputs are INCOMPLETE vs 64% baseline — the 500-char truncation confound heavily affects this comparison
3. Mistral-7B INCORRECT rate increases with steering (24% vs 16%), but INCOMPLETE rate *decreases* (24% vs 40%), suggesting steering may produce more complete but sometimes wrong code
4. Llama-8B baseline has 0% INCORRECT (all non-incomplete outputs are correct or partial), while steered has 16% INCORRECT

## Interpretation (Analyst)

**Mistral-7B**: Steering does NOT degrade functional correctness. The +8pp improvement suggests that steering toward secure patterns (snprintf) doesn't break code logic. This is an encouraging result — security comes "for free" in terms of correctness.

**Llama-8B**: Results are heavily confounded by the 500-char output truncation. With 76% INCOMPLETE on steered outputs, we cannot draw reliable conclusions about functional correctness. The apparent -28pp drop may be an artifact of steering producing longer outputs that get truncated. A re-evaluation with full (untruncated) outputs would be needed to draw valid conclusions.

## Files

### Code
- [01_evaluate_correctness.py](../../src/experiments/02-27_functional_correctness/01_evaluate_correctness.py) - GPT-4o judge evaluation script

### Results
- [correctness_results_20260227_174107.json](../../src/experiments/02-27_functional_correctness/results/correctness_results_20260227_174107.json) - Full results with per-output verdicts

---

## Experiment 25b: Llama-8B Re-evaluation (Untruncated)

To address the truncation confound, Exp 25b regenerated Llama-8B outputs fresh from the model with FULL outputs (no 500-char cap).

### Methods
- Re-extracted activations at L31 for all 210 samples
- Computed 7 LOBO fold-specific directions
- Regenerated 25 steered (α=3.5) + 25 baseline (α=0.0) outputs using same prompt IDs as Exp 25
- `max_new_tokens=512`, outputs stored in full (avg steered: 2432 chars, avg baseline: 2171 chars)
- All 50 outputs exceeded 500 chars — confirming Exp 25's truncation was the problem

### Exp 25b Results

| Condition | N | CORRECT | PARTIAL | INCORRECT | INCOMPLETE | Functional% |
|-----------|---|---------|---------|-----------|------------|-------------|
| Steered (α=3.5) | 25 | 24.0% | 0.0% | 36.0% | 40.0% | **24.0%** |
| Baseline (α=0.0) | 25 | 56.0% | 4.0% | 12.0% | 28.0% | **60.0%** |

### Comparison: Exp 25 vs Exp 25b

| Condition | Exp 25 (truncated) | Exp 25b (full) | Change |
|-----------|-------------------|----------------|--------|
| Steered functional | 8.0% | 24.0% | +16pp |
| Baseline functional | 36.0% | 60.0% | +24pp |
| Steered−Baseline | -28pp | **-36pp** | — |

### Interpretation (Analyst)

Truncation WAS a significant confound: both conditions improved substantially with full outputs (steered +16pp, baseline +24pp). However, the steering penalty is actually LARGER than Exp 25 suggested: -36pp vs -28pp. Steering clearly degrades Llama-8B's functional correctness.

Notably, the INCOMPLETE rate remains substantial even with full outputs (40% steered, 28% baseline). This may reflect the model generating multiple attempts/explanations that GPT-4o judges as incomplete functions.

### Combined Cross-Model Summary

| Model | Baseline | Steered | Diff | Notes |
|-------|----------|---------|------|-------|
| Mistral-7B (Exp 25) | 44.0% | 52.0% | **+8pp** | Steering preserves/improves correctness |
| Llama-8B (Exp 25b) | 60.0% | 24.0% | **-36pp** | Steering degrades correctness |

### Exp 25b Files
- [02_exp25b_llama8b_rerun.py](../../src/experiments/02-27_functional_correctness/02_exp25b_llama8b_rerun.py) - Full regeneration + evaluation script
- [exp25b_correctness_results_20260227_180052.json](../../src/experiments/02-27_functional_correctness/results/exp25b_correctness_results_20260227_180052.json) - Full results

## Reproducibility

```bash
cd src/experiments/02-27_functional_correctness
# Exp 25 (from stored outputs)
python 01_evaluate_correctness.py
# Exp 25b (regenerate Llama-8B fresh)
python 02_exp25b_llama8b_rerun.py
```

Requires: OpenRouter API key set in scripts, `openai` Python package, GPU for Exp 25b.
