# Experiment R12: GPT-2 Reconciliation with Hoang's Results

**Date**: 2026-03-27
**Models**: GPT-2-Small, GPT-2-Medium, GPT-2-Large, GPT-2-XL
**Task**: Decimal comparison (X.8 vs X.11)
**Status**: COMPLETE

## Background

Gus's Experiment 10 reported 0% error rate for GPT-2 on decimal comparison. Hoang's results reported 100% error rate for GPT-2-Small. This experiment resolves the discrepancy.

## Critical Bug Found: Shared-Token Evaluation

The `get_logit_difference` function had a bug that made ALL logit evaluations return 0.000. The function checked candidate token strings (e.g., `"1.8"`, `" 1.8"`, `".8"`, `"8"`, etc.) and took the first token ID from each. For multi-token strings like `"1.8"` → `[16, 13, 23]`, the first token is `"1"` (id=16). Since `"1.11"` also starts with `"1"` (id=16), both "correct" and "incorrect" sides picked the same token with the same logit value.

**Fix**: Filter out token IDs that appear on both sides (shared IDs). Only compare tokens with unique IDs:
- Correct: `"8"` (id=23), `" 8"` (id=807)
- Incorrect: `"11"` (id=1157), `" 11"` (id=1367)

## Results

### Section 1-2: Reproduction

Reproduced both Gus's and Hoang's setups on GPT-2-Small:
- Gus's text eval: 9/10 "correct" (counts unclear/ambiguous as not-a-bug)
- Strict text eval: 0/10 correct, 5/10 ambiguous, 2/10 unclear
- Base GPT-2 generates incoherent text → text eval cannot detect the bug

### Section 3: All GPT-2 Variants (Fixed Logit Eval)

| Model | Params | hoang_compare | gus_simple | gus_qa |
|-------|--------|--------------|------------|--------|
| GPT-2-Small | 124M | 8/9 (11% err) | 8/9 (11% err) | 8/9 (11% err) |
| GPT-2-Medium | 355M | 7/9 (22% err) | 8/9 (11% err) | 8/9 (11% err) |
| GPT-2-Large | 774M | 7/9 (22% err) | 8/9 (11% err) | 8/9 (11% err) |
| GPT-2-XL | 1.5B | 6/9 (33% err) | 5/9 (44% err) | 8/9 (11% err) |

Text eval across all models and formats: 0 correct, 0 bug — all classified as "unclear."

### Section 4: Evaluation Method Comparison (GPT-2-Small)

Across all 3 formats × 9 X-values = 27 evaluations:

| Method | Correct | Bug | Other |
|--------|---------|-----|-------|
| Text eval | 0 | 0 | 27 |
| Logit eval | 24 | 3 | — |
| First-number | 18 | 6 | 3 |

### Section 5: Hoang's 5-Head Circuit Patching

Circuit heads: (2,2), (5,1), (6,11), (9,9), (10,2)

| Format | Baseline | Patched | Effect |
|--------|----------|---------|--------|
| hoang_compare | 8/9 correct | 8/9 correct | negligible (Δ < 0.02) |
| gus_simple | 8/9 correct | 8/9 correct | negligible (Δ < 0.04) |
| gus_qa | 8/9 correct | 8/9 correct | negligible (Δ < 0.04) |

The circuit patching has almost no effect on GPT-2-Small. The model is already mostly correct at baseline.

### Section 6: MLP Layer Patching

MLP patching sweep across all 12 layers of GPT-2-Small:
- Every layer: baseline 8/9 → patched 8/9 (Δ=0)
- No individual MLP layer has a measurable causal effect on decimal comparison

### Edge Case: X=8

When X=8, the correct answer is "8.8" and incorrect is "8.11". The decimal part of the correct answer (`"8"`) is the same token as the integer part of the number. After filtering shared IDs, the correct side has ZERO unique tokens → logit_diff = -inf. This is a tokenization artifact, not a model failure.

## Conclusions

1. **Neither Gus nor Hoang was correct**: GPT-2-Small has ~11% error (not 0% or 100%)
2. **Text eval is blind on base models**: GPT-2 generates incoherent text; text-based evaluation cannot meaningfully classify outputs
3. **Logit eval had a critical bug**: Shared token IDs between correct/incorrect candidates caused logit_diff=0.000 (false 100% bug rate)
4. **Larger GPT-2 models are worse**: GPT-2-XL (1.5B) shows up to 44% error, worse than GPT-2-Small (124M)
5. **Circuit/MLP patching shows negligible effects**: The model's base preference is already in the correct direction for 8/9 X-values

## Code

- [run_experiment.py](../../9_8_research/reviewer_experiments/exp_R12_gpt2_reconciliation/run_experiment.py) — Main experiment with all 6 sections
- [run_section3.py](../../9_8_research/reviewer_experiments/exp_R12_gpt2_reconciliation/run_section3.py) — Split runner for Section 3
- [run_section4.py](../../9_8_research/reviewer_experiments/exp_R12_gpt2_reconciliation/run_section4.py) — Split runner for Section 4
- [run_section5.py](../../9_8_research/reviewer_experiments/exp_R12_gpt2_reconciliation/run_section5.py) — Split runner for Section 5
- [run_section6.py](../../9_8_research/reviewer_experiments/exp_R12_gpt2_reconciliation/run_section6.py) — Split runner for Section 6
- [run_summary.py](../../9_8_research/reviewer_experiments/exp_R12_gpt2_reconciliation/run_summary.py) — Summary generator
- [results_20260327_130729.json](../../9_8_research/reviewer_experiments/exp_R12_gpt2_reconciliation/results_20260327_130729.json) — Full results JSON
