# Experiment 10b: LOBO Alpha Extension for Python CWEs

**Date**: 2026-02-14 to 2026-02-16
**Model**: meta-llama/Meta-Llama-3.1-8B-Instruct, Layer 31
**GPU**: A100-80GB

## Research Question

Does extending the LOBO alpha sweep beyond alpha=5 yield further improvements for Python CWE steering vectors? What is the relationship between direction norm, alpha, and effective steering magnitude?

## Background

Experiment 10 Step 3 swept alpha in {0, 1, 2, 3, 4, 5} and found alpha=5 was best for all three Python CWEs. The three steering vectors have very different norms:

| Direction | Norm |
|-----------|------|
| Py-CWE-89 | ~2.8 |
| Py-CWE-78 | ~5.0-5.6 |
| Py-CWE-79 | ~7.0-7.2 |

This means the effective steering magnitude (norm x alpha) at alpha=5 varies substantially: CWE-89 is only at ~14, while CWE-79 is already at ~35. This experiment tests whether CWE-89 benefits from higher alphas where its effective magnitude would catch up.

## Methods

- **Protocol**: 7-fold Leave-One-Base-Out cross-validation
- **Alpha grid (new)**: {6, 7, 8, 10, 12, 15}
- **Alpha grid (prior, from Exp 10)**: {0, 1, 2, 3, 4, 5}
- **Prompts per fold**: 15 (insecure-variant prompts)
- **Seeds per prompt**: 10
- **Generations per alpha per CWE**: 1,050 (7 folds x 15 prompts x 10 seeds)
- **Generation config**: temperature=0.6, top_p=0.9, max_new_tokens=512
- **Seeds**: [42, 123, 456, 789, 1000, 1111, 2222, 3333, 4444, 5555]
- **Steering method**: Mean-difference direction, added at Layer 31

### Completion Status

| CWE | Folds Completed | Notes |
|-----|-----------------|-------|
| CWE-89 | 7/7 | Complete |
| CWE-78 | 6/7 | Process killed; pattern conclusive (coherence collapse) |
| CWE-79 | 7/7 | Ran in separate script |

## Results (No Interpretation)

### CWE-89 (SQL Injection) — Direction Norm ~2.8

Full alpha curve (prior + extension), N=1,050 per alpha:

| Alpha | Secure% | Insecure% | Other% | Eff. Magnitude |
|-------|---------|-----------|--------|----------------|
| 0.0 | 57.0% | 42.9% | 0.2% | 0.0 |
| 1.0 | 61.0% | 38.9% | 0.1% | ~2.8 |
| 2.0 | 64.5% | 35.5% | 0.0% | ~5.6 |
| 3.0 | 68.1% | 31.8% | 0.1% | ~8.4 |
| 4.0 | 68.9% | 31.1% | 0.0% | ~11.2 |
| 5.0 | 70.3% | 29.7% | 0.0% | ~14.0 |
| 6.0 | 71.9% | 27.9% | 0.2% | ~16.8 |
| 7.0 | 73.4% | 26.4% | 0.2% | ~19.6 |
| 8.0 | 74.3% | 24.9% | 0.9% | ~22.4 |
| 10.0 | 76.5% | 20.7% | 2.9% | ~28.0 |
| **12.0** | **78.5%** | **15.1%** | **6.4%** | **~33.6** |
| 15.0 | 75.6% | 11.4% | 13.0% | ~42.0 |

- **New best**: alpha=12 at 78.5% secure (+8.2pp over prior best alpha=5 at 70.3%)
- **Improvement continues** from alpha=5 through alpha=12; secure rate still climbing
- **Other rate** stays negligible through alpha=8 (0.9%), rises at alpha=10 (2.9%), alpha=12 (6.4%), and becomes significant at alpha=15 (13.0%)
- **Alpha=15 shows rollover**: secure rate drops to 75.6% as coherence collapse (13.0% other) begins eating into outputs

### CWE-78 (OS Command Injection) — Direction Norm ~5.0-5.6

Full alpha curve (prior + extension). Alpha 0-5: N=1,050. Alpha 6-12: N=900 (6/7 folds):

| Alpha | Secure% | Insecure% | Other% | Eff. Magnitude |
|-------|---------|-----------|--------|----------------|
| 0.0 | 14.3% | 85.5% | 0.2% | 0.0 |
| 1.0 | 16.5% | 83.4% | 0.1% | ~5.0 |
| 2.0 | 18.8% | 81.0% | 0.2% | ~10.0 |
| 3.0 | 20.2% | 79.1% | 0.7% | ~15.0 |
| 4.0 | 21.7% | 77.0% | 1.3% | ~20.0 |
| **5.0** | **22.0%** | **76.3%** | **1.7%** | **~25.0** |
| 6.0 | 22.6% | 67.1% | 10.3% | ~30.0 |
| 7.0 | 20.4% | 41.7% | 37.9% | ~35.0 |
| 8.0 | 14.7% | 18.9% | 66.4% | ~40.0 |
| 10.0 | 3.7% | 0.0% | 96.3% | ~50.0 |
| 12.0 | 0.0% | 0.0% | 100.0% | ~60.0 |

- **Prior alpha=5 remains best** at 22.0% secure (with only 1.7% other)
- Alpha=6 shows marginal secure gain (22.6%) but 10.3% other rate -- not worth the coherence cost
- **Coherence collapse** begins at alpha=7 (37.9% other), accelerates rapidly
- By alpha=10, 96.3% other; by alpha=12, 100% other (complete collapse)
- Process killed before alpha=15; pattern was conclusive

### CWE-79 (Cross-Site Scripting) — Direction Norm ~7.0-7.2

Full alpha curve (prior + extension), N=1,050 per alpha:

| Alpha | Secure% | Insecure% | Other% | Eff. Magnitude |
|-------|---------|-----------|--------|----------------|
| 0.0 | 0.2% | 98.4% | 1.4% | 0.0 |
| 1.0 | 0.5% | 97.3% | 2.2% | ~7.1 |
| 2.0 | 1.7% | 94.8% | 3.5% | ~14.2 |
| 3.0 | 10.7% | 85.6% | 3.7% | ~21.3 |
| 4.0 | 19.3% | 66.9% | 13.8% | ~28.4 |
| **5.0** | **30.5%** | **34.0%** | **35.5%** | **~35.5** |
| 6.0 | 27.8% | 12.2% | 60.0% | ~42.6 |
| 7.0 | 14.7% | 1.5% | 83.8% | ~49.7 |
| 8.0 | 4.6% | 0.1% | 95.3% | ~56.8 |
| 10.0 | 0.0% | 0.0% | 100.0% | ~71.0 |
| 12.0 | 0.0% | 0.0% | 100.0% | ~85.2 |
| 15.0 | 0.0% | 0.0% | 100.0% | ~106.5 |

- **Prior alpha=5 remains best** at 30.5% secure
- Alpha=6 already worse (27.8% secure, 60.0% other)
- Rapid collapse: alpha=8 has 95.3% other, alpha>=10 is 100% other
- CWE-79's large direction norm means it hits the effective magnitude ceiling earlier

## Summary: Optimal Alpha by CWE

| CWE | Direction Norm | Best Alpha | Best Secure% | Eff. Magnitude at Best | Other% at Best |
|-----|---------------|------------|--------------|----------------------|----------------|
| CWE-89 | ~2.8 | 12.0 | 78.5% | ~33.6 | 6.4% |
| CWE-78 | ~5.0 | 5.0 | 22.0% | ~25.0 | 1.7% |
| CWE-79 | ~7.1 | 5.0 | 30.5% | ~35.5 | 35.5% |

### Key Finding: Effective Steering Magnitude Sweet Spot

The effective steering magnitude (direction_norm x alpha) appears to have a sweet spot around ~30-35:

- **CWE-89** reaches its optimum at effective magnitude ~33.6 (2.8 x 12)
- **CWE-78** at alpha=5 has effective magnitude ~25.0 (still below the sweet spot, but coherence collapse from its larger norm prevents reaching it)
- **CWE-79** at alpha=5 has effective magnitude ~35.5 (right at the sweet spot, already with 35.5% other)

This explains why CWE-89 benefits from much higher alphas: its small direction norm means it needs a higher multiplier to reach the effective magnitude range where steering is most impactful. The larger-norm vectors (CWE-78, CWE-79) hit coherence collapse before or right at this sweet spot.

## Data Verification

- Datasets and scorers were triple-checked
- The "other" rate represents genuine coherence/domain collapse (incoherent, off-topic, or degenerate outputs), not scorer bugs
- CWE-89/78 results parsed from process output logs (process killed mid-CWE-78)
- CWE-79 results from dedicated re-run with full completion

## Code / Files Generated

### Scripts
- [03b_lobo_alpha_extension.py](../../src/experiments/02-10_python_cwe_steering/03b_lobo_alpha_extension.py) - Main alpha extension script (CWE-89 and CWE-78)
- [03c_lobo_alpha_cwe79_only.py](../../src/experiments/02-10_python_cwe_steering/03c_lobo_alpha_cwe79_only.py) - CWE-79 dedicated run
- [parse_alpha_extension_output.py](../../src/experiments/02-10_python_cwe_steering/parse_alpha_extension_output.py) - Output parser for CWE-89/78 results from logs

### Result Files
- [alpha_extension_partial_89_78.json](../../src/experiments/02-10_python_cwe_steering/results/alpha_extension_partial_89_78.json) - CWE-89/78 fold-level data
- [alpha_extension_results_partial.json](../../src/experiments/02-10_python_cwe_steering/results/alpha_extension_results_partial.json) - CWE-89/78 aggregated + merged with prior alphas
- [alpha_extension_full_20260215_015309.json](../../src/experiments/02-10_python_cwe_steering/results/alpha_extension_full_20260215_015309.json) - CWE-79 fold-level data
- [alpha_extension_results_20260215_015309.json](../../src/experiments/02-10_python_cwe_steering/results/alpha_extension_results_20260215_015309.json) - CWE-79 summary
- [alpha_curve_merged_20260215_015309.json](../../src/experiments/02-10_python_cwe_steering/results/alpha_curve_merged_20260215_015309.json) - All CWEs merged alpha curves (0-15)
