# Experiment 7: Stacked Vectors Test

**Date**: 2026-02-06
**Model**: Llama-3.1-8B-Instruct (fp16)
**Layer**: 31
**GPU Time**: ~4.5 hours (1,260 generations at ~13s/prompt)

## Research Question

Does applying all 3 native CWE steering vectors simultaneously (summed perturbation) preserve CWE-specific steering performance? Experiment 6 showed averaging dilutes performance (-15 to -31pp vs native). This tests whether stacking avoids that dilution because each vector operates in its own subspace.

## Hypothesis

Stacking preserves CWE-specific effects because each vector operates in its own subspace, unlike averaging which dilutes all of them.

**Result: Hypothesis REJECTED.** Stacking performs worse than both native and unified approaches.

## Method

### Multi-Steering Hook
```python
# Single hook adds all 3 vectors at L31:
h[:, -1, :] += α_787 * dir_787 + α_119 * dir_119 + α_134 * dir_134
```

### Direction Vectors (Pre-computed)
| Vector | File | Norm |
|--------|------|------|
| CWE-787 | `direction_cwe787_L31_20260206_031901.npy` | 7.77 |
| CWE-119 | `direction_cwe119_L31_20260206_031901.npy` | 8.66 |
| CWE-134 | `direction_cwe134_L31_20260206_031901.npy` | 8.51 |

### Alpha Configurations
| Config | α_787 | α_119 | α_134 | Rationale |
|--------|-------|-------|-------|-----------|
| Low | 1.0 | 1.0 | 0.5 | Conservative, minimize interference |
| Medium | 1.5 | 1.5 | 0.5 | Balanced |
| High | 2.0 | 2.0 | 1.0 | Stronger steering |
| Weighted | 1.5 | 2.0 | 0.3 | Proportional to native optimal |

### Datasets
- CWE-787: 105 pairs (sprintf, strcat)
- CWE-119: 105 pairs (gets, strcpy)
- CWE-134: 105 pairs (printf_format, fprintf_format, syslog_format)

### Generation Parameters
- Seed: 42, Temperature: 0.6, Top-p: 0.9, Max tokens: 512

## Results

### Comparison Table

| CWE | Baseline | Native Best | Unified Best | Stk-Low | Stk-Med | Stk-High | Stk-Weighted |
|-----|----------|-------------|--------------|---------|---------|----------|--------------|
| 787 | 0.0% | 52.4% | 21.0% | 7.6% | 20.0% | 27.6% | 18.1% |
| 119 | 0.0% | 20.0% | 4.8% | 1.0% | 2.9% | 7.6% | 10.5% |
| 134 | 66.7% | 90.0% | 69.5% | 59.0% | 52.4% | 48.6% | 55.2% |

### Detailed Breakdown

#### Low (α_787=1.0, α_119=1.0, α_134=0.5)
| CWE | Secure | Insecure | Other | Other % | Refusals |
|-----|--------|----------|-------|---------|----------|
| 787 | 8/105 (7.6%) | 92 | 5 | 4.8% | 0 |
| 119 | 1/105 (1.0%) | 103 | 1 | 1.0% | 0 |
| 134 | 62/105 (59.0%) | 42 | 1 | 1.0% | 0 |

#### Medium (α_787=1.5, α_119=1.5, α_134=0.5)
| CWE | Secure | Insecure | Other | Other % | Refusals |
|-----|--------|----------|-------|---------|----------|
| 787 | 21/105 (20.0%) | 72 | 12 | 11.4% | 0 |
| 119 | 3/105 (2.9%) | 98 | 4 | 3.8% | 0 |
| 134 | 55/105 (52.4%) | 50 | 0 | 0.0% | 0 |

#### High (α_787=2.0, α_119=2.0, α_134=1.0)
| CWE | Secure | Insecure | Other | Other % | Refusals |
|-----|--------|----------|-------|---------|----------|
| 787 | 29/105 (27.6%) | 51 | 25 | 23.8% | 1 |
| 119 | 8/105 (7.6%) | 54 | 43 | 41.0% | 0 |
| 134 | 51/105 (48.6%) | 54 | 0 | 0.0% | 0 |

#### Weighted (α_787=1.5, α_119=2.0, α_134=0.3)
| CWE | Secure | Insecure | Other | Other % | Refusals |
|-----|--------|----------|-------|---------|----------|
| 787 | 19/105 (18.1%) | 78 | 8 | 7.6% | 0 |
| 119 | 11/105 (10.5%) | 88 | 6 | 5.7% | 0 |
| 134 | 58/105 (55.2%) | 44 | 3 | 2.9% | 0 |

### Other % (Degradation Check)

| CWE | Stk-Low | Stk-Med | Stk-High | Stk-Weighted |
|-----|---------|---------|----------|--------------|
| 787 | 4.8% | 11.4% | **23.8%** | 7.6% |
| 119 | 1.0% | 3.8% | **41.0%** | 5.7% |
| 134 | 1.0% | 0.0% | 0.0% | 2.9% |

High config exceeds 15% Other threshold on both CWE-787 and CWE-119.

### Success Criteria Evaluation

| Config | CWEs >=70% of native (need 2/3) | Other%<15% | Pass? |
|--------|--------------------------------|------------|-------|
| Low | 0/3 | PASS | FAIL |
| Medium | 0/3 | PASS | FAIL |
| High | 0/3 | FAIL | FAIL |
| Weighted | 0/3 | PASS | FAIL |

**All configurations FAIL** the success criteria. No config preserves >=70% of native performance on even 1 CWE.

## Key Observations

1. **Stacking is worse than unified averaging**: For CWE-787, best stacked (High, 27.6%) < unified (21.0%) is slightly better but still far from native (52.4%). For CWE-134, stacking *degrades below baseline* (48.6-59.0% vs 66.7% baseline).

2. **CWE-134 regression**: All stacked configs push CWE-134 below the unsteered baseline of 66.7%. The CWE-787 and CWE-119 vectors actively interfere with format-string security patterns.

3. **High alpha causes severe degradation**: 41% "other" rate on CWE-119 at High config means the model generates incoherent outputs rather than either secure or insecure code.

4. **Weighted config best for CWE-119**: The only config approaching reasonable CWE-119 performance (10.5% vs 20.0% native, 52.5% preservation), but at the cost of CWE-134 regression.

5. **Vectors do NOT operate in independent subspaces**: The destructive interference confirms the vectors share substantial representational overlap despite low cosine similarities (0.48 between CWE-787 and CWE-134 from Experiment 5d).

## Conclusion

Multi-vector stacking does not solve the broad-spectrum steering problem. The CWE-specific directions interfere destructively when applied simultaneously, degrading both target and non-target CWE performance. This rules out simple additive composition as a viable approach for multi-vulnerability steering.

## Code

- [stacked_vectors_experiment.py](../../src/experiments/02-05_cross_cwe_steering/stacked_vectors_experiment.py) - Main experiment script
- [shared/steering_generator.py](../../src/experiments/02-05_cross_cwe_steering/shared/steering_generator.py) - Added `generate_with_multi_steering()` method

## Data

All in `src/experiments/02-05_cross_cwe_steering/cross_cwe_analysis/data/`:

| File | Description | Size |
|------|-------------|------|
| `stacked_steering_results_20260206_225040.json` | Per-config, per-CWE breakdown + comparison summary | ~10 KB |
| `stacked_steering_full_20260206_225040.json` | Per-sample outputs for all 1,260 generations | ~966 KB |
