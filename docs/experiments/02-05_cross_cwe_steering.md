# Experiment 5: Cross-CWE Steering Validation (CWE-119, CWE-134)

**Date**: 2026-02-05/06
**Model**: Llama-3.1-8B-Instruct
**Datasets**: CWE-119 (105 pairs), CWE-134 (105 pairs)
**Reference**: CWE-787 LOBO results (0% → 52.4% at α=3.5)

## Overview

This experiment tests whether mean-difference activation steering generalizes across different CWE types. We created new datasets for CWE-119 (buffer operations) and CWE-134 (format strings) to evaluate if steering vectors from one vulnerability class can improve secure code generation for related but distinct vulnerability patterns.

## Research Question

Does mean-difference activation steering for secure code generation generalize across CWE types?
1. CWE-119: Improper Restriction of Operations within Memory Buffer Bounds
2. CWE-134: Use of Externally-Controlled Format String

## Datasets

### CWE-119 Dataset

7 validated prompt pairs targeting buffer operations without bounds checking:

| Short Name      | Vulnerability | Context                    |
|-----------------|---------------|----------------------------|
| user_input      | gets          | Read line from stdin       |
| command_parser  | gets          | CLI command input          |
| config_reader   | gets          | Configuration line input   |
| username_copy   | strcpy        | Username string copy       |
| filepath_copy   | strcpy        | File path string copy      |
| error_msg_copy  | strcpy        | Error message copy         |
| hostname_copy   | strcpy        | Hostname string copy       |

**Secure patterns**: `fgets()`, `strncpy()`
**Insecure patterns**: `gets()`, `strcpy()`

### CWE-134 Dataset

7 validated prompt pairs targeting format string vulnerabilities:

| Short Name      | Vulnerability    | Context                     |
|-----------------|------------------|-----------------------------|
| print_message   | printf_format    | Print user message          |
| print_status    | printf_format    | Display server status       |
| print_error     | printf_format    | Show error description      |
| log_to_file     | fprintf_format   | Log message to file         |
| write_report    | fprintf_format   | Write report entry to file  |
| system_log      | syslog_format    | Log to syslog               |
| audit_log       | syslog_format    | Record audit event          |

**Secure patterns**: `printf("%s", var)`, `fprintf(fp, "%s", var)`, `syslog(pri, "%s", var)`
**Insecure patterns**: `printf(var)`, `fprintf(fp, var)`, `syslog(pri, var)`

## Methodology

### Pipeline (identical for both CWEs)

1. **Baseline**: Generate on all 105 vulnerable prompts with no steering
2. **Activation Collection**: Collect activations at all 32 layers for all 210 prompts
3. **Layer Sweep**: Train linear probes at each layer, select best by accuracy + direction norm
4. **Pilot LOBO**: 2-fold pilot to check if steering works
5. **Full LOBO**: 7-fold cross-validation

### Configuration

- **Model**: meta-llama/Meta-Llama-3.1-8B-Instruct
- **Steering layer**: 31 (last hidden layer, selected via layer sweep)
- **Alpha grid**: [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
- **Generation**: temp=0.6, top_p=0.9, max_new_tokens=512, 1 generation per prompt
- **Validation**: LOBO (Leave-One-Base-ID-Out) 7-fold cross-validation

## Results

### CWE-119: Buffer Operations (Full 7-Fold LOBO)

#### Baseline
- **Strict Secure**: 0.0% (0/105)
- **Strict Insecure**: 100.0% (105/105)
- **Refusals**: 0.0%

#### Layer Sweep
Best layer: **31** (100% probe accuracy, 8.66 direction norm)

#### Full LOBO Results (STRICT Scoring)

| Alpha | N | Secure% | Insecure% | Other% |
|-------|---|---------|-----------|--------|
| 0.0 | 105 | 0.0% | 100.0% | 0.0% |
| 0.5 | 105 | 0.0% | 99.0% | 1.0% |
| 1.0 | 105 | 0.0% | 100.0% | 0.0% |
| 1.5 | 105 | 1.0% | 98.1% | 1.0% |
| 2.0 | 105 | 1.0% | 99.0% | 0.0% |
| 2.5 | 105 | 1.0% | 96.2% | 2.9% |
| 3.0 | 105 | 3.8% | 93.3% | 2.9% |
| 3.5 | 105 | 7.6% | 87.6% | 4.8% |
| **4.0** | **105** | **20.0%** | **72.4%** | **7.6%** |
| 5.0 | 105 | 20.0% | 29.5% | 50.5% |

**Best result**: 20.0% secure at α=4.0 (+20pp from baseline)

**Key observation**: Unlike CWE-787, steering for CWE-119 shows minimal effect. The model strongly defaults to insecure patterns (gets/strcpy) and resists steering. At α=5.0, output quality degrades (50.5% "other") rather than becoming secure.

### CWE-134: Format Strings (Pilot 2-Fold)

#### Baseline
- **Strict Secure**: 66.7% (70/105)
- **Strict Insecure**: 33.3% (35/105)
- **Refusals**: 0.0%

**Note**: Unlike CWE-787 (0% baseline) and CWE-119 (0% baseline), the model already generates secure format string code most of the time.

#### Layer Sweep
Best layer: **31** (100% probe accuracy, 8.51 direction norm)

#### Pilot LOBO Results (STRICT Scoring)

| Alpha | N | Secure% | Insecure% | Other% | Refusal% |
|-------|---|---------|-----------|--------|----------|
| 0.0 | 30 | 66.7% | 33.3% | 0.0% | 0.0% |
| 0.5 | 30 | 73.3% | 26.7% | 0.0% | 0.0% |
| 1.0 | 30 | 76.7% | 23.3% | 0.0% | 0.0% |
| **1.5** | **30** | **90.0%** | **10.0%** | **0.0%** | **0.0%** |
| 2.0 | 30 | 80.0% | 20.0% | 0.0% | 0.0% |
| 2.5 | 30 | 86.7% | 10.0% | 3.3% | 0.0% |
| 3.0 | 30 | 86.7% | 13.3% | 0.0% | 0.0% |
| 3.5 | 30 | 90.0% | 10.0% | 0.0% | 0.0% |
| 4.0 | 30 | 83.3% | 13.3% | 3.3% | 0.0% |
| 5.0 | 30 | 36.7% | 6.7% | 50.0% | 6.7% |

**Best result**: 90.0% secure at α=1.5 (+23.3pp from baseline)

**Key observations**:
- Model already has strong format string safety (66.7% baseline)
- Steering improves secure rate by ~23pp at optimal α
- Lower α (1.5) works better than higher α
- Over-steering (α=5.0) causes significant degradation

## Cross-CWE Summary

| CWE | Vulnerability Type | Baseline | Best Steered | Best α | Improvement |
|-----|-------------------|----------|--------------|--------|-------------|
| CWE-787 | sprintf → snprintf | 0.0% | 52.4% | 3.5 | +52.4pp |
| CWE-119 | gets/strcpy → fgets/strncpy | 0.0% | 20.0% | 4.0 | +20.0pp |
| CWE-134 | printf(var) → printf("%s", var) | 66.7% | 90.0% | 1.5 | +23.3pp |

## Key Findings

1. **Steering effectiveness varies by CWE type**: CWE-787 shows strong steering (52pp), CWE-134 shows moderate steering (23pp), CWE-119 shows weak steering (20pp)

2. **Baseline matters**: CWE-134 has high baseline security (66.7%), so steering has less room to improve. CWE-119 and CWE-787 both start at 0% but respond differently to steering.

3. **CWE-119 is resistant to steering**: Despite 0% baseline and 100% probe accuracy, steering only achieves 20% secure at best. The model may have deeply ingrained preferences for gets()/strcpy().

4. **Layer selection is consistent**: Layer 31 (last hidden layer) is optimal for all CWEs.

5. **Optimal α varies by CWE**: CWE-134 works best at low α (1.5), CWE-787 at medium α (3.5), CWE-119 requires high α (4.0+) for any effect.

6. **Over-steering collapse is universal**: At high α (5.0+), all CWEs show output degradation rather than increased security.

## Interpretation

The results suggest that mean-difference steering is CWE-dependent. The technique works well when:
- The secure pattern is structurally similar to the insecure one (sprintf→snprintf)
- The model has some latent understanding of the secure alternative

CWE-119 resistance may be explained by:
- gets() and strcpy() are more fundamental/common in training data
- fgets() and strncpy() have different function signatures (extra parameters)
- The transformation requires more than just prefix change

CWE-134's high baseline suggests format string security may already be well-represented in the model's training, making steering less necessary but still effective for the remaining insecure outputs.

## Code Location

`src/experiments/02-05_cross_cwe_steering/`
- [datasets/cwe119/](../src/experiments/02-05_cross_cwe_steering/datasets/cwe119/) - CWE-119 validated pairs and expanded dataset
- [datasets/cwe134/](../src/experiments/02-05_cross_cwe_steering/datasets/cwe134/) - CWE-134 validated pairs and expanded dataset
- [experiment_cwe119_llama8b/](../src/experiments/02-05_cross_cwe_steering/experiment_cwe119_llama8b/) - CWE-119 steering experiment
- [experiment_cwe134_llama8b/](../src/experiments/02-05_cross_cwe_steering/experiment_cwe134_llama8b/) - CWE-134 steering experiment
- [shared/](../src/experiments/02-05_cross_cwe_steering/shared/) - Shared utilities (model loading, steering)

## Data Location

### CWE-119 Data
- Expanded dataset: `datasets/cwe119/data/cwe119_expanded_20260205_151207.jsonl` (105 pairs)
- Baseline: `experiment_cwe119_llama8b/data/baseline_20260205_151629.json`
- Activations: `experiment_cwe119_llama8b/data/activations_20260205_155319.npz`
- Layer sweep: `experiment_cwe119_llama8b/data/layer_sweep_results.json`
- Full LOBO: `experiment_cwe119_llama8b/data/lobo_results_20260205_173625.json`
- Per-fold results: `experiment_cwe119_llama8b/data/fold_results/fold_pair_*.json`

### CWE-134 Data
- Expanded dataset: `datasets/cwe134/data/cwe134_expanded_20260205_151207.jsonl` (105 pairs)
- Pilot results: `experiment_cwe134_llama8b/data/pilot_results_20260205_231906.json`
- Layer sweep: `experiment_cwe134_llama8b/data/layer_sweep_results.json`
- Per-fold results: `experiment_cwe134_llama8b/data/fold_results/pilot_fold_*.json`

## Status

- CWE-119: **Complete** (7/7 folds)
- CWE-134: **Pilot complete** (2/7 folds) - Full LOBO stopped early, can be resumed later
