# Experiment 11: C-134 Transfer Matrix Diagonal Investigation

**Date**: 2026-02-13/14
**Model**: meta-llama/Meta-Llama-3.1-8B-Instruct
**Type**: Forensic investigation + Full 7-fold LOBO with extended alpha sweep

---

## Research Question

Why did CWE-134 score 0% on the diagonal of the 6x6 transfer matrix (Experiment 10), when Experiment 8.5 reported 100% secure for CWE-134?

## Background

The 6x6 transfer matrix from Experiment 10 showed:
```
| vec\prompts | C-787 | C-119 | C-134 | Py-89 | Py-78 | Py-79 |
|-------------|-------|-------|-------|-------|-------|-------|
| C-134       | 0.0%  | 0.7%  | 0.0%  | 69.3% | 4.0%  | 0.0%  |
```

The C-134 diagonal (0.0%) appeared to contradict Exp 8.5 which showed CWE-134: 100% secure.

## Phase 1: Investigation Methodology

Phase 1 was a pure forensic analysis (no new model runs), comparing:
- Transfer matrix code and results (Exp 10)
- E2E pipeline results (Exp 8.5)
- CWE-134 pilot LOBO results (Exp 5)
- Neutral baseline and steered results (Exp 8)
- The actual prompt datasets used in each experiment

## Findings

### Finding 1: Different Prompt Types (Root Cause)

The transfer matrix and Exp 8.5 used **completely different prompts**:

**Transfer Matrix** used `cwe134_expanded_*.jsonl` with the `"vulnerable"` key:
> "Task: Write a C function that prints a user-provided message to stdout.
> Requirements: ... Pass the message directly to printf for simplicity"

These prompts **explicitly instruct** the model to use the insecure `printf(var)` pattern.

**Exp 8.5** used `neutral_eval_prompts.jsonl` with neutral code prefixes:
```c
void display_error(const char *error_msg) {
```

These give no instruction about which pattern to use — the model defaults to secure code.

### Finding 2: CWE-134 Has a Ceiling Effect on Neutral Prompts

| Condition | Secure Rate | Source |
|---|---|---|
| Neutral prompts, NO steering | 100.0% (140/140) | neutral_baseline_results_20260207_134440.json |
| Neutral prompts, α=1.0 steering | 100.0% (140/140) | neutral_steered_results_20260207_140550.json |
| Neutral prompts, Exp 8.5 E2E (α=1.0) | 100.0% (70/70) | e2e_pipeline_results_20260207_212212.json |

The model already generates `printf("%s", var)` on neutral prompts without any steering. CWE-134 steering on neutral prompts was a **no-op** — it can't improve beyond 100%.

### Finding 3: Alpha Values Across Experiments

| Experiment | Alpha | Context | Result |
|---|---|---|---|
| Pilot LOBO (Exp 5) | 1.5 (best) | 2-fold, 30 samples, mixed base_ids | 90% secure |
| Pilot LOBO (Exp 5) | 3.5 (tied) | Same | 90% secure |
| Exp 8 neutral steering | 1.0 (best) | Neutral prompts | 100% (but baseline was already 100%) |
| Exp 8.5 E2E | 1.0 | format_string route | 100% (ceiling effect) |
| Transfer matrix (Exp 10) | 1.5 | Insecure prompts, first 15 from pair_01 | 0% |

The α=1.5 was taken from the pilot LOBO, but the **full LOBO was never completed** for CWE-134 (only 2-fold pilot ran; comment in Exp 5: "Full LOBO stopped early, can be resumed later").

### Finding 4: Prompt Subset Bias

The transfer matrix took the **first 15 prompts** from the expanded dataset, which are all variations of `pair_01_print_message`. The pilot LOBO used mixed base_ids across its 2 folds. This means:

- Pilot LOBO tested across diverse prompt types (print_message, print_status, print_error, log_to_file, write_report, system_log, audit_log)
- Transfer matrix tested only print_message variants

### Finding 5: Entire C-134 Row is Weak

The C-134 vector at α=1.5 is essentially inert on **all** C prompts:
- C-134 → C-787: 0.0%
- C-134 → C-119: 0.7%
- C-134 → C-134: 0.0%

It only shows effect on Py-89 (69.3%), likely because SQL injection prompts are inherently easier to steer secure.

## Phase 1 Resolution

The 0% C-134 diagonal is **NOT a bug**. It results from:

1. **Prompt mismatch**: Transfer matrix tests insecure-variant prompts (which explicitly instruct the vulnerability), while Exp 8.5 tested neutral prompts (where the model already writes secure code).
2. **Ceiling effect**: CWE-134 baseline on neutral prompts is already 100% — steering adds nothing.
3. **Undertrained vector**: Only a 2-fold pilot LOBO was run. The 90% result from 30 samples had high variance.
4. **Subset bias**: Only pair_01 variants were tested, vs. mixed base_ids in the pilot.

---

## Phase 2: Full 7-Fold LOBO with Extended Alpha Sweep

### Methodology

- Full 7-fold LOBO across all base_ids (print_message, print_status, print_error, log_to_file, write_report, system_log, audit_log)
- Extended alpha sweep: α = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- 3 seeds per (fold, alpha) combination → 45 generations per cell
- 256 max new tokens, temperature=0.6, top_p=0.9
- Layer 31 activations, mean-difference steering direction
- Total: 7 folds × 11 alphas × 15 prompts × 3 seeds = 3,465 generations

### Aggregated Results (N=315 per alpha)

| Alpha | Secure Rate | Insecure Rate | Other Rate | Refusal Rate |
|-------|-------------|---------------|------------|--------------|
| 0.0   | **70.2%**   | 29.5%         | 0.3%       | 0.0%         |
| 1.0   | 73.3%       | 23.2%         | 3.5%       | 0.0%         |
| 2.0   | 71.1%       | 22.5%         | 6.3%       | 0.0%         |
| **3.0** | **74.9%** | **21.3%**     | 3.8%       | 0.0%         |
| 4.0   | 61.0%       | 21.0%         | 18.1%      | 1.6%         |
| 5.0   | 22.2%       | 6.3%          | 71.4%      | 7.9%         |
| 6.0   | 0.6%        | 0.0%          | 99.4%      | 0.6%         |
| 7.0+  | 0.0%        | 0.0%          | 100.0%     | 0.0%         |

**Best alpha: 3.0** with 74.9% secure (+4.8pp over baseline 70.2%)

### Per-Fold Breakdown (at best alpha per fold)

| Fold | Baseline (α=0) | Best α | Best Rate | Improvement |
|------|----------------|--------|-----------|-------------|
| pair_01_print_message | 80.0% | 3.0 | 86.7% | +6.7pp |
| pair_02_print_status  | 84.4% | 1.0 | 88.9% | +4.4pp |
| pair_03_print_error   | 84.4% | 1.0 | 93.3% | +8.9pp |
| pair_04_log_to_file   | 80.0% | 2.0 | 84.4% | +4.4pp |
| pair_05_write_report  | 75.6% | 3.0 | 82.2% | +6.7pp |
| pair_06_system_log    | 46.7% | 3.0 | 48.9% | +2.2pp |
| pair_07_audit_log     | 40.0% | 3.0 | 48.9% | +8.9pp |

### Key Observations

1. **Modest improvement**: Best alpha (3.0) only improves by +4.8pp aggregate. This confirms CWE-134 steering is weak on adversarial prompts.
2. **Hard folds**: system_log and audit_log have ~40-47% baselines and barely improve with steering (~49%). These use `syslog()` and `fprintf()` which have different format string patterns.
3. **Output collapse**: At α≥5, outputs become garbled ("other"). At α≥7, 100% destruction. Higher alpha cannot compensate for a weak direction.
4. **Pilot LOBO was biased**: Pilot only tested 2 "easy" folds (print_message, print_status with 80-84% baselines), inflating the apparent 90% secure rate. Full 7-fold reveals the true average is 74.9%.
5. **Direction norms are consistent**: All folds produce directions with norms ~8.3-8.7, suggesting the representation is stable but the direction itself has limited steering power on explicit vulnerability instructions.

### Implications for Transfer Matrix

The full LOBO best alpha (3.0) achieves 74.9% secure on held-out CWE-134 prompts — substantially above the 0% shown in the transfer matrix (which used α=1.5 on only pair_01 prompts). If the transfer matrix were re-run with α=3.0 using diverse prompts, the C-134 diagonal would likely be ~75%, not 0%.

However, even at best alpha, CWE-134 steering on adversarial prompts remains the weakest of all CWE types tested.

---

## Phase 3: Transfer Matrix Row Re-run with α=3.0

### Methodology

Re-ran only the C-134 row (6 cells) of the 6×6 transfer matrix with α=3.0 (from full LOBO), keeping all other parameters identical to the original:
- Same direction vector (cross_cwe_analysis, trained on all data)
- 10 seeds, 15 prompts per cell, 512 max_new_tokens
- Same prompt sets and scorers as original transfer matrix

### Results

| C-134 → | C-787 | C-119 | C-134 | Py-89 | Py-78 | Py-79 |
|----------|-------|-------|-------|-------|-------|-------|
| Original (α=1.5) | 0.0% | 0.7% | 0.0% | 69.3% | 4.0% | 0.0% |
| **Updated (α=3.0)** | **0.0%** | **0.0%** | **0.0%** | **62.0%** | **6.7%** | **0.0%** |
| Delta | 0.0pp | -0.7pp | 0.0pp | -7.3pp | +2.7pp | 0.0pp |

### Key Observations

1. **C-134 diagonal still 0%**: But the failure mode changed — at α=1.5, outputs were recognizable insecure code; at α=3.0, all 150 outputs are "other" (garbled). The higher alpha *destroyed* the output rather than making it secure.
2. **Py-89 dropped**: Cross-language transfer to SQL injection prompts decreased from 69.3% to 62.0%, suggesting α=3.0 is too aggressive for cross-CWE transfer.
3. **C columns unchanged**: All C prompt sets still show 0% secure — the C-134 vector simply cannot steer C code to be secure when prompts explicitly instruct the vulnerability.
4. **Matrix summary unchanged**: Overall diagonal (49.9%) and off-diagonal (13.0%) are essentially the same since C-134 was already 0%.

## Final Resolution

The 0% C-134 diagonal is **confirmed as a legitimate finding**, not a bug or alpha selection error:

1. At α=1.5 (original): outputs are insecure code (model follows instructions)
2. At α=3.0 (optimal LOBO): outputs are garbled (model is disrupted but not redirected)
3. At α≥5: complete output destruction

The CWE-134 vector works modestly in the LOBO setting (+4.8pp when testing on held-out base_ids from the same distribution), but **completely fails** on the transfer matrix's insecure-variant prompts which explicitly instruct the vulnerability pattern. This represents a fundamental limit: activation steering cannot overcome explicit task instructions for format-string vulnerabilities.

## Recommendations

For the paper:
1. Note CWE-134 as a **ceiling effect** case on neutral prompts (100% baseline)
2. On adversarial prompts, CWE-134 is the weakest CWE — even optimal alpha produces 0% on transfer matrix diagonal
3. The updated transfer matrix (with α=3.0) confirms the original finding; no further re-runs needed
4. Hard folds (system_log, audit_log) reveal function-specific difficulty: `syslog()` and `fprintf()` are harder than `printf()`
5. The distinction between LOBO improvement (+4.8pp) and transfer matrix failure (0%) highlights that steering effectiveness depends heavily on prompt type

## Code / Files Generated

- [c134_investigation_20260213.json](../../src/experiments/02-10_python_cwe_steering/results/c134_investigation_20260213.json) - Phase 1 forensic investigation results
- [run_full_lobo.py](../../src/experiments/02-13_c134_full_lobo/run_full_lobo.py) - Full LOBO experiment script
- [c134_full_lobo_20260213_222152.json](../../src/experiments/02-13_c134_full_lobo/results/c134_full_lobo_20260213_222152.json) - Aggregated LOBO results
- [c134_full_lobo_full_20260213_222152.json](../../src/experiments/02-13_c134_full_lobo/results/c134_full_lobo_full_20260213_222152.json) - Full per-fold breakdown
- [activations_cwe134_L31_20260213_221204.npz](../../src/experiments/02-13_c134_full_lobo/data/activations_cwe134_L31_20260213_221204.npz) - Layer 31 activations (210 prompts)
- [rerun_c134_transfer_row.py](../../src/experiments/02-13_c134_full_lobo/rerun_c134_transfer_row.py) - Transfer matrix row re-run script
- [c134_transfer_row_20260214_121747.json](../../src/experiments/02-13_c134_full_lobo/results/c134_transfer_row_20260214_121747.json) - Row re-run results
- [transfer_matrix_updated_20260214_121747.json](../../src/experiments/02-13_c134_full_lobo/results/transfer_matrix_updated_20260214_121747.json) - Updated full transfer matrix

## Data Sources

- [transfer_matrix_20260210_201615.json](../../src/experiments/02-10_python_cwe_steering/results/transfer_matrix_20260210_201615.json) - Transfer matrix results
- [04_transfer_matrix.py](../../src/experiments/02-10_python_cwe_steering/04_transfer_matrix.py) - Transfer matrix code
- [pilot_results_20260205_231906.json](../../src/experiments/02-05_cross_cwe_steering/experiment_cwe134_llama8b/data/pilot_results_20260205_231906.json) - CWE-134 pilot LOBO
- [e2e_pipeline_results_20260207_212212.json](../../src/experiments/02-08_probe_routing_v2/results/e2e_pipeline_results_20260207_212212.json) - Exp 8.5 E2E
- [neutral_baseline_results_20260207_134440.json](../../src/experiments/02-05_cross_cwe_steering/neutral_eval/results/neutral_baseline_results_20260207_134440.json) - Neutral baseline
- [cwe134_expanded_20260205_151207.jsonl](../../src/experiments/02-05_cross_cwe_steering/datasets/cwe134/data/cwe134_expanded_20260205_151207.jsonl) - CWE-134 expanded dataset
- [neutral_eval_prompts.jsonl](../../src/experiments/02-05_cross_cwe_steering/neutral_eval/data/neutral_eval_prompts.jsonl) - Neutral eval prompts
