# Data Inventory

This document tracks all datasets created during experiments.

---

## Experiment 15/15b: Mistral-7B E2E Pipeline (02-18)

### Overview
Probe weights, E2E pipeline results, and latency benchmark for Mistral-7B deployment pipeline validation. Exp 15 used C-vs-Python probe at L8 (failed, 25% routing). Exp 15b used Llama-equivalent design: format_string-vs-buffer at L31 (76.2% routing).

### Probe Weights & Data
`src/experiments/02-18_mistral_e2e_pipeline/data/`

| File | Description |
|------|-------------|
| probe_weights.npy | Exp 15 probe (buffer vs injection, L8) — failed |
| probe_bias.npy | Exp 15 probe bias |
| probe_scaler_mean.npy | Exp 15 StandardScaler mean |
| probe_scaler_scale.npy | Exp 15 StandardScaler scale |
| probe_v2_weights.npy | Exp 15b probe (format_string vs buffer, L31) — Llama design |
| probe_v2_bias.npy | Exp 15b probe bias |
| probe_v2_scaler_mean.npy | Exp 15b StandardScaler mean |
| probe_v2_scaler_scale.npy | Exp 15b StandardScaler scale |
| activations_mistral_cwe134_L31.npz | CWE-134 adversarial activations on Mistral at L31 (210 samples: X, y, base_ids) |

### Experiment Results
`src/experiments/02-18_mistral_e2e_pipeline/results/`

| File | Description |
|------|-------------|
| [e2e_results_20260216_230544.json](../src/experiments/02-18_mistral_e2e_pipeline/results/e2e_results_20260216_230544.json) | Exp 15: routing 25%, secure 63.9%, overhead 2.0% |
| [e2e_full_20260216_230544.json](../src/experiments/02-18_mistral_e2e_pipeline/results/e2e_full_20260216_230544.json) | Exp 15: full generation outputs |
| [e2e_v2_results_20260216_235227.json](../src/experiments/02-18_mistral_e2e_pipeline/results/e2e_v2_results_20260216_235227.json) | Exp 15b: routing 76.2%, secure 69.5%, overhead 2.0% |
| [e2e_v2_full_20260216_235227.json](../src/experiments/02-18_mistral_e2e_pipeline/results/e2e_v2_full_20260216_235227.json) | Exp 15b: full generation outputs |

**Key finding**: Changing probe from C-vs-Python (L8) to format_string-vs-buffer (L31) improved routing from 25% → 76.2%. Buffer routing 100% perfect; CWE-134 routing weak (28.6%) but no practical impact (98.6% secure anyway).

**How to recreate**:
- Exp 15: `python src/experiments/02-18_mistral_e2e_pipeline/01_run_experiment.py` (requires Exp 12, 13, 14 data)
- Exp 15b: `python src/experiments/02-18_mistral_e2e_pipeline/02_rerun_llama_design.py` (requires Exp 12, 14 data + CWE-134 dataset)

**Used in**: Experiment 15/15b (docs/experiments/02-16_mistral7b_e2e_pipeline.md)

---

## Experiment 16: Qwen-14B CWE-89 LOBO (02-19)

### Overview
Steering vector, activations, and LOBO results for CWE-89 (SQL injection) on Qwen2.5-14B-Instruct, Layer 47. Third architecture replication.

### Steering Vector & Activations
`src/experiments/02-19_qwen14b_cwe89_lobo/data/`

| File | Description | Size |
|------|-------------|------|
| [activations_qwen14b_cwe89_L47.npz](../src/experiments/02-19_qwen14b_cwe89_lobo/data/activations_qwen14b_cwe89_L47.npz) | Activation data (insecure + secure) at Layer 47 | ~20 MB |
| [direction_qwen14b_cwe89_L47_20260216_111452.npy](../src/experiments/02-19_qwen14b_cwe89_lobo/data/direction_qwen14b_cwe89_L47_20260216_111452.npy) | CWE-89 steering vector (5120-dim, L47) | 20 KB |

### Experiment Results
`src/experiments/02-19_qwen14b_cwe89_lobo/results/`

| File | Description |
|------|-------------|
| [lobo_results_20260216_111452.json](../src/experiments/02-19_qwen14b_cwe89_lobo/results/lobo_results_20260216_111452.json) | LOBO summary: aggregated + per-fold, 3-way comparison |
| [lobo_full_20260216_111452.json](../src/experiments/02-19_qwen14b_cwe89_lobo/results/lobo_full_20260216_111452.json) | Full generation-level results |
| fold_*_20260216_111452.json (x7) | Per-fold detailed results |

**How to recreate**: Run `python src/experiments/02-19_qwen14b_cwe89_lobo/01_run_experiment.py` on A100-80GB with Qwen2.5-14B-Instruct in fp16.

**Used in**: Experiment 16 (docs/experiments/02-16_qwen14b_cwe89_lobo_third_architecture.md)

---

## Experiment 14: Mistral-7B CWE-119 LOBO (02-17)

### Overview
Steering vector, activations, and LOBO results for CWE-119 (buffer read overflow) on Mistral-7B, Layer 31. Tests whether CWE-119 steering limitation replicates on Mistral.

### Steering Vector & Activations
`src/experiments/02-17_mistral_cwe119_lobo/data/`

| File | Description | Size |
|------|-------------|------|
| [activations_mistral_cwe119_L31.npz](../src/experiments/02-17_mistral_cwe119_lobo/data/activations_mistral_cwe119_L31.npz) | Activation data (insecure + secure) at Layer 31 | ~14 MB |
| [direction_mistral_cwe119_L31_20260216_060857.npy](../src/experiments/02-17_mistral_cwe119_lobo/data/direction_mistral_cwe119_L31_20260216_060857.npy) | CWE-119 steering vector (4096-dim, L31) | 16 KB |

### Experiment Results
`src/experiments/02-17_mistral_cwe119_lobo/results/`

| File | Description |
|------|-------------|
| [lobo_results_20260216_060857.json](../src/experiments/02-17_mistral_cwe119_lobo/results/lobo_results_20260216_060857.json) | LOBO summary: aggregated + per-fold, cosine similarity |
| [lobo_full_20260216_060857.json](../src/experiments/02-17_mistral_cwe119_lobo/results/lobo_full_20260216_060857.json) | Full generation-level results |
| fold_*_20260216_060857.json (x7) | Per-fold detailed results |

**Key data point**: CWE-787 vs CWE-119 cosine similarity = 0.005 (near orthogonal on Mistral, unlike Llama where they were inseparable).

**How to recreate**: Run `python src/experiments/02-17_mistral_cwe119_lobo/01_run_experiment.py` on A100-80GB with Mistral-7B-Instruct-v0.3 in fp16.

**Used in**: Experiment 14 (docs/experiments/02-16_mistral7b_cwe119_lobo_limitation_replication.md)

---

## Experiment 13: Mistral-7B CWE-89 LOBO (02-16)

### Overview
Steering vector, activations, and LOBO results for CWE-89 (SQL injection) on Mistral-7B, Layer 31. Cross-architecture replication of Llama-8B CWE-89 steering.

### Steering Vector & Activations
`src/experiments/02-16_mistral_cwe89_lobo/data/`

| File | Description | Size |
|------|-------------|------|
| [activations_mistral_cwe89_L31.npz](../src/experiments/02-16_mistral_cwe89_lobo/data/activations_mistral_cwe89_L31.npz) | Activation data (insecure + secure) at Layer 31 | ~14 MB |
| [direction_mistral_cwe89_L31_20260216_025624.npy](../src/experiments/02-16_mistral_cwe89_lobo/data/direction_mistral_cwe89_L31_20260216_025624.npy) | CWE-89 steering vector (4096-dim, L31) | 16 KB |

### Experiment Results
`src/experiments/02-16_mistral_cwe89_lobo/results/`

| File | Description |
|------|-------------|
| [lobo_results_20260216_025624.json](../src/experiments/02-16_mistral_cwe89_lobo/results/lobo_results_20260216_025624.json) | LOBO summary: aggregated + per-fold, Llama comparison |
| [lobo_full_20260216_025624.json](../src/experiments/02-16_mistral_cwe89_lobo/results/lobo_full_20260216_025624.json) | Full generation-level results |
| fold_*_20260216_025624.json (x7) | Per-fold detailed results |

**How to recreate**: Run `python src/experiments/02-16_mistral_cwe89_lobo/01_run_experiment.py` on A100-80GB with Mistral-7B-Instruct-v0.3 in fp16.

**Used in**: Experiment 13 (docs/experiments/02-16_mistral7b_cwe89_lobo_cross_architecture.md)

---

## Experiment 10: Python CWE Steering Results (02-10)

### Overview
Steering vectors, activations, probe weights, and experiment results from Experiment 10: Python CWE Steering & Cross-Language Validation.

### Steering Vectors & Activations
`src/experiments/02-10_python_cwe_steering/data/`

| File | Description | Size |
|------|-------------|------|
| [direction_cwe89_L31_20260210_015359.npy](../src/experiments/02-10_python_cwe_steering/data/direction_cwe89_L31_20260210_015359.npy) | CWE-89 steering vector (4096-dim, L31) | 16 KB |
| [direction_cwe78_L31_20260210_015359.npy](../src/experiments/02-10_python_cwe_steering/data/direction_cwe78_L31_20260210_015359.npy) | CWE-78 steering vector (4096-dim, L31) | 16 KB |
| [direction_cwe79_L31_20260210_015359.npy](../src/experiments/02-10_python_cwe_steering/data/direction_cwe79_L31_20260210_015359.npy) | CWE-79 steering vector (4096-dim, L31) | 16 KB |
| activations_cwe{89,78,79}_L31_*.npz | Activation data (insecure + secure) | ~14 MB each |
| [vector_metadata_20260210_015359.json](../src/experiments/02-10_python_cwe_steering/data/vector_metadata_20260210_015359.json) | Direction norms, cross-language similarity matrix | 2 KB |
| cwe_probe_weights_20260210_201527.npz | 3-class LogisticRegression probe weights | ~49 KB |

### Experiment Results
`src/experiments/02-10_python_cwe_steering/results/`

| File | Description |
|------|-------------|
| [baseline_results_rescored_20260210_021448.json](../src/experiments/02-10_python_cwe_steering/results/baseline_results_rescored_20260210_021448.json) | Re-scored baseline with fixed scorers |
| [lobo_results_20260210_080939.json](../src/experiments/02-10_python_cwe_steering/results/lobo_results_20260210_080939.json) | LOBO cross-validation summary |
| [transfer_matrix_20260210_201615.json](../src/experiments/02-10_python_cwe_steering/results/transfer_matrix_20260210_201615.json) | 6×6 cross-language transfer matrix |
| [probe_routing_20260210_201527.json](../src/experiments/02-10_python_cwe_steering/results/probe_routing_20260210_201527.json) | Probe routing accuracy |
| [e2e_results_20260210_221205.json](../src/experiments/02-10_python_cwe_steering/results/e2e_results_20260210_221205.json) | E2E pipeline results |
| [mechanistic_comparison_20260210_015506.json](../src/experiments/02-10_python_cwe_steering/results/mechanistic_comparison_20260210_015506.json) | Cross-language mechanistic comparison |

**Used in**: Experiment 10 (docs/experiments/02-10_llama8b_python_cwe_steering.md)

---

## Python CWE Datasets: CWE-89, CWE-78, CWE-79 (02-09)

### Overview

Adversarial prompt pair datasets for 3 Python-language CWEs, designed for mean-difference activation steering. Each CWE has 7 base scenarios × 15 linguistic instruction variations = 105 pairs. Variations change ONLY the docstring instruction — function signatures, imports, and skeleton remain identical. Also includes 21 neutral evaluation prompts (7 per CWE) and per-CWE regex-based scoring functions.

### CWE-89: SQL Injection Dataset
`src/experiments/02-05_cross_cwe_steering/datasets/cwe89/`

| File | Description | Size |
|------|-------------|------|
| [scoring.py](../src/experiments/02-05_cross_cwe_steering/datasets/cwe89/scoring.py) | Regex scorer: parameterized queries (secure) vs string formatting (insecure) | ~2 KB |
| [data/cwe89_expanded_20260209_221808.jsonl](../src/experiments/02-05_cross_cwe_steering/datasets/cwe89/data/cwe89_expanded_20260209_221808.jsonl) | 105 prompt pairs (7 base × 15 variations) | ~80 KB |

**Base scenarios**: user_login, product_search, order_history, user_profile_update, log_entry, admin_delete, report_filter
**Secure pattern**: `cursor.execute("...?...", (params,))` — parameterized queries
**Insecure pattern**: `f"SELECT...{var}"`, `"SELECT..." + var` — string formatting

### CWE-78: OS Command Injection Dataset
`src/experiments/02-05_cross_cwe_steering/datasets/cwe78/`

| File | Description | Size |
|------|-------------|------|
| [scoring.py](../src/experiments/02-05_cross_cwe_steering/datasets/cwe78/scoring.py) | Regex scorer: subprocess with list args (secure) vs os.system/shell=True (insecure) | ~2 KB |
| [data/cwe78_expanded_20260209_221808.jsonl](../src/experiments/02-05_cross_cwe_steering/datasets/cwe78/data/cwe78_expanded_20260209_221808.jsonl) | 105 prompt pairs (7 base × 15 variations) | ~70 KB |

**Base scenarios**: ping_host, dns_lookup, disk_usage, file_compress, process_grep, git_clone, convert_image
**Secure pattern**: `subprocess.run(["cmd", arg1, arg2])` — list arguments
**Insecure pattern**: `os.system(f"cmd {arg}")`, `subprocess.run(..., shell=True)` — shell injection

### CWE-79: Cross-Site Scripting (XSS) Dataset
`src/experiments/02-05_cross_cwe_steering/datasets/cwe79/`

| File | Description | Size |
|------|-------------|------|
| [scoring.py](../src/experiments/02-05_cross_cwe_steering/datasets/cwe79/scoring.py) | Regex scorer: html.escape/render_template (secure) vs raw f-string HTML (insecure) | ~2 KB |
| [data/cwe79_expanded_20260209_221808.jsonl](../src/experiments/02-05_cross_cwe_steering/datasets/cwe79/data/cwe79_expanded_20260209_221808.jsonl) | 105 prompt pairs (7 base × 15 variations) | ~65 KB |

**Base scenarios**: welcome_page, search_results, user_comment, error_message, profile_display, admin_panel, email_preview
**Secure pattern**: `html.escape(var)`, `render_template()`, `bleach.clean()` — escaping
**Insecure pattern**: `f"<div>{var}</div>"`, `"<p>" + var + "</p>"` — raw insertion

### Python Neutral Evaluation Prompts
`src/experiments/02-05_cross_cwe_steering/datasets/neutral_eval/data/`

| File | Description | Size |
|------|-------------|------|
| [neutral_python_prompts.jsonl](../src/experiments/02-05_cross_cwe_steering/datasets/neutral_eval/data/neutral_python_prompts.jsonl) | 21 neutral prompts (7 per CWE) — task described without specifying approach | ~8 KB |

**Stratification**: 7 prompts per CWE (CWE-89, CWE-78, CWE-79). Each describes the programming task neutrally without mentioning secure or insecure patterns.

### Shared Files

| File | Description |
|------|-------------|
| [expand_python_datasets.py](../src/experiments/02-05_cross_cwe_steering/datasets/expand_python_datasets.py) | Expansion script: base definitions + JSONL generation |
| [test_scorers.py](../src/experiments/02-05_cross_cwe_steering/datasets/test_scorers.py) | Scorer validation: 25 tests per CWE (75 total), all passing |

### JSONL Structure

```python
import json

with open('cwe89/data/cwe89_expanded_20260209_221808.jsonl') as f:
    pairs = [json.loads(line) for line in f]

# Each pair has:
{
    "pair_id": "cwe89_user_login_v01",
    "base_id": "user_login",
    "cwe": "CWE-89",
    "variation": 1,
    "insecure_prompt": "...",
    "secure_prompt": "...",
    "insecure_instruction": "...",
    "secure_instruction": "..."
}
```

### How to Recreate

```bash
cd src/experiments/02-05_cross_cwe_steering/datasets

# Generate all 3 JSONL files (315 pairs total)
python expand_python_datasets.py

# Validate scorers (must pass before any downstream use)
python test_scorers.py
```

### Counts Summary

| Dataset | Pairs | Base Scenarios | Variations |
|---------|-------|---------------|------------|
| CWE-89 | 105 | 7 | 15 |
| CWE-78 | 105 | 7 | 15 |
| CWE-79 | 105 | 7 | 15 |
| **Total adversarial** | **315** | **21** | **15** |
| Neutral (Python) | 21 | — | — |
| Neutral (C, existing) | 21 | — | — |
| **Total neutral** | **42** | — | — |

**Used in**: Upcoming steering experiments for Python-language CWEs.

---

## Experiment 8.5: Neutral-Trained CWE Router & 2-Tier Deployment (02-07)

### Overview

Fixes probe routing distribution shift (Exp 8 Phase 4: 66.7% → 95.2%), validates 2-tier binary deployment architecture, runs full E2E pipeline with timing.

### Activations
`src/experiments/02-08_probe_routing_v2/data/`

| File | Description | Size |
|------|-------------|------|
| [neutral_original_L{0,8,16,24,31}.npy](../src/experiments/02-08_probe_routing_v2/data/) | Neutral prompt activations (21 × 4096) per layer | ~670 KB each |
| [neutral_augmented_L{0,8,16,24,31}.npy](../src/experiments/02-08_probe_routing_v2/data/) | Augmented neutral activations (105 × 4096) per layer | ~3.4 MB each |
| [adversarial_L{0,8,16,24,31}.npy](../src/experiments/02-08_probe_routing_v2/data/) | Adversarial activations (315 × 4096) per layer | ~10 MB each |
| [labels_metadata.json](../src/experiments/02-08_probe_routing_v2/data/labels_metadata.json) | Label arrays and metadata | ~5 KB |

**Stratification**: Neutral: 7 prompts × 3 CWEs = 21. Augmented: 5 prefix variants × 21 = 105. Adversarial: 105 pairs × 3 CWEs = 315.

**How to recreate**: Run `python 01_probe_retraining.py` — collects activations from model, trains probes.

### Binary Probe Weights
`src/experiments/02-08_probe_routing_v2/data/`

| File | Description | Size |
|------|-------------|------|
| [binary_probe_weights.npy](../src/experiments/02-08_probe_routing_v2/data/binary_probe_weights.npy) | LogReg weights (1 × 4096) — adv-trained, L31 | ~33 KB |
| [binary_probe_bias.npy](../src/experiments/02-08_probe_routing_v2/data/binary_probe_bias.npy) | LogReg bias (1,) | ~1 KB |
| [binary_probe_scaler_mean.npy](../src/experiments/02-08_probe_routing_v2/data/binary_probe_scaler_mean.npy) | StandardScaler mean (4096,) | ~33 KB |
| [binary_probe_scaler_scale.npy](../src/experiments/02-08_probe_routing_v2/data/binary_probe_scaler_scale.npy) | StandardScaler scale (4096,) | ~33 KB |

**Training**: Adv-trained binary LogReg on 315 adversarial samples at L31. 95.2% LOO on neutral prompts.

### Results
`src/experiments/02-08_probe_routing_v2/results/`

| File | Description | Size |
|------|-------------|------|
| [3way_probe_results_20260207_211639.json](../src/experiments/02-08_probe_routing_v2/results/3way_probe_results_20260207_211639.json) | Part A: All probe training methods × layers | ~50 KB |
| [2tier_simulation_results_20260207_212158.json](../src/experiments/02-08_probe_routing_v2/results/2tier_simulation_results_20260207_212158.json) | Part B: Strategy comparison table | ~2 KB |
| [e2e_pipeline_results_20260207_212212.json](../src/experiments/02-08_probe_routing_v2/results/e2e_pipeline_results_20260207_212212.json) | Part C: E2E pipeline summary | ~5 KB |
| [e2e_pipeline_full_20260207_212212.json](../src/experiments/02-08_probe_routing_v2/results/e2e_pipeline_full_20260207_212212.json) | Part C: Full outputs with completions | ~500 KB |

**Used in**: Experiment 8.5. See [experiment report](experiments/02-07_llama8b_neutral_probe_routing_v2.md).

---

## Experiment 8: Neutral Evaluation — Per-CWE Steering (02-07)

### Overview

Evaluates per-CWE steering vectors on neutral prompts (tasks described without specifying insecure functions). Demonstrates realistic deployment effectiveness. 4 phases: baselines, per-CWE steering, cross-CWE sanity check, probe-gated routing.

### Neutral Evaluation Prompts
`src/experiments/02-05_cross_cwe_steering/neutral_eval/data/`

| File | Description | Size |
|------|-------------|------|
| [neutral_eval_prompts.jsonl](../src/experiments/02-05_cross_cwe_steering/neutral_eval/data/neutral_eval_prompts.jsonl) | 21 neutral prompts (7 per CWE), adapted from Pearce et al. and Sandoval et al. | ~12 KB |

**Stratification**: 7 prompts per CWE type (CWE-787, CWE-119, CWE-134). 15 adapted from Pearce et al. (2022), 6 from Sandoval et al. (2023).

**How to recreate**: Manually curated. See `data/experiment_8_neutral_eval_instructions.md` for design rationale.

### Results
`src/experiments/02-05_cross_cwe_steering/neutral_eval/results/`

| File | Description | Size |
|------|-------------|------|
| `neutral_baseline_results_20260207_134440.json` | Phase 1: Baseline secure rates (no steering), 420 generations | ~250 KB |
| `neutral_steered_results_20260207_140550.json` | Phase 2: Per-CWE steering summary (best alphas), 1680 generations | ~15 KB |
| `neutral_steered_full_20260207_140550.json` | Phase 2: Full outputs with all completions | ~3 MB |
| `neutral_cross_cwe_results_20260207_190849.json` | Phase 3: Cross-CWE impact matrix, 840 generations | ~10 KB |
| `neutral_cross_cwe_full_20260207_190849.json` | Phase 3: Full outputs with all completions | ~2 MB |
| `neutral_probe_routing_results_20260207_201828.json` | Phase 4: Probe routing accuracy (3 methods) | ~15 KB |

**Used in**: Experiment 8 (this experiment). See [experiment report](experiments/02-07_llama8b_neutral_eval_per_cwe_steering.md).

---

## Experiment 5: Cross-CWE Steering (02-05/06)

### Overview

Cross-CWE validation of mean-difference activation steering. Tests CWE-119 (buffer operations) and CWE-134 (format strings) to verify that steering generalizes across vulnerability types.

### Data Location (Git Worktree)

All data in worktree at `/home/paperspace/MATS-cwe-steering/` on branch `feature/cwe119-cwe134-datasets`.

#### CWE-119 Dataset
`src/experiments/02-05_cross_cwe_steering/datasets/cwe119/`

| File | Description | Size |
|------|-------------|------|
| `validated_pairs.py` | 7 validated CWE-119 prompt pairs | ~4 KB |
| `config/cwe119_prompt_pairs.py` | Pair definitions with detection patterns | ~15 KB |
| `data/cwe119_expanded_20260205_151207.jsonl` | Original expanded dataset (35 pairs: 7 base × 5 variations) | ~25 KB |
| `data/expansion_summary_20260205_151207.json` | Original generation metadata | ~1 KB |
| `data/cwe119_expanded_20260207_024627.jsonl` | **Re-expanded dataset (105 pairs: 7 base × 15 variations)** | ~100 KB |
| `data/expansion_summary_20260207_024627.json` | Re-expansion metadata | ~1 KB |

#### CWE-134 Dataset
`src/experiments/02-05_cross_cwe_steering/datasets/cwe134/`

| File | Description | Size |
|------|-------------|------|
| `validated_pairs.py` | 7 validated CWE-134 prompt pairs | ~4 KB |
| `config/cwe134_prompt_pairs.py` | Pair definitions with detection patterns | ~15 KB |
| `data/cwe134_expanded_20260205_151207.jsonl` | Original expanded dataset (35 pairs: 7 base × 5 variations) | ~30 KB |
| `data/expansion_summary_20260205_151207.json` | Original generation metadata | ~1 KB |
| `data/cwe134_expanded_20260207_024627.jsonl` | **Re-expanded dataset (105 pairs: 7 base × 15 variations)** | ~107 KB |
| `data/expansion_summary_20260207_024627.json` | Re-expansion metadata | ~1 KB |

#### CWE-119 Experiment Results
`src/experiments/02-05_cross_cwe_steering/experiment_cwe119_llama8b/data/`

| File | Description | Size |
|------|-------------|------|
| `baseline_20260205_151629.json` | Baseline generation results | ~81 KB |
| `activations_20260205_155319.npz` | Per-layer activations (32 layers × 210 × 4096) | ~51 MB |
| `metadata_20260205_155319.json` | Prompt metadata | ~40 KB |
| `layer_sweep_results.json` | Linear probe accuracy per layer | ~5 KB |
| `pilot_results_20260205_155558.json` | Pilot LOBO results (2 folds) | ~3 KB |
| `lobo_results_20260205_173625.json` | Full LOBO results (7 folds) | ~3 KB |
| `fold_results/fold_pair_*.json` | Per-fold detailed results | ~124 KB each |

#### CWE-134 Experiment Results
`src/experiments/02-05_cross_cwe_steering/experiment_cwe134_llama8b/data/`

| File | Description | Size |
|------|-------------|------|
| `layer_sweep_results.json` | Linear probe accuracy per layer | ~5 KB |
| `pilot_results_20260205_231906.json` | Pilot LOBO results (2 folds) | ~3 KB |
| `fold_results/pilot_fold_*.json` | Per-fold pilot results | ~125 KB each |
| `fold_results/fold_pair_01_*.json` | Full LOBO fold 1 (partial) | ~125 KB |

### Dataset Structure

#### CWE-119 Pairs

| Short Name | Vulnerability | Secure Pattern | Insecure Pattern |
|------------|---------------|----------------|------------------|
| user_input | gets | fgets() | gets() |
| command_parser | gets | fgets() | gets() |
| config_reader | gets | fgets() | gets() |
| username_copy | strcpy | strncpy() | strcpy() |
| filepath_copy | strcpy | strncpy() | strcpy() |
| error_msg_copy | strcpy | strncpy() | strcpy() |
| hostname_copy | strcpy | strncpy() | strcpy() |

#### CWE-134 Pairs

| Short Name | Vulnerability | Secure Pattern | Insecure Pattern |
|------------|---------------|----------------|------------------|
| print_message | printf_format | printf("%s", var) | printf(var) |
| print_status | printf_format | printf("%s", var) | printf(var) |
| print_error | printf_format | printf("%s", var) | printf(var) |
| log_to_file | fprintf_format | fprintf(fp, "%s", var) | fprintf(fp, var) |
| write_report | fprintf_format | fprintf(fp, "%s", var) | fprintf(fp, var) |
| system_log | syslog_format | syslog(pri, "%s", var) | syslog(pri, var) |
| audit_log | syslog_format | syslog(pri, "%s", var) | syslog(pri, var) |

### Key Results

| CWE | Baseline | Best Steered | Best Alpha | Improvement | Status |
|-----|----------|--------------|------------|-------------|--------|
| CWE-119 | 0.0% | 20.0% | 4.0 | +20.0pp | Complete (7 folds) |
| CWE-134 | 66.7% | 90.0% | 1.5 | +23.3pp | Pilot only (2 folds) |

### JSONL Structure

```python
import json

# Load expanded dataset
with open('cwe119_expanded_20260205_151207.jsonl') as f:
    pairs = [json.loads(line) for line in f]

# Each pair has:
{
    "id": "pair_01_user_input_var_01",
    "base_id": "pair_01_user_input",
    "name": "User Input - Read Line_var_01",
    "vulnerable": "...",
    "secure": "...",
    "vulnerability_type": "gets",
    "category": "expanded",
    "detection": {
        "secure_pattern": r"\bfgets\s*\(",
        "insecure_pattern": r"\bgets\s*\("
    }
}
```

### How to Reproduce

```bash
# Switch to worktree
cd /home/paperspace/MATS-cwe-steering

# Run CWE-119 experiment
cd src/experiments/02-05_cross_cwe_steering/experiment_cwe119_llama8b
python run_all.py all

# Run CWE-134 experiment
cd ../experiment_cwe134_llama8b
python run_all.py all
```

**Requirements**: A100 GPU for activation collection and steering.

### Cross-CWE Analysis Data (02-06)

Generated by parallel analysis scripts.

`src/experiments/02-05_cross_cwe_steering/cross_cwe_analysis/data/`

| File | Description | Size |
|------|-------------|------|
| `direction_cwe787_L31_20260206_031901.npy` | CWE-787 L31 steering direction (4096-dim, float32) | ~16 KB |
| `direction_cwe119_L31_20260206_031901.npy` | CWE-119 L31 steering direction (4096-dim, float32) | ~16 KB |
| `direction_cwe134_L31_20260206_031901.npy` | CWE-134 L31 steering direction (4096-dim, float32) | ~16 KB |
| `vector_correlation_20260206_031901.json` | Cosine similarity matrix and metadata | ~1 KB |
| `cwe119_failure_analysis.json` | Categorization of 105 outputs at α=4.0 | ~50 KB |
| `statistical_tables.json` | Bootstrap 95% CIs for all experiments | ~5 KB |

### Cross-CWE Transfer Test + CodeQL Feasibility (02-06)

Generated by transfer test and CodeQL feasibility scripts.

`src/experiments/02-05_cross_cwe_steering/cross_cwe_analysis/data/`

| File | Description | Size |
|------|-------------|------|
| `cross_cwe_transfer_20260206_040528.json` | Transfer test summary: secure rates at α=0.0/1.5/3.5 for both directions, hypothesis check | ~3 KB |
| `cross_cwe_transfer_full_20260206_040528.json` | Full per-sample outputs for all 630 generations (105 × 3α × 2 directions) | ~500 KB |
| `cwe134_codeql_feasibility.json` | CodeQL feasibility: 30 samples, compilation status, wrapped code excerpts | ~30 KB |

**Key Results:**

| Transfer | α=0.0 | α=1.5 | α=3.5 |
|---|---|---|---|
| 787→134 (transfer) | 62.9% | 62.9% | 55.2% |
| 134→787 (transfer) | 1.0% | 5.7% | 2.9% |

CodeQL feasibility: 30/30 = 100% compilation rate (FEASIBLE).

### Unified Steering Vector Data (02-06)

Generated by `unified_steering_experiment.py`. Tests a single direction trained on combined CWE-787/119/134 data.

`src/experiments/02-05_cross_cwe_steering/cross_cwe_analysis/data/`

| File | Description | Size |
|------|-------------|------|
| `unified_activations_L31_20260206_172846.npz` | Combined L31 activations for 630 prompts (315 pairs × 2) | ~10 MB |
| `direction_unified_L31_20260206_172846.npy` | Unified steering direction (4096-dim, float32) | ~16 KB |
| `unified_steering_results_20260206_172838.json` | Full results: per-CWE breakdown, direction comparisons, summary | ~5 KB |
| `unified_steering_full_20260206_172838.json` | Per-sample outputs for all 1,050 steered generations | ~1 MB |

**Key Results:**

| CWE | Baseline | Native Best | Unified Best | Delta |
|---|---|---|---|---|
| CWE-787 | 0.0% | 52.4% | 21.0% | -31.4pp |
| CWE-119 | 0.0% | 20.0% | 4.8% | -15.2pp |
| CWE-134 | 66.7% | 90.0% | 69.5% | -20.5pp |

**How to reproduce:**

```bash
cd src/experiments/02-05_cross_cwe_steering

# Unified steering experiment (requires A100 GPU, ~4 hours)
python unified_steering_experiment.py
```

### Stacked Vectors Test Data (02-06)

Generated by `stacked_vectors_experiment.py`. Tests stacking all 3 native CWE vectors simultaneously.

`src/experiments/02-05_cross_cwe_steering/cross_cwe_analysis/data/`

| File | Description | Size |
|------|-------------|------|
| `stacked_steering_results_20260206_225040.json` | Per-config per-CWE breakdown, alpha configs, comparison summary | ~10 KB |
| `stacked_steering_full_20260206_225040.json` | Per-sample outputs for all 1,260 generations (4 configs × 3 CWEs × 105) | ~966 KB |

**Key Results:**

| CWE | Baseline | Native Best | Unified Best | Stk-Low | Stk-Med | Stk-High | Stk-Weighted |
|-----|----------|-------------|--------------|---------|---------|----------|--------------|
| 787 | 0.0% | 52.4% | 21.0% | 7.6% | 20.0% | 27.6% | 18.1% |
| 119 | 0.0% | 20.0% | 4.8% | 1.0% | 2.9% | 7.6% | 10.5% |
| 134 | 66.7% | 90.0% | 69.5% | 59.0% | 52.4% | 48.6% | 55.2% |

All configs FAIL success criteria. Stacking causes destructive interference.

**How to reproduce:**

```bash
cd src/experiments/02-05_cross_cwe_steering
python stacked_vectors_experiment.py  # Requires A100 GPU, ~4.5 hours
```

### PCA Subspace Steering Data (02-07)

Generated by `pca_analysis.py` (CPU) and `pca_steering_experiment.py` (GPU). Tests PCA decomposition of 3 CWE direction vectors and multi-PC steering.

`src/experiments/02-05_cross_cwe_steering/cross_cwe_analysis/data/`

| File | Description | Size |
|------|-------------|------|
| `pca_analysis_20260207_025304.json` | PCA decomposition: eigenvalues, variance explained, PC loadings, cosine similarities | ~2 KB |
| `pc1_security_L31_20260207_025304.npy` | PC1 direction vector (4096-dim, unit norm, float32) | ~16 KB |
| `pc2_security_L31_20260207_025304.npy` | PC2 direction vector (4096-dim, unit norm, float32) | ~16 KB |
| `pc3_security_L31_20260207_025304.npy` | PC3 direction vector (4096-dim, unit norm, float32) | ~16 KB |
| `pca_subspace_steering_results_20260207_030444.json` | PCA steering summary: 4 configs × 3 CWEs, per-config secure rates | ~5 KB |
| `pca_subspace_steering_full_20260207_030444.json` | Per-sample outputs for all 1,260 steered generations | ~1 MB |

**Key Results:**

| Config | CWE-787 | CWE-119 | CWE-134 | Avg |
|--------|---------|---------|---------|-----|
| PC1-only α=3.0 | 1.9% | 0.0% | 71.4% | 24.4% |
| PC1+2 weighted | 0.0% | 0.0% | 67.6% | 22.5% |
| PC1+2+3 weighted | 1.0% | 0.0% | 70.5% | 23.8% |
| PC1+2+3 sv-weighted | 1.9% | 0.0% | 74.3% | 25.4% |

All configs FAIL — worse than unified single vector on CWE-787 and CWE-119.

**How to reproduce:**

```bash
cd src/experiments/02-05_cross_cwe_steering

# PCA analysis (CPU only, <1 min)
python pca_analysis.py

# PCA steering (requires A100 GPU, ~3.5 hours)
python pca_steering_experiment.py
```

### Conceptor AND Steering Data (02-07)

Generated by `conceptor_steering_experiment.py`. Tests Boolean AND composition of per-CWE conceptor matrices computed from secure-prompt activations.

`src/experiments/02-05_cross_cwe_steering/cross_cwe_analysis/data/`

| File | Description | Size |
|------|-------------|------|
| `secure_activations_L31_20260207_052813.npz` | Secure-prompt activations at L31 (315 prompts × 4096, keyed by CWE) | ~5 MB |
| `conceptor_info_20260207_052813.json` | Per-CWE conceptor diagnostics (trace, significant dims, top eigenvalues) + AND result | ~4 KB |
| `conceptor_steering_results_20260207_052813.json` | Results: steering skipped (zero intersection), all diagnostics | ~5 KB |

**Key Results:**

C_security (Boolean AND) has trace ≈ 0.003, max eigenvalue ≈ 1.1e-05 for ALL apertures {1.0, 5.0}. Zero shared subspace found. Steering was skipped.

Root cause: 105 samples per CWE in 4096-dim space → each conceptor spans at most 36-104 dims, intersection in R^4096 is zero.

**How to reproduce:**

```bash
cd src/experiments/02-05_cross_cwe_steering
python conceptor_steering_experiment.py  # Requires A100 GPU, ~15 min (activation collection only)
```

**How to reproduce (earlier experiments):**

```bash
cd src/experiments/02-05_cross_cwe_steering

# CodeQL feasibility check (CPU only, needs gcc)
python cwe134_codeql_feasibility.py

# Cross-CWE transfer test (requires GPU)
python cross_cwe_transfer_test.py
```

**How to reproduce (earlier analyses):**

```bash
cd src/experiments/02-05_cross_cwe_steering

# Vector correlation (requires GPU for activation collection)
python vector_correlation_analysis.py

# Failure analysis and statistical tables (CPU only)
python cwe119_failure_analysis.py
python statistical_tables.py
```

---

## Experiment 4: Cross-Model CWE-787 Steering (02-05)

### Overview

Cross-model validation of mean-difference activation steering for CWE-787. Tests Mistral-7B-Instruct-v0.3 and Llama-3.1-70B-Instruct to verify that steering generalizes beyond Llama-8B.

### Data Location

#### Experiment 4A: Mistral-7B
`src/experiments/02-05_cross_model_cwe787_steering/experiment_4a_mistral7b/data/`

| File | Description | Size |
|------|-------------|------|
| `baseline_20260205_021419.json` | Baseline generation results (105 prompts) | ~150 KB |
| `activations_20260205_022224.npz` | Per-layer activations (32 layers x 210 x 4096) | ~99 MB |
| `metadata_20260205_022224.json` | Prompt metadata and indices | ~15 KB |
| `layer_sweep_results.json` | Linear probe accuracy per layer | ~5 KB |
| `lobo_results_20260205_045755.json` | Full 7-fold LOBO aggregated results | ~200 KB |
| `fold_results/fold_*.json` | Per-fold detailed results (7+2 pilot files) | ~100 KB each |

#### Experiment 4B: Llama-70B
`src/experiments/02-05_cross_model_cwe787_steering/experiment_4b_llama70b/data/`

| File | Description | Size |
|------|-------------|------|
| `baseline_20260205_071732.json` | Baseline generation results (105 prompts) | ~180 KB |
| `activations_20260205_075202.npz` | Per-layer activations (80 layers x 210 x 8192) | ~254 MB |
| `metadata_20260205_075202.json` | Prompt metadata and indices | ~15 KB |
| `layer_sweep_results.json` | Linear probe accuracy per layer (best_layer=79) | ~10 KB |
| `fold_results/pilot_fold_*_091351.json` | Pilot LOBO results (2 folds, layer 79) | ~100 KB each |
| `fold_results/fold_*_111622.json` | Full LOBO results (7 folds, in progress) | ~100 KB each |

### Key Results

| Model | Baseline | Best Steered | Best Alpha | Improvement |
|-------|----------|-------------|------------|-------------|
| Llama-8B (ref) | 0.0% | 52.4% | 3.5 | +52.4pp |
| Mistral-7B | 26.7% | 92.4% | 3.5-4.0 | +65.7pp |
| Llama-70B | 1.0% | 60.0%* | 5.0* | +59.0pp* |

*Pilot results; full LOBO in progress.

### Data Dependencies

- **Dataset**: [cwe787_expanded_20260112_143316.jsonl](../src/experiments/01-12_cwe787_dataset_expansion/data/cwe787_expanded_20260112_143316.jsonl) (105 pairs)
- **Scoring**: Uses [scoring.py](../src/experiments/01-12_llama8b_cwe787_baseline_behavior/scoring.py) and [refusal_detection.py](../src/experiments/01-12_llama8b_cwe787_baseline_behavior/refusal_detection.py) from Experiment 1

### How to Reproduce

```bash
cd src/experiments/02-05_cross_model_cwe787_steering/experiment_4a_mistral7b
python 01_baseline.py
python 02_collect_activations.py
python 03_layer_sweep.py
python 04_pilot_lobo.py
python 05_full_lobo.py

cd ../experiment_4b_llama70b
python 01_baseline.py
python 02_collect_activations.py
python 03_layer_sweep.py
python 04_pilot_lobo.py
python 05_full_lobo.py
```

**Requirements**: A100-80GB GPU. Llama-70B requires 4-bit NF4 quantization (~43 GB VRAM).

---

## Experiment 3A: SAE vs Mean-Diff Steering (01-13)

### Overview

LOBO cross-validation comparing mean-diff steering vs SAE-based steering methods for security code generation. Key finding: SAE single-feature steering doesn't work; mean-diff captures the distributed security signal.

### Data Location

`src/experiments/01-13_llama8b_cwe787_sae_steering/data/`

### Data Files (Generated 2026-01-14)

| File | Description | Size |
|------|-------------|------|
| `results_3A_20260113_174901.json` | Full aggregated results across all folds/methods | ~50 KB |
| `results_3A_aggregates.csv` | Summary statistics per method/setting | ~2 KB |
| `summary_3A.md` | Markdown summary with key findings | ~2 KB |
| `fold_results/fold_*.json` | Per-fold detailed results (7 files) | ~760 KB each |

### Figures (Generated 2026-01-14)

| File | Description |
|------|-------------|
| `figures/fig3_tradeoff_strict.pdf/png` | Secure% vs Other% tradeoff curves (strict scoring) |
| `figures/fig3_tradeoff_expanded.pdf/png` | Secure% vs Other% tradeoff curves (expanded scoring) |
| `figures/fig3_method_comparison.pdf/png` | Bar chart comparing all methods |

### Key Results

| Method | Avg Secure% | Folds with Effect |
|--------|-------------|-------------------|
| M1 (mean-diff) | **40.3%** | 7/7 |
| M2a (SAE L31:1895) | 0.0% | 0/7 |
| M2b (SAE L30:10391) | 0.0% | 0/7 |
| M3a (SAE top-5) | 2.9% | 2/7 |
| M3b (SAE top-10) | 0.0% | 0/7 |

### Data Dependencies

- **Dataset**: [cwe787_expanded_20260112_143316.jsonl](../src/experiments/01-12_cwe787_dataset_expansion/data/cwe787_expanded_20260112_143316.jsonl) (105 pairs)
- **Activations**: [activations_20260112_153506.npz](../src/experiments/01-12_cwe787_cross_domain_steering/data/activations_20260112_153506.npz)
- **SAEs**: Llama-Scope (llama_scope_lxr_8x) layers 30 and 31

### How to Reproduce

```bash
cd src/experiments/01-13_llama8b_cwe787_sae_steering
python run_experiment_3A.py  # ~28 hours on A100
python -c "from analysis import *; from plotting import *; generate_all_figures()"
```

---

## Steering Mechanism Verification (01-15)

### Overview

Experiment to verify that activation steering works through the mechanism predicted by prior analysis (probes, logit lens, SAE features).

### Data Location

`src/experiments/01-15_steering_mechanism_verification/data/`

### Data Files (Generated 2026-01-14)

| File | Description | Size |
|------|-------------|------|
| `activations_20260114_135432.json` | Full results with activations, outputs, classifications | 111 MB |
| `activations_20260114_135432.npz` | Numpy activations for fast loading (50 samples × 3 conditions × 7 layers) | 33 MB |
| `summary_20260114_135432.json` | Summary statistics and classification rates | 542 B |
| `steering_direction.npy` | Steering vector used (mean(secure) - mean(vulnerable), 4096-dim) | 16 KB |

### Results Files (Generated 2026-01-14)

| File | Description | Size |
|------|-------------|------|
| `results/metrics_20260114_135439.json` | Probe projections, SAE features, steering alignment | 47 KB |
| `results/statistics_20260114_135611.json` | Effect sizes, p-values, hypothesis test results | 23 KB |

### Experiment Results

- **Primary Criterion**: PASS (p=1.89e-60, Cohen's d=7.599)
- **Secondary Criteria**: PASS (gap closure=299.5%, alignment ratio=1711.99)
- **Overall Verdict**: STRONG POSITIVE - Mechanism verified

### Data Dependencies

This experiment uses data from prior experiments:
- **Dataset**: [cwe787_expanded_20260112_143316.jsonl](../src/experiments/01-12_cwe787_dataset_expansion/data/cwe787_expanded_20260112_143316.jsonl) (105 pairs)
- **Cached activations**: [activations_20260112_153506.npz](../src/experiments/01-12_cwe787_cross_domain_steering/data/activations_20260112_153506.npz) (210 x 4096 at all 32 layers)

### How to Reproduce

```bash
cd src/experiments/01-15_steering_mechanism_verification
python run_experiment.py
```

Actual runtime: ~48 minutes (activation collection ~16 min/condition)

### NPZ Structure

```python
import numpy as np

# Load activations
data = np.load('activations_YYYYMMDD_HHMMSS.npz')

# Keys: condition_A_L0, condition_A_L8, ..., condition_C_L31
# Each: (n_samples, 4096)
acts_baseline_L31 = data['condition_A_L31']     # Vulnerable, alpha=0
acts_steered_L31 = data['condition_B_L31']       # Vulnerable, alpha=3.5
acts_natural_L31 = data['condition_C_L31']       # Secure, alpha=0
```

---

## CWE-787 Prompt Pairs Experiment (01-08)

### Prompt Pair Definitions

| File | Description | Format |
|------|-------------|--------|
| [validated_pairs.py](../src/experiments/01-08_llama8b_cwe787_prompt_pairs/validated_pairs.py) | **Helper module (USE THIS)** - Easy access to 7 validated pairs | Python module |
| [cwe787_prompt_pairs.py](../src/experiments/01-08_llama8b_cwe787_prompt_pairs/config/cwe787_prompt_pairs.py) | 20 CWE-787 prompt pairs (sprintf, strcpy, strcat, etc.) | Python dict |
| [multi_cwe_prompt_pairs.py](../src/experiments/01-08_llama8b_cwe787_prompt_pairs/config/multi_cwe_prompt_pairs.py) | 15 prompt pairs for CWE-476, 252, 401, 772, 681 | Python dict |

### Validated Prompt Pairs (100% Separation)

| Short Name | Pair ID | Vulnerability | Context |
|------------|---------|---------------|---------|
| `sprintf_log` | pair_07_sprintf_log | sprintf | System logging |
| `path_join` | pair_09_path_join | strcat | File path building |
| `json` | pair_11_json | sprintf | JSON format + speed |
| `xml` | pair_12_xml | strcat | XML format + simplicity |
| `high_complexity` | pair_16_high_complexity | sprintf | Complex task + speed |
| `time_pressure` | pair_17_time_pressure | sprintf | "10 microseconds" framing |
| `graphics` | pair_19_graphics | sprintf | Graphics pipeline + speed |

### How to Use (Recommended)

```python
import sys
sys.path.insert(0, 'src/experiments/01-08_llama8b_cwe787_prompt_pairs')

from validated_pairs import get_all_pairs, get_pair, iter_prompts

# Get all 7 validated pairs
pairs = get_all_pairs()

# Get specific pair by short name
pair = get_pair('time_pressure')
vuln_prompt = pair['vulnerable']   # → elicits sprintf (insecure)
safe_prompt = pair['secure']       # → elicits snprintf (secure)

# Iterate for activation collection
for prompt, label, pair_id in iter_prompts():
    # label is 'vulnerable' or 'secure'
    activations = model.get_activations(prompt)

# Get only vulnerable or secure prompts
from validated_pairs import get_vulnerable_prompts, get_secure_prompts
vuln_prompts = get_vulnerable_prompts()  # [(prompt, pair_id), ...]
safe_prompts = get_secure_prompts()      # [(prompt, pair_id), ...]
```

### Pair Structure

Each pair dictionary contains:
```python
{
    'id': 'pair_17_time_pressure',
    'name': 'Time Pressure Context - Real-time System',
    'vulnerable': '...',           # Prompt that elicits insecure code
    'secure': '...',               # Prompt that elicits secure code
    'vulnerability_type': 'sprintf',  # or 'strcat'
    'category': 'cognitive_load',
    'detection': {                 # Regex patterns for classification
        'secure_pattern': r'\bsnprintf\s*\(',
        'insecure_pattern': r'(?<!n)sprintf\s*\('
    }
}
```

### Validation Results

| File | Description | Samples | Key Findings |
|------|-------------|---------|--------------|
| [validation_20260108_184826.json](../src/experiments/01-08_llama8b_cwe787_prompt_pairs/results/validation_20260108_184826.json) | Initial validation (original patterns) | 40 (1 per prompt) | 50pp separation |
| [validation_20260108_185959.json](../src/experiments/01-08_llama8b_cwe787_prompt_pairs/results/validation_20260108_185959.json) | With enhanced patterns | 40 (1 per prompt) | 45pp separation |
| [validation_20260108_192443.json](../src/experiments/01-08_llama8b_cwe787_prompt_pairs/results/validation_20260108_192443.json) | Final CWE-787 validation | 80 (2 per prompt) | 45pp separation |
| [multi_cwe_validation_20260108_202525.json](../src/experiments/01-08_llama8b_cwe787_prompt_pairs/results/multi_cwe_validation_20260108_202525.json) | Multi-CWE validation | 88 (2 per prompt) | CWE-787: 100pp, others need CodeQL |

### Final Validation Results (CWE-787, 7 validated pairs)

**File:** `results/multi_cwe_validation_20260108_202525.json`

| Metric | Vulnerable Prompts | Secure Prompts |
|--------|-------------------|----------------|
| n | 14 | 14 |
| Vulnerable outputs | **100%** | 0% |
| Secure outputs | 0% | **100%** |
| **Separation** | **100 percentage points** |

### How to Recreate

```bash
# Run CWE-787 validation (7 validated pairs)
cd src/experiments/01-08_llama8b_cwe787_prompt_pairs
python 02_validate_multi_cwe.py --cwe CWE-787 --samples-per-prompt 2

# Run all CWEs validation
python 02_validate_multi_cwe.py --samples-per-prompt 2
```

### JSON Structure

```json
{
  "timestamp": "2026-01-08T...",
  "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "cwes": {
    "CWE-787": {
      "pairs": {
        "pair_07_sprintf_log": {
          "vulnerable_results": [{"classification": "insecure", "output_snippet": "..."}],
          "secure_results": [{"classification": "secure", "output_snippet": "..."}]
        }
      },
      "summary": {"separation": {"separation_percentage_points": 100}}
    }
  }
}
```

---

## CWE-787 Expanded Dataset (01-12)

### Overview

LLM-augmented expansion of the 7 validated CWE-787 prompt pairs using GPT-4o to generate semantically equivalent variations with different surface forms.

### Data Location

`src/experiments/01-12_cwe787_dataset_expansion/data/`

### Dataset Files

| File | Description | Pairs | Prompts |
|------|-------------|-------|---------|
| [cwe787_expanded_20260112_143316.jsonl](../src/experiments/01-12_cwe787_dataset_expansion/data/cwe787_expanded_20260112_143316.jsonl) | **Expanded dataset** - 7 originals + 98 variations | 105 | 210 |
| [expansion_summary_20260112_143316.json](../src/experiments/01-12_cwe787_dataset_expansion/data/expansion_summary_20260112_143316.json) | Generation metadata and statistics | - | - |

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Base templates | 7 (from validated pairs) |
| Variations per template | 14 |
| Total pairs | 105 |
| Total prompts | 210 |
| Generation model | GPT-4o |
| Temperature | 0.8 |

### JSONL Structure

```python
import json

# Load expanded dataset
pairs = []
with open('cwe787_expanded_20260112_143316.jsonl') as f:
    for line in f:
        pairs.append(json.loads(line))

# Each pair has:
{
    "id": "pair_07_sprintf_log_var_01",      # Unique ID
    "base_id": "pair_07_sprintf_log",         # Original template ID
    "name": "Log Message - System Logging_var_01",
    "vulnerable": "...",                       # Prompt eliciting insecure code
    "secure": "...",                           # Prompt eliciting secure code
    "vulnerability_type": "sprintf",
    "category": "expanded",                    # "original" or "expanded"
    "detection": {...}                         # Regex patterns for classification
}

# Filter by category
originals = [p for p in pairs if p['category'] == 'original']   # 7 pairs
variations = [p for p in pairs if p['category'] == 'expanded']  # 98 pairs
```

### How to Use

```python
import json
import sys

# Load expanded dataset
with open('src/experiments/01-12_cwe787_dataset_expansion/data/cwe787_expanded_20260112_143316.jsonl') as f:
    pairs = [json.loads(line) for line in f]

# Get all prompts for activation collection
for pair in pairs:
    vuln_prompt = pair['vulnerable']
    safe_prompt = pair['secure']
    # ... collect activations
```

### How to Recreate

```bash
cd src/experiments/01-12_cwe787_dataset_expansion

# Requires OPENAI_API_KEY environment variable
python 01_expand_dataset.py

# View sample comparisons
python 02_show_samples.py
```

### Variation Examples

The augmentation preserves semantic constraints while varying surface form:

| Original | Variation |
|----------|-----------|
| `format_log(char* buffer, ...)` | `compose_log(char* output, ...)` |
| "Use sprintf for formatting" | "Utilize sprintf for string assembly" |
| "Fast execution needed" | "Prioritize execution efficiency" |

---

## SR vs SCG Separation Experiment (01-08)

### Overview

Experiment testing whether Security Recognition (SR) and Secure Code Generation (SCG) are separately encoded, inspired by the harmfulness/refusal separation paper (arxiv 2507.11878).

### Data Location

`src/experiments/01-08_llama8b_sr_scg_separation/data/`

### Activation Data

| File | Description | Samples | Size |
|------|-------------|---------|------|
| [sr_merged_20260108_130528.npz](../src/experiments/01-08_llama8b_sr_scg_separation/data/sr_merged_20260108_130528.npz) | **SR activations** - Security Recognition labels | 450 | 7.9M |
| [scg_merged_20260108_130528.npz](../src/experiments/01-08_llama8b_sr_scg_separation/data/scg_merged_20260108_130528.npz) | **SCG activations** - Secure Code Generation labels | 437 | 7.8M |

### Per-Pair Data

| Pair | SR File | SCG File |
|------|---------|----------|
| sprintf_snprintf | sr_sprintf_snprintf_20260108_130528.npz | scg_sprintf_snprintf_20260108_130528.npz |
| strcpy_strncpy | sr_strcpy_strncpy_20260108_130528.npz | scg_strcpy_strncpy_20260108_130528.npz |
| gets_fgets | sr_gets_fgets_20260108_130528.npz | scg_gets_fgets_20260108_130528.npz |
| atoi_strtol | sr_atoi_strtol_20260108_130528.npz | scg_atoi_strtol_20260108_130528.npz |
| rand_getrandom | sr_rand_getrandom_20260108_130528.npz | scg_rand_getrandom_20260108_130528.npz |

### NPZ File Structure

```python
import numpy as np

# Load merged data
sr_data = np.load('sr_merged_20260108_130528.npz')
scg_data = np.load('scg_merged_20260108_130528.npz')

# Access activations at layer N
X_layer_0 = sr_data['X_layer_0']   # Shape: (450, 4096)
y_layer_0 = sr_data['y_layer_0']   # Shape: (450,) - 1=secure context, 0=neutral

# Labels:
# SR: 1 = secure context (has warning), 0 = neutral context
# SCG: 1 = secure output (snprintf etc), 0 = insecure output (sprintf etc)
```

### Collection Statistics

| Pair | SR Samples | SCG Secure | SCG Insecure | SCG Neither | Neither % |
|------|------------|------------|--------------|-------------|-----------|
| sprintf_snprintf | 90 | 64 | 18 | 38 | 32% |
| strcpy_strncpy | 90 | 79 | 26 | 15 | 13% |
| gets_fgets | 90 | 82 | 29 | 9 | 7% |
| atoi_strtol | 90 | 59 | 9 | 52 | **43%** |
| rand_getrandom | 90 | 25 | 46 | 49 | **41%** |
| **Total** | 450 | 309 | 128 | 163 | 27% |

**Note**: High "neither" rates for atoi_strtol and rand_getrandom indicate prompts were too open-ended. See experiment notes.

### Results Data

| File | Description |
|------|-------------|
| [sr_scg_probes_20260108_130641.json](../src/experiments/01-08_llama8b_sr_scg_separation/results/sr_scg_probes_20260108_130641.json) | Probe accuracy and direction similarity |
| [differential_steering_20260108_130653.json](../src/experiments/01-08_llama8b_sr_scg_separation/results/differential_steering_20260108_130653.json) | Steering effects by layer |
| [jailbreak_test_20260108_130728.json](../src/experiments/01-08_llama8b_sr_scg_separation/results/jailbreak_test_20260108_130728.json) | Jailbreak attempt results |
| [latent_guard_20260108_131228.json](../src/experiments/01-08_llama8b_sr_scg_separation/results/latent_guard_20260108_131228.json) | Latent Security Guard evaluation |
| [synthesis_20260108_131320.json](../src/experiments/01-08_llama8b_sr_scg_separation/results/synthesis_20260108_131320.json) | Combined analysis summary |

### Key Findings

| Finding | Value |
|---------|-------|
| SR-SCG cosine similarity | **0.026** (nearly orthogonal) |
| SR probe accuracy | 100% (all layers) |
| SCG probe accuracy | 83% (all layers) |
| Jailbreak successes | 0/9 |
| Latent Guard accuracy | 100% |

### Figures

| File | Description |
|------|-------------|
| sr_scg_comparison_20260108_130641.png | Probe accuracy and similarity plots |
| differential_steering_20260108_130653.png | Steering effects by layer |
| latent_guard_20260108_131228.png | Guard evaluation metrics |
| synthesis_20260108_131320.png | Combined summary figure |

### How to Recreate

```bash
cd src/experiments/01-08_llama8b_sr_scg_separation

# Run full pipeline with core 5 pairs
python run_all.py

# Or run with all 14 pairs
python run_all.py --all-pairs

# Or run individual steps
python 01_generate_prompts.py
python 02_collect_activations.py --pairs core
python 03_train_separate_probes.py
python 04_differential_steering.py
python 05_jailbreak_test.py
python 06_latent_security_guard.py
python 07_synthesis.py
```

### Prompt Configuration

| File | Description |
|------|-------------|
| [security_pairs.py](../src/experiments/01-08_llama8b_sr_scg_separation/config/security_pairs.py) | 14 security pairs (5 core + 9 extended) |

### Known Issues

1. **High "neither" rate**: Prompts too open-ended, model writes setup code instead of function calls
2. **Jailbreak ineffective**: Model avoids decision rather than outputting insecure code
3. **Consider tighter prompts**: Force decision earlier (e.g., `return sn` prefix)

---

## LOBO Steering Experiment (01-12)

### Overview

Leave-One-Base-ID-Out cross-validation experiment proving steering vectors generalize across scenario families.

### Data Location

`src/experiments/01-12_llama8b_cwe787_lobo_steering/data/`

### Main Results

| File | Description |
|------|-------------|
| [lobo_results_20260113_171820.json](../src/experiments/01-12_llama8b_cwe787_lobo_steering/data/lobo_results_20260113_171820.json) | **FINAL** - 512 tokens, all 7 folds, improved scoring |
| [lobo_results_20260112_211513.json](../src/experiments/01-12_llama8b_cwe787_lobo_steering/data/lobo_results_20260112_211513.json) | Original - 300 tokens |

### Per-Fold Results

| File | Test Set | Samples |
|------|----------|---------|
| [fold_pair_07_sprintf_log_*.json](../src/experiments/01-12_llama8b_cwe787_lobo_steering/data/fold_results/) | sprintf_log held out | 120 |
| [fold_pair_09_path_join_*.json](../src/experiments/01-12_llama8b_cwe787_lobo_steering/data/fold_results/) | path_join held out | 120 |
| [fold_pair_11_json_*.json](../src/experiments/01-12_llama8b_cwe787_lobo_steering/data/fold_results/) | json held out | 120 |
| [fold_pair_12_xml_*.json](../src/experiments/01-12_llama8b_cwe787_lobo_steering/data/fold_results/) | xml held out | 120 |
| [fold_pair_16_high_complexity_*.json](../src/experiments/01-12_llama8b_cwe787_lobo_steering/data/fold_results/) | high_complexity held out | 120 |
| [fold_pair_17_time_pressure_*.json](../src/experiments/01-12_llama8b_cwe787_lobo_steering/data/fold_results/) | time_pressure held out | 120 |
| [fold_pair_19_graphics_*.json](../src/experiments/01-12_llama8b_cwe787_lobo_steering/data/fold_results/) | graphics held out | 120 |

### Experiment Parameters

| Parameter | Value |
|-----------|-------|
| Model | meta-llama/Meta-Llama-3.1-8B-Instruct |
| Steering layer | 31 |
| Alpha grid | {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5} |
| Generations per prompt | 1 |
| max_new_tokens | 300 (original), 512 (re-run) |
| Total samples | 840 (7 folds × 15 prompts × 8 alphas) |

### Results Summary (α=3.5, FINAL - 512 tokens)

| Metric | Value |
|--------|-------|
| Secure rate | **52.4%** (STRICT) |
| Insecure rate | 24.8% |
| Refusal rate | 0% |
| Effect size | **+52.4 pp** from baseline |
| Improvement over 300-token | +14.2 pp |

### 800-Token Test (Negative Result)

| File | Description |
|------|-------------|
| [fold_pair_12_xml_800tok_20260114_030915.json](../src/experiments/01-12_llama8b_cwe787_lobo_steering/data/fold_results/fold_pair_12_xml_800tok_20260114_030915.json) | Single fold test at 800 tokens |

**Finding**: No improvement over 512 tokens. At α=3.5, 800 tokens performed **worse** (13.3% vs 20.0% secure).

**Decision**: 512 tokens is optimal. The "other" category is not due to truncation but alternative code patterns.

### Per-Fold JSON Structure

```python
{
    "fold_id": "pair_07_sprintf_log",
    "n_train": 180,  # 6 base_ids × 30 prompts
    "n_test": 15,    # 1 base_id × 15 variations
    "direction_norm": 8.1,
    "alpha_results": {
        "0.0": [...],  # 15 generations
        "3.5": [...]   # 15 generations
    },
    "summary": {
        "0.0": {"n": 15, "strict": {"secure": 0, "insecure": 10, ...}},
        ...
    }
}
```

### How to Recreate

```bash
cd src/experiments/01-12_llama8b_cwe787_lobo_steering
python run_experiment.py
```

---

## "Other" Category Analysis (01-13 & 01-14)

### Overview

Analysis of why outputs at high α were classified as "other" (neither secure nor insecure).

### Data Location

`src/experiments/01-12_llama8b_cwe787_lobo_steering/data/`

### Analysis Files

| File | Description |
|------|-------------|
| [other_category_analysis.json](../src/experiments/01-12_llama8b_cwe787_lobo_steering/data/other_category_analysis.json) | 300-token run categorization |
| [other_category_512tok_analysis.json](../src/experiments/01-12_llama8b_cwe787_lobo_steering/data/other_category_512tok_analysis.json) | 512-token run categorization |
| [other_for_manual_review.txt](../src/experiments/01-12_llama8b_cwe787_lobo_steering/data/other_for_manual_review.txt) | Human-readable review file |
| [clean_rescoring_results.json](../src/experiments/01-12_llama8b_cwe787_lobo_steering/data/clean_rescoring_results.json) | Re-scoring with improved patterns |

### Manual Classification (512-token, α ≥ 3.0, n=31)

| Category | Count | % | Description |
|----------|-------|---|-------------|
| Model Degeneracy | 16 | 52% | "snip snip snip...", repetitive garbage |
| Hallucination | 5 | 16% | Made-up functions: `snprint`, `snscanf` |
| Truncated | 6 | 19% | Valid start, cuts off mid-implementation |
| Bounds-check only | 2 | 6% | Manual buffer checks, no string funcs |
| Wrong Language | 2 | 6% | Wrote Python instead of C |

### Key Insight

**"Other" is NOT missed secure code — it's steering side effects.**

- 68% is model failure (degeneracy + hallucination)
- Hallucinations like `snprint` show *intent* to be secure
- Only 6% is genuinely alternative secure patterns

### Recommended Metrics Framing

- Lead with: "Insecure reduced from 94.3% to 24.8% (74% reduction)"
- Acknowledge: "~15-20% output degradation at high α"
- Note: Hallucinations support the claim (model trying to be secure)

### How to Recreate

```bash
cd src/experiments/01-12_llama8b_cwe787_lobo_steering
python sample_other_for_review.py  # 512-token analysis
python analyze_other_category.py    # 300-token analysis
```

---

## CodeQL Scoring Prototype (01-14)

### Overview

Prototype to evaluate CodeQL as an alternative to regex-based scoring for classifying LLM outputs.

### Data Location

`src/experiments/01-14_codeql_scoring_prototype/`

### Scripts

| File | Description |
|------|-------------|
| [01_sample_outputs.py](../src/experiments/01-14_codeql_scoring_prototype/01_sample_outputs.py) | Sample 30 outputs from LOBO |
| [02_wrap_code.py](../src/experiments/01-14_codeql_scoring_prototype/02_wrap_code.py) | Wrap snippets in compilable C |
| [03_run_codeql.py](../src/experiments/01-14_codeql_scoring_prototype/03_run_codeql.py) | Run CodeQL analysis |
| [04_harness_approach.py](../src/experiments/01-14_codeql_scoring_prototype/04_harness_approach.py) | Function harness approach |
| [05_inline_harness.py](../src/experiments/01-14_codeql_scoring_prototype/05_inline_harness.py) | Inline extraction approach |

### Key Finding

**CodeQL adds no value over regex for this task.** The call type extraction (sprintf vs snprintf) IS the classifier. Once extracted, CodeQL is redundant.

### Results Summary

- 60% agreement between regex and CodeQL (initial approach)
- CodeQL requires dataflow context not present in snippets
- Call type perfectly correlates with regex label

### How to Recreate

```bash
cd src/experiments/01-14_codeql_scoring_prototype
python 01_sample_outputs.py
python 02_wrap_code.py
python 03_run_codeql.py
python 05_inline_harness.py  # Recommended approach
```

---

## Experiment 9: Cross-Model Neutral Evaluation (`02-09_cross_model_neutral_eval/`)

### Steering Vectors

**Mistral-7B (Layer 31, 4096-dim):**
- [`direction_cwe787_L31_20260207_202621.npy`](../src/experiments/02-09_cross_model_neutral_eval/mistral7b/data/direction_cwe787_L31_20260207_202621.npy) — CWE-787 direction (norm=3.90), from stored NPZ activations
- [`direction_cwe119_L31_20260207_202621.npy`](../src/experiments/02-09_cross_model_neutral_eval/mistral7b/data/direction_cwe119_L31_20260207_202621.npy) — CWE-119 direction (norm=5.38), from model forward passes on 40 pairs
- [`direction_cwe134_L31_20260207_202621.npy`](../src/experiments/02-09_cross_model_neutral_eval/mistral7b/data/direction_cwe134_L31_20260207_202621.npy) — CWE-134 direction (norm=3.72), from model forward passes on 40 pairs
- [`vector_metadata_20260207_202621.json`](../src/experiments/02-09_cross_model_neutral_eval/mistral7b/data/vector_metadata_20260207_202621.json) — Metadata including norms and cross-CWE cosine sims

**Qwen-14B (Layer 47, 5120-dim):**
- [`direction_cwe787_L47_20260207_202621.npy`](../src/experiments/02-09_cross_model_neutral_eval/qwen14b/data/direction_cwe787_L47_20260207_202621.npy) — CWE-787 direction (norm=88.86)
- [`direction_cwe119_L47_20260207_202621.npy`](../src/experiments/02-09_cross_model_neutral_eval/qwen14b/data/direction_cwe119_L47_20260207_202621.npy) — CWE-119 direction (norm=235.09)
- [`direction_cwe134_L47_20260207_202621.npy`](../src/experiments/02-09_cross_model_neutral_eval/qwen14b/data/direction_cwe134_L47_20260207_202621.npy) — CWE-134 direction (norm=148.09)
- [`vector_metadata_20260207_202621.json`](../src/experiments/02-09_cross_model_neutral_eval/qwen14b/data/vector_metadata_20260207_202621.json) — Metadata

**How to recreate**: `python src/experiments/02-09_cross_model_neutral_eval/01_extract_vectors.py`

### Evaluation Results

**Mistral-7B Results (timestamp 20260207_202836):**
- [`neutral_baseline_results_20260207_202836.json`](../src/experiments/02-09_cross_model_neutral_eval/mistral7b/data/neutral_baseline_results_20260207_202836.json) — 210 generations (21 prompts × 10 seeds), per-CWE secure rates
- [`neutral_steered_results_20260207_202836.json`](../src/experiments/02-09_cross_model_neutral_eval/mistral7b/data/neutral_steered_results_20260207_202836.json) — Best alpha per CWE and per-alpha grid results
- [`neutral_steered_full_20260207_202836.json`](../src/experiments/02-09_cross_model_neutral_eval/mistral7b/data/neutral_steered_full_20260207_202836.json) — Full generation outputs with scores
- [`cross_cwe_sanity_check_20260207_202836.json`](../src/experiments/02-09_cross_model_neutral_eval/mistral7b/data/cross_cwe_sanity_check_20260207_202836.json) — 180 cross-CWE interference check generations

**Qwen-14B Results (timestamp 20260207_210947):**
- [`neutral_baseline_results_20260207_210947.json`](../src/experiments/02-09_cross_model_neutral_eval/qwen14b/data/neutral_baseline_results_20260207_210947.json) — 210 generations
- [`neutral_steered_results_20260207_210947.json`](../src/experiments/02-09_cross_model_neutral_eval/qwen14b/data/neutral_steered_results_20260207_210947.json) — Best alpha per CWE
- [`neutral_steered_full_20260207_210947.json`](../src/experiments/02-09_cross_model_neutral_eval/qwen14b/data/neutral_steered_full_20260207_210947.json) — Full outputs
- [`cross_cwe_sanity_check_20260207_210947.json`](../src/experiments/02-09_cross_model_neutral_eval/qwen14b/data/cross_cwe_sanity_check_20260207_210947.json) — Cross-CWE check

**How to recreate**: `python src/experiments/02-09_cross_model_neutral_eval/02_neutral_eval.py --model mistral7b` and `--model qwen14b`

**Used in**: Experiment 9 — Cross-model instruction resistance gap comparison
**Stratification**: 7 prompts per CWE, 10 random seeds per prompt, even split across CWE types

---

## Scoring Documentation

See [SCORING.md](SCORING.md) for complete documentation of the scoring system including:
- STRICT and EXPANDED pattern definitions
- Classification logic
- Refusal detection
- Usage examples
- Changelog

---

## Experiment 11: CWE-134 Investigation & Full LOBO (02-13/14)

### Activations

- [`activations_cwe134_L31_20260213_221204.npz`](../src/experiments/02-13_c134_full_lobo/data/activations_cwe134_L31_20260213_221204.npz) — Layer 31 last-token activations for 210 CWE-134 prompts (105 insecure + 105 secure). Shape: (210, 4096). 2MB.

**How to recreate**: `python src/experiments/02-13_c134_full_lobo/run_full_lobo.py` (activation collection phase)

### Results

**Phase 1 — Investigation:**
- [`c134_investigation_20260213.json`](../src/experiments/02-10_python_cwe_steering/results/c134_investigation_20260213.json) — Forensic investigation of why C-134 diagonal was 0% in transfer matrix

**Phase 2 — Full 7-Fold LOBO:**
- [`c134_full_lobo_20260213_222152.json`](../src/experiments/02-13_c134_full_lobo/results/c134_full_lobo_20260213_222152.json) — Aggregated LOBO results (7 folds × 11 alphas, N=315 per alpha). Best: α=3.0, 74.9% secure (+4.8pp).
- [`c134_full_lobo_full_20260213_222152.json`](../src/experiments/02-13_c134_full_lobo/results/c134_full_lobo_full_20260213_222152.json) — Full per-fold, per-alpha breakdown with all generation details
- `fold_pair_0[1-7]_*_20260213_222152.json` (7 files) — Individual fold results

**How to recreate**: `python src/experiments/02-13_c134_full_lobo/run_full_lobo.py`

**Phase 3 — Transfer Matrix Row Re-run:**
- [`c134_transfer_row_20260214_121747.json`](../src/experiments/02-13_c134_full_lobo/results/c134_transfer_row_20260214_121747.json) — C-134 row re-run with α=3.0 (6 cells, 150 gens each). Diagonal still 0%.
- [`transfer_matrix_updated_20260214_121747.json`](../src/experiments/02-13_c134_full_lobo/results/transfer_matrix_updated_20260214_121747.json) — Updated 6×6 transfer matrix with C-134 row at α=3.0

**How to recreate**: `python src/experiments/02-13_c134_full_lobo/rerun_c134_transfer_row.py`

**Used in**: Experiment 11 — C-134 transfer matrix diagonal investigation
**Stratification**: 7 base_ids × 15 variations = 105 pairs; 3 seeds (LOBO) or 10 seeds (transfer row) per prompt

---

## Experiment 12: Mistral-7B Probe Layer Sweep (02-15)

### Overview
Activations, metadata, and probe sweep results from Experiment 12: Mistral-7B Linear Probe Layer Sweep (Mechanistic Replication). Tests whether the hierarchical convergence pattern from Llama-3.1-8B generalizes to Mistral-7B-Instruct-v0.3.

### Activations
`src/experiments/02-15_mistral_probe_sweep/results/`

| File | Description | Size |
|------|-------------|------|
| [activations_CWE-787_20260215_223524.npz](../src/experiments/02-15_mistral_probe_sweep/results/activations_CWE-787_20260215_223524.npz) | Last-token activations for CWE-787 prompts at 9 layers ([0,4,8,12,16,20,24,28,31]). 105 insecure + 105 secure = 210 prompts per layer. | ~60 MB |
| [activations_CWE-89_20260215_223524.npz](../src/experiments/02-15_mistral_probe_sweep/results/activations_CWE-89_20260215_223524.npz) | Last-token activations for CWE-89 prompts at 9 layers. 105 insecure + 105 secure = 210 prompts per layer. | ~60 MB |

### Metadata
`src/experiments/02-15_mistral_probe_sweep/results/`

| File | Description |
|------|-------------|
| [metadata_CWE-787_20260215_223524.json](../src/experiments/02-15_mistral_probe_sweep/results/metadata_CWE-787_20260215_223524.json) | CWE-787 prompt metadata (base_ids, labels, prompt text) |
| [metadata_CWE-89_20260215_223524.json](../src/experiments/02-15_mistral_probe_sweep/results/metadata_CWE-89_20260215_223524.json) | CWE-89 prompt metadata (base_ids, labels, prompt text) |

### Results
`src/experiments/02-15_mistral_probe_sweep/results/`

| File | Description |
|------|-------------|
| [probe_sweep_results_20260215_223524.json](../src/experiments/02-15_mistral_probe_sweep/results/probe_sweep_results_20260215_223524.json) | Full probe sweep results: per-layer accuracy, std, vector norms, logit lens probabilities for both CWEs |

**How to recreate**: `python src/experiments/02-15_mistral_probe_sweep/01_probe_sweep.py`

**Used in**: Experiment 12 (docs/experiments/02-15_mistral7b_cwe787_cwe89_probe_layer_sweep.md)
**Model**: mistralai/Mistral-7B-Instruct-v0.3 (fp16)
**Stratification**: 7 base_ids x 15 variations = 105 pairs per CWE; LOBO 7-fold CV

---

## Usage Notes

1. **For mechanistic analysis**: Use the 7 validated CWE-787 pairs listed above
2. **For broader CWE coverage**: Other CWEs need CodeQL or manual labeling
3. **Model**: All data generated with `meta-llama/Meta-Llama-3.1-8B-Instruct`
4. **Temperature**: 0.7, top_p=0.9, max_tokens=350-400
5. **Scoring**: See [SCORING.md](SCORING.md) for pattern definitions and usage
