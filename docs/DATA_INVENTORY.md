# Data Inventory

This document tracks all datasets created during experiments.

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

## Scoring Documentation

See [SCORING.md](SCORING.md) for complete documentation of the scoring system including:
- STRICT and EXPANDED pattern definitions
- Classification logic
- Refusal detection
- Usage examples
- Changelog

---

## Usage Notes

1. **For mechanistic analysis**: Use the 7 validated CWE-787 pairs listed above
2. **For broader CWE coverage**: Other CWEs need CodeQL or manual labeling
3. **Model**: All data generated with `meta-llama/Meta-Llama-3.1-8B-Instruct`
4. **Temperature**: 0.7, top_p=0.9, max_tokens=350-400
5. **Scoring**: See [SCORING.md](SCORING.md) for pattern definitions and usage
