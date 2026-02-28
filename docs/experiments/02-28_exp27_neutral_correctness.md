# Experiment 27: Functional Correctness on Neutral Prompts

**Date**: 2026-02-28
**Models**: Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3
**Judge**: GPT-4o (gpt-4o-2024-05-13)
**CWEs tested**: CWE-787, CWE-119, CWE-134

## Research Question

Does activation steering degrade functional correctness on neutral (non-adversarial) coding prompts? Is the correctness penalty observed in Exp 25b/25d specific to adversarial prompts, or does it generalize to normal coding tasks?

**Hypothesis**: The steered model should be closer to its training distribution on neutral prompts (which naturally ask for secure code), so the correctness penalty should be smaller than on adversarial prompts.

## Methods

### Prompts
- 21 neutral evaluation prompts from the E2E pipeline
- Source: `src/experiments/02-05_cross_cwe_steering/neutral_eval/data/neutral_eval_prompts.jsonl`
- Distribution: 7 CWE-787 (buffer overflow), 7 CWE-119 (buffer over-read), 7 CWE-134 (format string)
- These are completion-style prompts: headers + function signature, model generates the body

### Steering Direction
- **Overall mean difference** (secure − vulnerable) from ALL 210 CWE-787 training pairs
- No LOBO holdout: neutral prompts are entirely out-of-distribution, no data leakage concern
- Llama direction norm: ~8.0 (extracted fresh)
- Mistral direction norm: 3.90 (from pre-computed activations)

### Models & Alphas
| Model | Alpha (steered) | Alpha (baseline) | Layer |
|-------|----------------|------------------|-------|
| Llama-3.1-8B-Instruct | 3.0 (Pareto-optimal from Exp 25c) | 0.0 | 31 |
| Mistral-7B-Instruct-v0.3 | 3.5 (best from LOBO experiments) | 0.0 | 31 |

### Generation
- temperature=0.6, top_p=0.9, max_new_tokens=512
- 1 generation per prompt per condition
- Full outputs stored, no truncation
- Total: 21 × 2 conditions × 2 models = 84 generations

### Evaluation
- **Judge**: GPT-4o via OpenAI API (temperature=0.0, max_tokens=10)
- **Critical fix**: Original prompt + model completion concatenated before sending to judge. Initial evaluation sent only the raw output (function body without header), producing a floor-effect with 62-81% INCOMPLETE across ALL conditions including baselines. With prompt context, baseline functional rates rose to 62-76%, confirming the fix was necessary.
- **Degeneration detection**: Improved detector that excludes pure whitespace repetition (which was causing false positives for code indentation)

### Rating Scale (same as Exp 25)
- **CORRECT**: Code would compile and correctly implement the intended function
- **PARTIALLY_CORRECT**: Code has minor issues but core logic is sound
- **INCORRECT**: Code has significant logical errors or would not compile
- **INCOMPLETE**: Code is truncated or missing essential parts

## Results

### Per-Condition Breakdown

| Condition | N | CORRECT | PARTIAL | INCORRECT | INCOMPLETE | Functional% |
|-----------|---|---------|---------|-----------|------------|-------------|
| Llama-8B steered (α=3.0) | 21 | 11 | 4 | 6 | 0 | **71.4%** |
| Llama-8B baseline (α=0.0) | 21 | 12 | 4 | 5 | 0 | **76.2%** |
| Mistral-7B steered (α=3.5) | 21 | 9 | 2 | 7 | 3 | **52.4%** |
| Mistral-7B baseline (α=0.0) | 21 | 10 | 3 | 2 | 6 | **61.9%** |

### Correctness Penalty: Neutral vs Adversarial

| Model | Neutral (Exp 27) | Adversarial (Exp 25b/d) | Ratio |
|-------|------------------|------------------------|-------|
| Llama-8B | **-4.8pp** | -36.0pp | ~7.5× smaller |
| Mistral-7B | **-9.5pp** | -12.0pp | ~1.3× smaller |

### Per-CWE Breakdown

| CWE | Llama Steered | Llama Baseline | Llama Diff | Mistral Steered | Mistral Baseline | Mistral Diff |
|-----|---------------|----------------|------------|-----------------|------------------|-------------|
| CWE-787 (in-dist) | 86% | 100% | **-14pp** | 86% | 86% | **0pp** |
| CWE-119 (related) | 57% | 57% | **0pp** | 43% | 71% | **-29pp** |
| CWE-134 (OOD) | 71% | 71% | **0pp** | 29% | 29% | **0pp** |

### Degeneration (Steered Outputs Only)

| Model | Degenerate | Rate | Patterns |
|-------|------------|------|----------|
| Llama-8B | 7/21 | 33% | Block duplication, EOF repetition, hex repetition |
| Mistral-7B | 2/21 | 10% | Dash repetition, block duplication |

Note: Degeneration detector was improved for this experiment to exclude pure whitespace repetition (e.g., `'    ' x3`), which was flagging normal code indentation as degenerate in the initial run.

#### Llama-8B Degenerate Details
- neutral_119_01: large block duplication
- neutral_119_03: `// EOF` repeated 23×
- neutral_119_07: large block duplication
- neutral_134_01: large block duplication
- neutral_134_04: `EOF` repeated 66×
- neutral_134_05: hex pattern `0A0D0A0D...` repeated 14×
- neutral_134_07: large block duplication

#### Mistral-7B Degenerate Details
- neutral_787_07: dashes repeated 3×
- neutral_134_04: large block duplication

### Output Length Statistics

| Condition | Min | Max | Mean | >500 chars |
|-----------|-----|-----|------|------------|
| Llama steered | 734 | 2566 | 1920 | 21/21 |
| Llama baseline | 1265 | 2426 | 1912 | 21/21 |
| Mistral steered | 540 | 1756 | 1276 | 20/21 |
| Mistral baseline | 308 | 1713 | 1280 | 20/21 |

## Key Observations (No Interpretation)

1. Neutral prompt correctness penalties are much smaller than adversarial: Llama -4.8pp (vs -36pp adversarial), Mistral -9.5pp (vs -12pp adversarial)
2. Baseline functional rates are higher on neutral prompts (~62-76%) vs adversarial (~48-60%)
3. Degeneration rate in Llama steered is 33% even on neutral prompts (vs 40% INCOMPLETE rate on adversarial Exp 25b steered)
4. Per-CWE: Penalties are concentrated on in-distribution CWEs (CWE-787 for Llama, CWE-119 for Mistral), while out-of-distribution CWE-134 shows 0pp penalty for both models
5. Initial evaluation without prompt context was invalid — demonstrates importance of providing completion context to the judge

## Interpretation (Analyst)

**The hypothesis is supported**: Steering causes much smaller correctness degradation on neutral prompts than on adversarial ones. For Llama-8B, the penalty shrinks from -36pp to -4.8pp (~7.5× smaller). For Mistral-7B, it shrinks from -12pp to -9.5pp (~1.3× smaller).

This suggests the correctness penalty is partially an artifact of the adversarial prompt structure. On deployment-relevant neutral prompts, steering toward secure patterns has a modest impact on code quality. However:
- The penalty is not zero — steering still has a measurable cost
- Degeneration remains a concern (33% for Llama on neutral prompts)
- Small sample sizes (N=21 per condition) limit statistical confidence

**CWE-specific observation**: The correctness penalty appears to concentrate on CWEs closely related to the steering direction's training data (CWE-787/119, both buffer-related). CWE-134 (format string), which is structurally different from the CWE-787 training data, shows no correctness penalty — the steering direction simply doesn't affect these prompts much.

## Methodology Note: Evaluation Context

The initial evaluation (without prompt context) produced near-zero functional rates across ALL conditions, including baselines. This was because neutral prompts are completion-style: the model output starts mid-function without the header/signature. GPT-4o judged these headless function bodies as INCOMPLETE.

The fix — concatenating the original prompt with the model's completion — restored meaningful functional rates (62-76% baseline) and is the methodologically correct approach for completion-style evaluation.

## Code

- [05_exp27_neutral_correctness.py](../../src/experiments/02-27_functional_correctness/05_exp27_neutral_correctness.py) - Generation script (both models)
- [07_exp27_evaluate_with_context.py](../../src/experiments/02-27_functional_correctness/07_exp27_evaluate_with_context.py) - Evaluation with prompt+completion context (final)
- [06_exp27_evaluate_only.py](../../src/experiments/02-27_functional_correctness/06_exp27_evaluate_only.py) - Initial evaluation without context (invalid, kept for reference)

## Results Files

- [correctness_27_20260228_013017.json](../../src/experiments/02-27_functional_correctness/results/correctness_27_20260228_013017.json) - Final results (with prompt context)
- [exp27_raw_outputs_20260227_234501.json](../../src/experiments/02-27_functional_correctness/results/exp27_raw_outputs_20260227_234501.json) - Raw generation outputs

## Reproducibility

```bash
cd src/experiments/02-27_functional_correctness
# Generate outputs (requires GPU, ~45min)
python 05_exp27_neutral_correctness.py
# Evaluate with prompt context (requires OpenAI API key)
python 07_exp27_evaluate_with_context.py
```

Requires: OpenAI API key in script, `openai` Python package, GPU for generation.
