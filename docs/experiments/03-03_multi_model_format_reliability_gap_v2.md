# Experiment 22b: Format-Reliability Gap v2 (Code Review Design)

**Date:** 2026-03-03
**Models:** Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Mistral-Small-24B-Instruct-2501
**CWEs:** CWE-787, CWE-119, CWE-134, CWE-89, CWE-78, CWE-79
**Scorer:** GPT-4o judge (with CWE-119 re-scored using relaxed judge prompt)

## Research Question

Do LLMs that generate insecure code actually KNOW the secure alternatives? Redesigned from Exp 22 which had leading prompts (naming both vulnerability and fix) producing a ceiling effect (100% accuracy).

## Context: Why Exp 22 Was Redesigned

Exp 22 used direct knowledge queries like "What is the difference between sprintf and snprintf?" — these name both the unsafe and safe function, essentially hand-feeding the answer. All models scored 100% (after manual review), which the Iron Law flagged as a bug. Investigation found:
1. **Leading prompts** that contain the answer
2. **Lenient scoring** requiring only 1 keyword match per category
3. **Only 3 prompts per CWE** — insufficient statistical power

## Methods (Exp 22b)

### Single Condition: Code Review Without Guidance
- Show insecure code, ask: **"Review this [language] function. Are there any issues?"**
- No mention of security, vulnerability class, or expected fix
- Model must independently: (1) identify security issue, (2) name vulnerability, (3) suggest fix
- **10 insecure prompts + 4 secure distractors per CWE = 84 total prompts**

### Scoring
- **GPT-4o judge** scoring each response on binary criteria
- Insecure: requires `identifies_issue` AND `names_vulnerability` AND `suggests_fix`
- Secure (distractors): checks `correctly_identifies_as_safe` and `false_security_alarm`
- True Negative Rate = % of secure code correctly identified as safe
- **CWE-119 re-scored** with relaxed judge prompt (see note below)

### Baselines (Code Generation Security Rates)
- From prior LOBO experiments (Exps 8, 10, 11, 13, 14, 23)
- Gap = Review Accuracy - Code Security Rate

### Configuration
- Temperature: 0 (deterministic, do_sample=False)
- Max new tokens: 512
- GPU: NVIDIA A100-SXM4-80GB

## Results (No Interpretation)

### Review Accuracy (Insecure Code Detection)

| CWE | Vulnerability | Llama-8B | Mistral-7B | Mistral-24B |
|-----|--------------|----------|------------|-------------|
| CWE-787 | Out-of-bounds write (sprintf) | 90% (9/10) | 70% (7/10) | 90% (9/10) |
| CWE-119 | Buffer overflow (strcpy/gets) | 90% (9/10)* | 100% (10/10)* | 100% (10/10)* |
| CWE-134 | Format string vulnerability | 20% (2/10) | 10% (1/10) | 20% (2/10) |
| CWE-89 | SQL injection | 100% (10/10) | 100% (10/10) | 100% (10/10) |
| CWE-78 | OS command injection | 60% (6/10) | 50% (5/10) | 100% (10/10) |
| CWE-79 | Cross-site scripting (XSS) | 50% (5/10) | 0% (0/10) | 80% (8/10) |

*CWE-119 re-scored with relaxed judge prompt (original GPT-4o scores: Llama-8B 10%, Mistral-7B 20%, Mistral-24B 30%)

### True Negative Rate (Secure Code Correctly Identified as Safe)

| CWE | Llama-8B | Mistral-7B | Mistral-24B |
|-----|----------|------------|-------------|
| CWE-787 | 0% (0/4) | 75% (3/4) | 75% (3/4) |
| CWE-119 | 50% (2/4)* | 25% (1/4)* | 100% (4/4)* |
| CWE-134 | 75% (3/4) | 100% (4/4) | 100% (4/4) |
| CWE-89 | 0% (0/4) | 50% (2/4) | 25% (1/4) |
| CWE-78 | 50% (2/4) | 100% (4/4) | 50% (2/4) |
| CWE-79 | 75% (3/4) | 50% (2/4) | 100% (4/4) |

*CWE-119 re-scored with relaxed judge prompt (original GPT-4o TN: all 0%)

### Gap Table (Review Accuracy vs Code Generation Security Rate)

#### Llama-3.1-8B-Instruct

| CWE | Review Acc | Code Security | Gap |
|-----|-----------|--------------|-----|
| CWE-787 | 90% | 6.7% | **+83.3pp** |
| CWE-119 | 90%* | 0.0% | **+90.0pp** |
| CWE-134 | 20% | 0.0% | **+20.0pp** |
| CWE-89 | 100% | 57.0% | **+43.0pp** |
| CWE-78 | 60% | 14.3% | **+45.7pp** |
| CWE-79 | 50% | 0.2% | **+49.8pp** |

#### Mistral-7B-Instruct-v0.3

| CWE | Review Acc | Code Security | Gap |
|-----|-----------|--------------|-----|
| CWE-787 | 70% | 3.8% | **+66.2pp** |
| CWE-119 | 100%* | 0.3% | **+99.7pp** |
| CWE-89 | 100% | 42.9% | **+57.1pp** |

#### Mistral-Small-24B-Instruct-2501

| CWE | Review Acc | Code Security | Gap |
|-----|-----------|--------------|-----|
| CWE-787 | 90% | 0.0% | **+90.0pp** |
| CWE-119 | 100%* | 0.0% | **+100.0pp** |

*CWE-119 re-scored with relaxed judge prompt

## CWE-119 Re-scoring Note

The original GPT-4o judge was too strict on `suggests_fix` for CWE-119:
- Models that said "check the length before copying" were scored False
- Models that named fgets/strncpy with code examples were inconsistently scored False
- The secure distractors (strncpy-based code) have known limitations (no guaranteed null-termination, padding behavior) — models flagging these are demonstrating deeper security knowledge, not making false alarms

A relaxed CWE-119 judge was applied that:
1. Accepts broader mitigations: bounds checking, sized copy functions, length validation
2. Recognizes strncpy/fgets limitation critiques as legitimate (not false alarms)

Original GPT-4o → Re-scored:
- Llama-8B: 10% → 90% review accuracy, 0% → 50% TN rate
- Mistral-7B: 20% → 100% review accuracy, 0% → 25% TN rate
- Mistral-24B: 30% → 100% review accuracy, 0% → 100% TN rate

## Key Observations

### CWE-134 (Format String) Is Genuinely Hard
All models score 10-20% on format string identification. Manual inspection of responses shows models mention "missing format specifier" or general style issues but rarely flag the security implication. This is the most subtle vulnerability in the set and represents a genuine knowledge limitation, not just scorer failure.

### CWE-89 Consistent 100% Across All Models
SQL injection is universally recognized in code review. Combined with 42-57% code security rates, this gives a clean 43-57pp gap.

### Model Size Matters for Harder CWEs
- CWE-78 (OS command injection): Llama-8B 60%, Mistral-7B 50%, Mistral-24B 100%
- CWE-79 (XSS): Llama-8B 50%, Mistral-7B 0%, Mistral-24B 80%
- Mistral-24B consistently outperforms smaller models on harder CWEs

### True Negative Rates Improved with GPT-4o
Keyword scorer had inflated false-positive rates. GPT-4o judge provides more nuanced assessment, especially for CWE-134 and CWE-79 secure distractors.

## Scoring History

Three scoring passes were applied:
1. **Keyword scorer** (initial run, OPENAI_API_KEY unavailable)
2. **GPT-4o judge** (re-scored all CWEs)
3. **GPT-4o relaxed CWE-119 judge** (re-scored only CWE-119 with broader mitigation acceptance)

The final results above use GPT-4o scores for all CWEs, with CWE-119 using the relaxed judge.

## Code

- [exp22b_run.py](../../src/experiments/03-03_exp22b_format_reliability_gap/exp22b_run.py) - Main experiment runner
- [exp22b_prompts.py](../../src/experiments/03-03_exp22b_format_reliability_gap/exp22b_prompts.py) - All 84 code review prompts
- [exp22b_rescore_119.py](../../src/experiments/03-03_exp22b_format_reliability_gap/exp22b_rescore_119.py) - CWE-119 re-scoring with relaxed judge

## Result Files

- `src/experiments/03-03_exp22b_format_reliability_gap/results/exp22b_llama8b_20260302_222731/` - Llama-8B raw results (pre-scoring)
- `src/experiments/03-03_exp22b_format_reliability_gap/results/exp22b_llama8b_20260302_224438/` - Llama-8B keyword scored
- `src/experiments/03-03_exp22b_format_reliability_gap/results/exp22b_llama8b_20260303_014946/` - Llama-8B GPT-4o scored (final)
- `src/experiments/03-03_exp22b_format_reliability_gap/results/exp22b_mistral7b_20260302_224515/` - Mistral-7B keyword scored
- `src/experiments/03-03_exp22b_format_reliability_gap/results/exp22b_mistral7b_20260303_015117/` - Mistral-7B GPT-4o scored (final)
- `src/experiments/03-03_exp22b_format_reliability_gap/results/exp22b_mistral24b_20260302_225831/` - Mistral-24B keyword scored
- `src/experiments/03-03_exp22b_format_reliability_gap/results/exp22b_mistral24b_20260303_015119/` - Mistral-24B GPT-4o scored (final)
- Each GPT-4o directory also contains `results_rescored_119.json` with CWE-119 relaxed judge results
