# Experiment 24: Mistral-Small-24B Logit Lens

**Date**: 2026-02-27
**Model**: mistralai/Mistral-Small-24B-Instruct-2501
**Task**: Track P("sn") emergence across all 40 layers

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Mistral-Small-24B-Instruct-2501 |
| Layers | 40 |
| Quantization | None (fp16) |
| Primary token | "sn" (first part of snprintf split) |
| Secondary tokens | "sprintf", "printf", and 15 others |
| Static prompts | 2 (secure with snprintf hint, vulnerable without) |
| Dataset prompts | 5 secure/vulnerable pairs from CWE-787 |
| Prompt style | Raw completion (no chat template) |

## Key Finding: Distributed Emergence Pattern (Matches Mistral-7B)

P("sn") on the secure static prompt shows:
- Near zero through L25 (~64% depth)
- Gradual rise L26-L34 (0.05% to 3.6%)
- **Peak at L35 = 46.6%** (rank 0, top-1 predicted token)
- Oscillation in final layers: L36=23.7%, L37=36.1%, L38=27.0%, L39=6.3%

This matches the **Mistral-7B distributed pattern**, NOT the Llama sudden-emergence pattern.

## Static Prompt Results — P("sn") Trajectory

### Secure Prompt

| Layer | Depth% | P(sn) | Rank | P(sprintf) | Top-1 Token |
|-------|--------|-------|------|------------|-------------|
| 0 | 0.0% | 0.000001 | 118799 | 0.000002 | 随着时间的 |
| 5 | 12.8% | 0.000000 | 125474 | 0.000001 | *„ |
| 10 | 25.6% | 0.000000 | 130984 | 0.000001 | „* |
| 15 | 38.5% | 0.000001 | 120627 | 0.000001 | Oosten |
| 18 | 46.2% | 0.000007 | 38612 | 0.000004 | cerem |
| 22 | 56.4% | 0.000020 | 8628 | 0.000011 | mér |
| 25 | 64.1% | 0.000104 | 389 | 0.000011 | mér |
| 26 | 66.7% | 0.000458 | 15 | 0.000133 | mér |
| 28 | 71.8% | 0.001127 | 5 | 0.000072 | mér |
| 29 | 74.4% | 0.007382 | 0 | 0.000060 | **sn** |
| 31 | 79.5% | 0.006223 | 0 | 0.000155 | **sn** |
| 32 | 82.1% | 0.023221 | 0 | 0.000137 | **sn** |
| 34 | 87.2% | 0.035658 | 1 | 0.000770 | char |
| **35** | **89.7%** | **0.465902** | **0** | 0.000194 | **sn** |
| 36 | 92.3% | 0.236686 | 0 | 0.000366 | **sn** |
| 37 | 94.9% | 0.361050 | 1 | 0.000113 | **sn** |
| 38 | 97.4% | 0.269832 | 0 | 0.000996 | **sn** |
| 39 | 100% | 0.062883 | 6 | 0.018881 | // |

### Vulnerable Prompt

| Layer | Depth% | P(sn) | Rank | P(sprintf) | Top-1 Token |
|-------|--------|-------|------|------------|-------------|
| 0-17 | 0-44% | ~0.000001 | >120000 | ~0.000001 | various |
| 25 | 64.1% | 0.000088 | 542 | 0.000008 | mér |
| 29 | 74.4% | 0.008652 | 0 | 0.000062 | **sn** |
| 32 | 82.1% | 0.005616 | 2 | 0.000140 | char |
| **35** | **89.7%** | **0.033545** | **3** | 0.000242 | size |
| 37 | 94.9% | 0.096761 | 2 | 0.000504 | size |
| 39 | 100% | 0.013549 | 9 | 0.017128 | node |

## Secure vs Vulnerable Differentiation

| Layer | Depth% | Sec P(sn) | Vul P(sn) | Diff | Notes |
|-------|--------|-----------|-----------|------|-------|
| 29 | 74.4% | 0.74% | 0.87% | -0.13% | Vuln slightly higher |
| 32 | 82.1% | 2.32% | 0.56% | **+1.76%** | First clear divergence |
| 34 | 87.2% | 3.57% | 0.91% | +2.65% | |
| **35** | **89.7%** | **46.59%** | **3.35%** | **+43.24%** | **Peak divergence** |
| 37 | 94.9% | 36.11% | 9.68% | +26.43% | |
| 39 | 100% | 6.29% | 1.35% | +4.93% | Attenuated |

## Cross-Model Emergence Comparison

| Model | Params | Total Layers | Peak Layer | Depth% | Pattern | Peak P(secure token) |
|-------|--------|-------------|------------|--------|---------|---------------------|
| Mistral-7B | 7B | 32 | ~L28 | 87.5% | Distributed | ~high |
| **Mistral-24B** | **24B** | **40** | **L35** | **89.7%** | **Distributed** | **46.6%** |
| Llama-8B | 8B | 32 | L31 | 96.9% | Sudden | 37% |
| Llama-70B | 70B | 80 | ~L75 | 93.8% | Late | ~2% |

Key: Mistral family shows distributed emergence peaking at ~87-90% depth, while Llama family shows sudden emergence at 94-97% depth.

## Dataset Prompt Results

Dataset prompts showed near-zero P(sprintf) and P(sn) across all layers. This is expected because most prompts could not be properly truncated before the critical function call (the `truncate_before_call` function didn't find the target in many cases, falling back to full code where the next token is end-of-code, not a function call).

The static prompts are more informative for emergence analysis since they're carefully crafted to end right before the function call.

## Code

- [03_logit_lens.py](../../src/experiments/02-27_mistral24b_cwe787_lobo/03_logit_lens.py) - Logit lens analysis script

## Results Files

- `src/experiments/02-27_mistral24b_cwe787_lobo/results/logit_lens_24b_20260227_170823.json` - Full results
