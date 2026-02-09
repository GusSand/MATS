# Experiment 9b: Probe-Then-Steer Architecture

**Date**: 2026-02-09
**Model**: meta-llama/Meta-Llama-3.1-8B-Instruct
**GPU**: A100-80GB
**Dataset**: 21 neutral C-language prompts (CWE-787, CWE-119, CWE-134)

## Research Question

Can we reduce the ~100% generation overhead observed in Exp 8.5 by decoupling probe classification from the generation loop and replacing per-token Python hooks with hook-free steering methods?

## Architecture

Two-phase pipeline:
1. **Probe pass** (~28ms): Single forward pass extracts Layer 31 activations, binary probe classifies "buffer" vs "format_string"
2. **Steered generation**: Generate with steering vector applied via hook-free method (no Python callbacks during generation)

## Steering Methods Tested

| Option | Method | Description |
|--------|--------|-------------|
| A | Monkey-patch | Replace `layer.forward` with Python wrapper |
| B | torch.compile | Compile the steered forward with kernel fusion |
| C | Layernorm bias | Patch `post_attention_layernorm.forward` |
| D | Weight bias | Add steering vector as `mlp.down_proj.bias` (zero Python wrapper) |

## Key Finding: Exp 8.5 Overhead Was a Measurement Artifact

The ~100% overhead reported in Exp 8.5 was caused by **unequal token counts**, not by hook dispatch overhead.

**Token count diagnostic:**
- Baseline (no steering): generates **32/64** tokens (hits EOS early)
- Steered (any method): generates **64/64** tokens (full max_new_tokens)

This creates an apparent 2x overhead that has nothing to do with hooks or Python wrappers. When token counts are equalized with `min_new_tokens=64`:

## Benchmark Results (min_new_tokens=64, 50 iterations)

| Method | Mean (ms) | Std (ms) | P95 (ms) | Overhead |
|--------|-----------|----------|----------|----------|
| Baseline (no steering) | 1522.7 | 20.3 | 1556.1 | +0.0% |
| Hook-based (Exp 8.5) | 1569.5 | 88.6 | 1741.4 | **+3.1%** |
| Full hook pipeline | 1547.3 | 14.0 | 1573.3 | +1.6% |
| Probe-then-steer A (monkeypatch) | 1544.4 | 22.1 | 1582.6 | **+1.4%** |
| Probe-then-steer B (compiled) | 1551.9 | 78.3 | 1569.0 | +1.9% |
| Probe-then-steer D (weight_bias) | 1548.8 | 18.3 | 1580.5 | +1.7% |
| Persistent weight_bias | 1525.5 | 24.7 | 1573.4 | **+0.2%** |
| Persistent monkeypatch | 1515.9 | 21.4 | 1557.6 | **-0.4%** |

All methods pass the <10% overhead target. The actual overhead from activation steering is ~1-3%, regardless of implementation approach.

## Routing Accuracy

| Prompt | CWE | Predicted | Confidence | Status |
|--------|-----|-----------|------------|--------|
| neutral_787_01 | CWE-787 | buffer | 1.000 | OK |
| neutral_787_02 | CWE-787 | buffer | 1.000 | OK |
| neutral_787_03 | CWE-787 | buffer | 1.000 | OK |
| neutral_787_04 | CWE-787 | buffer | 1.000 | OK |
| neutral_787_05 | CWE-787 | format_string | 0.847 | **MISS** |
| neutral_787_06 | CWE-787 | buffer | 1.000 | OK |
| neutral_787_07 | CWE-787 | buffer | 0.971 | OK |
| neutral_119_01-07 | CWE-119 | buffer | 0.917-1.000 | ALL OK |
| neutral_134_01-07 | CWE-134 | format_string | 0.607-1.000 | ALL OK |

**Accuracy: 20/21 (95.2%)** — matches Exp 8.5

## E2E Security Validation (21 prompts × 10 seeds)

| CWE | Secure Rate | Secured/Total | Insecure | None | Routing |
|-----|------------|---------------|----------|------|---------|
| CWE-787 | 98.6% | 69/70 | 1 | 0 | 6/7 |
| CWE-119 | 67.1% | 47/70 | 12 | 11 | 7/7 |
| CWE-134 | 100.0% | 70/70 | 0 | 0 | 7/7 |
| **Overall** | **88.6%** | **186/210** | 13 | 11 | **20/21** |

**Exactly matches Exp 8.5** baseline (88.6%, delta = -0.0pp).

## Success Criteria

| Criterion | Target | Result | Status |
|-----------|--------|--------|--------|
| Generation overhead | <10% | +1.4% (monkeypatch) | **PASS** |
| Routing accuracy | >=95.2% | 95.2% | **PASS** |
| Overall secure rate | >=87% | 88.6% | **PASS** |

## Errors and Fixes During Development

1. **IndexError: too many indices for tensor of dimension 2** — During generation decode steps, tensors are 2D `(batch, hidden)` not 3D `(batch, seq_len, hidden)`. Fixed by adding `if h.dim() == 3` check.

2. **TypeError: can only concatenate tuple (not "Tensor") to tuple** — `output[1:]` fails on `BaseModelOutputWithPast`. Fixed by using in-place tensor operations (`h.add_()`, `h[:, -1, :] +=`) and returning the original output object.

3. **torch.compile RuntimeError with CUDA graphs** — `mode="reduce-overhead"` uses CUDA graphs incompatible with dynamic KV cache. Fixed by using default compile mode.

4. **100% overhead was token count confound** — Baseline hits EOS at ~32 tokens, steered models generate all 64. Fixed by adding `min_new_tokens=64` to equalize.

## Configuration

```python
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LAYER = 31
ALPHAS = {"buffer": 4.0, "format_string": 1.0}
SEEDS = [42, 123, 456, 789, 1000, 1111, 2222, 3333, 4444, 5555]
```

## Code

- [probe_router.py](../../src/experiments/02-09b_probe_then_steer/probe_router.py) — Binary probe router (single forward pass classification)
- [steered_generator.py](../../src/experiments/02-09b_probe_then_steer/steered_generator.py) — Hook-free steered generation (Options A, B, C, D)
- [benchmark.py](../../src/experiments/02-09b_probe_then_steer/benchmark.py) — Timing benchmark (8 conditions × 50 iterations)
- [e2e_pipeline.py](../../src/experiments/02-09b_probe_then_steer/e2e_pipeline.py) — E2E security validation (21 prompts × 10 seeds)

## Result Files

- `results/benchmark_results_20260209_232132.json` — Final benchmark summary
- `results/benchmark_full_20260209_232132.json` — Full timing data (all 50 iterations per condition)
- `results/e2e_results_20260209_233229.json` — E2E validation summary
- `results/e2e_full_20260209_233229.json` — Full generation outputs
