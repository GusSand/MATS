# R16 — Does position 0 explain the L40S vs A100 gap in R13?

**Status:** PLANNED (pre-registered 2026-11-22, before any compute)
**Script:** `9_8_research/reviewer_experiments/exp_R16_position0/run_experiment.py` (imports R13's script
by path; R13's sha256 must equal `1ce5b0cc…`, the one both R13 runs used)
**Jobs:** `run_a100.sbatch` (`a100_tandon`), then `run_l40s.sbatch` (`l40s_public`, needs job 1's delta file).
**Decision:** Gus, 2026-11-22 ("we need to run another experiment: the position-0 zeroing experiment").

## Research question

R13's old-hook cells give different counts on the two GPUs with identical software. Example: `post_wo/prefill`
repairs 120 of the 1,000 R8 subsets on A100 and 83 on L40S. Every class change goes from bug (L40S) to
correct (A100). Is the cause the patch at **position 0**?

Why position 0: with causal attention, position 0 sees only BOS. So in exact arithmetic every head's output
at position 0 is identical in the clean and buggy prompts, and copying it is a no-op. Measured at layer 10
(R13 `bos_max_abs_diff_clean_vs_buggy`): **0.0 on L40S, 6.1e-5 (one fp16 ULP) on A100**. The kernels'
arithmetic depends on prompt length (17 vs 19 tokens). So on A100 every patch also copies a one-ULP change
at position 0, and in the all-steps cells writes it into every decode step.

## Simplest (boring) hypothesis, written first

The gap comes from arithmetic everywhere (every position and every layer runs slightly differently on the
two GPUs), not from the position-0 copy. Skipping position 0 would then change little.

## Setup

- Model, prompts, hook, alignment, configs (1,307), greedy, 20 tokens, fp16: exactly R13's, via R13's own
  `Patcher`, `build_configs`, `is_correct`. Same snapshot (0e9e39f2…), torch 2.13.0, transformers 5.15.1.
- **The only change is the donor tensor's position 0.** R13's hook is untouched.
  - *copy* (R13 as run): donor = clean activations.
  - *keep0*: donor position 0 := the buggy run's own position-0 activation (layer 10, post-W_O). In
    prefill, the patch then writes the buggy run's own value at position 0, which leaves position 0
    unpatched. Positions 1–16 are patched as before. In all-steps cells, decode steps (which R13 fills from
    donor position 0) get the buggy run's own position-0 value instead of the clean one.
  - *inject* (L40S only): donor position 0 := L40S buggy position 0 + (A100 clean − A100 buggy) at
    position 0, computed in fp32 and cast to fp16. This puts the A100's one-ULP difference onto the L40S.
- The captured buggy position 0 must equal the live buggy run's position 0. R13's identity gate already
  shows that capture and generation are bitwise consistent.

| Job | GPU | Cells (each 1,307 configs) |
|---|---|---|
| 1 | A100 | `post_wo/prefill` copy (gate), `post_wo/prefill` keep0, `post_wo/all_steps` copy (gate), `post_wo/all_steps` keep0; saves position-0 vectors |
| 2 | L40S | `post_wo/prefill` copy (gate), `post_wo/prefill` inject, `post_wo/all_steps` copy (gate), `post_wo/all_steps` inject |

## Gates

- **Environment:** each *copy* cell must reproduce, output for output (all 1,307 texts), the same cell
  from R13 on that GPU: `results/r13_24084222` (A100) and `results/r13_24014365` (L40S). Fail → that GPU's
  comparisons are not interpreted (node or library drift); INBOX.
- **Position-0 difference recorded:** max |clean − buggy| at position 0 (layer 10, post-W_O) on each GPU. On
  L40S, if it is exactly 0, keep0 is bitwise identical to copy and is not run (stated, not assumed: checked
  in job 2).
- **R13 script hash** equals `1ce5b0cc2e2cdc74c21de37de5340cd00f5b14e27ae13cf6893d2e09f280348d`.

## Pre-registered decision rules

Let **D** = the configurations whose class differs between R13's L40S and A100 runs in that cell (from the
two R13 result files; `post_wo/prefill` has 40, `post_wo/all_steps` 60, all bug → correct).

- **A100 keep0, per cell:** r = fraction of D that take their L40S class; N_new = configs outside D whose
  class changes relative to A100 copy.
  - r ≥ 0.8 and N_new ≤ 4 → "position 0 explains the GPU gap in this cell".
  - r ≤ 0.2 → "position 0 does not explain the gap; the difference comes from arithmetic elsewhere".
  - otherwise → "position 0 explains part of the gap (r = …)".
  Also reported: the R8-subset count (A100 copy 120, L40S 83 for `post_wo/prefill`), and for `all_steps` the
  R8 reproduction agreement (A100 1,000, L40S 942).
- **L40S inject (reverse test):** r_rev = fraction of D that take their A100 class; N_new as above relative
  to L40S copy. Same thresholds, reading "the A100's position-0 difference reproduces the gap on L40S".
- Both directions pass → "shown from both sides".

## Compute estimate

8 cells × 1,307 ≈ 10,500 generations at ~0.45 s ≈ 1.3 GPU-h, split across two GPUs. Walltime 2 h each.

## Limitations known before the run

- Only layer 10's post-W_O position 0 is changed. The A100's other layers also compute position 0
  slightly differently, but those differences sit in both the patched and unpatched runs alike.
- One GPU node per type; the copy gates check that it behaves like R13's nodes.

## Results

(none yet)
