# R16 — Does the layer-10 position-0 write explain the L40S vs A100 gap in R13?

**Status:** PLANNED (pre-registered 2026-11-22, before any compute; amended after design review round 1)
**Design review:** round 1 FIXES NEEDED — fresh subagent, 2026-11-22 — docs/reviews/11-22_llama8b_position0_gpu_gap_design.md
**Script:** `9_8_research/reviewer_experiments/exp_R16_position0/run_experiment.py`. It imports R13's script by
path, and R13's sha256 must equal `1ce5b0cc…`, the hash both R13 runs used. R13's hook is untouched.
**Jobs:** `run_a100.sbatch` (`a100_tandon`, pinned to `ga004`, R13's A100 node), then `run_l40s.sbatch`
(`l40s_public`, pinned to `gl044`, R13's L40S node) with `--dependency=afterok`. The A100 job exits 2 on any
gate failure or absent premise. The L40S job also refuses to start unless the A100 `run.json` shows that all
gates passed and the position-0 difference is nonzero.
**Decision:** Gus, 2026-11-22 ("we need to run another experiment: the position-0 zeroing experiment").

## Research question

R13's old-hook cells give different counts on the two GPUs, with identical software. Example: `post_wo/prefill`
repairs 120 of the 1,000 R8 subsets on A100 and 83 on L40S. Across the cells there are 100 class changes, all
going bug (L40S) → correct (A100). Does the **layer-10 position-0 write** cause the gap?

Why position 0: with causal attention, position 0 sees only BOS. So in exact arithmetic its value is identical
in the clean and buggy prompts, and copying it is a no-op. Measured at layer 10, post-W_O: **0.0 on L40S and
6.1e-5 (one fp16 ULP) on A100**, because the kernels' arithmetic depends on prompt length (17 vs 19 tokens).
So on A100 every R13 patch also copies a one-ULP change at position 0. In the all-steps cells it also writes
that change into every decode step.

**Prior against the hypothesis, from R13.** On A100 the true-head prompt-only cell (`pre_wo/prefill`) copies
the same 6.1e-5 at position 0, yet it differs from L40S on only 2 configs, against 40 for `post_wo/prefill`.
This is not decisive. `pre_wo` patches individual heads, while `post_wo` patches 128-dim residual blocks, so
the two differ in what a one-ULP change touches.

**Scope of the claim.** keep0 removes the clean run's position-0 value at the **receiving** end, at layer 10.
A second channel stays open: in the clean donor capture, positions 1–16 attended to the clean run's BOS keys
and values at layers 0–10. Those may also differ by a ULP on A100. So keep0 tests "the layer-10 position-0
write explains the gap", not "position 0 anywhere explains it". The L40S `donorA100` cell separates the donor
side from the receiver side.

## Simplest (boring) hypothesis, written first

The configs that differ between the GPUs sit on a decision boundary, so any tiny perturbation flips some of
them. The gap then comes from arithmetic everywhere, not from the position-0 write. The placebo cells measure
this baseline directly.

## Setup

- Model, prompts, hook, alignment, the 1,307 configs, greedy decoding, 20 tokens and fp16 are exactly R13's,
  via R13's own `Patcher`, `build_configs`, `is_correct` and `reproduction`. Same snapshot (0e9e39f2…),
  torch 2.13.0, transformers 5.15.1.
- **Only the donor tensor changes:**
  - *copy*: R13 as run.
  - *keep0* (A100): donor position 0 := the buggy run's own position 0. In prefill, position 0 is then
    written with its own value. In all-steps, decode steps (which R13 fills from donor position 0) get
    buggy0, the exact-arithmetic counterpart of clean0, so bug B stays and only the ULP goes.
  - *placebo* (both GPUs): donor position 0 := buggy0 moved one ULP at **other** coordinates of the same
    128-dim blocks as d0's nonzero entries, with the same count and the same signs. It gives the boundary
    fragility baseline.
  - *inject* (L40S): donor position 0 := L40S buggy0 + the A100's (clean0 − buggy0), computed in fp32 and cast.
    The cast is checked to reproduce the delta exactly.
  - *donorA100* (L40S): the A100's whole clean donor tensor (17 × 4,096). It splits the gap into donor and
    receiver parts.
  - *a100clean0* (L40S, only if L40S buggy0 ≠ A100 buggy0 bitwise): donor position 0 := the A100's clean0.
- **Exposure.** In `post_wo` a "head" is a 128-dim residual block. A config is *exposed* if its blocks include
  a block where d0 ≠ 0. On L40S the A100's d0 is used. Only exposed configs can receive the position-0
  difference.

| Job | GPU | Variants, each run in both `post_wo/prefill` and `post_wo/all_steps` (1,307 configs per cell) |
|---|---|---|
| 1 | A100 (ga004) | copy, keep0, placebo; saves position-0 vectors and the full clean donor |
| 2 | L40S (gl044) | copy, inject, placebo, donorA100 (+ a100clean0 if needed) |

## Gates (all recorded; any failure makes the job exit 2)

- **Capture equals live**, bitwise: the captured buggy layer-10 output at position 0 must equal the value seen
  during unpatched `generate()` prefill. The clean prompt is recorded the same way. Without this, keep0 is
  not a no-op.
- **Premise:** on A100 the position-0 difference must be nonzero. If it is zero, the job records "premise
  absent on this node" and stops. On L40S, d0 = 0 is recorded as the reverse test's premise.
- **Copy reproduces R13:** all 1,307 texts must equal R13's for that GPU. If this fails, the comparisons with
  R13's D are not interpreted. As a fallback, labelled exploratory, D′ is defined from this run's copy against
  the other GPU's R13 classes.
- **Unexposed identical:** for keep0, inject and placebo, every unexposed config's text must equal copy's.
  This doubles as the implementation check.
- **Inject exactness** and the node and GPU-string match with R13 are recorded.

## Pre-registered decision rules

D is the set of configs whose class differs between R13's L40S and A100 runs in that cell: 40 in `prefill`
(37 of them R8 subsets) and 60 in `all_steps` (58 R8). Rules are computed on **exposed D**.

- **keep0 (A100) and inject (L40S):** r is the fraction of exposed D that takes the other GPU's class.
  N_new is the number of configs outside D whose class changes relative to this run's copy. Flips outside D
  are reported in each direction.
  - No exposed D config → "refuted by construction".
  - r ≥ 0.8, N_new ≤ 4 and r − r_placebo ≥ 0.5 → "explains the gap".
  - r ≤ 0.2 → "does not explain the gap", or, if N_new > 4, "perturbs, does not explain".
  - Otherwise → "explains part of the gap (r = …)".
- Both keep0 and inject "explain" → "shown from both sides".
- **donorA100 (descriptive):** r on all of D. High r means the gap comes from the donor's computation;
  low r means it comes from the receiving run's arithmetic.
- Also reported:
  - for `all_steps`, the R8/R1 reproduction agreement of every variant (A100 copy 1,000/200/60; L40S copy
    942/199/59);
  - R8-subset correct counts (`prefill` 120 vs 83);
  - the number of variant texts equal to the other GPU's R13 texts.

## Compute estimate

A100: 3 variants × 2 cells; L40S: 4 or 5 variants × 2 cells. That is about 14 cells × ~620 s ≈ 2.4 GPU-h,
split across two GPUs. Walltime 3 h each. Pinning to the R13 nodes may lengthen the queue wait.

## Limitations known before the run

- The donor-side channel described above (the clean run's BOS keys and values reaching positions 1–16) is not
  removed by keep0. donorA100 bounds it from the L40S side.
- One node per GPU type, pinned to R13's nodes.

## Results

(none yet)
