# R16 — Does the layer-10 position-0 write explain the L40S vs A100 gap in R13?

**Status:** PLANNED (pre-registered 2026-11-22, before any compute; amended after design review rounds 1–4)
**Design review:** round 1 FIXES NEEDED — fresh subagent, 2026-11-22 — docs/reviews/11-22_llama8b_position0_gpu_gap_design.md
**Design review:** round 2 FIXES NEEDED — fresh subagent, 2026-11-22 — docs/reviews/11-22_llama8b_position0_gpu_gap_design_v2.md
**Design review:** round 3 FIXES NEEDED — fresh subagent, 2026-11-22 — docs/reviews/11-22_llama8b_position0_gpu_gap_design_v3.md
**Design review:** round 4 FIXES NEEDED — fresh subagent, 2026-11-22 — docs/reviews/11-22_llama8b_position0_gpu_gap_design_v4.md
**Design review:** PASS (round 5) — fresh subagent, 2026-11-22 — docs/reviews/11-22_llama8b_position0_gpu_gap_design_v5.md
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
  - *placebo* (both GPUs): built on **this GPU's copy donor** (clean0). It applies the test variant's own
    per-coordinate steps at **other** coordinates of the same 128-dim blocks: −d0 on A100, mirroring keep0,
    and +dA on L40S, mirroring inject. A target coordinate qualifies only if the step lands there exactly
    and nonzero, i.e. fp16(x + s) − x = s. Every move is recorded. A source is unplaceable when its step is
    finer than the fp16 grid of every free target in its block; this happens to small-magnitude sources
    even when the difference is sparse; a difference with about half of each block nonzero exceeds the
    tolerance. **The placebo is valid if the unplaced steps carry at most 1% of the
    total |step|.** The unplaced count and fraction are recorded. An invalid placebo is not a gate: it only
    blocks positive labels. It gives the
    boundary-fragility baseline. Round 2 caught a first version built on buggy0, which on A100 also removed d0.
  - *inject* (L40S): donor position 0 := L40S buggy0 + the A100's (clean0 − buggy0), computed in fp32 and cast.
    The cast is checked to reproduce the delta exactly.
  - *donorA100* (L40S): the A100's whole clean donor tensor (17 × 4,096). It splits the gap into donor and
    receiver parts.
  - *a100clean0* (L40S, only if L40S buggy0 ≠ A100 buggy0 bitwise): donor position 0 := the A100's clean0.
    Descriptive, like donorA100.
- **Exposure.** In `post_wo` a "head" is a 128-dim residual block. A config is *exposed* if its blocks include
  a block where d0 ≠ 0. On L40S the A100's d0 is used. Only exposed configs can receive the position-0
  difference.

| Job | GPU | Variants, each run in both `post_wo/prefill` and `post_wo/all_steps` (1,307 configs per cell) |
|---|---|---|
| 1 | A100 (ga004) | copy, keep0, placebo; saves position-0 vectors and the full clean donor |
| 2 | L40S (gl044) | copy, inject, placebo, donorA100 (+ a100clean0 if needed) |

## Gates (all recorded; any failure makes the job exit 2, and `afterok` then blocks the L40S job)

- **Capture equals live**, bitwise: the captured buggy layer-10 output at position 0 must equal the value seen
  during unpatched `generate()` prefill. The clean prompt is recorded the same way. Without this, keep0 is
  not a no-op.
- **Premise:** on A100 the position-0 difference must be nonzero. If it is zero, the job records "premise
  absent on this node", saves the position-0 vectors and stops. On L40S, d0 must be exactly 0; otherwise
  the L40S copy already carries its own difference and inject is not the reverse test.
- **Copy reproduces R13:** all 1,307 texts must equal R13's for that GPU. If this fails, the comparisons with
  R13's D are not interpreted. As a fallback, labelled exploratory, D′ is defined from this run's copy against
  the other GPU's R13 classes.
- **Unexposed identical:** for keep0, inject and placebo, every unexposed config's text must equal copy's.
  This doubles as the implementation check.
- **Inject exactness:** L40S inject position 0 minus buggy0 must equal the A100 delta exactly (a gate).
  The node and GPU-string match with R13 are recorded.

## Pre-registered decision rules

D is the set of configs whose class differs between R13's L40S and A100 runs in that cell: 40 in `prefill`
(37 of them R8 subsets) and 60 in `all_steps` (58 R8). Rules are computed on **exposed D**.

- **keep0 (A100) and inject (L40S):** n = |exposed D|; moved = exposed D configs that take the other GPU's
  class; moved_pl = the same count for the placebo, on the same exposed D. N_new is the number of configs outside D whose class
  changes relative to this run's copy. Flips outside D are reported in each direction (under the hypothesis:
  correct → not correct for keep0, the reverse for inject). Comparisons use integers. The first matching
  rule, in this order:
  1. n = 0 → "refuted by construction: no D config is exposed to the position-0 difference".
  2. n < 10 → "descriptive only".
  3. moved ≥ 0.8 n:
     - N_new > 4 → "perturbs broadly";
     - else, placebo invalid → "not distinguishable from fragility (no valid placebo)";
     - else 2(moved − moved_pl) < n → "not distinguishable from fragility";
     - else, if no D config is unexposed or moved ≥ 0.8 |D| → "explains the gap";
     - else → "explains the exposed part (moved/|D| = …)".
  4. moved ≤ 0.2 n → "perturbs, does not explain" if N_new > 4, else "does not explain the gap".
  5. Otherwise (0.2 n < moved < 0.8 n):
     - N_new > 4 → "perturbs broadly";
     - placebo invalid → "not distinguishable from fragility (no valid placebo)";
     - 2 moved_pl ≥ moved (the placebo moves at least half as many) → "not distinguishable from fragility";
     - 2(moved − moved_pl) < n → "partial, below the placebo margin (r = …, r_pl = …)";
     - else → "explains part of the gap (r = moved/n)".
  The placebo itself is scored descriptively and gets no label.
  `test_decide_placebo.py` asserts the branches, including the invalid-placebo and dense-difference cases.
- **interpreted:** a cell's labels count only if every gate holds: the copy gate, capture equals live,
  unexposed-identical for keep0, inject and placebo, and, on L40S, d0 = 0 and inject exactness.
- Both keep0 and inject "explain" → "shown from both sides" (read from the two `run.json` files).
- **donorA100 and a100clean0 (descriptive, no label):** r on all of D. High r means the gap comes from the donor's computation;
  low r means it comes from the receiving run's arithmetic.
- If the copy gate fails, the exploratory D′ verdicts use a placebo scored on D′ too.
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
- Known label inconsistency, left as is: in rule 3 (r ≥ 0.8) the placebo margin is 2(moved − moved_pl) < n,
  while rule 5 uses 2·moved_pl ≥ moved. For 1–3 values of moved_pl per moved, a stronger effect can get the
  weaker label (review round 5, note 1). The raw moved and moved_pl counts are always reported.

## Results

(none yet)
