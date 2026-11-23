# R17 — Where does Llama-3.1-8B decide 9.8 vs 9.11? A corrected activation-patching map

**Status:** DESIGN (pre-registered 2026-11-23, before any compute; amended after design review rounds 1–3)
**Design review:** round 1 FIXES NEEDED — fresh subagent, 2026-11-23 — docs/reviews/11-23_llama8b_localization_map_design.md
**Design review:** round 2 FIXES NEEDED — fresh subagent, 2026-11-23 — docs/reviews/11-23_llama8b_localization_map_design_v2.md
**Design review:** PASS WITH MINOR FIXES (round 3, fixes applied) — fresh subagent, 2026-11-23 — docs/reviews/11-23_llama8b_localization_map_design_v3.md
**Script:** `9_8_research/reviewer_experiments/exp_R17_localization_map/run_experiment.py`. It imports R13's script
by path only for the R8/R1 classifier (`is_correct`, `is_bug`); R13's sha256 must equal `1ce5b0cc…`.
**Job:** `run.sbatch` (`a100_tandon`, 2 h walltime).
**Decision:** Gus, 2026-11-23: "yup let's do this" (stage 1 of a three-stage search for what does repair the error
under correct patching, instead of withdrawing the Even Heads paper).

## Research question

R13 tested one layer (10) and one component (attention heads), the place the buggy procedure pointed to, and found
no repair. It never asked where the 9.8-vs-9.11 decision is actually made. Under correct patching, which
(component, layer, token position) carries the difference between the simple format (answers 9.8) and the Q&A
format (answers 9.11)?

## Simplest (boring) hypothesis, written first

The decision is spread over many prompt positions and layers, and no single site carries it. Then only patches that
replace many positions at once (the whole prompt's residual stream at an early layer) repair, and stage 2 (heads)
has no target.

## Setup

- Model `meta-llama/Meta-Llama-3.1-8B-Instruct`, snapshot 0e9e39f2…, fp16, one A100. Hugging Face forward hooks.
- Donor = `Which is bigger: 9.8 or 9.11?\nAnswer:` (17 tokens; R13's clean prompt). Recipient =
  `Q: Which is bigger: 9.8 or 9.11?\nA:` (19 tokens; R13's buggy prompt).
- **Teacher-forced answer prefix.** Both prompts' greedy answers begin with the tokens `" "`, `"9"`, `"."`
  (ids 220, 24, 13: checked by a gate), and the next token is `"8"` (23) or `"11"` (806). Both sequences get the
  prefix appended (20 and 22 tokens). **Metric:** Δ = logit(`8`) − logit(`11`) at the last position.
  r = (Δ_patched − Δ_buggy) / (Δ_clean − Δ_buggy), where Δ_clean is the donor's own Δ.
- **Alignment.** Donor position i → recipient position i + 2, for i ≥ 1. The aligned tokens are identical except
  donor 1 `Which` vs recipient ` Which` and donor 15 `Answer` vs recipient `A` (gate). Recipient positions 1–2
  (`Q`, `:`) have no counterpart. BOS is never patched.
- **Components**, at every layer L = 0…31:
  - *resid*: the input of decoder layer L (pre-hook);
  - *attn*: the output of `self_attn` (= `o_proj` output, the whole attention sublayer);
  - *mlp*: the output of `mlp`.
- **Position sets:**
  - *prompt*: each single prompt position (donor 1…16), and all 16 together (`prompt_all`);
  - *answer*: each answer-prefix position (`a0` = " ", `a1` = "9", `a2` = "."), and all three (`answer_all`).
  Answer-position patches act on tokens the model generates, so they are decode-time interventions. They are
  reported separately and are never called a prompt repair.
- 3 components × 32 layers × 21 position sets = 2,016 single forward passes.
- **Generation check.** Every prompt cell with r ≥ 0.5 **or Δ_patched > 0** (the argmax flips at
  r_flip = −Δ_buggy / gap, which is reported) is re-run as greedy generation (20 tokens) of the plain recipient
  prompt, with the patch applied on the prefill pass only (the hook asserts the prefill length is 19 and then sees
  only length-1 decode calls, which it leaves alone). The donor values at prompt positions come from the prefixed
  donor capture; under causal attention they equal a prompt-only capture, and the measured difference between the
  two (scaled by the residual norm at that layer and position) is reported. The text is classified with R13's
  classifier; repairs that pass only through its fallback rule (first line mentions 9.8 and not 9.11) are flagged,
  and every repairing text is hand-checked in the report.
- **Controls** (teacher-forced, for every cell that reaches the generation check):
  - *random direction* (the repair veto): the recipient's own value plus a random unit vector (seed 17) scaled to
    the norm of the real patch's change, at each patched position; its r is control_max_r;
  - *misaligned donor* (descriptive, single-position cells only): the donor value from position i − 1 or i + 1
    written at recipient position i + 2. A cell is *position-specific* if every available shifted r < r / 2 (p1 and p16 have one shift). Not a veto:
    neighbouring positions (e.g. donor 15 "Answer" and 16 ":") can carry the same format signal, so a shifted
    donor is not a null.
- **Reverse direction** (necessity, descriptive): for the same cells, the recipient's values are written into the
  donor run at the mirrored positions; reverse_r = (Δ − Δ_clean) / (Δ_buggy − Δ_clean). Stage 1 tests
  sufficiency; reverse_r is reported, not used in the label.
- **Input substitution** (excluded from the label, reported separately): donor positions 1 and 15 hold different
  tokens from the recipient's. Patching there, and patching anything at layer 0, mostly swaps one token's identity
  for another. Excluded: every *resid* cell at layer 0; every *resid* cell that includes position 1 or 15 (this
  includes all *resid* `prompt_all` cells); *attn* and *mlp* cells at layer 0 that include position 1 or 15.
- **Near input** (flagged): *attn* and *mlp* cells at layers ≤ 2 that include position 1 or 15 (this includes
  every *attn*/*mlp* `prompt_all` cell at layers 1–2). They still encode mostly token identity, so they carry no
  label and no stage-2 trigger; their repairs are listed.
- A prompt cell **repairs** if r ≥ 0.5, the generation is classified correct, and control_max_r < 0.2.

## Gates (any failure: exit 2, no sweep is interpreted)

1. **Alignment:** prompt lengths 17/19, total 20/22, mismatches only at donor positions {1, 15}.
2. **Phenomenon:** unpatched greedy generations: donor correct and recipient bug (R13 classifier); the donor's
   first four tokens are [220, 24, 13, 23] and the recipient's [220, 24, 13, 806].
3. **Baseline:** Δ_buggy < 0 < Δ_clean, gap ≥ 1.0 logit; in the teacher-forced runs the argmax at the last
   position is 23 for the donor and 806 for the recipient (top 5 recorded); capturing does not change the
   recipient's logits (bitwise).
4. **Decomposition:** resid[L+1] = resid[L] + attn[L] + mlp[L] within 1% of each non-BOS position's norm, for all
   L, with the final-norm input in place of resid[32]. Also recorded: whether replaying the layer's own fp16 adds,
   (resid + attn) + mlp, reproduces resid[L+1] bitwise.
5. **Identity:** patching the recipient with its own values (each component; layers 0, 10, 31; all positions)
   leaves the logits bitwise unchanged.
6. **Active patch:** with the donor's prompt values patched in (each component; layers 0, 10, 31), a probe hook
   registered after the patch sees exactly the donor values at the patched positions.
7. **Generation identity:** a prefill-only patch of the recipient's own prompt values (each component, layer 10)
   reproduces the unpatched generation token for token.
8. **Positive control (by construction):** writing the donor's final-norm input at the last position into the
   recipient gives r ≥ 0.99. Reported, not a gate: *resid* at layer 31 on all answer positions (expected ≥ 0.8).
9. **Patch has effect:** for each component, at least one sweep cell moves Δ by more than 1e-3.
10. **Hooks removed:** after the sweep, the unpatched recipient's logits equal the first measurement bitwise.

Gates 1–8 run before the sweep; 9–10 after it. An exception at any point after start-up is recorded in `run.json`
(with the cells measured so far) and the job exits 2.

## Pre-registered decision rules (prompt cells, input-substitution cells excluded)

- **Label** (first match):
  1. some single-position, non-near-input prompt cell repairs → "localized: single-position prompt patches
     repair" (listed);
  2. else some non-near-input `prompt_all` cell repairs → "distributed across prompt positions: only all-position
     patches repair (non-near-input cells)";
  3. else → "no prompt-only patch repairs".
- **Stage-2 trigger** (which layers get a head-level search):
  - every layer where a non-near-input *attn* prompt cell repairs;
  - if there is none, the up-to-3 layers with the highest *attn* `prompt_all` r, provided r ≥ 0.2 and, if the
    cell was checked, its random-direction r < 0.2, near-input cells excluded;
  - otherwise none. Stage 2 is then not run on prompt positions.
- **Also reported, no label:**
  - near-input repairs, position-specific repairs, input-substitution cells that repair, and the latest layer at which *resid* `prompt_all` repairs (the layer
    after which the prompt's residual stream alone no longer carries the difference);
  - cells whose generation is correct and Δ > 0 but r < 0.5 ("flips without r");
  - repairs that pass only through the classifier's fallback rule;
  - the first layer at which *resid* at `a2` (".") alone gives r ≥ 0.5, and the layers at which *attn* at an
    answer position gives r ≥ 0.5 (decode-time candidates, reported as such);
  - r, controls and reverse_r for every checked cell; heat maps per component; the number-token positions
    (donor 8 = "8", donor 13 = "11") layer by layer.
- Late-layer cells at the last prompt position (p16, ":") sit next to the answer; they are reported with that
  caveat. A p1 effect (the position right after BOS) needs its controls to be read.
- `decide()` implements these rules; `decide_selftest()` asserts every branch, the exclusions and the reported
  lists at start-up.

## What this can and cannot show

- One prompt pair, deterministic decoding: each cell is one measurement. Stage 1 locates; it does not generalize.
  Before stage 2 rests on a single-position site, the repairing cells are re-run on other prompt pairs. R2/R4
  (legacy, March) found the format-dependent pattern on no other decimal pair, so the candidates are paraphrases
  and other formats that show it; if none exists, the result is stated as specific to this pair.
- r ≥ 0.5 on the logit metric with a correct generation is the repair criterion. A cell with r ≥ 0.5 whose
  generation is not correct is reported as such, with its text.
- `prompt_all` *resid* at layer 0 replaces the recipient's question embeddings by the donor's (including `Answer`
  for `A`). The only remaining difference is the two `Q:` tokens, so the result says whether that prefix alone
  holds the error.

## Compute estimate

Under 15 minutes of A100 time: 2,016 passes of 22 tokens, plus generations and 4 control passes for each checked cell (at most a few hundred).
Walltime 2 h.

## Results

(none yet)
