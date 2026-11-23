Reviewer: fresh subagent (Claude Code)
Agent-ID: a1a54209459190d42
Verdict: VERIFIED-WITH-FLAGS
Transcript-SHA256: 42906ceb289f4c3d2f64aa13222f55557ae255dcf7333c11e873417838cbf93a

## Brief given

You are an independent verifier for a pre-registered mechanistic-interpretability experiment (R17) in ~/github/MATS. Read-only: do NOT edit any files, do NOT ssh. Plain python3 is available (no torch needed: all results are JSON).

Inputs:
- Pre-registration (with decision rules, gates, exclusions): docs/experiments/11-23_llama8b_localization_map.md
- Script: 9_8_research/reviewer_experiments/exp_R17_localization_map/run_experiment.py
- Output: 9_8_research/reviewer_experiments/exp_R17_localization_map/results/r17_24213609/run.json (all 2,016 cells with delta, r, flags; generation texts, controls and reverse_r for checked cells; gates; decision), Slurm job 24213609, A100 ga007, commit 623c45f.
- Token ids: donor positions 0..19 are [BOS, Which, is, bigger, :, ' ', 9, ., 8, or, ' ', 9, ., 11, ?\n, Answer, :, ' ', 9, .]; recipient = donor shifted by +2 with 'Q', ':' at 1-2, ' Which' and 'A' at the two mismatched positions.

Tasks:
1. Check every gate in run.json against the prereg's thresholds; confirm provenance (script sha256 must equal the sha256 of the local script; R13 sha256 1ce5b0cc…).
2. Recompute r = (delta − d_buggy)/(d_clean − d_buggy) for all cells from the stored deltas; recompute input_substitution and near_input flags from the prereg definitions; recompute which cells needed a generation check (r ≥ 0.5 or delta > 0) and confirm all of them have one; recompute `repair` (r ≥ 0.5, gen_class correct, random-direction r < 0.2); hand-check that each repairing gen_text really says 9.8 is bigger (not via classifier fallback).
3. Re-derive the pre-registered label, stage-2 layers and every reported list in run.json["decision"] yourself from the cells, independent of decide(), and compare.
4. Scrutinize the interpretation of the headline: residual-stream patches at donor position 13 (the "11" token of 9.11) repair at layers 4–14. Report the r curve by layer for resid/attn/mlp at p13; the random-direction controls; the shift controls (they are very large, r≈5–6: explain what that means — the shifted patch writes a different token's residual at the "11" position); reverse_r (necessity). Assess carefully whether "the format information is carried by the '11' token's residual at layers ~4–14 and read out by ~layer 15" is supported, and what alternative readings exist (e.g., any perturbation that weakens the '11' representation pushes toward 8; check whether the random-direction controls, which have the same norm, rule that out, and whether they are comparable in size to the real patch). Also look at: resid L0 p1 (excluded input substitution, ' Which' vs 'Which') repairing; mlp L3 prompt_all repairing; the answer-position results (resid a2 first layer ≥0.5, attn answer layers); and stage-2 layers [8, 11, 12] chosen by the weak fallback rule.
5. List any problems: computation errors, rule mismatches, overclaims.

Report: a top-line verdict on its own line, exactly one of VERIFIED, VERIFIED-WITH-FLAGS, or NOT-VERIFIED. Then concise sections: gates; recomputation; label derivation; interpretation assessment; flags. Be concrete with numbers.

## Report

VERIFIED-WITH-FLAGS

The run is clean. Every gate passes, provenance matches, and my independent recomputation agrees with every stored value and every entry in `run.json["decision"]`. The flags are about interpretation. The headline "format information is carried by the '11' token's residual at L4–14 and read out by ~L15" is broadly supported as a location claim, but "format" overstates it, and the controls do not rule out a salience reading.

## Gates and provenance
- **Provenance:**
  - Local `run_experiment.py` sha256 is `6bf85484…a8c862`. That equals `provenance.script_sha256` and the `git show 623c45f:` version.
  - Local R13 sha256 is `1ce5b0cc…348d`, which matches.
  - Slurm job 24213609 ran on ga007 (A100-80GB), model snapshot `0e9e39f2…`.
- **Alignment:** prompt lengths are 17/19, totals 20/22, and the mismatches are at [1, 15]. Passes.
- **Phenomenon:** the donor generates " 9.8 is bigger than 9.11…" (correct). The recipient generates " 9.11 is bigger than 9.8…" (bug). The first four tokens are [220,24,13,23] and [220,24,13,806]. Passes.
- **Baseline:** d_clean = 1.1719, d_buggy = −0.9063, gap = 2.078 (≥ 1). The argmax is 23 for the donor and 806 for the recipient. Capturing leaves the logits bitwise equal. r_flip = 0.436. Passes.
- **Decomposition:** max relative error 4.96e-4 (< 1e-2), and the bitwise replay is equal. Passes.
- **Identity, active patch and generation identity:** all 9/9/3 cells are true.
- **Positive control:** r = 1.0 (≥ 0.99). The soft control (resid L31 answer_all) also gives r = 1.0.
- **Patch has effect and hooks removed:** both true. `all_gates_ok` is true.
- **Descriptive:** the difference between the donor prompt-only capture and the prefixed capture is 0.0049 of the residual norm.

## Recomputation (all 2,016 cells; 3 components × 32 layers × 21 sets, no duplicates)
- **r:** `(delta − d_b)/gap` matches the stored r exactly (max error 0).
- **Exclusion flags:** recomputing `input_substitution` and `near_input` from the pre-registration text gives 0 mismatches.
- **Generation checks:** the set needing one (prompt cells with r ≥ 0.5 or Δ > 0) is exactly the 32 checked cells. None is missing and none is extra.
- **Classification:** I reimplemented the R13 classifier inline. `gen_class`, `fallback`, `control_max_r == random_dir`, `position_specific` and `repair` all match (0 errors).
- **Hand check of texts:** all 32 checked generations are the identical string " 9.8 is bigger than 9.11.\nExplanation: 9.8 is bigger". It matches the primary pattern "9.8 is bigger", so none passes only through the fallback rule. Every one is correct.
- **The generation check never vetoed anything.** Every checked cell, including ones with Δ = +0.03, generated the correct answer. Since the 0.5 threshold sits above r_flip = 0.436, the check added no discrimination.

## Label derivation (independent of `decide()`)
- **Repairing counted cells:** resid p13 at L4, 6, 7, 8, 10, 11, 12, 13, 14, plus mlp L3 prompt_all.
  - Resid p13 at L5 and L9 has r = 0.684 and 0.707 but fails the random-direction veto (0.639 and 0.203).
  - Resid p13 at L3 has r = 0.451, below 0.5.
- **Label:** "localized: single-position prompt patches repair". This matches.
- **Every reported list matches**, in the same order:
  - near_input_repairs: [].
  - position_specific_repairs: [].
  - input_substitution_repairs: 12 cells.
  - resid prompt_all latest repairing layer: 14.
  - flips_without_r: 6 cells.
  - fallback repairs: [].
  - Answer-position fields: the first resid a2 layer with r ≥ 0.5 is 25, and attn answer layers with r ≥ 0.5 are [14].
- **Stage-2 layers [8, 11, 12]:** no attn cell repairs, so the fallback rule applies. The candidates are attn prompt_all cells with r ≥ 0.2: L8 0.263, L12 0.233, L11 0.226, L3 0.211, L5 0.203.

## Interpretation assessment

**R curves at p13, L0→L31:**
- resid: 0.00, .06, .20, .45, **.57 .68 .55 .71 .58 .71 .74 .77 .84 .85 .80** (L4–L14), then drops to .26 at L15 and decays to ~0 by L31.
- attn: never above 0.29. It is noisy and changes sign (L5 −0.35, L7 −0.11; L3 .29, L11 .24, L12 .23).
- mlp: peaks at L3 0.45 and L8 0.50; L9–L13 sit around 0.25–0.34.
- Reading: the p13 difference is built up gradually over L2–L13 by many sublayers. No single attn or mlp output carries it.

**Specificity over position is genuine.** At L4–14, every other resid single-position patch stays at |r| < 0.35. The only exceptions are p11 at L5 (−0.54) and p16 at L6–7 (0.46 and 0.44). p8, the "8" token, stays at |r| ≤ 0.2.

**Readout around L14–15 is consistent across three signals:**
- resid p13 falls from 0.80 at L14 to 0.26 at L15.
- attn at answer a0 (" ") is 0.62 at L14, and answer_all is 0.65; this is the only attn answer layer with r ≥ 0.5.
- resid answer_all jumps from 0.16 at L14 to 0.89 at L15.

More precisely, the information is moved by layer-14 attention into the answer-prefix positions, mostly a0. Resid a2 alone reaches 0.5 only at L25.

**Reverse_r at resid p13 L4–14 is 0.90–1.53.** Writing the recipient's p13 residual into the donor run fully re-creates the error and often overshoots. So the site is both sufficient and necessary on this one pair. Note there is no random-direction control in the reverse direction.

**The shift controls are r ≈ 5.5–6.6 at every resid p13 layer (Δ ≈ +10.6–12.9 logits).**
- Shift −1 writes the donor's "." residual at the "11" slot; shift +1 writes the "?\n" residual there.
- This erases the identity of the "11" token at the place the model would copy "11" from. logit("11") collapses and Δ shoots far past d_clean.
- So shifted donors are not nulls here. They show the metric is extremely sensitive to the integrity of the "11" representation at p13.
- It also makes the pre-registered `position_specific` flag uninformative (false for every p13 cell, since the shifts are much larger than r/2). The empty `position_specific_repairs` list must not be read as "not position-specific".

**"Any weakening of '11' pushes toward 8": are the random controls enough?**
- They argue against it only weakly.
- At resid p13 L4–14, the same-norm random-direction r has mean −0.05 and SD 0.34, against a real-patch mean of about 0.71, so the real patch is about 2 SD above the random draws. One random draw (L5: 0.639) nearly equals the real patch (0.684). Across all 32 checked cells, random r has mean −0.15 and SD 0.49.
- There is one seed per cell, and a random direction in 4096 dimensions is an off-manifold, weak null.
- The patch norm relative to the residual norm is not stored, so I cannot tell whether these perturbations are small or large.
- The data cannot separate "the donor's '11' carries format information" from "the donor's '11' is a less copy-salient 11", that is, a change in the magnitude or salience of the copy source.
- Useful follow-ups:
  - Patch "11"-token residuals taken from other unrelated contexts.
  - Interpolate along the donor−recipient difference and compare with several random and in-manifold directions (multiple seeds).
  - Log the patch-to-residual norm ratio.
  - Measure logit(8) and logit(11) separately to see which one moves.

**"Format" is an overclaim; describe what p13 can actually see.**
- Under causal attention, donor p13 sees only "[BOS] Which is bigger: 9.8 or 9." It cannot see "Answer:" versus "A:".
- What the p13 patch transfers is therefore only the effect of the "Q:" prefix, the ' Which'/'Which' tokenization, and the +2 absolute-position shift. It is not the answer cue.
- The resid L0 p1 result sharpens this. Swapping just the ' Which' embedding for 'Which' gives r = 0.865, with a correct generation, random control −1.10, shift +1 −0.99, and reverse_r 0.41. It is excluded as input substitution, correctly per the pre-registration.
  - So "Q: Which… A:" with the 'Which' token in place of ' Which' is answered correctly. Much of the bug may follow the leading-space "Which" token rather than Q&A format as such.
  - Resid p1 at L1 gives r = 0.00, so this token effect is fully passed to other positions within layer 0.
  - Resid prompt_all at L0 gives 1.08: the "Q:" prefix alone does not hold the error, which answers the pre-registered question.
  - Resid p15 (Answer vs A) at L0 is −0.18, so that swap does nothing.
- Recommended wording: "the difference induced upstream (the Q: prefix / ' Which' token) is carried by the '11' token's residual at L4–14 and read out by layer-14 attention into the answer prefix." Avoid "format information".

**mlp L3 prompt_all repair is an isolated, fragile cell.**
- It has r = 0.647, random −0.617 and reverse 0.58.
- Neighbouring mlp prompt_all values swing in sign: L0 −0.83, L1 +0.68 (vetoed, random 0.39), L2 .28, L4 −0.32, L5 −0.58, L7 .41, L8 .49.
- mlp p13 at L3 alone is 0.45, so most of the effect is p13.
- It should not be read as a distributed MLP site.

**Resid prompt_all is non-monotone.** By layer it goes 1.08, 0.13, 0.19, −0.39, then 0.95 at L4. The dip at L1–3 is unexplained.
- One hypothesis I could not test (no norms stored): the donor's p1 sits right after BOS and may carry massive or attention-sink activations. Writing it at recipient position 3 is then not a like-for-like substitution. This also bears on reading the p1 effect.

**Stage-2 layers [8, 11, 12] are weakly motivated.**
- They follow the rule correctly, but the attn prompt_all r values are only 0.26, 0.23 and 0.23, inside a noisy band: L3 .21, L5 .20, L7 −0.23, L1 −0.20.
- None of the three was generation-checked or given the random-direction control, because r < 0.5 and Δ < 0. The rule's "if checked" clause means the veto never applied to them.
- At L11 and L12, attn p13 alone gives about the same r (0.24 and 0.23), so the signal is mainly p13 again.
- Given the resid/attn/mlp profile at p13, a head search at those layers is unlikely to find a single carrier. Tracing which heads write into p13 at L2–13, and which head at L14 reads p13 into a0, is better targeted.

## Flags
1. **No computation errors and no rule mismatches.** `decide()` and every stored flag reproduce exactly.
2. **Overclaim, "format information":** p13 cannot see "Answer/A:", and resid L0 p1 (' Which' → 'Which') alone repairs at r = 0.865. The carried variable is the upstream prefix/tokenization difference, possibly mostly ' Which'.
3. **Salience alternative not excluded:**
   - The shift controls (r ≈ 5.5–6.6) show the metric is dominated by whether "11" is intact at p13.
   - The random-direction null is one seed, off-manifold, and has SD of about 0.35–0.49 in r. One draw (L5) matched the real patch.
   - The real patch is only about 2 SD above it, and no norm ratio is stored.
4. **`position_specific` is uninformative** because the shifted donors change token identity. Specificity should instead be argued from the near-zero r of the neighbouring aligned positions (p12, p14).
5. **The repair band has two veto gaps (L5, L9)** caused only by the random control, so "L4–14" is really L4, 6–8, 10–14.
6. **mlp L3 prompt_all is an isolated cell** in a sign-flipping series. It should not be presented as a second site.
7. **Stage-2 [8, 11, 12] rests on r ≈ 0.23–0.26** that never received the control veto. It is weak, and p13-dominated where checked.
8. **Generalization:** this is one prompt pair with deterministic single measurements. The pre-registration already requires re-running on other pairs before stage 2 relies on p13.
9. **Minor:** Δ is quantized at fp16 (steps of 1/64), so r is quantized at about 0.0075. Two cells, resid L3 p13 and mlp L3 p13 (r = 0.451, Δ = 0.031), sit right at the flip boundary.

Files read: `/Users/gus/github/MATS/docs/experiments/11-23_llama8b_localization_map.md`, `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R17_localization_map/run_experiment.py`, `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R17_localization_map/results/r17_24213609/run.json`. Verifier script: `/private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/v.py`. No repo files were modified.
