Reviewer: fresh subagent (Claude Code)
Agent-ID: a1c8bf3f10d8d3b75
Verdict: VERIFIED-WITH-FLAGS
Transcript-SHA256: 69a6a71261deb152b902b892740885c092895fa9e4788457e41380d20b8ce693

## Brief given

You are an independent verifier for a pre-registered mechanistic-interpretability experiment (R19) in ~/github/MATS. Read-only: do NOT edit files, do NOT ssh. Plain python3 suffices (JSON results).

Inputs:
- Pre-registration (design, gates, decision rule, held-out rule, caveats): docs/experiments/11-23_llama8b_last_fraction.md
- Script: 9_8_research/reviewer_experiments/exp_R19_last_fraction/run_experiment.py (expected sha256 a4effdba53010f882d4c00ce243fc8618105acc9a7bece3047f9b3979a03b20e)
- Output: 9_8_research/reviewer_experiments/exp_R19_last_fraction/results/r19_24227404/run.json (Slurm 24227404, A100 ga024, commit c1d4a42)
- Background: R17/R18 (docs/experiments/11-23_llama8b_localization_map.md, 11-23_llama8b_eleven_token.md) found, on Llama-3.1-8B's canonical 9.8-vs-9.11 prompt, that the Q&A-vs-simple flip lives in the residual stream at the "11" token (layers ~9-14); R18 post hoc suggested the site is the question's last fractional token (LFT). R19 tests that on held-out questions.

Tasks:
1. Gates and provenance (script sha256 equals the expected value and the provenance field; R18 sha prefix 1c5e555b; R17 reproduction gate; hooks).
2. Recompute independently from rows: eligibility/skip reasons; in_r18_grid flags; each row's LFT/OFT best band layer, r, primary margin (max over every other prompt position and all band layers 9-14 from position_map_band), same-layer margin; repair flags (r ≥ 0.5, generation class correct, random ctrl mean < 0.2, margin ≥ 0.3); per-stratum rates and Wilson CIs; the label; r18_grid_overlap; by_wording. Also screen counts (format-dependent by order/wording). Spot-check ~30 generation texts (gen and qa/simple texts) by hand for classifier errors.
3. Interpretation, concretely:
   (a) big_first: LFT repairs 33/37 held-out (CI ~0.75-0.96), OFT 0; the LFT is the argmax position in every question. How strong and how general (by wording) is this? Any confound (e.g., LFT is always the token before '?'; mismatched-token positions in the competitor map)?
   (b) small_first: LFT 14/116; the position map suggests the decision is split between the LFT (mean max-band r ~0.32) and the final prompt token ':' (LFT+3, mean ~0.31). Verify and characterize (how many questions have their argmax at each; any question where ':' alone repairs; r values). What does this say about where the flip lives when the multi-digit number comes first?
   (c) The screen: 389/9600 format-dependent (48 big-first, 341 small-first), 105 reverse-dependent; ~18-21% of questions wrong in each format. Scope implications.
4. Flags: computation errors, rule mismatches, overclaims.

Report: top-line verdict on its own line, exactly one of VERIFIED, VERIFIED-WITH-FLAGS, NOT-VERIFIED. Then concise sections: gates; recomputation (recomputed vs logged); label; interpretation; flags.

## Report

VERIFIED-WITH-FLAGS

I recomputed everything from `run.json` with an independent python3 script, and every logged number matches. The flags below are about interpretation and scope. None is a computation error.

## Gates
- **Script sha:** the on-disk script hashes to `a4effdba…3b20e`. That equals the expected value, `provenance.script_sha256`, and the blob at commit `c1d4a42`.
- **R18 sha:** the file hashes to `1c5e555bc8f2…`, which matches the required prefix and the provenance field.
- **Provenance:** host `ga024`, Slurm `24227404`, git `c1d4a42`, A100-80GB.
- **R17 reproduction:** the canonical LFT is at position 13. Its r at layers 4–14 differs from R17's p13 cells by 0.000 at most. Δ_clean is 1.171875 and Δ_buggy is −0.90625, both exact. Gate ok.
- **Hooks:** `hooks_clean` and `hooks_removed` are both true, and `all_gates_ok` is true.
- **Report-only checks:** batch agreement is 1.0 (128 of 128). One sampled question per order failed the unbatched re-check.

## Recomputation (recomputed vs logged, all equal)
- **Screen:** 9,600 questions.
  - Simple: 7,905 correct, 1,695 bug.
  - Q&A: 7,619 correct, 1,978 bug, 3 neither.
  - Format-dependent: 389 (big-first 48, small-first 341). By wording: W1 107, W2 123, W3 159. Reverse-dependent: 105.
- **Sampling:** I re-derived the picks with seed 19. They match the 201 rows exactly, in order.
  - big_first: the pool outside R18's grid has 42 questions, all of them used; 6 are in the grid.
  - small_first: the pool outside the grid has 308 questions, of which 120 were sampled; 33 are in the grid.
- **Skips (9 total):**
  - big_first: 4 on baseline deltas, 1 on the unbatched re-check.
  - small_first: 3 on baseline deltas, 1 on the unbatched re-check.
  - I read all 9 texts and deltas, and each skip reason is right.
- **`in_r18_grid` flags:** all correct.
- **Per-row checks:** zero discrepancies across all rows, both sites.
  - LFT/OFT best band layer, primary margin (over every other position × layers 9–14) and same-layer margin all match.
  - Site r equals the position-map entry at the same position and layer.
  - Repair flags match the rule (r ≥ 0.5, generation correct, control mean < 0.2, margin ≥ 0.3).
  - The LFT always sits 5 tokens after the OFT. The map has 16 positions for W1/W2 and 17 for W3, as pre-registered.
- **Strata outside the grid:**
  - big_first: n = 37, LFT 33 (Wilson 0.753–0.957), OFT 0 (0–0.094).
  - small_first: n = 116, LFT 14 (0.073–0.192), OFT 0 (0–0.032).
- **R18-grid overlap:** big_first 6/6 LFT repairs, small_first 10/33 (0.17–0.47), OFT 0 in both.
- **`by_wording`:**
  - big_first: W1 4/4, W2 16/19, W3 13/14.
  - small_first: W1 4/26, W2 9/37, W3 1/53.
- **Classifier:** I checked 494 texts (simple, Q&A and site generations) against an independent "first number generated" rule and found 0 disagreements. Every "X is bigger than Y" pair is the right pair. I also read 32 random texts by hand, and all were classified correctly.

## Label
"holds for big_first only" is the correct first match. big_first holds (0.89 ≥ 0.5, and LFT − OFT = 0.89). small_first has n ≥ 10 and does not hold. The label string does not say how badly small_first failed: its LFT rate is 0.12, below even the 0.2 "not supported" threshold, with an upper CI of 0.19. The write-up should state that.

## Interpretation

**(a) big_first.** The effect is strong and clean.
- Repairs: 33/37, and 35/37 reach r ≥ 0.5. The four misses are:
  - r = 0.49 and r = 0.42 (near misses);
  - r = 1.62 with control mean 0.2046 (fails the control threshold by 0.005);
  - r = 0.76 where the generation is still the bug.
- Mean best-band r is 1.04 (median 0.98), with best layer 9 in 29 of 37. Mean r across layers 6–14 is about 0.8–1.1. The OFT gets about 0 from layer 6 up and never reaches 0.5. Random controls average about 0.003. Mean reverse r is 2.4.
- The LFT is the argmax position in 37/37 held-out questions and 6/6 grid ones.
- All 33 repairs also pass the same-layer margin, and the smallest primary margin among repairs is 0.35.
- Every other position scores ≤ 0.08 on average, including the two mismatched-token positions ("Which", "Answer"). At the final ":", r is negative (about −0.1 at its best layer, −0.2 to −0.38 per layer).
- It is consistent across wordings, but W1 has only 4 held-out questions. The sample is the entire held-out pool of 37 eligible questions (42 before skips), not a random draw.

There are two confounds:
1. In big_first, the LFT is both the multi-digit token (R18's "11" reading) and the token before "?". big_first alone cannot tell apart "last fractional token", "token before ?" and "the multi-digit token".
2. small_first rules out the multi-digit token. When it comes first (as the OFT), it carries about 0.04. So the site is positional, near the end of the question, not "the 11-like token". LFT versus "token before ?" stays unresolved, as the pre-registration says.

The mismatched-token competitors do not matter in big_first.

**(b) small_first.** The decision is split, mostly between questions rather than within them.
- Argmax position: 61 questions at the LFT and 55 at the final ":" (LFT+3). No other position is ever the argmax. In the grid: 27 LFT, 5 ":", 1 "Answer".
- Mean best-band r: LFT 0.319 and ":" 0.312, which confirms your figures (medians 0.34 and 0.29). The mismatched "Answer" position (+2) averages 0.17.
- Per question, LFT r and ":" r are anti-correlated (r = −0.78). No question has both at ≥ 0.5: the LFT reaches 0.5 in 33 questions and ":" in 24. The two summed average about 0.63, or about 0.80 with +2 added. The rest probably sits at the teacher-forced " X." answer positions, which the map does not include.
- Layer profiles differ. ":" is about 0.27 at layers 9–11 and fades to about 0.02 by layer 13. The LFT stays flat at about 0.25–0.29, and its best layer is most often 13.
- ":" was never tested with generation or random controls, because only the LFT and OFT get those tests. On the map alone, 16 questions have ":" at r ≥ 0.5 with a ≥ 0.3 margin over every other position. Whether ":" alone repairs them is untested.
- LFT outcomes: 83 have r < 0.5; 19 have r ≥ 0.5 but the generation is still the bug (2 of these also fail the margin); 14 repair. The margin rule and the conservative competitor set are not what sinks small_first. Weak r is.
- By wording, W3 is worst (1/53 repairs). Its argmax splits 27 LFT versus 26 ":", against W2's 23 versus 14.
- **Reading:** when the multi-digit number comes first, the flip has no single site in the prompt. It is spread across the question's end (the LFT) and the answer-cue colon, and which one carries it varies by question. Part of it lies outside the mapped prompt.
- The grid rate (0.30) against the held-out rate (0.12) suggests R18's post hoc small_first observation was partly selection.

**(c) Screen and scope.**
- Simple is wrong on 17.7% of questions, Q&A on 20.6%. 1,589 questions (16.6%) are wrong in both formats.
- Only 4.1% of questions are format-dependent.
- In big_first, Q&A does not hurt on net. 90 questions are reverse-dependent (Q&A fixes them) against 48 format-dependent, and 718 are wrong in both formats. In W3 the split is 45 against 17.
- In small_first, Q&A does hurt (25.3% wrong against 18.5%, 341 versus 15 reverse-dependent).
- So the clean big_first result covers about 1% of big_first questions (48/4,800), a minority direction of the format effect. The canonical 9.8/9.11 prompt sits in that minority. Neither order's analysis says anything about the dominant error that appears in both formats.

## Flags
1. The label text hides that small_first fails outright (LFT 0.12, CI upper bound 0.19 < 0.2). Report it as "small_first not supported".
2. Do not present big_first as confirming the "LFT" reading over the R18 "multi-digit/11-token" reading, or over "token before ?". It is the small_first OFT null that removes the multi-digit-token reading. LFT versus pre-"?" remains confounded.
3. Scope: the big_first finding covers 48/4,800 format-dependent questions against 90 reverse-dependent ones. Avoid generalising it to "where Q&A causes the 9.11 bug".
4. "Split between LFT and ':'" holds for the averages but is mostly either/or per question (corr −0.78, zero questions with both ≥ 0.5). The ":" site was never tested for repair (no generation or controls). About 0.2–0.4 of the r is unlocated, probably at the answer-prefix tokens.
5. big_first W1 has n = 4. The "general across wordings" claim rests on W2 (19) and W3 (14).
6. Mean r ≈ 1.04 and reverse r ≈ 2.4 overshoot. This is fine for "carries the flip", but r > 1 means the patch moves the logit difference beyond the clean run's value, so "r = fraction of the gap restored" is not the right gloss there.

There are no computation errors and no mismatches between the code and the pre-registered rule.

My scripts are in `/private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/` (`v.py` to `v4.py`).
