# R19 — Does the format flip live at the question's last fractional token, in both number orders?

**Status:** DESIGN (pre-registered 2026-11-23, before any compute; amended after design review rounds 1–2)
**Design review:** round 1 FIXES NEEDED — fresh subagent, 2026-11-23 — docs/reviews/11-23_llama8b_last_fraction_design.md
**Design review:** PASS WITH MINOR FIXES (round 2, fixes applied) — fresh subagent, 2026-11-23 — docs/reviews/11-23_llama8b_last_fraction_design_v2.md
**Script:** `9_8_research/reviewer_experiments/exp_R19_last_fraction/run_experiment.py`. It imports R18's script by
path (sha256 must start `1c5e555b`) for the pair classifier, unbatched generation and resid capture, and R17's
`Model` through R18.
**Job:** `run.sbatch` (`a100_tandon`, 3 h walltime). Expected script sha256: `a4effdba53010f882d4c00ce243fc8618105acc9a7bece3047f9b3979a03b20e`.
**Decision:** Gus, 2026-11-23: "yes. keep designing and running until we find smth."

## Background and hypothesis

R17 and R18 found, on the canonical pair, that the residual stream at the "11" token carries the Q&A-vs-simple flip.
It is a specific direction, clean at layers 9–14, and it is read out by layer-14 attention. In R18's grid (39
format-dependent questions), the "11"-analogue site repaired 6 of 6 big-first questions. In the 33 small-first ones
the strongest site was the *last* fractional token before "?", i.e. the single digit. That observation was post
hoc.

**H (pre-registered here):** in format-dependent questions, the flip is carried by the question's last fractional
token (LFT), whichever number it belongs to. The other fractional token (OFT) carries much less.

## Simplest (boring) hypotheses, written first

- Any late-position patch helps, at some layer. This is tested by the margin rule: every other prompt position at
  every band layer is a competitor, plus random directions. Note: in every prompt the LFT is also the token right
  before "?\n", so R19 cannot separate "last fractional token" from "the token before the question mark".
- The R18 big-first result (n = 6) was luck, and in a larger sample the site is mixed or order-dependent.

## Setup

- Model, snapshot, fp16 and hooks are R17's; the metric is R18's. For big = X.Y and small = X.Z,
  Δ = logit(Y) − logit(Z) after the prefix " X.", and r = (Δ_patched − Δ_buggy) / (Δ_clean − Δ_buggy).
- **Questions:**
  - wordings W1 "Which is bigger:", W2 "Which is larger:", W3 "Which number is bigger:";
  - X ∈ 1..20, Y ∈ 2..9, Z ∈ 10..19 (so X.Y > X.Z always);
  - both orders (big_first = "X.Y or X.Z", small_first = "X.Z or X.Y");
  - 9,600 questions.
  - Simple = "{W} {n1} or {n2}?\nAnswer:"; Q&A = "Q: {W} {n1} or {n2}?\nA:" (natural tokenization).
- **Screen:** greedy, 20 tokens, left-padded batches of 64, classified with R18's classifier. *Format-dependent* =
  simple correct and Q&A bug. Batching can differ numerically from single runs, so a seeded sample of 64 questions
  is re-run unbatched (gate), and every analysed question is re-checked unbatched (eligibility).
- **Sample:** per order, a seeded random sample (seed 19) of up to 120 format-dependent questions outside R18's grid
  (the decision sample).
- **Held-out rule:** R18's grid (W1 with Y ∈ 6..9 and Z ∈ 10..13) generated the hypothesis. Up to 40 of its
  format-dependent questions per order are analysed the same way and reported separately (with Wilson intervals); they
  never enter the decision.
- **Expected n:** R18's rates suggest about 10% of small-first and about 2% of big-first questions are
  format-dependent. That gives ≈ 480 small-first (sample capped at 120) and roughly 20–90 big-first, before
  exclusions and eligibility. Big-first may therefore fall below n = 10 (then it is descriptive).
- **Eligibility** (unbatched), all required:
  - simple correct and Q&A bug;
  - both answers start with " X." followed by Y (simple) or Z (Q&A);
  - single-token fractional parts;
  - alignment offset 2, with mismatches only at the first word ("Which" / " Which") and the ending
    ("Answer" / "A");
  - exactly two "." in the prompt, the LFT directly before "?\n";
  - Δ_buggy < 0 < Δ_clean, gap ≥ 0.5.
- **Per eligible question,** *resid* patches (simple → Q&A, offset 2) at the LFT and at the OFT, layers 4–14. The
  primary band is layers 9–14 (R18: the direction is specific there); a site's best layer is chosen within it.
- **Position map (always computed):** every prompt position except BOS, at every band layer (16 positions for W1/W2 and 17 for W3,
  i.e. 15/16 competitors besides the site, × 6 layers). This includes the two mismatched-token positions ("Which", "Answer"), which makes the
  test conservative; the teacher-forced " X." answer tokens are not prompt positions and are not included.
- **Margin (primary)** = the site's r at its best band layer − the largest r of any other position at any band
  layer. The same-layer margin and the best competitor (position, layer, token, r) are also reported.
- **At each site's best band layer, if r ≥ 0.5:**
  - prefill-only generation, classified;
  - 5 random same-norm directions (mean and SD);
  - reverse direction (Q&A value into the simple run), reported.
- A site **repairs** a question if r ≥ 0.5, the generation is correct, the random mean < 0.2 and the margin ≥ 0.3.

## Gates (any failure: exit 2)

1. **R17 reproduction:** for the canonical question (W1, 9.8 vs 9.11, big-first), the LFT sits at position 13, its r
   at layers 4–14 is within 0.03 of R17's p13 values, and Δ_clean and Δ_buggy are within 0.04 of R17's.
2. **Hooks clean** (before) and **hooks removed** (after): the canonical recipient's logits are bitwise unchanged.
Report-only (not a gate, since every analysed question is re-checked unbatched): batched vs unbatched agreement on
64 seeded questions × 2 formats; and, per order, how many sampled candidates fail the unbatched re-check.

Any exception exits 2; after `run.json` exists, it is recorded there (with the rows so far). A wrong R18
sha256, or a script whose sha256 differs from the committed one (checked in `run.sbatch`), also exits 2.

## Pre-registered decision rule

Per order stratum (eligible questions outside R18's grid): n, LFT and OFT repair rates with Wilson 95% intervals
(also for descriptive strata).
A stratum *holds* if n ≥ 10, LFT rate ≥ 0.5 and LFT − OFT ≥ 0.3. (Each repair already needs a 0.3 margin over every
other position, so LFT − OFT ≥ 0.3 is nearly implied.) First match:

1. both strata hold → "last-fractional-token site holds in both orders";
2. at least one holds → "holds for <order> only", plus "; <order> descriptive (n < 10)" for any small stratum;
3. no stratum has n ≥ 10 → "descriptive only (too few eligible questions)";
4. every stratum with n ≥ 10 has LFT rate < 0.2 → "not supported" (plus the descriptive note);
5. otherwise → "partial" (plus the descriptive note).

Reported, no label:
- per-wording counts;
- the screen's class counts, the format-dependent counts by order and wording, and the reverse-dependent count;
- per-question r curves, position maps, margins (primary and same-layer), best competitors, controls and reverse r;
- the R18-grid overlap (n, LFT and OFT repairs per order);
- skipped questions with reasons.

`decide()` has a start-up selftest covering every label.

## What this can and cannot show

- It tests where the flip is carried on questions that have one. It does not explain the much more common error that
  appears in both formats (R18: 183/640).
- One model (Llama-3.1-8B-Instruct), greedy decoding, our three wordings.
- If big-first format-dependent questions stay rare, the big-first stratum may be descriptive only.
- The LFT is always the token before "?\n" here; a claim about "last fractional token" vs "token before the
  question mark" needs a different question form.

## Compute estimate

- Screen: 19,200 generations in batches of 64, about 10 min.
- Sites: up to 240 questions × (2 unbatched generations, ~95 position-map passes, 2 × 11 site passes, 2 × 6 control
  and reverse passes, 2 generations), about 25–35 min.
- Walltime 3 h.

## Results

(none yet)
