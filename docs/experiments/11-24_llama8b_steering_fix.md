# R20 — Can a steering vector at the end of the question fix the decimal error on held-out questions?

**Status:** RUNNING (pre-registered 2026-11-24, before any compute; amended after design review rounds 1–2)
**Design review:** PASS WITH MINOR FIXES (round 1, fixes applied) — fresh subagent, 2026-11-24 — docs/reviews/11-24_llama8b_steering_fix_design.md
**Design review:** PASS WITH MINOR FIXES (round 2, fixes applied) — fresh subagent, 2026-11-24 — docs/reviews/11-24_llama8b_steering_fix_design_v2.md
**Script:** `9_8_research/reviewer_experiments/exp_R20_steering_fix/run_experiment.py`. It imports R19's script by path
(full sha256 `a4effdba…3b20e` checked) for the question set and wordings, R18 (through R19) for the pair classifier,
and R17's `Model` hooks (through R18).
**Job:** `run.sbatch` (`a100_tandon`, 4 h walltime). **Slurm job:** 24373179, commit 995daa2. Log `/scratch/gs157/MATS/logs/24373179_r20_steer.log`; results `/scratch/gs157/MATS/results/r20_24373179/`.
Expected script sha256: `effa9b292313395ab19519a0f810a021d76e53f656d6dd1e72b7ed7966fcf5f1`.
**Decision:** Gus, 2026-11-24: "try r20" (after R19; standing instruction "keep designing and running until we find smth").

## Background and hypothesis

R19 (pre-registered, held out) found that in big-first format-dependent questions, the Q&A flip is carried by the
residual stream at the last fractional token (LFT, the token before "?") at layers 9–14. Patching that one vector
from the simple-format run repairs 33/37. But patching needs a per-question donor run with the right answer's
format. That is a diagnosis, not a fix.

R20 asks whether a **fixed vector**, learned once from training questions and added at one prompt position and layer,
fixes errors on questions with unseen integer parts (and one unseen wording) without breaking correct answers. Prior
work on 9.11 > 9.8 (Transluce neuron suppression; KSHSeek; prompting) did not test a held-out targeted activation fix
with a break rate.

**H1:** a direction learned at the LFT (or final prompt token) gives a significant net gain (fixes − breaks) in
Q&A-format generation on held-out questions, beats random directions of the same norm, and breaks ≤ 2% of correct
answers.

## Simplest (boring) hypotheses, written first

- **Format swap, not a fix.** The simple − Q&A direction makes the Q&A prompt behave like the simple prompt, including
  the simple prompt's own errors. Then it fixes format-dependent questions and *breaks* reverse-dependent ones
  (simple bug, Q&A correct), and the steered answer agrees with the simple-format answer. Reported: the
  steered answer's agreement with the unsteered simple-format class, and breaks among reverse-dependent questions.
- **Any perturbation helps.** Large pushes knock the model into a different answer at random; tested with 3 random
  directions of the same norm at the same site, layer and α. Isotropic random vectors are a soft null, so fd and cb
  also get a **matched label null**: the same mean-difference recipe with labels destroyed (cb: correct/bug labels
  shuffled within order on train; fd: a random train subset of the same size as the format-dependent set), rescaled to
  ‖v‖ (label-null seed 21). The fd label null is ≈ fmt plus sampling noise, so for fd it tests whether selecting
  format-dependent questions adds anything beyond a generic format push; cos(v, null) and cos(fd, fmt) are saved.
  The sign-flipped vector −v is reported.
- **The trivial fix is the prompt.** Dropping "Q:" is free. From R19's classes, a pure Q&A → simple swap on test would
  score F = 179, B = 35 (net 144, break rate 1.1%). The headline label carries the qualifier "(<= prompt swap)" when
  the steered net does not exceed the prompt-swap net.
- **The dominant error is untouched.** Most errors (R19: 16.6% of questions) appear in both formats. A format direction
  should not touch them; the correctness direction (cb) might.

## Setup

- Model, snapshot, fp16 and hooks: R17's. Questions: R19's 9,600. Wordings: W1 "Which is bigger:", W2 "Which is
  larger:", W3 "Which number is bigger:". X ∈ 1..20, Y ∈ 2..9, Z ∈ 10..19 (X.Y > X.Z), both orders. Prompts: simple =
  `{W} {n1} or {n2}?\nAnswer:`, Q&A = `Q: {W} {n1} or {n2}?\nA:`.
- **Split by X** (seed 20, X = 9 forced into test so the canonical prompt is held out):
  - train X = {1, 3, 4, 5, 7, 10, 11, 12}, W1 and W2 (2,560 questions);
  - dev X = {2, 13, 15, 16}, W1 and W2 (1,280);
  - test X = {6, 8, 9, 14, 17, 18, 19, 20}, W1, W2 and W3 (3,840). W3 is never seen in train or dev.
  - From R19's screen (classes only; used to size the split, not to tune anything): train has 74 format-dependent and
    559 both-bug questions, dev 45 and 99, test 177 and 563.
- **Dev sees almost none of the R19 site.** From R19's classes, all 45 dev format-dependent questions are small-first
  (where R19 found the LFT does *not* carry the flip), and dev has 6 reverse-dependent questions. Train has 14
  big-first fd, test 27. So selection is driven by small-first and both-bug questions; the big-first fd result on test
  is reported separately (order × category).
- **Baseline:** batched greedy generation (20 tokens, left-padded batches of 64, the same order and batches as R19)
  for all 9,600 questions in both formats, classified with R18's pair classifier. This full pass gives the categories and gate 4. The fix/break reference is a
  second unsteered pass over the dev and test subsets in exactly the batches the steered runs use (so fp16
  batch-composition noise cannot count as a fix); disagreements between the two passes are reported. Categories: format-dependent (simple
  correct, Q&A bug), both-bug, reverse-dependent (simple bug, Q&A correct), both-correct, other (any "neither").
- **Directions** (from train only; per layer L ∈ {4, 6, 8, 9, …, 14, 16, 18, 20} and site ∈ {LFT, final prompt token
  ":"}; resid = input of decoder layer L):
  - **fmt**: mean over all train questions of resid_simple − resid_Q&A at the site (label-free).
  - **fd**: the same mean over train format-dependent questions only.
  - **cb**: in Q&A format, mean resid of questions answered correctly minus mean of those answered with the bug,
    computed within each order and averaged over the two orders.
- **Steering:** add α·v to the resid at the site in the prompt, on the prefill pass only (later tokens see it through
  the KV cache). α ∈ {0.5, 1, 2, 4, 8}. Same vector for every question.
- **Dev selection** (Q&A format, dev questions only):
  1. Proxy grid, every direction × site × layer × α (360 configs): teacher-forced Δ = logit(Y) − logit(Z) after
     " X."; proxy net = #(Δ ≤ 0 → > 0) − #(Δ > 0 → ≤ 0).
  2. The top 3 proxy configs per direction (ties: smaller α, lower layer, LFT before ":") are run with generation.
     Each direction's config is the one with the highest dev generation net (fixes − breaks, by classifier class);
     ties keep the better proxy rank.
  3. **Primary direction** = highest dev generation net (ties: fmt, fd, cb in that order).
- **Test** (Q&A format, test questions), for each direction's chosen config: generation with the vector, and with 3
  random directions of the same norm (seed 20) at the same site, layer and α, the label null (fd, cb) and −v. A *fix* = baseline class not correct →
  steered correct; a *break* = baseline correct → steered not correct.

## Gates (any failure: exit 2)

1. Question tokenization: every question has two "." tokens with the LFT directly before "?\n", single-token Y and Z,
   and the same LFT token in both formats.
2. Zero vector: steering with α·v = 0 leaves the batched logits bitwise unchanged and the batched generations identical.
3. Active: a nonzero vector changes the logits.
4. Baseline matches R19's screen in ≥ 99% of classes (both formats).
5. Hooks removed at the end: probe logits bitwise unchanged.
In-code asserts (exception → exit 2): the steered padded position holds the expected token; the hook fires once on
prefill with the full length, then only on length-1 decode steps.
Report-only: batch vs single-row logits on the canonical prompt; batched vs single-row steered generation class on up
to 32 test rows whose class the primary vector changed, plus 32 seeded random rows.

## Pre-registered decision rule (per direction; the primary direction's label is the headline)

Over all test questions (Q&A format): fixes F, breaks B, net = F − B, two-sided sign test p on F vs B, break rate =
B / #baseline-correct. First match:

1. net ≤ 0 or p ≥ 0.01 → **"no reliable fix"**
2. net ≤ the best control net (3 random directions and, for fd/cb, the label null) → **"no better than controls"**
3. break rate > 0.02 → **"net-positive but damaging"**
4. fix rate among both-bug questions ≥ 0.3 → **"fixes errors beyond the format flip"**
5. fix rate among format-dependent questions ≥ 0.5 → **"fixes the format flip only"**
6. otherwise → **"small net improvement"**

Qualifier: for labels 3–6, if net ≤ the prompt-swap net, the label gets " (<= prompt swap)".
Break rate is reported with a Wilson 95% interval. `decide()` has a start-up selftest that covers every label.

Reported, no label:
- per category × order fixes and breaks (especially big-first fd, and breaks among reverse-dependent);
- every control's fixes, breaks and net;
- by order and by wording (W3 unseen);
- format-swap index on the discordant set (fd + reverse-dependent, ≈ 212 in test): how often the steered class equals
  the simple-format class, vs unsteered;
- the primary vector applied to simple-format prompts (fixes and breaks);
- the prompt-swap comparator;
- the canonical 9.8/9.11 Q&A prompt's steered text;
- direction norms vs resid norms (norm of the mean and mean per-token norm);
- per category, how many subset-baseline classes differ from the full-pass class;
- the whole dev proxy grid and dev generations;
- texts: all baseline outputs (both formats), every steered test output for each direction's config, and changed rows
  for controls and the simple-format run (for an independent re-classification at verification).

## What this can and cannot show

- One model, greedy decoding, three wordings, one question template family (decimals sharing an integer part, one
  single-digit and one two-digit fractional part). It does not test other number shapes or tasks.
- Selection over 360 proxy configs and 9 generation configs happens on dev; test is touched once per direction.
- A "fix" that only swaps format (and breaks reverse-dependent questions) shows up as a low net, a high break rate
  or a high agreement with the simple format.
- Generalization is across integer parts (X) and one wording; Y and Z values are shared across splits.

## Compute estimate

- Baseline 19,200 generations (~10 min, as R19); direction capture 5,120 forwards (~1 min).
- Dev proxy 360 × 1,280 teacher-forced forwards in batches of 64 (~10 min).
- Dev generation 9 × 1,280; subset baselines 2 × 3,840 + 1,280; test 17 × 3,840 (steered, 3 random, label null,
  −v per direction) + 3,840 simple-format: ≈ 90k generations, ~15–30 min.
- Walltime 4 h.

## Results

(pending)
