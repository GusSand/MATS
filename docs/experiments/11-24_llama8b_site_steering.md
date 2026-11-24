# R22 — Can a fixed vector at R19's site repair big-first format errors, and is R19's patch question-specific?

**Status:** PRE-REGISTERED (2026-11-24, before any compute; amended after design review rounds 1–3)
**Design review:** round 1 FIXES NEEDED — fresh subagent, 2026-11-24 — docs/reviews/11-24_llama8b_site_steering_design.md
**Design review:** round 2 FIXES NEEDED — fresh subagent, 2026-11-24 — docs/reviews/11-24_llama8b_site_steering_design_v2.md
**Design review:** PASS WITH MINOR FIXES (round 3; no script change, document and preflight fixes applied) — fresh subagent, 2026-11-24 — docs/reviews/11-24_llama8b_site_steering_design_v3.md
Reviewed script sha256: `aa9c9b02d66b6f4028b56d16745512fd9ccab809a361c8dd59571b329417dba2`.
**ID:** first drafted as R21. That ID already belonged to docs/experiments/11-24_llama8b_gpu_margin_offset.md, so this is R22.
**Script:** `9_8_research/reviewer_experiments/exp_R22_site_steering/run_experiment.py` (imports R20, full sha256 checked, and R19/R18/R17 through it).
**Job:** `run.sbatch` (`a100_tandon`, 4 h walltime). **Slurm job:** —
**Decision:** Gus, 2026-11-24: "your reco" (test R19's site before rewriting; docs/decisions.md).

## Question

R19: patching each big-first format-dependent question's own simple-format residual at the last fractional token (LFT),
layers 9–14, repaired 33/37. R20 tested fixed vectors, but its dev set had no big-first format-dependent questions and
no chosen config was at that token and those layers, so it did not test R19's site. R22 does.

- **Q1 (primary).** Does the mean patch over training questions (fd) repair held-out big-first format-dependent
  questions at the LFT, layers 9–14, better than controls, and without damage?
- **Q2.** Does a question's own patch repair more than another question's patch (question-specific)?

## Boring hypotheses, written first

- These questions sit near the decision boundary, so any push flips them. In R20, random vectors (other layers, scales,
  and for fmt another site) fixed 0 to 25 of 27. Controls: 20 random same-norm vectors per config, and 20 random pushes with
  each question's own patch norm. The own patch must beat the latter by 0.3, or the own patch counts as not distinguishable from random pushes at this margin. Differences are evaluated in floating point.
- The vector pushes toward the first-named number, which in big-first is always correct. It would then break small-first
  questions. Check: the same vector on held-out small-first questions.
- A generic format push does it. Control: 5 mean-difference vectors from non-format-dependent train questions matched to the fd set on wording and Z.
- "Question-specific" is really wording-specific. Control: other questions' patches come from test questions with the same wording.

## Setup

- Model, snapshot, fp16, hooks: R17's. Steering, batching and the classifier: R20's and R18's. Greedy, 20 tokens, prefill-only.
- **Questions:** X 1..40, Y 2..9, Z 10..19 (3,200 triples; X.Y > X.Z is checked as a fraction for each), both orders, wordings W1–W9 (in the script).
- **Split (seed 21):** triples 30% train, 20% dev, 50% test. Train and dev use W1–W5, test uses W6–W9. No triple or wording
  is shared between test and train/dev. The (wording, triple) combinations in test are new. The numbers are not: triples with X ≤ 20 ran in R19 and R20 under
  W1–W3, and X, Y and Z values recur across splits. X 21–40 is new. "Which is greater:" (W4) was run on the canonical pair before.
- **Expected big-first format-dependent questions:** R19's rate on X 1..20 is 48/4,800 (1.0%), and X 11..20 is about 0.4%. Applying the seed-21 split to R19's screen and scaling to the wordings: train ≈ 30–39, dev ≈ 8–14, test ≈ 33–45. Dev may fall below 10, which affects only the secondary configs; the seed is not changed to avoid it. New wordings could lower this. The
  minimum-n rule then gives "descriptive only".
- **Screen:** both formats, classified; categories as in R20. R18's classifier follows an "Explanation:" line over a bare answer (R20 flag 3); this shifts some categories.
- **Directions** (train, big-first, LFT, layers 9–14): **fd** = mean of (simple − Q&A) over format-dependent
  questions; **fmt** = the same over all; **cb** = Q&A correct − Q&A bug mean.
- **Headline config, fixed in advance: fd, α = 1, layer 9** (R19's modal best layer). No selection.
- **Secondary configs:** for fd, fmt and cb, layer × α ∈ {0.5, 1, 1.5, 2, 3} chosen on dev by generation, highest population-weighted net summed over
  big-first and small-first (ties: lower layer, then smaller α). Dev big-first uses all format- and reverse-dependent questions plus seeded samples
  (100 both-bug, 300 both-correct, 50 other); small-first caps format- and reverse-dependent at 100 each.
- **Test:** all big-first format- and reverse-dependent questions plus seeded samples (300 both-bug, 1,500
  both-correct, 100 other). Small-first has the same caps, with format- and reverse-dependent capped at 300. Each row is weighted N_category / n_sampled. Baselines are re-run in the same batches.
- **Conditions:** own patch (own simple − Q&A, added) at each layer 9–14; at layer 9, 20 random pushes of each question's own patch norm; on the format-dependent
  questions, 5 draws of another same-wording test question's patch (different triple) and 5 of a train format-dependent question's; for each of the four configs: the vector, −v, 20 random same-norm vectors,
  5 nulls (fd: matched non-format-dependent; cb: shuffled labels; fmt: none), and the vector on small-first. Each draw index has its own seed; the fd and cb nulls share it, and so do the two donor types. Unpaired rows in the saved `other` classes are self-patches and are ignored.
- **Relation to R19.** R19 reported 33/37 after unbatched eligibility, with a per-question best layer, r, a random-control mean and a margin. Here "repair" is generation class only,
  at one fixed layer, over all screened format-dependent questions. The two rates are not the same quantity.

## Gates (any failure: exit 2)

Tokenization, X.Y > X.Z and offset-2 alignment for every question; zero vector leaves logits bitwise unchanged on the canonical question; in a mixed-length
left-padded batch, adding the own patch gives the simple-format residual at each row's LFT (relative error < 1%) and the zero vector leaves generations identical;
adding the own patch on the canonical question reproduces R17's p13 r within 0.05 at layers 9–14, and Δ_clean, Δ_buggy within 0.04;
hooks removed at the end. Report-only: batched vs screen baseline classes. A wrong import hash or a missing snapshot also stops the run (exit 2, except the R17 hash and the snapshot check, which exit 1).

## Decision rule (headline: fd, α 1, layer 9; the three dev-selected configs get the same rule as secondary)

Fix rate = share of big-first format-dependent test questions answered correctly under steering. First match:

1. fewer than 20 test or 10 train big-first format-dependent questions → "descriptive only". The dev-selected configs also need 10 dev questions; the headline uses no dev data
2. own patch at layer 9 fixes < 0.5 → "site does not replicate on fresh questions"; it beats the mean of the own-norm random pushes by < 0.3 → "own patch not distinguishable from random pushes"
3. fix rate < 0.5 → "vector does not repair the site"
4. fix rate ≤ the best of the 20 random vectors → "no better than random"
5. fix rate ≤ the best of the 5 nulls → "no better than a generic format vector" (for cb this means shuffled labels; fmt has no null test)
6. big-first weighted net > 0 and above all random and null nets, big-first plus small-first weighted net > 0, and weighted break rate (point estimate) ≤ 0.02 in big-first and in small-first → "usable fix on big-first questions"
7. otherwise → "repairs the site but not usable"

**Q2** (questions with a same-wording partner, for own, other and random alike): fewer than 20 → "not testable (too few questions)"; own < 0.5 or own − mean(own-norm random) < 0.3 → "not testable (own patch does not beat random
pushes)"; own − mean(same-wording other) ≥ 0.3 → "question-specific"; difference ≤ 0.1 and mean(other) > best own-norm random → "shared across questions"; otherwise "mixed".

`decide()` and `decide_specific()` have start-up selftests. The break-rate threshold is a point estimate; Wilson intervals on both-correct breaks are reported for both orders. Secondary labels carry their config, and a dev-selected config equal to the headline is flagged as a duplicate.
Breaking every reverse-dependent question alone is about 2.2% of correct big-first answers (R19: 90/4,034), so a pure format swap fails rule 6.

Reported, no label: per-category fixes and breaks, Wilson interval on the fix rate, how many format-dependent questions are already correct in the batched baseline, fd–fmt and fd–null cosines,
norms, the prompt-swap comparator (its big-first net was negative in R19: 48 − 90), the dev grid and dev classes (`dev.json`), the vectors (`vectors.pt`, `own_fd.pt`), screen texts (`screen.json`), and test texts (baseline and changed).

## Limits

One model, greedy, one template family. With 20 random vectors, "beats all random" is a one-sided rank of 1/21, not a
small p. "Usable" is on big-first questions only and would need to know the order at inference. The question sets are
related to R17–R20's, so this is fresh in (wording, triple) combinations, not in numbers or task.

The fd null is matched on wording and Z but not X; R19's format-dependent set is skewed to X = 8 (14 of 48).

## Compute estimate

Screen: 3,200 triples × (50% × 5 + 50% × 4 wordings) × 2 orders × 2 formats = 57,600 generations (R19: 19,200 in 168 s, so about 9 min). Dev: 90 configs × about 1,000 = 90k (about 13 min).
Test: 6 + 20 + 27 + 27 + 22 + 27 = 129 conditions × about 2,000 = 258k (about 38 min), plus small-first and the format-dependent-only draws. About 65–80 min, 1.3 GPU-h. Walltime 4 h.

## Results

(pending)
