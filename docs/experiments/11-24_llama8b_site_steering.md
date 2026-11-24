# R21 — Can a fixed vector at R19's site repair big-first format errors, and is R19's patch question-specific?

**Status:** PRE-REGISTERED (2026-11-24, before any compute)
**Design review:** pending
**Script:** `9_8_research/reviewer_experiments/exp_R21_site_steering/run_experiment.py` (imports R20, full sha256 checked, and R19/R18/R17 through it).
**Job:** `run.sbatch` (`a100_tandon`, 4 h walltime). **Slurm job:** —
**Decision:** Gus, 2026-11-24: "your reco" (R21 over rewriting now).

## Question

R19: patching each big-first format-dependent question's own simple-format residual at the last fractional token (LFT),
layers 9–14, repaired 33/37. R20 tested fixed vectors, but its dev set had no big-first format-dependent questions and
no chosen config was at that token and those layers, so it did not test R19's site. R21 does.

- **Q1 (primary).** Does the mean patch over training questions (fd) repair held-out big-first format-dependent
  questions at the LFT, layers 9–14, better than controls, and without damage?
- **Q2.** Does a question's own patch repair more than another question's patch (question-specific)?

## Boring hypotheses, written first

- These questions sit near the decision boundary, so any push flips them. In R20, random vectors at other layers and
  scales fixed 0 to 25 of 27. Control: 20 random same-norm vectors per direction.
- The vector pushes toward the first-named number (here always the correct one). It would then "repair" big-first
  questions and break small-first ones. Control: the same vector on held-out small-first questions, break rate ≤ 2%.
- A generic format push does it. Control: 5 mean-difference vectors from random non-format-dependent train questions.

## Setup

- Model, snapshot, fp16, hooks: R17's. Steering, batching and the classifier: R20's and R18's. Greedy, 20 tokens, prefill-only.
- **Questions:** X 1..20, Y 2..9, Z 10..29 (3,200 triples), both orders. Wordings W1–W9 (in the script). Z 20..29 and
  W4–W9 are new.
- **Split (seed 21):** triples 30% train, 20% dev, 50% test. Train and dev use W1–W5, test uses W6–W9. Test shares no
  triple and no wording with train or dev. Test questions use wordings no earlier experiment ran. Some number pairs
  appeared in R18–R20 under other wordings, and R19's site itself was found on X 1..20.
- **Screen:** both formats, classified; categories as in R20. Directions and tests use big-first only.
- **Directions** (train, big-first, LFT, layers 9–14): **fd** = mean of (simple − Q&A) over format-dependent
  questions, the **pre-specified primary**; **fmt** = the same over all; **cb** = Q&A correct − Q&A bug mean.
- **Dev selection:** layer × α ∈ {0.5, 1, 1.5, 2, 3} per direction, by generation, highest population-weighted net
  (ties: lower layer, then smaller α). Dev uses all format- and reverse-dependent questions and seeded samples of
  the rest (100 both-bug, 300 both-correct, 50 other). The own-patch layer is the one with the highest dev repair rate.
- **Test:** all format- and reverse-dependent big-first questions plus seeded samples (300 both-bug, 600 both-correct,
  100 other), each weighted N_category / n_sampled. Baseline re-run in the same batches.
- **Conditions on test:** own patch at each layer 9–14 (own simple − Q&A, added); at the own-patch layer, 5 draws
  of another train format-dependent question's delta and 5 random vectors of each question's own delta norm; for each
  direction's config: the vector, −v, 20 random same-norm vectors, 5 nulls (fd: random non-format-dependent train
  subsets of the same size; cb: shuffled labels; fmt: none), and the vector on up to 600 small-first test questions.
- **Prompt swap:** simple format instead of Q&A, over the whole big-first test population (screen classes).

## Gates (any failure: exit 2)

Tokenization and offset-2 alignment for every question; zero vector leaves logits bitwise unchanged; adding the own
delta on the canonical question reproduces R17's p13 r within 0.05 at layers 9–14, and Δ_clean, Δ_buggy within 0.04; hooks removed
at the end. Report-only: subset-batched vs screen baseline classes.

## Decision rule (per direction; fd is the headline). Fix rate = share of test format-dependent questions answered
correctly under steering. First match:

1. fewer than 20 test, 10 dev or 10 train big-first format-dependent questions → "descriptive only"
2. own patch fix rate at the dev-chosen layer < 0.5 → "site does not replicate on fresh questions"
3. fix rate < 0.5 → "fixed vector does not repair the site"
4. fix rate ≤ the best of the 20 random and 5 null fix rates → "no better than controls"
5. weighted net > 0, above the prompt-swap net and above all 20 random nets, and weighted break rate ≤ 0.02 in big-first and in
   small-first → "usable fix on big-first questions"
6. otherwise → "repairs the site but not usable"

Q2 label: own < 0.5 → "not testable"; own − mean(other) ≥ 0.3 → "question-specific"; ≤ 0.1 → "shared across questions";
otherwise "mixed". `decide()` and `decide_specific()` have start-up selftests.

Secondary: fmt and cb get the same label and are not the headline. Reported: every condition's per-category fixes and
breaks, Wilson interval on the fix rate, the cosine of individual fd deltas to their mean, norms, dev grid, all
baseline texts and changed texts.

## Limits

One model, greedy, one template family. With 20 random vectors, "beats all random" is a one-sided rank of 1/21, not a
small p. "Usable" is on big-first questions only and would need to know the order at inference. The question sets are
related to R17–R20's, so this is fresh in wording and triples, not in task.

## Compute estimate

Screen: 3,200 triples × 2 orders × 2 formats; 50% × 5 wordings + 50% × 4 wordings ≈ 57,600 generations (R19: 19,200 in 168 s,
so ≈ 9 min). Dev: 90 configs + 6 own-patch layers × ≤ ~500 questions ≈ 48k (≈ 8 min). Test: about 103 conditions × ≤ ~1,200 ≈ 124k
(≈ 20 min). About 40 min, 0.7 GPU-h. Walltime 4 h.

## Results

(pending)
