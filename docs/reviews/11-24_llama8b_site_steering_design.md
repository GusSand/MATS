Reviewer: fresh subagent (Claude Code)
Agent-ID: a4064591cc8d8cd59
Verdict: FIXES NEEDED
Transcript-SHA256: 390303dbc669fcd4e8a306e59fb320c79a95d77eddb04560e4efa11bc18a43c7

## Brief given

You are an adversarial design reviewer for a pre-registered experiment that has not run yet, in `~/github/MATS`. Read only: no edits, no GPU, no submission. Local python for arithmetic is fine.

Review `docs/experiments/11-24_llama8b_site_steering.md`, `9_8_research/reviewer_experiments/exp_R21_site_steering/{run_experiment.py,run.sbatch}`, and `scripts/{submit.sh,dry_run_check.py}`. Sources to check facts against: `docs/experiments/11-23_llama8b_last_fraction.md`, `docs/experiments/11-24_llama8b_steering_fix.md`, `docs/research_journal.md`, `docs/decisions.md`, and the imported scripts exp_R17_localization_map, exp_R18_eleven_token, exp_R19_last_fraction, exp_R20_steering_fix (with their results/run.json).

Checklist:
1. Fact-check every number against its source.
2. Recompute all arithmetic, including the compute estimate, generation counts and expected category counts in each split (use the R19 screen in its run.json to project the big-first format-dependent count in train, dev and test; say whether the minimum-n rule is likely to trigger).
3. Terminology and sign consistency.
4. Operational definitions: every "matched", "null", "weighted", "held-out" term.
5. Circularity and leakage: selection vs test, split construction, wording/triple overlap, reuse of earlier data.
6. Category-check of cited facts.
7. Matching controls: are they pre-registered, adequate, and implemented as described?
8. Code correctness: left-padded positions, per-question vectors aligned to batches, capture positions, label-null construction, subset weights, fix/break accounting, first-match rule vs doc, selftests, device/dtype, JSON saving cost (run.json is rewritten after every condition), runtime within 4 h, smoke-only branches.
9. Any path where the rule yields a misleading label.

First line: PASS, PASS WITH MINOR FIXES, or FIXES NEEDED. Then numbered findings with severity (blocking / should-fix / minor), file:line and a concrete fix.

## Report

FIXES NEEDED

I read the doc, `run_experiment.py`, `run.sbatch`, `submit.sh`, `dry_run_check.py`, the R17–R20 scripts (R17 lines 1–260, R18 lines 1–330), and the R19 and R20 `run.json`. Counts, weights and conditions were recomputed locally. Nothing was edited or submitted. Hashes: the R20 pin `effa9b29…` matches. The R21 script is `e6759f3f…` and matches the smoke log header.

"Script" below means `9_8_research/reviewer_experiments/exp_R21_site_steering/run_experiment.py`. "Doc" means `docs/experiments/11-24_llama8b_site_steering.md`.

## Blocking

**1. The ground truth is wrong for Y=2 with Z≥20** (script:33, 127; doc "Questions").
- `big = X.Y` is hard-coded. X.2 = X.20 is a tie, and X.2 < X.21…X.29.
- 200 of 3,200 triples are mislabelled (20 ties, 180 reversed): train 61, dev 41, test 98. All enter the screen, so test alone has 98×4 = 392 big-first questions.
- There a correct model answer is classed "bug". This corrupts every category, the directions (fd, cb), weights, break rates and the "first-named is always correct" premise. R18–R20 were safe because Z ≤ 19.
- Fix: exclude Y=2 for Z≥20 (3,000 triples), or use Y 3..9. Add a start-up gate that X.Y > X.Z for every question as a Fraction. Update all counts (3,000 triples gives 900/600/1,500 and about 54,000 screen generations).

**2. Dev selection optimises the pre-registered confound and not the Q1 estimand** (script:329–343; doc "Dev selection").
- Dev is big-first only, so the correct answer is always first-named, and the objective is weighted net.
- Projected dev big-first (from the R19 screen): fd about 24–32 at weight 1. Both-bug about 479 in 100 samples (weight ≈4.8). Both-correct about 2,629 in 300 samples (weight ≈8.8).
- So one both-bug fix is worth about 5 fd fixes. All fd together (≤32) are swamped by up to 479 both-bug. The chosen config will be a strong first-number push if one exists, and the small-first break check sees it only after it is fixed on test.
- Conversely, one or two sampled both-correct breaks (8.8 each) can make selection drop an fd-repairing config.
- Rule 3, "fixed vector does not repair the site", would then be a statement about a net-selected config. R20 had the same selection/site mismatch.
- Q1 asks about "the mean patch", which is α=1 at R19's layers, but that config is not guaranteed to be tested.
- Fix: pre-specify the headline config as fd, α=1, at the own-patch layer (or a fixed layer 9, where R19's map peaks). Report the net-selected config as secondary. If dev selection stays in the headline, include small-first dev questions, constrain dev small-first break rate ≤2%, and separately select by dev fd fix rate. Restrict label 3 to the configs actually tested.

**3. Own-patch and Q2 labels ignore the any-push control** (script:81, 92–101, 385–391; doc rules 2 and Q2).
- `own_norm_random` is run and never enters a decision. R20 random vectors fixed up to 25/27 at an early layer.
- If random pushes of the own norm also fix ≥0.5, then rule 2 passes "site replicates", and other ≈ own gives "shared across questions". Both are misleading.
- Fix: Q2 needs own ≥0.5 AND own − mean(own_norm_random) ≥0.3 (else "not specific: any push"). "Shared" needs other > max(own_norm_random). Apply the same test to rule 2, and add selftests.

## Should-fix

**4. Minimum-n projection.** R19's big-first fd rate is 48/4,800 = 1.0%. With the seed-21 split applied to R19's screen (Z<20 triples, W1–W3):

| | train | dev | test |
|---|---|---|---|
| fd in R19's screen (W1–W3) | 14 | 14 | 20 |
| scaled to the wordings used here | ≈23 | ≈23 | ≈27 |
| adding Z 20..29 at the Z 15–19 rate | ≈30 | ≈28 | ≈36 |

- A naive 1.0% flat rate gives 48/32/64, which is too optimistic. Z 10–14 has 41 of 48 fd and Z 15–19 only 7 (7/2,400), so Z 20–29 may add little or nothing.
- The rule is not the expected outcome, but it is a real risk. Poisson P(test<20) is about 8–10% at mean 26–27, under 1% at 32. Wording variation is extra: W1 11, W2 20, W3 17, so four new wordings behaving like W1 give about 18 test fd, which triggers. P(dev<10) is negligible at about 23.
- Projected test eval set (flat R19 rates): fd 64, reverse 120, both-bug 957 capped to 300, both-correct 5,259 capped to 600, other ≈0. Total ≈1,084. With fd ≈30, ≈1,050.
- Fix: Z 10..19 with more X (X 1..40) raises n and removes finding 1. If Z 20..29 stays, state the projection in the doc. Expected counts per split are missing (R19 and R20 docs had them).

**5. Break-rate check is underpowered and the prompt-swap clause is vacuous** (script:39–41, 84–88, 355; doc rule 5).
- Small-first samples 150 both-correct of about 4,760, weight ≈31.7. The 2% threshold is 3 sampled breaks (3 pass, 4 fail). Wilson upper bound at 0/150 is 2.5% and at 3/150 is 5.7%, so even 0 observed breaks cannot show ≤2%.
- Big-first both-correct n=600: 12/600 has an upper bound of 3.5%.
- "Up to 600 small-first" is wrong. The cap is 150 per category over 5 categories (max 750). Projected about 470.
- Breaking every reverse-dependent question alone is about 2.2% (90/4,034 in R19), so a perfect format swap already fails. State that.
- The big-first prompt-swap net is negative (48−90 = −42 in R19), so "above prompt-swap net" is implied by net>0. The relevant trivial baseline in big-first is "answer the first number", which is 100% correct.
- Fix: both-correct caps ≥1,500 in both orders (compute is cheap). Require the Wilson upper bound ≤2%, or call it "point estimate ≤2%". Drop or relabel the prompt-swap clause.

**6. "Question-specific" is confounded with wording** (script:382–384). "Other" deltas come from train (W1–W5) and are applied to W6–W9 test questions, so a wording difference looks like question-specificity. Fix: add leave-one-out donors from other test fd questions, same wording and different triple. Give Q2 a minimum n (it currently labels even at n_test_fd=3), and gate it on the same counts as rule 1.

**7. fd null is not matched** (script:313–317; doc Setup). The fd set is Z-skewed (R19: 35 of 48 at Z 10–13) and X-skewed (14 of 48 at X=8). The non-fd pool is uniform over Z 10..29. The LFT token in big-first is Z itself, so fd vs null can differ by token content and not by format dependence. Fix: sample nulls matched to the fd (Z, wording) histogram. Save cos(fd, null) and cos(fd, fmt) as R20 did. Same caution for cb, whose correct and bug sets differ in Z.

**8. Rule 4 merges two different readings, and rule 5 omits the nulls** (script:84, 86; doc rules 4 and 5). Losing to a random vector and tying a generic-format null both print "no better than controls". At ceiling (both 1.0) a tie with the null gives that label even if the format vector clearly works. Rule 5 compares net to the 20 randoms only. Fix: two labels ("no better than random" / "no better than a generic format vector"), and include null nets in rule 5.

**9. Held-out wording and triple claims are overstated** (doc Setup, Limits).
- 799 of 1,600 test triples have Z<20 and ran in R19 and R20 under W1–W3. "Fresh in wording and triples" contradicts the Setup paragraph.
- Nothing within the triple is held out: 100% of test (X,Y) pairs, 90% of (X,Z) pairs and 99% of (Y,Z) pairs occur in train triples.
- W4 "Which is greater:" was run on Llama with 9.8/9.11 in both formats (`exp_R2_generalization/run_experiment.py:205–206`). W6 was run in Pythia (`test_memorization_hypothesis.py:81`). So "W4–W9 are new" and "wordings no earlier experiment ran" are false as written.
- Fix: say "new as (wording, triple) combinations; numbers shared", or hold out X as R20 did.

**10. R19 comparability.** 33/37 is after unbatched eligibility (5 of 48 screened dropped: 4 baseline-delta, 1 unbatched), with a per-question best layer, r ≥0.5, random mean <0.2 and margin ≥0.3. R21's "repair" is generation-only at one layer on all screened fd. Generation alone was 34/37 in R19. At fixed layers the r≥0.5 counts are 35/37 at L9 and 34/37 at L10–14. Fix: state the population and criterion difference, and define "repair" once. Optionally report an R19-style unbatched eligibility subset.

**11. Experiment ID collision.** R21 is already `exp_R21_gpu_margin` (jobs 24375606 and 24375607, `docs/decisions.md:9–10`, journal 4709 and 4732). This doc, job name, `R21_*` env vars and the `results/r21_<job>` prefix reuse it. Rename to R22. Also, the journal (4734) records "your reco" → test R19's site. "(R21 over rewriting now)" has no source, and `decisions.md` has no line. Add one.

**12. R20's reproducibility flag is repeated** (script:334–351). Vectors (fd, fmt, cb, nulls) and dev generations and classes are again not saved. R20's blind re-verification flagged exactly this. Save vectors to a `.pt`, save dev classes and changed texts, and report the dev subset-baseline disagreement (only test and small are stored now, although the doc lists it as report-only).

**13. Steering gates run only on a single unpadded canonical row** (script:252–269). Left-padding and per-question alignment are never tested in a mixed-length batch. `Steer`'s token assert is tautological (`r20:208`). I read the padding and index code as correct, but add a gate: a mixed-wording batch, own delta added, captured residual equals the simple-format residual at each row's LFT. Also zero vectors leave logits and generations unchanged on that batch.

**14. `scripts/dry_run_check.py` and `submit.sh` gaps.**
- The experiment file is not in the committed-check (`:17`), so the pre-registration can be uncommitted.
- A review PASS is not tied to the script hash. R20's "script changed after review round 2, no third round" (journal 4734 item 3) would pass. Fix: put the reviewed sha256 in the review line and compare it, as the R19 and R20 docs do with "Expected script sha256" (missing here).
- R17–R20 imports and `R17_REF` are neither checked nor copied to Torch, so they must already exist there.
- The `git fetch` return code is ignored (`:19`).
- `smoke_cpu.log` is gitignored (`.gitignore:38`), so the smoke evidence is not committed.
- `DIR` must be repo-root relative because `submit.sh` cds first.

## Minor

15. Condition count: 6 own + 5 other + 5 own-norm + 27 (fd) + 22 (fmt) + 27 (cb) = 92 conditions on the test subset, plus 1 baseline and 4 small-first runs. The doc says 103 and 124k. Projected ≈93×1,084 + 4×470 ≈ 103k, so 124k is a safe upper bound. Screen 57,600 is right (54,000 after finding 1). 168.4 s per 19,200 gives 8.4 min. Dev 97×≈492 ≈ 48k. R20's job took 31 min wall-clock. Total ≈40 min holds, well inside 4 h.
16. JSON saving: I measured ≈15 MB final and ≈0.2 s per full `dump` locally (more on Torch). At 92 saves it is tens of seconds to a few minutes, not a walltime risk. Writing the screen to its own file still avoids rewriting it.
17. `decide` selftest covers `n_dev_fd` but not `n_test_fd` or `n_train_fd`, nor the "no better" boundary. First-match order and rules otherwise match the doc.
18. Doc says any failure exits 2, but `r18.load_r17` uses `sys.exit(str)` (exit 1). The `nfd==0` early return sets no primary-direction entry.
19. Boring-hypothesis wording: "at other layers and scales" is loose. fmt's randoms were at L9 but at ":" (other site). R20's 0–25 set is quoted correctly.
20. R18's classifier calls some "Explanation:" simple answers bugs (R20 flag 3), which shifts fd membership. State it and report `neither` rates per new wording.
21. fd fix rate counts subset-baseline-correct questions as fixed (script:173). Report how many fd have `base == correct`.
22. Sequential RNG: `g` is shared by own-norm and all random draws, and `nrng` by nulls and donors, so reordering conditions changes draws. Use a seed per condition.

## Checked and fine

- Signs: fd and fmt = simple − Q&A, cb = correct − bug. Both are added to Q&A, so they point the intended way. −v is the correct flip.
- Left-pad arithmetic in `capture`, `steer` indexing by `id(q)` and batching, pos + pad, device/dtype handling.
- Weights, including the empty-category path. Fix/break accounting. Tie-breaks ("lower layer, then smaller α", strict >) match the doc.
- Random vectors scaled to `‖α·v‖`. cb null shuffles labels over correct/bug only.
- Own-delta gate reproduces R17's p13 in structure and tolerance as documented. Model and env chain R21→R20→R19→R18→R17/R13.
- The R20 facts quoted in the Question section (no big-first fd in dev; chosen configs fd LFT L4, cb LFT L6, fmt ":" L9) match `run.json`.
- Alignment gate logic holds for all nine wordings (smoke ran with the real tokenizer).
- R20's `binom_two_sided`, `transitions` and similar are not used.
