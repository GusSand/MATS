Reviewer: fresh subagent (Claude Code)
Agent-ID: ad02bc5fd63796fab
Verdict: FIXES NEEDED
Transcript-SHA256: a8b4814b4a36629447518ff5c6a47c08ceb7bf2d81497723177bbd378864089f

## Brief given

You are an adversarial design reviewer (round 2) for a pre-registered experiment that has not run, in `~/github/MATS`. Read only: no edits, no GPU, no submission. Local python for arithmetic is fine.

Review `docs/experiments/11-24_llama8b_site_steering.md`, `9_8_research/reviewer_experiments/exp_R22_site_steering/{run_experiment.py,run.sbatch}`, and `scripts/{submit.sh,dry_run_check.py}`. Round-1 report: `docs/reviews/11-24_llama8b_site_steering_design.md`. Sources: `docs/experiments/11-23_llama8b_last_fraction.md`, `docs/experiments/11-24_llama8b_steering_fix.md`, `docs/research_journal.md`, `docs/decisions.md`, the imported R17–R20 scripts and their run.json files.

Check: (1) every round-1 finding is correctly resolved, or state which remain; (2) new bugs (per-question vectors, leave-one-out donors, matched nulls, weights, mixed-batch gate and its hook ordering, the headline config, seeds, dev selection over both orders, early exits, saved files); (3) doc and code agree on every rule and number; (4) arithmetic and compute, including projected counts per split from the R19 screen; (5) any path to a misleading label; (6) whether the preflight and submit scripts do what they claim.

First line: PASS, PASS WITH MINOR FIXES, or FIXES NEEDED. Then numbered findings with severity (blocking / should-fix / minor), file:line and a concrete fix. State the sha256 of the run_experiment.py you reviewed.

## Report

FIXES NEEDED

Reviewed `run_experiment.py` sha256: `968e33d022773699d2f07f41b3fe378f1f78893497a585293fe73dcef660a0bc`. It matches the header of `smoke_cpu.log`. Nothing was edited or submitted. I ran local arithmetic and the read-only preflight (which does a `git fetch`). I did not run the model or contact Torch.

There is one blocking finding, and the fix is about two lines. The steering and patching code is correct as far as I could trace it. A diff-level re-check of the new hash is enough; a full third round is not needed.

"Script" = `9_8_research/reviewer_experiments/exp_R22_site_steering/run_experiment.py`. "Doc" = `docs/experiments/11-24_llama8b_site_steering.md`.

## Blocking

**1. The dev minimum-n rule can void the headline, which uses no dev data. The doc's dev projection is wrong for seed 21** (script:78, script:338; doc:38–39, doc:66).
- The doc's projection is a flat 1.0% on X ≤ 20, plus 0–0.4% on X 21–40. It ignores the actual split.
- I applied the seed-21 split to R19's screen (48 big-first format-dependent questions, W1–W3), then scaled to the wordings used:

| Split | In R19's screen | X ≤ 20, scaled to wordings | + X 21–40 at 0–0.4% | Doc says |
|---|---|---|---|---|
| train | 18 | 30 | 30–39 | 24–33 |
| **dev** | **5** | **8.3** | **8–14** | **16–22** |
| test | 25 | 33 | 33–45 | 32–45 |

- The 5 dev questions are W2 ×2 and W3 ×3. Treating them as known, P(dev < 10) is about:
  - 0.76 if X 21–40 gives none;
  - 0.05 at the X 11–20 rate (9/2,400 = 0.375%);
  - 0.014 at the X 16–20 rate (0.5%).
- This is Poisson. Hits cluster by triple across wordings, so the true tails are wider.
- P(test < 20) is about 0.005 or lower, if W6–W9 behave like W1–W3 on average.
- The headline (fd, α 1, layer 9) involves no dev selection. With dev = 8 and test = 35, it would still print "descriptive only (too few format-dependent questions)". That is a misleading label.
- The early exit at script:338 also aborts Q1 and Q2 entirely if dev = 0.
- Fix:
  - Apply `n_dev_fd` only to the three `_selected` configs. For example, pass `n_dev_fd=None` for the headline and skip the check when it is None.
  - Make the early exit test train and test only.
  - Update doc rule 1, the projection and the selftest.
  - Do not change the seed to get a better dev count.

## Should-fix

**2. Rule 2 merges two different failures** (script:80–81; doc:67).
- Own 0.95 with own-norm random mean 0.70 prints "site does not replicate on fresh questions". But the site did repair; any push also does.
- Fix: two labels. Own < 0.5 → "site does not replicate on fresh questions". Own − mean(random) < 0.3 → "own patch not distinguishable from random pushes". Update the selftests at 114–115.

**3. "Usable" can be net-negative over both orders** (script:88–90; doc:71).
- Small-first is checked only by break rate. 2% of the projected 4,760 small-first both-correct questions is about 95 breaks. Big-first fd fixes are about 33–45.
- `small_net` is computed (script:495) but never used in a rule.
- Fix: also require `net + small_net > 0`, or state in the label and rule that small-first net is not checked.
- Also report a small-first both-correct Wilson interval. Only big-first is saved (script:494), and doc:77 does not say so.

**4. The preflight does not tie a PASS to the reviewed hash** (scripts/dry_run_check.py:33–36).
- Any `**Design review:** PASS` or `FIXED` line anywhere passes, even with a later REJECTED or FIXES NEEDED line. `check_reviews.py` uses the last occurrence.
- `FIXED` is written by the doer, and so is the `Reviewed script sha256` line. R20's problem (script changed after round 2, no third round) still passes if the doer types the new hash.
- Fix: take the last design-review line, open its `docs/reviews/*.md`, require that report to exist and to be committed, and require `sha` to appear in its `## Report` section.
- The doc's current line format ("round 1 FIXES NEEDED", "round 2 …") never matches the regex, so the final line must start with exactly `**Design review:** PASS`.

**5. The preflight and submit scripts leave the rest of round-1 finding 14 open.**
- Extra files passed to `submit.sh` are copied but never checked for being committed (dry_run_check.py:17 covers only script, sbatch and doc).
- The R17–R20 and R13 scripts and the `R17_REF` run.json are neither checked nor copied. A missing or stale one is found only after queueing. The script hashes are checked at run time; `R17_REF` is not hashed at all.
- `smoke_cpu.log` is git-ignored (`.gitignore:38`). R17–R20's logs are tracked, R22's is not, and the preflight reads only the local copy.
- `submit.sh` uses `DIR` raw. With an absolute path, `rsync -R` copies to `/scratch/gs157/MATS/Users/…`, the hash comparison still passes there, and the job starts in a directory without the imports.
- Fix: add extra files, imports and `R17_REF` to the porcelain check; compare their hashes on Torch; `git add -f` the smoke log; reject absolute `DIR`.

**6. `decide_specific` donors fall back silently** (script:450–451).
- If a test question has no same-wording partner, the donor comes from another wording, or is the question's own patch if it is alone. The result still enters "other_same_wording". That is the confound round-1 finding 6 asked to remove.
- Fix: count fallbacks and save the count, or exclude those questions from both own and other.

## Minor

**7.** Doc:15 "R21 does." should be R22. `submit.sh:2` still says "e.g. R21".

**8.** Doc:51 "Every random draw has its own seed" is false. `nrng` (script:355) is shared by the fd and cb nulls, and `trng` (script:445) by both donor types and all draws. Random vectors are individually seeded. Either seed per draw or correct the sentence.

**9.** Doc:60 says R18 or R19 hash failures exit "2 or 1". In fact R20, R19 and R18 exit 2 and R17 exits 1 (`exp_R18/run_experiment.py:86`). A missing snapshot exits 1 (`run.sbatch`). "Any failure: exit 2" is therefore not literally true.

**10.** Secondary configs are chosen by population-weighted net. Projected dev weights: big-first both-bug 4.8, both-correct 8.8; small-first fd 2.3, both-bug 5.8, both-correct 7.9. Rule 3 on those configs ("vector does not repair the site") says more than was tested. Prefix secondary labels with the config, e.g. "fd_selected L12 α 3: …".

**11.** If dev selects fd, α 1, layer 9, then `fd` and `fd_selected` are the same vector with different random draws. They can get different labels. State that `fd` is the headline and flag the duplicate.

**12.** The fd null is matched on (wording, Z) as documented, but not X. R19's fd set is X-skewed (14 of 48 at X = 8). Say so in Limits, or also match on X where the pool allows.

**13.** `vectors.pt` omits `ev_own` (the test own patches), so own-norm random and donor conditions cannot be rebuilt without the model. Saving only the headline layer for the fd rows is cheap. `changed_texts` omits unchanged-class texts, which is acceptable.

**14.** The journal entry for R22 says round-1 findings were "All fixed". See the next section.

## Round-1 findings: status

- **Resolved:** 1, 3, 5, 8, 9, 10, 12, 13, 15, 16, 17, 18 (the decision-dict part; exit text is finding 9 above), 19, 20, 21.
- **Partly resolved:**
  - 2: headline fixed in advance; secondary still net-dominated (finding 10).
  - 4: projections added but wrong for the split (finding 1).
  - 6: same-wording donors and minimum n added; silent fallback (finding 6).
  - 7: (wording, Z) matching and cosines added; X unmatched (finding 12).
  - 11: renamed; stale ID at doc:15.
  - 14: experiment file, fetch check and sha line added; the rest is in findings 4 and 5.
  - 22: random vectors seeded; nulls and donors still sequential.

## Checked and correct

- **Ground truth:** Z is 10..19 and the Fraction gate covers every question. The LFT/offset gate holds for all nine wordings in the smoke run.
- **Mixed-batch gate and hook order:** the steer pre-hook is registered before the grab hook and pre-hooks run in registration order, so grab sees the steered residual. It discriminates: R20's saved norms at L9 LFT are 0.58–0.63 for the mean delta and about 6.3 for the residual, so an unsteered run would miss by roughly 9–10% against the 1% tolerance. The canonical R17 gate uses independently hard-coded positions.
- **Per-question vectors:** `id(q)` lookup, batch slicing, pad arithmetic, per-layer closures, and `ev_own[pick, kh]` / `fd_deltas[[…], kh]` indexing are right.
- **Donors, nulls, weights:** leave-one-out is right apart from finding 6. Nulls are rescaled to the steered norm. Weights are N_category / n_sampled with empty categories handled. Dev selection sums both orders, and strict `>` gives lower layer then smaller α.
- **Headline:** fd, α 1, layer 9 is fixed in code. R19's modal best layer is 9 (29 of 37; 35/37 with r ≥ 0.5 at L9).
- **Other rules:** `decide()` rules 3–7 and Q2 match the doc in order and thresholds.
- **Arithmetic:** 57,600 screen generations. 129 conditions × projected 1,984 rows = 256k. Dev 90 × about 1,004 = 90k. At 114 generations/s that is about 8.4, 13 and 37 min. Small-first is about 2,120 rows × 4 configs. 65–80 min holds inside 4 h. 48 − 90 and 90/4,034 = 2.2% check out.
- **Preflight as run now:** exits 1 with three messages (uncommitted, no passing review line, no reviewed sha). That is the right behaviour before commit. I did not run `submit.sh`.
