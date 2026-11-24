Reviewer: fresh subagent (Claude Code)
Agent-ID: ae683b8e0ea73c9f7
Verdict: PASS WITH MINOR FIXES
Transcript-SHA256: 67a0c39be7f78b259b8c7345ec6ace3b57bb58288402a7c80db6719ed55e6318

## Brief given

You are a fresh, skeptical design reviewer for a pre-registered interpretability experiment (repo `~/github/MATS`). Do NOT run GPU compute or edit files; read only (you may run `python3` locally for quick checks, e.g. on `9_8_research/reviewer_experiments/exp_R19_last_fraction/results/r19_24227404/run.json`).

Read:
- Pre-registration: `docs/experiments/11-24_llama8b_steering_fix.md` (R20)
- Script: `9_8_research/reviewer_experiments/exp_R20_steering_fix/run_experiment.py` and `run.sbatch` in that dir
- Context it builds on: `docs/experiments/11-23_llama8b_last_fraction.md` (R19 results), R19 script `9_8_research/reviewer_experiments/exp_R19_last_fraction/run_experiment.py`, R18 helpers (`exp_R18_eleven_token/run_experiment.py`: classify_pair, load_r17) and R17's `Model` (`exp_R17_localization_map/run_experiment.py`, `_register`, resid pre-hook).
- CPU smoke output was fine (tiny random model; mechanics only).

Question: is this design sound enough to run, and will its decision rule answer "can a fixed steering vector fix Llama-3.1-8B's 9.8 vs 9.11 error on held-out questions without breaking correct answers"? Look especially for:
1. Correctness bugs in the code (hook positions under left padding, position_ids in the logits-only forward vs generate, prefill-only steering, direction construction, train/dev/test leakage, classification/transition counting, the sign test, random controls, tie-breaking, smoke-mode branches that could leak into the real run).
2. Whether the decision rule can mislabel (e.g., a format swap scoring as a fix; thresholds; multiple comparisons across 3 directions; is the random control matched well).
3. Whether baseline batching reproduces R19 (gate 4) and whether batched steered generation is trustworthy.
4. Missing controls or reported quantities a skeptical reader would demand; anything that's over-scoped.
5. Runtime risk within 4 h on one A100 (8B fp16).

Return a verdict on the first line: PASS, PASS WITH MINOR FIXES, or FIXES NEEDED. Then numbered findings, each with severity (blocking / should-fix / minor), file:line, and a concrete fix. Be concise and concrete; no praise.

## Report

PASS WITH MINOR FIXES

I found no blocking bugs. I checked the left-pad position mapping, the prefill-only hook, the logits-forward position_ids, the train-only directions, the selection tie-breaks, the sign test, the transition counting and the smoke-mode branches. Checks I ran locally:
- `split_xs()` returns train [1,3,4,5,7,10,11,12], dev [2,13,15,16], test [6,8,9,14,17,18,19,20]. These match the pre-registration.
- R19's `run.json` gives per-split class counts that match the doc (train fd 74 / both-bug 559, dev 45 / 99, test 177 / 563).
- R19's `run.json` is git-tracked, so gate 4 can find it on the cluster.
- The model env chain R19→R18→R17 resolves to the pinned snapshot.
- Query order and batch boundaries match R19's screen, so gate 4 should pass at about 100%.
- R19's screen took 168 s for 300 batches. R20 needs about 1,260 generation batches plus about 7,200 proxy forwards, which is well under 1 h. There is no walltime risk.

Findings:

1. **should-fix — `run.sbatch:27`.** Results go to `${SCRATCH}/MATS/results/r19_${SLURM_JOB_ID}`. That is the R19 prefix, so R20 results will be filed as R19. Fix: change it to `r20_${SLURM_JOB_ID}`.

2. **should-fix — `run_experiment.py:458` and `:433`.** The fix/break baseline is the class from the 9,600-question batching (R19's batch boundaries). Steered runs use different batches: test and dev are subsets, and 160-question X-blocks don't align with 64-row batches. Some fixes and breaks can then come from fp16 batch-composition noise rather than from steering. That noise feeds straight into the break rate and the sign test. Fix: run one unsteered `batched(M, test, "q")` and one `batched(M, dev, "q")` (about 80 batches, under 1 min) and use those as `tbase` / `dev_base`. Report how many disagree with the 9,600-batch classes. Keep `cat` from the full baseline for stratification. Do the same for `sbase` in `primary_on_simple` (`:503`).

3. **should-fix — random control is weak (`:467-473`, decision rule step 2).**
   - Isotropic Gaussian directions of matched norm are a soft null. Difference-of-means vectors concentrate in a few high-variance or outlier residual dimensions; random ones spread over 4,096. At α=4–8, random vectors usually either do nothing or break things (negative nets), so "beats the best of 3 random" is close to guaranteed.
   - Fix (keep the 3 random directions and add): a matched mean-difference null at the same config. For cb, shuffle the correct/bug labels within order on train. For fd, use a random train subset of the same size (74), giving simple − Q&A. Rescale to ‖v‖.
   - Also report the full random fix and break counts, not just the net, and the −v (sign-flipped) result.
   - If you add the shuffled-label null to the rule, add it to `decide()` and its selftest.

4. **should-fix — the decision rule can't tell steering apart from just dropping "Q:" (`:508-510`).**
   - I computed a pure format swap on test from R19's classes: F=179, B=35, net=144, break rate 1.1%, fd fix rate ≈1.0. That passes steps 1–3 and is labelled "fixes the format flip only". The label is honest, but the prompt-swap comparator is report-only.
   - Fix: pre-register a qualifier on the headline, "(≤ prompt swap)" when steered net ≤ prompt-swap net.
   - Compute the format-swap index on the discordant set only: fd + reverse-dependent, n≈212 in test. The current `agree_with_simple_base` (`:494`) is over all 3,840 questions and dominated by both-correct agreement (~80%), so it can't move much.

5. **should-fix — dev has no big-first format-dependent questions.**
   - From R19's classes, dev fd is 45, all small-first. Train fd is 14 big-first and 60 small-first. Test fd is 27 big-first and 150 small-first. Dev reverse-dependent is 6.
   - So dev selection is driven by small-first and both-bug questions, which is where R19 found the LFT does *not* carry the flip (12%). Dev also barely penalizes format-swap breaks.
   - This is not a bug, but the doc's motivation ("R19 found the LFT…") does not match what selection can see. Fix: state this in the pre-registration. Add order × category fixes and breaks to the test report (`:474-485` only has the marginals), so the big-first fd result (n=27) is visible.

6. **should-fix — no texts saved (`:302-306`, `:463-464`, `:470-471`).** Only classes are stored. R19's verification re-classified the raw texts with an independent classifier; R20 can't be audited the same way. Steered outputs are also likely to be odd (echoed questions, a new "Q:"). Fix: save the baseline texts (both formats) and the steered test texts for each direction's config. Random-control texts can be limited to rows that changed.

7. **minor — `:516-524`.** The steered batched-vs-single check samples 32 random test questions, about 80% of which are both-correct and unchanged, so it says little about the transitions that matter. Fix: sample about 32 from rows whose class changed under the primary vector (fixes and breaks), plus 32 random.

8. **minor — the 5% break threshold is lenient (`MAX_BREAK`, `:59`).** It allows about 155 breaks out of 3,098 base-correct, about the same as the total fixable fd (177), while the free prompt swap breaks 1.1%. Fix: report a Wilson CI on the break rate and per-category break rates in the headline. Optionally tighten the threshold to 0.02 before the run, not after.

9. **minor — the tie-break for dev-generation net is unstated.** At `:447`, `rec["net"] > best["net"]` keeps the earlier proxy-ranked config on ties. Fix: say this in the doc's dev-selection step 2.

10. **minor — hook mechanics: correct, no fix needed.**
    - Logits mode uses `cumsum-1` clamped at 0 for pad positions; HF generate fills them with 1. Pads are masked and only the last position is read, so this is fine.
    - The steer-position assert (`:187-188`) and the prefill/decode length asserts are correct.
    - Smoke-only branches (`:256-257`, `:275-277`, `:316-319`, `stop_if_failed`) are all gated on `a.smoke`. `run.sbatch` does not pass `--smoke`.

11. **minor — process.** `exp_R20_steering_fix/` and the doc are untracked. Commit them before computing `R20_SCRIPT_SHA`. After fixes 1–6, the script sha256 will change.

Over-scope: none significant. The 360-config proxy costs roughly 10–15 min.
