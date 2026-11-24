Reviewer: fresh subagent (Claude Code)
Agent-ID: adb9f63ef3241637f
Verdict: PASS WITH MINOR FIXES
Transcript-SHA256: 366138906ec26eb251780904f56994e0699a6710261b55a049ef2edebfc85af2

## Brief given

You are a fresh, skeptical design reviewer (round 2) for a pre-registered interpretability experiment in repo `~/github/MATS`. Do NOT run GPU compute, do NOT edit repo files, do NOT run git commit/push. Read only; you may run plain `python3` locally (no torch) on saved JSON files.

Read:
- Pre-registration (amended after round 1): `docs/experiments/11-24_llama8b_gpu_margin_offset.md` (R21)
- Round-1 review and the doer's response: `docs/reviews/11-24_llama8b_gpu_margin_offset_design.md`
- Scripts: `9_8_research/reviewer_experiments/exp_R21_gpu_margin/run_experiment.py`, `analyze.py`, `run.sbatch`, `smoke_cpu.log`
- Imported unchanged: `9_8_research/reviewer_experiments/exp_R13_true_head_patching/run_experiment.py`; R13 results in `exp_R13_true_head_patching/results/r13_24014365` (L40S) and `r13_24084222` (A100).

Tasks:
1. For each of the 11 round-1 findings, check that the fix in the code matches what the experiment file says, and that it is correct. Name any finding that was not fixed or was fixed wrongly.
2. Look for new bugs introduced by the amendments: the determinism re-run block in run_experiment.py (does it reuse `measure`/`stash` correctly, does it read the right saved file, could it ever mutate the saved results), the class gate, analyze.py's `score` (permutation control loop, recall at quartiles, density window, which-margin switch, verdict order matching the doc's rules 0–5), and the provenance assertions.
3. Check that every decision rule and reported quantity in the doc has a matching implementation, and vice versa (nothing the code labels that the doc does not pre-register).
4. Is anything still missing that would make the result uninterpretable?

Return a verdict on the first line: PASS, PASS WITH MINOR FIXES, or FIXES NEEDED. Then numbered findings with severity (blocking / should-fix / minor), file:line, and a concrete fix. Be concise; no praise.

## Report

PASS WITH MINOR FIXES

Nothing blocks the run. One reported H0 quantity cannot fail, the permutation loop has a fragile cache, and round-1 finding 9 is still open. All fixes are in `analyze.py` or the docs, except finding 3, which touches `run_experiment.py`; land that before computing `R21_SCRIPT_SHA`.

**Round-1 findings, checked against the code**
- 1: The window is fixed (`w` = median |Δ|). The "expected reverse flips" count added with it is wrong; see finding 1 below.
- 2: Fixed. The switch at `analyze.py:130` matches the doc's rule (c == 0 or more than 20% of Δ exactly 0), and both margins are reported.
- 3a: Fixed, with one gap: the unpatched re-run does not compare text (finding 3).
- 3b: Implemented, but the D-set cache is unsafe (finding 2).
- 4: Fixed. The expected outcome is in the doc, and recall/precision at Q1, median and Q3 are reported. `st.quantiles` uses the exclusive method, and for n = 1307 its q[1] equals the median.
- 5: Fixed. `decide()` follows doc rules 0–5 in order.
- 6: Fixed. There are guards for an empty M, and ties are counted (`D_with_tie`). `st.quantiles` still raises if M has exactly one config; that cannot happen with the real data.
- 7: Fixed, and stricter than the doc says (finding 4).
- 8: Mostly fixed (finding 6).
- 9: **Not done.** `exp_R21_gpu_margin/`, the experiment file and the review are untracked in git, and `docs/decisions.md` and `docs/research_journal.md` have no R21 entry.
- 10, 11: Fixed. The limitations are in the doc, and `max_abs_logit` is recorded.

**Checked and correct**
- `wrapped` takes only keyword arguments. R13's `Patcher.generate` makes exactly one `model.generate(**ids, ...)` call. `capture` calls `model(...)`, not generate, so the stash is never stale.
- `measure` pops the stash, so nothing carries over between calls.
- The determinism block reads the prefill file it just wrote, under `a.out`. Floats survive the JSON round trip exactly. It writes only to `run["determinism"]`; neither the saved cell files nor `run["unpatched"]` are touched.
- R13's sha256 is still `1ce5b0cc…348d`. The unpatched baseline is " 9.11 is bigger…" on both GPUs, so the unpatched run has a decision step.
- I ran `analyze.py` end to end on synthetic run directories built from R13's texts. It runs, and the verdict order and the which-margin switch behave as the doc says.

**Findings**

1. **should-fix**, `analyze.py:103` and doc lines 107–109. `expected_reverse_flips_own_delta` = #{0 < m_L < −Δ_k}. Since m_L + Δ_k = m_A, this counts configs with m_A < 0 < m_L, which are the reverse flips actually observed in R21. The copy and class gates, together with R13 having 0 reverse flips, force it to 0 (it was 0 in my synthetic run too). The "expected vs observed 0" comparison therefore cannot fail. Fix: drop it, or replace it with an expectation under H0. Example: symmetrize Δ (pool Δ and −Δ), then compute Σ over k with m_L(k) > 0 of the fraction of pooled values below −m_L(k), reported next to the observed 0. Update the doc text to match. The permutation's `mean_reverse_flips` is the non-symmetrized version and can stay.

2. **should-fix**, `analyze.py:88, 110–117`. `Dset_cache` is keyed on `id(D)`. D is a local list in `analyze_cell`, and CPython reuses the ids of freed objects. If the all_steps D lands at the prefill D's address, the permutation recall for all_steps silently uses prefill's D. My synthetic run happened to get distinct ids, so this is a hazard I did not observe. Fix: `Dset = set(D)` once at the top of `score` and test `k in Dset`; delete `_DC` and `Dset_cache`.

3. **minor**, `run_experiment.py:192`. The unpatched determinism check compares m and lens but not text, while the doc (line 72) says text, m and lens. Fix: add `and r2["output"] == run["unpatched"]["output"]`. Also say in the doc that only prefill configs are re-run; all_steps is not.

4. **minor**, `run_experiment.py:172–179` vs doc lines 55–56. `gate_class` needs `len(cls_ok) == len(done)`, so any config without a decision step fails the gate and the job exits 2. The doc says such configs are "excluded from M (counted)". Fix: in the doc, state that a missing decision step fails the class gate, so M = 1,307 is required for interpretation.

5. **minor**, process (round-1 finding 9). Commit `exp_R21_gpu_margin/`, the experiment file and the review, and add the R21 decision and journal entries, before taking `R21_GIT_SHA`. Nothing checks that `R21_GIT_SHA` matches the code that ran; `analyze.py` only checks that both runs carry the same value. Also have `run.sbatch` print `sha256sum analyze.py`, or record the expected hash in the doc, because the sbatch does not freeze `analyze.py`.

6. **minor**, `analyze.py:150–168`.
   - Also assert that `r13_script_sha256` equals `1ce5b0cc…348d` and that `token_ids` are equal across the two runs.
   - Copy `same_node_as_r13` and `same_gpu_string_as_r13` into `res["runs"]`. Note that `same_gpu_string_as_r13` (`run_experiment.py:146`) compares only the GPU name, not the UUID.
   - The verdict is printed even when `interpreted` is false. Fix: when gates fail, set each cell's verdict to "not interpreted: gates failed" and keep the raw verdict under another key.

7. **minor**, `analyze.py:79, 162`.
   - If more than 50% of Δ on the m margin are exactly 0, w is 0 and both density counts are empty. That is acceptable, because the lens_final block covers it, but say so in the doc.
   - `unpatched_delta` is computed only on m. When lens_final rules, the doc's "unpatched Δ vs c" compares different margins. Fix: also report the unpatched Δ of `lens[-1]`.

**Doc vs code coverage.** Every rule and reported quantity in the doc has an implementation, and every label in `decide()` is pre-registered. Apart from finding 1, nothing would make the result uninterpretable. The permutation control and the lens profile are descriptive with no thresholds, as the doc says.

**Files**
- `~/github/MATS/9_8_research/reviewer_experiments/exp_R21_gpu_margin/analyze.py`
- `~/github/MATS/9_8_research/reviewer_experiments/exp_R21_gpu_margin/run_experiment.py`
- `~/github/MATS/docs/experiments/11-24_llama8b_gpu_margin_offset.md`

I made no repo edits and ran no git writes or GPU jobs. The synthetic test data and scripts are in `/private/tmp/claude-502/-Users-gus-Documents-obsidian-global/b1c8d5ca-2b8b-4ac2-b532-97d0357c961a/scratchpad/`.

## Doer response

All seven findings applied before any compute:
1. The tautological count is replaced by an H0 expectation: pooled ±Δ, summed share below −m_L40S over L40S-correct configs, reported next to the observed reverse flips.
2. `Dset_cache` removed; `Dset = set(D)` inside `score`.
3. The unpatched determinism check now compares text too; the doc says only prefill configs are re-run.
4. The doc now says a missing decision step fails the class gate, so M = 1,307 is required.
5. Code, experiment file and reviews committed with the decision and journal entries before taking R21_GIT_SHA; limitation on freezing analyze.py stated.
6. analyze.py asserts R13's sha256 and equal token ids; copies node and GPU-name match flags; replaces verdicts with "not interpreted: gates failed" when gates fail.
7. w = 0 case documented; unpatched Δ also on lens_final.
Re-tested on the synthetic offset and noise datasets: verdicts unchanged.
