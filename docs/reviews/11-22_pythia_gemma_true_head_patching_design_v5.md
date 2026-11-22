Reviewer: fresh subagent (Claude Code)
Agent-ID: adbaeb3d1000114a6
Verdict: PASS
Transcript-SHA256: ca58355eeb5837d816e3bc0c6d3f3efacfa12d878c24330137c802cc103e4c16

## Brief given

You are a design reviewer for an experiment that has not run yet. This is round 5, and it is narrow. Round 4 (docs/reviews/11-22_pythia_gemma_true_head_patching_design_v4.md) required two small fixes. Verify that they are correct, and that nothing else in the design changed in a way that breaks it. Do not edit any files.

Repo: ~/github/MATS.
- Pre-registration: docs/experiments/11-22_pythia_gemma_true_head_patching.md
- Script: 9_8_research/reviewer_experiments/exp_R14_cross_model_true_heads/run_experiment.py
  - `git diff d330a52 d51b0ed -- <script> <pre-reg>` shows the round-4 changes.
  - Run `python -I run_experiment.py --selftest`; it needs only the standard library.
- Smoke log: smoke_cpu.log, in the same folder. It was run on commit d51b0ed.
- Round 4's mock and simulation scripts are in /private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/r14v4/. Rerun the relevant mocks, especially lt.py, screendiff and loadcrash.

Check:
1. `late_training_verdict`: does it now return "untestable" when step143000's all-heads patch does not repair? Does the screen-mismatch caveat fire? Does the order of the checks still match the pre-registration list exactly?
2. `gemma_parity_independent` now requires repair. Is `parity_interpreted` correct?
3. The load-failure record now has `matches_main_run` and a reason.
4. Does the pre-registration text match the code everywhere?

Write your review as markdown. Start with a verdict, PASS or FIXES NEEDED, then any required fixes with file:line, then optional notes. Return the full review as your final message; it will be saved verbatim.

## Report

# R14 design review, round 5: true per-head patching in Pythia-160M and Gemma-2B

**Verdict: PASS.** Both round-4 fixes are in and correct. Round 4's optional load-failure item is also in. Nothing else in the design changed.

## What I checked

- **Scope of change.** `git diff d330a52 d51b0ed` touches only:
  - `late_training_verdict` (`run_experiment.py:658-661`)
  - `rules()` (`:263`, `:265`)
  - the initialisation of the `dynamics()` record (`:703`)
  - four selftest asserts and the case count (`:875-883`)
  - the pre-reg status line, Gemma rule (`:153`), late-training list (`:158`, `:164-166`) and Limitations (`:180-183`)
  
  `d51b0ed..HEAD` changes only `smoke_cpu.log` for R14. `run.sbatch` is unchanged, and the working tree is clean for these files.
- **Selftest.** `python -I run_experiment.py --selftest` prints: "selftest OK: classifier (18 cases), rules on 8 synthetic truths, noise test ({6,8,10}: 1/40 false passes; >=3 even: 40/40), late-training verdict (11 cases)".
- **Smoke log.** The script's sha256 is `20819e06…`, the same as the header of `smoke_cpu.log`, which was run on `d51b0ed`. All four smoke commands and the negative `--main-run` test behave as in round 4. The dynamics smoke ends "untestable: no interpretable full-subset run at step143000", which is expected under `--limit`.
- **Mocks rerun on the current script:** `lt.py`, and `mock.py` in all 9 scenarios (normal, mismatch, gatefail, crash, nomain, noscreen, screendiff, loadcrash, nomaincfg). I added a few extra cases in `/private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/r14v5_g.py`.

## 1. `late_training_verdict` (`run_experiment.py:651-675`)

- **No repair at step143000 now returns "untestable".**
  - `lt.py` case: the even-majority truth has `repair=False` and `parity_holds=True`. Both late-training calls now return "untestable: all-heads patch does not repair at step143000; head rules not interpreted". Round 4 returned "supported" here.
  - The check (`:660-661`) comes after `ok(c143)` and before the parity check, as required.
  - The new selftest case at `:875` covers it.
- **A screen mismatch now adds the caveat.**
  - The caveat condition at `:658-659` now includes `screen_matches_main_run is False`.
  - Mock `screendiff` now gives "refuted: parity already holds at step120000 (not interpreted: step143000 differs from the main run; INBOX)". In round 4 it had no caveat.
  - The caveat also attaches to the new no-repair branch. I checked this for a screen mismatch and for a subset mismatch.
  - `None`, meaning the check was skipped, adds no caveat (mock `noscreen` and `nomaincfg`). That matches the pre-reg's "differ" wording.
- **Order matches the pre-reg list (`:157-163`) exactly:**
  1. not ok
  2. no repair at 143k
  3. parity fails at 143k
  4. 120k errored
  5. 120k has no pair
  6. 120k gates failed
  7. 120k does not repair
  8. 120k parity holds → "refuted"
  9. otherwise → "supported"
  
  The case count in the selftest output (11) matches the asserts.
- **Other mocks are unchanged from round 4:**
  - `mismatch`: "not supported … (not interpreted …)"
  - `gatefail`, `crash`, `nomain`, `loadcrash`: "untestable: no interpretable full-subset run at step143000"
  - `normal`, `noscreen`, `nomaincfg`: "refuted"

## 2. Gemma parity-independence and `parity_interpreted` (`run_experiment.py:263-266`)

- `gemma_parity_independent = repair and all_even and all_odd`.
  - A Gemma truth where exactly 4 heads repair (`len(h)==4`) has even-4 and odd-4 correct, but all 8 heads fail. It now gives `gemma_parity_independent=False`.
  - The `len(h)>=4` truth still gives True, and the existing selftest assert at `:841` still holds.
- `parity_interpreted = repair and parity_holds` is correct:
  - The even-majority Pythia truth gives `parity_holds=True`, `parity_interpreted=False`, verdict "repair fails…".
  - The ≥3-even truth gives True/True, verdict "parity holds".
  - The field is set for both families. It agrees with `parity_verdict` branches 1–2 by construction.

## 3. Load-failure record (`run_experiment.py:703`)

- Mock `loadcrash`: the step143000 record is `{"matches_main_run": None, "matches_main_run_reason": "checkpoint errored", "error": "OSError: no revision cached"}`.
- On a normal load, `dynamics_checkpoint` overwrites these with "not step143000" or "not reached" at `:724-725`, before any other work. So the per-branch reasons in all other mocks are unchanged:
  - "compared 4095 configs"
  - "gates failed"
  - "not reached" after the mid-run crash
  - "main run has no pair"
  - "main run has no complete right-aligned results"

## 4. Pre-registration vs code

These now match:
- Gemma rule (`:153`) and `:264`
- late-training list (`:157-163`) and `:657-675`, including the new no-repair branch
- caveat sources (`:164-166`) and `:658-659`
- Limitations on `parity_interpreted` (`:182-183`) and `:263`
- residual false-positive numbers (`:180-181`), which match the round-4 simulation
- status line, which now lists rounds 1–4

I found no new mismatches.

## Optional notes (none block the run)

1. **"Every verdict carries" the caveat (pre-reg `:165`).** The first branch, "untestable: no interpretable full-subset run", does not carry it (`:656-657`). That branch claims nothing, so this is harmless. "Every verdict past 'no interpretable run'" would be exact.
2. **"Always compared" for the screen (pre-reg `:165`).** The screen comparison is `None` when the main run has no screen, and absent when the checkpoint fails to load. "Compared whenever both screens exist" would be exact.
3. **Gemma selftest gap.** No selftest case isolates the new Gemma repair condition. The existing no-repair truth (`:843-844`) also has `all_even` false. Adding `verdict(8, lambda h: len(h) == 4, "gemma")` and asserting `not r["gemma_parity_independent"]` would pin it down. I checked by hand that the current code gives False.
4. **Skipped consistency check (round-4 optional 5, still open).** When `matches_main_run` and `screen_matches_main_run` are both `None`, the late verdict carries no qualifier. The reason is recorded in the checkpoint record, so this is a reporting choice only.

## Files

- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R14_cross_model_true_heads/run_experiment.py`
- `/Users/gus/github/MATS/docs/experiments/11-22_pythia_gemma_true_head_patching.md`
- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R14_cross_model_true_heads/smoke_cpu.log`
- Mocks: `/private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/r14v4/` and `/private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/r14v5_g.py`
