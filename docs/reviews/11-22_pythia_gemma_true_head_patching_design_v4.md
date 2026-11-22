Reviewer: fresh subagent (Claude Code)
Agent-ID: a8dde408d1e77cd2a
Verdict: FIXES NEEDED
Transcript-SHA256: 366caadaee74dad0df77da6062947b392378c6c44bb844b6b80de57d7da1ad52

## Brief given

You are a design reviewer for an experiment that has not run yet. This is round 4. Rounds 1–3 required fixes, and the design and script have since been revised. Check whether the round-3 fixes are correct and complete, and look for remaining problems. Be adversarial but proportionate: flag only issues that could change a verdict, lose data, or make the pre-registration disagree with the code. Do not edit any files.

Repo: ~/github/MATS.
- Prior reviews, in order: docs/reviews/11-22_pythia_gemma_true_head_patching_design.md, _design_v2.md, _design_v3.md. Read round 3 carefully.
- Pre-registration: docs/experiments/11-22_pythia_gemma_true_head_patching.md
- Script: 9_8_research/reviewer_experiments/exp_R14_cross_model_true_heads/run_experiment.py. Run `python -I run_experiment.py --selftest`; it needs only the standard library and takes about 5 s.
- Job file: run.sbatch, in the same folder. It now uses `a100_tandon`.
- Smoke log: smoke_cpu.log, in the same folder, run on commit d330a52.
- Round 3's simulation and mock scripts are in /private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/r14v3/. Reuse them.

Check:
1. Round 3's required fixes:
   - Fix 1: the "no split scores higher" clause in (d), plus the noise selftest. Rerun your noise simulation on the new rules, including the graded truths.
   - Fix 1b: verdict branches 5–6. Does a noisy true-parity truth now avoid any claim against parity?
   - Fix 2: the late-training caveat, the error and gate reasons, and `matches_main_run_reason` in each branch.
   - Fix 3: branch order and the truth count in the text.
2. The adopted optional items:
   - partial checkpoint records filled in place;
   - the guarded main-run load;
   - dumping configs before summarize;
   - `screen_matches_main_run`;
   - the Gemma verdict field.
   Re-drive your mock dynamics scenarios.
3. Any remaining mismatch between the pre-registration text and the code.

Write your review as markdown. Start with a verdict, PASS or FIXES NEEDED, then required fixes with file:line, then optional suggestions. Return the full review as your final message; it will be saved verbatim.

## Report

# R14 design review, round 4: true per-head patching in Pythia-160M and Gemma-2B

**Verdict: FIXES NEEDED** (two small fixes. Everything else from round 3 is in and correct.)

`python -I run_experiment.py --selftest` passes and prints 18 classifier cases, 8 synthetic truths, the noise test ({6,8,10}: 1/40; ≥3 even: 40/40) and 9 late-training cases. The script's sha256 (`04aa7ead…`) is the same as the one in `smoke_cpu.log`. The script has not changed since `d330a52`, and HEAD `8c08fe0` only adds R15 files.

Beyond the selftest:
- I reran the round-3 noise simulation against the new verdict wording. It covers 12 truths × p ∈ {0, 0.01, 0.03, 0.10} × 40 seeds, plus 7 graded truths and 2 graded truths with an extra head-8 bias, 60 seeds each.
- I drove `dynamics()` end to end with the mock `Patcher` and stub `torch`, in 9 scenarios: normal, mismatch, gate failure at 143k, crash during 143k subsets, missing `run.json`, `run.json` without a screen, screen-only mismatch, model-load crash at 143k, and a main run without complete configs.

Scripts: `/private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/r14v4/` (`sim.py`, `sim2.py`, `mock.py`, `lt.py`).

## Required fixes

**1. The late-training verdict can say "supported" or "refuted" when the all-heads patch does not repair at step143000.**
Where: `run_experiment.py:655-660` (`late_training_verdict`), with `:261-262`. Pre-reg `:124-125` and `:154-161`.

- **What goes wrong.** `rules()` computes `parity_holds` from (a)–(d) only. It does not check `repair`. `late_training_verdict` then reads `c143["rules"]["parity_holds"]` and never checks `c143["rules"]["repair"]`. Step120000 does get a repair check (`:667`).
- **Example, from `lt.py`.** An even-majority truth gives all-heads 6/6, so the patch is not correct and the subset rows are not correct. Under that truth:
  - The step143000 `verdict` is "repair fails at this layer; head rules not interpreted".
  - But `parity_holds` is True, and `late_training_verdict` returns **"supported: parity holds at step143000, not at step120000"** whenever step120000 repairs without parity.
- **Why it matters.** This contradicts the pre-registered "Layer-L repair fails → head rules reported, not interpreted". It also errs in the paper's favour. The case is plausible: in the CPU smoke, the all-heads patch at L6 did not repair (layer sweep correct at [0, 1, 2, 3, 7]). Dynamics always uses L6, with no fallback layer.
- **Fix.**
  - In `late_training_verdict`, right after the `ok(c143)` check and before the parity check, add: `if not c143["rules"]["repair"]: return "untestable: all-heads patch does not repair at step143000; head rules not interpreted" + caveat`.
  - Add this case to the pre-reg's late-training list (`:156`) and add a selftest case: `ck(True, False, True)` vs `ck(True, True, False)` must start with "untestable".
- **Same gap in Gemma.** `gemma_parity_independent` (`:264`) is `all_even and all_odd`, with no repair condition. Either AND it with `r["repair"]`, or say in the pre-reg (`:152`) that it is read only when repair holds.
- **Stored field.** Optionally, store `parity_holds` as False, or add `"parity_interpreted": repair and parity_holds`, so that a write-up quoting the field cannot contradict the verdict.

**2. A mismatch in the screen outputs does not add the "not interpreted" caveat, though the pre-reg text says it should.**
Where: `run_experiment.py:657-658`, `:726-727`. Pre-reg `:162-164`.

- The pre-reg says: "If step143000's outputs differ from the main run, every verdict carries '(not interpreted…)'. The six screen outputs are also compared … as `screen_matches_main_run`." The six screen outputs are step143000 outputs, so a reader expects a screen mismatch to add the caveat.
- The code adds it only when `matches_main_run is False`.
- Mock `screendiff`: `screen_matches_main_run: False`, and the verdict is "refuted: parity already holds at step120000" with no caveat.
- This matters most when the subset check is skipped. With `main_cfgs` empty (mock `nomaincfg`), the screen comparison is the only evidence of a weights or numerics difference, and today it is ignored.
- Fix in code (preferred): add the caveat when `c143.get("matches_main_run") is False or c143.get("screen_matches_main_run") is False`, plus a selftest case. Or fix the text instead: say the caveat comes only from the 4,095-config comparison and the screen comparison is descriptive.

## Round-3 fixes: status

| Fix | Status | Evidence |
|---|---|---|
| 1. "No split scores higher" in (d), plus noise selftest | **Fixed** | Code at `:258-260`. Selftest at `:845-858`; its thresholds (≤ 2/40, ≥ 36/40) match pre-reg `:150-151`. My rerun at p = 0.01/0.03/0.10 gave "parity holds" (`holds`/`holds+nec`): {6,8,10} 1/1/0 of 40; 8 alone 1/0/0; {0,2} 1/0/0. True parity: ≥3 even 39/37/27; ≥2 even 39/35/18; any even 39/37/28. Graded 70/10/40: 1 of 23 repairing seeds pass, as in Limitations `:176`. Residual: **"2 of {6,8,10}" still passes 4/2/2 of 40** (optional 1). |
| 1b. Branches 5–6 | **Fixed** | Branch 5 needs the necessary heads to be a non-empty proper subset of the even set (`:284`). **Branch 5 fired in 0 runs out of 4,056** across all noisy truths and all graded truths, including the two with a head-8 bias. Noisy true parity lands in "holds", "not uniquely the most specific split" (descriptive), "inconclusive", or "not supported" (one-output (a)–(c) failures, already a known limitation at `:178`). No claim against parity is made. Noiseless head 8 alone and {0,2} still give branch 5, as intended. |
| 2. Late-training caveat, reasons, `matches_main_run_reason` | **Fixed, except required fixes 1–2** | Mismatch caveat on every branch past the first (mock `mismatch`: "...(not interpreted: step143000 differs from the main run; INBOX)"). "errored or did not run" at `:661`. "gates failed" at `:665-666`. Pre-reg lists both (`:157-159`). Reasons are set per branch: "main run has no pair", "gates failed", "no phenomenon at this checkpoint", "main run has no complete right-aligned results", "compared 4095 configs", and "not reached" after a mid-run crash. All were confirmed in the mocks. |
| 3. Branch order and truth count | **Fixed** | Pre-reg `:132-146` lists branches 1–6 in the code's order (`:275-288`). The qualifiers match (`!= even_set` for branch 2, `< even_set` for branch 5). "Eight synthetic truths" matches the counter and the printout. The 9 late-training cases match the printout. |

## Adopted optional items: status

| Item | Status | Mock evidence |
|---|---|---|
| Partial checkpoint records filled in place | Works (`:700-706`) | `crash`: the 143k record keeps screen, old protocol, gates, `true_heads` and the error. `subsets_step143000.json` holds 2,500 configs, partial. 120k and 1k run in full. |
| Guarded main-run load | Works (`:679-686`) | `nomain`: prints "main run unreadable"; screens and old protocol run at every checkpoint; reason "main run has no pair"; verdict "untestable". |
| Configs dumped before summarize | Works (`:556`, `:762`) | Read; summary dump follows. |
| `screen_matches_main_run` | Recorded at 143k, None when main has no screen (`noscreen`) | `screendiff` gives False, but see required fix 2. |
| Gemma verdict field | Renamed to `pythia_style_parity_descriptive` (`:263-265`), matching pre-reg `:147` | — |

`loadcrash`: a model-load failure at 143k leaves `{"error": ...}` only, with no `matches_main_run`/reason. The error string serves as the reason, so this is acceptable (optional 3).

## Optional suggestions

1. **Residual false-pass rate in Limitations.** "Any 2 of {6,8,10}" still gives "parity holds" in 4/40 seeds at 1% flips (10%), 2/40 at 3%, and 2/40 at 10%. Round 3 gave these numbers for option (b), so this is the cost of the adopted design, not a regression. The pre-reg mentions only {6,8,10}. One line in Limitations would make the residual risk explicit.
2. **Necessary-head qualifier under noise.** For "8 + any other even head" at 1% flips, 30/40 seeds give bare "parity holds", without "with heads [8] necessary": one stray minimal set empties the intersection. This was flagged descriptively in round 3 and has not changed. Reporting heads present in ≥ 90% of minimal sets would keep that information.
3. **Load-failure record.** Initialise `r["matches_main_run"], r["matches_main_run_reason"] = None, "checkpoint errored"` in `dynamics()` before calling `dynamics_checkpoint`, so the field is present even when `Patcher` fails to load. This is likely if a `stepN` revision is missing from the offline cache; round 1's preflight suggestion is still open.
4. **Selftest scope.** The noise test counts `parity_holds`, not the verdict, so a seed that flips the all-heads output counts as a "pass". That is conservative for the false-pass bound and harmless. Required fix 1 is the same gap with real consequences.
5. **Skipped consistency check.** When `matches_main_run` is None at 143k, for example because the main run's right cell is incomplete, the late verdict is interpreted with no consistency evidence. Consider adding "(consistency check skipped: <reason>)" to the verdict.

## Pre-registration vs code: checked and consistent

These match:
- rules (a)–(d), including the new clause;
- the 6-branch order and wording, the necessary-head qualifiers, and the Gemma descriptive field;
- the selftest truths and noise thresholds;
- late-training cases (except required fixes 1–2);
- dynamics order, try/except and in-place records;
- the guarded load, run conditions, comparison of all 4,095 configs, and the per-branch reasons;
- `run.sbatch`: `a100_tandon`, the same account as R13's A100 replication (job 24084222), `python -I`, a directory for `--main-run`, and `|| echo` continuation.

The smoke log matches the final script's sha256.

Files:
- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R14_cross_model_true_heads/run_experiment.py`
- `/Users/gus/github/MATS/docs/experiments/11-22_pythia_gemma_true_head_patching.md`
- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R14_cross_model_true_heads/run.sbatch`
- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R14_cross_model_true_heads/smoke_cpu.log`
- Simulation and mock scripts: `/private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/r14v4/`
