Reviewer: fresh subagent (Claude Code)
Agent-ID: a1d5eb0864d781deb
Verdict: PASS
Transcript-SHA256: 1968de1ffc55ca8a47898c6e312c0a0d11bb02ec2cabee6fbf118ba7a1763041

## Brief given

You are a design reviewer for an experiment that has not run yet. This is round 4, and it is narrow. Round 3 (docs/reviews/11-22_mistral7b_true_head_patching_design_v3.md in ~/github/MATS) required one fix: M1a must not turn a refutation into "uninterpretable". Verify that fix and the small changes made with it, and confirm nothing else broke. Do not edit any files.

- Use `git diff a354921 0460eaa` to see the changes to the R15 script and pre-registration. They are:
  - the M1a fix;
  - a try/except around `decisions_before_M7`;
  - new `decide_selftest` asserts covering M1a with a failed sanity transplant, the 5/60 vs 4/60 boundary inside `decide()`, `M5_specific` False, and no-pair M7 with data;
  - pre-reg wording for the sanity gate, M1a, the "even-only rate split by H22", and "classified output".
- Files:
  - Script: 9_8_research/reviewer_experiments/exp_R15_mistral_true_heads/run_experiment.py
  - Pre-registration: docs/experiments/11-22_mistral7b_true_head_patching.md
  - Smoke log: smoke_cpu.log in the R15 folder, run on 0460eaa.
- Round 3's harness is in /private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/r15/. Rerun t3.py against the current script.

Write your review as markdown. Start with a verdict, PASS or FIXES NEEDED, then any required fixes with file:line. Return the full review as your final message; it will be saved verbatim.

## Report

# Design review round 4: R15, true per-head patching on Mistral-7B

**Verdict: PASS.** Nothing needs fixing.

The M1a fix is correct in the code, the pre-registration and the selftest. The try/except, the new asserts and the wording changes do what they claim. No other part of `decide()` or the run order changed. Round 3's full harness gives the same results as before, except for the M1a case, which is now fixed.

## How I checked

- **Commits.** `git diff a354921 0460eaa` touches only the script lines named in the brief, the pre-registration and the round-3 review. HEAD (`08e74dc`) changes only `smoke_cpu.log` on top of `0460eaa`.
- **Smoke log is current.** `sha256(run_experiment.py)` = `1fba89a41bec6887…`, the same at HEAD and at `0460eaa`. That equals the R15 sha in `smoke_cpu.log`, whose header says commit `0460eaa`. Its R14 sha `20819e06…` is unchanged.
  - Both smoke runs print `r15 decide selftest OK`.
  - The forced-pair run had a failed sanity transplant and no correct layer, and printed `M1a_no_layer_repairs_gen: 'uninterpretable: no layer repairs and …'`. That is the intended output.
- **Harness.** I reran `t3.py` against the current script. The system Python has no numpy, so I used the scratch venv `…/scratchpad/r13env/bin/python`. `decide_selftest()` passes. Results:

| World | Result | Same as round 3? |
|---|---|---|
| M1a: sanity fails, no layer correct | `uninterpretable: no layer repairs and the all-layers sanity transplant does not repair either` | yes |
| **M1a: sanity fails, layer 2 correct** | **`False`**, `M1_correct_layers = [2]` | **fixed** (was "uninterpretable") |
| M2 (c): 4/60 vs 0/60 | "H22's role not shown" (p = 0.0594) | yes |
| M2 (c): 5/60 vs 0/60 | "parity holds" (p = 0.0287) | yes |
| H22 and noH22 both succeed often | "parity holds" | yes |
| H22 plus 4 noH22 | "carried by H22" | yes |
| Count-driven worlds | p = 0.16 and 0.75; neither gives a false "carried by H22" | yes |
| All Δ None, parity and layer sweep | "not supported" | yes |
| Δ > 0 only at even k = 16 | "role not shown" | yes |
| M1b: five layers on top, five layers low, only the five layers Δ > 0, 3 of five Δ > 0 | holds / non-specific / holds / not supported | yes |
| No-pair M7, with and without generalization data | `1 of 1 eligible pairs repaired` / `M7: not run` | yes |
| M4 "H22 special" per config | True / False | yes |
| M5: 3/10 random 5-layer sets correct | `M5_specific` False | yes |

## The changes, one by one

1. **M1a fix** (`run_experiment.py:286-288`). The order is right: any correct layer → `False`. Otherwise the result is `True` if the sanity transplant is correct, and "uninterpretable" if not.
   - A correct layer now refutes M1a whatever the sanity result.
   - The null is still gated on the sanity transplant.
   - Pre-reg `:90-92` states the same rule. The sanity gate line `:83-84` now says the gate is not a stop and only decides whether M1a's null is interpretable. That matches the code, which has no sanity stop.
2. **try/except around `decisions_before_M7`** (`:573-577`).
   - An exception from `decide()` is now saved as `{"error": repr(e)}`, and M7 still runs.
   - It catches only analysis errors. The gates and selftests run earlier, outside the try, and still stop the job.
3. **New `decide_selftest` asserts** (`:406-437`). Each one tests what it claims:
   - `m1a("neither", True)` is `False`. This is the case round 3 required.
   - `m1a("neither", False)` is "uninterpretable", and `m1a("correct", False)` is `True`.
   - `M5_specific` is False with 3/3 random 5-layer sets correct.
   - On the no-pair path with data, `M7_report_as` is "1 of 1".
   - The 5/60 vs 4/60 boundary is now tested inside `decide()` at layer 2, giving "parity holds" vs "role not shown". The `c_noH22_vs_odd[0] == 5` check confirms the world is built as intended.
   - `m1a` uses empty `single_heads` and `parity`, and `decide()` handles them without error. That also shows `decide()` copes with partial results, which the try/except is there for.
4. **Pre-reg wording.**
   - The status line now cites rounds 1–3 and links the v3 review.
   - "Every classified output" (`:60`) answers round-3 note 5.
   - The split-by-H22 sentence now reads "every even-only rate is also split by H22 ∈ S" (`:101`).

## Optional notes (none change a verdict or lose data)

1. **One "every output" left in the pre-reg** (`:42`, in Setup): "every output records `hit_max_new` and `says_less`". This is the same imprecision round 3 fixed at `:60`. The gate outputs are compared as strings and never classified. Change it to "every classified output" to match.
2. **The split by H22 is pooled, not per k** (`:338-341`).
   - The code splits the even-only rates into one pooled pair, `even_with_H22` / `even_without_H22`, across all k. The without-H22 subsets also get per-k rates (`even_noH22_k*`).
   - There is no per-k with/without split of the `even` family, so "every even-only rate is also split" slightly overstates what is reported.
   - Each parity row saves its `heads`, so a per-k split can be computed later. No data is lost.
3. **The final `decide()` is not wrapped** (`:597`, after M7). If it raised, `decisions` and `finished` would not be written. All M7 data are already saved, because `save()` runs after every M7 row, and `decisions_before_M7` exists. So the risk is cosmetic.
4. **Stale docstring.** `decide_selftest`'s docstring still says "review rounds 1-2" (`:383`).

## Files

- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R15_mistral_true_heads/run_experiment.py`: M1a fix `:286-288`, rates `:332-341`, selftest `:382-437`, `decisions_before_M7` `:573-577`, final decide `:597`
- `/Users/gus/github/MATS/docs/experiments/11-22_mistral7b_true_head_patching.md`: `:42` (optional wording), `:60`, `:83-84`, `:90-92`, `:101`
- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R15_mistral_true_heads/smoke_cpu.log`: sha matches the current script
- Harness: `/private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/r15/t3.py`. Run it with `/private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/r13env/bin/python`.
