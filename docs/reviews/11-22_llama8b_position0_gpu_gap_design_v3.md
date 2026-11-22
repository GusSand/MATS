Reviewer: fresh subagent (Claude Code)
Agent-ID: aba5459b6513799ce
Verdict: FIXES NEEDED
Transcript-SHA256: 80280d185a01d99a52b7d43c8f8a91201efe8625ef3749679ce60c242a578a62

## Brief given

You are a design reviewer for an experiment that has not run yet. This is round 3. Round 2 (docs/reviews/11-22_llama8b_position0_gpu_gap_design_v2.md in ~/github/MATS) required six fixes. Verify them and look for remaining problems. Flag only issues that could change a verdict, lose data, or make the pre-registration disagree with the code. Do not edit any files.

Files:
- Pre-registration: docs/experiments/11-22_llama8b_position0_gpu_gap.md
- Script: 9_8_research/reviewer_experiments/exp_R16_position0/run_experiment.py
- Unit test: test_decide_placebo.py. Run it as `python -I test_decide_placebo.py run_experiment.py`; it needs torch. Round 2's test scripts and venv are in /private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/ (t1.py, t2.py, t3.py). Rerun them against the new code.
- Job files: run_a100.sbatch and run_l40s.sbatch. Smoke log: smoke_cpu.log. All are in the R16 folder.

Check:
1. Round 2's fixes, one by one:
   - placebo built on each GPU's copy donor, mirroring the variant's steps, with exponent matching and exactness;
   - minimum exposed D and the coverage condition;
   - the new labels and integer comparisons;
   - donorA100 and a100clean0 made descriptive, with r on all D;
   - L40S d0 = 0 and inject exactness as gates;
   - the D′ placebo.
2. Test `decide` on further synthetic cases: ties at each threshold, D partly exposed, a large N_new, and the order of the branches against the pre-registration's ordered list.
3. `make_placebo`:
   - What if no coordinate with the same exponent is free?
   - What if d0 has many nonzero entries in one block?
   - What about negative steps at small values?
   - Does the placebo ever touch a d0 coordinate?
4. Any new mismatch between the pre-registration and the code, and any data-loss path.

Write your review as markdown. Start with a verdict, PASS or FIXES NEEDED, then required fixes with file:line, then optional notes. Return the full review as your final message; it will be saved verbatim.

## Report

# R16 design review, round 3

## Verdict: FIXES NEEDED

All six of round 2's fixes are in the code, and the branch order of `decide` matches the pre-registration's ordered list exactly. Three problems remain, and each can change a verdict or stop the run:

- `make_placebo` crashes if any 128-dim block has more than 64 nonzero d0 entries.
- `make_placebo` does not enforce the exactness that the pre-registration promises. A move can silently come out as zero, which makes the placebo weaker and "explains" easier to reach.
- Rule 5 (the 0.2 < r < 0.8 band) ignores both the placebo and N_new.

None of the fixes needs new cells or more compute.

**What I checked.** Current `run_experiment.py` has sha256 `64100caf…`, the same as in `smoke_cpu.log`. I used torch 2.15 on CPU (`r13env`) and ran:
- `python -I test_decide_placebo.py run_experiment.py`: passes.
- Round 2's `t1.py` (`ulp_step`): still matches `np.nextafter` everywhere.
- Round 2's `t2.py` and `t3.py`: both now fail. The causes are API changes (`moves` entries are dicts, and `placebo_r` became `placebo_moved`), not bugs. I wrote adapted versions:
  - `t4_placebo.py` covers the placebo edge cases;
  - `t5_decide.py` covers 26 synthetic `decide` cases.

Both are in `/private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/`.

---

## Status of round 2's fixes

| Round 2 fix | Status | Evidence |
|---|---|---|
| 1. Placebo built on each GPU's copy donor, mirroring the variant, with exponent matching | **Done, but exactness is not guaranteed** | `:240` calls `make_placebo(clean0_A, -d0)` and `:262` calls `make_placebo(clean0_L, +dA)`. In t4, A100 placebo − copy is nonzero only at `[950, 1060]`, with d0 at `[1000, 1100]`: the same blocks, the same size and the opposite sign. It never touches a d0 coordinate. Exponent matching works in the common case, but see required fixes 1 and 2. |
| 2. Minimum exposed D and the coverage condition | **Done** | `MIN_EXPOSED = 10` (`:100`, `:128`). The coverage test at `:135` is `D − n == 0 or 5·moved ≥ 4·\|D\|`. With 10 of 40 D exposed and all moving, the label is "explains the exposed part (10/40)". With 32 of 40, it is "explains the gap". With 31 of 40, it is "exposed part". With 9, it is "descriptive only". |
| 3. New labels and integer comparisons | **Done** | `:130`, `:133` and `:139` use only integers. The ties give the expected labels: 8/10 and 32/40 count as ≥ 0.8; 2/10 and 8/40 count as ≤ 0.2; 2(m − p) = n counts as distinguishable. N_new = 4 gives "explains" and N_new = 5 gives "perturbs broadly". |
| 4. donorA100 and a100clean0 descriptive, with r on all D | **Done** | `:306-309` pass `label=False`, so `De = list(D)` and the verdict is "descriptive". |
| 5. L40S d0 = 0 and inject exactness as gates | **Done** | Both are appended to `gates_ok` at `:255-256`. See optional note A about `interpreted`. |
| 6. D′ placebo | **Done** | `:301` computes `pr` on the same `Dset`, inside the D/D′ loop. |

---

## Required fixes

### 1. `make_placebo` crashes when a block has more than 64 nonzero d0 entries, and run.json is lost
**Where:** `run_experiment.py:83-86`, called at `:240` (A100) and `:262` (L40S).

**The problem.** `taken` starts with every d0 coordinate and grows by one target per source. A block with k nonzero entries therefore needs 2k ≤ 128. In t4:
- k = 40 and k = 64 run;
- k = 65 and k = 128 raise `IndexError: list index out of range` at `(same or cands)[0]`.

R13 recorded only the maximum of d0 (`bos_max_abs_diff_clean_vs_buggy.post_wo = 6.1e-5`), not how many entries are nonzero. A kernel tiling change between 17 and 19 tokens could easily make d0 dense.

**Consequences on A100:**
- The crash happens after `position0.pt` is saved (`:234`) but before the first `dump(run, path)` (`:272`).
- `capture_vs_live`, `pos0_n_nonzero` and the block list are lost.
- The job exits 1, so the L40S job is blocked.
- No verdict exists, and the run is not described anywhere.

**Fix.**
- Move a `dump(run, path)` to before the placebo is built.
- Pre-register what happens when a block cannot hold a matched placebo. Two options:
  - mark the placebo "not constructible" and make rule 3 output "not distinguishable (no placebo)";
  - allow fewer moves and gate on the count.
- Have the code follow that rule instead of crashing.

### 2. Placebo exactness is stated in the pre-registration but not enforced, and inexact moves can be zero
**Where:** `run_experiment.py:85-92`; pre-registration `:55-56` ("so the step is the same size and exact"); the gates list (`:73-87`) has no placebo item.

**The problem.** The code records `exact` and `achieved` but nothing uses them. t4 found three ways to get an inexact move. In each case the move is silently weakened or dropped.

1. **No same-exponent coordinate is free.** The code falls back to `cands[0]`. In a block of values near 1.0 with the source at 0.08, the result is `achieved = 0.0`, and the placebo equals copy bit for bit.
2. **The source sits at the bottom of a binade.** Say clean0 = 0.0625 exactly, so the A100 keep0 step is −3.05e-5, half a ULP of clean0's binade. A target with the same exponent then rounds by round-to-even to `achieved = 0.0`, even though `same_exponent` is True. The L40S has the mirror case: when buggy0 sits at the bottom of a binade, dA is half a ULP of the base's binade.
3. **The source is zero.** `frexp(0)` returns exponent 0, the same as values in [0.5, 1). A source equal to 0 can therefore be matched to a target near 0.7, where a step of about 6e-8 rounds to 0.

**How often.** In a dense-d0 simulation (30% of coordinates, one ULP up or down), 37 of 1,243 moves came out as zero and 39 had the wrong size.

**Why it changes the verdict.** If d0 has only one or two nonzero entries, which round 1 thought possible, one zero move makes the placebo identical to copy. Then moved_pl = 0, rule 3's fragility check cannot fail, and keep0 or inject can reach "explains the gap" without a real baseline.

**Fix.**
- Choose c2 by the property that matters: `fp16(x[c2] + step) − x[c2] == step` exactly, with a nonzero result. Do not use the frexp exponent.
- Add "all placebo moves exact" to the gates, or to `interpreted`, in both the code and the pre-registration.

### 3. Rule 5 (0.2 < r < 0.8) ignores the placebo and N_new
**Where:** `run_experiment.py:141-142`; pre-registration `:107`.

**The problem.** Code and pre-registration agree here, but the rule mislabels these cases (from t5):
- moved = 30/40, placebo 40/40 → "explains part of the gap (r = 30/40)". The placebo moved more configs than the variant did.
- moved = 20/40, placebo 20/40 → "explains part of the gap".
- moved = 20/40, N_new = 500 → "explains part of the gap".

Round 1 already warned that the partial band "may be exactly where noise lands". This is the band where the placebo matters most, and it is the only band that never checks it. Rules 3 and 4 do check N_new > 4; rule 5 does not.

**Fix.** Pre-register and code, before rule 5's label:
- N_new > 4 → "perturbs broadly";
- a placebo condition such as `2*(moved - moved_pl) < n`, or `moved <= moved_pl` if you want a weaker bar → "not distinguishable from fragility".

Add asserts for both to `test_decide_placebo.py`.

---

## Optional notes

**A. `interpreted` does not reflect all the gates.** At `:297`, `interpreted` covers only the copy gate and capture vs live. On L40S, the d0_L = 0 gate, inject exactness and the unexposed-identical gate only change the exit code. The L40S job has no dependent job, so its run.json would still show inject verdicts as if valid. Fold all gates into `interpreted`, or say in the pre-registration that any gate failure voids the labels.

**B. The pre-registration claims more test coverage than exists.** Pre-registration `:108` says `test_decide_placebo.py` "asserts every branch". It has no assert for "perturbs broadly" or for "explains part of the gap", and none for the exactness flags. Either add them (`t5_decide.py` has ready-made cases) or soften the sentence.

**C. The docstring is out of date.** `run_experiment.py:9-12` still describes the round 1 placebo ("buggy0 moved one ULP … same count and signs as d0"; "as above, A100 blocks").

**D. The placebo gets a verdict label of its own.** `decide(out["placebo"], …)` at `:301` runs with `placebo_moved=None`, which is treated as 0. The placebo can therefore be labelled "explains the gap". This is harmless, but `label=False` or a "baseline" label would stop readers from misreading it.

**E. The smoke test never ran a non-empty D, a dense d0, or a100clean0.** A smoke run with a synthetic d0 of more than 64 entries in one block would have caught required fix 1.

**F. Things checked and found fine:**
- The placebo never touches a d0 or dA coordinate, because `taken` is seeded with them.
- All fallback moves stay in the source's block, so the unexposed-identical gate stays valid for the placebo.
- Negative one-ULP steps at small values, at ±0.0625, at 0, at ±6e-5 and at subnormals are exact when the target binade matches.
- The L40S start check, `afterok`, the atomic `dump`, and L40S walltime (at most 10 cells × about 645 s) are as reviewed in round 2.
- N_new excludes D configs, including unexposed ones.
