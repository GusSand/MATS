Reviewer: fresh subagent (Claude Code)
Agent-ID: a57c5feb6be741e4d
Verdict: FIXES NEEDED
Transcript-SHA256: 5d45c50badf66f793eb949c615252ad97cc0cc9cfe6faaad91711220e001552c

## Brief given

You are a design reviewer for an experiment that has not run yet. This is round 4, and it is narrow. Round 3 (docs/reviews/11-22_llama8b_position0_gpu_gap_design_v3.md in ~/github/MATS) required three fixes:
1. `make_placebo` must not crash on a dense d0, and must not lose `run.json`.
2. Placebo moves must be exact, or the placebo is marked invalid.
3. The 0.2 < r < 0.8 band must check N_new and the placebo.

Round 3's optional notes A–D were also addressed: `interpreted` now folds in all gates, the test now covers more, the docstring is updated, and the placebo is scored descriptively.

Verify all of this, and check that nothing else broke. Do not edit any files.

- Use `git diff ab993ff 2f4a01c` to see the changes to the script, the unit test and the pre-registration (docs/experiments/11-22_llama8b_position0_gpu_gap.md).
- Script folder: 9_8_research/reviewer_experiments/exp_R16_position0/
- Run `python -I test_decide_placebo.py run_experiment.py`.
- Round 3's scripts t4_placebo.py and t5_decide.py, and the r13env venv, are in /private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/. Adapt them to the current API and rerun them. In particular, cover:
  - the dense case;
  - the three inexact-move cases;
  - the 26 decide cases;
  - the zero-source case;
  - whether a placebo can ever be marked valid with a zero or wrong-size move.
- Check that the pre-registration text matches the code exactly, including the ordered rule list and the definition of `interpreted`.

Write your review as markdown. Start with a verdict, PASS or FIXES NEEDED, then any required fixes with file:line. Return the full review as your final message; it will be saved verbatim.

## Report

# R16 design review, round 4

## Verdict: FIXES NEEDED

All three of round 3's required fixes are in the code and work as specified. Notes A–D are addressed. The ordered rule list and the definition of `interpreted` in the pre-registration match the code.

Two problems remain. Both come from the new fixes, and either can change a verdict:

1. **Placebo validity fails at much lower density than the pre-registration says.** The pre-registration says an invalid placebo needs "more than half a block nonzero". In my simulations it happens far sooner. With 2% of coordinates differing, 4 of 20 placebos are invalid; with 5%, 12 of 20 are. One small-magnitude source with no smaller value in its block is enough. When the placebo is invalid, keep0 and inject can never get a positive label.
2. **Rule 5 says "fragility" when the placebo moved nothing.** The fragility check `2(moved − moved_pl) < n` sits inside the 0.2 < r < 0.8 band. So every r in (0.2, 0.5) gets "not distinguishable from fragility", even when moved_pl = 0.

Both fixes are small. Neither needs compute.

**What I checked.** Commit `2f4a01c`, with a clean working tree. Script sha256 is `26a1d253…`, the same hash `smoke_cpu.log` records. I used torch on CPU (`r13env`).
- `python -I test_decide_placebo.py run_experiment.py`: passes, exit 0.
- `t5_decide.py`, all 26 cases: it still runs against the new API. The output is in the rule-5 section below.
- `t4_placebo.py`, ported to the `{"moves", "unplaced", "valid"}` API as `t6_placebo_v4.py`. It includes an independent float64 audit of every move and a fuzz over 800 placebos.
- `t7_matching.py`: compares greedy placement with a maximum bipartite matching.
- `t8_why.py`: which sources go unplaced, and how often a placebo is valid at each density.

All of these are in `/private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/`.

---

## Status of round 3's fixes

| Round 3 item | Status | Evidence |
|---|---|---|
| **1.** `make_placebo` must not crash on a dense d0, and must not lose run.json | **Done** | `run_experiment.py:85-101`: an unplaceable source goes to `unplaced` instead of raising. `dump(run, path)` now runs before the placebo is built, on A100 (`:256`) and on L40S (`:279`). In t6, nothing crashes for one block with k = 40, 64, 65, 100 or 128 nonzero entries, for random 30%, 60% and 100% density, or for an all-zero step (`valid=False`). It takes at most 0.1 s on CPU. |
| **2.** Placebo moves exact, or the placebo is marked invalid | **Done** | `:89-91`: a target qualifies only if `fp16(x + s) − x == s` and the result is nonzero. `:101` sets `valid = not unplaced and bool(moves)`. The float64 audit (t6) checks every move: exact size, nonzero, a different coordinate in the same block, no d0 coordinate touched, no target used twice, and placed + unplaced = number of sources. |
| — inexact case (a): no target with the same exponent is free | **Done** | A block of 1.0s with the source at 0.08: unplaced, `valid=False`. If one value in the block can take the step, the move goes there and is exact. |
| — inexact case (b): source at the bottom of a binade (half-ULP step) | **Done** | If a smaller-binade target exists, the move is placed exactly (step −3.05e-5 onto a target near 0.05). If none exists, `valid=False`. The L40S mirror case (+half ULP) also gives `valid=False`. |
| — inexact case (c): source equal to zero (±0.0) | **Done** | `frexp` is no longer used, so a zero source no longer matches targets near 0.7. Its step (±5.96e-8) lands only on a zero or subnormal target. If the block has none, `valid=False`. |
| — can a placebo be valid with a zero or wrong-size move? | **No** | Fuzz over 400 seeds × 2 bases: 5 scales (1e-4 to 30), sparse and dense, ± signs, and a run of zeros in some seeds. Result: 348 valid, 452 invalid, and **0** placebos with a zero move, a wrong-size move or a miscount. Targets that would overflow (65504 + 32 → inf) are rejected. |
| **3.** The 0.2 < r < 0.8 band checks N_new and the placebo | **Done, but see required fix 2** | `:151-158`. N_new = 500 at r = 0.5 gives "perturbs broadly". r = pl = 0.5, r = pl = 0.75, and r = 0.75 with pl = 1.0 all give "not distinguishable from fragility". |
| A. `interpreted` folds in all gates | **Done** | `:314-315` and `:338-339` combine the copy gate, capture = live, unexposed-identical for keep0, inject and placebo, and on L40S d0 = 0 and inject exactness. This matches pre-registration `:118-119` item by item. |
| B. Test coverage | **Done, with one gap** | It adds asserts for the dense case, rule 3 with an invalid placebo, the rule-5 placebo check, and N_new in rules 3 and 5. There is still no assert for the rule-5 invalid-placebo branch (`:153-154`). See optional note 3. |
| C. Docstring | **Done** | `:9-13` describe the current placebo. |
| D. Placebo scored descriptively | **Done** | `:320` passes `label=False`, and the smoke log shows `'verdict': 'descriptive'` for the placebo. |

**`t5_decide.py`, all 26 cases.** The ties land as round 3 found:
- r = 0.8 and r = 0.2 ties;
- 2(m − p) = n counts as distinguishable;
- N_new = 4 gives "explains", N_new = 5 gives "perturbs broadly";
- the coverage cases (10/40, 32/40, 31/40, 9, 0).

The rule-5 cases now give the new labels. Branch order matches pre-registration `:102-115` exactly:
- rule 3: N_new, then invalid placebo, then fragility, then coverage;
- rule 4: no placebo check;
- rule 5: N_new, then invalid placebo, then fragility, then "explains part".

---

## Required fixes

### 1. The placebo is invalid far more often than the pre-registration says, so keep0 and inject may never get a label

**Where:**
- `run_experiment.py:85-101`, where `valid` requires all sources placed;
- pre-registration `:57-59` ("A dense difference, with more than half a block nonzero, can leave sources unplaced").

**The problem.** The placement rule is exact, but strict. A source is unplaceable whenever its step is finer than the fp16 grid of every free target. This happens to the smallest-magnitude source in a block that has no smaller value. All the unplaced sources in t8 look like this: clean0 ≈ 2e-4, 3e-4 or −0.0, with steps of 6e-8 to 2.4e-7, in blocks whose max |x| ≈ 0.16. Validity rate on Gaussian blocks (0.05·randn, random ±1-ULP d0, 20 seeds each):

| fraction of coordinates nonzero | 0.1% | 0.3% | 1% | 2% | 5% | 30% |
|---|---|---|---|---|---|---|
| placebo valid | 20/20 | 20/20 | 19/20 | 16/20 | 8/20 | 0/20 (42 of 1,243 unplaced) |

R13 recorded only `max|d0| = 6.1e-5`, not the count. A d0 caused by prompt length (17 vs 19 tokens) changing the kernels' reduction order is plausibly not sparse. If d0 has more than about 50 nonzero entries, the most likely outcome is:
- any r > 0.2 gets "not distinguishable from fragility (no valid placebo)" in both cells;
- the experiment uses about 2.4 GPU-h and cannot say "explains".

The pre-registration's "more than half a block" makes this look like an unlikely corner case. It is not, so as written the pre-registration does not describe what the code does.

Greedy first-fit placement adds to this. A maximum bipartite matching places more sources: 61 vs 55 of 64 in one block, and 1,222 vs 1,201 of 1,243 at 30% density. It does not rescue validity, because the unplaceable sources have no qualifying target at all.

**Fix.** Choose a rule before the run. Either:
- **(a) Tolerate tiny unplaced steps** (recommended). Make the placebo valid if the unplaced steps carry at most a small fraction of the total step size, for example `sum|s_unplaced| ≤ 0.01 · sum|s|`. Record that fraction and the unplaced count in `run.json`. The unplaced steps are 100 to 500 times smaller than the typical one-ULP step at |x| ≈ 0.05, so leaving them out barely weakens the placebo. Add the same sentence to the pre-registration, and a unit test for a block with one tiny unplaceable source plus many normal ones.
- **(b) Keep the strict rule.** Then correct pre-registration `:57-59` to say the placebo is invalid whenever any single source has no target with an exact landing. Add that this is expected unless d0 is very sparse (roughly fewer than 50 nonzero entries), and that keep0 and inject then cannot be labelled above "does not explain".

Optionally, use a maximum matching (augmenting paths within each block) in place of first-fit. `t7_matching.py` has a 15-line version.

### 2. Rule 5 labels r in (0.2, 0.5) as "fragility" even when the placebo moved nothing

**Where:** `run_experiment.py:155-156`; pre-registration `:114`.

**The problem.** In the 0.2 < r < 0.8 band, the check `2(moved − moved_pl) < n` needs r − r_pl ≥ 0.5 before it gives "explains part". So any r ≤ 0.5 fails it, whatever the placebo did. From t5:
- n = 10, moved = 3, moved_pl = 0 → "not distinguishable from fragility";
- n = 40, moved = 12, moved_pl = 0 → the same label.

In both cases the placebo moved zero configs, so the result is clearly distinguishable from fragility. The label states the opposite of the evidence. Readers will take "fragility" to mean the boring hypothesis won. "explains part of the gap" is now reachable only for 0.5 ≤ r < 0.8 with a near-inert placebo. Code and pre-registration agree, so this is a design error and not a mismatch. Round 3 offered this margin as one option; inside rule 5 it is too strict a bar.

**Fix.** Separate "the placebo moves as much" from "the margin is small". For example, in rule 5:
- `2 * moved_pl >= moved` (the placebo moves at least half as many) → "not distinguishable from fragility";
- else `2 * (moved - moved_pl) < n` → "partial, below the placebo margin (r = moved/n, r_pl = moved_pl/n)";
- else → "explains part of the gap (r = moved/n)".

Change pre-registration `:111-115` to match, and assert in the test that n = 10, moved = 3, moved_pl = 0 does **not** give "fragility". Rule 3 is not affected: there r ≥ 0.8, so the 0.5 margin still lets moved_pl reach 0.3 n.

---

## Optional notes

1. **moved_pl is counted on all of D, not on exposed D.** `:320` scores the placebo with `label=False`, so `pr["moved"]` counts all of D. The docstring (`:113`) and pre-registration `:98` say moved_pl is "the same count" on exposed D.
   - When `interpreted` is true the two are identical. The copy gate gives copy class = R13 class on D. The placebo's unexposed-identical gate then means no unexposed D config moves.
   - When a gate fails they can differ, but labels are void then anyway.
   - To match the text exactly, pass the exposed-D count: call `decide(..., label=True)` for the placebo and keep only its numbers, or add a `moved_exposed` field.
2. **`placebo_valid` is recorded in the cell (`:316`) but is not a gate and is not in `interpreted`.** This matches the pre-registration: an invalid placebo only changes labels, through `decide`. It is worth one sentence in the pre-registration's gates section, so nobody expects exit code 2 from it.
3. **One test gap.** `test_decide_placebo.py` has no assert for the rule-5 invalid-placebo branch (`:153-154`). Pre-registration `:117` says the test covers "the invalid-placebo … cases". Add one line: `half` with `placebo_valid=False` → "… (no valid placebo)".
4. **Header.** `run_experiment.py:6` lists only design review round 1. This is cosmetic.
5. **Checked and found fine:**
   - The L40S `position0_l40s.pt` save comes after `make_placebo` (`:281`). That is now safe, because `make_placebo` cannot raise on any input I tried.
   - The cost on GPU is at most 4,096 × 128 scalar checks, which takes seconds.
   - The job files are unchanged since `ab993ff`.
   - The smoke hash matches the script.
   - The rule-3 order and the integer thresholds are unchanged from round 3.
