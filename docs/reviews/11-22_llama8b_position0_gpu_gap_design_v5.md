Reviewer: fresh subagent (Claude Code)
Agent-ID: a1190ef225061a47f
Verdict: PASS
Transcript-SHA256: b799a324e27eff4a48283d1d2a908330e900a1942a394009252bbe057346d01d

## Brief given

You are a design reviewer for an experiment that has not run yet. This is round 5, and it is narrow. Round 4 (docs/reviews/11-22_llama8b_position0_gpu_gap_design_v4.md in ~/github/MATS) required two fixes:
1. The placebo is too often invalid under strict placement. It is now valid if the unplaced steps carry at most 1% of the total |step|.
2. Rule 5 mislabelled weak but real effects as "fragility". It is now "fragility" only when 2·moved_pl ≥ moved, and otherwise either "partial, below the placebo margin" or "explains part".

Optional notes 1, 3 and 4 were also addressed: moved_pl is now counted on exposed D, there is a test for the rule-5 invalid-placebo branch, and the header was updated.

Verify these, and check that nothing else broke. Do not edit any files.

- Use `git diff 2f4a01c HEAD -- 9_8_research/reviewer_experiments/exp_R16_position0 docs/experiments/11-22_llama8b_position0_gpu_gap.md` to see the changes.
- Run `python -I test_decide_placebo.py run_experiment.py` in the R16 folder.
- Rerun your round-4 scripts against the current code: t5_decide.py, t6_placebo_v4.py, t7_matching.py and t8_why.py, in /private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/, using r13env. Report the validity rate by density under the 1% tolerance.
- Check that a placebo can't become "valid" while weak enough to make "explains" easy. Compare the placed step mass with what was intended.
- Check that the pre-registration text matches the code exactly: the ordered rules and the placebo definition.

Write your review as markdown. Start with a verdict, PASS or FIXES NEEDED, then any required fixes with file:line. Return the full review as your final message; it will be saved verbatim.

## Report

# R16 design review, round 5

## Verdict: PASS

Both of round 4's required fixes are in the code, and both work. Optional notes 1, 3 and 4 are done too. The pre-registration matches the code word for word on the placebo definition and on all five ordered rules. I found nothing that the changes broke. Three optional notes follow, and none of them blocks the run.

**What I checked.** HEAD is `0c541b2` and the working tree is clean for R16. Only `outputs/` files are untracked. The script's sha256 is `bd47abab…`, the same hash `smoke_cpu.log` records. The job files are unchanged since `2f4a01c`. I ran everything on CPU with `r13env`.
- `python -I test_decide_placebo.py run_experiment.py` passes, exit 0.
- I reran `t5_decide.py`, `t6_placebo_v4.py`, `t7_matching.py` and `t8_why.py` against the current code.
- I wrote a new script, `t9_tol_v5.py`, for three things:
  - the validity rate by density under the 1% tolerance;
  - an independent float64 audit of the placed and unplaced step mass;
  - a fuzz over 300 seeds × 5 scales × both GPU directions.

All scripts are in `/private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/`.

---

## Status of round 4's items

| Item | Status | Evidence |
|---|---|---|
| **Fix 1.** Placebo valid if the unplaced steps carry ≤ 1% of total \|step\| | **Done** | `run_experiment.py:72` sets `UNPLACED_TOL = 0.01`. `:105-108` computes `unplaced_step_fraction` and sets `valid = bool(moves) and frac <= UNPLACED_TOL`. The whole info dict goes into `run.json` (`:295`). An empty step gives frac = 1.0 and `valid=False`. The pre-registration (`:58-62`) states the same rule, says the count and fraction are recorded, and says an invalid placebo only blocks positive labels. |
| **Fix 2.** Rule 5 says "fragility" only when 2·moved_pl ≥ moved | **Done** | `:162-167` checks `2*pm >= moved` → fragility, then `2*(moved-pm) < n` → "partial, below the placebo margin (r = …, r_pl = …)", then "explains part". The pre-registration (`:117-119`) has the same three branches in the same order. |
| Note 1. moved_pl counted on exposed D | **Done** | `:330-331` adds `moved_exposed`, summed over `k in Dset if k in exposed`. This is the same set and comparison `decide` uses for `moved` when `label=True` (`:121-122`). It is what gets passed as `placebo_moved` (`:337`), including in the exploratory D′ path. The pre-registration (`:101`) now says "on the same exposed D". |
| Note 3. Test for the rule-5 invalid-placebo branch | **Done** | The test calls `half` (r = 20/40) with `placebo_valid=False` and gets "(no valid placebo)". It also adds two new rule-5 asserts: n=10, m=3, pl=0 must give "partial…" and must not contain "fragility"; m=3, pl=2 gives fragility. |
| Note 4. Header | **Done** | `run_experiment.py:6` now points to `…_design*.md (rounds 1-4)`. The pre-registration status line and the review list include round 4. |

### `t5_decide.py`: all 26 cases

All 26 decide cases behave as expected:
- the r = 0.8 and r = 0.2 ties;
- 2(m − p) = n in rule 3 counts as distinguishable;
- N_new = 4 vs 5;
- coverage at 10/40, 32/40, 31/40, 9 and 0.

Two cases changed label, both as intended:
- "r=0.2+ n10" (m=3, pl=0) is now "partial, below the placebo margin (r = 3/10, r_pl = 0/10)". Round 4 got "fragility" here.
- "r=.5 pl=.5", "r=.75 pl=.75" and "r=.75 pl=1.0" still give "not distinguishable from fragility", now through the `2·pm ≥ moved` branch.

### `t6`, `t7`, `t8`

**t6.** The exactness audit is clean for every case: dense blocks with k = 40/64/65/100/128, 30/60/100% density, inexact cases (a)/(a′)/(b)/(b′)/(b″)/(c)/(c′), the empty step, and the overflow targets.

t6 does print `ANY PROBLEM: True`, with 286 fuzz flags and a flag on "dense random 30%". These are **stale checks**, not defects. t6 still asserts round 3's strict rule, that a valid placebo has no unplaced sources (`valid_but_unplaced_or_empty`, `valid_but_count_differs`). Round 4's fix deliberately removed that rule. The t9 audit, written for the new rule, finds zero defects (see below).

**t7.** Greedy placement now gives `valid=True` at every density from 5% to 30% (12 of 12 runs). Maximum matching still places a few more sources, e.g. 1,222 vs 1,201 of 1,243. That only matters for validity near the 1% boundary.

**t8.** The unplaced sources are still the tiny ones: |x| ≈ 2e-4 to 3e-4 or −0.0, with steps of 6e-8 to 2.4e-7. At densities of 0.1–5%, 20 of 20 placebos are valid. Round 4 had 20/20, 20/20, 19/20, 16/20 and 8/20.

---

## Validity rate by density under the 1% tolerance (t9)

Setup: Gaussian blocks (0.05·randn, fp16), A100 direction (base clean0, step −d0), 20 seeds per row. Rows marked "ulp" use random ±1-ULP d0 (±3-ULP gave the same validity rates). Rows marked "noise" use the more realistic d0 = fp16(x·(1 + 1e-3·randn)) − fp16(x) on the chosen subset. "Strict" is round 3's all-placed rule, shown for comparison.

| d0 density | ulp: valid (1%) | ulp: strict | ulp: max unplaced frac among valid | ulp: min placed mass among valid | noise: valid (1%) | noise: strict |
|---|---|---|---|---|---|---|
| 0.1% | 20/20 | 20/20 | 0 | 1.0000 | 19/20* | 19/20 |
| 0.3% | 20/20 | 19/20 | 1.5e-4 | 0.9999 | 20/20 | 20/20 |
| 1% | 20/20 | 18/20 | 4.9e-5 | 1.0000 | 20/20 | 20/20 |
| 2% | 20/20 | 14/20 | 3.8e-4 | 0.9996 | 20/20 | 16/20 |
| 5% | 20/20 | 4/20 | 6.1e-4 | 0.9994 | 20/20 | 6/20 |
| 10% | 20/20 | 1/20 | 5.8e-4 | 0.9994 | 20/20 | 2/20 |
| 30% | 20/20 | 0/20 | 5.1e-3 | 0.9949 | 20/20 | 0/20 |
| 50% | 0/20 (median frac 6.0e-2) | 0/20 | — | — | 20/20 | 0/20 |
| 60% | 0/20 (median 2.4e-1) | 0/20 | — | — | 13/20 | 0/20 |
| 100% | 0/20 | 0/20 | — | — | 0/20 | 0/20 |

\* The one invalid noise seed at 0.1% drew zero nonzero coordinates. With no sources there are no moves, so `valid=False`. This is correct. On A100, d0 = 0 is caught earlier by the premise gate in any case.

Up to 30% density the placebo is now valid in every run. Round 4 had 8/20 valid at 5%. The placebo fails only above roughly half-block density, where it should.

## Can a valid placebo be too weak, making "explains" easy?

No. For every placebo in t9 (density sweep plus fuzz, both GPU directions, about 4,200 audits), I recomputed each move in float64 and summed the placed and unplaced mass independently.

- **Zero defects.** No inexact or zero move, no move into another block, no source coordinate touched, no target used twice, and no change outside the recorded moves. Placed plus unplaced equals the source count. The code's `unplaced_step_fraction` matches the float64 value to within 1e-6, and placed + unplaced mass equals the intended mass.
- **Placed mass is always ≥ 99% of intended** when `valid=True`. The worst valid case delivered 0.9949 of the intended L1 step mass, at 30% density. Every placed step has exactly the size of its source step, so the shortfall is only the unplaced tiny steps.
- **The boundary holds against an adversarial case.** I built a block with 60 tiny unplaceable sources (|x| = 2e-4) plus 4 normal ones. It had an unplaced fraction of 1.44%, placed mass of 0.9856, and was marked **invalid**. So the tolerance cannot be satisfied by count alone.

The placebo therefore removes or adds at least 99% of the variant's absolute step mass, at the same sizes and in the same blocks. A placebo that moves fewer configs because it is weaker is ruled out to within 1%.

## Pre-registration against code

- **Placebo definition.** Pre-registration `:55-63` matches `make_placebo` (`:75-108`) on all of these: the base is this GPU's clean0, the step is −d0 on A100 and +dA on L40S, targets are other coordinates in the same 128-dim block, the landing must be exact and nonzero (fp16(x + s) − x = s), the tolerance is 1% of total |step|, and the count and fraction are recorded. The pre-registration also says an invalid placebo is not a gate. That matches the code: `placebo_valid` is recorded in the cell (`:325`) but is not in `gates_ok` and not in `interpreted` (`:349`).
- **Ordered rules.** Pre-registration `:105-119` matches `decide` (`:141-167`) branch by branch:
  - rule 1: n = 0;
  - rule 2: n < 10;
  - rule 3: `5m ≥ 4n`, then N_new > 4, then invalid placebo, then `2(m − pm) < n`, then coverage;
  - rule 4: `5m ≤ n`;
  - rule 5: N_new > 4, then invalid placebo, then `2pm ≥ m`, then `2(m − pm) < n` → partial, then explains part.

  The label strings match. The definitions of moved_pl (`:101`) and `interpreted` (`:122-123`) match the code (`:330-337`, `:349`).

---

## Required fixes

None.

## Optional notes

1. **Rule 3 can still say "fragility" when the placebo moved less than half as many.** See `run_experiment.py:150-151` and pre-registration `:110`. Round 4 said rule 3 was unaffected. That is true for the r < 0.5 problem, but the labels are now inconsistent across bands. In rule 3, if m − n/2 < pm < m/2, the verdict is "not distinguishable from fragility". Rule 5 would call the same ratio "partial…".
   - Example: n = 40, moved = 36, moved_pl = 17 gives "fragility". But n = 40, moved = 31, moved_pl = 14 gives "partial, below the placebo margin". The stronger effect gets the weaker label.
   - The reverse case: m = n, pm = n/2 gives "explains the gap", while rule 5 would call r_pl = r/2 "fragility".
   - The window is only 1–3 integer values of moved_pl for each moved, so this is unlikely to matter.
   - For uniform wording, apply rule 5's three-way split in rule 3 as well: `2pm ≥ m` → fragility; `2(m − pm) < n` → partial; else explains. This needs one more test assert.
2. **Test the other side of the tolerance.** The new unit test checks that a tiny unplaced source stays valid. It does not check that an unplaced mass above 1% is invalid. My adversarial block (1.44% → invalid) shows the code is correct. Adding one assert would pin it down.
3. **Pre-registration wording on dense d0.** `:58-60` now covers only small-magnitude sources. In t9, a ±1-ULP d0 at 50% or higher density gives 0/20 valid placebos. One clause would keep readers from being surprised by "(no valid placebo)" if d0 turns out dense, for example: "a difference with about half of each block nonzero exceeds the tolerance".

Also, t6 now gives misleading output. If anyone reuses it, drop its two strict-rule checks (`valid_but_unplaced_or_empty`, `valid_but_count_differs`) in favour of t9's mass checks.
