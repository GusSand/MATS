Reviewer: fresh subagent (Claude Code)
Agent-ID: a6fc9b302b18a73a4
Verdict: FIXES NEEDED
Transcript-SHA256: 6e26ac6fbda2475b297a5d6e50fef96df79f39c6a0a5a576b4c048d459a7f83d

## Brief given

You are a design reviewer for an experiment that has not run yet. This is round 3. Round 2 required fixes. Verify the fixes and look for remaining problems. Flag only issues that could change a verdict, lose data, or make the pre-registration disagree with the code. Do not edit any files.

Repo: ~/github/MATS.
- Prior reviews: docs/reviews/11-22_mistral7b_true_head_patching_design.md (round 1) and _design_v2.md (round 2). Read round 2 first.
- Pre-registration: docs/experiments/11-22_mistral7b_true_head_patching.md
- Script: 9_8_research/reviewer_experiments/exp_R15_mistral_true_heads/run_experiment.py
  - `git diff 18f3df2 HEAD -- <script> <pre-reg> run.sbatch` shows the round-2 changes.
  - `decide()` and the new `decide_selftest()` can be run without a model.
- Job file: run.sbatch. Smoke log: smoke_cpu.log, run on commit a354921. Both are in the R15 folder.
- Round 2's test harness is in /private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/r15/. Reuse it.

Check:
1. Round 2's required fixes:
   - Fisher-based M2 (c), and the `fisher_greater` implementation;
   - the H22-role condition and the new "H22's role not shown" branch;
   - M1a uninterpretable without sanity; M1b specificity;
   - M7 on the no-pair path;
   - the `says_less` and `hit_max_new` flags everywhere.
   Test each with synthetic worlds, including edge cases: H22 and noH22 both succeed often; exactly 5/60 vs 0/60; Δ readings with None deltas.
2. The adopted optional items:
   - the noH22 curve;
   - "H22 special" counted per config;
   - "M5 specific";
   - the M7 "k of n" report;
   - the R15 self-hash gate;
   - `decisions_before_M7`;
   - the wording fixes.
3. Is `decide_selftest` correct, and does it cover the reviewer's counterexamples?
4. Any remaining mismatch between the pre-registration text and the code.

Write your review as markdown. Start with a verdict, PASS or FIXES NEEDED, then required fixes with file:line, then optional notes. Return the full review as your final message; it will be saved verbatim.

## Report

# Design review round 3: R15, testing the Mistral-7B head claims with true per-head patching

**Verdict: FIXES NEEDED.** One small fix is required, a one-line change in the code plus one sentence in the pre-registration.

All five round-2 fixes are in, and every optional item that was adopted is in. I tested each with synthetic worlds. The code matches the pre-registration everywhere except one rule. In that rule (M1a), the "uninterpretable" condition is applied too broadly, so it can discard a refutation. The code and the text agree with each other on it, but both are wrong.

## What I checked and found correct

**Setup and provenance**
- `decide_selftest()` passes. `fisher_greater` matches a brute-force hypergeometric enumeration on 286 tables (0 mismatches). It gives p = 0.5 for 1/60 vs 0/60, p = 0.0287 for 5/60 vs 0/60, and p = 0.0594 for 4/60 vs 0/60.
- **Smoke is current.** `smoke_cpu.log` records R15 sha `9f7b34cc…`. That equals the sha of both `a354921` and HEAD; the script has not changed since. It also records R14 sha `20819e06…`, the required one. The round-2 provenance note is resolved.

**Fix 1: M2 (c) now uses Fisher** (`run_experiment.py:316-318`). Tested at layer 2 with the even k = 16 set succeeding and no odd successes:
- 4/60 noH22 successes vs 0/60 odd gives "even-only k = 16 repairs; H22's role not shown" (p = 0.059).
- 5/60 vs 0/60 gives "parity holds" (p = 0.029).
- 1/60 with H22 working alone gives "carried by H22". The selftest covers this case.
- When H22 and noH22 both succeed often (noH22 40/60, even-with-H22 30/30, H22 alone correct), the verdict is "parity holds" because the first matching rule wins. That is what the pre-registration says.

**Fix 2: H22's role** (`:319-328`).
- If only the even k = 16 set succeeds, the verdict is "H22's role not shown". The selftest covers this.
- H22-only plus 4 noH22 successes gives "carried by H22" (with-H22 30/30 vs noH22 4/60).
- **Δ readings with every Δ None:** p = 1 and "not supported", which is conservative.
- If only the even k = 16 set has Δ > 0, the result is "role not shown".

**Fix 3: sanity and M1b specificity** (`:282-300`).
- When sanity fails and no layer is correct, M1a is "uninterpretable".
- M1b is correct in every case I tried:

  | Synthetic world | M1b result |
  |---|---|
  | All 32 layers Δ > 0, the five layers the largest | holds |
  | All 32 layers Δ > 0, the five layers the smallest | non-specific |
  | Only the five layers Δ > 0 | holds |
  | 3 of the five Δ > 0 | not supported |
  | Every Δ None | not supported |
  | Baseline Δ > 0 | untestable |

**Fix 4: M7 on the no-pair path** (`:253-262`, `:272`). `m7_decisions` now runs before the early return:
- With a generalization list and no pair, M7 bug counts, eligibility, `repair_transfers` and `M7_report_as` are all computed. The smoke's no-pair run shows them.
- With no generalization data, M7 is "not run".
- M7 data collection (`:555-571`) sits outside the pair block, so it runs on both paths.

**Fix 5: output flags.**
- `rec` (`:463-464`) records `hit_max_new` and `says_less`; so do the M7 format rows (`:563-564`) and `five_layer_repair` (`:568-569`).

**Adopted optional items**
- **noH22 curve.** `ten_even_heads_curve_noH22` reports k = 4, 8, 12 (`:346`). The pre-registration says k = 16 cannot be tested without H22.
- **"H22 special" per config** (`:362-370`).
  - If H22 is correct only in the five-layer config and 4 other heads are correct only in the all-layers config, the result is True: the all-layers config is not counted, because H22 fails there.
  - If H22 and 4 others are correct in the same config, that is 5 heads, and the result is False.
- **"M5 specific."** With 3/10 random 5-layer sets correct it is False; with 0/10 it is True.
- **M7 "k of n"** is reported, and 0 eligible pairs gives False.
- **R15 self-hash gate** (`:439-441`): it runs unless `--limit` is set, and `run.sbatch:23` requires the variable.
- **`decisions_before_M7`** (`:551-552`) is computed and saved before M7.
- **Wording.** All three wording fixes are in (pre-reg `:63-65`, `:87`, `:120`).

## Required fix

### 1. M1a turns a refutation into "uninterpretable" (pre-reg `:89-91`, script `:286-287`)

When the all-layers sanity transplant fails, `M1a_no_layer_repairs_gen` is "uninterpretable" even when some layer's all-heads patch *is* correct. I tested this: sanity *neither*, layer 2 *correct* gives "uninterpretable", with `M1_correct_layers = [2]`.

The pre-registration gives the reason in its own text: "a null from an ineffective patch says nothing". That reason applies only to the null. If a single layer repairs, the patch clearly works and the claim "no single layer fully repairs" is refuted.

This case is plausible. Transplanting all 32 layers puts the simple prompt's attention output into every layer. That can make the reply look like a simple-format continuation that the classifier marks *neither*, while one layer alone gives *correct*.

**Fix.** Code:

```python
d["M1a_no_layer_repairs_gen"] = (False if d["M1_correct_layers"] else
    True if d["sanity_correct"] else "uninterpretable: ...")
```

Pre-registration `:90`: change it to "a correct layer refutes M1a regardless; only the null is uninterpretable when the sanity transplant is not *correct*". Also update the sanity gate line (`:82-83`, "treat single-layer nulls with caution") to point to this rule.

## Optional notes

1. **`decide_selftest` coverage** (`:381-415`).
   - **Correctness.** It is correct and passes.
   - **Round-2 counterexamples covered:** the 1/60 case and the k = 16-only case.
   - **Not covered:**
     - M1a with a failed sanity transplant;
     - M1b non-specific;
     - the no-pair M7 path with data;
     - the 5/60 boundary inside `decide()` (only checked at the `fisher_greater` level);
     - "H22 special" across configs;
     - `M5_specific` False.

   I tested all of these and the code handles them. Adding a few asserts, including the fixed M1a case, would keep them from regressing.
2. **`decisions_before_M7` can block M7.** If `decide()` raised at `:551`, the job would die before the 500 M7 generations. I found no input that makes it raise, and the smoke's forced-pair run passed through it. Still, `try: … except Exception as e: run["decisions_before_M7"] = {"error": repr(e)}` costs nothing and protects M7's data.
3. **The H22-role Fisher test compares unequal k mixes.**
   - The with-H22 even subsets at k ≤ 12 are 5/11/14 at k = 4/8/12. The noH22 subsets are 20/20/20.
   - In a count-driven world this tilts the test toward "H22 helps".
   - The verdict reaches this test only when (c) fails, which means noH22 ≤ 4/60. That bounds the bias, and my count worlds gave p ≈ 0.16–0.75, never a false "carried by H22".
   - A stratified comparison at k = 12 only (14 vs 20) would remove the bias.
4. **"Every rate is reported split by H22 ∈ S"** (pre-reg `:99-100`). Only the even rates are split; mixed subsets also contain H22. Either say "every even-only rate" or add the split for mixed.
5. **"Every output records `hit_max_new` and `says_less`"** (pre-reg `:59`). The gate outputs (`unpatched`, `identity_output`, `determinism_outputs`) don't. They are compared as strings, never classified, so only the wording is affected.

## Files

- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R15_mistral_true_heads/run_experiment.py`: `:286-287` (required fix), `:381-415` (selftest), `:551` (`decisions_before_M7`)
- `/Users/gus/github/MATS/docs/experiments/11-22_mistral7b_true_head_patching.md`: `:82-83`, `:89-91`, `:99-100`
- Test harness (scratch): `/private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/r15/t3.py` (round-3 worlds), next to the round-2 files `t_decide.py`, `t2.py` and `load.py`.
