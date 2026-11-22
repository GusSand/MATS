Reviewer: fresh subagent (Claude Code)
Agent-ID: a3ccbb8941251c250
Verdict: FIXES NEEDED
Transcript-SHA256: 9559c021f682dc4b2825333200b261cd4d2bc48f27ce7ac6a966d88eee0d27a6

## Brief given

You are a design reviewer for an experiment that has not run yet. This is round 2. Round 1 (docs/reviews/11-22_llama8b_position0_gpu_gap_design.md in ~/github/MATS) required fixes, and the design and script have since been revised. Verify the fixes and look for new problems. Flag only issues that could change a verdict, lose data, or make the pre-registration disagree with the code. Do not edit any files.

Files:
- Pre-registration: docs/experiments/11-22_llama8b_position0_gpu_gap.md
- Script: 9_8_research/reviewer_experiments/exp_R16_position0/run_experiment.py
- Job files: run_a100.sbatch and run_l40s.sbatch, in the same folder.
- Smoke log: smoke_cpu.log, in the same folder. It used a random tiny Llama on CPU. The A100 role used a synthetic one-ULP delta, and the L40S role read the A100 role's `position0.pt`. With `--limit 40`, D is empty by construction; the copy gates fail by design.
- R13 script and results: 9_8_research/reviewer_experiments/exp_R13_true_head_patching/

Check:
1. Each of round 1's required fixes 1–8: is it implemented correctly?
   - Fix 1: the premise gate, exit codes, and the L40S check of the A100 `run.json`.
   - Fix 2: the capture-vs-live hook. Is a forward hook on `p.attn` during unpatched `generate()` the right comparison, and is the first call the prefill?
   - Fix 3: exposure and the unexposed-identical gate.
   - Fix 4: the all_steps reproduction statistic via R13's functions.
   - Fix 5: the placebo construction (`ulp_step` and `make_placebo`). Test them yourself on real fp16 tensors, including negative values and values near binade edges. Does placebo give the boundary-fragility baseline intended? Note that placebo is the same object on both GPUs.
   - Fix 6: the narrowed claim plus the donorA100 cell.
   - Fix 7: the inject exactness check and the a100clean0 condition.
   - Fix 8: node pinning and the exploratory D′ fallback.
2. `decide()`. Test it on synthetic dicts. Check that r uses exposed D; that the placebo_r threshold, the "perturbs" label and the outside-flip directions are right; and that the verdict labels match the pre-registration exactly.
3. Are there exit paths that lose data, or a job-2 start condition that can wrongly pass or fail?
4. Do the pre-registration and the code match everywhere?

Write your review as markdown. Start with a verdict, PASS or FIXES NEEDED, then required fixes with file:line, then optional notes. Return the full review as your final message; it will be saved verbatim.

## Report

# R16 design review, round 2

## Verdict: FIXES NEEDED

Round 1's fixes 1, 2, 4, 7 and 8 are implemented correctly. Fixes 3 and 6 are mostly right. Fix 5, the placebo, is built wrongly for the A100 role. Round 1's own wording caused this: it said "buggy0, plus a one-ULP perturbation". Because of that error, the A100 keep0 cell cannot reach "explains the gap" in the case where position 0 really is the cause. There are also three verdict-logic problems, and the pre-registration and code disagree in four places. All of them can be fixed without new cells or extra compute.

Everything below was checked against the current files: R16 sha256 `e73b99a6…`, which matches the smoke log, and R13 `1ce5b0cc…`. I tested `ulp_step`, `make_placebo` and `decide` in a scratch venv with torch 2.15 on CPU. I recomputed the R13 facts from `results/r13_24014365` and `results/r13_24084222`, and all the pre-registration numbers check out: D = 40/60, R8 in D = 37/58, R8-correct 120 vs 83 and 499 vs 441, reproduction 1000/200/60 vs 942/199/59. R13 ran on nodes `ga004`/`gl044`, and its first hook call has shape 19, which is the prefill.

---

## Status of round 1 fixes

| Fix | Status | Notes |
|---|---|---|
| 1. Premise gate, exit codes, L40S reads the A100 `run.json` | **Partly done** | A100 d0 = 0 → dumps and exits 2 (`:199-202`). End-of-run `sys.exit(2)` on any gate failure (`:291-292`). L40S refuses to start unless `all_gates_ok` and `pos0_n_nonzero > 0` (`:140-142`), before `makedirs`. The delta sha256 is recorded. **Missing:** a nonzero d0 on L40S is still not a gate. See required fix 5. |
| 2. Capture vs live | **Done, correct** | A forward hook on `p.attn` during unpatched `generate()` is the right comparison. The patch acts on the layer-10 output, so layers 0–10 up to that output are identical in patched and unpatched runs. The first call is the prefill: R13 recorded `hook_call_shapes` = `[19, 1, 1, …]` on both GPUs. `all_equal` (`torch.equal` on the full tensor) would also fail on any shape mismatch. The lambda returns `None`, so it does not modify the output. |
| 3. Exposure and the unexposed-identical gate | **Mechanics correct; verdict use flawed** | The block mapping (block h = coordinates h·128…h·128+127) matches R13's `_swap` view. The gate holds by construction for keep0, inject and placebo, in both prefill and all_steps. But the verdict ignores how much of D is exposed. See required fix 2. |
| 4. all_steps R8/R1 reproduction | **Done, correct** | `r8` is kept (`:151`). `reproduction_gate(reproduction(out[v], reference_outcomes(r8, r1)))["values"]` is computed for every variant (`:273-276`). The configs carry `heads`, so heads matching works. |
| 5. Placebo | **Wrong on A100** | See required fix 1. `ulp_step` itself is correct. |
| 6. Narrowed claim and donorA100 | **Claim done; donorA100 decided with the wrong rule** | The pre-registration (`:30-34`) states the narrowed claim. The donorA100 donor is the A100's full `clean_full` (1×17×4096). The code judges it like keep0, on exposed D only. See required fix 4. |
| 7. Inject exactness and a100clean0 | **Done, with one gap** | It checks `exact_delta`, `n_changed` against `n_delta_nonzero`, and `buggy0`/`clean0` equality, and saves `position0_l40s.pt`. a100clean0 runs only when `buggy0_L ≠ buggy0_A`. When `buggy0_L == buggy0_A`, inject equals clean0_A bit for bit, because fp32 holds the fp16 difference and the sum exactly. Exactness is recorded but does not gate anything, and a100clean0 has no pre-registered rule. See required fixes 4 and 5. |
| 8. Node pinning and the D′ fallback | **Done, one bug** | `--nodelist=ga004` / `gl044` are set. `same_node_as_r13` and `same_gpu_string_as_r13` match R13's hostnames (`gl044.hpc.nyu.edu`, `ga004.hpc.nyu.edu`) and GPU names. D′ (`:262`) is defined as pre-registered, but its verdict uses the placebo_r computed on D. See required fix 6. |

### `ulp_step` tested on fp16

I compared it against `np.nextafter` on real fp16 tensors. It matched on 200k random values and on ±1, ±0, ±0.0625 (a binade edge), ±2⁻¹⁴, ±smallest subnormal and ±65504. Up always gives a larger value and down a smaller one. Negative values go the right way: up from −1 gives −0.99951. Both +0 and −0 step to ±the smallest subnormal. Up from −(smallest subnormal) gives −0. Up from ±max gives ±inf, which does not matter here. The non-contiguous slice `buggy[:,0]` is handled.

---

## Required fixes

### 1. The A100 placebo is keep0 plus noise, not a baseline for copy
**Where:** `run_experiment.py:71-88`, `:209`, `:229`, `:232-234`; pre-registration `:52-54`.

**The problem.** `make_placebo(buggy0, d0)` returns buggy0 with one-ULP moves at other coordinates. That is right on L40S, where copy writes clean0_L = buggy0_L, so placebo − copy is the matched perturbation p. On A100, copy writes clean0_A = buggy0_A + d0, so placebo − copy = −d0 + p. The A100 placebo therefore contains keep0's entire change plus an extra perturbation.

I confirmed this on a synthetic case:
- d0 nonzero at coordinates 1000 and 1100.
- A100 placebo − copy is nonzero at coordinates `[936, 1000, 1036, 1100]`.
- keep0 − copy is nonzero at `[1000, 1100]`.
- L40S placebo − copy is nonzero at `[936, 1036]`.

**Why it changes the verdict.** Suppose the hypothesis is true: removing d0 sends the exposed D configs to the L40S class. Then keep0 gives r ≈ 1. The placebo also removes d0, so its r is also ≈ 1, unless p happens to restore what d0 did. So r − r_placebo ≈ 0 < 0.5, and `decide` returns "explains part of the gap (r = 1.00)". I checked this on a synthetic dict. "Explains the gap" on A100 can only be reached if an unrelated perturbation p restores the A100 class, which is the opposite of what the test intends. "Placebo is the same object on both GPUs" holds only when `buggy0_L == buggy0_A`, and the A100 copy donor is still not buggy0.

**Fix.** Build the placebo on each GPU's copy donor, and mirror the change the test variant makes relative to copy:
- **A100:** placebo0 := clean0_A, moved at other coordinates of the same blocks, in the direction of −d0, so it mirrors keep0.
- **L40S:** placebo0 := clean0_L, moved in the direction of +dA, so it mirrors inject.

Keep the d0 coordinates excluded (`taken`). Update the pre-registration sentence to match.

**Also match the size.** "One ULP" at the new coordinate can be much smaller than d0. In my test, the move at coordinate 1036 (value −3e-4) was −2.4e-7, against a d0 entry of 6.1e-5, 256× smaller. For values drawn from N(0, 0.05), one ULP is 0.03–1.0× of 6.1e-5, with median 0.5×. A smaller placebo lowers r_placebo and makes "explains" easier to reach.
- Choose c2 in the same block with the same fp16 exponent as the d0 coordinate's value, so that one ULP has the same absolute size. Alternatively, write `fp16(x[c2] + d0[c])` and check that the step is exact.
- Record the achieved placebo deltas next to `placebo_moves`.

### 2. "Explains the gap" can be declared from a few exposed configs
**Where:** `decide` `:97-108`; pre-registration `:85-94`.

**The problem.** r is computed on exposed D only, with no minimum size and no coverage term. In R13, the D configs' block usage is uneven:
- **prefill:** all 40 configs are even-only; blocks 4 and 14 are each used by only 4 configs, and no odd block is used.
- **all_steps:** blocks 13 and 25 are each used by 1 config; the even blocks by 19–35.

If d0 is nonzero only in block 13, then exposed D = 1 config in all_steps. With r = 1 and N_new ≤ 4, the verdict is "explains the gap" on the strength of 1 of 60 configs. More generally, any D_unexposed > 0 means the layer-10 position-0 write cannot explain those configs at all (they are gated identical). "Explains the gap" then overclaims.

**Fix.** Pre-register two things:
- **A minimum exposed-D size** (for example 10). Below it, the result is descriptive only.
- **A coverage condition for the full verdict:** "explains the gap" needs D_unexposed = 0, or moved/|D| ≥ 0.8. Otherwise the label is "explains the exposed part (moved/|D| = k/n)".

### 3. The "otherwise" bucket hides failures of the placebo and N_new conditions, and the threshold has a floating-point edge
**Where:** `decide` `:107-112`; pre-registration `:92-94`.

**The problem.** With r ≥ 0.8 and r − r_placebo < 0.5, the verdict is "explains part of the gap (r = 1.00)". I confirmed this with placebo_r = 0.6 and with placebo_r = 1.0. That label is wrong: the result cannot be told apart from boundary fragility, which is exactly the simplest hypothesis the pre-registration writes down first. Likewise, r ≥ 0.8 with N_new > 4 becomes "explains part" instead of a broad-perturbation label.

**Fix.** Add two labels:
- "not distinguishable from fragility", when r − r_placebo < 0.5;
- "perturbs broadly", when r ≥ 0.8 and N_new > 4.

**Floating-point edge.** `r - placebo_r >= 0.5` fails at exact ties. For example, 19/20 − 9/20 evaluates to 0.4999… I found 47 such (moved, moved_pl, n) cases with n ≤ 60. Compare integers instead: `2*(moved - moved_pl) >= n_De`. The 0.8 and 0.2 thresholds have no float problem for n ≤ 60.

### 4. donorA100 and a100clean0 are judged by the keep0/inject rule on A100-exposed D
**Where:** `:266-272`; pre-registration `:57-59`, `:96-97`.

**donorA100.** The pre-registration says it is "descriptive: r on all of D". The code gives it a verdict label and the placebo threshold, and computes r on exposed D only. donorA100 replaces every position and every block, so exposure does not apply to it. Report r on all of D and give it no verdict label.

**a100clean0.** It writes clean0_A. When it runs (that is, when `buggy0_L ≠ buggy0_A`), it differs from the L40S copy wherever clean0_A ≠ clean0_L, and those coordinates can be in blocks outside the A100 d0 set. Its r on A100-exposed D therefore leaves out configs it does change. The pre-registration gives it no rule at all.

**Fix.** Pre-register a rule for a100clean0:
- descriptive, with r on all D;
- or exposure redefined as the blocks where `clean0_A != clean0_L`.

Then make the code follow it.

### 5. Gates that the pre-registration lists but the code does not enforce
**Where:** `:214`, `:217-222`; pre-registration `:69`, `:75`, `:81`.

**The mismatch.** The pre-registration heading says "Gates (all recorded; any failure makes the job exit 2)". Under it, it lists the L40S d0 = 0 premise and inject exactness. The code only records them.

**L40S d0.** Round 1 asked for a nonzero L40S d0 to be flagged as a gate failure. If d0_L ≠ 0 in the exposed blocks, then:
- the L40S copy already carries its own position-0 difference;
- inject (built on buggy0_L) removes it before adding dA;
- r_inject is no longer the reverse test.

d0_L ≠ 0 in unexposed blocks would be caught by the unexposed gate. d0_L ≠ 0 in exposed blocks would not be caught.

**Fix.**
- Append `run["pos0_n_nonzero"] == 0` to `gates_ok` on L40S, or mark the inject and placebo verdicts as not interpreted.
- Do the same for `inject_check["exact_delta"]`.
- Edit the pre-registration so the heading and the list agree.

### 6. The exploratory D′ verdict uses a placebo r computed on D
**Where:** `:264`, `:271-272`.

**The problem.** `placebo_r=pr["r"]` comes from R13's D, while the variant's r comes from D′. The difference r − r_placebo then compares two different sets.

**Fix.** Compute `decide(out["placebo"], copy, ref_other, Dx, exposed)` and pass its r.

---

## Exit paths and the job-2 start condition

- **No data-loss paths found.** Each variant's results are written atomically as they finish. `run.json` is dumped after each cell. A failed copy gate does not stop the run early. The premise-absent exit dumps `run.json` before exiting. `torch.load` with the torch ≥ 2.6 default `weights_only=True` loads a dict of tensors without trouble. Every field written to JSON is serializable.
- **The job-2 start condition cannot wrongly pass:**
  - `afterok` and `all_gates_ok` are written only at the end of the run.
  - `pos0_n_nonzero > 0` is checked.
  - The delta path is fixed by the job ID.
- **It can block L40S when it arguably should not.** A failed A100 copy gate, where the pre-registered fallback is D′, also blocks the whole L40S arm. inject, placebo and donorA100 depend only on `position0.pt` and on the A100 passing capture vs live. This is consistent with the pre-registration, but consider gating L40S only on premise, capture vs live and unexposed-identical.
- **A blocked L40S job stays pending** with `DependencyNeverSatisfied` and must be cancelled by hand.
- **Walltime is enough:** L40S runs at most 10 cells × ~645 s (R13 L40S) ≈ 108 min plus model load, within 3 h.

## Optional notes

- The `outside_flips_to_bug` count (`:101`, `:115`) is really "correct → not correct". A not-correct text need not be a bug text, so rename it `to_not_correct` or count `is_bug`. Under the hypothesis, the expected direction is correct → not correct for keep0 and the reverse for inject. Stating that in the pre-registration would help.
- The "shown from both sides" rule (pre-registration `:95`) combines two jobs, and no code computes it. Say where it is computed: a small post-hoc script, or by hand from the two `run.json` files.
- When `capture_vs_live.buggy.pos0_equal` is false, set `interpreted: false` on the keep0 verdicts too. At present only the exit code shows it.
- On the premise-absent exit (`:199-202`), save `position0.pt` anyway for diagnosis.
- Record `store[0].shape` in `capture_vs_live`, and assert `N_BLOCKS*BLOCK == p.n_heads*p.head_dim`.
- The smoke test never ran with a non-empty D or the a100clean0 branch. A smoke run with synthetic R13 reference JSONs that put a few configs in D would exercise `decide`'s non-empty branches end to end.

Files reviewed:
- `/Users/gus/github/MATS/docs/reviews/11-22_llama8b_position0_gpu_gap_design.md`
- `/Users/gus/github/MATS/docs/experiments/11-22_llama8b_position0_gpu_gap.md`
- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R16_position0/run_experiment.py`
- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R16_position0/run_a100.sbatch`
- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R16_position0/run_l40s.sbatch`
- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R16_position0/smoke_cpu.log`
- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R13_true_head_patching/run_experiment.py`
- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R13_true_head_patching/results/r13_24014365/` and `r13_24084222/`

The test scripts are in `/private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/`: `t1.py` (`ulp_step`), `t2.py` (placebo against copy) and `t3.py` (`decide`).
