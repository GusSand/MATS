Reviewer: fresh subagent (Claude Code)
Agent-ID: a567afcbbb86f3ef7
Verdict: FIXES NEEDED
Transcript-SHA256: 822840e931fbced5e8307d9048d3ef67473a0b680996794a4253541753883ad1

## Brief given

You are a fresh, skeptical design reviewer (round 2) for a pre-registered mechanistic-interpretability experiment (R17) in ~/github/MATS. Read-only: do NOT edit files, do NOT ssh or run GPU jobs. You may run small local Python checks (no torch is installed system-wide; `uv run -q --no-project --with torch --with transformers --with numpy python -I ...` works, and a tiny random Llama with the real Llama-3 tokenizer is at /private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/tiny_llama3).

Read:
- Pre-registration: docs/experiments/11-23_llama8b_localization_map.md
- Script: 9_8_research/reviewer_experiments/exp_R17_localization_map/run_experiment.py, run.sbatch, smoke_cpu.log
- Round-1 review: docs/reviews/11-23_llama8b_localization_map_design.md (12 findings). Check that each was addressed correctly in both the script and the pre-registration, and that the fixes did not introduce new bugs.
- Context: R13 (docs/experiments/11-21_llama8b_decimal_true_head_patching.md; script 9_8_research/reviewer_experiments/exp_R13_true_head_patching/run_experiment.py). Earlier head-patching claims about Llama-3.1-8B's 9.8-vs-9.11 format bug came from hooking post-W_O attention output reshaped into "heads" and leaving the hook active during generate(). R17 is stage 1 of a search for what DOES repair the bug: a component × layer × position activation-patching map with a teacher-forced 8-vs-11 logit metric, generation confirmation, controls.

Check especially: code/prereg consistency; that every gate can actually fail when it should; the controls' implementation (misaligned shift, random direction norm, reverse-direction positions); the input-substitution exclusion; decide() vs the written rules; anything that would make results uninterpretable on the real 32-layer fp16 A100 run (e.g., a gate likely to fail spuriously in fp16, such as bitwise identity or the 1% decomposition/donor-capture tolerances, or the positive control threshold).

Output: first line exactly one of PASS, PASS WITH MINOR FIXES, FIXES NEEDED. Then numbered findings with file:line, severity (blocking / should-fix / minor), and a concrete fix. Be concise.

## Report

FIXES NEEDED

Paths are relative to `~/github/MATS`. "Script" is `9_8_research/reviewer_experiments/exp_R17_localization_map/run_experiment.py`; "prereg" is `docs/experiments/11-23_llama8b_localization_map.md`.

**Round-1 findings: all 12 are addressed, in both the script and the prereg.**
- **#1 no-op patch:** three new gates cover it. `patch_has_effect` (script:433), `active_patch` (script:397-404) and `generation_identity` (script:406-411). The probe is registered after the patch, and PyTorch hands the modified args/kwargs and outputs on to later hooks, so the probe sees the patched tensor. If patches did nothing, every Δ would equal Δ_buggy exactly and gate 10 would fail.
- **#2 input substitution:** `input_substitution()` (script:208-213) matches the prereg text.
- **#3 fourth token and argmax:** both are checked (script:362-363, 377-378). I confirmed with the real Llama-3 tokenizer that " 9.8" / " 9.11" tokenize to [220,24,13,23] / [220,24,13,806]. Donor position 8 is "8", 13 is "11", 15 is "Answer", 16 is ":".
- **#4 flip threshold:** generation is triggered on `r>=0.5 or delta>0`, and `r_flip` is recorded.
- **#5–#12:** controls, reverse direction, positive control, per-position non-BOS decomposition including the `final` capture, the donor-capture gate, report caveats, selftest branches, the fallback flag and the scope text are all present.
- **Other checks:**
  - R13's sha256 matches `1ce5b0cc…`, and R13 guards its main block.
  - The reverse-direction mirror is correct (`pm = {dp: cap_r[..][dp+OFFSET]}`, run on `d_ids`).
  - The random-direction norm is the per-position ‖donor − recipient‖, which is correct.
  - `decide()` implements the written rules line for line.
- **Bitwise gates are low risk in fp16 (identity, capture_equals_plain, hooks_removed).** The patch writes identical values into a clone on the same deterministic kernels. R13's exact-identity gates passed on A100.

**Findings**

1. **The shift control vetoes real repairs, which biases the headline label toward the null (script:453-455, 464, 469; prereg "Controls" and the repair rule). Severity: blocking.**
   - The ±1 misaligned donor is not a null control for the cells most likely to matter.
   - At mid and late layers, the format difference ("Answer:" vs "A:") sits at both donor 15 and donor 16.
   - For `p16`, only shift −1 runs, and it writes donor 15's ("Answer") activation at the ":" position. That probably carries the same format signal, so control r ≥ 0.2 and the cell is vetoed.
   - For `prompt_all`, a ±1 shift keeps almost all the donor's information, only moved by one position. So for attn/mlp the "distributed" label and the attn stage-2 trigger are close to unreachable.
   - The result would read "no prompt-only patch repairs", which is exactly the kind of null R13 warned about.
   - Fix (prereg and code):
     - The repair veto uses only `random_dir < 0.2`.
     - Shift controls are recorded as positional specificity. Run them only for single-position cells, and report them per cell, e.g. a cell is "position-specific" if max shift r < r/2.
     - Optionally, add a same-format null donor (a Q:/A: paraphrase whose answer is also the bug) as a second veto.

2. **Gate 4 (donor captures agree) can fail spuriously in fp16 and abort the whole run (script:379-380, 314-317). Severity: should-fix.**
   - The error is divided by each component's own per-position norm. attn/mlp outputs, especially at early layers, can have a norm far below the residual's.
   - Sequences of length 17 and 20 can get different GEMM/SDPA tilings, so the shared positions are not guaranteed to be bitwise equal. Any small-norm position then inflates the error over a max of 3×32×16 values.
   - Fix (either works):
     - Normalize by the residual norm at that position and layer (the scale the patch is written into).
     - Or take the generation donor values from `cap_d[c][L][:nd_prompt]`, which is mathematically identical under causal attention, and make gate 4 descriptive.
   - Gate 5 already normalizes by the resid norm and is fine. Its fp16 add-rounding error is about 1e-3 even with some cancellation.

3. **The positive control is a judgment call, not true by construction (script:413-415; prereg gate 9). Severity: minor.**
   - Patching the resid input to layer 31 at the answer positions still leaves layer 31's attention reading the recipient's K/V at the prompt positions. r is very likely above 0.9, but if it lands below 0.8 the whole run is lost.
   - Fix: add a second control that holds by construction: patch the final-norm input (`model.model.norm` pre-hook) at the last position, require r ≥ 0.99, and gate on that. Keep the L31 `answer_all` r ≥ 0.8 as a softer check, or lower it to 0.5.

4. **The input-substitution boundary is uneven (script:208-213). Severity: minor.**
   - resid is excluded at p15 at every layer, but mlp/attn at p15 are counted from layer 1 onward. Early MLPs at p15 still mostly encode "Answer" vs "A" token identity.
   - Fix: add a sentence to the prereg saying that mlp/attn cells at p1/p15 with L ≤ 2 are flagged as "near-input" in the report. They stay counted, but cannot carry the localized label on their own.

5. **The stage-2 fallback ignores controls (script:250-253). Severity: minor.**
   - The top-3 attn `prompt_all` layers by r ≥ 0.2 are chosen without looking at `random_dir`.
   - Fix: also require `control_max_r < 0.2` (after fix 1, the random-direction r) whenever the cell was checked. Otherwise state in the prereg that controls are not applied there.

6. **A failed assertion crashes the job instead of exiting 2 (script:178, 192). Severity: minor.**
   - An `AssertionError` in `generate()` mid-sweep raises with a traceback and exit code 1, and the cells measured so far are never dumped.
   - Fix: wrap the sweep and generation phase in try/except, dump `run.json` with the error, then `sys.exit(2)`.
