Reviewer: fresh subagent (Claude Code)
Agent-ID: aa6a447261d6164fb
Verdict: PASS WITH MINOR FIXES
Transcript-SHA256: 8911b45d7dc346ca7b9a77f07999fdb541eef0460383cb9e0f1daba838aef533

## Brief given

You are a fresh, skeptical design reviewer (round 3) for a pre-registered mechanistic-interpretability experiment (R17) in ~/github/MATS. Read-only: do NOT edit files, do NOT ssh or run GPU jobs. You may run small local Python checks (`uv run -q --no-project --with torch --with transformers --with numpy python -I ...` works; a tiny random Llama with the real Llama-3 tokenizer is at /private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/tiny_llama3).

Read:
- Pre-registration: docs/experiments/11-23_llama8b_localization_map.md
- Script: 9_8_research/reviewer_experiments/exp_R17_localization_map/run_experiment.py, run.sbatch, smoke_cpu.log
- Prior reviews: docs/reviews/11-23_llama8b_localization_map_design.md (round 1, 12 findings) and docs/reviews/11-23_llama8b_localization_map_design_v2.md (round 2, 6 findings). Verify that the round-2 findings were addressed correctly in BOTH script and prereg and that no new bugs were introduced (the sweep was moved into a separate function `sweep()` wrapped in try/except; the final-norm patch target was added; the decision rule 2 condition changed).
- Context: R17 is stage 1 of a search for what repairs Llama-3.1-8B's 9.8-vs-9.11 format bug under correct activation patching (earlier head-level claims were artifacts of a buggy hook; see docs/experiments/11-21_llama8b_decimal_true_head_patching.md).

Focus on: code/prereg consistency, gates that can fail spuriously or pass vacuously on the real 32-layer fp16 A100 run, correctness of decide() and the controls, and whether the label can be reached by a trivial effect. Don't relitigate design choices that previous rounds settled unless you find a concrete defect.

Output: first line exactly one of PASS, PASS WITH MINOR FIXES, FIXES NEEDED. Then numbered findings with file:line, severity (blocking / should-fix / minor), and a concrete fix. Be concise.

## Report

PASS WITH MINOR FIXES

Files: the script is `~/github/MATS/9_8_research/reviewer_experiments/exp_R17_localization_map/run_experiment.py` and the prereg is `~/github/MATS/docs/experiments/11-23_llama8b_localization_map.md`.

**Round-2 findings, checked in both the script and the prereg.** Five of the six are fixed as specified. #6 is only partly fixed (finding 2 below).
- **#1:** the repair veto is now `random_dir` only (`:512`, `:517`). The shift controls are descriptive, run only on single-position cells, and set `position_specific = max(shift) < r/2` (`:498-503`). The prereg matches.
- **#2:** the donor-capture comparison is descriptive, normalized by the resid norm (`:405-408`). Generation takes its donor values from `cap_d`; under causal attention that is correct for prompt positions.
- **#3:** a positive control that holds by construction (final-norm input at the last position, r ≥ 0.99) is now the gate (`:441-443`). The L31 `answer_all` check is reported only (`:445-447`). The `final` pre-hook gets the positional arg, which matches `self.norm(hidden_states)`. Smoke r = 1.0.
- **#4:** `near_input` is added (`:214-216`, `:475`). Rule 1 excludes it (`:240`), and there is a selftest for it.
- **#5:** the stage-2 fallback now requires `control_max_r < 0.2` when the cell was checked; unchecked cells pass. This matches the prereg wording, and there is a selftest (`:263-264`, `:313-315`).
- **#6:** see finding 2.

**Other checks**
- The smoke log's sha256 (`883f5d82…`) equals the current script. R13's sha256 still matches `1ce5b0cc…`.
- `SystemExit` from the gate-9/10 failure path (`:527`) is not swallowed by `except Exception`, which is correct.
- Prompt patches at L31 cannot reach the last-position logits, so late-layer p16 cells cannot repair trivially through the final layer.
- I found no new logic bugs in `decide()` beyond finding 1.

**Findings**

1. **`run_experiment.py:244`, `:258`; prereg "Decision rules" rule 2 and stage-2 trigger. Should-fix.** A near-input cell can carry the "distributed" label and the stage-2 trigger.
   - Every attn/mlp `prompt_all` cell at L1–2 includes p1 and p15, so `near_input=True`.
   - Rule 2 (`any(c["set"]=="prompt_all" for c in rep)`) does not filter `near_input`. Neither does `attn_rep` (`:258`) nor the fallback candidates (`:263`).
   - So an mlp-L1 `prompt_all` patch can produce "distributed across prompt positions". That patch mostly swaps the identities of "Answer"→"A" and "Which"→" Which", which is the trivial effect the near-input flag exists to fence off. An attn-L1/L2 cell could also become a stage-2 layer.
   - The selftest at `:309-311` also shows the label text being false. A near-input single cell repairs, yet the label says "only all-position patches repair".
   - Fix:
     - Build `rep_nn = [c for c in rep if not c.get("near_input")]` and use it for rule 2, `attn_rep` and the fallback `cand`. Keep near-input repairs in `near_input_repairs`.
     - Change the label-2 text to "...only all-position patches repair (among non-near-input cells)".
     - Say in the prereg that near-input cells carry no label and no stage-2 trigger.
     - Add a selftest: an mlp-L1 `prompt_all` repair alone gives "no prompt-only patch repairs".

2. **`run_experiment.py:452-461`, `:466`, `:519`. Should-fix.** The round-2 #6 fix is incomplete: the cells measured before a crash are still lost.
   - `cells` is local to `sweep()` and is only attached to `run` at `:519`. On an exception (for example the `generate()` assert at `:184`/`:198` during a generation check), the except block dumps `run` without any cells. The smoke note shows this path ran once, but with a NameError, so there were no cells to lose.
   - Fix: at the top of `sweep()`, write `cells = run["cells"] = []`.

3. **`run_experiment.py:384-450`. Minor.** The gate phase is outside the try. `M.generate` with a patch (generation identity, `:438`) can raise `AssertionError`, which exits with code 1 and leaves `run.json` with provenance only.
   - Fix: wrap everything from `M = Model()` to `stop_if_failed()` in the same try/except. Alternatively, move the try to the top of `main()` after the first `dump`.

4. **`run_experiment.py:410-415` (gate 4). Minor, low risk.** The 1% decomposition tolerance can fail spuriously if a non-BOS position carries a large residual that a sublayer mostly cancels. fp16 rounding of the large terms would then be large relative to `||next||`.
   - I can't confirm this for Llama-3.1-8B without the model. Its massive activations are mainly at BOS, which the gate excludes.
   - Fix (removes the risk entirely): replay the layer's own fp16 order, `(resid[L] + attn[L]) + mlp[L]`, which matches the decoder layer's `residual + hidden_states` adds, and require `torch.equal` against `resid[L+1]`/`final`. Keep the fp32 1% number as a descriptive field.

5. **Prereg "Controls", misaligned donor. Minor wording.** The prereg says "both shifted r < r/2", but p1 and p16 have only one valid shift, and the code uses the one that exists (`:500-503`). Fix: write "every available shifted r < r/2 (p1, p16: one shift)".

The design is otherwise sound for the real run.
- The bitwise gates (capture equals plain, identity, hooks removed) are bitwise-safe in fp16: the patch writes the same values into a clone.
- The generation-identity gate compares tokens, not bitwise values, so small numeric differences between the `cap_rp` capture and `generate()`'s prefill are tolerated.
- The worst-case generation-check load (about 1,600 cells × (1 generate + 4 passes)) fits in the 2 h walltime.

Findings 1 and 2 should be fixed before submission. Findings 3–5 are optional.
