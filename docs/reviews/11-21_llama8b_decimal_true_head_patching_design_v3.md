Reviewer: fresh subagent (Claude Code)
Agent-ID: a55686677e88dd02e
Verdict: PASS
Transcript-SHA256: 9a04ba4e3cf767e7d65783eee7ac13e1527fb0bf1cc6cfab0a5d16c409b388f4

## Brief given

You are a design reviewer for a planned GPU experiment (round 3, after a second amendment). You did not write it and have no stake in it. Review adversarially before compute. Do not edit repo files, do not run git commit/push; read-only commands only (scratch files only under /Users/gus/github/MATS/outputs/).

Experiment file: /Users/gus/github/MATS/docs/experiments/11-21_llama8b_decimal_true_head_patching.md
Script: /Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R13_true_head_patching/run_experiment.py
Job file: same folder, run.sbatch. CPU smoke log: same folder, smoke_cpu.log.
Round-2 review (required fixes and optional suggestions at the end): /Users/gus/github/MATS/docs/reviews/11-21_llama8b_decimal_true_head_patching_design_v2.md
Round-1 review: /Users/gus/github/MATS/docs/reviews/11-21_llama8b_decimal_true_head_patching_design.md
Reference data: /Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R8_threshold_analysis/ and exp_R1_arbitrary_heads/
Methodology: /Users/gus/github/research_template/global/research-workflow.md ("Design review").
CPU python: /private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/r13env/bin/python (run with -I); random 32-head Llama at .../scratchpad/tiny32 (env R13_MODEL=<path>, R13_GIT_SHA=anything, flag --smoke). A cached Llama-3 tokenizer may exist under ~/.cache/huggingface (allenai/Llama-3.1-Tulu-3-8B-SFT).

Tasks:
A. For each round-2 required fix (1: right alignment must not write clean BOS into a content position; 2: reproduction gate evaluated as a boolean with per-criterion values and run.json persisted after each cell; wording "any threshold"; odd-only gated or reported) state RESOLVED / PARTIAL / NOT RESOLVED with evidence, verifying the right-alignment mapping at the real 17/19 Llama-3 token lengths.
B. Check that the file and script agree everywhere the amendment touched (primary-cell designation, hypothesis-2 negative branch, head-isolation gate, BOS diff record, all-heads equivalence first-token record, --smoke flag semantics, compute arithmetic, capture-pass counts).
C. Any new bug introduced by this amendment, and anything that would make the primary cell's result uninterpretable.

End with one verdict line: "VERDICT: PASS" or "VERDICT: FIXES NEEDED" or "VERDICT: REJECTED", then required fixes (if any) and optional suggestions, separately.

## Report

## R13 design review, round 3

I made no repo edits. My only scratch file is `/Users/gus/github/MATS/outputs/r13_v3_align.py`. The CPU smoke log was produced by the current script: the `script_sha256` in the smoke `run.json` matches `run_experiment.py`, `1ce5b0cc…348d`.

### A. Round-2 required fixes

**Fix 1, right alignment: RESOLVED.** `_swap` (`run_experiment.py:142-147`) sets `n = min(sl, ts) - 1` and writes `hr[:, sl-n:] = sr[:, ts-n:]`. I checked it with the cached Llama-3.1 tokenizer (Tulu-3-8B-SFT): the clean prompt is 17 tokens and the buggy prompt is 19. I ran `_swap` on position-tagged tensors and got this mapping:
- Buggy positions 0, 1, 2 (BOS, `Q`, `:`) are left unpatched.
- Buggy 3..18 ← clean 1..16. This is token-aligned (`ĠWhich`←`Which`, …, `?Ċ`←`?Ċ`, `A`←`Answer`, `:`←`:`).
- No clean BOS lands in a content position.
- With head [5], only head 5 changes.
- In the identity control, buggy is written into buggy at positions 1..18, so it stays exact.

The file's description (`Right alignment pairs the last 16 non-BOS positions …`) matches.

**Fix 2, reproduction gate: RESOLVED.**
- `reproduction_gate()` (`:256-266`) returns a boolean `reproduction_gate_ok` with per-criterion `values` and `thresholds`: 950 / 190 / 57.
- It also requires that all 1,260 reference subsets are matched by head set (`all_heads_matched`).
- Odd-only results are kept separately as `oddonly_reported_not_gated`.
- `run.json` is rewritten after every cell (`:369-370`).
- The file now says "Below any threshold" and "R1's 5 odd-only subsets are reported, not gated".
- The smoke `run.json` contains all of these fields.

I also confirmed that the R8 and R1 protocols match the `post_wo/all_steps` cell: same prompts, fp16, `self_attn` forward hook, `[:, :ml]` left slice, 20 greedy tokens, same classifier.

### B. Does the file agree with the script?

- **Primary cell.** The file names `pre_wo/prefill_right` as primary and `pre_wo/prefill` as secondary, both fixed in advance. This is a file-level designation and nothing in the script contradicts it. The script docstring does not mention it, which is fine.
- **Hypothesis-2 negative branch.** Present ("Otherwise: BOS injection alone does not produce the repair"). `decode_only` uses `saved = buggy_saved` with scope `all_steps`, which matches the rule.
- **Head-isolation gate.** It matches: fail → exit (`:354-356`). The file says it patches "into the real model's o_proj input", but the check only runs `_swap` on captured real `o_proj` inputs. It does not run a hooked forward pass, and it tests only left alignment. The wording slightly overstates what the check does.
- **BOS diff record.** Matches, for both hooks, recorded and not gated.
- **All-heads equivalence first-token record.** Present (`same_text`, `same_first_token`, `same_class`). The first token is taken by re-tokenizing the decoded text, not from the generated ids, so it is approximate (see the suggestions).
- **`--smoke` flag.** Not contradictory, but not documented. The script's only effect is that the phenomenon gate is recorded but not enforced, and `--limit` has the same effect, which the help text does not say. The file mentions neither flag's effect on gating, only "no `--limit`" for the smoke run.
- **Compute arithmetic.** Checked and correct:
  - Configs: 1,307 × 6 = 7,842.
  - Control generations: 6 cells × 5 (unpatched, identity, 3× determinism) + 2 baselines = 32, giving 7,874 generations in total.
  - Capture passes in cells: 5×2 + 1 = 11 (`decode_only` captures only the buggy prompt).
  - Head isolation: 2 capture passes. BOS check: 4 passes.
  - Time: 7,874 × 0.48 s ≈ 1.05 h; doubled is 2.1 h, inside the 4 h walltime.

### C. New bugs and threats to interpreting the primary cell

- I found no new correctness bug in the amended code.
- Prompt-only scope, shape, identity and determinism gates behave as specified. The smoke run had identity True in the three prompt-only cells and False in the all-steps and `decode_only` cells, as expected.
- **Primary cell.** A null result there would now be clean: there is no BOS injection and the alignment is token-aligned. The "L10 repair fails" branch is pre-registered, so an all-heads failure is still an interpretable result.
- **Small gaps** (none of them blocks the primary cell):
  1. No automated check targets the right-aligned path. The all-heads equivalence compares only the two left-aligned prompt-only cells, and the identity control cannot detect an offset error because both lengths are 19. Correctness depends on the mapping verified above and on the token lengths staying 17/19. The script neither asserts nor records those lengths.
  2. `head_isolation` and `bos_max_abs_diff…` are written to `run.json` only after cell 1, or on an isolation failure. If cell 1 crashes, they are lost.
  3. If a prompt-only gate aborts the job, `run.json` does not record it; only the cell JSON does.
- **Operational issue.** The script, the experiment file and `docs/reviews/` are untracked in the MATS repo. `R13_GIT_SHA` must be a SHA that actually contains this script, so commit before you submit.

VERDICT: PASS

**Required fixes:** none.

**Optional suggestions:**
1. Commit the R13 folder and the experiment file before setting `R13_GIT_SHA`. They are currently untracked. Make sure `/scratch/$USER/MATS/logs` exists before you submit.
2. Record and assert the token lengths (clean 17, buggy 19) and the right-alignment position map in `run.json` or the cell JSON. The primary cell then documents its own alignment.
3. Call `save(run)` right after the head-isolation and BOS checks, and record an aborted cell in `run.json` before `sys.exit`.
4. For the equivalence diagnostic, record the first generated token id, or the max abs difference of the first-step logits, instead of re-tokenizing the decoded text.
5. Reword the head-isolation description to "tensor-level check on captured `o_proj` inputs". Optionally also test head 5 with `align="right"`.
6. Add one line to the file on `--smoke` and `--limit`: the phenomenon gate is recorded but not enforced, and nothing else changes. Optionally refuse `--smoke` when CUDA is available.
7. Optionally pre-register a tolerance for the BOS diff, so a nonzero GPU fp16 value has a stated reading for `decode_only`.

## Doer response

No required fixes. Optional suggestions 2–7 are deferred, not adopted: each would change the
reviewed code or design and need another round, and none affects the primary cell's validity
(the reviewer verified the right-alignment map at real 17/19 token lengths). Suggestion 1 (commit
before submitting; logs directory exists) is done. Deferred items are listed in BACKLOG.md.
