Reviewer: fresh subagent (Claude Code)
Agent-ID: ac7f8ddfb0fe1912f
Verdict: FIXES NEEDED
Transcript-SHA256: 98f5cea347e95ccdddb3bbb7f5bc3f31bbc251d7fb0f10a0cc997c778df3b296

## Brief given

You are a design reviewer for a planned GPU experiment. You did not write it and have no stake in it. Review it adversarially BEFORE any compute is spent. Do not edit any repo files and do not run git commit/push; you may only read files and run read-only commands (you may write scratch files under /Users/gus/github/MATS/outputs/ if needed).

Experiment file: /Users/gus/github/MATS/docs/experiments/11-21_llama8b_decimal_true_head_patching.md
Script: /Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R13_true_head_patching/run_experiment.py
Job file: /Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R13_true_head_patching/run.sbatch
Prior scripts/results it cites: /Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R8_threshold_analysis/ (run_experiment.py and results_20260314_091723.json), /Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R1_arbitrary_heads/, /Users/gus/github/MATS/9_8_research/working_scripts/test_minimal_even_heads.py
Paper draft it would feed: /Users/gus/github/phd_thesis/preamble/surgeon_dilemma/sections/02_study.tex
Repo context: /Users/gus/github/MATS (CLAUDE.md, docs/research_journal.md, docs/decisions.md). The shared methodology is /Users/gus/github/research_template/global/research-workflow.md (see "Design review" and "Skepticism checklist").

Checklist (answer each explicitly):
1. Fact-check every number in the experiment file against its cited source files (e.g. R8 counts 202/500 and 297/500, config counts, seeds). Flag anything untraceable.
2. Recompute all arithmetic (config counts per cell, total generations, GPU-hour estimate vs. walltime).
3. Terminology/sign consistency (repair, correct, bug, success; "pre_wo"/"post_wo"; "prefill"/"all_steps").
4. Completeness: is every "control", "matched", "reproduction" term operationally defined? Does the script actually implement what the file claims (e.g. does build_configs really reproduce R8's subsets in order; does the pre_wo hook truly isolate heads for Llama-3.1-8B with GQA; does "prefill" really patch only the prompt pass under HF generate with KV cache; does the identity control test what it says)?
5. Circularity: does any decision rule reference a quantity not independently defined, or does the screening criterion equal the confirmation criterion?
6. Category check: are cited facts from the right experiment/dataset?
Also: is the matching control pre-registered? Are the bug A / bug B claims about the old scripts correct when you read the old code? Any failure mode that would make the corrected cell's result uninterpretable?

End your report with a single verdict line: "VERDICT: PASS" or "VERDICT: FIXES NEEDED" or "VERDICT: REJECTED", followed by a numbered list of required fixes (if any) and optional suggestions, kept separate.

## Report

## R13 design review: true per-head patching at L10 (Llama-3.1-8B)

**Bottom line:** The core mechanics are sound. I checked on a CPU stand-in: a small randomly initialised Llama (12 layers, 32 query heads, 8 KV heads, run on the real Llama-3 tokenizer, script at `/Users/gus/github/MATS/outputs/r13_review/check_hooks.py`). On it the `pre_wo` hook isolates exactly one query head, "prefill" fires only on the prompt pass, and R8's subsets come back one-to-one. The design still has gaps. The paper's main parity control (matched spatial pairs) is missing. The random 8-subsets use a different seed than R1, so the paper's 13/200 can't be checked against them. Hypothesis 2 can't be settled by the planned comparison. Several gates have no machine-recorded outcome and no action defined if they fail.

### 1. Fact-check of numbers
- **R8 202/500 and 297/500:** both verified in `exp_R8_threshold_analysis/results_20260314_091723.json`. Every R8 rate is exactly 0.0 or 1.0, so applying `>=0.5` loses nothing. The script docstring's 40.4% and 59.4% match.
- **Seed 42 and R8 call order:** verified. `build_configs` regenerates all 500 of the 7-subsets and all 500 of the 8-subsets in order. All seven transition configs (base `[4,8,10,12,14,16,22]` plus {0,1,2,3,5,9,16} odd heads) match R8 exactly.
- **Duplicates in R8's subsets:** only 493 of the 7-subsets and 489 of the 8-subsets are distinct. The "1,000 matched subsets" therefore include 18 repeats. R8 counted them the same way, but the file should say so.
- **Seed 1234 for the random 8-subsets is new and matches nothing.** 0 of 200 match R1's `section1_random_mixed_8`. R1 used `RandomState(42)` over `range(32)`, and that call reproduces R1's 200 subsets exactly (I checked). The paper's "6.5% (13/200)" (`02_study.tex` l.123–124) comes from R1. With seed 1234 there is no subset-level reproduction of that number.
- **Config count 1,242 per cell:** verified, 3 + 32 + 500 + 500 + 7 + 200.
- **"~30 control runs":** the actual figure is 22 generations (2 baseline + 4 cells × (2 identity + 3 determinism)) plus 8 capture passes. Minor.
- **Untraceable:** "seen in the CPU smoke test". There is no log or results file, only a `__pycache__` (cpython-315). The claim that R8's hardware is unrecorded is consistent with the JSON, which has no provenance.
- **Not checkable from here:** account `torch_pr_173_tandon_advanced`, partition `l40s_public`, and whether the weights are pre-downloaded.

### 2. Arithmetic
- **Total generations:** 1,242 × 4 = 4,968, plus about 22 controls, so about 4,990. The file's "≈5,000" is fine.
- **Speed:** R8's own timing was 24,150 s for 50,350 generations, about 0.48 s each including hook overhead, on unknown hardware.
- **GPU hours:** at that speed R13 needs about 0.67 GPU-h. The file's 0.7–1.4 GPU-h is consistent.
- **Walltime:** 4 h is at least 2.8× the high estimate. Adequate.

### 3. Terminology
- **"Classified bug" has no operational definition.** The phenomenon gate requires the unpatched buggy prompt to be "classified bug", but `is_correct` only returns correct-and-not-bug. "Not correct" covers both "bug" and "neither".
- **Mixed up terms:** "repair", "correct" and "success" are used interchangeably. Pick one and define it as `is_correct == True`.
- **"Mixed-parity 8-subsets" are really uniform random 8 of 32.** Under seed 1234, 2 of the 200 have 7 even heads. Call them "random 8-subsets of all 32 heads", as R1 does.
- **`post_wo` "heads":** in the `post_wo` cells these are 128-wide blocks of residual coordinates, not heads. Label them that way in outputs.
- **`prefill` / `all_steps`:** used consistently. `prefill` is correctly implemented as "first hook call". I confirmed the `o_proj` input shapes in order: `(1,19,·)` on the first call, then `(1,1,·)` on each of the 19 decode steps.

### 4. Completeness, and whether the script does what the file claims
- **`pre_wo` with GQA:** correct. The `o_proj` input is `[b, s, 32·128]` laid out head by head. In the test, patching head 5 changed only slice 5. Note that query heads 4g..4g+3 share KV head g, so at head level "even vs odd" means within-group positions {0,2} vs {1,3}.
- **All-heads equivalence:** an all-heads `pre_wo` patch equals an all-coordinates `post_wo` patch in `prefill` (verified; `o_proj` has no bias). In the real run, `all_heads` should therefore give the same result in `pre_wo/prefill` and `post_wo/prefill`. This free consistency check is not pre-registered.
- **The identity control cannot detect head mis-indexing.** It replaces all heads with their own values, so any permutation also passes.
- **What the identity control can detect:** the BOS activation is bit-identical for the clean and buggy prompts (verified; position 0 is causal and attends only to itself). So bug B injects a prompt-independent constant. In the `all_steps` cells, the identity run is exactly "bug B alone" for all heads. The script saves that output but never classifies it.
- **Hypothesis 2 is not tested as written.** The file states it as "the BOS overwrite, not the prompt patch, flips the answer". Comparing `all_steps` with `prefill` only detects an interaction (prompt patch plus BOS injection). It cannot show that BOS injection alone flips the answer. That needs a decode-only condition: `all_steps` scope with the buggy run's own saved activations, per config.
- **Gates are not recorded in the results.** The baseline (phenomenon gate) and the `compare_to_r8()` output are printed to the log only. Neither is written to any results JSON. Nothing in the job stops a later step when a gate fails.
- **The R8 gate can be silently skipped.** `compare_to_r8` returns `None` if the R8 JSON is not present in the scratch copy.
- **The failure actions are not stated.** The file says what the identity (`prefill`) and determinism checks must show, but not what to do if they fail.
- **Stale results can be mixed in.** `run_cell` reuses any existing `results_<tag>.json` as already-done configs, with no provenance check. A `--limit` smoke run or an earlier run on other hardware in that folder would silently merge with the new results.
- **The commit hash can be missing.** `R13_GIT_SHA` defaults to `"unset"`, and the job does not fail when it is unset.

### 5. Circularity
- **No circular decision rule.** In the corrected cell, the even-16 hypothesis is fixed in advance and tested in a new basis (actual heads).
- **The reproduction gate is a legitimate environment check.** It is a different quantity from the decision rules.
- **Screening and confirmation share one prompt pair.** H\* was found on this pair and is confirmed on it. Any positive result is in-sample with respect to prompts. That is acceptable only if the paper says so.
- **Even-16 vs odd-16 is one comparison each.** Rule (c) carries the statistical weight.

### 6. Category check
- **Two of the paper's numbers come from R1, not R8.** The 13/200 random 8-subsets and the matched-spatial pairs (21/30 even pairs succeed, 0/30 odd) are R1 results. R13 drops the matched control and changes the seed. As designed, the paper's "Matched-spatial control" paragraph (l.141–155) cannot be updated from R13.
- **v1's script used a different classifier.** `test_minimal_even_heads.py` has fewer patterns and no first-line fallback. R13 uses R8's classifier, which is fine, but don't compare R13 numbers directly with v1 numbers.

### Is the matching control pre-registered?
Only partly. The random 8-subsets are a size-matched comparison set. The parity control the paper actually relies on is R1 section 4: 30 even 8-subsets, each paired with the same set shifted by +1. That control is not in R13. Neither is R1's set of odd-only subsets at k = 8–16, which backs the paper's "no complementary subset succeeds at any size" claim. At head level the +1 shift keeps each head in the same KV group, so it is a well-matched control.

### Are the bug A and bug B claims correct?
Yes. R8, R1, `test_minimal_even_heads.py` and `validate_even_heads.py` all forward-hook `layers[10].self_attn`, which is post-`o_proj`. They reshape to `[32, d/32]` and keep the patch active through `generate()`. On decode steps, `min_seq_len = 1` writes clean position 0 into the patched blocks. One refinement: bug B overwrites only the patched blocks, not the whole activation, and what it writes does not depend on the prompt.

### Failure mode that could make the corrected cell uninterpretable
The buggy prompt has 19 tokens and the clean prompt 17. Left alignment pairs buggy "Q" with clean "Which" and so on, and leaves the buggy prompt's last two positions ("A", ":") unpatched. That is where the answer is produced. Under `prefill` only, a null result in `pre_wo/prefill`, even for `all_heads`, could come from this misalignment rather than from bug B. Before R13, `all_steps` hid this, because the decode positions received BOS. The rule "if all-heads fails → the L10 headline depended on bug B" then over-reads a negative result.

### Aside, outside this review
Two places in the paper draft don't match the code or data:
- `02_study.tex` l.58 says "Hooks are implemented via nnsight". The scripts use HuggingFace hooks.
- l.152 says the paired difference is "+70.0pp on every pair". R1 shows 21 of 30 pairs at +100pp and 9 at 0.

VERDICT: FIXES NEEDED

Required fixes:
1. **Add the matched-spatial control.** Use R1 section 4's 30 even 8-subsets and their +1 shifts; hard-code the heads from R1's JSON. Also add R1's odd-only subsets (k = 8, 10, 12, 14, 16). Pre-register a decision rule on the pairs, for example "even succeeds and odd fails on ≥ X of 30", in the H\* rule.
2. **Draw the random 8-subsets with `RandomState(42)` over `range(32)`**, which reproduces R1's 200 subsets exactly (verified). Add a second reproduction check: in `post_wo/all_steps`, the result must agree with R1's 13/200, subset by subset. If seed 1234 is kept, the paper's 13/200 cannot be checked from this run.
3. **Test hypothesis 2 as stated.** Add a decode-only condition: `all_steps` scope patched with the buggy run's own saved activations, for every config or at least all_heads, even16, odd16 and the 1,000 R8 subsets (about 1,000–1,250 extra generations, roughly 10–20 min). Classify the identity outputs. Pre-register how to read it: decode-only success means bug B alone flips the answer.
4. **Define "classified bug"** (a bug pattern matched and not correct) or change the gate to "not correct". Settle on one term, "correct", for the outcome.
5. **Record every gate and define what happens on failure.** Write the baseline outputs, the gate verdict, the R8 agreement (and R1 agreement, from fix 2), and the identity and determinism results into the results JSON. Fail loudly if the R8 JSON is missing. State what a failed identity or determinism check in a `prefill` cell means (stop and escalate, as with the R8 gate).
6. **Pre-register the all-heads equivalence check:** `pre_wo/prefill` must equal `post_wo/prefill` for `all_heads` (exact string, or the same classification). This is the only check on the real model that ties the new hook to the old one.
7. **Address the alignment failure mode before reading a negative result.** Either add a right-aligned (suffix-aligned) `prefill` variant for all_heads, even16 and odd16, or reword the "L10 repair survives" rule. The reworded failure branch should read "fails under left-aligned overlap with the last two positions unpatched", not "depended on bug B".
8. **Stop stale results from merging in.** Refuse to start if `results_<tag>.json` already exists without a matching provenance and commit hash, or write to a run-specific file name. Make the job fail when `R13_GIT_SHA` is unset, or compute it with `git rev-parse` and record a sha256 of the script.
9. **Back or remove the "seen in the CPU smoke test" claim.** Attach the smoke-test log, or drop the claim and mark the `all_steps` identity failure as a prediction.

Optional suggestions:
- Add an automated head-isolation check before the grid starts: patch one head and assert that only that 128-slice of the `o_proj` input changes. My check script does this.
- In the `prefill` scope, assert that the first hook call has `seq_len == T_buggy` and every later call has 1.
- In the file, note the 18 duplicate subsets and the GQA grouping (query heads 4g..4g+3 share KV head g) when interpreting "parity".
- Add random 16-subsets of all 32 heads (for example 100, as in R1 section 2) to test whether even-16 is special at head level, not just better than odd-16.
- Pin the transformers version in the job environment. HF's attention code changes often, and the `o_proj` call must stay positional for `args[0]` to work.
- Correct "~30 control runs" to 22 generations plus 8 capture passes.
- Fix the two paper inconsistencies in the aside separately (nnsight vs HF hooks; "+70.0pp on every pair").
