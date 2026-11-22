Reviewer: fresh subagent (Claude Code)
Agent-ID: aa9191c50cef479e7
Verdict: FIXES NEEDED
Transcript-SHA256: 0a340aa9e2e81ef04aaea38e85ab3e0f83050a66863ee5e585213daa02a96e03

## Brief given

You are a design reviewer for an experiment that has not run yet. This is round 2. Round 1 required fixes, and the design and script have since been revised. Check whether the fixes are correct and complete, and look for new problems. Flag only issues that could change a verdict, lose data, or make the pre-registration disagree with the code. Do not edit any files.

Repo: ~/github/MATS.
- Round-1 review: docs/reviews/11-22_mistral7b_true_head_patching_design.md. Read it first.
- Pre-registration: docs/experiments/11-22_mistral7b_true_head_patching.md
- Script: 9_8_research/reviewer_experiments/exp_R15_mistral_true_heads/run_experiment.py. It imports R14's script by path, ../exp_R14_cross_model_true_heads/run_experiment.py. All decision rules are computed in `decide()`.
- Job file: run.sbatch. Smoke log: smoke_cpu.log. Both are in the R15 folder. The smoke used a random tiny Mistral and covers the no-pair path and a forced-pair path.

You can test `decide()` without a model. Import the script with the R14 module available; importing needs numpy and the standard library only, and torch is imported lazily inside R14's Patcher. Then feed `decide()` synthetic run/res dicts. Do so for every rule:
- S and M1a/M1b;
- M2 in both readings, including the H22-excluded (c), the "carried by H22" branch, and the untestable Δ branch;
- M3 and M3b with the draft top-5 ranks;
- M4 and "H22 special";
- M5 with the random 5-layer control;
- M6 requiring *correct*;
- M7.

Check that each one matches the pre-registration text exactly, and that none can pass vacuously.

Also check:
- `parity_subsets`: counts, 262 per layer, distinctness, and the rule that mixed subsets contain both parities;
- `random_layer_sets`;
- snapshot pinning and its offline loading via `revision=<commit hash>` in transformers 5.x / huggingface_hub;
- the R14 sha check and the sbatch export;
- `hit_max_new` and `says_less`;
- the chat-template check fix (BatchEncoding);
- save points;
- compute and walltime (5 h on A100 for about 2,180 generations of 64 tokens, plus Δ passes).

Write your review as markdown. Start with a verdict, PASS or FIXES NEEDED, then required fixes with file:line, then optional suggestions. Return the full review as your final message; it will be saved verbatim.

## Report

# Design review round 2: R15, testing the Mistral-7B head claims with true per-head patching

**Verdict: FIXES NEEDED**

Most round-1 fixes are in and work. These include snapshot pinning, the R14 sha gate, the direct `R14_GIT_SHA` assignment, alignment recording, the H22-excluded subsets, the Δ readings with an "untestable" branch, M6 and S requiring *correct*, the count-matched controls, `hit_max_new` and the BatchEncoding fix. Five problems remain, and each can change a verdict or makes the code disagree with the pre-registration:

- M2's new (c) still passes on one success out of 60.
- M2's "carried by H22" label is assigned without any evidence about H22.
- M1a can pass because the patching does nothing, and M1b can pass even when every layer passes.
- The no-pair path never computes M7.
- M7 outputs don't record `says_less`, although the pre-registration says every output does.

All five are small code and text changes.

## What I checked and found correct

**`decide()` with synthetic run/res dicts.** I imported the script with R14 present (numpy and the standard library only) and tested every rule:

- **S.** Passes on the base case. It fails when qa and simple are both *neither*, and when comparative is *neither*.
- **M1a/M1b.** The `delta_testable` branch works. A baseline Δ > 0 or `None` gives "untestable" for M1b, M2-Δ and M3b.
- **M2.**
  - H22-only world: "even-only repair carried by H22" in both readings. Rates come out as `even_with_H22` 31/31 and `even_without_H22` 0/30.
  - Real parity (even and noH22 correct at k ≥ 8): "parity holds".
  - One odd success: "not supported".
  - Count-only world (all k = 16 correct): "not supported".
- **M3.** H22 alone at layers 2 and 17 passes. Adding one more correct head fails it. M3b ranks correctly, `draft_top5_ranks` is filled, and a tie with a lower-index head gives False, which is the conservative outcome.
- **M4 and "H22 special".** These match the pre-registration text.
- **M5.** Passes, and the random control is reported.
- **M6.** A *neither* reverse gives False, and M6 is False when M5 fails.
- **M7.** Counts are right, and zero eligible pairs gives False.

**`parity_subsets()`.**
- 262 per layer: even and odd 20/20/20/1, mixed 20×4, even_noH22 20×3.
- Subsets are distinct within each (kind, k) and every subset has the right size.
- Mixed subsets always contain both parities. even_noH22 never contains H22.
- The function is deterministic, and no even_noH22 set duplicates an even set.
- H22 appears in 5, 11, 14 and 1 of the even subsets at k = 4, 8, 12, 16.

**`random_layer_sets()`.** 10 distinct 5-layer sets, none equal to FIVE, overlapping FIVE by 0–2 layers. `gen_pairs()` gives 100 distinct pairs.

**Snapshot.** `63a8b081895390a26e140280378bc85ec8bce07a` is the real, current `main` of the public repo (HF API; lastModified 2025-07-24). I downloaded its config and tokenizer files only into a scratch `HF_HOME`. With `HF_HUB_OFFLINE=1` and transformers 5.12.1 / huggingface_hub 1.21:
- `AutoConfig`/`AutoTokenizer.from_pretrained(repo, revision=<hash>)` resolve offline. A download by hash writes no `refs/main`, and loading still works.
- `_commit_hash` is the full hash, so the `startswith(SNAPSHOT[:8])` check (`run_experiment.py:354`) passes.
- `sliding_window` is null, and `generation_config` has only bos/eos.

**Tokenizer, on the real v0.2 tokenizer under transformers 5.12.1:**
- **Chat-template check.** `apply_chat_template(tokenize=True)` returns a `BatchEncoding`. The fix at `:374` gives a flat list, and the string path and `tokenize=True` agree: `<s> ▁[ INST ] …`.
- **Token counts.** chat 23, comparative 18, qa 21, simple 20.
- **Δ prefix check.** Passes on all four formats, and on all 100 generalization pairs × 4 formats (0 failures).
- **Alignment.** simple→chat aligns exactly on the question; qa→chat is off by one, as the pre-registration states.

**R14 sha gate and sbatch.**
- The committed R14 (`d51b0ed`, equal at `18f3df2`) has sha256 `20819e06…`.
- `:348` exits on a mismatch unless `--limit` is set, and the sbatch requires the variable to be set.
- `:352` overwrites `R14_GIT_SHA`, so a stale shell value can no longer reach provenance.
- I ran the committed R14 `selftest()` locally. It passes: "late-training verdict (11 cases)".

**`hit_max_new`** (`:214`). True only when 64 tokens were generated and the last one is not EOS. That is correct.

**Save points.** The script saves:
- after the pair is chosen (before the gates);
- after the gates (before the gate exit);
- after the layer sweep;
- after each single-head layer;
- after each parity layer;
- after the M4–M6 block (about 81 generations, about 3 min);
- after each M7 pair.

`decide()` is pure, so it can be rerun offline after a crash.

**Compute.**
- Generations: 7 + 5 + 1 + 32 + 1,024 + 524 + 1 + 10 + 64 + 6 + 400 + ≤100 ≈ 2,175.
- At about 1.5–2.5 s per 64-token greedy generation on an A100 with HF `generate`, that is about 1–1.5 h.
- Δ needs 2 short forward passes for each of about 1,670 `rec` calls, which is a few minutes.
- 5 h leaves more than 3× margin.

## Required fixes

### 1. M2 (c) passes on 1 vs 0 (pre-reg `:87-89`, script `:279`)

When (b) holds, odd-only successes are 0, so (c) reduces to "at least one of 60 even_noH22 subsets succeeds". I checked this: an H22-only world plus one noH22 success gives "parity holds".

Under boring hypothesis 2 or H22-only, one sporadic success is not evidence: Fisher one-sided p for 1/60 vs 0/60 is about 0.5. This is the same redundancy round 1 flagged for the old (c).

**Fix:** pre-register a margin and code it. Either:
- one-sided Fisher exact p < 0.05 on noH22 vs odd (k = 4, 8, 12), which needs about 5/60 vs 0/60; or
- a fixed threshold, for example noH22 rate ≥ 10/60.

### 2. "Even-only repair carried by H22" is assigned without evidence about H22 (pre-reg `:89`, script `:280-281`)

The branch fires on (a) ∧ (b) ∧ ¬(c), and none of these involve H22. Test case: only the even k = 16 set succeeds, which means about 16 even heads are needed together and H22 plays no role. The verdict is still "carried by H22", with `even_with_H22` = 1/31 and `even_without_H22` = 0/30.

**Fix:** "carried by H22" also requires that even subsets with H22 at k ≤ 12 succeed more often than even_noH22 subsets (with the same margin as fix 1), or that M3 shows H22 alone correct at that layer. Otherwise the verdict is "even-only k = 16 repairs; H22's role not shown".

### 3. M1a can pass because patching does nothing, and M1b can pass when every layer passes (pre-reg `:78-79`, `:85-86`; script `:262-265`)

**M1a.** "No layer's all-heads patch is *correct*" is a pure null. It passes trivially if the intervention is ineffective. The sanity transplant (all layers, all heads) runs (`:418`) but is not in `decide()`, and the pre-registration only says "treat with caution".

Fix:
- add `d["sanity_correct"]`;
- make M1a "uninterpretable" when sanity is not *correct*;
- pre-register that.

**M1b.** "≥4 of the five layers have Δ > 0" passes even when all 32 layers have Δ > 0 (I tested this). That is likely if chat's baseline Δ is just below 0. The draft names five specific layers, which implies the others don't fix.

Fix: add a specificity condition and code it. For example, ≥4 of the five are among the top 8 layers by Δ, or report "non-specific" when more than 5 of the other 27 layers also have Δ > 0.

### 4. The no-pair path never computes M7 (pre-reg `:55-56`, `:101-102`; script `:252-254`)

`decide()` returns right after S when `layer_sweep` is absent. M7's bug counts and "repair transfers" are then never computed, even though the M7 data is collected and the pre-registration says M7 still runs. The smoke log's run 1 confirms this: the decisions contain only S and the note.

**Fix:** move the M7 block above the early return, or run it in both branches.

### 5. M7 outputs lack `says_less`, and the M7 repairs lack both flags (pre-reg `:40-41`, script `:465`, `:469`)

The pre-registration says *every* output records `hit_max_new` and `says_less`:
- the generalization format rows record only `hit_max_new`;
- `five_layer_repair` records neither.

These are the 500 outputs where "neither" caused by truncation or by "less" phrasing matters most for comparing against the draft's (2, 19, 0, 0)/100.

**Fix:** add both flags to both dicts. This is one line each.

## Optional suggestions

1. **"≈10 even heads for > 50%"** (pre-reg `:90-91`). As written, it is read from the even-only rate by k, which is the H22-confounded curve: the share of subsets containing H22 is k/16. Read it from the even_noH22 rates instead (k = 4, 8, 12), and say that k = 16 cannot be tested without H22.
2. **"H22 special"** (`:313`). It can be True when H22 fails in the all-32-layers config while 4 other heads succeed there. Count the ≤4 condition only in configs where H22 itself is *correct*, or report H22's status per config.
3. **M5 vs the random control.** No rule decides boring hypothesis 2 for M5. Pre-register "M5 specific" as M5 plus ≤2 of 10 random 5-layer sets *correct*, so a write-up can't quote M5 when 10/10 random sets also repair.
4. **M7 "repair transfers" on 1/1.** With the draft's 2/100 chat errors, eligibility will be tiny. Report the count next to the boolean in the write-up rule ("k of n eligible").
5. **Stale smoke again.** `smoke_cpu.log:2` records R14 sha `04aa7ead…` (`d330a52`). The job requires `20819e06…` (`d51b0ed`, committed one minute after the smoke). The diff touches only `rules`, `late_training_verdict` and `selftest`. `Patcher` is unchanged, and the new selftest passes locally (about 5 s), so this has no impact on the mechanics. Rerun or annotate the smoke for provenance.
6. **R15 self-hash.** The R15 script's sha256 is recorded but not checked against `R15_GIT_SHA`. The scratch copy is synced by hand, as for R14. An `R15_EXPECTED_SHA256` gate would close that gap.
7. **Decisions written only at the end.** Compute and save `decide()` once before M7 as well, so a crash during M7 still leaves M1–M6 decisions in `run.json`. They can be recomputed offline, but this saves a step.
8. **Wording.**
   - Pre-reg `:81` says "(all on generation outcomes)", but Δ readings now exist.
   - `:76` counts 6 gate generations; the code runs 5.
   - Δ's not-None rate (`:61`) is not computed. It is all-or-none per prompt and shows in the screen's Δ, so this only matters for consistency.

## Key file references

- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R15_mistral_true_heads/run_experiment.py`:
  - `:252-254` early return (fix 4);
  - `:262-265` M1 (fix 3);
  - `:277-281` M2 a/b/c and the verdict (fixes 1 and 2);
  - `:313` H22 special;
  - `:418` sanity, not used in decisions;
  - `:465` and `:469` M7 records (fix 5).
- `/Users/gus/github/MATS/docs/experiments/11-22_mistral7b_true_head_patching.md`:
  - `:78-79` sanity;
  - `:85-91` M1 and M2;
  - `:40-41` output flags;
  - `:101-102` M7.
- Test harness (scratch, not in the repo): `/private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/r15/t_decide.py`, `t2.py`, `t_subsets.py`, `t_tok.py`.
