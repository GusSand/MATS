Reviewer: fresh subagent (Claude Code)
Agent-ID: a9bdfc30f464994f2
Verdict: FIXES NEEDED
Transcript-SHA256: 5f2883726b7102e449a79e5595455a25430b8c72ee33497ec20c054150337e9d

## Brief given

You are a design reviewer for a planned GPU experiment (round 2, after an amendment). You did not write it and have no stake in it. Review it adversarially BEFORE any compute is spent. Do not edit any repo files and do not run git commit/push; only read files and run read-only commands (scratch files only under /Users/gus/github/MATS/outputs/).

Experiment file: /Users/gus/github/MATS/docs/experiments/11-21_llama8b_decimal_true_head_patching.md
Script: /Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R13_true_head_patching/run_experiment.py
Job file: /Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R13_true_head_patching/run.sbatch
CPU smoke log: /Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R13_true_head_patching/smoke_cpu.log
Round-1 review (its required fixes are listed at the end): /Users/gus/github/MATS/docs/reviews/11-21_llama8b_decimal_true_head_patching_design.md
Reference data: /Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R8_threshold_analysis/ and exp_R1_arbitrary_heads/ (scripts + results JSONs)
Methodology: /Users/gus/github/research_template/global/research-workflow.md ("Design review", "Skepticism checklist").
A CPU python with torch+transformers exists at /private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/r13env/bin/python and a random-weight 32-head Llama at .../scratchpad/tiny32 (use env R13_MODEL=<path>) if you want to execute checks; run python with -I.

Tasks:
A. For each of the 9 round-1 required fixes, state whether it is resolved in the current files (RESOLVED / PARTIAL / NOT RESOLVED) with evidence.
B. Re-run the checklist on the amended design: (1) fact-check every number against sources (reference counts 202/500, 297/500, 13/200, 21/30, 0/30; config counts; compute estimate); (2) recompute arithmetic; (3) terminology consistency; (4) completeness — does the script implement every gate, cell, alignment and decision rule exactly as the file states (including the right-alignment indexing, decode_only semantics, gate failure actions, all-heads equivalence); (5) circularity; (6) category check (R1 vs R8 provenance of each reference number). Is the matching control pre-registered?
C. Any new bug introduced by the amendment.

End with one verdict line: "VERDICT: PASS" or "VERDICT: FIXES NEEDED" or "VERDICT: REJECTED", then required fixes (if any) and optional suggestions, separately.

## Report

## R13 design review, round 2

**Bottom line:** 7 of the 9 round-1 fixes are fully resolved and 2 are partly resolved. The mechanics hold up on the 32-head random-weight Llama. Two problems remain, both cheap to fix:
- **Right alignment puts the clean prompt's BOS activation into buggy position 2 (`:`).** This affects the corrected cell the amendment added.
- **The reproduction gate is not machine-evaluated or saved promptly.**

The matched control is now pre-registered: rule (c) uses R1's 30 pairs, read from R1's JSON, in both corrected cells.

Checks I ran are in `/Users/gus/github/MATS/outputs/r13_review2/check.py`, on `tiny32` and the CPU env. I also tokenized both prompts with the real Llama-3.1 tokenizer, taken from the cached `allenai/Llama-3.1-Tulu-3-8B-SFT`. I edited no repo files and made no commits.

### A. The 9 round-1 fixes

1. **Matched-spatial control and odd-only subsets: RESOLVED.**
   - `build_configs` at `run_experiment.py:207-210` reads `section4_matched_controls` and `section3_odd_only` from R1's JSON. All 30 odd sets equal even+1 (verified).
   - Rule (c) is pre-registered and computed by `summarize`, which I ran on a reduced config set and confirmed works.
   - At head level, +1 keeps every head in its KV group (0/2 mod 4 becomes 1/3 mod 4).
2. **Random 8-subsets match R1: RESOLVED.** The 200 subsets are read verbatim from `section1_random_mixed_8`. The reproduction gate checks them (≥190/200), and all 200 heads matched in the smoke run.
3. **Hypothesis 2 tested as stated: RESOLVED.**
   - `post_wo/decode_only` uses `saved = buggy_saved` with `all_steps` generation (`:257-259`). The prompt pass is an identity patch; decode steps get the buggy BOS. On the CPU model, BOS is bit-identical between the prompts (max diff 0.0 for both hooks).
   - Identity outputs are classified (`identity_correct`).
   - The rule is pre-registered. It has no explicit negative branch; see the suggestions.
4. **"Bug" defined: RESOLVED.** `is_bug` (`:73-75`) is used by the phenomenon gate (`:309`), and "correct" is used consistently as the outcome.
5. **Record every gate and define failure actions: PARTIAL.**
   - Done: the phenomenon gate, per-cell identity/determinism/hook-shape gates, R8/R1 refs failing loudly (`load_refs`), and failure actions are all in place.
   - Missing (a): the reproduction gate's thresholds (≥950/1000 R8 combined, ≥190/200, ≥57/60) are never evaluated. `reproduction()` reports per-group counts, with 7 and 8 separate, and no pass/fail flag.
   - Missing (b): `run["reproduction_vs_R8_R1"]` is computed after cell 1, but `run.json` is only rewritten at the very end (`:332`). If a later prompt-only cell aborts via `sys.exit` (`:277`), or the job crashes or times out, the gate result never reaches `run.json`. It can still be recomputed from `results_post_wo__all_steps.json`.
6. **All-heads equivalence pre-registered: RESOLVED.** It is in the file and recorded at `:326-330` (`same_text` and `same_class`). On `tiny32` it gave the same text.
7. **Alignment failure mode: RESOLVED.** The `pre_wo/prefill_right` cell was added and the failure branch reworded ("fails … with <left|right> alignment"). It introduces a new issue; see C1.
8. **Stale results / SHA: RESOLVED.** `os.makedirs(exist_ok=False)` plus a per-file existence check, and the SHA is required in both the sbatch (`:?`) and the script (`:302`). The script's sha256, GPU, versions and Slurm job ID are recorded.
9. **Smoke-test claim backed: RESOLVED, with caveats.** `smoke_cpu.log` is attached, but:
   - It shows 1,300 configs per cell, `summary: {}`, pairs `heads_matched 58` and odd-only 0. So it was run with `--limit 1300`: the full path (`summarize`, the last 7 configs) was not exercised, and the command line is not in the log. I exercised `summarize` and `reproduction` separately; both work.
   - The `tiny32` tokenizer is not Llama-3. It gives 19/21 tokens, not 17/19, so the 17/19 alignment and the "seq_len 19" gate were not tested at their real lengths.

### B. Checklist on the amended design

**(1) Fact-check of reference numbers (all verified against the JSONs)**

| Number in the file | Source | Status |
|---|---|---|
| 202/500 and 297/500 | R8 | Verified; every rate is 0 or 1 |
| 13/200 | R1 `section1_random_mixed_8` | Verified |
| 21/30 even, 0/30 odd | R1 `section4_matched_controls` | Verified |
| Odd-only k = 8–16 all 0 | R1 | Verified |
| 493 / 489 distinct subsets | R8 | Verified |
| 17 / 19 prompt tokens | Real Llama-3.1 tokenizer | Verified |
| 50,350 generations, 24,150 s → 0.48 s each | R8: (500+500+7)×50, 24,149.8 s | Verified. R8 repeated a greedy generation 50×, sequentially, so the timing is comparable. |

The classifier is identical to R1's and R8's `is_correct`.

**(2) Arithmetic**
- Configs per cell: 3 + 32 + 500 + 500 + 7 + 200 + 60 + 5 = 1,307. `build_configs` returns 1,307 (run).
- Generations: 7,842 + 6×5 + 2 = 7,874.
- Capture passes: 5×2 + 1 = 11.
- Compute: 7,874 × 0.48 s = 3,780 s ≈ 1.05 h; 2× is 2.1 h, inside the 4 h walltime. Correct.

**(3) Terminology**
- The reproduction gate lists three thresholds, then says "Below either threshold". It should say "any".
- The script also reports an `oddonly_` group that has no threshold in the file.
- "L10 repair survives" is only a rule name, which is fine.
- Output records the `unit` per hook ("128-dim block of post-W_O output"). Good.

**(4) Completeness: script vs file**
- **Left alignment:** maps clean 0..16 to buggy 0..16 (BOS to BOS, `Which`→`Q`, …), leaving `A` and `:` unpatched. Matches the file.
- **Right alignment** (`_swap`, `:142-143`): maps clean 0..16 to buggy 2..18. With the real tokenizer this is token-aligned (`Which`→` Which`, …, `Answer`→`A`, `:`→`:`), except that clean BOS lands on buggy position 2 (`:`). See C1.
- **Prompt-only scope:** only hook call 1 is patched; decode calls are untouched (verified). In `all_steps`, decode calls receive clean BOS exactly (diff 0.0).
- **Head isolation:** patching head 5 changes only head 5's slice (verified).
- **`decode_only` semantics:** correct.
- **Gate failure actions:** match the file. Prompt-only cells exit on identity, determinism or shape failure; `all_steps`/`decode_only` cells only record.
- **All-heads equivalence:** recorded, but it is a human escalation, not automated. That is acceptable as written.
- **Gaps:** the reproduction-gate evaluation and its persistence (fix 5 above).

**(5) Circularity**
- None found. The H* thresholds are lenient relative to the old-protocol references, and the screening quantity differs from the decision quantity.
- The prompt pair is in-sample, and the Limitations section discloses this.
- Two corrected cells, each with its own verdict, create a forking path: no primary cell is designated (see the suggestions).

**(6) Category check**
- 202/500 and 297/500 are correctly attributed to R8.
- 13/200, 21/30 and 0/30 and the odd-only results are correctly attributed to R1.
- Rule (d) mixes R8's even 8-subsets with R1's random 8-subsets, and both are labelled correctly.

### C. New issues introduced by the amendment

**C1. Right alignment writes the clean BOS-position head outputs into buggy position 2 (`:`), a content position.** BOS is the attention sink, and its per-head output is atypical. This puts an out-of-distribution value into the residual stream at position 2 in the very cell meant to fix the alignment confound. A null result in `pre_wo/prefill_right` could then come from this injection. The alignment wanted is clean 1..16 → buggy 3..18, with BOS left unpatched on both sides. That is a one-line change in `_swap` for `align == "right"`; otherwise, the file must state the BOS→`:` mapping and pre-register it as a limitation.

**C2.** Reproduction thresholds are not evaluated, and `run.json` is not saved after cell 1 (fix 5 above).

**C3. Minor.** The smoke log does not record its command or `--limit`, and it used a non-Llama-3 tokenizer. The file should say so.

**No issues found** in `summarize`, `reproduction`, the decode-only wiring, or the determinism and shape gates.

VERDICT: FIXES NEEDED

**Required fixes:**
1. Fix right alignment so the clean prompt's BOS is not written into buggy position 2: map clean 1..16 → buggy 3..18 and leave both BOS positions alone. Update the file's description ("Right alignment pairs the last 16 non-BOS positions"). If you keep the current mapping instead, disclose it explicitly in the file and pre-register it as a limitation.
2. In the script, evaluate the reproduction gate as the file defines it and write a boolean `reproduction_gate_ok` with per-criterion values:
   - R8 7+8 combined ≥ 950/1000
   - R1 random ≥ 190/200
   - R1 pairs ≥ 57/60

   Rewrite `run.json` immediately after cell 1 (and ideally after every cell), so the gate survives a later abort or timeout. In the file, change "either threshold" to "any threshold", and say whether `oddonly_` agreement is gated or only reported.

**Optional suggestions:**
- Name one corrected cell (right-aligned, once fixed) as primary for the paper's headline. Keep left-aligned as the paper-protocol replication, to avoid choosing between them after the fact.
- Add a negative branch to the hypothesis-2 rule: "otherwise, BOS injection alone does not produce the repair".
- Record the max abs difference between the clean and buggy position-0 captures for both hooks in `run.json`. On GPU fp16, different sequence lengths may break the bit-identity that `decode_only` relies on (it was 0.0 on CPU fp32).
- The all-heads equivalence uses exact text. fp16 GEMMs over 17 vs 19 rows may differ in the last bits, so also record a logit or first-token comparison. That makes a text mismatch diagnosable rather than an automatic escalation.
- Re-run the smoke test without `--limit` on a reduced config list. Record the command in the log, and note in the file that the smoke tokenizer gives 19/21 tokens.
- Still not adopted from round 1: an automated single-head isolation assert before the grid, and pinning the `transformers` version in the env. The `o_proj` call must stay positional for `args[0]` to work.
- Make sure `/scratch/$USER/MATS/logs` exists before submitting, or Slurm cannot open the `--output` file.
