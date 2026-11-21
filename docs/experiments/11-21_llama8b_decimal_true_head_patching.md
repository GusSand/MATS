# R13 — True per-head patching at Layer 10 (Llama-3.1-8B-Instruct, 9.8 vs 9.11)

**Status:** PLANNED (pre-registered 2026-11-21, before any compute; amended after design review rounds 1 and 2)
**Design review:** round 1 FIXES NEEDED — fresh subagent, 2026-11-21 — docs/reviews/11-21_llama8b_decimal_true_head_patching_design.md
**Design review:** round 2 FIXES NEEDED — fresh subagent, 2026-11-21 — docs/reviews/11-21_llama8b_decimal_true_head_patching_design_v2.md
**Design review:** PASS — fresh subagent, 2026-11-21 — docs/reviews/11-21_llama8b_decimal_true_head_patching_design_v3.md
**Script:** `9_8_research/reviewer_experiments/exp_R13_true_head_patching/run_experiment.py`
**Job:** `run.sbatch` (same folder). One job, one GPU, six cells sequentially in one process.
**Slurm job:** 24012066 failed at model load (cache had no `refs/main`; no experiment code ran).
**Slurm job:** 24014365 (commit a577123, `l40s_public`, model pinned to snapshot 0e9e39f2…). Log: `/scratch/gs157/MATS/logs/24014365_r13_heads.log`; results `/scratch/gs157/MATS/results/r13_24014365/`.
**CPU smoke test:** `exp_R13_true_head_patching/smoke_cpu.log` (random-weight 12-layer Llama, 32 query / 8 KV
heads; tokenizer `hf-internal-testing`, which gives 19/21 tokens rather than Llama-3's 17/19; full
config list, no `--limit`, command recorded in the log; checks mechanics only, outcomes mean nothing).

## Research question

Do the paper's head-level claims (the 16-head set H\*, even/odd parity, single heads fail, gradual
7→8 subset threshold, matched-spatial control) hold when the patch acts on actual attention heads
and only on the prompt pass?

## Why (the two bugs)

All head-subset scripts behind arXiv:2508.19414 and the v2 draft (`working_scripts/test_minimal_even_heads.py`,
`validate_even_heads.py`, `reviewer_experiments/exp_R1_arbitrary_heads`, `exp_R8_threshold_analysis`)
register a forward hook on `model.layers[10].self_attn`. The design review confirmed both bugs in each.

- **Bug A, hook point.** That output is post-`o_proj` (post-W_O). Slicing it as `[32, 128]` swaps blocks
  of residual-stream coordinates, not heads. Per-head patching acts on the `o_proj` input.
- **Bug B, decode steps.** The hook stays active through `generate()`. On decode steps `seq_len == 1`,
  so the patched blocks of each new token are overwritten with the clean prompt's position 0 (BOS).
  Position 0 is identical for both prompts, so this injects a prompt-independent constant.

## Simplest (boring) hypotheses, written first

1. The old head results are artifacts of bug A: "parity" is a property of residual coordinate blocks.
2. Bug B alone flips the answer: injecting BOS activations at decode steps, with no prompt patch,
   makes the buggy prompt answer correctly. Tested directly by the `decode_only` cell.

## Setup

- Model `meta-llama/Meta-Llama-3.1-8B-Instruct`, fp16, CUDA, Layer 10. GQA: query heads 4g..4g+3
  share KV head g, so at head level "even vs odd" means within-group positions {0,2} vs {1,3}.
- Clean prompt `"Which is bigger: 9.8 or 9.11?\nAnswer:"` (17 tokens), buggy prompt
  `"Q: Which is bigger: 9.8 or 9.11?\nA:"` (19 tokens).
- Greedy, 20 new tokens. One run per configuration (greedy is deterministic).
- **Outcome terms.** *correct* = `is_correct(text)` (R8/R1 classifier, verbatim). *bug* = a bug pattern
  matched and not correct. Rates below are always counts of *correct*.
- **Cells (hook × scope):**

| Cell | Hook | Patched passes | Patch source | Purpose |
|---|---|---|---|---|
| `post_wo/all_steps` | post-W_O blocks | prompt + every decode step | clean | old protocol; reproduction gate |
| `pre_wo/prefill` | true heads | prompt only, left-aligned | clean | corrected, paper's alignment |
| `pre_wo/prefill_right` | true heads | prompt only, right-aligned | clean | corrected, answer positions patched |
| `post_wo/prefill` | post-W_O blocks | prompt only, left-aligned | clean | attribution (bug A kept) |
| `pre_wo/all_steps` | true heads | prompt + every decode step | clean | attribution (bug B kept) |
| `post_wo/decode_only` | post-W_O blocks | prompt + every decode step | buggy run itself | bug B alone (hypothesis 2) |

  Left alignment pairs positions `0..16` of both prompts and leaves the buggy prompt's last two
  positions (`A`, `:`) unpatched. Right alignment pairs the last 16 non-BOS positions: clean 1..16 →
  buggy 3..18 (token-aligned: `Which`→` Which` … `Answer`→`A`, `:`→`:`); both BOS positions and buggy
  positions 1–2 (`Q`, `:`) are left unpatched, so no BOS activation is written into a content position.
- **Configs per cell (1,307):** all 32 heads; even 16; odd 16; 32 single heads; R8's 500 random
  7-subsets and 500 random 8-subsets of even heads (regenerated with R8's seed-42 call order;
  493 and 489 distinct, duplicates kept as in R8); R8's 7-even + {0,1,2,3,5,9,16}-odd transition (7);
  R1's 200 random 8-subsets of all 32 heads; R1's 30 matched pairs (even 8-subset and the same set
  shifted +1, which keeps every head in its KV group) = 60; R1's odd-only subsets at k = 8, 10, 12, 14, 16 (5).
  R1 subsets are read from R1's results JSON.
- Provenance in `run.json`: hostname, GPU name/UUID/driver, torch/CUDA/transformers versions,
  Slurm job, git SHA (job refuses to start without it), sha256 of the script.
- Output goes to a new run directory; the script refuses to reuse one. Raw text saved per config.

## Gates (recorded in `run.json` / each cell's JSON; failure actions fixed in advance)

- **Phenomenon gate:** unpatched clean is *correct* and unpatched buggy is *bug*. Fail → job exits.
- **Head isolation:** patching head 5 into the real model's `o_proj` input changes only head 5's slice.
  Fail → job exits. Also recorded (not gated): max |clean − buggy| at position 0 for both hooks, which
  `decode_only` assumes is ~0 (it was exactly 0 on CPU fp32).
- **Identity control:** patch all heads with the buggy run's own activations. In prompt-only cells it
  must equal the unpatched output exactly; fail → job exits, escalate. In all-steps cells it is expected
  to differ (bug B); its classification is recorded as "bug B alone, all blocks".
- **Determinism:** all-heads patch three times, identical text. Fail in a prompt-only cell → job exits.
- **Hook calls:** first hook call has `seq_len` 19, every later call 1. Fail in a prompt-only cell → exit.
- **All-heads equivalence:** in prompt-only scope, all-heads `pre_wo` equals all-blocks `post_wo`
  (o_proj has no bias). Must give the same text in `pre_wo/prefill` and `post_wo/prefill`.
  Fail → escalate before interpreting anything; the recorded first-token comparison tells fp16 GEMM
  noise (same first token) apart from a wrong hook.
- **Reproduction gate (`post_wo/all_steps`):** outcome agreement with the original runs, subset by
  subset: R8 7-subsets (ref 202/500 correct) and 8-subsets (ref 297/500) ≥ 950 of 1,000 combined;
  R1 random 8-subsets (ref 13/200) ≥ 190 of 200; R1 pairs (ref even 21/30, odd 0/30) ≥ 57 of 60.
  All 1,260 reference subsets must also be matched by head set. Evaluated by the script as
  `reproduction_gate_ok` and written to `run.json` right after the cell. R1's 5 odd-only subsets are
  reported, not gated. Below any threshold: library/hardware numerics differ from the original runs;
  the job finishes but no cell is interpreted until an INBOX decision.

## Pre-registered decision rules

Applied separately to `pre_wo/prefill_right` (**primary**: token-aligned, answer positions patched; its
verdict is the headline) and `pre_wo/prefill` (replication under the paper's left alignment; reported
as secondary). The primary cell was fixed before the run and is not chosen by outcome.

- **L10 repair survives:** all-heads patch is *correct*. If it fails in a cell, the verdict for that
  cell is "Layer-10 repair fails under true-head prompt-only patching with <left|right> alignment",
  and the head rules below are reported but not interpreted for that cell.
- **H\* holds at head level:** all of (a) even-16 *correct*; (b) odd-16 not *correct*;
  (c) matched pairs: even *correct* and shifted-odd not *correct* on ≥ 15 of 30 pairs, and odd
  *correct* on ≤ 3 of 30; (d) rate(R8 even 8-subsets) − rate(R1 random 8-subsets) ≥ 0.20, as counts
  out of 500 and 200. Any failing → "parity not supported at the head level" for that cell.
- **Bug B alone (hypothesis 2):** in `post_wo/decode_only`, count *correct* among all_heads, even-16,
  odd-16 and the 1,000 R8 subsets. If all-heads or ≥ 10% of R8 subsets are *correct*, the old repairs
  can be produced by BOS injection alone, with no information from the clean prompt. Otherwise:
  BOS injection alone does not produce the repair.
- **Descriptive only (no thresholds):** single heads (count of 32), 7- and 8-subset rates, odd-only
  subsets, transition, and the comparison of each corrected-cell count with the attribution cells.

## Compute estimate

1,307 configs × 6 cells = 7,842 generations, plus 6 × 5 control generations + 2 baselines = 32,
plus 11 capture passes in cells and 6 for the head-isolation (2) and BOS (4) checks. R8 measured ~0.48 s per generation including hooks (24,150 s / 50,350).
≈ 7,874 × 0.48 s ≈ 1.05 GPU-h; allow 2× for hardware differences → ≤ 2.1 GPU-h.
Requested walltime 4 h, `l40s_public`, account `torch_pr_173_tandon_advanced`.

## Limitations known before the run

- Single canonical prompt pair. H\* was found on this pair, so any positive result is in-sample with
  respect to prompts; the paper must say so.
- R8/R1 hardware is unrecorded; the reproduction gate also tests the environment.
- R13 uses the R8/R1 classifier; v1's `test_minimal_even_heads.py` used a narrower one, so R13 numbers
  are not compared directly with v1 numbers.

## Results

(none yet)
