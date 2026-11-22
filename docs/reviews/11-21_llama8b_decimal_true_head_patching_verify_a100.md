Reviewer: fresh subagent (Claude Code)
Agent-ID: ad484e7b23465c356
Verdict: VERIFIED-WITH-FLAGS
Transcript-SHA256: 18c4a78ea63907722a674bfcfdf2264bc8d7d64e9bf93e978627ac9bcf26e56a

## Brief given

You are an independent verifier. Recompute everything from raw files, and trust no summary. Do not edit repo files.

Repo: ~/github/MATS. Read the pre-registration first: docs/experiments/11-21_llama8b_decimal_true_head_patching.md, and especially its "Hardware replication" section. That section fixed, before submission, how this run would be read.

Inputs:
- Run to verify (A100, Slurm job 24084222): 9_8_research/reviewer_experiments/exp_R13_true_head_patching/results/r13_24084222/
- Earlier L40S run (job 24014365), which was already verified: …/results/r13_24014365/
- Script: …/exp_R13_true_head_patching/run_experiment.py
- References: 9_8_research/reviewer_experiments/exp_R8_threshold_analysis/results_20260314_091723.json and exp_R1_arbitrary_heads/results_20260314_023132.json

Tasks:
1. Provenance. Check the script sha256 against run.json; the run.json git_sha (10e89d6) exists; the GPU is an A100; and the run used the same model snapshot as the L40S run.
2. Recompute every A100 cell summary from the raw texts using the script's `is_correct`.
3. Recompute the reproduction gate against R8/R1, subset by subset and matched by head set.
4. Recompute every gate: phenomenon, head isolation, identity/determinism/shapes, and all-heads equivalence.
5. Evaluate each pre-registered decision rule on the A100 data:
   - L10 repair, and H\* (a)–(d), for pre_wo/prefill_right and pre_wo/prefill;
   - "bug B alone" for post_wo/decode_only.
   Compare each verdict with the L40S verdict. For each cell, list the config-level outcome differences between L40S and A100.
6. Apply the pre-registered "Hardware replication" reading: is each verdict replicated on two GPU types or hardware-sensitive, and does the gate result support the hardware explanation?
7. Flag anything odd. One example: the BOS max-abs diff is reported nonzero on A100 but 0.0 on L40S. Assess whether that matters.

Report as markdown. Put the verdict first: VERIFIED, VERIFIED-WITH-FLAGS, or DISCREPANCIES. Return the full report as your final message.

## Report

# R13 A100 replication (job 24084222): independent verification

## Verdict: **VERIFIED-WITH-FLAGS**

I recomputed every number from the raw per-config texts and found no discrepancies. On A100 the reproduction gate **passes perfectly**: R8 is 1,000/1,000, R1 random 8-subsets 200/200, R1 pairs 60/60, and odd-only 5/5. All three pre-registered verdicts come out the same on A100 as on L40S, so under the pre-registered reading each one is **replicated on two GPU types**, and the gate result **supports the hardware explanation** for the L40S shortfall. The flags (below) concern interpretation and record-keeping. None changes a verdict.

I did not edit any repo files. My scripts are in `/private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/` (`verify.py`, `v2.py`, `texts.json`).

---

## 1. Provenance

| Check | Result |
|---|---|
| Script sha256 (working tree, at `10e89d6`, at `a577123`) | All `1ce5b0cc…f280348d`, equal to `run.json` `script_sha256` on both runs. The script has one commit in its history (`a546b96`), so it did not change between runs. |
| `git_sha` 10e89d6 exists | Yes: commit `10e89d6916a8…`, "R13: plan A100 hardware replication (same script)", 2026-11-22 09:23:15. It touches only the experiment doc and adds the "Hardware replication" section. |
| Reading fixed before the run | Yes. The commit is at 09:23:15 and `run.json` `started` is 09:26:47. The job-ID line was added in `ab0df86` at 09:23:26, also before the start. The doc has no uncommitted diff. |
| GPU | `NVIDIA A100-SXM4-80GB, GPU-e4ae6c00-…, driver 610.43.02`, host `ga004`. Recorded in both `run.json` and the log header. |
| Model snapshot | Same path in both runs: `…/snapshots/0e9e39f249a16976918f6564b8830bc894c89659`. |
| Software | torch 2.13.0+cu130, CUDA 13.0, transformers 5.15.1, the same as L40S. Driver is also the same, so the GPU is the only recorded difference. |
| Reference files | The R8/R1 JSONs are clean in git (last commit `6d9d14c`). Their `success_rate` values are all exactly 0.0 or 1.0, so the `>= 0.5` threshold is not doing any work. |

## 2. Cell summaries (recomputed with the script's `is_correct`/`is_bug`, copied verbatim)

All 6 cells have 1,307 configs, and the key set equals my independent rebuild: R8 seed-42 call order regenerated, and R1 subsets read from its JSON. Every config's `heads` matches. The stored `correct`/`bug` flags disagree with reclassification 0 times. Every output is either *correct* or *bug* (0 are neither). Each recomputed summary, including the transition flags, equals both the JSON `summary` and the log line.

A100 counts of *correct* (L40S value in brackets where it differs):

| cell | all | even16 | odd16 | single | even7 | even8 | rand8 | odd-only | pairs E&!O | pairs O |
|---|---|---|---|---|---|---|---|---|---|---|
| post_wo/all_steps | 1 | 1 | 0 | 0/32 | 202 [166] | 297 [275] | 13 [12] | 0/5 | 21 [20] | 0 |
| pre_wo/prefill | 0 | 0 | 0 | 1/32 (h27) | 0 | 0 | 24 [23] | 1/5 | 0 | 7 [6] |
| pre_wo/prefill_right | 0 | 0 | 0 | 0/32 | 0 | 0 | 0 | 0/5 | 0 | 0 |
| post_wo/prefill | 0 | 1 | 0 | 0/32 | 38 [22] | 82 [61] | 1 | 0/5 | 7 [4] | 0 |
| pre_wo/all_steps | 1 | 0 | 1 | 1/32 (h27) | 3 | 9 | 29 | 3/5 | 0 | 10 [9] |
| post_wo/decode_only | 0 | 0 | 0 | 0/32 | 0 | 0 | 0 | 0/5 | 0 | 0 |

I also recomputed the L40S run. It matches the experiment doc's table exactly.

## 3. Reproduction gate (`post_wo/all_steps` vs R8/R1, subset by subset, matched by head set)

| subset | n_ref | heads matched | agree (A100) | agree (L40S) |
|---|---|---|---|---|
| R8 even7 | 500 | 500 | **500** | 464 |
| R8 even8 | 500 | 500 | **500** | 478 |
| R8 combined (≥950) | 1,000 | 1,000 | **1,000 PASS** | 942 FAIL |
| R1 rand8 (≥190) | 200 | 200 | **200 PASS** | 199 |
| R1 pairs (≥57) | 60 | 60 | **60 PASS** | 59 |
| R1 odd-only (not gated) | 5 | 5 | 5 | 5 |

All 1,260 gated reference subsets are matched by head set. The R8 reference heads equal my regenerated seed-42 subsets. `reproduction_gate_ok = true` is correct. A100 also reproduces the reference totals exactly: 202/500, 297/500, 13/200, and pairs 21 even / 0 odd.

## 4. Other gates

- **Phenomenon:** clean `" 9.8 is bigger than 9.11.\nExplanation: …"` is *correct*, and buggy `" 9.11 is bigger than 9.8.\nQ: …"` is *bug*. PASS. Both baseline texts are identical to L40S.
- **Head isolation:** recorded `changed_heads: [5]`, ok. Checking this needs the GPU, so I could not recompute it from the files.
- **Identity:** in all 6 cells the identity output equals the unpatched buggy baseline. That is required in the prompt-only cells. In the all-steps cells the pre-registration expected a difference. This is the same deviation the L40S verification already flagged.
- **Determinism:** in all 6 cells the three all-heads outputs are identical, and `configs["all_heads"]` equals them.
- **Hook shapes:** `[19, 1, 1, …]`, 20 calls, in every cell. PASS.
- **All-heads equivalence:** `pre_wo/prefill` and `post_wo/prefill` all-heads texts are identical (`" 9.11 is bigger than 9.8.\nQ: Which is bigger: 9"`). Same text, same first token, same class. PASS.

## 5. Decision rules on A100, compared with L40S

**`pre_wo/prefill_right` (primary)**
- L10 repair: all-heads is *bug*. Verdict: **"Layer-10 repair fails under true-head prompt-only patching with right alignment."** H\* rules are reported, not interpreted:
  - (a) even-16 is not correct: fail.
  - (b) odd-16 is not correct: pass.
  - (c) 0/30 pairs, with odd correct on 0/30: fail.
  - (d) 0/500 − 0/200 = 0.00: fail.
- Same as L40S: **yes**.
- Config-level differences: **0 class changes**. There are 2 text-only changes (`rand8_028`, `rand8_185`); in both, the bug text swaps between the "Q:" and "Explanation:" continuations.

**`pre_wo/prefill` (secondary)**
- L10 repair: all-heads is *bug*. Verdict: **"Layer-10 repair fails under true-head prompt-only patching with left alignment."** H\* is not interpreted:
  - (a) fail.
  - (b) pass.
  - (c) 0/30, with odd correct on **7/30** (L40S 6/30). Fails on both GPUs.
  - (d) 0/500 − 24/200 = **−0.120** (L40S −0.115). Fails on both GPUs.
- Same as L40S: **yes**.
- Config-level differences: **2 class changes**, both bug on L40S and correct on A100: `pair23_odd` and `rand8_072`.

**`post_wo/decode_only` (bug B alone)**
- all-heads, even-16 and odd-16 are all *bug*, and 0/1,000 R8 subsets are correct. Verdict: **"BOS injection alone does not produce the repair."**
- Same as L40S: **yes**.
- Config-level differences: **0 class changes** and 1 text-only change (`rand8_035`).

**Attribution cells** (descriptive, no verdicts):
- `post_wo/all_steps`: 60 class changes. These are exactly the 58 R8 + 1 rand8 + 1 pair subsets that failed reproduction on L40S. There are 69 text differences in total.
- `post_wo/prefill`: 40 class changes, 44 text differences.
- `pre_wo/all_steps`: 1 class change (`pair01_odd`), 20 text differences.

The qualitative descriptive statements in the doc still hold on A100:
- Single heads: 0/32 with right alignment and 1/32 with left (head 27 on both GPUs).
- Old-protocol transition: only the +5-odd config is correct.
- `post_wo/prefill`: even-16 is correct and all blocks are not.
- `pre_wo/all_steps`: all heads and odd-16 are correct, even-16 is not.

## 6. Pre-registered "Hardware replication" reading

- **Gate ≥ 950 on A100: yes (1,000).** Per the fixed reading, this **supports the hardware explanation** for the L40S shortfall (942). The match is exact, 1,260/1,260, and torch, transformers, driver and snapshot are the same on both runs. The L40S-vs-A100 difference therefore comes from the GPU, or the kernels chosen on it. The likely implication is that R8 and R1 ran on A100-class hardware, but their hardware is unrecorded, so that is an inference.
- **Verdicts: all three are replicated on two GPU types.** Primary right-aligned: repair fails. Secondary left-aligned: repair fails. Decode-only: BOS injection alone does not repair. No verdict is hardware-sensitive.
- The A100 cells pass the gate on their own merits, so they can be interpreted without relying on the waiver.

## 7. Flags

1. **The hardware effect has a direction, it is not symmetric noise.** All 103 class changes between L40S and A100, across every cell, go from bug on L40S to correct on A100. None goes the other way. The L40S shortfall against R8 had the same direction. The verdicts are unaffected because they rest on all-heads and on the 0/500 counts. But the descriptive counts near the decision boundary are GPU-dependent:
   - `post_wo/prefill` R8 subsets: 83 → 120 of 1,000.
   - `post_wo/prefill` pairs E&!O: 4 → 7.
   - `pre_wo/prefill` odd-pair count: 6 → 7.

   The write-up should give these counts with their GPU, or report both values.
2. **The waiver argument holds up on A100.** On L40S, the 58 R8 subsets that failed reproduction are correct 0 times in `post_wo/prefill`, against 83/942 for the rest. On A100 they are still correct 0/58 times, against 120/942 for the rest.
3. **BOS max-abs diff is 6.1035e-05 on A100 and 0.0 on L40S.** 6.1e-5 = 2⁻¹⁴, which is one fp16 ULP for values of magnitude 0.0625 to 0.125. Position 0 is mathematically identical for both prompts under causal attention, so this is a kernel numerics effect: A100 computes row 0 slightly differently at sequence length 17 versus 19 (different tiling or reduction). Assessment: **immaterial to the verdicts.**
   - `decode_only` injects the buggy run's own BOS, so it remains exactly "bug B alone" whatever the diff is.
   - The all-steps cells inject the clean BOS, which on A100 differs from the buggy BOS by one ULP somewhere. That makes "all-steps decode writes = the same constant as decode_only" approximate rather than exact on A100. Even so, the identity controls still equal unpatched, and decode_only is 0/1,307.
   - It is also direct evidence that this GPU's numerics depend on sequence length, which fits flag 1.
   - Note that the L40S verification said the identity behaviour "fits the BOS diff of 0". That argument does not need BOS diff = 0, since the identity control writes the buggy run's own BOS either way.
4. **The all-steps identity deviation recurs on A100.** The pre-registration expected the all-steps identity output to differ from unpatched, and it is equal again. This is the same flag as on L40S, and it is consistent across GPUs.
5. **Head isolation is taken from the record, not recomputed.** It needs the model on a GPU.
6. **Bookkeeping:**
   - `run.sbatch` still says `--partition=l40s_public`. The A100 override was made on the command line and is not saved in `run.json` or `command`. The GPU name proves A100 was used, but the submit line itself is not recorded.
   - `results/r13_24084222/` is untracked in git.
   - The doc's Status line and Results section do not yet report the A100 results.
