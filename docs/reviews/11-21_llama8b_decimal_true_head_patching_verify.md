Reviewer: fresh subagent (Claude Code)
Agent-ID: a4657449103c445a0
Verdict: VERIFIED-WITH-FLAGS
Transcript-SHA256: e640145345788d0bb089ec8b6537fdfcf69e8eb51dd4c0808d612d1e388921d2

## Brief given

You are an independent verifier for a finished experiment. Recompute everything from the raw files and do not trust any summary. Do not edit any files except the one report noted at the end.

Repo: ~/github/MATS. Pre-registration (read it first, including the gates and decision rules): docs/experiments/11-21_llama8b_decimal_true_head_patching.md
Script: 9_8_research/reviewer_experiments/exp_R13_true_head_patching/run_experiment.py
Raw results (Slurm job 24014365): 9_8_research/reviewer_experiments/exp_R13_true_head_patching/results/r13_24014365/ (six results_<hook>__<scope>.json files, run.json, job log)
Reference runs: 9_8_research/reviewer_experiments/exp_R8_threshold_analysis/results_20260314_091723.json and exp_R1_arbitrary_heads/results_20260314_023132.json

Tasks:
1. Check provenance: confirm that run.json's script_sha256 equals the sha256 of the script file, and that git_sha a577123 exists in the repo.
2. For each cell, recompute the summary counts from the per-config raw text using the script's is_correct classifier. Import it from the script or copy it verbatim. Counts: all_heads, even_16, odd_16, single heads, even7, even8, rand8, odd-only, and pairs (even correct & odd not / even correct / odd correct). Compare with the logged summaries, which are in the job log.
3. Recompute the reproduction gate yourself, subset by subset against R8/R1, matched by head set and not by index. Report agreement counts, the direction of every disagreement (reference correct → now incorrect, or the reverse), and whether head sets all match. Also check whether the disagreeing R8 subsets look borderline, e.g. what their R8 outputs and R13 outputs were.
4. Check the gates recorded in each cell (identity, determinism, shapes), the all-heads equivalence, phenomenon, head isolation.
5. Report what the pre-registration says must happen when the reproduction gate fails. Then evaluate each pre-registered decision rule mechanically for pre_wo/prefill_right, pre_wo/prefill and post_wo/decode_only, labelled clearly as "conditional on the gate being waived". Give the numbers each rule depends on.
6. Look for anything off, e.g. duplicated configs, empty texts, truncated outputs, or a classifier that is mislabelling. Sample about 15 raw texts across cells, including several counted correct and several not.

Write your report as markdown, verdict first (VERIFIED / DISCREPANCIES). Save it by running:
python scripts/check_reviews.py --save-report docs/reviews/11-21_llama8b_decimal_true_head_patching_verify.md < report.md
If that flag's usage differs, check `python scripts/check_reviews.py --help`. Write report.md in a temp dir, not in the repo. Return the full report in your final message.

## Report

# R13 verification: job 24014365

**Verdict: VERIFIED-WITH-FLAGS.** Every logged number matches what I recomputed from the raw per-config text. I found no discrepancies. The main flag is that the **reproduction gate failed** (942 of 1,000 against a threshold of 950), as `run.json` records. Under the pre-registration, no cell may be interpreted until an INBOX decision is made.

## 1. Provenance
- `shasum -a 256 run_experiment.py` = `1ce5b0cc…f280348d`. This equals `run.json` `script_sha256`. The copy at `git show a577123:<script>` has the same hash.
- `a577123` exists. It is a commit: "R13: load model by pinned snapshot path (job 24012066 failed: cache has no refs/main)". The script has no uncommitted changes. Only `results/` is untracked.
- Model is pinned to snapshot `0e9e39f2…`. Run on an L40S, torch 2.13.0+cu130, transformers 5.15.1.

## 2. Summary counts recomputed from raw text
I extracted `is_correct`/`is_bug` verbatim from the script via AST. The script's classifier is identical to R8's and R1's. Each cell has 1,307 configs. In every cell the stored `correct`/`bug` flags match reclassification (0 mismatches). There are no empty outputs. "Neither" outcomes: 0. The recomputed summary equals both the log line and the JSON `summary`, and the transition flags also match.

| cell | all | even16 | odd16 | single | even7 | even8 | rand8 | odd-only | pairs E&!O | pairs E | pairs O |
|---|---|---|---|---|---|---|---|---|---|---|---|
| post_wo/all_steps | 1 | 1 | 0 | 0/32 | 166/500 | 275/500 | 12/200 | 0/5 | 20/30 | 20/30 | 0/30 |
| pre_wo/prefill | 0 | 0 | 0 | 1/32 | 0/500 | 0/500 | 23/200 | 1/5 | 0/30 | 0/30 | 6/30 |
| pre_wo/prefill_right | 0 | 0 | 0 | 0/32 | 0/500 | 0/500 | 0/200 | 0/5 | 0/30 | 0/30 | 0/30 |
| post_wo/prefill | 0 | 1 | 0 | 0/32 | 22/500 | 61/500 | 1/200 | 0/5 | 4/30 | 4/30 | 0/30 |
| pre_wo/all_steps | 1 | 0 | 1 | 1/32 | 3/500 | 9/500 | 29/200 | 3/5 | 0/30 | 0/30 | 9/30 |
| post_wo/decode_only | 0 | 0 | 0 | 0/32 | 0/500 | 0/500 | 0/200 | 0/5 | 0/30 | 0/30 | 0/30 |

Transition in post_wo/all_steps: only +5 odd is correct. That matches R8's transition 7 of 7, and the head sets are identical.

## 3. Reproduction gate (post_wo/all_steps), matched by head set
I matched each reference subset by its sorted head tuple, not by index. Index alignment also gives identical head sets for all 1,000 R8 subsets. All 1,265 reference subsets were matched: R8 500 + 500, R1 rand8 200, pairs 60, odd-only 5.

| group | ref correct | R13 correct | agree | ref correct → now incorrect | ref incorrect → now correct | threshold |
|---|---|---|---|---|---|---|
| R8 7-subsets | 202/500 | 166/500 | 464 | 36 | 0 | |
| R8 8-subsets | 297/500 | 275/500 | 478 | 22 | 0 | |
| R8 combined | | | **942/1000** | 58 | 0 | ≥950 **FAIL** |
| R1 rand8 | 13/200 | 12/200 | 199 | 1 (rand8_057) | 0 | ≥190 pass |
| R1 pairs | 21 even / 0 odd | 20 / 0 | 59 | 1 (pair02_even) | 0 | ≥57 pass |
| R1 odd-only (not gated) | 0/5 | 0/5 | 5 | 0 | 0 | |

- My recomputation matches `run.json` exactly: `reproduction_gate_ok: false` and `all_heads_matched: true`.
- **All 60 disagreements go one way.** In each, the reference was correct and R13 now gives the canonical bug text `" 9.11 is bigger than 9.8.\nQ: Which is bigger: 9"`. No subset moved the other way.
- **Borderline?** The R8 reference rates cannot show it. Every R8 subset is 0/50 or 50/50, so the 50 greedy trials were one deterministic output repeated. All 58 lost subsets had a reference rate of 1.0. R8 saved no sample texts for the 7- and 8-subsets. R1 did, and the two lost R1 configs had the reference outputs `"9.8 is bigger than 9.11.\nQ: Which is bigger:"` (rand8_057) and `"9.8 is bigger than 9.11.\nExplanation: 9.8 is"` (pair02_even).
- **Indirect evidence that the lost subsets are marginal.** I took the 499 R8-correct subsets and checked them in post_wo/prefill, which is the same patch without decode-step injection:
  - The 441 that reproduced were correct there 83 times (18.8%).
  - The 58 lost ones were correct 0 times. If the rate were 18.8%, a run of 58 misses would happen with probability about 6e-6.
  - So the lost subsets are the ones whose repair is weakest. This fits small numerical drift between environments (fp16 on different hardware or library versions) tipping near-threshold subsets, rather than a protocol difference.
  - Head frequencies in the lost vs reproduced sets show no single-head driver. Head 14 is over-represented in the lost set (0.45 vs 0.20), and head 16 is under-represented (0.33 vs 0.49).
- **Protocol check.** The R8 `_patch_hook` (`exp_R8_threshold_analysis/run_experiment.py:86-99`) uses the same post-`self_attn` forward hook, the same left alignment `[:, :ml]`, and is active on every decode step. Greedy decoding, 20 tokens, fp16. The only difference is that R8 held the saved tensor on CPU and moved it to the device, which does not change values. The protocol is equivalent, so the drift is environmental.
- **Duplicates.** There are 493 and 489 distinct 7- and 8-subsets, as the pre-registration states. Duplicate head sets give identical outputs in every R13 cell: 1,286 distinct sets, none with more than one output. Duplicates in R8 are also consistent with each other.

## 4. Gates
- **Phenomenon.** Clean gives `" 9.8 is bigger than 9.11.\nExplanation: 9.8 is bigger"` (correct). Buggy gives `" 9.11 is bigger than 9.8.\nQ: Which is bigger: 9"` (bug). Pass. I recomputed the classification myself.
- **Head isolation.** `changed_heads: [5]`, ok. This is the recorded value; checking it needs a GPU, so I did not recompute it. The BOS max-abs-diff is 0.0 for both hooks.
- **Identity, determinism and shapes** are true in all 6 cells:
  - Determinism: the 3 outputs are identical in each cell.
  - Shapes: `[19, 1×19]`, 20 calls in total.
  - Identity in the prompt-only cells equals the unpatched output, as required.
- **All-heads equivalence.** pre_wo/prefill and post_wo/prefill give the same all-heads text, both `" 9.11 is bigger than 9.8.\nQ: …"`. `same_text`, `same_first_token` and `same_class` are all true. Pass.
- **Flag on the identity control in the all-steps cells.** The pre-registration expected this control to differ from unpatched, because bug B writes BOS into all blocks on each decode step. It did not differ: in both post_wo/all_steps and pre_wo/all_steps it equals the unpatched buggy text exactly. The hook does fire on decode steps: there are 20 calls, and decode_only produces 5 distinct texts across configs, so decode-step writes do change the text. That means writing BOS into all blocks is simply too small to change greedy output. This fits the BOS diff of 0 and the decode_only all-heads result of bug. It is recorded rather than gated and is not a defect, but the write-up should state it, because it disagrees with what the pre-registration expected.

## 5. Pre-registered decision rules
**When the reproduction gate fails, the pre-registration says:** "Below any threshold: library/hardware numerics differ from the original runs; the job finishes but no cell is interpreted until an INBOX decision." The gate failed: 942 < 950. Everything in this section is therefore **conditional on the gate being waived**.

**pre_wo/prefill_right (primary), conditional on waiver**
- L10 repair: all-heads is not correct (bug text). So the verdict is **"Layer-10 repair fails under true-head prompt-only patching with right alignment"**, and the H\* rules are reported but not interpreted.
- H\* rules, for the record:
  - (a) even-16 correct: 0, fail.
  - (b) odd-16 not correct: pass.
  - (c) even correct and odd not on 0/30, needed ≥15, fail. Odd correct 0/30 ≤ 3, pass.
  - (d) 0/500 − 0/200 = 0.00 < 0.20, fail.
- Every one of the 1,307 configs here is bug.

**pre_wo/prefill (secondary, left alignment), conditional on waiver**
- L10 repair: all-heads is not correct. Verdict: **"Layer-10 repair fails under true-head prompt-only patching with left alignment"**. H\* is not interpreted.
- H\* rules, for the record:
  - (a) even-16: 0, fail.
  - (b) odd-16 not correct: pass.
  - (c) 0/30 needed ≥15, fail. Odd correct 6/30 needed ≤3, fail.
  - (d) 0/500 − 23/200 = −0.115 < 0.20, fail.

**post_wo/decode_only (bug B alone), conditional on waiver**
- all-heads 0/1, even-16 0/1, odd-16 0/1, R8 subsets 0/1000 (0% < 10%).
- Verdict: **"BOS injection alone does not produce the repair."** Hypothesis 2 is not supported.

## 6. Anomaly checks and text sampling
- There are no empty outputs and no duplicated config names; the script asserts this.
- Output lengths are 45–80 characters. Every output is cut at 20 tokens (`MAX_NEW=20`), mid-sentence, but the answer sentence is always complete before the cut.
- Across all 7,842 generations there are only **13 distinct texts**. I checked the classification of every one, not just a sample:
  - Correct: `"9.8 is bigger than 9.11.\nExplanation: 9.8 is bigger"` (587), `"9.8 is bigger than 9.11.\nQ: …"` (72), `"9.8 is bigger than 9.11.\nExplanation: 9.11 is smaller"` (1).
  - Bug: the other ten all begin `"9.11 is bigger"`.
  - None needed the first-line fallback branch of the classifier, and none is mislabelled.
- Random sample of 15 (seed 0):
  - 7 counted correct: all_steps even8_410/474/184/453 and even7_131, post_wo/prefill even7_294, pre_wo/prefill rand8_149.
  - 8 counted not correct: decode_only even8_010/388, prefill_right even7_346/even8_295, post_wo/prefill even7_480/even8_238, pre_wo/all_steps even7_119, pre_wo/prefill even8_424.
  - All are labelled correctly.
- The experiment file's Results section is still "(none yet)".

## Saving this report
`scripts/check_reviews.py --save-report` does not read stdin. Its usage is `--save-report AGENT_ID OUT --verdict VERDICT`. It extracts the report verbatim from the finished subagent's transcript and refuses to run while the agent is still running. The verifier therefore cannot save its own report. After this agent finishes, run `python scripts/check_reviews.py --list-agents` to get its ID, then `python scripts/check_reviews.py --save-report <id> docs/reviews/11-21_llama8b_decimal_true_head_patching_verify.md --verdict "VERIFIED-WITH-FLAGS"`. A copy of this report is at `/private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/report.md`. I did not edit any repo files.
