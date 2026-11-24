Reviewer: fresh subagent (Claude Code)
Agent-ID: a027b451cf44c8d49
Verdict: PASS WITH CAVEATS
Transcript-SHA256: 2124cefed9af1c6c4671173894839a5e0b10d15e25e474cf20a503831d2531cd

## Brief given

You are an independent verifier for a pre-registered experiment in the repo ~/github/MATS. Do not trust any summary; recompute from raw files. Do not edit any file in the repo. Use `python3 -I` for scripts (write any scripts to /private/tmp/claude-502/r21verify/, not the repo).

Read first:
- Pre-registration: ~/github/MATS/docs/experiments/11-24_llama8b_gpu_margin_offset.md (decision rules, which-margin rule, gates)
- Measurement script: ~/github/MATS/9_8_research/reviewer_experiments/exp_R21_gpu_margin/run_experiment.py
- Analysis script: ~/github/MATS/9_8_research/reviewer_experiments/exp_R21_gpu_margin/analyze.py
- Raw results: .../exp_R21_gpu_margin/results/r21_a100_24375606/ and r21_l40s_24375607/ (run.json, results_post_wo__prefill.json, results_post_wo__all_steps.json)
- R13 reference results (source of D): .../exp_R13_true_head_patching/results/r13_24084222 (A100) and r13_24014365 (L40S), results_post_wo__{prefill,all_steps}.json
- analyze.py output: .../exp_R21_gpu_margin/results/verdicts.json

Claims to check independently (write your own code; do not import analyze.py):
1. Both runs: all gates true, finished, not smoke runs, same r21_script_sha256 / r21_git_sha / token_ids, R13 sha 1ce5b0cc2e2cdc74c21de37de5340cd00f5b14e27ae13cf6893d2e09f280348d. Texts in each cell equal R13's texts on the same GPU for all 1307 configs.
2. D (configs whose R13 `correct` differs between GPUs) = 40 prefill, 60 all_steps; every D config is correct on A100 and buggy on L40S.
3. Per cell, on m: Δ = m_A100 − m_L40S. Claimed prefill: 1291 positive, 0 negative, 16 zero; median 0.03125; all_steps: 1271/0/36, median 0.03125. On lens_final (lens[-1]): all 1307 Δ > 0 in both cells, medians ≈ 0.0334 (prefill) and ≈ 0.0290 (all_steps).
4. The which-margin rule picks m in both cells (zero share < 20%, c ≠ 0), so the pre-registered verdict is rule 2 "offset present, does not alone predict which configs flip" in both cells. Recompute recall/precision of D_pred = {m_L < 0 < m_L + c}: claimed prefill 17/40 hit, D_pred 18; all_steps 28/60, D_pred 29. On lens_final, all_steps meets rule 1 (recall 0.8, precision 0.8) and prefill does not (recall 0.775) — but lens_final is not the primary margin.
5. Ties: claimed 25 of 40 prefill D configs and 37 of 60 all_steps D configs have m_A100 exactly 0 (logit8 == logit11 in fp16) with argmax = "8". Check the token ids of "8" and "11" and whether argmax chose the lower id on a tie. Check whether any non-D config has a tie on either GPU, and what it decoded to.
6. Unpatched: m_A100 = −0.890625, m_L40S = −0.9375, Δ = 0.046875.
7. Logit-lens median-Δ profile (index 0..32 of `lens`): claimed ≈ 0 (|·| ≤ 0.012) through index 22 and ≈ 0.05 from index 23 on, with a peak ≈ 0.13 at index 30, in both cells. Index l is hidden_states[l]; index 0 is embeddings, so index 23 = output of decoder layer 22 (0-based). Confirm that indexing from the script.
8. Observed reverse flips (m_A < 0 < m_L) = 0; expected under symmetric H0 ≈ 7.0 (prefill, m) / 14.7 (all_steps, m). Recompute the expected number with your own implementation of the pre-registered definition.
9. Determinism block: 50/50 identical on both GPUs, unpatched identical.

Also look for anything wrong I have not asked about: bugs in analyze.py relative to the pre-registration, mismatched config keys, off-by-one in t_dec, or anything that would change the verdict.

Report: for each claim, CONFIRMED / WRONG (with your number) / CANNOT CHECK. Then a list of other problems, most severe first. Then a one-line overall verdict: PASS, PASS WITH CAVEATS, or FAIL. Be concise.

## Report

## R21 independent verification

I recomputed everything with my own code (`/private/tmp/claude-502/r21verify/v.py` and `v2.py`), reading the raw R21 and R13 JSONs. I did not import `analyze.py` and edited nothing in the repo.

### Claims

1. **CONFIRMED.**
   - Both runs: all 8 gates are true, `all_gates_ok` is true, both are finished and neither has `--limit`.
   - Both runs share the same `r21_script_sha256` (`c76475e9…`, which matches the file on disk), `r21_git_sha` `e89ad0a` and `token_ids` `[23, 806]`. The R13 sha is `1ce5b0cc…` in both provenances and on disk.
   - All 1307/1307 texts equal R13's texts on the same GPU, in both cells and on both GPUs. The config key sets are identical across all four files, and `heads` agree across GPUs.
   - `analyze.py` has no diff from `e89ad0a`. Its recorded sha equals the file's.
2. **CONFIRMED.** D is 40 for prefill and 60 for all_steps. Every D config is correct on A100 and buggy on L40S, and R13's `bug` flag is true for all of them on L40S.
3. **CONFIRMED.**

   | Margin | Cell | pos / neg / zero | Median Δ | Δ range |
   |---|---|---|---|---|
   | m | prefill | 1291 / 0 / 16 | 0.03125 | [0, 0.078125] |
   | m | all_steps | 1271 / 0 / 36 | 0.03125 | [0, 0.078125] |
   | lens_final | prefill | 1307 / 0 / 0 | 0.033419 | — |
   | lens_final | all_steps | 1307 / 0 / 0 | 0.029005 | — |

4. **CONFIRMED.**
   - Zero share is 1.2% (prefill) and 2.8% (all_steps), and c ≠ 0, so the rule picks m. The verdict is rule 2 in both cells.
   - On m: prefill hits 17/40 with D_pred = 18 (recall 0.425, precision 0.944). all_steps hits 28/60 with D_pred = 29 (recall 0.467, precision 0.966).
   - On lens_final: all_steps has recall 0.800 and precision 0.800, which meets rule 1. Prefill has recall 0.775 and precision 0.705, so it is rule 2.
5. **CONFIRMED.**
   - The token ids are "8" = 23 and "11" = 806.
   - 25/40 prefill and 37/60 all_steps D configs have logit8 == logit11 exactly on A100, all with argmax 23, so the tie went to the lower id. No D config ties on L40S.
   - Non-D ties exist and all decode " 9.8 is…", i.e. to "8":
     - A100: 1 in prefill (`even7_128`) and 2 in all_steps (`even8_015`, `rand8_106`).
     - L40S: 15 in prefill and 28 in all_steps (including `all_heads` in all_steps).
6. **CONFIRMED.** m_A100 is −0.890625 and m_L40S is −0.9375, so Δ = 0.046875. On lens_final the unpatched Δ is 0.04380.
7. **CONFIRMED.**
   - Through index 22 the largest |median Δ| is 0.0117 (prefill) and 0.0096 (all_steps). From index 23 on, the smallest value is 0.0334 and 0.0290.
   - Index 23 is 0.052 (prefill) and 0.046 (all_steps). The peak is at index 30: 0.1287 and 0.1140.
   - The indexing checks out:
     - `t_dec` = 3 for every config. `gen_ids[:3]` = [220, 24, 13] (" 9.") and `gen_ids[3]` equals argmax for all configs, so there is no off-by-one.
     - `hidden_states[3]` is the decode step whose output is `logits[3]`.
     - `lens[0]` is one identical value in every config, on both GPUs and in the unpatched run, as the embedding of "." would be.
     - The first index that varies across configs is 12 for prefill (the layer-10 patch reaches the current position through layer 11's attention) and 11 for all_steps (the layer-10 output is patched directly). Both match "index l = output of decoder layer l−1", so index 23 = layer 22 (0-based).
     - The lens gate (|lens[32] − m| ≤ 0.0156) confirms index 32 is the final normed state.
   - I could not read the HF source because transformers is not installed locally.
8. **CONFIRMED.** Observed reverse flips are 0. My implementation of the pre-registered expectation gives 7.028 (prefill, m), 14.733 (all_steps, m), 14.011 and 33.666 (lens_final).
9. **CONFIRMED.** Determinism is 50/50 identical with the unpatched run identical, on both GPUs.

### Other problems, most severe first

1. **The pre-registered D_pred rule mis-handles exact ties, and fixing it would change the label on m. This is a flaw in the pre-registration, not an `analyze.py` bug.**
   - D_pred uses `m_L < 0 < m_L + c` with a strict upper bound. On A100 a tie (m = 0) decodes to "8" (correct), as shown in claim 5. So a config with m_L + c == 0 is predicted to flip but is left out.
   - That is exactly where most misses sit, because c = 0.03125 and many D configs have m_L = −0.03125. Prefill has 19 of its 23 misses there; all_steps has 29 of its 32.
   - With `0 <= m_L + c`, the m results become:
     - prefill: D_pred = 47, hit 36, recall 0.90, precision 0.77
     - all_steps: D_pred = 72, hit 57, recall 0.95, precision 0.79
   - Both cells would then meet rule 1 ("shared offset explains"). This is post hoc and cannot replace the pre-registered verdict. But the write-up should say that rule 2 on m comes from tie handling, not from Δ spreading as widely as c. The same strict bound is also used in the permutation control and in `recall_at_quartiles`.
2. **The H0 reverse-flip expectation leaves out L40S configs with m_L == 0** (15 in prefill, 28 in all_steps). They all decode "8" and would reverse-flip under any negative Δ. Leaving them out understates the expectation, so 7.0 and 14.7 are conservative. Counting them would roughly double the figures: my rough estimate is about +7 for prefill and +14 for all_steps, which I did not compute exactly. The direction only strengthens the case against H0.
3. **`D_sign_ok` (15/40, 23/60) is low only because of ties.** All 40/40 and 60/60 D configs satisfy m_A ≥ 0 > m_L. m_L on D runs from −0.046875 to −0.015625, and m_A from 0 to 0.03125.
4. **The `top2_gap` field is misleading on ties.** `torch.topk` lists 806 first (gap 0.0) while `argmax` returns 23. All 26 (A100 prefill), 15, 39 and 28 mismatches are exactly the tie configs. `argmax` and the class are correct.
5. **The lens_final all_steps rule-1 result sits exactly on the threshold** (48/60 = 0.800), so it is fragile. It is also not the primary margin.
6. **Bookkeeping.** `results/` (including `verdicts.json`) is untracked in git, and the pre-registration's Results section still reads "None yet".

I found no mismatched config keys, no `t_dec` off-by-one, and no departure of `analyze.py` from the pre-registered definitions (the which-margin rule uses > 20% zeros, as written).

**Overall verdict: PASS WITH CAVEATS.** Every claimed number reproduces, but rule 2 on m is caused by the pre-registration's strict inequality meeting fp16 ties. With ties counted as flips, m meets rule 1 in both cells, and the write-up must say so, labelled as post hoc.
