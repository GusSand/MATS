Reviewer: fresh subagent (Claude Code)
Agent-ID: ab6531662a08f9525
Verdict: VERIFIED-WITH-FLAGS
Transcript-SHA256: 599fc9df35e626e2ed555f97e1d1751dcfcd183dd04b5d38e2897af9c162be2f

## Brief given

You are an independent verifier for a pre-registered experiment (R16) in the repo ~/github/MATS. Read-only: do NOT edit any files, do NOT ssh anywhere. Your job: check that the reported results are correct and that the verdicts follow the pre-registered rules, and flag anything wrong or overclaimed.

Inputs:
- Pre-registration: docs/experiments/11-22_llama8b_position0_gpu_gap.md (read the whole thing; "Results" is not yet written).
- Script: 9_8_research/reviewer_experiments/exp_R16_position0/run_experiment.py (and test_decide_placebo.py). It imports R13's script by path: 9_8_research/reviewer_experiments/exp_R13_true_head_patching/run_experiment.py.
- Outputs: 9_8_research/reviewer_experiments/exp_R16_position0/results/r16_a100_24095012/ (A100, job 24095012) and results/r16_l40s_24104901/ (L40S, job 24104901). Each has run.json and per-cell results_post_wo__<cell>__<variant>.json.
- R13 reference outputs: 9_8_research/reviewer_experiments/exp_R13_true_head_patching/results/r13_24014365/ (L40S) and r13_24084222/ (A100).

Things to check, recomputing from per-config JSON where possible with plain python3 (no torch is installed locally; for .pt files you can run `uv run -q --no-project --with torch python -I <script>` with the script in /private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/):
1. Copy gate: each GPU's copy texts equal R13's for that GPU on all 1,307 configs, both cells (post_wo/prefill, post_wo/all_steps).
2. D sets: configs whose class differs between R13 L40S and A100 in each cell (expected 40 prefill, 60 all_steps). Recompute.
3. For keep0 (A100), inject (L40S), placebo (both), donorA100 and a100clean0 (L40S): recompute moved (exposed D configs taking the other GPU's class), N_new (configs outside D whose class changes vs this run's copy, with direction), R8-subset correct counts (copy/variant/other GPU: expected prefill 120/119/83 on A100), and the number of texts equal to the other GPU's R13 texts. Compare with the logged values in run.json.
4. Apply the pre-registered decision rules yourself to keep0 and inject in each cell, and check the labels match. Check whether `interpreted` is set correctly. NOTE: the L40S inject exactness gate failed (`inject_check.exact_delta: false`). An offline check found the injected position-0 delta exact on 1,462 of 1,466 nonzero coordinates, with sum|diff−dA| = 9.6e-6 vs sum|dA| = 9.6e-3; L40S buggy0 differs from A100 buggy0 on 2,681 coordinates, so the A100 delta lands on a different fp16 grid. Confirm this independently from the .pt files (results/r16_a100_24095012/position0.pt with keys clean0,buggy0,delta_fp32,clean_full; results/r16_l40s_24104901/position0_l40s.pt with clean0,buggy0) and state what the pre-registration says must happen to the inject labels when this gate fails. Also check that the L40S job exited 2 for this reason only.
5. Note the "unexposed identical" gate is vacuous because d0 is nonzero in all 32 blocks (all configs exposed). Confirm.
6. Check the placebo validity (unplaced_step_fraction ≤ 0.01) on each GPU.
7. Anything else that looks wrong: mismatched provenance (sha256s: R16 bd47abab…, R13 1ce5b0cc…), wrong node, etc.

Report: a top-line verdict on its own line, exactly one of VERIFIED, VERIFIED-WITH-FLAGS, or NOT-VERIFIED. Then a table of recomputed vs logged numbers per cell/variant, the rule-by-rule label derivation, the inject-gate finding, and any flags. Be concise and concrete.

## Report

VERIFIED-WITH-FLAGS

Every logged number matches my recomputation from the per-config JSON and the .pt files. Both keep0 labels follow the pre-registered rules. The inject labels follow the rules arithmetically, but the inject exactness gate failed, so under the pre-registration they do not count, and run.json correctly records `interpreted: false` for both L40S cells. The flags below are about how the results should be read. None of them is a computation error.

**Gates and provenance (all checked)**
- **Copy gate:** this run's texts equal R13's for the same GPU on 1307/1307 configs, in both GPUs and both cells. Heads and classes also match.
- **D sets (recomputed from R13 L40S vs A100):**
  - prefill: |D| = 40 (37 of them R8 subsets).
  - all_steps: |D| = 60 (58 R8).
  - Every D config is correct on A100 and not correct on L40S.
  - R13 R8-subset correct counts: prefill 120 vs 83, all_steps 499 vs 441.
- **Hashes:**
  - The local R16 script is `bd47abab…`; R13 is `1ce5b0cc…`. Both run.json files record these.
  - The R13 reference-file sha256s in run.json match the local files.
  - The L40S `delta_file_sha256` (`8a138557…`) equals the local A100 `position0.pt`.
- **Nodes:** A100 ran on ga004 and L40S on gl044. The jobs ran at different git commits (cd272dc and 61dd72b), but the only difference is a one-line message in `run_l40s.sbatch`, and the script hash is the same in both.

**Recomputed vs logged** (identical in every field; `->corr` / `->notcorr` = direction of the N_new flips)

| GPU / cell / variant | moved / n | N_new (->corr / ->notcorr) | R8 correct: variant / own copy / other GPU | texts equal to other GPU's R13 |
|---|---|---|---|---|
| A100 prefill keep0 | 6/40 | 5 (5/0) | 119/120/83 | 1263 |
| A100 prefill placebo | 10/40 | 8 (8/0) | 118/120/83 | 1267 |
| A100 all_steps keep0 | 10/60 | 7 (7/0) | 497/499/441 | 1243 |
| A100 all_steps placebo | 12/60 | 8 (8/0) | 496/499/441 | 1242 |
| L40S prefill inject | 6/40 | 3 (0/3) | 84/83/120 | 1265 |
| L40S prefill placebo | 5/40 | 2 (0/2) | 86/83/120 | 1266 |
| L40S prefill donorA100 | 5/40 | 2 (0/2) | 86/83/120 | 1267 |
| L40S prefill a100clean0 | 4/40 | 4 (0/4) | 83/83/120 | 1261 |
| L40S all_steps inject | 9/60 | 3 (0/3) | 448/441/499 | 1242 |
| L40S all_steps placebo | 8/60 | 6 (0/6) | 445/441/499 | 1236 |
| L40S all_steps donorA100 | 4/60 | 6 (0/6) | 441/441/499 | 1235 |
| L40S all_steps a100clean0 | 5/60 | 6 (0/6) | 442/441/499 | 1234 |

The R8/R1 reproduction counts for the copy cells match the pre-registration: A100 1000/200/60, L40S 942/199/59.

**Label derivation** (exposed D = all of D in every cell, because all configs are exposed)

| Cell | Check | Derived label | Logged label | Counts? |
|---|---|---|---|---|
| A100 prefill keep0 | n=40, moved=6. Rule 3 fails (5·6=30 < 32). Rule 4 applies (30 ≤ 40). N_new=5 > 4. | "perturbs, does not explain" | same | Yes (`interpreted: true`) |
| A100 all_steps keep0 | n=60, moved=10. Rule 4 applies (50 ≤ 60). N_new=7 > 4. | "perturbs, does not explain" | same | Yes (`interpreted: true`) |
| L40S prefill inject | n=40, moved=6. Rule 4 applies. N_new=3. | "does not explain the gap" | same | No |
| L40S all_steps inject | n=60, moved=9. Rule 4 applies (45 ≤ 60). N_new=3. | "does not explain the gap" | same | No |

- The invalid placebo does not affect any label: it only blocks positive labels, and rule 4 never consults the placebo.
- "Shown from both sides" cannot apply: neither variant explains the gap, and inject is not interpreted.

**Inject gate (confirmed independently from the .pt files)**
- On A100, `delta_fp32` equals clean0 − buggy0: 1,466 nonzero coordinates, max 6.1e-5, every one of the 32 blocks has 38–54 nonzero.
- On L40S, clean0 equals buggy0 bitwise, so d0 = 0 there.
- L40S buggy0 differs from A100 buggy0 on 2,681 coordinates (clean0 on 2,673).
- The injected delta is exact on 1,462 of 1,466 nonzero coordinates. Zero coordinates are unchanged. sum|diff − dA| = 9.596e-6 against sum|dA| = 9.618e-3.
- The four failing coordinates are 1034, 1103, 2810 and 3260. At each, the A100 step is smaller than or off the fp16 grid at the L40S buggy0 value. For example, at 2810 the step is 7.6e-6 but the result moved 1.5e-5. Coordinate 2810 alone carries most of the error.
- **What the pre-registration requires:** inject exactness is a gate, and labels count only if every gate holds ("on L40S, d0 = 0 and inject exactness"). So both inject labels are uninterpreted and may be reported only as raw descriptive counts. This also blocks the "shown from both sides" conclusion.
- **Exit code:** run.json shows exactly one false gate, `exact_delta`. The other L40S gates all hold: capture equals live, d0 = 0, both copy gates, and the vacuous unexposed gates. `finished` is written, so the job did not crash, and the code then exits 2 when `all_gates_ok` is false. I could not see the Slurm log or exit code myself (no ssh). The exit-2 claim rests on that code path and on the research journal entry.

**Other checks**
- **Unexposed-identical gate (item 5):** confirmed vacuous. d0 is nonzero in all 32 blocks, every config has at least one head, so `n_exposed_configs` = 1307 and `unexposed_text_identical` = [0, 0] everywhere. It passes trivially and provides no implementation check.
- **Placebo validity (item 6):** invalid on both GPUs. Re-running `make_placebo` gives 948 moves, 518 unplaced, unplaced_step_fraction 0.1498 > 0.01 on each GPU. The recomputed moves and unplaced lists match the logged ones exactly.

**Flags**
1. **Inject labels must not be reported as verdicts.** "does not explain the gap" on L40S is descriptive only. The gate failure is small (0.1% of the step mass, 4 coordinates), and it arises because the A100 delta cannot be placed exactly on the L40S fp16 grid at those coordinates. A writeup may say that, but not upgrade the label.
2. **The placebo is invalid on both GPUs** (15% of the step mass unplaced), so the boundary-fragility baseline is not available as a valid comparison. The pre-registration anticipated this for dense differences. Report placebo counts as raw only.
   - Descriptively, the placebo moves about as many D configs as the test variant: 10 vs 6 and 12 vs 10 on A100, 5 vs 6 and 8 vs 9 on L40S.
   - Its N_new is as large or larger than keep0's (8 vs 5, 8 vs 7). This is consistent with the boring hypothesis, but that claim cannot be made formally.
3. **The "perturbs" half of the keep0 labels needs context.**
   - It comes from N_new = 5 and 7, just above the threshold of 4, and the placebo perturbs at least as broadly. So the label says nothing specific about position 0.
   - The outside-D flips run opposite to the pre-registered prediction. Keep0 flips went not-correct → correct (5/0, 7/0), where the hypothesis predicted correct → not-correct. Inject flips went correct → not-correct (0/3), where it predicted the reverse.
4. **Not the same A100 as R13.** The R16 A100 GPU UUID is `GPU-d7fd4247…`; R13's A100 run used `GPU-e4ae6c00…`, a different card on the same node ga004. `same_gpu_string_as_r13` only compares the model name, so it reports true. This is harmless here, since the copy gate reproduces R13 1307/1307, but it should not be described as "the same GPU". The L40S UUID matches R13.
5. **The donor-side result points away from the hypothesis.** donorA100 and a100clean0 move only 4–5 of 40 and 4–5 of 60 D configs, and leave the L40S R8 count at 86/83 and 441/442 respectively, against A100's 120 and 499. That fits the gap arising in the receiving run's arithmetic rather than in the donor. These cells carry no label by design.

My verification scripts are `/private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/recompute.py` and `/private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/pt_check.py`.
