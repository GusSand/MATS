Reviewer: fresh subagent (Claude Code)
Agent-ID: a86be01e7daf06c8c
Verdict: VERIFIED-WITH-FLAGS
Transcript-SHA256: cb5bce97fcbbcccf9e9530ab22939467aeb694a9b5dd9f997b94a7d676b343e7

## Brief given

You are an independent verifier for a finished experiment. Recompute everything from the raw files, and trust no summary. Do not edit repo files except as noted at the end.

Repo: ~/github/MATS.
- Read the pre-registration first: docs/experiments/11-22_pythia_gemma_true_head_patching.md. It has gates, ordered verdict rules and a late-training rule list.
- Script: 9_8_research/reviewer_experiments/exp_R14_cross_model_true_heads/run_experiment.py. Use its own functions (`classify`, `summarize`, `rules`, `late_training_verdict`) to recompute. `--selftest` runs without a model.
- Raw results (Slurm job 24092087, A100): 9_8_research/reviewer_experiments/exp_R14_cross_model_true_heads/results/r14_24092087/
  - subfolders pythia160m/, gemma2b/, gemma2b_it/ and pythia160m_dynamics/
  - the job log
- Original September results:
  - 9_8_research/cross_model_validation/pythia_160m_validation_20250926_174151.json
  - 9_8_research/cross_model_validation/gemma_2b_validation_20250926_174510.json
  - 9_8_research/training_dynamics_analysis/results/pythia_training_dynamics_20250926_191344.json

Tasks:
1. **Provenance.** Check that the script sha256 equals run.json, that git_sha 2504146 exists, and record the GPU and the weights commits.
2. **Screens.** For every model and checkpoint, reclassify each screen output and check the pair rule.
   - **Look closely at gemma2b_it F5:** "Sure, here's the difference between 9.11 and 9.99". It is classified *bug*, and that made the pair F1 → F5. Is that a classifier mislabel, and does the output actually claim 9.11 is bigger?
   - State what the pre-registered rule mechanically gives, and separately whether the phenomenon is real for Gemma-IT. List any other outputs where you disagree with the classifier.
3. **Reproduction gates** for pythia160m and gemma2b. Recompute them against the September references with `orig_fixed`. Pythia's gate is recorded as failed (7/10): confirm this and show the disagreeing outputs.
4. **Every cell.** For each cell, including pythia's fallback layer-0 cell and the full-subset dynamics checkpoints:
   - recompute `summarize` and `rules` from the raw configs;
   - check the gates (identity, determinism, shapes, head isolation, all-heads equivalence, n_effective);
   - give the pre-registered verdict;
   - for the fallback cell, report minimal sufficient sets and necessary heads.
5. **Dynamics.**
   - Check `phenomenon_present` per checkpoint.
   - Check the old-protocol reproduction against TD_REF.
   - Check `matches_main_run` at step143000 by comparing all 4,095 outputs yourself.
   - Give the late-training verdict.
6. **Sanity of the rules.** State what each pre-registered rule concludes for each model. Then state what is and is not interpretable, given the failed Pythia reproduction gate (the pre-registration says: not interpreted until an INBOX decision).
7. **Anything odd.**

Report as markdown. Put the verdict first: VERIFIED, VERIFIED-WITH-FLAGS, or DISCREPANCIES. Return the full report as your final message.

## Report

# R14 independent verification (Slurm 24092087)

## Verdict: VERIFIED-WITH-FLAGS

I recomputed every recorded number from the raw files using the script's own `classify`, `summarize`, `rules` and `late_training_verdict`, and every one matches: screens, pairs, both reproduction gates, all five subset cells, both full-subset checkpoints, all 11 checkpoint records and the late-training verdict. Across all 20,895 subset outputs, re-running the classifier changes zero labels, and `--selftest` passes locally. No pre-registered verdict changes under the mechanical rules.

The flags are about labels and interpretation, not arithmetic:
1. **Gemma-IT F5 is a classifier mislabel.** Read correctly, Gemma-IT has no format-dependent bug, so the pre-registered outcome would be "no phenomenon", not "repair fails".
2. **The Pythia fallback layer was triggered by an ambiguous output.** The layer-0 all-heads output that counts as "correct" is a multiple-choice listing.
3. **The Pythia reproduction gate fails (7/10)**, as recorded. Pythia therefore stays uninterpreted until an INBOX decision.
4. **The training-dynamics reproduction is 9/11, and both misses are the two late checkpoints.** The September "parity appears at 143k" signal does not reproduce, even under the old protocol.

No repo files were edited. My scratch scripts are `/private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/v1.py` through `v5.py`.

---

## 1. Provenance
- The sha256 of `run_experiment.py` on disk and at `git show 2504146:` is `20819e0620464ec3b97407a897c71c48f23e26a672646c4b2e09bed92bc20fc2`. It equals `script_sha256` in all four `run.json`/`dynamics.json` files.
- Commit `2504146` exists ("R14: design review PASS (round 5)"), and `git_sha` = `2504146` everywhere.
- GPU: `NVIDIA A100-SXM4-80GB, GPU-c4dfba88-5a6c-d39f-b76e-ce70fc1b60db, driver 610.43.02`. Host `ga014.hpc.nyu.edu`, torch 2.13.0+cu130, transformers 5.15.1, sdpa attention.
- Dtypes are as pre-registered: fp16 for Pythia and Gemma-2B, bf16 for Gemma-IT.
- Weights commits:

| Model / revision | Commit |
|---|---|
| `pythia-160m@main` | `50f5173d932e8e61f858120bcb800b97af589f46` |
| `gemma-2b@main` | `9cf48e52b224239de00d483ec8eb84fb8d0f3a3a` |
| `gemma-2b-it@main` | `96988410cbdaeb8d5093d1ebdc5a8fb563e02bad` |
| Pythia `step143000` | `b56d9bee…` |
| Pythia `step120000` | `3df7d2c5…` |
| Pythia `step1000` | `f711eaa4…` |
| Pythia `step2000` | `a4817f95…` |
| Pythia `step4000` | `b0f0e024…` |
| Pythia `step8000` | `efd022a4…` |
| Pythia `step16000` | `13af8625…` |
| Pythia `step32000` | `dfead30e…` |
| Pythia `step64000` | `aac0de45…` |
| Pythia `step80000` | `c2b7e67a…` |
| Pythia `step100000` | `c507d0e6…` |

  The `main` and `step143000` commit hashes differ (different branches), but all 4,095 outputs are byte-identical, so the weights are the same (see §5).

## 2. Screens and pair rule
Every screen class, `orig_fixed` and `td_fixed` value recomputes identically, and every pair recomputes as recorded.

| Model | Screen classes | Pair (mechanical rule) |
|---|---|---|
| pythia160m | F1 bug, F2 bug, F3 correct, F4–F6 neither | F3 → F2 |
| gemma2b | F1 neither, F2 bug, F3 bug, F4 neither, F5 neither, F6 bug | none → F2: no donor, so "no phenomenon" |
| gemma2b_it | F1–F4 correct, F5 bug, F6–F7 correct | F1 → F5 |

**Gemma-IT F5: `"\n\nSure, here's the difference between 9.11 and 9.99"`**
- **Why it is labelled bug:** there is no comparison phrase, so `answer_line` decides. The first line names 9.11, and "9.99" does not match the 9.8 regex, so the line counts as naming 9.11 alone.
- **What the output actually says:** it never claims 9.11 is bigger. It is the start of a "difference" (subtraction) answer, it misquotes 9.8 as 9.99, and it is cut off at 20 tokens. This is a classifier mislabel.
- **What the rule mechanically gives:** pair F1 → F5, then a Gemma-IT verdict of "Layer-8 repair fails; `gemma_parity_independent` = False".
- **Whether the phenomenon is real:** no. Every other Gemma-IT format, including the chat template F7, answers "9.8 is bigger" (or "larger"/"greater"). Read correctly, the screen has no (correct, bug) pair, and the pre-registered outcome would be "Gemma-2B-IT shows no format-dependent bug on these formats". The Gemma-IT cells patch toward a non-bug and should not be cited as a failed repair.
- **A curiosity in the patched outputs:**
  - Right cell: 253/255 subsets reproduce the unpatched F5 text exactly. The other 2 (`h0_1_2`, `h0_1_2_6`) give a genuine bug, "The answer is 9.11. The number 9.11 is greater".
  - Left cell: the same genuine bug appears in `h0_1_6` and `h0_1_2_6`.
  - Layer sweep: layer 3 gives the same genuine bug.
  - So patching from the correct donor sometimes *creates* the bug.
  - Also in the left cell: 8 subsets give `"…difference:\n\n9.11 - 9.99"`. That is a subtraction, but it is also labelled bug.

**Other labels I disagree with or consider ambiguous:**
- **Pythia, ambiguous number lists.** The answer-line rule labels lists that alternate the two numbers. Examples: L0 `" 9.8\nB: 9.11\nC: 9.8\nD: 9."` and the 120k recipient F2 `" 9.11\n\nA: 9.8\n\nA: 9.11"`. Outputs naming both numbers with no comparison phrase:

  | Cell | Labelled correct | Labelled bug |
  |---|---|---|
  | L0 fallback | 1,040 | 125 |
  | step120000 subsets | 899 | 2,772 |
  | L6 left | 50 | 0 |
  | L6 right | 6 | 0 |

  Step100000 F2 and step16000 F2 are the same kind of output.
- **Pythia, code labelled as answers.** For example, `"$a = 9.11;"` is labelled bug (13 outputs). L6 right has 19 code-like outputs labelled bug and 8 labelled correct.
- **step2000 F1 `"-9.11"`** is counted as 9.11, because the lookbehind does not exclude a minus sign. It has no effect (no donor at that checkpoint).
- **The screen labels themselves are reasonable.** Pythia F1 bug, Pythia F3 correct, gemma2b F3 bug (repeats the question after answering 9.11), gemma2b F6 bug ("The greater number is 9.11") and gemma2b F1/F5 neither (reasoning cut off) all look right to me.

**Strict re-scoring (my own sensitivity check, not pre-registered).** I treated ambiguous and code outputs as neither and re-ran the rules:
- L6 right, L6 left and 143k: verdicts unchanged.
- L0: becomes "repair fails" (the all-heads output is a listing). Under this reading the layer sweep's "correct" at L0 and L2 is also a listing (L2: `" 9.8\n\nA: 9.11\n\nA:"`), so the fallback cell would not have been triggered at all.
- 120k: still "parity not supported".

## 3. Reproduction gates (`orig_fixed` against the September JSONs)
The `ORIG_REFS` and `TD_REF` values in the script match the September files exactly.

**Pythia: 7/10, failed (confirmed).** The must-agree config `all_even` fails. The disagreeing outputs:

| Config | Sept | Now (`orig_fixed`) | Output now | September output |
|---|---|---|---|---|
| `all_even` [0,2,4,6,8,10] L6 | 1 | 0 | `"\n\nI would say that the answer is 9.11.\n\nA:\n\nI"` | "I would say that the answer is 9.8." |
| `second_half_even` [6,8,10] L6 | 1 | 0 | `"\n\nI think the answer is 9.11.\n\nA:\n\nI think the"` | "I think the answer is:\n\n9.8" |
| `sweep_L6` (same config as `all_even`) | 1 | 0 | identical to `all_even` | same as `all_even` |

- `sweep_L6` and `all_even` are the same configuration, so one miss is counted twice. On distinct configs the score is 7/9 (78%); the gate fails either way.
- The unpatched screen outputs are prefixes of the September samples, so the base model reproduces and the patched runs diverge. Curiously, the all-12-heads `x_all` output now reproduces September's `all_even` text verbatim ("…answer is 9.8."), so the result sits on a one-token knife-edge.
- The original hook code (`test_pythia_160m.py:67-126`) is equivalent to `orig_cell`. A transformers or attention-backend change is the plausible cause; I did not verify this.

**Gemma-2B: 15/15, passed**, and both must-agree configs (`even4`, `odd4`) agree. Two caveats:
- Under R14's strict classifier, the September "fixed" outputs (`" 9.8\n\nQ: …"`) are also correct.
- `orig_fixed` is lenient: `x_single_{1,3,5,7}` (" 9.11\n\nQ: Which is bigger: 9.8 or 9.") count as fixed only because the repeated question contains "9.8".

## 4. Every cell (recomputed `summarize` and `rules` equal both the results file and `run.json`)
All gates pass in every cell:
- Identity: the patched-with-own-activations output equals the unpatched output, and that equals the screen recipient's output.
- Determinism: one distinct output across 3 runs.
- Shapes: `[T, 1×19]`.
- Head isolation: changed heads = [1] for Pythia and Gemma-IT.
- All-heads equivalence: `same_text` = True for Pythia L6 and Gemma-IT L8. It was only checked at the original layer, not at L0.
- n_effective: Pythia right 14 / left 15; Gemma-IT right 14 / left 14 (the BOS pair is the no-op). No cell is vacuous.

| Cell | n_correct | all / even / odd | even-only vs odd-only | D(e−o), D_contig, max_D | (a)(b)(c)(d) | Verdict |
|---|---|---|---|---|---|---|
| pythia `pre_wo__prefill_right` L6 (primary) | 95 | F / F / F (bug) | 5 vs 4 | +1, 2, 10 | F T T F | repair fails at this layer; head rules not interpreted |
| pythia `pre_wo__prefill` L6 | 247 | F / F / F (bug) | 2 vs 3 | −1, 0, 12 | F T F F | repair fails at this layer |
| pythia `pre_wo__prefill_right__L0` (fallback) | 2,990 | **T** / F / **T** | 29 vs 48 | −19, 27, 50 | F F F F | **parity not supported** |
| gemma2b_it right L8 | 0 / 255 | F / F / F | 0 vs 0 | 0, 0, 0 | — | repair fails; `gemma_parity_independent` = False |
| gemma2b_it left L8 | 0 / 255 | F / F / F | 0 vs 0 | 0, 0, 0 | — | same |
| dyn step143000 (4,095) | 95 | F / F / F | 5 vs 4 | +1, 2, 10 | F T T F | repair fails at this layer |
| dyn step120000 (4,095) | 1,008 | T / F / T | 0 vs 23 | −23, 17, 30 | F F F F | parity not supported |

**Pythia L0 fallback (layer sweep: all-heads correct at L0 and L2; first = 0):**
- Minimal sufficient sets: `[0], [3], [4], [7], [9], [10], [11], [5,6,8]`. Necessary heads: none (empty intersection).
- Single heads that repair: [0, 3, 4, 7, 9, 10, 11].
- Top split: |D| = 50 for {0,3,5,9,10,11}. The balance leans odd (majority-rate gap −0.18).
- The all-heads "correct" output is the listing `" 9.8\nB: 9.11\nC: 9.8\n\nA: 9"` (see §2).

**Other minimal sufficient sets:**
- L6 right / 143k: 24 sets, smallest `[0,11],[1,11],[2,9],[4,10],[6,10],[7,9]`; necessary none.
- L6 left: 27 sets; necessary none.
- 120k: 33 sets; necessary none.

## 5. Dynamics
- **`phenomenon_present`** (main pair F3 → F2) recomputes as recorded: True at 80k, 100k, 120k and 143k; False at 1k–64k. Gates pass at every checkpoint, there are no errors, and n_effective = 14 everywhere.
  - The 120k recipient label rests on the ambiguous listing `" 9.11\n\nA: 9.8\n\nA: 9.11"`.
  - At 80k and 100k, the true-head all-heads patch is not correct, so the repair fails there too (descriptive only).
- **Old protocol against `TD_REF` (`td_fixed`): 9/11 agree.** Both misses are the checkpoints the claim rests on:
  - 120k: now (0,1,0) vs reference (0,0,0). The even output is "9.8 is the largest number of people in the world".
  - 143k: now (0,0,0) vs reference (0,1,0). The even output is "…answer is 9.11", identical to the main-run `orig all_even`.
  - So September's "even repairs only at 143k" does not reproduce even under the old protocol. This check is reported, not gated.
- **`matches_main_run` at 143k:** I compared all 4,095 outputs against `pythia160m/results_pre_wo__prefill_right.json` myself and found 0 differences (True, as recorded). The screen outputs also match (`screen_matches_main_run` True), and the gate outputs are identical. The `true_heads` rows equal the corresponding subset outputs at both checkpoints.
- **Late-training verdict** (recomputed = recorded, no INBOX caveat): **"untestable: all-heads patch does not repair at step143000; head rules not interpreted"**. Under strict scoring, 120k would also become "no clear pair". That does not change the verdict.

## 6. What each pre-registered rule concludes, and what is interpretable
- **Pythia-160M:**
  - Layer-6 repair fails under true-head, prompt-only patching, in both alignments.
  - The parity rules are not interpreted (verdict 1).
  - Fallback L0 gives "parity not supported" (odd-leaning), a secondary result.
  - Late-training claim: untestable.
  - **But the Pythia reproduction gate failed, so per the pre-registration none of this is interpreted until an INBOX decision.** It is ambiguous whether that hold covers the dynamics run, which uses the same model but whose reproduction is "reported, not gated"; I would treat the dynamics as held too.
  - What can be stated without interpretation: the outputs exist, all gates pass, the run is internally consistent (143k equals main exactly), and nothing in the run supports the draft's parity or late-emergence claims.
- **Gemma-2B (base):** **No phenomenon**: no format answers correctly, so no corrected cell was run. "Its head-level claims have nothing to repair." The reproduction gate passed, so this is interpretable: the draft's "parity-independent" result came from F1 → F2 with a donor that gives no answer, under bugs A and B.
- **Gemma-2B-IT:** the mechanical verdict is "Layer-8 repair fails; not parity-independent". Correctly read, there is **no phenomenon** (F5 mislabel). There is no reproduction gate, so this is interpretable. I recommend reporting it as no format-dependent bug, with the mislabel stated, rather than as a failed repair.

## 7. Anything odd
- In Gemma-IT, patching from the correct donor occasionally induces a genuine "9.11 is greater" (2 subsets per alignment, plus layer 3 of the sweep).
- The answer-line rule does not cut model-invented `A:`/`B:` turns (only `Q:`). That drives the ambiguous labels at L0, 100k and 120k, and the fallback-layer trigger.
- The Pythia gate double-counts `all_even`/`sweep_L6`, so the "10 references" are 9 distinct configs. It does not matter for the result.
- `run.json` for the main Pythia run has no `verdict_input` field, which is fine. The experiment doc still says "Results: (none yet)" and Status SUBMITTED.
