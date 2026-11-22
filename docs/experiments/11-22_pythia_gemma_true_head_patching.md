# R14 — True per-head patching in Pythia-160M and Gemma-2B (9.8 vs 9.11)

**Status:** PLANNED (pre-registered 2026-11-22, before any compute; amended after design review rounds 1 and 2)
**Design review:** round 1 FIXES NEEDED — fresh subagent, 2026-11-22 — docs/reviews/11-22_pythia_gemma_true_head_patching_design.md
**Design review:** round 2 FIXES NEEDED — fresh subagent, 2026-11-22 — docs/reviews/11-22_pythia_gemma_true_head_patching_design_v2.md
**Script:** `9_8_research/reviewer_experiments/exp_R14_cross_model_true_heads/run_experiment.py`
**Job:** `run.sbatch` (same folder). One job, one GPU, all models sequentially.
**Decision:** Gus, 2026-11-22: "rerun" (INBOX 2, archived). Audit behind it:
`docs/reviews/11-22_cross_model_head_scripts_audit.md`.

## Research question

Do the v2 draft's non-Llama head claims hold under true per-head, prompt-only patching?
- Pythia-160M: even-head subsets repair, all-odd fail (`02_study.tex:289-293`); parity appears late in
  training (`02_study.tex:295-299`).
- Gemma-2B: even-only and odd-only subsets both repair ("parity-independent", `02_study.tex:310-314`).

## Why

The original scripts (`cross_model_validation/test_pythia_160m.py`, `test_gemma_2b.py`,
`training_dynamics_analysis/.../test_pythia_training_dynamics.py`) have R13's bug A (post-W_O
forward hook on `attention`/`self_attn`) and bug B (hook left on through `generate`). Two further
problems found while designing this:
- **The donor was wrong too.** In both original result files the donor prompt
  `"Which is bigger: 9.8 or 9.11?"` has `bug_rate: 1.0`. The old "repairs" copied activations from a
  prompt that also gave the wrong answer.
- **Lenient scoring.** The original classifier counts any output containing `"9.8"` as correct unless
  a bug phrase appears (`test_pythia_160m.py:145-168`).

## Simplest (boring) hypotheses, written first

1. Neither model has a format-dependent bug on these prompts: every format is wrong (or every format is
   right), so there is nothing for a donor to repair.
2. The old Pythia/Gemma repairs come from writing donor position-0 activations into the
   decode steps (bug B) or residual blocks (bug A), not from heads.

## Setup

- Models (CUDA, greedy, 20 new tokens, one run per configuration; fp16 for the two models with an original
  run, as in September; bf16 for Gemma-IT, its native dtype, since it has nothing to reproduce):
  - `EleutherAI/pythia-160m` (revision `main`; 12 layers, 12 heads, head_dim 64; `attention.dense`
    has a bias, which pre-dense patching leaves untouched). Original layer 6.
  - `google/gemma-2b` (base, the model the original script used; 18 layers, 8 query heads, 1 KV head,
    head_dim 256, `o_proj` without bias). Original layer 8.
  - `google/gemma-2b-it` (the model behind the draft's "~5% bug rate"). No original run; layer 8.
- Pythia training checkpoints: step1000, 2000, 4000, 8000, 16000, 32000, 64000, 80000, 100000,
  120000, 143000 (the original list), layer 6.
- **Classifiers.**
  - Text preparation: strip Markdown `*`, `` ` ``, `_`; cut the text at the first line starting with `Q:`
    (a turn the model invents). Numbers have boundaries: `9.8`, `9.80`, `9.800` count as 9.8; `19.8`,
    `29.11` count as nothing.
  - *Answer line* = the first non-empty line naming exactly one of 9.8 / 9.11; a line naming both ends
    the scan with no answer line.
  - *correct* = a correct phrase (`9.8 is [the] bigger|larger|greater`) and no bug phrase; with no
    phrase, the answer line is 9.8.
  - *bug* = not correct and a bug phrase; with no phrase, the answer line is 9.11. Else *neither*.
  - Changed from R13 because Pythia answers as `"I think the answer is:\n\n9.11"`, which R13's
    first-line rule calls *neither* (review round 1, fix 2). `--selftest` asserts 18 cases.
  - *orig_fixed* = the original `check_bug_fixed` verbatim, used **only** for the reproduction gate.
  - *td_fixed* = the training-dynamics script's classifier verbatim, used only to reproduce that run.
- **Phenomenon screen (stage 1).** Each model generates on six fixed formats, in this order:
  F1 `Which is bigger: 9.8 or 9.11?`; F2 `Q: Which is bigger: 9.8 or 9.11?\nA:`;
  F3 `Which is bigger: 9.8 or 9.11?\nAnswer:`; F4 `Compare 9.8 and 9.11. Which is larger?`;
  F5 `9.8 vs 9.11 - which is greater?`; F6 `Between 9.8 and 9.11, which is greater?`;
  Gemma-IT adds F7 = F1 in its chat template. **Pair rule (mechanical):** recipient = F2 if *bug*,
  else the first *bug* format in list order; donor = the first *correct* format in list order.
  No donor or no recipient → the model has no format-dependent bug on these formats; no corrected
  cell is run for it.
- **Cells per model (stage 2):**

| Cell | Hook | Scope | Pair | Configs | Purpose |
|---|---|---|---|---|---|
| `orig/post_wo/all_steps` | post-W_O blocks | prompt + decode | F1 → F2 (original) | original configs + all/even/odd/singles | reproduction gate |
| `pre_wo/prefill_right` | true heads | prompt only, right-aligned | screen pair | **every non-empty subset** (Pythia 4,095; Gemma 255) | **primary** |
| `pre_wo/prefill` | true heads | prompt only, left-aligned | screen pair | every non-empty subset | secondary |
| `layer_sweep` | true heads | prompt only, right-aligned | screen pair | all heads, every layer | descriptive |
| `pre_wo/prefill_right__L<f>` | true heads | prompt only, right-aligned | screen pair | every non-empty subset | fallback, secondary |

  Gemma-IT has no `orig` cell. Right alignment pairs the last `min(T_donor, T_recipient) − 1` positions
  and never patches position 0 (BOS for Gemma; the first token, the attention sink, for Pythia).
  Left alignment patches position 0 too (for Pythia, the first token). Aligned token pairs and
  *n_effective* (patched pairs whose activations can differ; a pair at the same absolute position with
  identical token prefixes is a causal no-op) are recorded per run for each alignment, and per checkpoint
  for right alignment. **A left-aligned cell with n_effective = 0 is vacuous: not run, not reported as a
  replication.** Right alignment cannot reach 0 unless the prompts are identical or one token long.
  **Fallback layer:** if the all-heads patch is not correct at the original layer but the layer sweep finds
  layers where it is, the primary cell is repeated at the first such layer, reported as secondary.
- **Pythia training dynamics:** per checkpoint (revision `stepN`; the original used `main` for 143000,
  both commits recorded): the six-format screen; old protocol F1 → F2 with even/odd at layer 6 scored by
  *td_fixed* (reproduces the original JSON, reported); identity/determinism/shape gates; then, on the
  `main` screen pair, true-head `pre_wo/prefill_right` with all/even/odd/singles, **run only if, at that
  checkpoint, the donor is *correct* and the recipient *bug* (`phenomenon_present`) and the gates pass.**
  Otherwise the checkpoint is "no phenomenon" or "gate failed: not interpreted" (printed); a gate failure
  does not stop the other checkpoints. At step120000 and step143000 (the two the late-training claim rests
  on) all 4,095 subsets run under the same condition, so the full parity rule applies there. These two run
  first; each checkpoint is wrapped so an error is recorded and the next one runs.
  **Consistency check:** step143000's outputs are compared with the `main` run's (all 4,095 when both are
  complete). A mismatch means weights or numerics differ: the 143k verdict is reported but not interpreted
  until an INBOX decision. A skipped check is recorded as null with its reason.

## Gates

- **Head isolation** per model: patching head 1 into the dense/o_proj input changes only head 1's slice.
  Fail → that model stops.
- **Identity / determinism / hook shapes** per prompt-only cell, as in R13. Fail → that model stops.
- **All-heads equivalence:** all-heads `pre_wo/prefill_right` equals the all-blocks `post_wo` patch with
  the same scope and alignment (same text). Holds even with Pythia's dense bias, which both paths keep.
  Fail → escalate before interpreting.
- **Reproduction gate (`orig` cells, original models only):** every original configuration at the
  original layer, scored by *orig_fixed*, gives the original outcome (rate ≥ 0.5 ↔ fixed). References:
  Pythia L6 all_even 1, all_odd 0, first_half_even 0, second_half_even 1; layer sweep L6 1, L7–11 0.
  Gemma L8 even4 1, odd4 1, even2 0, even3 0, mixed4 1; layer sweep L8 1, L9 0, L10 1, L11–17 0.
  Gate: ≥ 90% of these agree, and all_even/all_odd (Pythia) and even4/odd4 (Gemma) must agree.
  Below → numerics differ from September 2025; the job finishes but that model is not interpreted
  until an INBOX decision. The training-dynamics reproduction is reported, not gated.

## Pre-registered decision rules (per model, on `pre_wo/prefill_right`; `pre_wo/prefill` reported as replication)

- **No phenomenon:** the screen finds no (correct, bug) pair → "<model> shows no format-dependent bug on
  these six formats; its head-level claims have nothing to repair."
- **Layer-L repair:** all-heads patch is *correct*. Fails → "Layer-L repair fails under true-head
  prompt-only patching in <model>"; the head rules are reported, not interpreted.
- **Pythia parity holds** only if all four hold: (a) all-even correct; (b) all-odd not correct;
  (c) more even-only subsets correct than odd-only subsets (63 each);
  (d) **split specificity:** for each of the 462 balanced bipartitions (A, B) of the 12 heads,
  D = #correct subsets inside A − #correct inside B. The even/odd split must have D > 0 (even side),
  |D| strictly greater than the contiguous split {0–5}|{6–11}, and rank in the top 5% of |D| with ties
  counted against it. (a)–(c) alone pass when one even head carries the repair (review round 1, fix 1).
  Verdict wording (`parity_verdict`), given repair:
  - all four hold → "parity holds", qualified "with heads X necessary" when the necessary heads (the
    intersection of all minimal sufficient sets) are non-empty and are not the full even set;
  - (a)–(c) fail → "parity not supported";
  - (a)–(c) hold, (d) fails, and |D(even/odd)| ≤ 3 → "inconclusive: parity evidence rests on 3 or fewer
    subsets" (a true "all six even needed" effect has D = 1, so a few stray outputs sink (d));
  - (a)–(c) hold, (d) fails, and some split strictly beats even/odd, or a minimal set contains an odd head,
    or the necessary heads are a non-empty proper part of the even set → "even-only subsets repair, but the
    effect is carried by specific heads, not by parity";
  - otherwise → "inconclusive: the even/odd split ties with other splits".
  Minimal sufficient sets, necessary heads and the top 5 splits are reported.
  Pre-compute history: a ≥ 0.20 even-majority vs odd-majority gap was dropped (a pure "all six even heads"
  effect scores 0.04 on it); `--selftest` asserts the rule on seven synthetic truths, including head 8
  alone (fails), {6, 8, 10} needed (fails), all six even needed (holds), any two even (holds).
- **Gemma parity-independence holds:** even-4 and odd-4 both correct. The split rank (d) over Gemma's 35
  bipartitions is reported as descriptive (top 5% of 35 means a unique maximum).
- **Pythia late-training claim** ("parity appears only in the final 15%"), computed by
  `late_training_verdict`:
  - step143000 has no phenomenon / gate failure / error → "untestable".
  - parity fails at step143000 → "not supported".
  - Given parity at step143000, step120000: (i) no correct/bug pair → "untestable"; (ii) pair present,
    all-heads patch does not repair → "repair itself appears late; parity emergence not separable";
    (iii) repair present, parity fails → "supported"; (iv) parity holds → "refuted".
  Earlier checkpoints are descriptive.
- **Descriptive only:** three-way class counts by size, the even-majority vs odd-majority rate gap, single heads, count of correct subsets by size, layer sweep, per-checkpoint counts.

## Compute estimate

Pythia: up to 3 × 4,095 generations (right, left, fallback) at ~0.1 s, plus checkpoints (11 × ~40, and
2 × 4,095 at step120000/143000). Gemma: 2 models × (3 × 255 + ~40 + 18) at ~0.3 s.
Total ≈ 0.7 GPU-h; walltime 3 h on `l40s_public`.

## Limitations known before the run

- Rules (a) and (b) each rest on one output (all-even, all-odd). In simulation a noisy parity effect passes
  (d) far more often than (a). Size-wise class counts are reported next to them.
- If the screen pair is F3 → F1 (Pythia's fp32 CPU smoke gave this), the result concerns F1, not the
  `Q:/A:` phrasing the paper names, and the left cell is vacuous. Said in the write-up if it happens.

- One prompt pair per model, chosen by the mechanical rule above; positive results are in-sample.
- Different prompt formats have different token counts, so alignment pairs non-corresponding tokens
  somewhere in the prompt; the two alignments bracket this.
- Mistral head claims are not tested: no code or results exist (decision 2026-11-22).
- The Pythia "0% on 50 decimal pairs" claim is not rerun; its scoring made 0% certain (audit, point C).

## Results

(none yet)
