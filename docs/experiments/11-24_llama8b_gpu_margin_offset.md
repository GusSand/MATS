# R21 — Is R13's one-way L40S vs A100 gap a shared logit-margin offset?

**Status:** PRE-REGISTERED 2026-11-24, before any compute (amended after design review rounds 1–2)
**Design review:** FIXED — fresh subagent, 2026-11-24 — docs/reviews/11-24_llama8b_gpu_margin_offset_design.md
**Design review:** FIXED (round 2) — fresh subagent, 2026-11-24 — docs/reviews/11-24_llama8b_gpu_margin_offset_design_v2.md
**Script:** `9_8_research/reviewer_experiments/exp_R21_gpu_margin/run_experiment.py` (measurement, one job per GPU)
and `analyze.py` (pre-registered rules, CPU). The run script imports R13's script by path; R13's sha256 must equal
`1ce5b0cc…`, the hash both R13 runs and R16 used. R13's hook and generation code are untouched.
**Jobs:** `run.sbatch`, submitted twice and independently: A100 (`a100_tandon`, pinned to `ga004`, R13's A100 node)
and L40S (`l40s_public`, pinned to `gl044`, R13's L40S node).
**Slurm jobs:** 24375606 (A100, ga004) and 24375607 (L40S, gl044), commit e89ad0a, R21 sha256 c76475e9…,
R13 sha256 1ce5b0cc…. Logs `/scratch/gs157/MATS/logs/<job>_r21_margin.log`; results
`/scratch/gs157/MATS/results/r21_<gpu>_<job>/`.
**Decision:** Gus, 2026-11-24 ("check", in reply to the proposal to find where the two GPUs' runs diverge and
why all differences go one way).

## Research question

R13's old-hook cells give different classes on the two GPUs, with identical software. Across all R13 cells there
are 103 class changes and every one goes bug (L40S) → correct (A100). R16 ruled out the layer-10 position-0 write
(keep0 leaves the A100 R8 prefill count at 119 vs 120; L40S is 83) and found that the gap arises in the receiving
run's own arithmetic (donorA100 on L40S: 86). Why is the gap one-directional?

Fact from the saved R13 texts (computed 2026-11-24 from `results_post_wo__{prefill,all_steps}.json` of
r13_24014365 and r13_24084222): in all 40 `post_wo/prefill` and all 60 `post_wo/all_steps` D configs, the two
GPUs' texts agree up to " 9." and split at the next token ("8" on A100, "11" on L40S). The gap is therefore one
logit comparison at one decoding step.

Design review fact-check: every one of the 1,307 texts in both cells on both GPUs starts with " 9.", and the
class equals "next token is 8" in all of them. So M should be all 1,307 and the margin is the decision itself.

## Expected outcome, stated before the run (review finding 4)

R13's own counts already make two-sided noise unlikely: 40:0 and 60:0 flips, while 89 L40S-correct configs
never flip back. So p_pos ≥ 0.9 is the expected result and is not the informative part. The informative parts:
- **Rule 1 tests "a constant offset suffices"**, not "biased vs noise". It can fail even under H1 when Δ_k
  spreads as widely as c. "Explains" means "is described by an offset", not a mechanism.
- **Where the gap appears** in the logit-lens profile.
- **The permutation control**, which tests whether Δ is independent of where a config sits near the boundary.

## Hypotheses

- **H1, shared offset (proposed).** Every config runs the same buggy prompt with the same clean donor; only the set
  of patched blocks differs. Much of the arithmetic is shared across configs, so the GPU's rounding may act as a
  near-constant shift of the "8" vs "11" margin, the same sign for nearly every config. Configs whose L40S margin
  lies just below zero then cross on A100, all in one direction.
- **H0, boring (written first).** Per-config differences are two-sided noise; the one-way flips come from an
  asymmetric density of margins near zero (many configs just below, few just above). Detected by p_pos near 0.5
  together with the near-boundary counts below.

## Setup

- Model, prompts, hook (`post_wo`), alignment, the 1,307 configs, greedy decoding, 20 tokens, fp16: exactly R13's,
  via R13's own `Patcher.generate`, `build_configs`, `is_correct`, `is_bug`. Same snapshot (0e9e39f2…),
  torch 2.13.0, transformers 5.15.1 (the Torch env R13 and R16 used).
- The only change: `model.generate` is wrapped to also return raw logits and hidden states
  (`return_dict_in_generate`, `output_logits`, `output_hidden_states`) and hand R13 back `sequences`.
- Cells per GPU: the unpatched buggy prompt (1 run), `post_wo/prefill` and `post_wo/all_steps` (1,307 each).
- **Decision step** t_dec: the first generated position t whose generated prefix decodes to exactly " 9.".
  A config without one fails the class gate (below), so interpretation requires M = all 1,307 configs.
- **Margin** m = logit(id "8") − logit(id "11") at t_dec, raw logits cast to fp32. Both strings must be single
  Llama-3 tokens (asserted).
- **Logit lens** (descriptive): for each of the 33 hidden states at t_dec, last position, apply the final RMSNorm
  (layers 0–31; HF's last hidden state is already normed) and dot with the two lm_head rows in fp32.
- Also recorded: generated ids, argmax at t_dec, top-2 ids and gap at every step up to t_dec.

## Gates (job exits 2 on any failure)

- **Copy gate:** in each cell, all 1,307 texts equal R13's texts on the same GPU. This also shows the wrapper
  changes nothing.
- **Unpatched gate:** the unpatched text equals R13's baseline buggy text.
- **Lens gate:** for every config with a margin, |lens at the final hidden state − m| ≤ 0.05 (fp32 lens vs
  the model's fp16 lm_head; one fp16 step at |logit| ≈ 20 is 0.0156).
- **Class gate (review finding 7):** for every config, argmax at t_dec ∈ {"8", "11"} and R13's class `correct`
  equals (argmax == "8").
- **Determinism control (review finding 3a):** in the same job, the unpatched prompt and the first 50 prefill
  configs (prefill only; all_steps is not re-run) are run again; text, m and all 33 lens values must be
  bit-identical. Without this, Δ could include
  run-to-run noise.
- Interpretation requires all gates on both GPUs (`interpreted` in analyze.py). analyze.py refuses smoke runs
  (`--limit`), unfinished runs, cells without 1,307 configs, and runs with different R21 script or git SHAs, and
  records its own sha256.

## Pre-registered decision rules (analyze.py, per cell)

D = configs whose R13 class (`correct`) differs between the two GPUs, read from R13's own files (40 prefill,
60 all_steps). M = configs with a margin on both GPUs. Δ_k = m_A100(k) − m_L40S(k).
p_pos = (# Δ > 0) / (# Δ ≠ 0); zeros reported. c = median Δ over M.
Offset prediction: D_pred = {k ∈ M : m_L40S(k) < 0 < m_L40S(k) + c} (mirror image if c < 0).
recall = |D ∩ D_pred| / |D|, precision = |D ∩ D_pred| / |D_pred|. First matching rule:

0. no nonzero Δ → **"no measurable difference"**
1. p_pos ≥ 0.9 and recall ≥ 0.8 and precision ≥ 0.5 → **"shared offset explains the one-way flips"**
2. p_pos ≥ 0.9 → **"offset present, does not alone predict which configs flip"**
3. p_pos ≤ 0.1 → **"reverse offset: contradicts the direction of D"**
4. 0.3 ≤ p_pos ≤ 0.7 → **"no shared offset: differences are two-sided"**
5. otherwise → **"partial offset (p_pos = …)"**

**Which margin rules (review finding 2).** m is the difference of two fp16-rounded logits, so it sits on a grid
of about 0.008–0.016 and many Δ could be exactly 0. Primary: m. If c == 0 or more than 20% of Δ are exactly 0,
the rules are applied to the fp32 lens margin at the final hidden state instead (`lens_final`). Both are always
reported; `primary_margin` records which one ruled.

Why precision 0.5, not 0.8: a constant c over-predicts configs whose own Δ is below the median, so some D_pred
members legitimately do not flip. Not circular: D comes from R13's texts, c from the margins of all ~1,300
configs, of which D is under 5%.

**Also reported (descriptive, no label):** Δ quartiles and range; Δ for each D config; D_sign_ok (D configs with
m_A100 > 0 > m_L40S) and D configs with an exact tie; recall and precision at c = Q1, median and Q3 of Δ;
unpatched Δ vs c; the median-Δ logit-lens profile over the 33 hidden states and the first layer where its
magnitude reaches |c|/2.
- **H0 density check (review finding 1):** window w = median |Δ|, independent of c. Counts of m_L40S in (−w, 0)
  and (0, w). If more than half of Δ on m are exactly 0, w = 0 and the counts are empty; the lens_final block
  then carries the check.
- **Expected reverse flips under H0 (round-2 finding 1):** pool Δ and −Δ (symmetric noise); for each config with
  m_L40S > 0, add the share of pooled values below −m_L40S. Reported next to the observed reverse flips
  #{m_A100 < 0 < m_L40S} (0 by R13 and the gates). A large expectation against 0 observed is evidence against H0.
- Unpatched Δ is reported on both m and lens_final.

**Controls.**
- **Determinism** (gate above): run-to-run noise within one GPU must be exactly zero.
- **Permutation control (review finding 3b):** shuffle Δ across configs 1,000 times (seed 0) and predict flips as
  {m_L < 0 < m_L + Δ_π(k)}. Reported: mean number of D configs recovered (out of |D|) and mean number of
  predicted reverse flips. If a config-independent offset accounts for D, the shuffles recover about |D| and
  predict about 0 reverse flips. Descriptive, no label.

## Compute estimate

Per GPU: 2 cells × ~620 s (R16's per-cell time) plus logits and hidden-state copies, about 25–30 min. Two jobs,
≈ 1 GPU-h total; walltime 2 h each. Node pinning may lengthen the queue wait.

## Limitations known before the run

- One node per GPU type. R16's A100 job ran on a different card on ga004 than R13 and still reproduced R13.
- The lens is descriptive and fp32. At t_dec the KV cache already carries the GPU differences from the prefill and
  earlier decode steps, so "first layer where |median Δ| ≥ |c|/2" locates where the gap in the current step's
  residual first reaches half its final size, not where it was created (review finding 11). Max |logit| is
  recorded so a lens-gate failure can be diagnosed; the fp16 tolerance was not tested by the fp32 CPU smoke.
- The node is pinned, the card is not. GPU UUIDs are in the provenance (review finding 10).
- analyze.py is frozen by the commit only (the sbatch hashes run_experiment.py); its sha256 is recorded in its
  output and must match the committed file. Nothing checks R21_GIT_SHA against the running code beyond
  both runs carrying the same value and the script sha256 gate.
- `run_experiment.py` writes R21's git SHA into R13's provenance field `git_sha` (it sets `R13_GIT_SHA`).
- Only the two `post_wo` cells with D ≥ 40 are measured. The `pre_wo` cells (D = 2 and 1) are not.

## Results

None yet.
