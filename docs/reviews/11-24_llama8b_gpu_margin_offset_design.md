Reviewer: fresh subagent (Claude Code)
Agent-ID: a38e87a4f441f3a93
Verdict: PASS WITH MINOR FIXES
Transcript-SHA256: 8313fa4a77350a28ad4697cb437af54f3ba8dc06848b26a6f8cca93014779aeb

## Brief given

You are a fresh, skeptical design reviewer for a pre-registered interpretability experiment (repo `~/github/MATS`). Do NOT run GPU compute, do NOT edit any repo files, do NOT run git commit/push. Read only; you may run `python3` locally for quick checks on saved JSON results (plain `python3`, no torch available locally).

Read:
- Pre-registration: `docs/experiments/11-24_llama8b_gpu_margin_offset.md` (R21)
- Scripts: `9_8_research/reviewer_experiments/exp_R21_gpu_margin/run_experiment.py`, `analyze.py`, `run.sbatch`
- What it builds on: R13's script `9_8_research/reviewer_experiments/exp_R13_true_head_patching/run_experiment.py` (imported unchanged; Patcher.generate, build_configs, is_correct) and R13's saved results in `exp_R13_true_head_patching/results/r13_24014365` (L40S) and `r13_24084222` (A100); R16's experiment file `docs/experiments/11-22_llama8b_position0_gpu_gap.md` and its script `exp_R16_position0/run_experiment.py`. The journal is `docs/research_journal.md`, decisions in `docs/decisions.md`.
- A CPU smoke on a random-weight 12-layer, 32-head tiny Llama ran both GPU roles with `--limit 20 --smoke-tdec 2`: mechanics ran, final-layer lens matched the margin to ~1e-7, copy gates failed by design (CPU vs GPU references).

Question: is this design sound enough to run, and will its decision rules answer "is the one-directional L40S→A100 class gap in R13 explained by a shared offset in the '8' vs '11' logit margin at the decision step"?

Apply this checklist:
1. Fact-check every number in the experiment file (D counts 40/60, 103 total changes, R16's 119/120/83/86, sha256s, the "texts split right after ' 9.'" claim) against the repo's files. Recompute from the R13 JSONs where possible.
2. Recompute the arithmetic (compute estimate, thresholds).
4. Completeness: is every term (decision step, margin, M, D_pred, p_pos, offset) operationally defined, and does the code implement exactly the written definition?
5. Circularity: does any decision rule reuse what it is supposed to test?
Plus: is the matching control adequate? Look for code bugs: the generate wrapper (does adding return_dict_in_generate / output_logits / output_hidden_states risk changing numerics or R13's decode of `out[0][...]`?), the t_dec search, logit indexing (logits[t] predicts gen[t]?), hidden_states indexing per step and the final-norm convention, the smoke flag leaking into real runs, analyze.py's D/M/D_pred logic and edge cases (c<0, zeros, configs without a decision step), runtime/memory risk of output_hidden_states over 20 steps on an 8B fp16 model on L40S (48GB) within 2 h.

Return a verdict on the first line: PASS, PASS WITH MINOR FIXES, or FIXES NEEDED. Then numbered findings, each with severity (blocking / should-fix / minor), file:line, and a concrete fix. Be concise and concrete; no praise.

## Report

**PASS WITH MINOR FIXES**

Nothing blocks the run. The mechanics are correct, and the numbers I could check against the repo's files match. The fixes below mostly concern whether the labels can say anything beyond what R13 already shows, and they all live in `analyze.py` or the experiment file, so they can land before the run without touching the sbatch hash.

**Fact-check (recomputed from the R13 JSONs)**
- D counts: `post_wo/prefill` D = 40 and `post_wo/all_steps` D = 60, all going False→True (L40S bug → A100 correct). The total across R13's six cells is 103 (40 + 60 + 1 + 2 + 0 + 0), all in that direction.
- The " 9." split claim holds for all 100 D configs: the common prefix is exactly `' 9.'`, then `'8 is'` (A100) against `'11 i'` (L40S).
- Stronger result: every one of the 1,307 texts in both cells on both GPUs starts with " 9.", and the class equals "next token is 8" in all of them. So M should be all 1,307 and D ⊂ M. The other text differences outside D (9 in all_steps, 4 in prefill) split much later and do not change the class.
- R13's sha256 is `1ce5b0cc…348d` ✓. R16's 119/120/83/86 ✓ (these are R8-subset counts in the prefill cell, R16 doc lines 160–166).
- Per-cell time: R13 took 585–648 s per cell (A100 / L40S). Two cells plus model load comes to about 25 min per GPU, so the estimate and the 2 h walltime are fine.

**Mechanics checked and OK**
- `logits[t]` predicts `gen[t]`. `hidden_states[t_dec][:, -1]` is the forward pass that produced `logits[t_dec]`.
- In transformers 5.x (I read `utils/output_capturing.py` in a local 5.12.1 install, not the cluster's 5.15.1), the 33 entries are: embeddings, the outputs of layers 0–30, then `last_hidden_state`, which is already normed. The norm on `l < 32` matches that.
- The capture hooks are inert forward hooks on the decoder layers. They do not touch R13's `self_attn` hook or its `prefill_only` call counter.
- The wrapper returns `.sequences`, so R13's `out[0][...]` decode is unchanged. The copy gate would catch any drift anyway.
- `--smoke-tdec` cannot reach real runs (it needs `--limit`, which the sbatch never passes).
- GPU memory for the extra outputs is about 20 MB per call (20×128k fp32 logits plus 33×20×4096 fp16 hidden states). No memory or time risk on the L40S.

**Findings**

1. **should-fix**, `analyze.py:60-61`, doc line 77: the H0 density check uses the window ±|c|. Under H0, c ≈ 0, so both windows are empty exactly when H0 is true; the check cannot detect what it is for. Fix: use a window that does not depend on c, e.g. ±median|Δ| (or the IQR of Δ). Report the counts of m_L40S in (−w, 0) and (0, w), plus the expected number of reverse flips, #{k : 0 < m_L < −Δ_k}, against the observed 0.

2. **should-fix**, `run_experiment.py:117-119`, `analyze.py:42-44`: m is the difference of two fp16-rounded logits (HF casts fp16 `outputs.logits` to fp32). So m and Δ sit on a grid of about 0.0078–0.0156. If many Δ are 0, then c can be 0, which makes D_pred empty: recall is 0 and the run is labelled rule 2 or rule 4 for a numerical reason. Fix: pre-register what happens when c == 0 or n_zero/M > 0.2. Also pre-register a secondary analysis on the near-continuous fp32 margin `lens[-1]`, and state which one rules on a disagreement.

3. **should-fix**, doc lines 81–83, no control in code: the doc says "no placebo possible", but there are two cheap controls.
   - (a) Within-GPU logit determinism: re-run the unpatched prompt plus about 50 configs a second time in the same job and require bit-identical m. Without this, Δ may include run-to-run noise. R13's determinism gate checked only texts, and only for the all-heads config.
   - (b) A permutation control for "shared offset": shuffle Δ across configs and predict flips as {m_L < 0 < m_L + Δ_π(k)}, over about 1,000 permutations. If a config-independent offset explains D, the shuffled predictions should recover about |D| one-way flips and about 0 reverse flips. This tests the actual claim, that Δ is independent of where a config sits relative to the boundary. A constant-c recall threshold does not test it.

4. **should-fix**, doc lines 67–74 and research question: R13's 40:0 and 60:0 one-way counts already all but rule out symmetric noise (89 L40S-correct configs, none flips back). So p_pos ≥ 0.9 is close to forecastable, and rule 3 is nearly unreachable. The real information is in recall/precision and in where the gap appears across layers. Also, rule 1 can fail even if H1 is true, whenever Δ_k has a spread comparable to c: D configs with Δ_k > c and m_L in (−Δ_k, −c) are missed. Fix: write the expected outcome into the doc. Say that rule 1 tests "a constant suffices", not "biased versus noise". Say that "explains" here means "is described by an offset", not a mechanism. Optionally report recall at c = Q1, median and Q3 of Δ.

5. **minor**, `analyze.py:33-35`:
   - p_pos < 0.3 is labelled "partial offset", which is wrong: that is a reverse offset, contradicting D's direction. Add a mirror rule for p_pos ≤ 0.1.
   - p_pos defaults to 0.0 when every Δ is 0 (`:43`), which also falls through to "partial offset". Label that case "no measurable difference".

6. **minor**, `analyze.py:44,54,64`: `st.median` and `st.quantiles` raise on an empty M, and `A[M[0]]` raises IndexError. Fine for the real data, where M = 1,307, but guard them. `D_sign_ok` uses strict inequalities. On an exact fp16 tie the lower id wins: assuming the ids are as I remember (not checked, no tokenizer here), "8" is about 23 and "11" about 806, so "8" wins. Count m == 0 separately.

7. **minor**, `run_experiment.py:175`: `class_matches_argmax` is only recorded. Given finding (fact-check bullet 3), make "argmax at t_dec ∈ {8, 11} and the class equals argmax == 8" a gate. Otherwise m does not mean "the decision".

8. **minor**, `analyze.py:77-87`: nothing rejects smoke or partial run directories.
   - Assert `"--limit" not in run["command"]` and `n_configs == 1307` per cell.
   - Assert both runs carry the same `r21_script_sha256` and `r21_git_sha`.
   - Record analyze.py's own sha256. The sbatch hashes only `run_experiment.py` (`run.sbatch:24`), so analyze.py is not frozen by anything except the commit.

9. **minor**, process: `exp_R21_gpu_margin/` and the experiment file are untracked, and there is no R21 entry in `docs/decisions.md` or `docs/research_journal.md` (the doc cites Gus's "check"). Commit them before taking `R21_GIT_SHA`. `run_experiment.py:72` writes R21's SHA into R13's provenance `git_sha`; note that in the doc.

10. **minor**, doc Limitations: the node is pinned but the card is not. R16 already landed on a different A100 card. Texts matched, but logit-level Δ could differ by card. Record the GPU UUID (already in provenance) and note it.

11. **minor**, lens (descriptive):
    - At t_dec the KV cache already carries the GPU differences from the prefill and the first decode steps. So "first layer where |median Δ| ≥ |c|/2" locates where the gap in the current step's residual first reaches half its final size. It does not locate where the gap was created. State this.
    - `LENS_TOL = 0.05` is right for logits below 32 (≤ 0.016 rounding error). It is tight if logits are 32 or more (≤ 0.031). Record the max |logit| so a gate failure can be diagnosed.
    - The CPU smoke ran in fp32, so the fp16 tolerance has not been tested yet.

I edited no files and ran no git or GPU commands. Everything above was checked by reading the files plus local `python3` over the R13 JSONs.

## Doer response

All eleven findings applied before any compute:
1. H0 density window is now w = median |Δ|; expected reverse flips under each config's own Δ are reported.
2. Primary margin m; the fp32 `lens_final` margin rules if c == 0 or more than 20% of Δ are 0. Both reported.
3. (a) Determinism gate: unpatched + first 50 prefill configs re-run in the same job, bit-identical m, lens and
   text. (b) Permutation control, 1,000 shuffles, seed 0, descriptive.
4. Expected outcome written into the experiment file; rule 1 described as "a constant suffices"; recall at
   Q1/median/Q3 reported.
5. Mirror rule "reverse offset" for p_pos ≤ 0.1; "no measurable difference" when all Δ are 0.
6. Guards for empty M; exact ties counted separately.
7. Class gate added.
8. analyze.py rejects smoke, unfinished and mismatched runs, asserts 1,307 configs, records its own sha256.
9. Files committed before taking R21_GIT_SHA; the git_sha note is in Limitations.
10–11. Card and lens-locality limitations stated; max |logit| recorded.
Synthetic tests: an offset dataset gives rule 1 in both cells, a noise dataset gives "no shared offset".
