# R15 — Testing the Mistral-7B head claims with true per-head, prompt-only patching

**Status:** DONE (pre-registered 2026-11-22, before any compute; amended after design review rounds 1–3)
**Design review:** round 1 FIXES NEEDED — fresh subagent, 2026-11-22 — docs/reviews/11-22_mistral7b_true_head_patching_design.md
**Design review:** round 2 FIXES NEEDED — fresh subagent, 2026-11-22 — docs/reviews/11-22_mistral7b_true_head_patching_design_v2.md
**Design review:** round 3 FIXES NEEDED (1 small) — fresh subagent, 2026-11-22 — docs/reviews/11-22_mistral7b_true_head_patching_design_v3.md
**Design review:** PASS (round 4) — fresh subagent, 2026-11-22 — docs/reviews/11-22_mistral7b_true_head_patching_design_v4.md
**Script:** `9_8_research/reviewer_experiments/exp_R15_mistral_true_heads/run_experiment.py` (imports R14's
patching code; the job refuses to start unless R14's and R15's sha256 equal `R14_EXPECTED_SHA256` and
`R15_EXPECTED_SHA256`, computed at submit time from the committed files). `decide()` has its own synthetic
selftest, run at every start. All decision rules are computed in the script (`decide`) into `run.json`.
**Job:** `run.sbatch` (same folder), `a100_tandon`.
**Slurm job:** 24092912 (commit e1c139e, `a100_tandon`, R15 sha256 1fba89a4…, R14 sha256 20819e06…). Log `/scratch/gs157/MATS/logs/24092912_r15_mistral.log`; results `/scratch/gs157/MATS/results/r15_24092912/`.
**Decision:** Gus, 2026-11-22: "the nam result can we run that ourselves after downloading the model?"

## Research question

Do the v2 draft's Mistral-7B-Instruct-v0.2 claims (`02_study.tex:224-280`) hold when we measure them
ourselves? They came from a collaborator by email; no code, prompts or results exist anywhere
(audit: `docs/reviews/11-22_cross_model_head_scripts_audit.md`, claims 6–7). So this run **tests the
claims; it cannot reproduce the numbers.** The draft's Δ metric and prompt strings are unknown.

Claims under test (draft wording, abridged):
- **S** Format profile: chat and comparative formats err, Q&A and simple do not.
- **M1** No single layer's attention transplant fully repairs; five layers fix it individually
  (ℓ = 2, 17, 13, 10, 11, in decreasing effect).
- **M2** Even-only subsets fix at ℓ = 2 and 17; odd-only subsets fail at k ∈ {4, 8, 12, 16}; about 10 even
  heads needed for > 50% success.
- **M3** H22 is the only single head that fixes the bug (at ℓ = 2 and 17).
- **M4** H22 alone across 7 layers gives a full fix.
- **M5** Clean → buggy five-layer transplant restores correct generation.
- **M6** Asymmetry: buggy → clean transplant does not induce the bug.
- **M7** On 100 random X.D vs X.DD pairs (shorter larger), errors are rare (chat 2/100, comparative 19/100,
  Q&A 0/100, simple 0/100), and the five-layer transplant repairs the chat-buggy pairs.

## Simplest (boring) hypotheses, written first

1. Mistral has no format-dependent bug on our versions of the formats (the draft's strings are unknown).
2. Repairs, where they occur, depend on how many heads/layers are written, not on head identity.

## Setup

- `mistralai/Mistral-7B-Instruct-v0.2`, loaded **by snapshot 63a8b081…** (the job refuses another), bf16
  (native), A100 (`a100_tandon`), greedy, **64** new tokens (the instruct model often reasons before
  answering); every classified output records `hit_max_new` and `says_less` ("is less/smaller/lower", which the
  classifier counts as *neither*). 32 layers, 32 query heads,
  8 KV heads (GQA groups 4g..4g+3), head_dim 128, `o_proj` without bias.
- Patching: R14's `Patcher` (true heads = `o_proj` input via forward-pre-hook; prompt pass only; right
  alignment, position 0 never patched). Multi-layer configs register one pre-hook per layer, each with its
  own donor activations; all hooks act on the prompt pass only.
- **Classifier:** R14's (`classify`, with its 18-case selftest).
- **Formats (fixed):** *chat* = the tokenizer's chat template around `Which is bigger: 9.8 or 9.11?`;
  *comparative* = `Compare 9.8 and 9.11. Which is larger?`; *qa* = `Q: Which is bigger: 9.8 or 9.11?\nA:`;
  *simple* = `Which is bigger: 9.8 or 9.11?\nAnswer:`.
- **Pair rule:** recipient = chat if *bug* (the draft's target format); donor = simple if *correct*, else qa
  if *correct*. Aligned token pairs are recorded. simple (20 tokens) → chat (23) right-aligned is exactly
  token-aligned on the question; a qa donor is one position off throughout and is reported as
  "off-by-one aligned".
- **Descriptive screen:** comparative, qa and simple wordings inside the chat template (separates template
  from wording). The chat-template token ids from the string path and from `tokenize=True` are recorded. No pair → stop: "no format-dependent bug in chat on our formats; M1–M6 have nothing to
  repair". M7's screen still runs.
- Every classified output, including M7's, records `hit_max_new` and `says_less`. `decide()` runs before M7 too
  (`decisions_before_M7`), so a crash in M7 leaves M1–M6 decided.
- **Δ** = teacher-forced log P(` 9.8`) − log P(` 9.11`) as the reply's first words, with the same prompt-only
  patch (continuation positions unpatched). Mistral tokenizes ` 9.8` as `▁ 9 . 8`, so Δ compares `8`
  against `1 1` after `▁9.`. **Δ readings are testable only if chat's unpatched Δ ≤ 0** (the draft has
  −1.19); otherwise every Δ "fix" is vacuous and those readings are "untestable". Δ is None when the
  continuation does not extend the prompt's tokens (all-or-none per prompt, so visible in the screen's Δ).

| Cell | Configs | Tests |
|---|---|---|
| screen | 4 formats, unpatched | S |
| layer sweep | all 32 heads at each layer (32) | M1 |
| single heads | each head at each layer (1,024) | M3 |
| parity | layers 2 and 17 × {even-only, odd-only, mixed (both parities present)} × k ∈ {4, 8, 12, 16} × up to 20 distinct seeded subsets (one even-only and one odd-only set at k = 16), plus even-only subsets **without H22** at k = 4, 8, 12 (20 each): 262 per layer | M2 |
| multi-layer | all heads at {2, 10, 11, 13, 17}; 10 seeded random 5-layer sets (all heads); **each head h** at {2, 10, 11, 13, 17} and at all 32 layers (64) | M4, M5 |
| reverse | chat activations into the simple prompt (all heads): each of the five layers, and all five together | M6 |
| generalization | 100 seeded X.D vs X.DD pairs (shorter larger, X ∈ 1..9) × 4 formats; on each pair with chat *bug* and simple *correct*, the five-layer transplant | M7 |

## Gates

- `selftest` (R14) passes; head isolation (head 1 at layer 2); identity (own activations at all heads,
  layers {2, 10, 11, 13, 17}, equals unpatched), determinism (×3) and hook shapes (prompt pass only) for the
  five-layer config. Any failure → stop.
- Sanity: the all-heads, all-32-layers transplant is expected to be *correct* (it replaces every attention
  output on the overlapping positions). Not a stop; it decides whether M1a's null is interpretable (see M1a).

## Pre-registered decision rules (computed by `decide()`)

Generation reading: success = *correct*. Δ reading (only when testable): success = Δ > 0.
- **S holds:** chat *bug*, comparative *bug*, qa and simple not *bug*, and at least one of them *correct*.
- **M1a** ("no single layer fully repairs", generation): no layer's all-heads patch is *correct*.
  A *correct* layer refutes M1a regardless. Only the null needs the all-layers sanity transplant to be
  *correct*; otherwise the null is "uninterpretable" (a null from an ineffective patch says nothing).
  **M1b** ("five layers fix individually", Δ): at least 4 of {2, 10, 11, 13, 17} have Δ > 0 **and** at least
  4 of them are among the 8 layers with the largest Δ. Δ > 0 at the five without that → "non-specific".
- **M2 at ℓ ∈ {2, 17}**, generation and Δ readings: (a) the even-only k = 16 set succeeds; (b) no odd-only
  subset succeeds; (c) **even-only subsets without H22 succeed more often than odd-only subsets**, one-sided
  Fisher exact p < 0.05 (60 vs 60, k = 4, 8, 12; about 5/60 vs 0/60 is needed). H22's role is shown if even
  subsets containing H22 (k ≤ 12) beat the without-H22 subsets at Fisher p < 0.05, or H22 alone succeeds at ℓ.
  Verdict, first match: (a)(b)(c) → "parity holds"; (a)(b) and H22's role shown → "even-only repair carried
  by H22"; (a)(b) → "even-only k = 16 repairs; H22's role not shown"; else "not supported". Every rate is
  reported; even-only rates are also split by H22 ∈ S (pooled over k; per-k splits are computable from the saved heads). "≈10 even heads for > 50%" is read from the **without-H22** rates (k = 4, 8, 12;
  k = 16 cannot be tested without H22).
- **M3** (generation): at both ℓ = 2 and 17, H22 is *correct* and no other single head is.
  **M3b** (Δ): H22 has the largest Σ_ℓ (Δ_patched − Δ_unpatched) over the 32 × 32 single-head sweep; the
  draft's top five (H22, H29, H31, H10, H21) are ranked too.
- **M4** (generation): H22 at {2, 10, 11, 13, 17} or at all 32 layers is *correct* (the draft's 7 layers are
  not named). **"H22 special"**: M4 holds and, in each config where H22 is *correct*, at most 4 of the 32
  heads are. H22's result is reported per config.
- **M5** (generation): the five-layer all-heads transplant is *correct*. **"M5 specific"**: M5 holds and at
  most 2 of the 10 random 5-layer sets are *correct* (boring hypothesis 2).
- **M6**: M5 holds and the five-layer reverse transplant leaves simple *correct* (not merely "not bug";
  *neither* is reported separately), with its Δ.
- **M7:** error counts per format reported against the draft's (2, 19, 0, 0)/100; "repair transfers" holds
  if the five-layer transplant is *correct* on every eligible pair (and there is at least one), always
  reported as "k of n eligible pairs". M7 is decided on the no-pair path too.
- **Descriptive only:** Δ for every config; per-head counts across layers; GQA-group breakdown of correct
  single heads; mixed-subset rates.

## Compute estimate

≈ 7 screen + 5 gates + 1 sanity + 32 + 1,024 + 524 + 1 + 10 + 64 + 6 + 400 + ≤100 ≈ 2,180 generations ×
~2 s (64 tokens, 7B, A100) ≈ 1.2 h, plus 2 Δ forward passes per patched config (cheap). Walltime 5 h.

## Limitations known before the run

- Prompts, Δ definition and the "7 layers" are not the draft's; a mismatch with the draft can come from
  these alone. The write-up says "tested", not "reproduced".
- The draft describes an "attention pattern" / "z-pattern" transplant. This run patches z (the `o_proj`
  input). A softmax-pattern transplant across prompts of different lengths is a different intervention.
- GQA groups 4g..4g+3 each hold two even and two odd heads, so parity cannot be a disguised KV-group effect;
  it is the within-group slot {0, 2} vs {1, 3}.
- One canonical pair for M1–M6.
- The draft's "n = 20 per condition" and "n = 1000 per format" were repeats of greedy runs or unknown
  sampling; here every configuration runs once and n counts distinct subsets or pairs.

## Results

Job 24092912 (A100 ga006, commit e1c139e; R15 sha256 1fba89a4…, R14 sha256 20819e06…), raw outputs in
`results/r15_24092912/`. Verified from raw text by a fresh subagent: VERIFIED-WITH-FLAGS. All 2,073 labels
and every decision recompute exactly (docs/reviews/11-22_mistral7b_true_head_patching_verify.md). Gates and
the sanity transplant pass.

| Claim | Verdict | Numbers |
|---|---|---|
| S format profile | **not supported** | chat bug, qa and simple correct, comparative gives no answer within 64 tokens (unresolved, not clean). Inside the chat template, 3 of 4 wordings err in substance. |
| M1a no single layer repairs | **refuted** | only layer 7 repairs. The draft's layers 2, 17, 13, 10 and 11 each still give the bug. |
| M1b / M2-Δ / M3b (Δ readings) | **untestable** | chat baseline Δ = +3.75. Δ is not meaningful for chat: its reply names 9.8 first whether it is right or wrong. |
| M2 even-only at ℓ = 2, 17 | **not supported** | 0 correct among the 262 subsets per layer (even, odd, mixed, without H22). Even-16 does not repair. |
| M3 H22 the only fixing head | **false** | 0 of 1,024 single heads repair, at any layer. |
| M4 H22 across layers | **false** | H22 at the five layers and at all 32 layers: bug. 0 of 32 heads repair in either config. |
| M5 five-layer transplant | **holds, not specific** | it repairs, but 7 of 10 random 5-layer sets also repair. |
| M6 reverse does not induce the bug | **holds** | simple stays correct. Δ drops (12.37 → 3.00) where the draft says it increases. |
| M7 generalization | bug counts chat 6 / comparative 2 / qa 0 / simple 2 (draft 2/19/0/0); "3 of 4 eligible repaired" | In substance: one chat "bug" (8.5 vs 8.12) is correct, and its "failed repair" is a mislabel. One real chat bug (9.7 vs 9.12) was labelled *neither* and not patched. So 3 of 3 genuine tested pairs repaired. The comparative count is unreliable at 64 tokens. |

Flags (substance, no verdict changes):
- 187 patched outputs say "9.8 is smaller than 9.11", which is the bug, but are labelled *neither*. Every
  success criterion is *correct*, so no decision moves. Single-head and parity patches barely change the
  output at all (2 and 4 distinct texts).
- The right alignment is exact on the question tokens, but the recipient's last 4 positions (` [/INST]`)
  receive the donor's `\n Answer :` state. The layer-7 and multi-layer repairs may come from that tail
  rather than from question processing; this would fit 7/10 random 5-layer sets repairing.

