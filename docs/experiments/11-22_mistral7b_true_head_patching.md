# R15 — Testing the Mistral-7B head claims with true per-head, prompt-only patching

**Status:** PLANNED (pre-registered 2026-11-22, before any compute; amended after design review round 1)
**Design review:** round 1 FIXES NEEDED — fresh subagent, 2026-11-22 — docs/reviews/11-22_mistral7b_true_head_patching_design.md
**Script:** `9_8_research/reviewer_experiments/exp_R15_mistral_true_heads/run_experiment.py` (imports R14's
patching code; the job refuses to start unless R14's sha256 equals `R14_EXPECTED_SHA256`, computed at submit
time from the committed file). All decision rules are computed in the script (`decide`) into `run.json`.
**Job:** `run.sbatch` (same folder), `a100_tandon`.
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
  answering); every output records `hit_max_new` and `says_less` ("is less/smaller/lower", which the
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
- **Δ** = teacher-forced log P(` 9.8`) − log P(` 9.11`) as the reply's first words, with the same prompt-only
  patch (continuation positions unpatched). Mistral tokenizes ` 9.8` as `▁ 9 . 8`, so Δ compares `8`
  against `1 1` after `▁9.`. **Δ readings are testable only if chat's unpatched Δ ≤ 0** (the draft has
  −1.19); otherwise every Δ "fix" is vacuous and those readings are "untestable". Δ is None when the
  continuation does not extend the prompt's tokens; the not-None rate is reported.

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
- Sanity: the all-heads, all-32-layers transplant must be *correct* (it replaces every attention output on
  the overlapping positions). Fail → report, and treat single-layer nulls with caution.

## Pre-registered decision rules (all on generation outcomes)

Generation reading: success = *correct*. Δ reading (only when testable): success = Δ > 0.
- **S holds:** chat *bug*, comparative *bug*, qa and simple not *bug*, and at least one of them *correct*.
- **M1a** ("no single layer fully repairs", generation): no layer's all-heads patch is *correct*.
  **M1b** ("five layers fix individually", Δ): at least 4 of {2, 10, 11, 13, 17} have Δ > 0.
- **M2 at ℓ ∈ {2, 17}**, generation and Δ readings: (a) the even-only k = 16 set succeeds; (b) no odd-only
  subset succeeds; (c) **even-only subsets without H22 succeed more often than odd-only subsets** (60 vs 60,
  k = 4, 8, 12). Verdict: all three → "parity holds"; (a) and (b) only → "even-only repair carried by H22";
  else "not supported". (H22 is even, so without (c) H22 alone would pass M2; review fix 1.) Every rate is
  reported split by H22 ∈ S; "≈10 even heads for > 50%" is read from the even-only rate by k.
- **M3** (generation): at both ℓ = 2 and 17, H22 is *correct* and no other single head is.
  **M3b** (Δ): H22 has the largest Σ_ℓ (Δ_patched − Δ_unpatched) over the 32 × 32 single-head sweep; the
  draft's top five (H22, H29, H31, H10, H21) are ranked too.
- **M4** (generation): H22 at {2, 10, 11, 13, 17} or at all 32 layers is *correct* (the draft's 7 layers are
  not named). **"H22 special"**: M4 holds and at most 4 of the 32 heads succeed in each of the two configs.
- **M5** (generation): the five-layer all-heads transplant is *correct*; the 10 random 5-layer sets are
  reported as the count-matched control.
- **M6**: M5 holds and the five-layer reverse transplant leaves simple *correct* (not merely "not bug";
  *neither* is reported separately), with its Δ.
- **M7:** error counts per format reported against the draft's (2, 19, 0, 0)/100; "repair transfers" holds
  if the five-layer transplant is *correct* on every eligible pair (and there is at least one).
- **Descriptive only:** Δ for every config; per-head counts across layers; GQA-group breakdown of correct
  single heads; mixed-subset rates.

## Compute estimate

≈ 7 screen + 6 gates + 1 sanity + 32 + 1,024 + 524 + 1 + 10 + 64 + 6 + 400 + ≤100 ≈ 2,180 generations ×
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

(none yet)
