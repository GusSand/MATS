# R15 — Testing the Mistral-7B head claims with true per-head, prompt-only patching

**Status:** PLANNED (pre-registered 2026-11-22, before any compute)
**Script:** `9_8_research/reviewer_experiments/exp_R15_mistral_true_heads/run_experiment.py` (imports R14's
patching code; both sha256 recorded)
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

- `mistralai/Mistral-7B-Instruct-v0.2` (snapshot 63a8b081…), bf16 (native), A100 (`a100_tandon`),
  greedy, **40** new tokens (the instruct model often answers in a full sentence). 32 layers, 32 query heads,
  8 KV heads (GQA groups 4g..4g+3), head_dim 128, `o_proj` without bias.
- Patching: R14's `Patcher` (true heads = `o_proj` input via forward-pre-hook; prompt pass only; right
  alignment, position 0 never patched). Multi-layer configs register one pre-hook per layer, each with its
  own donor activations; all hooks act on the prompt pass only.
- **Classifier:** R14's (`classify`, with its 18-case selftest).
- **Formats (fixed):** *chat* = the tokenizer's chat template around `Which is bigger: 9.8 or 9.11?`;
  *comparative* = `Compare 9.8 and 9.11. Which is larger?`; *qa* = `Q: Which is bigger: 9.8 or 9.11?\nA:`;
  *simple* = `Which is bigger: 9.8 or 9.11?\nAnswer:`.
- **Pair rule:** recipient = chat if *bug* (the draft's target format); donor = simple if *correct*, else qa
  if *correct*. No pair → stop: "no format-dependent bug in chat on our formats; M1–M6 have nothing to
  repair". M7's screen still runs.
- **Secondary metric Δ** (continuous, descriptive): teacher-forced log P(` 9.8`) − log P(` 9.11`) as the
  first words of the reply, with the same prompt-only patch. Reported next to every generation outcome.

| Cell | Configs | Tests |
|---|---|---|
| screen | 4 formats, unpatched | S |
| layer sweep | all 32 heads at each layer (32) | M1 |
| single heads | each head at each layer (1,024) | M3 |
| parity | layers 2 and 17 × {even-only, odd-only, mixed} × k ∈ {4, 8, 12, 16} × up to 20 distinct seeded subsets (fewer when fewer exist: one even-only and one odd-only set at k = 16) | M2 |
| multi-layer | all heads at {2, 10, 11, 13, 17}; H22 at {2, 10, 11, 13, 17}; H22 at all 32 layers | M4, M5 |
| reverse | chat activations into the simple prompt (all heads): each of the five layers, and all five together | M6 |
| generalization | 100 seeded X.D vs X.DD pairs (shorter larger, X ∈ 1..9) × 4 formats; on each pair with chat *bug* and simple *correct*, the five-layer transplant | M7 |

## Gates

- `selftest` (R14) passes; head isolation (head 1 at layer 2); identity (own activations at all heads,
  layers {2, 10, 11, 13, 17}, equals unpatched), determinism (×3) and hook shapes (prompt pass only) for the
  five-layer config. Any failure → stop.
- Sanity: the all-heads, all-32-layers transplant must be *correct* (it replaces every attention output on
  the overlapping positions). Fail → report, and treat single-layer nulls with caution.

## Pre-registered decision rules (all on generation outcomes)

- **S holds:** chat *bug*, comparative *bug*, qa not *bug*, simple not *bug*.
- **M1:** reported as the set of layers whose all-heads patch is *correct*. "Five layers fix individually"
  holds if at least 4 of {2, 10, 11, 13, 17} are *correct*; "no single layer fully repairs" holds if none is.
  (Under generation these exclude each other; the draft may have meant Δ. Both are reported with Δ.)
- **M2 holds at layer ℓ** if (a) the even-only k = 16 set is *correct*, (b) no odd-only subset at any k is
  *correct*, and (c) more even-only than odd-only subsets are *correct* (equal numbers tested per k).
  "≈10 even heads for > 50%": reported as the even-only correct rate at each k.
- **M3 holds** if at both ℓ = 2 and 17, H22 is *correct* and no other single head is.
- **M4 holds** if H22 at {2, 10, 11, 13, 17} or H22 at all 32 layers is *correct* (the draft's 7 layers are
  not named).
- **M5 holds** if the five-layer all-heads transplant is *correct*.
- **M6 holds** if M5 holds and the five-layer reverse transplant leaves simple not *bug*.
- **M7:** error counts per format reported against the draft's (2, 19, 0, 0)/100; "repair transfers" holds
  if the five-layer transplant is *correct* on every eligible pair (and there is at least one).
- **Descriptive only:** Δ for every config; per-head counts across layers; GQA-group breakdown of correct
  single heads; mixed-subset rates.

## Compute estimate

≈ 32 + 1,024 + ~470 + 3 + 6 + 400 + ≤100 ≈ 2,050 generations × ~1.2 s (40 tokens, 7B, A100) ≈ 0.7 h, plus Δ
forward passes (cheap). Walltime 3 h.

## Limitations known before the run

- Prompts, Δ definition and the "7 layers" are not the draft's; a mismatch with the draft can come from
  these alone. The write-up says "tested", not "reproduced".
- One canonical pair for M1–M6.
- The draft's "n = 20 per condition" and "n = 1000 per format" were repeats of greedy runs or unknown
  sampling; here every configuration runs once and n counts distinct subsets or pairs.

## Results

(none yet)
