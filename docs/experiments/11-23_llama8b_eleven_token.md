# R18 — What does the "11" token carry, what puts it there, and does the site generalize?

**Status:** DESIGN (pre-registered 2026-11-23, before any compute; amended after design review rounds 1–2)
**Design review:** round 1 FIXES NEEDED — fresh subagent, 2026-11-23 — docs/reviews/11-23_llama8b_eleven_token_design.md
**Design review:** round 2 FIXES NEEDED — fresh subagent, 2026-11-23 — docs/reviews/11-23_llama8b_eleven_token_design_v2.md
**Script:** `9_8_research/reviewer_experiments/exp_R18_eleven_token/run_experiment.py`. It imports R17's script by
path (sha256 must start `6bf85484…`) for its hooks, captures, teacher-forced logits and prefill-only generation.
**Job:** `run.sbatch` (`a100_tandon`, 3 h walltime).
**Decision:** Gus, 2026-11-23: "yes keep working on this", after R17 (VERIFIED-WITH-FLAGS) and his pointer to the
Transluce Monitor post (https://transluce.org/observability-interface).

## Background

R17 (docs/experiments/11-23_llama8b_localization_map.md) located the decision on the canonical pair: the residual
stream at the "11" token of 9.11 at layers 4–14 is sufficient (r 0.55–0.85) and necessary (reverse_r 0.8–1.5), and
layer-14 attention reads it into the answer prefix. The verifier flagged four open points:
1. The "11" token cannot see "Answer:"/"A:". What it carries comes from the `Q:` prefix or the leading-space
   " Which" token (writing "Which" over " Which" at layer 0 alone repairs, r 0.87).
2. Content vs salience is open: a single-seed random-direction control is weak, and replacing the "11" with another
   token's residual pushes Δ far toward 8.
3. One prompt pair.
4. Stage-2 layers [8, 11, 12] are weakly motivated.

Transluce (Meng et al., 2024) found MLP neurons for Bible verses, dates (September 11) and software versions active
on these prompts. Zeroing the Bible-verse neurons raised accuracy on 1,280 X.Y-vs-X.Z questions from 55% to 76%.
Their hypothesis: Llama reads 9.11 as a verse or date (9:11). R18 tests whether the "11" token's state is of that
kind.

## Simplest (boring) hypotheses, written first

- **Salience:** anything that makes "11" a weaker copy source pushes the answer to 9.8. Foreign "11" residuals of any
  kind repair about equally, and random directions of the same norm do about as well as the real difference.
- **Tokenization:** the error follows the " Which" token, not the `Q:` prefix or the answer cue.
- **Single-prompt artifact:** the site does not carry over to other pairs.

## Setup

Model, snapshot, fp16, hooks and the teacher-forced metric are R17's. For a pair (big = X.Y, small = X.Z):
Δ = logit(Y) − logit(Z) after the prefix " X."; r = (Δ_patched − Δ_buggy) / (Δ_clean − Δ_buggy). Patches are
*resid* (the input of decoder layer L) at one position. Donor and recipient are aligned with offset 2, as in R17.

**Part E — upstream factorial (canonical pair).** 8 prompts: prefix {none, `Q:`} × first word {"Which" (id 23956),
" Which" (id 16299)} × ending {`\nAnswer:`, `\nA:`}, built at token level. Two more prompts add a neutral 2-token
prefix (two "\n" tokens, ids [198, 198]) with "Which" and with " Which" (ending `A:`). These separate the content
of `Q:` from its +2 position shift. Caveat: "\n\n" naturally tokenizes as one token (271), so two separate "\n"
tokens are rare in training and add their own perturbation. Reported contrasts: neutral2 − none and Q: − neutral2,
on Δ and site r, per first-word form, ending `A:`. The token-built R17 donor and recipient must equal the natural tokenization
(gate), and must generate correct and bug respectively (gate). For each prompt:
- greedy generation (20 tokens), its class, Δ;
- **site r**: the prompt's own "11" residual at layers 4–14 written into the recipient's "11" (mean over layers).
  This measures each factor's effect at the site, not only at the output.

Reported: the main effect of each factor on Δ and on site r (over the 8 factorial prompts), and the three 2-way
interactions on Δ.

**Part C — content vs reading-independence.** "11"-token residuals from 17 sentences in four categories. The words
that set the reading come *before* 9.11, because the "11" token cannot see later words:
- decimal: "The price is $9.11", "It weighs 9.11", "Let x = 9.11", "The temperature was 9.11",
  "The measured value was 9.11";
- verse: "Read the Bible verse John 9.11", "As written in Genesis 9.11", "Mark 9.11", "Revelation 9.11";
- date: "The attacks happened on 9.11", "We remember 9.11", "Never forget the attacks of 9.11",
  "On the morning of 9.11";
- version: "Please update to version 9.11", "Python 9.11", "iOS 9.11", "Firefox 9.11".

**Primary: length-matched versions.** Each sentence is preceded by complete neutral filler sentences (from a fixed
pool: "Birds fly.", "Water is wet.", "The sky is blue.", "Cats sleep.", "It rained.", "Snow is cold today.",
"We like long walks in the park.", "The door is open.", "Yes.", "Okay.", "Hello there."; the first ordered
combination of up to 4, in pool order, whose natural tokenization puts the sentence's "11" exactly at the recipient's
"11" position, 15). Example: "Birds fly. Yes. Read the Bible verse John 9.11". A sentence with no exact fit is
skipped (gate 8 counts). Secondary: the original short sentences ("11" at positions 4–9, near BOS). For each sentence and each
layer in {4, 6, 8, 10, 12, 14}:
- forward r (its "11" written at the recipient's "11");
- reverse r (written at the donor's "11"): (Δ − Δ_clean) / (Δ_buggy − Δ_clean);
- Δlogit(8), Δlogit(11), and ‖foreign 11‖ / ‖recipient 11‖.

Also reported: the correlation of r with the "11" position.

**Part D — direction tests (canonical pair, layers 4–14).** d = donor − recipient at the "11" token.
- Interpolation: recipient + α·d for α ∈ {0.25, 0.5, 0.75, 1, 1.25, 1.5, 2}.
- 20 random unit directions (seed 18) scaled to ‖d‖: all values, mean, max, SD.
- In-manifold null: the differences of all ordered pairs of the 5 length-matched decimal "11" residuals (20
  draws, both signs), each rescaled to ‖d‖. All values, mean and max; the sentence pairs are stored.
- ‖d‖ / ‖h_recipient‖; Δlogit(8) and Δlogit(11) for the real patch.
- cos(d, c), where c = mean verse/date/version "11" − mean decimal "11" (length-matched, 12 vs 5). Primary
  baseline: a label-permutation null over all C(17, 5) splits of the same 17 vectors (same group sizes), taking
  the 95th percentile of |cos|. Secondary: the largest |cos(d, e)| over the 20 decimal−decimal differences e.

**Part A — screen (Transluce's grid).** X ∈ 1..20, Y ∈ 6..9, Z ∈ 10..13, both orders (640 questions). Formats:
simple (donor form), Q&A (recipient form), and Q&A with "Which". Greedy, 20 tokens. Classified with a generalized
R13 classifier:
- number-bounded patterns "N is (the) bigger/larger/greater";
- else a symmetric first-line fallback: the first line names only one of the two numbers. The fallback is not used
  when a later line starts like a multiple-choice option B–E (`^\s*[B-E][:.)]\s`, R14's failure mode); a later
  "A:" (the next Q&A turn) does not block it.
- A fallback flag is recorded.

A question is **format-dependent** if simple is correct and Q&A is a bug.

**Part B — does the site generalize?** A seeded random sample (seed 18) of up to 100 format-dependent questions from
A. Eligible if:
- both answers start with the tokens of " X." followed by Y (simple) or Z (Q&A);
- Δ_buggy < 0 < Δ_clean with gap ≥ 0.5.

Sites, each patched as *resid* at layers 4–14:
- Z, the multi-digit fractional token (the analogue of "11");
- Y, the single-digit fractional token (the analogue of "8").

At each site's best layer, if r ≥ 0.5: a prefill-only generation and 5 random same-norm directions (mean and SD
reported; B uses 5 draws, not D's 20). At Z's best layer, every other aligned prompt position is also patched.
- **Z repairs** a question if the generation is correct, the mean random r < 0.2, and Z's r exceeds every other
  position's r by ≥ 0.3.
- **Y repairs** if the generation is correct and the mean random r < 0.2.

Each question's gap is reported (r is coarser when the gap is small).

## Gates (any failure: exit 2)

1. **Token build:** the token-built R17 donor and recipient equal the natural tokenization.
2. **Positions:** the recipient's "11" sits at donor "11" + 2.
3. **Baseline:** Δ_buggy < 0 < Δ_clean, gap ≥ 1.
4. **R17 baseline equal:** Δ_clean and Δ_buggy within 0.04 (about 2 fp16 ulps) of R17's (1.171875, −0.90625);
   bitwise equality is recorded. R17's torch, transformers and GPU strings are recorded next to R18's.
5. **R17 reproduction:** resid at the "11" token, layers 4–14, reproduces R17's r within 0.02 at every layer.
6. **Hooks clean** (before) and **hooks removed** (after): unpatched recipient logits bitwise equal.
7. **E canonical classes:** the token-built R17 donor generates correct and the recipient a bug.
8. **C contexts:** at least 2 sentences per category tokenize to end in "9", ".", "11", in both versions.

Gates 1–6 are checked before E; gates 7–8 after C, before D. An exception at any point is recorded in `run.json`
and the job exits 2.

## Pre-registered decision rules

- **C** (primary: length-matched; per-sentence mean forward r over the 6 layers; first match):
  1. decimal mean ≥ 0.5, every other category mean ≤ 0.2, decimal − max(others) ≥ 0.3, and an exact one-sided
     permutation test of mean(decimal) − mean(others) over the 17 sentences gives p < 0.05 → "content:
     decimal-context 11 repairs, verse/date/version 11 do not";
  2. every category mean ≥ 0.3 and (max − min) / mean of the category means < 0.25 → "reading-independent: every
     foreign 11 repairs about equally". This says the recipient's "11" state is what is abnormal, and that any
     "11" from elsewhere lacks it. It does not by itself say "salience"; that needs D.
  3. every category mean < 0.2 → "context-bound: no foreign 11 repairs";
  4. otherwise → "mixed".
  The same rule is applied to the original short contexts (secondary).
- **Transluce reading** (both pre-registered):
  - reverse: the mean of the three category means of reverse r (verse, date, version) ≥ 0.5 and the decimal
    category mean ≤ 0.2;
  - cosine: the mean over layers of cos(d, c) < 0 (d points away from the verse/date/version direction), and its
    absolute value exceeds the mean over layers of the permutation null's 95th percentile.

  Both → "supports the Transluce reading"; one → "partly supports (reverse only / cosine only)"; neither → "does
  not support the Transluce reading".
- **D:** a layer is *specific* if all four hold:
  - r(α=1) > every one of the 20 random r;
  - r(α=1) − random mean ≥ 0.3;
  - r(α=1) > every in-manifold r;
  - r(α=1) − in-manifold mean ≥ 0.3.

  A layer is *generic* if the random mean ≥ r(α=1) − 0.2. Label (first match): "specific direction" if ≥ 6 of
  the 11 layers are specific; "generic: random directions do about as well" if ≥ 6 are generic; else
  "inconclusive". Secondary, reported: the number of layers with r(α=1) > random mean + 3·SD.
- **Combined** (computed in code, `run["combined"]`): C "reading-independent" and D "generic" → "salience: foreign
  11s and random directions both repair"; D "specific" → "a specific direction carries the difference"; otherwise
  "no combined claim".
- **E and A:** descriptive (classes, main effects, interactions, counts, Which-swap fixes).
- **B** (eligible questions only; first match):
  1. n < 10 → "descriptive only";
  2. Z repair rate ≥ 0.5 and Z − Y ≥ 0.3 → "site generalizes: the multi-digit fractional token carries the
     decision";
  3. Z rate < 0.2 → "site does not generalize";
  4. else "partial".
- `decide_C` (with the permutation test), `decide_transluce`, `decide_D`, `decide_combined`, `decide_B` and the
  classifier have start-up selftests.

## What this can and cannot show

- C asks whether the "11" state can be swapped between readings. It cannot name neurons; that is stage 2.
- Length matching puts each "11" at the recipient's position, but the filler text differs from the question frame.
  "Context-bound" may therefore reflect the frame, not the reading.
- R2/R4 (legacy) found no other format-dependent decimal pair with the canonical prompt words. B may have few
  eligible questions. If so it reports descriptively, and the paper states the scope.

## Compute estimate

- A: 1,920 generations, about 15–20 min.
- C: 34 × 6 × 2 passes.
- D: 11 × (7 + 1 + 20 + 20).
- E: 10 × (11 passes + 1 generation).
- B: up to 100 × (22 + 16 passes) plus generations.

About 1 GPU-hour; walltime 3 h.

## Results

(none yet)
