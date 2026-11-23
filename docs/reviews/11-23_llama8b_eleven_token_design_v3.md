Reviewer: fresh subagent (Claude Code)
Agent-ID: aaeae74a74e0de7c0
Verdict: PASS WITH MINOR FIXES
Transcript-SHA256: aa7eb11742f5cf452ccff4d0f0745a16711090e579969dbefb39a62a7ded1c74

## Brief given

You are a fresh, skeptical design reviewer (round 3) for a pre-registered mechanistic-interpretability experiment (R18) in ~/github/MATS (committed at afb5e20). Read-only: do NOT edit files, do NOT ssh or run GPU jobs. You may run small local checks (`uv run -q --no-project --with torch --with transformers --with numpy python -I ...`; a tiny random Llama with the real Llama-3 tokenizer is at /private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/tiny_llama3; real tokenizer.json at .../scratchpad/tok/tokenizer.json).

Read:
- Pre-registration: docs/experiments/11-23_llama8b_eleven_token.md
- Script: 9_8_research/reviewer_experiments/exp_R18_eleven_token/run_experiment.py, run.sbatch, smoke_cpu.log
- Prior reviews: docs/reviews/11-23_llama8b_eleven_token_design.md (round 1) and docs/reviews/11-23_llama8b_eleven_token_design_v2.md (round 2, 10 findings). Verify the round-2 findings were addressed in BOTH script and prereg with no new bugs: length-matched C contexts now use complete filler sentences found by search so the natural tokenization puts "11" at the recipient's position; prereg corruption repaired; D is three-way (specific / generic / inconclusive) with a computed combined label; cosine baseline is a label-permutation null; neutral-prefix caveat and contrasts; listing regex B–E; R17 baseline tolerance 0.04; Transluce reverse wording.
- Context: R17 (docs/experiments/11-23_llama8b_localization_map.md) located Llama-3.1-8B's 9.8-vs-9.11 decision in the "11" token's residual at layers 4-14; R18 tests what it carries, what causes it, and whether it generalizes.

Don't relitigate settled choices unless you find a concrete defect. Focus on code/prereg consistency, reachability of each label, spurious/vacuous gates on the real fp16 A100 run, and runtime risk (e.g., the permutation-null loop cost, the filler search).

Output: first line exactly one of PASS, PASS WITH MINOR FIXES, FIXES NEEDED. Then numbered findings with file:line, severity, and a concrete fix. Be concise.

## Report

PASS WITH MINOR FIXES

All ten round-2 findings are fixed in both the script and the prereg, and I found no new blocking bugs. What I checked:
- **Provenance.** The `run_experiment.py` sha256 is `40c2c567…`, which matches `smoke_cpu.log`. R17's sha256 is `6bf85484…`. The script, sbatch file, smoke log and prereg are tracked and clean at `afb5e20`. The R17 reference `results/r17_24213609/run.json` is tracked, has the expected `cells` keys, and its baseline is 1.171875 / −0.90625.
- **C filler search, run with the real tokenizer.**
  - 17/17 length-matched contexts put "11" at position 15 and end in `[24, 13, 806]`.
  - Every context is made of complete sentences, e.g. "The sky is blue. Snow is cold today. Mark 9.11".
  - The search takes at most 36 tokenizer calls per sentence (about 5 ms in total), so it is not a runtime risk.
- **Prereg repaired.** The Tokenization bullet is restored and there is one compute estimate.
- **D is three-way** (`:203-217`). Specific and generic cannot both hold for a layer (gap ≥ 0.3 vs ≤ 0.2). Generic is not vacuous: R17's r_real at layers 4–14 is 0.55–0.85, so a generic layer needs a random mean of at least about 0.35.
- **Combined label** is computed at `:611` and has selftests.
- **Cosine null** (`:592-601`) is a label-permutation over all C(17,5) = 6188 splits. Its sign matches c (non-decimal mean − decimal mean). Cost is 11 × 6188 small GPU operations, a few seconds.
- **Exact `perm_p`** is about 6k Python iterations per call. Fine.
- **Other round-2 fixes are in place:**
  - Listing regex is `[B-E]` and has the "A:" selftest.
  - R17 baseline tolerance is 0.04, and bitwise equality is recorded.
  - Neutral-prefix contrasts are at `:482-485`, with the caveat in the prereg.
  - The reverse rule is worded "mean of the three category means".
- **Runtime.** A is about 1,920 × 20-token generations (about 15–20 min). B, C, D and E are a few thousand forward passes. That is well under 1 GPU-hour against a 3 h walltime. C's all-layer caches total about 140 MB.

Findings:

1. **Minor (interpretation): the filler is confounded with category in C's primary test.** Short sentences need long fillers, and the short ones are mostly verse and version.
   - "The sky is blue. Snow is cold today." precedes Mark, Revelation, Python, iOS and Firefox (3/4 version, 2/4 verse).
   - Most decimal and date sentences get "Birds fly. …" or "We like long walks in the park.", none of which are used for the short sentences.
   - If the filler changes r, that can show up as a category difference in the "content" or Transluce-reverse rules.
   - Fix: add a caveat to the prereg's "What this can and cannot show" section. Also report a descriptive breakdown of per-sentence `fwd_mean` grouped by filler (the text is already stored in `row["text"]`, `run_experiment.py:513`). Optionally pre-register that the "content" label is downgraded to "mixed" if the filler-group means differ by more than the category gap. Without that, a caveat is enough.

2. **Minor (possible spurious gate failure): gate 5's tolerance is tighter than gate 4's** (`run_experiment.py:438` vs `:431`).
   - Gate 4 accepts each Δ within 0.04 of R17. Gate 5 needs r within 0.02, which is only about 0.04 on Δ (the gap is 2.08).
   - If the environment is not bitwise identical to R17's (torch 2.13.0+cu130, transformers 5.15.1, A100-SXM4-80GB), small fp16 shifts in the baseline and patched Δ can fail gate 5 even though gate 4 passed.
   - Fix: either state in the prereg that gate 5 assumes the same environment (bitwise baselines), or loosen gate 5 to 0.03.

3. **Cosmetic: `decide_transluce` has two docstrings** (`run_experiment.py:191-193`). The second is a no-op string expression. Merge them.

4. **Note, no change needed: B's "every other aligned prompt position" loop** (`:710`) covers prompt positions only, not the teacher-forced " X." prefix tokens. This matches the prereg's wording ("prompt position"). Write it that way in the results so it is not read as "every position".

Every label in `decide_C`, `decide_transluce`, `decide_D`, `decide_combined` and `decide_B` can be reached, and each is exercised by the selftests.
