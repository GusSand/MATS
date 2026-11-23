Reviewer: fresh subagent (Claude Code)
Agent-ID: a1e8408d17f2d5bb3
Verdict: FIXES NEEDED
Transcript-SHA256: e80d7473f8b535ee0fb935126807e06a0f427d8586a2f27e1a3bc8c8218df756

## Brief given

You are a fresh, skeptical design reviewer (round 2) for a pre-registered mechanistic-interpretability experiment (R18) in ~/github/MATS. Read-only: do NOT edit files, do NOT ssh or run GPU jobs. You may run small local checks (`uv run -q --no-project --with torch --with transformers --with numpy python -I ...`; a tiny random Llama with the real Llama-3 tokenizer is at /private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/tiny_llama3; real tokenizer.json at .../scratchpad/tok/tokenizer.json).

Read:
- Pre-registration: docs/experiments/11-23_llama8b_eleven_token.md
- Script: 9_8_research/reviewer_experiments/exp_R18_eleven_token/run_experiment.py, run.sbatch, smoke_cpu.log
- Round-1 review: docs/reviews/11-23_llama8b_eleven_token_design.md (14 findings). Verify each was addressed correctly in BOTH script and prereg, and that the fixes introduced no new bugs (the classifier fallback is now symmetric with a multiple-choice-listing exclusion; C has length-matched contexts and a permutation test; D has 20 in-manifold draws and a rank-based rule; E adds site r and a neutral 2-token prefix; B samples randomly and compares Z against all other positions; new gates).
- Context: R17 (docs/experiments/11-23_llama8b_localization_map.md, results + verifier flags) located Llama-3.1-8B's 9.8-vs-9.11 Q&A-vs-simple decision in the residual stream of the "11" token at layers 4-14. R18 asks what that state carries (content per Transluce's verse/date reading vs reading-independent/salience), what upstream factor causes it, and whether the site generalizes.

Focus on: code/prereg consistency; decide_* functions vs written rules; whether each rule can actually reach each label; gates that can fail spuriously or pass vacuously on the real fp16 A100 run; correctness of the length-matched context construction, the neutral prefix, site_r, permutation test, cosine baseline, B's other-positions comparison. Don't relitigate settled choices unless you find a concrete defect.

Output: first line exactly one of PASS, PASS WITH MINOR FIXES, FIXES NEEDED. Then numbered findings with file:line, severity (blocking / should-fix / minor), and a concrete fix. Be concise.

## Report

FIXES NEEDED

Round-1 items #1–11, #13 and #14 are fixed in both the script and the prereg, and the code matches the written rules. Item #12 (commit the script before the job) is still open. Checks I ran:
- Script sha256 is `6807d631…` and matches the smoke log. R17's sha256 is `6bf85484…`.
- All 17 length-matched contexts put "11" at position 15, which is `zpos_r`.
- `perm_p` is exact: C(17,5) = 6188 splits, so the smallest possible p is 1/6188.
- The in-manifold null has 20 ordered pairs (10 pairs in both signs).
- At α=1, `h_r + d` is exactly the donor value.
- E's `site_r` for the recipient prompt itself gives 0 by construction, so it doubles as a built-in check.
- B's other-positions loop covers donor positions 1..len(dp)−1 mapped to +OFFSET and skips only Z.
- In `decide_B`, all four labels can be reached.
- The symmetric fallback and its selftests are correct.

Findings:

1. **Blocking: the length-matched C contexts (the primary C test) are run-on fragments with no sentence boundary** (`run_experiment.py:470-472`, prereg l.59-61). Built with the real tokenizer they read:
   - "…line of The price is $9.11"
   - "…plain text Mark 9.11"
   - "…plain text Python 9.11"
   - "…simple Read the Bible verse John 9.11"

   "text Mark 9.11" can read as the noun or verb "mark". Gluing a truncated filler straight onto the sentence weakens exactly the verse/date/version readings C is testing. That biases C toward "reading-independent"/"mixed" and the reverse-r Transluce rule toward "does not support".

   Fix: `[BOS] + filler[:need-1] + [SEP] + body`, where SEP is one boundary token: "\n" (198), or "." (13) followed by the body's leading-space word. Recompute `need` so "11" stays at 15. Gate on `pos11 == zpos_r` rather than the hard-coded `TARGET_POS`. Better still, use a filler made of complete short neutral sentences.

2. **Should-fix: the prereg text is corrupted.**
   - l.32: "- **Tokenizatio## Setup": the "Tokenization" boring hypothesis, and possibly more, was cut, and the Setup heading is fused into it.
   - l.177-180: a stale second compute estimate contradicts l.169-175 ("C: 17×6×2", "D: 11×(7+1+20+1)", "B: ≤100×22").

   Fix: restore the truncated hypothesis bullet(s) and delete l.177-180. This is the record, so it needs to be right before the job is submitted.

3. **Should-fix: the combined "Salience" reading is not implemented, and D's negative label is a catch-all** (prereg l.148; `run_experiment.py:199-206`, and nothing after `:567`).
   - No code computes "C reading-independent AND D not distinguishable".
   - "Not distinguishable" is reached whenever fewer than 6 layers pass all four tests. With R17's random SD of about 0.34, the max of 20 random draws is about mean + 1.9 SD ≈ 0.6, against r_real of 0.55–0.85. So the label can fire while random directions average about 0. That is the opposite of the salience prediction ("random directions do about as well", l.30).

   Fix:
   - Make D three-way: "specific" as now; "generic" if the median-layer `rand_mean ≥ r_real − 0.2` (or ≥ 0.5·r_real); otherwise "inconclusive".
   - State "salience" only for C reading-independent + D generic.
   - Compute this in code (`run["combined"]`), add a selftest, and update the prereg.

4. **Should-fix: the cosine baseline in the Transluce rule is not like-for-like** (`run_experiment.py:547,553-556`; prereg l.75-77, 135).
   - c is a 12-vs-5 mean difference, which averages away sentence noise.
   - Each baseline e is a 1-vs-1 difference (noisier, so lower |cos|), then maximized over 20 draws.
   - Which way this biases the rule is unknown.

   Fix: use a label-permutation null on the same 17 cached vectors, with no extra forward passes. Compute cos(d, mean(random 12) − mean(other 5)) over all C(17,5) splits (or 2,000 seeded ones) and require |cos_concept| > the 95th percentile of |cos|. Keep the current baseline as secondary.

5. **Minor: the neutral prefix [198, 198] is an unnatural token sequence** (`:55`, prereg l.40). "\n\n" tokenizes as the single token 271. Two separate "\n" tokens rarely occur in training, so this cell adds an out-of-distribution perturbation on top of the +2 shift. Natural candidates I checked ("---\n", "#\n", "...\n" before "Which") are all 1 token.

   Fix: either find a natural 2-token, content-free string (check that the natural tokenization equals the token build, as in gate 1), or add a third cell [271, 198]. At minimum, state the caveat in the prereg.

6. **Minor: E's neutral-prefix contrast is not computed** (`:449-456`). The prereg (l.40-41) says these cells "separate the content of Q: from its +2 position shift", but `E_effects` has no such contrast. Fix: report Δ and `site_r_mean` for neutral2 − none and Q: − neutral2, per Which form, all with ending A:.

7. **Minor: the multiple-choice exclusion also catches the Q&A answer marker** (`:126`).
   - `^\s*[A-E][:.)]` matches "A:" and "E.g.". I verified that " 9.8\nQ: Which is bigger: 9.7 or 9.12?\nA: 9.7", " 9.11\n\nA: 9.11" and " 9.8\nE.g. …" all give "neither".
   - With 20 tokens the "A:" line is usually just out of reach, so the effect is small. It can still undercount Q&A-format classifications in A.

   Fix: use `^\s*[B-E][:.)]\s` (R14's failure starts at "B:"), or require at least 2 option-like lines. Add a selftest for " 9.8\nQ: …\nA: 9.7" → "correct".

8. **Minor: the R17 baseline-equality gate (1e-6) is a bitwise check on fp16 logits** (`:407-408`). Logits around 15–25 have an fp16 spacing of 0.0156. Any kernel or node difference fails the gate and exits before E, which also makes the 0.02 tolerance on the reproduction gate redundant.

   Fix: either state in the prereg that bitwise equality is intended (environment pinning), or loosen it to |Δ diff| ≤ 0.04 (about 2 ulps) and record the exact values.

9. **Minor: the Transluce reverse sub-rule averages category means** (`:191`), while the prereg (l.134) says "the mean reverse r of verse, date and version". With 4/4/4 sentences the two agree, but they diverge if a sentence is dropped. Fix: say "mean of the three category means" in the prereg, or pool the sentences in code.

10. **Minor: provenance.**
    - The E line in `smoke_cpu.log` is cut off mid-tuple. Regenerate the log verbatim.
    - `exp_R18_eleven_token/` and the prereg are still untracked (round-1 #12). Commit both, then submit with that `R18_GIT_SHA`.
