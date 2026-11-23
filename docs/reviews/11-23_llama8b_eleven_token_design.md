Reviewer: fresh subagent (Claude Code)
Agent-ID: a2667f290b2926912
Verdict: FIXES NEEDED
Transcript-SHA256: 2bc1c4c9a5ab387c8dac0dad71fdb62101ca0480af685a8c76cd049be2302489

## Brief given

You are a fresh, skeptical design reviewer for a pre-registered mechanistic-interpretability experiment (R18) in ~/github/MATS. Read-only: do NOT edit files, do NOT ssh or run GPU jobs. You may run small local checks (`uv run -q --no-project --with torch --with transformers --with numpy python -I ...`; a tiny random Llama with the real Llama-3 tokenizer is at /private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/tiny_llama3, and the real tokenizer.json at .../scratchpad/tok/tokenizer.json).

Read:
- Pre-registration: docs/experiments/11-23_llama8b_eleven_token.md
- Script: 9_8_research/reviewer_experiments/exp_R18_eleven_token/run_experiment.py, run.sbatch, smoke_cpu.log
- It builds on R17: docs/experiments/11-23_llama8b_localization_map.md (results section and verification flags), docs/reviews/11-23_llama8b_localization_map_verify.md, and R17's script 9_8_research/reviewer_experiments/exp_R17_localization_map/run_experiment.py (Model class: hooks, capture, logits, generate).

R18 asks: on Llama-3.1-8B's 9.8-vs-9.11 Q&A-vs-simple bug, what does the "11" token's residual (shown by R17 to carry the decision at layers 4-14) carry — specific content (cf. Transluce's verse/date hypothesis) or salience — what upstream factor puts it there (Q: prefix vs ' Which' tokenization vs ending), and does the site generalize to other X.Y vs X.Z questions?

Review for: (1) code correctness (token building, position lookups, hook use via R17's Model, generation with explicit ids, classifier generalization incl. regex boundaries and order-reversed questions, part B eligibility and site positions, random controls, reverse direction); (2) whether each decision rule is unambiguous, implemented exactly, and actually discriminates the hypotheses it names (e.g., is the C 'content' vs 'salience' rule sound given foreign contexts differ in length/position; is D's in-manifold null adequate; is B's Y-site a fair comparison); (3) gates that can fail spuriously or pass vacuously (e.g., the R17-reproduction tolerance of 0.02 in fp16; the R17 results file must exist on the cluster at the relative path); (4) missing pieces a referee would demand. Don't relitigate R17.

Output: first line exactly one of PASS, PASS WITH MINOR FIXES, FIXES NEEDED. Then numbered findings with file:line, severity (blocking / should-fix / minor), and a concrete fix. Be concise.

## Report

FIXES NEEDED

The code is mostly correct. I checked these locally with the real Llama-3 tokenizer:
- `Q:` = [48, 25], "Which" = 23956, " Which" = 16299, " 9." = [220, 24, 13].
- "Q:Which…" and "Q: Which…" build as intended.
- All grid pairs X∈1..20, Y∈6..9, Z∈10..13 tokenize as " ", X, ".", frac, and `find_frac` is unambiguous even when X == Z (e.g. 11.11).
- All 8 C sentences I sampled end in "9", ".", "11", including " $9.11".
- The R17 sha256 is `6bf85484…a8c862`, which matches.
- The R17 reference `results/r17_24213609/run.json` is git-tracked, and its `cells` keys (`comp`, `set`="p13", `layer`, `r`) match the lookup at `run_experiment.py:339`.

The problems are in the decision rules: one is close to predetermined (D), and one mislabels what it measures (C).

**Findings**

1. **blocking: D's rule is near-vacuous in the negative direction** (`run_experiment.py:151`, prereg l.97).
   - R17 already measured the same-norm random r at L4–14 as about −0.05 ± 0.34, against r_real 0.55–0.85.
   - With the threshold at rand_mean + 3·SD, r_real would need about 1.0, so "not distinguishable from generic perturbation" is almost guaranteed.
   - That label would then be reported even though random directions average about 0, which is the opposite of the prereg's own salience prediction ("random directions … do about as well", l.30).
   - The rule mixes up the spread of the random effects with their mean.
   - Fix: call a layer specific if r_real exceeds all 20 random r (rank test, p < 0.05) and r_real − rand_mean ≥ 0.3 and r_real − r_inman ≥ 0.3. Report the 3·SD criterion as a secondary result.

2. **should-fix (borderline blocking): the in-manifold null is one draw with an arbitrary sign** (`:427-429`).
   - `dec[0] − dec[1]` is a single vector. Its sign is arbitrary, and the same-direction random draws have SD of about 0.3.
   - Fix: use all 10 decimal pairs in both signs (20 draws). Compare r_real with their mean and maximum, and store the sentence names used.

3. **should-fix: C's "salience" label claims more than the test shows** (`:142`, prereg l.92).
   - None of the foreign "11"s comes from a context that contains the "9.8 or …" comparison or the question frame.
   - So "every category repairs equally" is just as well explained by "the recipient's 11 state, induced by Q:/' Which', is what's abnormal, and any other 11 lacks it". That is reading-independence, not salience.
   - Salience is tested by D, not by C.
   - Fix: rename the label to "reading-independent: every foreign 11 repairs about equally". Reserve "salience" for a combined C+D statement.

4. **should-fix: C's spread threshold is absolute, and the categories are compared on means only** (`:142`).
   - R17's shifted residuals at p13 gave r ≈ 5–6, so foreign "11"s may well overshoot (r > 1).
   - An absolute max − min < 0.15 then fails mechanically, giving "mixed". There is also no check against within-category variation (4–5 heterogeneous sentences each).
   - Fix: use a relative spread, (max − min)/mean < 0.25. For "content", also require a between-category separation, e.g. every decimal sentence's r > every verse/date/version sentence's r, or a 17-sentence permutation test with p < 0.05.

5. **should-fix: the Transluce prediction is not in any rule** (prereg l.96, `:393`).
   - The Transluce reading predicts that verse/date/version 11s create the bug in the donor (reverse direction).
   - It also predicts that `cos_d_concept` < 0, since d = donor − recipient points away from verse-like.
   - Both are reported only, with no expected sign and no baseline.
   - Fix: pre-register the reverse sub-rule, e.g. "non-decimal reverse mean ≥ 0.5 and decimal reverse ≤ 0.2 → supports Transluce". Pre-register the sign of `cos_d_concept`, with a baseline cosine of d against the decimal−decimal differences (residuals are anisotropic, so a 4096-d isotropic baseline is wrong).

6. **should-fix: C's length/position confound is only noted as a caveat** (prereg l.113).
   - Contexts are 6–10 tokens, with "11" at positions 5–9, while the recipient has it at 15. This is near BOS, where massive-activation and attention-sink effects apply.
   - Fix: log ‖foreign 11‖ / ‖h_recipient‖ per row. Add length-matched variants, padding each sentence with a neutral clause before 9.11 so the "11" lands near position 15. Or regress r on position and report it.

7. **should-fix: E does not connect the upstream factors to the "11" site** (`:351-370`).
   - E measures only output Δ and class. The question "what puts it there" needs the factor's effect at the site.
   - Also, `Q:` is confounded with a +2 absolute position shift (the R17 verifier flagged this).
   - Fix 1: for each of the 8 prompts, write its "11" residual at L4–14 into the recipient's "11" and report r per factor. That is 88 passes, and it reuses `find_frac`.
   - Fix 2: add two cells with a neutral 2-token prefix (for example two `\n` tokens) in place of `Q:`, to separate content from position.
   - Also report the 2-way interactions, not only the main effects.

8. **should-fix: B compares Z only against Y** (`:511`, prereg l.74).
   - Calling Z "the" site needs a comparison with the other positions; R17's site was established against all positions.
   - Fix: at the Z site's best layer, also patch every other aligned prompt position (about 16 passes per question). Require Z's r to exceed every other position's by 0.3, or report Z's rank.
   - Y as a single comparison site is fair, since alignment and token identity are matched, but it is not enough on its own.

9. **minor: the classifier is asymmetric** (`:112-116`).
   - The first-line fallback can return "correct" but never "bug". A Q&A answer of just " 9.11" counts as "neither", which undercounts format-dependent questions in A.
   - It is consistent with R13, so it does not affect the canonical pair.
   - Fix: make the fallback symmetric (a first line naming only `small` → "bug") and record a `fallback` flag per row, as R17 did. Then add the order-reversed selftest case `classify_pair(" 1.10 is bigger than 1.6", "1.6", "1.10") == "bug"`.

10. **minor: B takes the first 100 questions in grid order** (`:476`).
    - If many questions are format-dependent, this oversamples X = 1..5 and the big-first order.
    - Fix: take a seeded random sample of 100 from `fd`, or stratify by X and order.

11. **minor: the R17-reproduction gate** (`:336-345`).
    - The tolerance of 0.02 is about 2.7 fp16 quanta of r (step about 0.0075) and should pass if the environment is unchanged.
    - It can fail spuriously if torch or transformers have changed since job 24213609 and the SDPA kernels differ.
    - Fix: also gate on d_clean/d_buggy equalling R17's `gates.baseline` (1.171875 / −0.90625), and record the R17 torch/transformers versions next to R18's so a failure can be diagnosed.
    - The reference file is git-tracked, so it will exist if `$SCRATCH/MATS` is a current checkout.

12. **minor: commit R18 before the job** (`run.sbatch:22`).
    - `exp_R18_eleven_token/` is untracked, so `R18_GIT_SHA` would not identify the script. The script sha256 is recorded, which mitigates this.
    - Fix: commit, then submit with that sha.

13. **minor: some gates are only checked at the end** (`:397`, `:302`).
    - The C_contexts gate is set after `stop_if_failed`, so C, D, A and B all run before the exit-2 at the end. With these sentences it passes (I checked), so it does not matter here.
    - The E prompts that equal the R17 prompts have no class gate (donor correct, recipient bug).
    - Fix: add the class gate and call `stop_if_failed()` after C.

14. **minor: B's random controls are 5 draws at one gap-normalized threshold** (`:523-529`).
    - With gaps as small as 0.5, r is noisier (quantum about 0.03).
    - Fix: report the control SD and the per-question gap, and say in the prereg that B uses 5 draws, not D's 20.

Code paths I checked and found correct:
- Token build and gates 1–2.
- Prefill-only generation in `gen_ids` (same assert logic as R17).
- `capture_resid`, and the reverse r formula (`:393`).
- D's α = 1 equalling the donor value.
- B's eligibility, position lookup and OFFSET for both question orders.
- Regex boundaries: "19.8" and "11.6" vs "1.6" are excluded correctly.
