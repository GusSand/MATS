Reviewer: fresh subagent (Claude Code)
Agent-ID: a93e7cb4664ff29eb
Verdict: VERIFIED-WITH-FLAGS
Transcript-SHA256: b6d463892d4e9ccb51029af5557334ecb9be907c3ecd6a5088983e7b147ea312

## Brief given

You are a fresh, independent verifier for a finished pre-registered experiment in `~/github/MATS`. Read only, except you may write scratch scripts under /tmp-free locations like `$TMPDIR` or the repo's `outputs/` is NOT allowed — just run `python3 -I` inline. No GPU.

Files:
- Pre-registration (rules fixed before compute): `docs/experiments/11-24_llama8b_steering_fix.md` (R20; Results section not written yet)
- Script: `9_8_research/reviewer_experiments/exp_R20_steering_fix/run_experiment.py` (committed sha256 effa9b29…; check that `provenance.script_sha256` in run.json matches the file)
- Raw output: `9_8_research/reviewer_experiments/exp_R20_steering_fix/results/r20_24373179/run.json` and the job log next to it
- Classifier: `classify_pair` in `9_8_research/reviewer_experiments/exp_R18_eleven_token/run_experiment.py`
- Design reviews: `docs/reviews/11-24_llama8b_steering_fix_design*.md`

Tasks:
1. Recompute from run.json, independently of the script's own summaries: every gate; the subset baselines; for each direction (fmt, fd, cb) the fixes/breaks/net, sign-test p, break rate + Wilson CI, control nets (random0-2, label_null, minus_v), per-category × order strata, by-order/by-wording, format-swap index; the prompt-swap comparator; and the pre-registered label for each direction by applying the doc's rule yourself (including the "(<= prompt swap)" qualifier). Check the dev selection (top-3 proxy per direction, dev-gen choice, primary = highest dev net) follows the doc.
2. Re-classify the saved texts (test steered `texts`, baseline `base_test[*].q_out_subset`/`s_out_subset`) with an independent classifier you write yourself (who is named as bigger first / which number appears as the answer), and report disagreements with the saved classes; look at a sample of fixed and broken texts to judge whether "fixes" are real answers or degenerate outputs.
3. Judge the interpretation I intend to write (challenge it):
   - Primary (fd) label "net-positive but damaging (<= prompt swap)"; control max net 91 is the label null, which is ≈ the fmt direction (cos 0.95), so fd barely beats it.
   - fd behaves like an order bias: fixes small-first (224) and breaks big-first (127), i.e. it pushes toward naming the second number, not toward correctness.
   - cb: label "net-positive but damaging (<= prompt swap)"; one random direction reached net 83.
   - fmt: clean (break rate 0.26%) but small, only small-first format-dependent fixes, ≤ prompt swap.
   - None of the vectors fix big-first format-dependent questions at the LFT (fd 2/27, fmt 0/27, cb 17/27) — where R19's per-question patch repaired 33/37 — suggesting the patch carried question-specific information that a mean direction does not. The canonical 9.8/9.11 Q&A prompt is not fixed.
   - Bottom line: no fixed steering vector is a usable fix here; none beats dropping "Q:".

First line: VERIFIED, VERIFIED-WITH-FLAGS, or DISCREPANCIES. Then numbered findings (with numbers), then flags on interpretation. Concise.

## Report

VERIFIED-WITH-FLAGS

Every number in run.json and the log reproduces from the raw classes and texts, and all three labels follow from the doc's rule. Four of your interpretation statements are wrong or unsupported by the data; details under "Flags".

**Findings**

1. **Provenance and gates.**
   - The script's sha256 is `effa9b29…66fcf5f1`. It matches `provenance.script_sha256` and the doc. The git sha is 995daa2 and the slurm job is 24373179.
   - All gates pass:
     - tokenization: n_bad 0
     - zero-steer logits and generations: identical
     - active steer: max diff 11.58
     - agreement with R19: 1.000
     - hooks removed: OK
   - Batch vs single: logits differ by 0.047; steered classes agree on 64/64 rows.
   - The split matches the doc: test fd 177, both-bug 563, reverse-dependent 35, both-correct 3063, other 2.
   - Subset baselines disagree with the full pass on 0 rows (dev Q&A, test Q&A, test simple). My recomputed categories also match the saved ones on 0 mismatches.
   - Prompt swap: F 179, B 35, net 144.

2. **Dev selection follows the doc.**
   - The proxy grid has 360 configs (120 per direction). I re-sorted it with the stated tie-breaks and got the same top 3 per direction.
   - The dev-generation choice and the primary match:

     | Direction | Chosen config | Dev net | Runners-up (dev net) |
     |---|---|---|---|
     | fmt | colon L9 α1 | 21 | colon L9 α4: −1136 (1136 breaks); colon L4 α2: 19 |
     | fd | LFT L4 α4 | 41 | L8: 37; L6: 30 |
     | cb | LFT L6 α2 | 25 | |

   - Primary = fd (highest dev net, 41).
   - The proxy ranked fmt colon L9 α4 first, but in generation it broke 1136 of 1280 dev questions, so the proxy is poorly aligned with generation at high α.
   - The best dev proxy nets at R19's site (LFT, L9–14) were only 0 (fmt), 10 (fd) and 0 (cb).

3. **Test results.** Baseline-correct n = 3098. All nets are ≤ 144 (the prompt swap), so the qualifier applies to every label. My labels equal the saved ones.

   | | F | B | net | sign p | break rate (Wilson 95%) | controls (net) | max control | fd fix rate | both-bug fix rate | label |
   |---|---|---|---|---|---|---|---|---|---|---|
   | fmt | 70 | 8 | 62 | 1.7e-13 | 0.26% (0.13–0.51%) | r 3, 6, −3 | 6 | 0.356 | 0.011 | small net improvement (<= prompt swap) |
   | fd | 227 | 132 | 95 | 6.1e-7 | 4.26% (3.60–5.03%) | r −37, −72, −294; label null 91 | 91 | 0.723 | 0.176 | net-positive but damaging (<= prompt swap) |
   | cb | 175 | 73 | 102 | 7.7e-11 | 2.36% (1.88–2.95%) | r 83, −73, 12; label null −583 | 83 | 0.525 | 0.144 | net-positive but damaging (<= prompt swap) |

   - −v nets: fmt −18, fd −392, cb −461.
   - Cosines: fd vs its label null 0.935; cb vs its label null −0.298; fd vs fmt 0.950.
   - All strata, by-order and by-wording numbers match the saved values.
     - fd by order: small-first F 224 / B 5; big-first F 3 / B 127.
     - fd by wording (net): W1 36, W2 22, W3 37. fmt W3 net 23; cb W3 net 26.
   - Format-swap index (212 discordant questions; unsteered agrees with the simple-format class on 0): fmt 67, fd 160, cb 115.
   - fd applied to simple-format prompts: F 157, B 128 out of 3242 correct.

4. **Independent re-classification.** My classifier uses the first line only, "A is bigger" means A, "A is smaller than B" means B, and a bare number counts as the answer.
   - Steered texts disagree with the saved classes on 0 (fmt), 1 (fd) and 0 (cb) rows. The fd case is "6.19 is smaller than 6.2": saved "neither", mine correct.
   - With my classifier the nets are fmt 60, fd 94, cb 100.
   - Baseline Q&A: 2 disagreements, both of the "is smaller than" type.
   - Baseline simple: 12 disagreements. Eleven are answers like " 6.4\nExplanation: 6.12 is bigger than 6.4", where R18 calls the row a bug because of the explanation line. Re-classified this way, the prompt swap becomes F 187 / B 35 / net 152. No label changes.
   - The fixes are real answers, not degenerate output. All fd and cb fixes are full "X.Y is bigger than X.Z" sentences; 66 of fmt's 70 are, and the other 4 are bare correct answers. Breaks are clean reversals ("X.Z is bigger than X.Y").

5. **Canonical prompt** (W1, 9.8 vs 9.11, big-first, in test).
   - fd: "9.11 is bigger than 9.8" (bug).
   - fmt: bug.
   - cb: "9.8 is bigger than 9.11" (correct). This is from cb's saved test texts; the pre-registration only reported the primary's canonical output.

**Flags on the interpretation**

A. **fd beats its controls only barely.** Calling it "≈ prompt swap / barely beats the label null" is right: 95 vs 91. Fix the cosines, though. The 0.95 is cos(fd, fmt); cos(fd, label null) is 0.935. cos(label null, fmt) was never measured, and fmt was not run at fd's config (LFT L4 α4). The label null stands in for "generic format push" by construction, not by a measured cosine.

B. **fd as an order bias is correct.** Of fd's 359 changed rows, 351 (98%) moved toward the second-named number. Big-first: 3/278 errors fixed (1%) and 127/1642 correct answers broken (7.7%). Small-first: 224/464 fixed (48%) and 5 broken (0.3%). This is not just where the errors are.
   - Say it is the same thing as the pre-registered "format swap" boring hypothesis. Most format-dependent questions are small-first (150/177) and most reverse-dependent ones are big-first (32/35). fd agrees with the simple-format class on 160/212 discordant questions and breaks 33/35 reverse-dependent ones.
   - It also overshoots the format swap: it breaks 95 big-first both-correct questions, which the prompt swap never breaks.
   - fmt and cb lean the same way: fmt big-first net −7 vs small-first +69; cb big-first net −5 vs small-first +107.
   - Random directions at the LFT also act mainly along this first/second axis, in either direction. fd random0: fixes 86 big-first, breaks 127 small-first.

C. **cb.** The label and the random net of 83 are correct. cb's label null is degenerate (net −583, cos −0.30), so random0 is the control that matters.

D. **fmt.** Correct as stated. Use its real label: "small net improvement (<= prompt swap)". It is the only direction well above its controls (62 vs 6). Note it sits at the colon (layer 9), not the LFT.

E. **"None of the vectors fix big-first format-dependent questions at the LFT" is wrong as written.**
   - cb fixes 17/27, and the canonical prompt with it. Its big-first net is still −5 because of 55 big-first breaks.
   - fmt is not at the LFT.
   - None of the chosen layers (4, 6, colon 9) is in R19's L9–14 range.
   - Random same-norm directions fix these questions easily: fd random0 21/27, fd random1 19/27, cb random0 25/27, and even fd's −v fixes 11. So big-first format-dependent questions are fragile, and fd simply pushes the wrong way on them.
   - That undercuts the inference that "the R19 patch carried question-specific information a mean direction lacks". Dev had no big-first format-dependent questions, and nothing at the R19 site or layers was taken to test. Present the R19 comparison as untested, not as a finding.

F. **The canonical claim needs correcting.** "The canonical prompt is not fixed" holds for fd (the primary) and fmt only. cb fixes it.

G. **The bottom line stands, with nuance.** No vector's net beats dropping "Q:" (144, or 152 with the first-line classifier), and the two that come closest break 2.4–4.3% of correct answers. fmt is the only clean one, and it is about 43% of the prompt swap's net.

My scratch verifier is `/private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/v.py`. No repository files were changed.
