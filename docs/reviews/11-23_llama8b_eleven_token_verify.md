Reviewer: fresh subagent (Claude Code)
Agent-ID: aa8017ed6ae21996e
Verdict: VERIFIED-WITH-FLAGS
Transcript-SHA256: 62d3aa83f34cc33fe3b4513c36cb30029811a0f0eaf0a7bb7a3b70f54ffb2776

## Brief given

You are an independent verifier for a pre-registered mechanistic-interpretability experiment (R18) in ~/github/MATS. Read-only: do NOT edit files, do NOT ssh. Plain python3 is enough (results are JSON).

Inputs:
- Pre-registration (setup, gates, decision rules, caveats): docs/experiments/11-23_llama8b_eleven_token.md
- Script: 9_8_research/reviewer_experiments/exp_R18_eleven_token/run_experiment.py
- Output: 9_8_research/reviewer_experiments/exp_R18_eleven_token/results/r18_24219672/run.json (Slurm job 24219672, A100 ga008, commit ddad516).
- Background: R17 (docs/experiments/11-23_llama8b_localization_map.md) found the canonical 9.8-vs-9.11 decision in Llama-3.1-8B's residual stream at the "11" token, layers 4-14.

Tasks:
1. Gates and provenance: check each gate against the prereg; the script sha256 in provenance equals the local file's sha256; R17 sha256 starts 6bf85484.
2. Recompute independently from the stored rows (not by calling the script's decide_* functions): C's per-category means (matched and original), the exact permutation p, the filler-group means/spread and filler-confound check, the label; the Transluce rule (reverse category means, cosine vs permutation p95); D's per-layer specific/generic flags and label from the stored rand_r/inman_r/r_real lists; the combined label; E's main effects, Q×space interaction, and neutral-prefix contrasts from the E rows; A's class counts, format-dependent count, Which-fix count (re-classify a random sample of ~60 A texts by hand/regex to check the classifier, and inspect any fallback-classified ones); B's eligibility, z/y repair flags (generation class, ctrl mean < 0.2, z margin ≥ 0.3 over other prompt positions) and the label.
3. Interpretation. Assess carefully and concretely what the data support and do not:
   (a) D 'specific direction' — note layers 4-6 have large in-manifold maxima; is 'specific' robust?
   (b) E: the bug appears only with BOTH 'Q:' and ' Which'; what do the neutral-prefix cells show (the ' Which' + neutral prefix '11' state has site r ≈ 0.06 yet the output isn't a bug)?
   (c) C 'mixed': date '11's repair (≈0.56), version '11's push strongly toward the bug (≈−0.55, reverse ≈2.1), 'Let x = 9.11' is strongly bug-ward; filler spread 0.88. What can and cannot be said (post hoc vs pre-registered)? Does this bear on the Transluce verse/date hypothesis?
   (d) A/B: only 39/640 questions are format-dependent; the Which swap fixes 4/39; B 'site does not generalize' with Z 6/39 vs Y 17/39. What does this imply for the scope of the R17 finding?
4. Flags: computation errors, rule mismatches, overclaims, anything a referee would hit.

Report: top-line verdict on its own line, exactly one of VERIFIED, VERIFIED-WITH-FLAGS, NOT-VERIFIED. Then concise sections: gates; recomputation (table of recomputed vs logged); labels; interpretation; flags.

## Report

VERIFIED-WITH-FLAGS

Every gate passes, every logged number and label recomputes exactly from the stored rows, and the script matches the pre-registration. The flags are about interpretation, not computation. The biggest one: B's pre-registered label "site does not generalize" is driven entirely by answer order. In the canonical order (bigger number first) the multi-digit token repairs 6 of 6 questions. In the other order (33 of 39) it does nothing, and the decision sits at the last fractional token instead.

## Gates and provenance
- **Hashes:** local `run_experiment.py` sha256 is `1c5e555b…ebeb88`, equal to `provenance.script_sha256`. `git diff ddad516` on the script is empty. R17's script hashes to `6bf85484302aafd9…`, which equals provenance and starts with the required prefix. Job 24219672, host ga008, `git_sha` ddad516, `all_gates_ok`: true, no error field.
- **Gate 1 (token build):** ok.
- **Gate 2 (positions):** ok; donor "11" at 13, recipient at 15.
- **Gates 3–4 (baseline, R17 equal):** Δ_clean = 1.171875 and Δ_buggy = −0.90625, bitwise equal to R17, even on a different A100 (GPU UUID differs). Torch and transformers versions are the same as R17's.
- **Gate 5 (R17 reproduction):** r at layers 4–14 is bitwise identical to R17's p13 resid values (0.571 … 0.797).
- **Gate 6 (hooks):** hooks_clean and hooks_removed both ok.
- **Gate 7 (E classes):** ok.
- **Gate 8 (C contexts):** 17 of 17 sentences pass in both versions; nothing was skipped.
- **Rule check:** I compared `decide_C`, `decide_transluce`, `decide_D`, `decide_combined` and `decide_B`, the filler check and the B repair conditions against the pre-registration and found no mismatches.

## Recomputation (mine vs logged)
| Item | Recomputed | Logged |
|---|---|---|
| C matched, forward category means (dec / verse / date / version) | 0.036 / 0.171 / 0.564 / −0.552 | same |
| C matched, reverse category means | 1.023 / 0.845 / 0.254 / 2.121 | same |
| C matched, exact permutation p (6188 splits) | 0.5507 | 0.5507 |
| C matched, filler groups and spread | 5 groups, spread 0.8837 | same |
| C original, forward means | −0.058 / 0.736 / 0.810 / −0.282 | same |
| C original, permutation p | 0.8985 | 0.8985 |
| C original, corr(r, "11" position) | 0.295 | 0.295 (matched is null because every "11" sits at position 15) |
| C label (both versions) | mixed (rule-2 ratio 20.3 in matched) | mixed |
| Transluce reverse test | non-decimal mean 1.07 ≥ 0.5, but decimal 1.02 > 0.2 → fails | false |
| Transluce cosine test | mean cos −0.038; \|cos\| < permutation p95 mean 0.082 → fails | false; "does not support" |
| D, specific / generic / 3·SD layers | 8 (layers 7–14) / 0 / 6 (layers 9–14) | 8 / 0 / 6, "specific direction" |
| Combined label | "a specific direction carries the difference" | same |
| E main effects on Δ (Q / space / ending A:) | −0.434 / −1.277 / −0.324 | same |
| E main effects on site r | −0.329 / −0.379 / 0.000 | same |
| E interactions on Δ (Q×space / Q×end / space×end) | −1.852 / 0.586 / 0.898 | same |
| E Q×space on site r (not logged) | −0.406 | — |
| E neutral-prefix contrasts | all four Δ and site-r values match | same |
| A counts: simple / qa / qa_Which (correct–bug) | 453–187 / 418–222 / 407–233, 0 neither | same |
| A format-dependent; fixed by Which | 39; 4 | 39; 4 |
| B: n / Z repairs / Y repairs | 39 / 6 (0.154) / 17 (0.436), no skips | same; "site does not generalize" |

**Classifier check:** I wrote an independent regex classifier ("N is bigger/larger/greater", else the first decimal named). It agreed with the logged class on all 1,920 A texts. A hand-read random sample of 60 was all correct. The 72 fallback-style texts are bare answers such as " 7.10\nHow do we know this?…", and every one is classified correctly.

**B repair flags:** I re-derived the best layers, generation classes, control means < 0.2 and z margins ≥ 0.3 for each of the 39 questions. All 78 flags match.

## Interpretation
**(a) Is D's "specific" robust?** Only for layers 7/9–14.
- At layers 4–6 the real r (0.57 / 0.68 / 0.55) loses to the in-manifold maximum (0.83 / 0.81 / 0.70). Every one of those maxima is a pair that includes "Let x = 9.11", the outlier sentence.
- At layers 4–6 random directions are also close: max 0.53 vs 0.57 at layer 4, with SD ≈ 0.33–0.36.
- Layers 7–8 pass all four conditions but fail the 3·SD secondary. The 3·SD count is 6 of 11, exactly at the threshold.
- So "specific" holds cleanly for layers 9–14 and is marginal earlier; early layers move a lot under almost any 10%-norm perturbation.
- "Specific" means only that donor − recipient beats isotropic random and decimal–decimal directions. It is not a named concept: cos with the verse/date/version axis is about 0.04. The combined label "a specific direction carries the difference" should be worded that narrowly.

**(b) E: the bug needs both `Q:` and " Which".**
- Only the two `Q:` + " Which" cells generate the bug (argmax 806 = "11"). Every other cell's argmax is "8".
- The three "neither" cells are multiple-choice listings (" 9.8\nB: 9.11…") with small positive Δ (0.06–0.20). They lean correct but are not clean "correct".
- Neutral2 + " Which": its "11" state is nearly the recipient's (site r 0.06; negative at layers 4, 6, 8). Yet Δ = +0.08 and the output is a listing, not the bug.
- What that shows: " Which" plus any 2-token prefix produces most of the bug-like "11" state. Q: − neutral2 changes site r by only −0.06, but changes Δ by −0.98. So the content of `Q:` acts downstream of, or in parallel with, the "11" site.
- The "11" state is therefore not sufficient for the bug. R17's "sufficient" holds only inside the `Q:` frame.
- Caveats: two separate "\n" tokens are unusual, the neutral prompts switch the model into multiple-choice mode, and +0.08 is a thin margin for "not a bug". Each cell is n = 1.
- Consistency check: site r is identical across the two endings (ending main effect on site r is exactly 0.0), as it should be.

**(c) C "mixed".**
- Correctly labelled under the pre-registered rules: p = 0.55, and the decimal "11"s do not repair (0.04).
- Pre-registered Transluce reading: not supported. Decimal "11"s break the donor as much as the others (reverse r ≈ 1.0), so any foreign "11" disrupts the donor's correct state.
- Two patterns hold in both versions, but they are post hoc:
  - Date "11"s repair: all 8 sentences positive, 0.43–1.17.
  - Version "11"s push toward the bug: all 8 negative, reverse r 1.5–2.6. This is not explained by filler: the original no-filler sentences show it too, and the verse sentences in the same filler group are not negative.
- They partly cut against Transluce. Verse, date and version readings all imply 9.11 > 9.8, yet date "11"s repair. "Let x = 9.11", a decimal context, is among the most bug-ward of all (−1.0 matched, −1.4 original).
- The layer profiles are very uneven, so a 6-layer mean hides most of the structure:
  - Version is −2.0 at layer 4 and +0.25 at layer 14.
  - Date's repair comes from layers 12–14, at 1.16–1.86 matched and up to 2.4 original. That overshoots the clean answer, which points to off-manifold effects at late layers.
  - Decimal goes from +1.1 at layer 4 to −0.56 at layer 12.
- Verse moves from 0.17 (matched) to 0.74 (original), so position and filler matter. The filler spread (0.88) exceeds every category contrast.
- Category is confounded with template: version sentences are short product names.
- Claimable: "no evidence that the '11' state encodes a verse/date reading; foreign '11' states move Δ in sentence- and layer-specific ways." Not claimable: "date contexts repair" or "version contexts cause the bug" as findings. They are hypotheses for a new pre-registered test, with n = 4 per category.

**(d) A/B scope.**
- Format dependence is a thin slice of the bug: 183 of 640 questions are bugs in both formats, 39 flip from simple to Q&A, and 4 flip the other way.
- R17's site explains the format-induced difference on a few questions, not the decimal bug in general.
- The Which swap fixes 4 of 39, and overall qa_Which is slightly worse than qa (407 vs 418 correct; 25 correct→bug vs 14 bug→correct). The canonical " Which" effect does not carry over to the grid.
- Order dominates the format-dependent set: 33 of 39 are small-first. Q&A bugs: small-first 154/320, big-first 68/320.
- **Key post hoc finding:** all 6 Z repairs are exactly the 6 big-first questions (including 9.8/9.11, 7.6/7.11, 8.8/8.10).
  - Big-first: mean Z r ≈ 1.0 at layers 6–14, Z repairs 6/6.
  - Small-first: Z r ≈ 0 at every layer, Z repairs 0/33. The top competing position is donor position 13, the last fractional token, which is Y in that order. Y repairs 15 of 33 small-first questions (mean r 0.4–0.56); in many others Y's r ≥ 0.5 but the generation stays a bug.
- The better description is that the decision sits at the last fractional token before "?" (donor position 13), whichever number it belongs to. The site generalizes fully in the canonical order on n = 6. The pre-registered label is correct as computed, but it tests the wrong token identity for small-first questions. Report the label, and show the order split as post hoc.

## Flags
1. **B label vs order:** "site does not generalize" will be read as R17 being a one-off. In fact Z repairs 6/6 big-first questions and the site follows the last fractional token. State the label and the post hoc split, with n = 6 for big-first.
2. **Overclaim risk in D/combined:** "specific direction" holds for layers 7/9–14, not 4–6. It carries no semantic content (cosine with the concept axis ≈ 0), and the random null is isotropic, which is weak. Do not cite all of R17's 4–14 band.
3. **Overclaim risk in C:** the date-repair and version-bug patterns are post hoc, n = 4 per category, confounded with template, and very layer-dependent, with overshoot r > 1. Do not present them as evidence for or against particular readings beyond the pre-registered "does not support Transluce".
4. **Sufficiency caveat:** E shows the bug-like "11" state is not sufficient without `Q:` (neutral2 + " Which"). Scope R17's sufficiency claim to the `Q:` frame.
5. **E "neither" cells:** three cells are multiple-choice listings with small positive Δ. Describe them as "8-leaning listings", not "correct".
6. **Scope:** format dependence is 39/640 (6%), and 183/640 are bugs in both formats. R17/R18 localize a format-triggered flip, not the decimal bug itself.
7. **Minor:** the matched corr_r_vs_pos11 is null by design (every "11" at position 15); say so. The Q×space interaction on site r (−0.41) is computable but not logged. B controls use 5 draws, as pre-registered.
8. **No problems found:** no computation errors, no rule mismatches, no provenance problems.

My scratch scripts are in `/private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/` (`v1.py`–`v7.py`).
