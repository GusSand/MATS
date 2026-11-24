Reviewer: fresh subagent (Claude Code)
Agent-ID: a2e50255571a7e2ef
Verdict: VERIFIED-WITH-FLAGS
Transcript-SHA256: 3945cfc71aeba7e94d90aa24e35b1ebb67b24bfbde890a828bdd0c72e9302d51

## Brief given

You are an independent verifier. Recompute an experiment's results from raw outputs and its pre-registered criteria only. Read-only on the repo `~/github/MATS`; no GPU; put any scratch script in `/private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/blind_r20/`.

Allowed inputs:
- Criteria: `git -C ~/github/MATS show 995daa2:docs/experiments/11-24_llama8b_steering_fix.md` (the version committed before the run).
- Raw outputs: `~/github/MATS/9_8_research/reviewer_experiments/exp_R20_steering_fix/results/r20_24373179/run.json` (baseline texts, steered texts, changed control texts, dev proxy grid, dev generations, provenance).
Do NOT read: the current version of that experiment file, anything in `docs/reviews/`, the journal, the mind map, or the experiment's script. Do not rely on the saved class labels, decisions or summary counts in run.json until you have derived your own from the texts; then compare.

Write your own classifier for who the text says is bigger, document its rules, and derive: baseline counts and categories on the test set; for each direction (fmt, fd, cb) fixes, breaks, net, sign-test p, break rate with Wilson 95% interval, every control's fixes/breaks/net, category × order strata, wording and order splits, the prompt-swap comparator, and the pre-registered label by applying the rule text yourself. Check the dev selection against the stated procedure. Check the standing gates: enough data, leakage between train/dev/test, label correctness, artifact provenance (git sha, script hash, model snapshot, gates, all from run.json's own metadata). Read at least 10 raw texts each of fixes, breaks and unchanged errors for each direction and say what they look like.

First line: VERIFIED, VERIFIED-WITH-FLAGS, or DISCREPANCIES. Then your numbers in tables, every mismatch with saved values, and what conclusions the data do and do not support, including any way each could be wrong. Concise.

## Report

VERIFIED-WITH-FLAGS

All three pre-registered labels reproduce from the raw texts, under two classifier rules. The flags concern interpretation, above all that fd is indistinguishable from its label null, and a few unverifiable items. Scripts: `/private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/blind_r20/{lib,main,a2,a3}.py`. I did not read the script, the log, the reviews, the journal or the current experiment file.

**My classifier** (first line of the answer only)
- "A is bigger/larger…" → A. Bare "A" → A. "A is smaller than B" → B. Full-number match to X.Y (correct) or X.Z (bug), otherwise neither.
- STRICT variant: "smaller" → neither. Used in the tables unless noted. SEMANTIC variant: as above.
- The baselines use only the bigger, bare and smaller forms. "Smaller" occurs 3 times in 19,200 baseline texts.

## Baseline (test, n = 3,840)

| | saved | mine (strict) |
|---|---|---|
| Q&A correct / bug / neither | 3098 / 740 / 2 (neither: "19.17 is smaller than 19.2" and one similar) | same (semantic 3100 / 740 / 0) |
| format-dep | 177 | 187 |
| both-bug | 563 | 553 |
| reverse-dep | 35 | 34 |
| both-correct | 3063 | 3064 |
| other | 2 | 2 |
| prompt swap F / B / net | 179 / 35 / 144 | 188 / 34 / 154 (semantic 187 / 35 / 152) |

- Category differences come from 11–13 simple-format rows where the bare answer contradicts its own explanation (" 6.4\nExplanation: 6.12 is bigger than 6.4"). The saved classifier follows the explanation. I follow the answer. Both are defensible.
- The same effect reaches train (saved 74 format-dep, mine 82) and dev (45 vs 47).
- Q&A baseline classes: 0 mismatches strict, 2 semantic.
- By order, Q&A errors are 462 small-first and 278 big-first. Dev is far more skewed: 124 vs 20. Train: 417 vs 216 (saved counts).

## Directions (test)

| | fmt (":", L9, α1) | fd, PRIMARY (LFT, L4, α4) | cb (LFT, L6, α2) |
|---|---|---|---|
| F / B / net | 70 / 8 / 62 | 227 / 132 / 95 | 175 / 73 / 102 |
| sign p | 1.7e-13 | 6.1e-07 | 7.7e-11 |
| break rate [Wilson 95%] | 0.26% [0.13, 0.51] | 4.26% [3.60, 5.03] | 2.36% [1.88, 2.95] |
| max eligible control net | 6 | 91 (label null) | 83 (random0) |
| fix rate both-bug, saved cats (mine) | .011 (.009) | .176 (.168) | .144 (.143) |
| fix rate format-dep, saved cats (mine) | .356 (.342) | .723 (.717) | .525 (.508) |
| label | small net improvement (<= prompt swap) | net-positive but damaging (<= prompt swap) | net-positive but damaging (<= prompt swap) |
| α·‖v‖ / token norm | 0.93 | 0.31 | 0.40 |

- Labels are identical under semantic scoring: 60, 94, 100 net, vs controls 5, 90, 82.
- Label 4 (both-bug ≥ .3) is never met. Saved decisions, p values, break rates and Wilson intervals all match mine (the interval agrees to 6 decimals).

**Controls (F / B / net)**

| | random0 | random1 | random2 | label null | −v |
|---|---|---|---|---|---|
| fmt | 12/9/+3 | 14/8/+6 | 14/17/−3 | n/a | 21/39/−18 |
| fd | 97/134/−37 | 56/128/−72 | 5/299/−294 | 165/74/+91 | 36/428/−392 |
| cb | 142/59/+83 | 84/157/−73 | 99/87/+12 | 142/725/−583 | 25/486/−461 |

Every control matches saved exactly under strict. cos(fd, null) = .935, cos(fd, fmt) = .950, cos(cb, null) = −.30 (saved values; vectors are not in the file).

**Category × order** (fixes or breaks / n; saved cats in brackets where different)

| | fmt | fd | cb |
|---|---|---|---|
| fd big-first (27), fixes | 0 | 2 | 17 |
| fd small-first (160 [150]), fixes | 64 [63] | 132 [126] | 78 [76] |
| both-bug big (251), fixes | 1 | 1 | 33 |
| both-bug small (302 [312]), fixes | 4 [5] | 92 [98] | 46 [48] |
| rev-dep big (32), breaks | 4 | **32** | 22 |
| rev-dep small (2 [3]), breaks | 0 | 1 | 0 |
| both-correct big (1610), breaks | 4 | 95 | 33 |
| both-correct small (1454 [1453]), breaks | 0 | 4 | 18 |
| other (2), fixes | 1 | 0 | 1 |

**Splits (F / B)**

| | fmt | fd | cb | swap (mine) |
|---|---|---|---|---|
| big-first | 1/8 | 3/127 | 50/55 | 27/32 |
| small-first | 69/0 | 224/5 | 125/18 | 161/2 |
| W1 | 11/0 | 79/43 | 51/18 | 51/13 |
| W2 | 32/4 | 62/40 | 60/17 | 70/9 |
| W3 (unseen) | 27/4 | 86/49 | 64/38 | 67/12 |

- All fmt/fd/cb order and wording splits match saved. W3 net is +23, +37, +26: the gain is not only on seen wordings.
- Format-swap index on 212 saved discordant rows: steered agrees with simple 67 / 160 / 115, unsteered 0. Mine on 221 rows: 68 / 166 / 117.
- Primary vector on simple prompts: saved 157 / 128, mine 154 / 127 (4 rows differ).
- Canonical Q&A 9.8/9.11 (W1, big-first): baseline bug. fd and fmt leave it as bug. cb fixes it. The W3 big-first version is unfixed by all three.

## Selection and gates
- **Dev selection follows the stated procedure.** The grid has 360 configs with net = fix − break throughout. The top 3 per direction match by order and tie rule, the chosen config has the highest dev generation net, chosen == test config, and primary = fd (41 vs 25 vs 21).
- The proxy is unreliable: the top fmt proxy (":", L9, α4, proxy net 25) scored dev generation 0 fixes, 1136 breaks, i.e. every correct row. The generation step caught it.
- **Cannot verify:** dev generation texts are not saved (counts only, though the criteria listed "dev generations"), and neither are the vectors. Control rows whose class did not change have no text. So I checked dev counts for consistency only.
- **Leakage:** train/dev/test X sets are disjoint and match the prereg; test has all 3,840 unique keys. Y and Z values are shared by design. Both orders and 3 wordings of one pair all sit in test, so rows are correlated: 640 pairs, 8 X values.
- **Enough data:** 3,098 correct and 740 errors in test, adequate for the overall label. Thin strata: big-first format-dep 27, reverse-dep 35 (3 small-first).
- **Provenance (run.json metadata):** git 995daa2 equals the criteria commit. Script sha effa9b29… equals the pre-registered hash. The R19 sha starts a4effdba…3b20e as stated. Snapshot 0e9e39f2…, A100-SXM4. Started 06:42, after the 06:31 commit. All gates ok: baseline vs R19 agreement 1.0, zero-vector and hook removal ok, batch vs single 64/64, max batch-vs-single logit difference 0.047. Full and subset baseline texts are identical in 3840/3840 for both formats.
- I took the script hash from metadata only. I did not hash the script file.

## Reading raw texts
I read a random 10 each of fixes, breaks and unchanged errors per direction (fmt has only 8 breaks, all read).
- **Fixes:** a clean flip of the first-line answer, same template. fmt often shortens to "17.5 is bigger." (simple-like; 910 such first lines, plus 23 bare numbers). fd and cb keep "N is bigger than M". Sampled fmt fixes are mostly format-dependent; fd and cb samples mix format-dependent and both-bug.
- **Breaks:** also clean flips, no garbling. fmt and fd breaks are mostly big-first and go to the second-listed X.Z. One fd break gives "bigger because it has more digits after the decimal point". cb breaks include 14.x/W3 runs in both orders.
- **Unchanged errors:** mainly both-bug, with the same stock template, often byte-identical. Explanations restate the wrong answer.
- No output degeneration at test configs.

## What the data support
1. **The primary label is right, and its meaning is "no better than its label null".** fd beats it by only 4 rows (95 vs 91). Paired: 107 rows only fd correct, 103 only null correct, McNemar p = .84. Pair bootstrap CI of the difference [−25, 33]. The rule's "net > control" passes on noise; a tie would have given label 2. Selecting format-dependent questions adds nothing demonstrable.
2. **fd behaves like an order push.** It moves 351 answers to the second-listed number and 7 to the first, for every wording. Small-first +219, big-first −124. It breaks all 32 big-first reverse-dependent rows. Net is positive because baseline errors lean small-first, and dev (124 vs 20) leans much further.
3. **fmt is the only direction meeting the H1 text** (significant net, beats randoms: McNemar p ≤ 6e-12, break rate ≤ 2%). It was not primary. It fixes format-dependent rows and ~nothing in both-bug (4–6 rows), in small-first only (big-first 1 fix, 8 breaks). It flips 76 to second, 1 to first, so it is mostly the prompt swap in miniature. fmt has no label null, and its small randoms are a weak null at ":".
4. **cb:** highest net, but not clearly above random0: 138 vs 119 paired, p = .26, pair CI of the difference [−18, 56]. It is clearly above random1 and random2. Only cb reaches big-first format-dep (17/27), but at this site randoms also fix 25 and 15 (random0, random2), and for fd randoms fix 21 and 19. So that stratum is not specific to any direction. Break rate 2.36%, CI spans 2%.
5. **None beats dropping "Q:"** (swap net 144 saved, 152–154 mine; steered minus swap −92, −59, −52 with pair-CI upper limits below 0).
6. Clustering does not overturn net > 0: pair CIs [44, 81], [53, 140], [66, 139]; X-cluster CIs [44, 82], [38, 157], [60, 157]. The test sign tests and Wilson intervals assume independent rows and overstate precision.

**Not supported:** a fix of the dominant error, a fix specific to format-dependence, "learned direction beats any direction" (cb's random spread is −73 to +83 from 3 draws, so "best random" is a soft bar), or general decimal competence. One model, one template, 3 wordings.

**Ways each could be wrong:** control set is tiny (3 randoms, one label-null draw, whose fd null is ≈ fmt); the α, layer and site were chosen on a dev set skewed to small-first; class noise on ~12 simple rows affects swap and category counts but no label; the break-rate bar is close for cb; fd's label 2 vs 3 outcome hangs on 4 rows.
