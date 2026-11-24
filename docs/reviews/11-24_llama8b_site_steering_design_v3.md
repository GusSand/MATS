Reviewer: fresh subagent (Claude Code)
Agent-ID: a212fe4fac72b1bc8
Verdict: PASS WITH MINOR FIXES
Transcript-SHA256: eb23a6650733749bff89d6e216a82d186fee78c0f5cf27dffdae41ef9357f75c

## Brief given

You are an adversarial design reviewer (round 3, focused) for a pre-registered experiment that has not run, in `~/github/MATS`. Read only: no edits, no GPU, no submission, no ssh.

Files: `docs/experiments/11-24_llama8b_site_steering.md`, `9_8_research/reviewer_experiments/exp_R22_site_steering/{run_experiment.py,run.sbatch}`, `scripts/{submit.sh,dry_run_check.py}`. Prior report: `docs/reviews/11-24_llama8b_site_steering_design_v2.md`. Imported scripts: exp_R17…exp_R20 in the same parent directory.

Check that each round-2 finding (1–14) is correctly resolved, that the fixes introduced no new bug (decide, decide_specific, the None dev minimum, partner/paired donor logic and its indexing between `fdi`, `fdq`, `paired` and class lists, small_net in rule 6, per-draw seeds, saved files, preflight logic), and that doc and code agree on every rule and number. Also give any other blocking problem you see. Before reviewing, compute the sha256 of run_experiment.py yourself (`shasum -a 256`), and write the full 64-character hash in your report.

First line: PASS, PASS WITH MINOR FIXES, or FIXES NEEDED. Then numbered findings with severity (blocking / should-fix / minor), file:line and a concrete fix.

## Report

PASS WITH MINOR FIXES

Reviewed `run_experiment.py` sha256, computed here with `shasum -a 256`: `aa9c9b02d66b6f4028b56d16745512fd9ccab809a361c8dd59571b329417dba2`. Its first 16 characters match the `smoke_cpu.log` header.

Also read:
- `run.sbatch` `b400250e6811ccea298e3d793ad7f4f81c70d45dc1ce49a423901a555f9bc84a`
- `submit.sh` `aea15e2a4b74899b325c13908dac0219b536fc91b8c2eb86131cdf9cd3f6941c`
- `dry_run_check.py` `c294410348375cfd27c6fd4ebcf1dda7b69cde7d5c16d3f2b92e095b5651f97b`

I edited nothing in the repo and used no GPU, ssh or submission. I ran the selftests with `torch` stubbed and did local arithmetic. I did not run the model, the preflight (it does a `git fetch`) or `submit.sh`.

There is no blocking finding. **No finding requires changing `run_experiment.py`.** The should-fix items are in the preflight and the submission instructions, so the hash above can stay as the reviewed one. If the script is edited anyway, the preflight will correctly demand a new report.

"Script" = `9_8_research/reviewer_experiments/exp_R22_site_steering/run_experiment.py`. "Doc" = `docs/experiments/11-24_llama8b_site_steering.md`.

## Findings

**1. Should-fix: the preflight's hash and verdict check can be satisfied by the doer, or missed** (`scripts/dry_run_check.py:33`, also :26). This is the remaining part of round-2 finding 4.
- The text after the last Report heading includes any "Doer response" section, which `docs/reviews/README.md` lets the doer write. A hash typed there passes.
- The saved report's own `Verdict:` header is never read. A doc line starting with PASS that links a FIXES NEEDED report naming the current hash passes.
- It splits on the last occurrence of the heading string, even inside prose. The round-2 report quotes that heading in its finding 4, so the hash it states is lost by the split. I confirmed this: 2 occurrences, hash found with a first-occurrence split, not found with the code's split. That one fails safe.
- Fix: reuse `check_reviews.py`: `sys.path.insert(0, "scripts"); from check_reviews import sections, header; s = sections(open(review).read())`. Then require `sha in s["Report"]` and `(header(s["_head"], "Verdict") or "").startswith("PASS")`.

**2. Should-fix: imports and the reference file are checked only if the caller remembers to pass them** (`scripts/dry_run_check.py:12,35`; `submit.sh:2,11`; `run.sbatch:12-13`). This is the remaining part of round-2 finding 5.
- With no extra arguments, the preflight is OK and nothing confirms that R20, R19, R18, R17 or `R17_REF` are on Torch and current.
- Script hashes fail at start, after queueing. `R17_REF` has no hash at run time.
- `run.sbatch:12-13` still gives a plain `sbatch` command, while `scripts/README.md` calls `submit.sh` "the only way to submit".
- Nothing records the exact `submit.sh` command.
- Fix: put the exact command in the doc and in the sbatch comment: `scripts/submit.sh <R22 dir> R22` plus the five files (R20, R19, R18 and R17 `run_experiment.py`, and `exp_R17_localization_map/results/r17_24213609/run.json`). Preferably have the sbatch carry a `# Needs: …` line that the preflight compares with its arguments.
- R13 is not needed. Round 2 listed it, but `load_r13` is called only in R17's own `main` (R17:380), which R22 never reaches. All five files are tracked and unmodified.

**3. Minor: operational, before submit.**
- `smoke_cpu.log` is still ignored (`.gitignore:38`, `*.log`) and untracked. It needs `git add -f`. The preflight now catches this.
- The doc's last design-review line must begin exactly `**Design review:** PASS` and link this saved report. The current "round 1/2 FIXES NEEDED" lines make the preflight refuse, which is right.
- For future experiments: `docs/reviews/README.md` and `check_reviews.py` allow `FIXED`, the preflight does not. Say so in the README row for `dry_run_check.py`.

**4. Minor: "Every random vector, null draw and donor draw has its own seed"** (doc, Conditions; script:364, 456). Each draw index has its own seed, but draw j of the fd null and the cb null share one stream, and so do draw j of the same-wording and the train donors. The result is reproducible and unbiased. Fix the sentence: "each draw index has its own seed; the fd and cb nulls, and the two donor types, share it."

**5. Minor: rule 5's label is wrong for cb** (script:90; doc rule 5). For `cb_selected` the nulls are shuffled-label vectors, so "no better than a generic format vector" misdescribes the failure. For `fmt_selected` there are no nulls and rule 5 never fires, though the doc's Conditions section says "fmt: none". Fix in the doc only: one sentence that for cb rule 5 means "no better than shuffled labels", and that fmt has no null test.

**6. Minor: exact ties at 0.3 and 0.1 are decided by floating point** (script:83, 102, 104, 106). An example is 0.7 − 0.4 < 0.3. For n in {20, 30, 40} with five equal donor rates, 29 of 150 exact-tie cases fall on the wrong side. An exact tie is unlikely with a mean over 5 or 20 draws. Either state in the doc that ties are evaluated in floating point, or check by hand if a difference prints as 0.3000 or 0.1000. Not worth a new hash.

**7. Minor: unpaired questions get their own patch as "other" in the saved classes** (script:459, 463). The rates exclude them correctly, but `classes` and `texts` include them with no marker, and only the count is saved. A verifier must rebuild `solo` from `run["eval"]`. Add a doc note: "unpaired rows in `other` classes are self-patches; ignore them."

**8. Minor: boring-hypotheses wording against rule 2.** The doc says that if own does not beat the pushes by 0.3, "R19's site does not count as replicated". Rule 2 now separates this as "own patch not distinguishable from random pushes". Also, 0.95 against 0.70 is distinguishable, just under the pre-registered margin. Align the sentence, and report the two rates next to the label.

**9. Minor: exit codes** (doc:60). The doc's account of R20/R19/R18 (2), R17 (1) and snapshot (1) is right. Unset `R22_GIT_SHA` or `R22_SCRIPT_SHA` also exits 1, and an sbatch script-hash mismatch exits 2. Add or ignore.

## Round-2 findings 1–14

1. **Resolved.** The headline passes `n_dev_fd=None` (script:503), `decide` skips it, and the early exit tests train and test only (343). Doc rule 1 and the projections are updated. I re-derived the seed-21 split of R19's 48: train 18, dev 5, test 25. An empty dev fd set is handled (`summarise` returns 0.0 and None).
2. **Resolved.** Two labels in the right order, with selftests.
3. **Resolved.** `net + small_net > 0` is in rule 6 in code and doc. Both nets are population-weighted over equal-size populations (6,400 each). The selftest covers it, and the small-first Wilson interval is saved and documented.
4. **Partly resolved** (finding 1). It now uses the last line, requires the report to exist and be committed, and requires the hash.
5. **Partly resolved** (finding 2). Passed extras are checked and hashed on Torch, absolute paths are rejected, and the smoke log must be tracked.
6. **Resolved.** Own, other and random use the same paired set.
7. **Resolved.** The doc says R22, and the `submit.sh` example says R22.
8. **Mostly resolved** (finding 4).
9. **Resolved.**
10. **Resolved.** Secondary labels carry config, layer and alpha.
11. **Resolved.** `duplicate_of_headline`.
12. **Resolved.** In Limits.
13. **Resolved.** `own_fd.pt` has the headline layer for fd rows, with indices into `eval`.
14. **Resolved.** The journal correction names the seven partly resolved findings and the blocking one.

## Checked and correct

- **Indexing:** `fdi` and `partner` use `ev` positions, and `pick` indexes `ev_own` correctly. `paired` holds positions within `fdi`/`fdq`, used directly on the donor `cls` and through `fdi[n]` on the full-`ev` class lists. `fdq` has the same order as both donor tensors. `fd_deltas[[…], kh]` and `ev_own[fdi, kh]` have shape [n, dim].
- **Q2:** own, other and random share the denominator `len(paired)`, and `n_paired` drives the minimum of 20. Thresholds and order match the doc. Zero paired questions give "too few", with no crash.
- **`decide()`:** rules 1–7 match the doc in order, thresholds and strictness. Dev selection and tie-breaking match the doc. The selftests pass on this hash.
- **Seeds:** no collisions between the 2100, 4200, 21000 and 210000 ranges that could reuse a vector. Config index separates the four random sets.
- **Hash chain:** R20, R19, R18 and R17 all match the pinned values or prefixes. The R17 reference has p13 resid cells at layers 9–14 (r 0.707–0.850).
- **Arithmetic:** 960/640/1,600 triples, 57,600 screen generations, 90 dev configs and 129 test conditions agree between doc and code. 4 h walltime is ample. Host memory for train captures is under about 2 GB against 48 GB.
- **Saved files:** `screen.json`, `dev.json`, `vectors.pt`, `own_fd.pt`, baseline and changed texts, all as the doc lists.

Reviewed script hash, repeated: `aa9c9b02d66b6f4028b56d16745512fd9ccab809a361c8dd59571b329417dba2`
