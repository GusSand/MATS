Reviewer: fresh subagent (Claude Code)
Agent-ID: aed69f0aa7964c2b5
Verdict: PASS WITH MINOR FIXES
Transcript-SHA256: 090e56cd65346d3534ef06a8893e69b49b26e621035899f48f0ba7755bedaa5a

## Brief given

You are a fresh, skeptical design reviewer (round 2) for a pre-registered experiment (R19) in ~/github/MATS (commit 5d5e5ee). Read-only: do NOT edit files, do NOT ssh or run GPU jobs. Local checks allowed (`uv run -q --no-project --with torch --with transformers --with numpy python -I ...`; tiny random Llama with the real Llama-3 tokenizer at /private/tmp/claude-502/-Users-gus-github-phd-thesis/d01546c8-ec8c-4e6d-bf61-056487a4472d/scratchpad/tiny_llama3; tokenizer.json at .../scratchpad/tok/tokenizer.json).

Read: docs/experiments/11-23_llama8b_last_fraction.md; 9_8_research/reviewer_experiments/exp_R19_last_fraction/run_experiment.py, run.sbatch, smoke_cpu.log; round-1 review docs/reviews/11-23_llama8b_last_fraction_design.md (9 findings). Verify each round-1 finding was addressed correctly in BOTH script and prereg, and that the fixes introduced no new bugs: competitor position map at every band layer always computed and used for the primary margin; R18-grid exclusion from the decision; decide() labels naming descriptive strata; Wilson CIs; batch-agreement gate made report-only; exit 2 on R18 sha mismatch; sbatch sha check; doc caveats (LFT = token before '?').

Context: R19 tests whether, in Llama-3.1-8B's format-dependent decimal questions (simple correct, Q&A wrong), the flip is carried by the residual stream at the question's last fractional token in both number orders. Background in docs/experiments/11-23_llama8b_eleven_token.md and 11-23_llama8b_localization_map.md. Don't relitigate settled choices unless there is a concrete defect.

Output: first line exactly one of PASS, PASS WITH MINOR FIXES, FIXES NEEDED. Then numbered findings with file:line, severity, concrete fix. Be concise.

## Report

PASS WITH MINOR FIXES

All 9 round-1 findings are fixed in both `run_experiment.py` and the prereg. I found no new correctness bugs. What I checked:
- `decide()` selftest passes.
- The script's sha256 is `2bf2d235…`. It matches `smoke_cpu.log` and is clean at commit `5d5e5ee`, and the R19 directory is now tracked.
- The R18 sha prefix still matches.
- `in_r18_grid` matches R18's grid exactly (W1, X 1..20, Y 6..9, Z 10..13; R18 `run_experiment.py:628`).
- The Wilson formula is correct.
- On canonical, R17's band-layer map has every position except p13 at |r| ≤ 0.29 (p16 ":" is negative). So the stricter any-band-layer margin is not structurally stacked against H.

Round-1 items, verified:
1. Position map. Fixed: computed for every prompt position at every band layer, always, before any gate (`run_experiment.py:258`). The primary margin is the maximum over all positions ≠ site and all band layers (`:265-266`). Same-layer margin and best competitor are reported (`:267-270`). Prereg `:58-62` matches.
2. LFT = token before "?\n". Caveat added (prereg `:25-26`, `:107-108`).
3. Descriptive-stratum labels. `decide()` appends "; <order> descriptive (n < 10)" on rules 2, 4 and 5 (`:113-123`). The selftest covers rules 2 and 4 (`:138-141`).
4. R18-grid exclusion. Grid rows are left out of `strata` and reported in `r18_grid_overlap` (`:387-394`; prereg `:43-44`, `:96`).
5. The competitor map is always computed (see 1). Generation and controls stay gated on r ≥ 0.5, as allowed.
6. Expected n and Wilson CIs. Both added (prereg `:45-47`, `:82`; script `:110`).
7. The batch-agreement check is report-only (`:359`). `stop_if_failed` only sees `gates`, so it cannot halt the run. Per-order unbatched-recheck failures are reported (`:395-397`).
8. The redundancy note is in the prereg (`:83-84`).
9. A wrong R18 sha now exits 2 (`:66-68`). `run.sbatch:22-24` requires `R19_SCRIPT_SHA` and exits 2 on mismatch. The prereg notes that the answer prefix is excluded (`:59-60`).

New minor findings:

1. Prereg `:58-59` has the position counts off by one (minor). Checked with the real tokenizer, the map covers 16 positions for W1/W2 (BOS + 16 tokens) and 17 for W3, so a site has 15 or 16 competitors. Fix: write "16 positions for W1/W2, 17 for W3 (15/16 competitors besides the site)".

2. The module docstring is stale (minor). `run_experiment.py:13-15` still says competitors are evaluated "at each site's best primary-band layer". Fix: say "every prompt position at every band layer (always computed) is the competitor map; primary margin = site's best band r − max over competitors and band layers".

3. The exit-2 promise is wider than the code (minor). Prereg `:77` says "An exception at any point is recorded in `run.json` … exits 2". But these all run before the `try` at `run_experiment.py:198`, and an exception in any of them exits 1 with no `run.json`:
   - `decide_selftest()` (`:187`)
   - R18 `exec_module` (`:72`)
   - `classify_selftest()`, `load_r17()` and `provenance()` (`:193-195`)

   Fix: wrap `main()`'s body in `try/except Exception: sys.exit(2)`, or reword the prereg to "after `run.json` is created".

4. Held-out rows use up the per-order cap (minor). `pool[:MAX_PER_ORDER]` (`:379-381`) is drawn before grid rows are dropped. In small-first, about 7% of the 120 slots (≈ 33/480) go to questions that are then excluded from the decision. That costs n but causes no bias. Fix: either cap at 120 non-grid questions and analyse the grid-overlap questions separately, or state in the prereg that the cap includes held-out rows.

5. `R19_SCRIPT_SHA` is not tied to `R19_GIT_SHA` (minor). The sbatch compares the file against a value the submitter types in (`run.sbatch:23-24`). Fix: record the expected value (`2bf2d23522412e22f919584da7f9f94f7917c1c1e13ce2f76818ba06b616b288`) in the prereg's Job line. Alternatively, if `$SCRATCH/MATS` is a git checkout, compute it with `git show $R19_GIT_SHA:<path> | sha256sum`.

6. Descriptive strata get no CI (minor). For n < 10 (`:104-107`) and for `r18_grid_overlap`, `decide()` reports counts only. Big-first is the stratum likely to be small, so add `wilson()` intervals there too. The function already handles n > 0.

None of these change the decision rule or its implementation. Items 1–3 and 5 are wording or provenance fixes to make before submitting.
