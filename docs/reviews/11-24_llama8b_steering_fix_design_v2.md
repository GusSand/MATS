Reviewer: fresh subagent (Claude Code)
Agent-ID: a34b3051073c27122
Verdict: PASS WITH MINOR FIXES
Transcript-SHA256: 76ae5285028ffe9b1a509e7dce80acabcefa341d113366bce4e7990466f45e6c

## Brief given

You are a fresh, skeptical reviewer (round 2) for a pre-registered experiment in `~/github/MATS`. Read only; no GPU compute, no edits (local `python3` checks are fine).

Files: pre-registration `docs/experiments/11-24_llama8b_steering_fix.md`; script `9_8_research/reviewer_experiments/exp_R20_steering_fix/run_experiment.py` and `run.sbatch`; round-1 review `docs/reviews/11-24_llama8b_steering_fix_design.md` (its fixes were just applied). Helpers it imports: R19 `exp_R19_last_fraction/run_experiment.py`, R18 `exp_R18_eleven_token/run_experiment.py` (classify_pair), R17 `exp_R17_localization_map/run_experiment.py` (Model._register). A CPU smoke run on a tiny random model completed with all code paths.

Check: (1) each round-1 finding is actually fixed correctly in code and doc; (2) any new bugs introduced by the fixes — especially the subset-batched baselines (`gen_cls`, `tbase`, `sbase`, `dev_base`), the label-null construction (`null_cb`, `null_fd`, accumulation keys `ncorrect_/nbug_/nfd_`), the `decide()` changes (controls, 0.02 break threshold, "(<= prompt swap)" qualifier, selftest), JSON-serializability of saved fields (dict keys ints are fine in json), and smoke-only branches; (3) the doc and code agree on every rule and number.

First line: PASS, PASS WITH MINOR FIXES, or FIXES NEEDED. Then numbered findings with severity (blocking / should-fix / minor), file:line, concrete fix. Concise, no praise.

## Report

PASS WITH MINOR FIXES

All 11 round-1 findings are fixed in both the code and the doc. The fixes introduced no blocking bugs.

What I checked:
- **Selftest.** I extracted `decide()` and ran `decide_selftest()` locally with no torch: it prints OK.
- **Every label is reachable.** The 0.02 boundary is exclusive (`br > MAX_BREAK`), which matches "breaks ≤ 2%" in H1. Both the doc and the code put the tie-breaks at line 508 (strict `>`, so a tie keeps the better proxy rank) and at line 513 (fmt, then fd, then cb).
- **Subset baselines.** `dev_base`, `tbase` and `sbase` (`run_experiment.py:371-373`) use the same list order and `BATCH` as the steered runs that are compared against them: dev generation at `:502`, test at `:528/534/539/542`, and `primary_on_simple` at `:582`. The batches line up. The smoke override at `:374-376` runs only after the real passes, so every code path still runs.
- **Label nulls.** Every train question with `q_class` correct or bug gets `null_cb` from the within-order shuffle (`:388-393`). So the `n{null_cb}_{order}` keys at `:434` always resolve to `ncorrect_*` / `nbug_*`, which `:456` reads. `null_fd` is a seeded subset of size `n_fd` (`:394-396`). Both nulls are rescaled to ‖v‖ and use the same α (`:539`). fmt has no null, as the doc says.
- **JSON.** The int keys in `changed_texts` serialize as strings, and every saved field is a plain Python type. No tensors leak into `run`.
- **Smoke mode.** Every smoke branch is gated on `a.smoke`, and `run.sbatch` does not pass `--smoke`. The results prefix is now `r20_`.

Findings:

1. **should-fix: the fd label null is roughly the fmt direction** (`run_experiment.py:394-396`, `:452-453`; doc line 36).
   - The fd null is the mean of (simple − Q&A) over 74 random train questions. That is fmt plus sampling noise.
   - So fd's "no better than controls" check really asks whether the fd-specific selection beats a generic format push at fd's config. It does not compare against a null.
   - The test is legitimate, but the label will be misread.
   - Fix: say this in the doc ("the fd label null is ≈ fmt; it tests whether selecting format-dependent questions adds anything"). Also save `cos(v, null)` and `cos(fd, fmt)` per chosen config in `res["controls"]["label_null"]`.

2. **minor: the qualifier also attaches to negative labels** (`:145-146`).
   - With net > 0 it appends " (<= prompt swap)" to "no reliable fix" and "no better than controls" too. The doc's wording (line 108) allows this, so the two agree.
   - The result reads oddly, e.g. "no reliable fix (<= prompt swap)".
   - Fix (doc and code together, before the run): apply the qualifier only to labels 4–6, or to every label that passes step 2. Add a selftest line for the case.

3. **minor: the compute estimate is stale** (doc line 138).
   - Test generation is now 1 steered + 3 random + 1 label null + 1 −v per direction: 5 for fmt and 6 each for fd and cb. That is 17 × 3,840, plus `primary_on_simple` (3,840) and the subset baselines (2 × 3,840 + 1,280). The total is about 78k generations, not the stated ~50k.
   - Walltime is still fine: about 1,220 batches at about 0.56 s each from R19, roughly 12 min.
   - Fix: update the line.

4. **minor: the label-null seed is not in the doc.**
   - The code uses `random.Random(SEED + 1)`, i.e. seed 21 (`:387`). The doc gives only "seed 20" for the random directions.
   - Fix: add "label nulls seed 21" to doc line 80.

5. **minor: −v saves no texts** (`:542-544`).
   - The doc (line 122) promises "changed rows for controls". `minus_v` stores only counts.
   - Fix: add `"changed_texts": changed_texts(rc, rt)` there.

6. **minor: confusing field names in `base_test`** (`:605-606`).
   - There, `q_text` / `s_text` hold the generated output. In every question dict, `q_text` / `s_text` are the prompt (`:252`).
   - A verifier re-classifying could pick up the wrong field.
   - Fix: rename them to `q_out_subset` / `s_out_subset`.

7. **minor: `resid_norm_q` is the norm of the mean residual, not a typical residual norm** (`:461`).
   - The mean is dominated by the shared offset and outlier dimensions, so it can overstate the scale that ‖v‖ is being compared with.
   - Fix: also accumulate the mean per-token ‖resid‖ in the capture loop (`:426`) and report it next to the current value.

8. **minor: strata use full-pass categories, but transitions use the subset baseline** (`:521`, `:548-554`, `:567`).
   - A format-dependent question whose `tbase` is already correct, because of batch noise, can never count as a fix, yet it stays in the `fd_fix_rate` denominator. A noisy both-correct question can likewise show up as a "fix".
   - The effect is small and `subset_baseline_disagreement` exposes it.
   - Fix: also report the per-category count where `tbase != q_class`. Leave the rule as it is.

9. **process: the files are still untracked.**
   - `git status` shows `exp_R20_steering_fix/`, the doc and the round-1 review as untracked.
   - Fix: commit them after these edits, then compute `R20_SCRIPT_SHA` from the committed file.

Doc and code agree on:
- the layers, α values, sites and 360 configs;
- TOPK = 3 and the 9 generation configs;
- `SIGN_P` 0.01, `MAX_BREAK` 0.02, `BOTH_FIX` 0.3 and `FD_FIX` 0.5;
- the split X sets and the 32 + 32 steered batch-vs-single sample;
- the gate list.

The only mismatches are items 3 and 4.
