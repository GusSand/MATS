Reviewer: fresh subagent (Claude Code)
Agent-ID: abf09667b747ce217
Verdict: FIXES NEEDED
Transcript-SHA256: d9694ed4fb2ac50f7665aca27179fc2f1612de0b8c8dd4dbadec33e0d3148360

## Brief given

You are a design reviewer for an experiment that has not run yet. Judge whether it will answer its question. Be adversarial and concrete, and flag issues that could change a verdict or lose data. Do not edit any files.

Repo: ~/github/MATS.
- Pre-registration: docs/experiments/11-22_llama8b_position0_gpu_gap.md
- Script: 9_8_research/reviewer_experiments/exp_R16_position0/run_experiment.py
- Job files: run_a100.sbatch and run_l40s.sbatch, in the same folder.
- CPU smoke log: smoke_cpu.log, in the same folder. It used a random tiny Llama; the gates fail by design there.

Background:
- R13 (docs/experiments/11-21_llama8b_decimal_true_head_patching.md; script 9_8_research/reviewer_experiments/exp_R13_true_head_patching/run_experiment.py) ran the same script on an L40S (results/r13_24014365) and an A100 (results/r13_24084222). Software was identical.
- In the post_wo cells, outcomes differ between the GPUs on 40 configs (`post_wo/prefill`) and 60 configs (`post_wo/all_steps`). All of them go bug on L40S → correct on A100.
- The verification reports, docs/reviews/11-21_llama8b_decimal_true_head_patching_verify*.md, record position-0 clean−buggy max diff at layer 10: 0.0 on L40S and 6.1e-5 on A100.
- R16 tests whether that position-0 difference causes the gap. It modifies only the donor tensor's position 0, and does not touch R13's hook.

Check at least:
1. Does replacing donor position 0 with the buggy run's own position 0 implement "don't patch position 0" in R13's `post_wo/prefill` cell (left-aligned, prefill only)? In `post_wo/all_steps`, R13's decode steps write donor position 0 into each new token. Is the keep0 replacement the right "exact-arithmetic no-op" counterpart there? Read R13's `_swap` and `generate`.
2. Is the captured buggy position 0 bitwise equal to the live buggy prefill's position 0? Consider separate forward calls, and `capture` vs `generate` paths, e.g. SDPA kernels with and without KV cache. If not, keep0 is not a pure no-op on A100. How could you check that inside the job cheaply?
3. Inject: L40S buggy0 + A100 delta, computed in fp32 and cast to fp16. Does that reproduce what the A100 writes, or something else? Should it instead write the A100's clean0 directly, or L40S clean0 + delta? Which is the right reverse test?
4. Gates: does the copy cell reproducing R13 output-for-output make sense on a possibly different node? What if it fails?
5. Decision rules: is r over D, together with N_new, sound? Are the thresholds sensible? Is the D computation right? Does the all_steps cell test the R8 reproduction gap the way the pre-registration says?
6. The limitation that only layer 10's position 0 is changed. Is the hypothesis even well posed? On A100, every layer's position-0 activations differ slightly between the clean and buggy runs, and so do the unpatched downstream computations. Could keep0 at layer 10 fail to remove the gap even if "position 0" is the cause? Is there a cheap additional cell that would make the result decisive? One example: keep0 applied at all layers via the residual stream.
7. Data safety, provenance, job chaining (`afterok`, the delta path), and walltime.

Write your review as markdown. Start with a verdict, PASS or FIXES NEEDED, then required fixes with file:line, then optional suggestions. Return the full review as your final message; it will be saved verbatim.

## Report

# R16 design review: position 0 and the L40S vs A100 gap

## Verdict: FIXES NEEDED

The core mechanics work. `keep0` puts the right value in the right place in both cells, and the chaining and data handling are basically sound. Five problems remain, and each could produce a wrong verdict:

- **Silent false negative.** If the new A100 node's position-0 difference (d0) is zero, the run reports "does not explain" without warning.
- **The no-op is assumed, not shown.** The pre-registration claims `keep0` is a bitwise no-op, but nothing supports that.
- **The `all_steps` R8 reproduction statistic is not computed.** The pre-registration promises it.
- **No baseline for how fragile D is.** D is selected for being near the decision boundary, so even an unrelated perturbation will flip some of it.
- **A free, sharp test is missing.** The block structure of the position-0 difference decides much of the question before any run.

All numbers below come from the R13 results files under `exp_R13_true_head_patching/results/`. I recomputed them.

### Facts from the R13 data that matter for this design

| cell | D (class differs, L40S vs A100) | text differs | direction | R8 configs in D |
|---|---|---|---|---|
| `post_wo/prefill` | 40 | 44 | all 40: L40S bug → A100 correct | 37 |
| `post_wo/all_steps` | 60 | 69 | all 60: L40S bug → A100 correct | 58 |
| `post_wo/decode_only` | 0 | 1 | none | 0 |
| `pre_wo/prefill` | 2 | 2 | bug → correct | 0 |
| `pre_wo/all_steps` | 1 | 20 | bug → correct | 0 |
| `pre_wo/prefill_right` | 0 | 2 | none | 0 |

- The R13 nodes were `ga004` (A100-SXM4-80GB) and `gl044` (L40S), both on driver 610.43.02.
- R13 cells took 586–649 s each. On A100, `bos_max_abs_diff` is 6.1e-5 for **both** `pre_wo` and `post_wo`.
- So `pre_wo/prefill` also copies a one-ULP difference at position 0 on A100, yet its gap is only 2 configs. This is a prior against the hypothesis, though not a decisive one, because `pre_wo` patches individual heads while `post_wo` patches 128-dim residual blocks. The write-up should say so.
- Every one of the 100 flips goes in one direction. Random arithmetic noise would flip both ways, so this suggests a systematic push. That fits the position-0 hypothesis, and it is a usable test statistic (see fix 5).

---

## Required fixes

### 1. Gate on a nonzero A100 d0, and make gate failures block job 2
**Where:** `run_experiment.py:100-111`, `:146`, `:151-152`; `run_a100.sbatch:14`; pre-registration lines 50-53.

**Zero d0 on A100.** If this A100 node gives d0 == 0, then `keep0` is bitwise identical to `copy`. The run then reports r = 0, which reads as "does not explain the gap". That would be a wrong verdict.
- Record the result as "premise absent on this node" and stop. Exit nonzero.
- The pre-registration says the d0 check happens "on L40S", but the code does not gate on it on **either** GPU. It only records d0.

**Nonzero d0 on L40S.** If the new L40S node gives d0 ≠ 0, the reverse test's premise ("L40S copy writes a zero difference") is false. Flag it as a gate failure.

**Exit codes.** The script never exits nonzero, so `afterok` cannot do its job.
- A failed A100 copy gate, or a zero delta, still lets job 2 run with a bad or zero `position0.pt`.
- With a zero delta, `inject` == `copy` and r_rev = 0: another false "does not explain".
- Fix: write all results, then `sys.exit(2)` when `gate.ok` is false in any cell or d0 == 0.

**Job 2 should check job 1's outputs.** The L40S job should read the A100 `run.json` next to `--delta`. It should refuse to run unless both A100 gates passed and `pos0_n_nonzero > 0`. It should also record `sha256(position0.pt)` in its provenance.

### 2. "Captured buggy position 0 == live position 0" is asserted, not shown
**Where:** pre-registration lines 38-39; `run_experiment.py:98-111`.

R13's identity gate compares **decoded text** after 20 greedy tokens. It cannot detect a one-ULP difference; the R13 verification itself found that writing all of BOS into every decode block does not change the text. So the claim "R13's identity gate already shows that capture and generation are bitwise consistent" is false as written.

`capture` calls `model(**ids)`. `generate` runs its prefill through the generation path, which has its own `cache_position` and mask construction and keeps only the last logit. Both probably dispatch the same SDPA and GEMM kernels at length 19, but that is an assumption. If they differ on A100, `keep0` writes a different one-ULP perturbation at position 0 instead of a no-op. The test would then compare two perturbations, not perturbation versus none.

**Cheap check (one generation):**
1. Register a temporary forward hook on `p.attn` that stores the first call's `out[0]`.
2. Run unpatched `p.generate(r13.BUGGY_PROMPT)`.
3. Record `torch.equal(live[:, 0], buggy[:, 0])` and `torch.equal(live, buggy)`.
4. Make the first a gate for `keep0`. Do the same for the clean prompt as a consistency check.

Fix the pre-registration sentence too.

### 3. Stratify by which blocks the d0 difference touches
**Where:** `run_experiment.py:100-104`, `decide` at `:48-64`; pre-registration decision rules.

In `post_wo`, a "head" is a 128-dim block of residual coordinates. The one-ULP difference at position 0 sits at specific coordinates (`pos0_n_nonzero` may well be 1). A config copies **any** position-0 difference only if its block set includes a block where d0 ≠ 0. That gives a sharp prediction at no extra cost:

- **Prefill.** For every config with no nonzero block, `keep0` must produce **bitwise the same** output as `copy`. This doubles as an implementation check. If it fails, something other than position 0 changed.
- **Explaining D.** A D config that contains no nonzero block cannot be explained by position 0 in prefill. For example, if d0 is nonzero only in odd blocks, the 37 even-only R8 configs in prefill D are ruled out before anything runs.
- **`all_steps` differs.** The decode writes also go only into the config's blocks, so the same exposure logic applies.

Changes:
- Record which blocks have d0 ≠ 0, as `(d0.view(-1, n_heads, head_dim) != 0).any(-1)`, on both GPUs.
- In `decide`, report r and N_new separately for exposed and unexposed configs.
- Pre-register: unexposed configs give text-identical `copy` and `keep0` (a gate), and the verdict is computed on exposed D. If no D config is exposed, the hypothesis is refuted by construction; say so.

### 4. The `all_steps` R8 reproduction number is not computed
**Where:** `run_experiment.py:61`, `:88`; pre-registration lines 66-67.

The pre-registration promises the R8 reproduction agreement for `all_steps` (A100 1,000, L40S 942). `decide` instead reports `r8_correct_*`, the **count of correct** even7/even8 configs (A100 499 and L40S 441 in `all_steps`). That is a different statistic.
- `load_refs` already returns R8, but the script discards it: `_, r1 = r13.load_refs()`.
- Fix: keep `r8`. For each variant in `all_steps`, compute `r13.reproduction(cfgs, r13.reference_outcomes(r8, r1))` and `r13.reproduction_gate(...)["values"]`.

### 5. The decision rule needs a fragility baseline (placebo)
**Where:** pre-registration lines 61-70; `decide`.

D is selected **because** its configs sit on the decision boundary. Any perturbation, including one unrelated to position 0, will flip a good share of them. Because all of D is "correct" on A100, any flip moves a config to the L40S class and counts toward r. So r has a nonzero null rate, and the 0.2–0.8 "partial" band may be exactly where noise lands.

N_new ≤ 4 offers some protection, but only in the "explains" branch. The "does not explain" branch (r ≤ 0.2) ignores N_new: a large N_new with a small r is "the perturbation does something, just not this", which is a different claim.

Add one placebo cell per scope on A100 (about 10 min each):
- donor position 0 := `buggy0`, plus a one-ULP perturbation with the same number of nonzero entries and the same signs as d0,
- placed in **other** coordinates, inside blocks that the D configs use.

Then:
- **Required:** report the direction of flips outside D in both directions. Under the position-0 hypothesis they should be one-directional; under the noise null they split both ways.
- **Optional:** require r_keep0 − r_placebo ≥ 0.5, alongside N_new ≤ 4, for "explains".
- **Wording:** give "r ≤ 0.2 with N_new > 4" a separate label, such as "position 0 perturbs, does not explain".

### 6. The limitation paragraph misdescribes what `keep0` removes
**Where:** pre-registration lines 78-79.

**For the receiving run, the paragraph is right.** In the patched buggy run, the only clean-run content is the layer-10 `post_wo` donor. Other layers' position-0 values are the buggy run's own, identical in `copy` and `keep0`. So "only layer 10 is changed" loses nothing there, and an "all layers via the residual stream" variant on the **buggy** side would be a no-op.

**It misses a second channel: the donor's positions 1–16.** In the clean capture, positions 1–16 at layers 0–10 attend to the clean run's BOS keys and values. On A100 those are ULP-different from the buggy run's (the same length-17 vs length-19 kernel effect). `keep0` leaves this channel open. So `keep0` can fail to remove the gap even if "position 0 on A100" is the cause.

The well-posed claim for `keep0` is narrower: "the layer-10 position-0 write explains the gap." State that. Then either:
- **(a)** add a `keep0_all` cell (recommended), or
- **(b)** restrict the conclusion to the layer-10 write.

**`keep0_all`:** capture the clean donor with position 0 forced to the buggy run's values at every layer 0–10, then apply `keep0`.
- It is a no-op in exact arithmetic.
- Residual-only replacement is **not** bit-exact, because the K/V projection of the forced residual at length 17 may differ by a ULP from length 19. Replace position 0 of the `k_proj`/`v_proj` outputs at each layer. RoPE at position 0 is exact (cos = 1, sin = 0).
- Also replace position 0 of each decoder-layer output.
- Cost: one capture plus two cells, about 20 min.

### 7. Inject: verify the fp16 cast and save the L40S position-0 vectors
**Where:** `run_experiment.py:113-119`.

`buggy0_L + delta_A` is the right reverse test **for the perturbation hypothesis**. A100 `copy` writes `clean0_A` where the live value is `buggy0_A`, a relative perturbation of `delta_A`. L40S `copy` writes a relative perturbation of 0. Writing `clean0_A` directly tests something else: it adds `buggy0_A − buggy0_L`, the cross-GPU difference, which is "arithmetic elsewhere". `clean0_L + delta` equals `buggy0_L + delta` whenever d0_L == 0 (fix 1 enforces that).

The fp32 → fp16 cast reproduces `delta` exactly only where `buggy0_L` is in the same binade as `buggy0_A`. Elsewhere a half-ULP can round away, or one ULP can double. The current `inject_pos0_max_abs_change_vs_buggy0` would not catch this.

Record:
- `torch.equal(inj0.float() - buggy0_L.float(), delta)`,
- the nonzero count of `(inj0 != buggy0_L)` versus `pos0_n_nonzero`,
- `torch.equal(buggy0_L, buggy0_A)` and `torch.equal(clean0_L, clean0_A)`. Save `clean0` and `buggy0` on L40S too.

If `buggy0_L == buggy0_A` bitwise, then `inject` == `clean0_A`, the two candidate tests coincide, and the result is clean. If not, also run an `a100clean0` variant (about 20 min). It covers the `all_steps` decode writes, where A100 writes `clean0_A` as an **absolute** value into new tokens rather than as a relative perturbation.

### 8. Copy gate: pin the node, and pre-register a fallback if it fails
**Where:** both sbatch files; pre-registration lines 48-50.

Bitwise reproduction on a different node of the same SKU and driver is likely, but not guaranteed. cuBLAS heuristics can change with GPU model or driver, and `a100_tandon` may contain more than one A100 variant.
- Add `#SBATCH --nodelist=ga004` and `--nodelist=gl044`, or at least `--constraint` for the exact GPU model.
- Check `prov["gpu"]` against R13's `run.json` and record the result.

As written, a gate failure throws away the whole GPU's data, yet the within-node comparison of `keep0` against `copy` stays valid.
- Pre-register a fallback, labeled exploratory: D' = configs whose class from **this run's** `copy` differs from the other GPU's R13 class.
- Compute N_new against this run's `copy`; the code already does this, but D comes from R13's files.

---

## Answers to the seven questions

1. **`keep0` semantics.** In `post_wo/prefill`, `_swap` (R13 `:148-149`) writes `donor[:, :17]` into the live `[:, :17]` on the first call only, so `keep0` makes position 0 a self-write. That is "don't patch position 0" **if** the captured value equals the live one (Q2). In `all_steps`, decode calls have `sl = 1`, so `ml = 1` and donor position 0 is written into every new token's patched blocks (bug B). `keep0` writes `buggy0` there. That is not "no patch". It is the exact-arithmetic-equivalent value (clean0 ≡ buggy0 in exact arithmetic), so it is the correct counterpart: bug B stays and only the ULP goes. The pre-registration wording is right. Note that R13's `decode_only` (whole donor = buggy) differs between the GPUs on only 1 text.
2. **Bitwise equality.** It is likely but unproven. See fix 2 for a one-generation in-job check.
3. **Inject.** `buggy0_L + delta` is right for the relative-perturbation hypothesis, provided the cast is verified (fix 7). `clean0_A` tests the absolute cross-GPU value. Run it as a second variant only if `buggy0_L ≠ buggy0_A`.
4. **Copy gate.** The gate is sensible, but pin the node and pre-register a fallback (fix 8). The gate is also evaluated only after a full 1,307-config cell. That is acceptable, because `copy` has to run anyway.
5. **Decision rules.**
   - The D computation is correct; I reproduced 40 and 60.
   - Thresholds need a placebo baseline and a label for large N_new (fix 5).
   - The R8 reproduction statistic is missing (fix 4).
   - The block-exposure split is needed (fix 3).
6. **Layer 10 only.** For the receiving run, only layer 10 imports clean content. The donor's positions 1–16 carry the clean run's BOS keys and values, which is a real channel `keep0` misses. The decisive extra cell is `keep0_all` on the **donor** capture, at the K/V level (fix 6). The free test from block exposure (fix 3) may settle much of the question before any run.
7. **Data safety, chaining, walltime.**
   - Good: `makedirs(exist_ok=False)`, atomic `dump`, per-variant files, and the R13 hash check.
   - Problem: the missing nonzero exit codes break `afterok` (fix 1).
   - If the A100 job times out or OOMs, the L40S job sits pending with `DependencyNeverSatisfied`; cancel it by hand.
   - The delta path (`r16_a100_$A/position0.pt`) matches the A100 `--out`, which is correct.
   - Walltime: 4 cells × ~620 s ≈ 42 min plus load, so 2 h is fine as written. With the cells added in fixes 5–7, set `--time=03:00:00`.

## Optional suggestions

- Report text-level metrics as well as class:
  - `keep0` text equal to `copy`, per config, to see how far the change reaches;
  - `keep0` text equal to the **L40S R13 text**, across all 1,307 configs.
  
  If position 0 were the whole story, many A100 `keep0` texts would match L40S's exactly, which is stronger than a class match.
- Cross-GPU donor transplant on L40S: a `donorA100` variant that writes the A100's **full** clean donor, saved in `position0.pt` (17×4096 fp16, tiny).
  - If L40S plus the A100 donor reproduces the A100 classes on D, the gap is in the donor (layers 0–10 of the clean run), and position 0 is the candidate within it.
  - If not, the gap is in the receiving run's arithmetic, and no donor-side position-0 fix can explain it fully.
  
  This splits the gap into donor and receiver parts for about 20 min.
- Record d0 (max, nonzero count, nonzero blocks) for the residual stream and K/V at every layer 0–10, on both GPUs. It is one forward pass with `output_hidden_states`.
- Write the R13 reference `results_*.json` sha256 values into the provenance, and check that `R16_GIT_SHA` matches `git rev-parse HEAD` with a clean tree.
- The CPU smoke run exercised `inject` only with a zero delta. Rerun it with a synthetic nonzero delta, and check that `inject` changes the output while unexposed configs stay identical.
- Put the R13 `pre_wo` gaps (2 and 1 configs, despite the same 6.1e-5 at position 0) in the pre-registration as context. A reader will ask why `post_wo` is so much more sensitive.

Files reviewed:
- `/Users/gus/github/MATS/docs/experiments/11-22_llama8b_position0_gpu_gap.md`
- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R16_position0/run_experiment.py`
- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R16_position0/run_a100.sbatch`
- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R16_position0/run_l40s.sbatch`
- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R16_position0/smoke_cpu.log`
- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R13_true_head_patching/run_experiment.py`
- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R13_true_head_patching/results/r13_24014365/`
- `/Users/gus/github/MATS/9_8_research/reviewer_experiments/exp_R13_true_head_patching/results/r13_24084222/`
- `/Users/gus/github/MATS/docs/reviews/11-21_llama8b_decimal_true_head_patching_verify.md`
- `/Users/gus/github/MATS/docs/reviews/11-21_llama8b_decimal_true_head_patching_verify_a100.md`
