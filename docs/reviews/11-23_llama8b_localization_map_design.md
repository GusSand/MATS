Reviewer: fresh subagent (Claude Code)
Agent-ID: a70a34e4090bdcfb2
Verdict: FIXES NEEDED
Transcript-SHA256: 131f441ef6586cc61b3b7a7f98e7d7434c44d92e6a5723990aff68f4b905e1ad

## Brief given

You are a fresh, skeptical design reviewer for a pre-registered mechanistic-interpretability experiment (R17) in ~/github/MATS. Read-only: do NOT edit files, do NOT ssh or run GPU jobs. You may run small local Python checks.

Read:
- Pre-registration: docs/experiments/11-23_llama8b_localization_map.md
- Script: 9_8_research/reviewer_experiments/exp_R17_localization_map/run_experiment.py, run.sbatch, smoke_cpu.log
- Context: R13 (docs/experiments/11-21_llama8b_decimal_true_head_patching.md and its script 9_8_research/reviewer_experiments/exp_R13_true_head_patching/run_experiment.py). R13 found that earlier head-patching claims about Llama-3.1-8B's 9.8-vs-9.11 format bug came from two code bugs: hooking post-W_O attention output and reshaping it into "heads", and leaving the hook active during generate() so decode steps got overwritten. Under corrected prompt-only patching, no head subset at layer 10, including all 32 heads, repairs the bug. R17 is stage 1 of a search for what DOES repair it: a full (component × layer × position) activation-patching map with a teacher-forced logit-difference metric, confirmed by generation.

Review for:
1. Code correctness: are the hooks capturing/patching the intended tensors in transformers 5.x Llama (decoder-layer pre-hook with kwargs for resid, self_attn output tuple, mlp output)? Is alignment right? Is the prefill-only generation patch correct and is the hook truly inactive at decode steps? Any way a gate can pass vacuously? Any re-introduction of the R13 bugs?
2. Metric validity: is Δ = logit("8") − logit("11") after teacher-forcing " 9." a sound proxy? Risks (e.g., the model's real continuation differs; tokens like "80"; the prefix being off-distribution for the donor)?
3. Decision rules: are they unambiguous, implemented as written in decide(), and do they answer the research question? Thresholds sensible? Anything that would let a trivial/uninteresting effect be labelled as a finding (e.g., late-layer resid at the last position, or answer-position patches)? Are positive controls adequate?
4. Anything missing that a referee would demand for stage 1 (e.g., noise/baseline variation, a control patch from a non-informative donor, reporting needs).

Output: first line exactly one of PASS, PASS WITH MINOR FIXES, FIXES NEEDED. Then numbered findings, each with file:line, severity (blocking / should-fix / minor), and a concrete fix. Be concise.

## Report

FIXES NEEDED

R17 does not reintroduce either R13 bug. Two gaps still need fixing before the run: a silent no-op patch would read as a null result, and the "localized" label can fire on what is really an input-token edit. Paths below are relative to `~/github/MATS`. "Script" is `9_8_research/reviewer_experiments/exp_R17_localization_map/run_experiment.py`; "prereg" is `docs/experiments/11-23_llama8b_localization_map.md`.

**What I checked and found sound**
- **Hooks.** The pre-hook on the decoder layer handles hidden states passed either positionally or as a keyword. `self_attn` is read as `out[0]` of its tuple, and `mlp` as a plain tensor (script:94-116).
- **The decomposition gate checks the captures.** resid[L+1] = resid[L] + attn[L] + mlp[L] tests that the three captured tensors are the right ones. It passed in the smoke run.
- **The attention patch is legitimate.** It patches the whole `o_proj` output as one component, with no reshaping into heads, so R13 bug 1 is not repeated.
- **Generation patches prefill only.** It asserts the first call has length T and every later call has length 1, then passes decode steps through untouched (script:161-179). So R13 bug 2 is not repeated.
- **Prompt patches are applied the same way in the metric and in generation.** In both, the answer positions are computed fresh against the patched cache.
- **`decide()` matches the prereg as written**, the `a2` naming lines up with `position_sets`, and R13's sha256 matches `1ce5b0cc…`.

**Findings**

1. **A no-op patch could pass every gate (script:321-327, 360). Severity: blocking.**
   - The identity gate (writing the recipient's own values back) passes bitwise even if the patch is silently ignored. The hooks-removed gate cannot catch that either.
   - If, say, the edited hidden-state keyword were ignored on the cluster's transformers version, every r would be 0. The run would then report "no prompt-only patch repairs", the same vacuous null that R13 uncovered.
   - Fix: add a gate requiring each component to have at least one sweep cell with |Δ_p − Δ_b| > 1e-3.
   - Also add an active-patch check: re-capture inside a patched run and assert that the tensor reaching the hook equals the donor value at the patched positions.
   - Add a generation-path identity check too: a prefill-only patch of the recipient's own values must reproduce the unpatched tokens exactly.

2. **A token substitution can earn the headline label (script:199-209; prereg:65-67, 86-90). Severity: blocking.**
   - Donor positions 1 and 15 hold different tokens ("Which" vs " Which", "Answer" vs "A"). A resid patch there, especially at layer 0, mostly swaps one token's identity for another. It is not localizing a mechanism.
   - A cell such as `resid L0 p15` would still earn "localized: single-position prompt patches repair".
   - Likewise `resid prompt_all` at early layers is close to editing the input. On its own it would earn "distributed".
   - Fix: pre-register that resid cells at positions {1, 15}, and all resid cells at L=0, count as "input substitution". They should be reported apart and kept out of rules 1 and 2.
   - For `resid prompt_all`, report the latest layer that still repairs, i.e. where the difference stops being only in the prompt's residual stream. That number answers the research question better than the label does.

3. **The phenomenon gate does not check the token the metric depends on (script:298-300). Severity: should-fix.**
   - It checks the first three generated tokens but not the fourth. Δ assumes the next token is "8" (23) for the donor and "11" (806) for the recipient.
   - Fix: require `g_d[0][3]==23` and `g_r[0][3]==806`.
   - In the teacher-forced runs, also require the argmax at the last position to be in {23, 806} for both prompts, and record the top 5. Otherwise Δ could be comparing two tokens that the model ranks low.

4. **r ≥ 0.5 does not line up with the answer actually flipping (script:354; prereg:45-49). Severity: should-fix.**
   - The argmax flips when Δ_p > 0, which happens at r > −Δ_b/(Δ_c − Δ_b), not at r = 0.5.
   - If Δ_b ≈ −0.3 and Δ_c ≈ +5, a cell with r = 0.1 already flips the answer and is never checked by generation. If Δ_b ≈ −5 and Δ_c ≈ +0.5, r = 0.5 can never flip it.
   - Fix: run the generation check on prompt cells with r ≥ 0.5 OR Δ_p > 0. Report r_flip = −Δ_b/gap. Keep the repair label as written, but also list cells that flip the answer without reaching r ≥ 0.5.

5. **No control for off-distribution disruption (prereg:63-80). Severity: should-fix; a referee will ask.**
   - Moving an activation from position i to i+2, or into a different context, can shift Δ for reasons unrelated to the format.
   - Fix: for every cell that reaches generation, run two controls and require r_control < 0.2 before calling it a repair:
     - a misaligned donor (donor i → recipient i+1 or i+3);
     - a random-direction replacement with the same norm as the donor patch.
   - Also run the reverse direction (buggy → clean, a necessity test) on the repairing cells, or state that stage 1 tests sufficiency only.

6. **No ungated positive control (script:330). Severity: should-fix.**
   - `resid` at layer 31 on `answer_all`, or on `a2`, should give r close to 1 by construction. Gate on r ≥ 0.8 there.
   - Current gates prove the hooks are harmless, not that they work.

7. **The decomposition tolerance is loose (script:315-320). Severity: minor.**
   - The error is scaled by max|resid|, and in Llama that maximum is dominated by the very large activation at BOS. A 1% tolerance is therefore large next to ordinary positions.
   - Fix: compute the error per position over non-BOS positions only, with a tolerance around 1e-2 of each position's norm. Also include L = n−1 by checking against the final-norm input.

8. **The two donor captures are never compared to a limit (script:311-313). Severity: minor.**
   - The difference between the capture with the answer prefix and the one without it is recorded but not thresholded. In fp16 the attention kernels can differ by sequence length.
   - Fix: assert maxdiff < 1e-2 × activation scale, or capture the donor with prefill-only patches and read positions [:17].

9. **Single-position and answer-position results need care in the report (prereg:73-80). Severity: minor.**
   - Late-layer `resid` at the last prompt position (p16, ":") and any answer-position cell should be presented as near-trivial or decode-time effects. The prereg already says this for answer positions.
   - Also note that donor position 1 sits right after BOS, where Llama tends to have unusual activations. A `p1` effect needs the controls from finding 5.

10. **The selftest leaves branches untested (script:229-240). Severity: minor.**
    - Add cases for the answer-position outputs (`a2` resid first layer, attn answer layers).
    - Add a case where only a non-attn component repairs, so stage 2 falls back to the r ≥ 0.2 rule.
    - Add tests for the exclusions from finding 2 once they exist.

11. **The classifier's fallback is lenient (R13 `run_experiment.py:66-69`). Severity: minor.**
    - When no pattern matches, a first line that mentions "9.8" but not "9.11" counts as correct. A broken generation from a patch could pass that way.
    - Fix: store the full generated text for every repairing cell (it already does) and hand-check them in the report. Flag repairs that only passed through the fallback rule.

12. **Scope: one prompt pair (prereg:84). Severity: minor.**
    - This is acceptable for stage 1 as pre-registered.
    - Before stage 2 rests on any single-position site, re-run the repairing cells on 3–5 other number pairs with the same token alignment (e.g. 8.7/8.12, 7.9/7.10). A site that holds for only one pair is not worth a head-level search.
