# BACKLOG

Claude's queue. Top = next. Add, reorder, and close freely.

- [ ] R13 follow-ups deferred from design review v3 (optional 2–7): assert/record 17/19 token lengths
      and the right-align map; save run.json after isolation/BOS checks and on cell abort; first-token
      id/logit diff for equivalence; document --smoke/--limit gating; BOS-diff tolerance
- [ ] v2 draft, cross-model fixes regardless of reruns (audit docs/reviews/11-22_cross_model_head_scripts_audit.md):
      Pythia "all 16 even heads" (Pythia-160M has 12); "0% on 50 decimal pairs" is a scoring artifact (no
      prompt contains 9.8); Gemma head result is base gemma-2b while the 5% bug rate is the IT model;
      method paragraph 02_study.tex:53-58 describes the post-W_O reshape
- [ ] Rewrite the v2 head section (`~/github/phd_thesis/preamble/surgeon_dilemma/sections/02_study.tex`) as a
      four-model cross-model table, one row per model (Gus, 2026-11-22):
      Llama: layer-10 prompt-only true-head patching repairs 0 of 1,307 configs, incl. all 32 heads (R13).
      Mistral: whole-layer patch at layer 7 repairs; 0 of 1,024 single heads and 0 of 524 subsets (R15);
        caveats: our prompts not Nam's, layer-7 repair may ride on the [/INST]-tail alignment, 5-layer repair non-specific.
      Pythia: Sept all-even repair does not reproduce under original protocol on current software (all_even → 9.11);
        replay disagrees at both late checkpoints → withdraw late-emergence claim; cause of difference open (R14).
      Gemma: neither base nor IT shows a correct-vs-error format pair on the tested formats → head test not possible (R14).
      Drop all Pythia/Gemma head-level and parity claims.
- [ ] Before citing any legacy (grandfathered) experiment's numbers in v2, re-review that file
- [ ] v2 draft fixes found in R13 design review: `02_study.tex` l.58 says hooks use nnsight (scripts
      use HF hooks); l.152 says "+70.0pp on every pair" (R1: 21/30 pairs at +100pp, 9 at 0)
- [ ] Build docs/reports/ (index, RQ page, R13 evidence page) and get it fact-checked
- [ ] Rebuild arXiv tarball and draft the reply to the reader (Even Heads part)
