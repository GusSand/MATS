# BACKLOG

Claude's queue. Top = next. Add, reorder, and close freely.

- [ ] R13 true per-head patching (docs/experiments/11-21_llama8b_decimal_true_head_patching.md):
      design review → fixes → commit → rsync to Torch → submit → verify → journal
- [ ] R13 follow-ups deferred from design review v3 (optional 2–7): assert/record 17/19 token lengths
      and the right-align map; save run.json after isolation/BOS checks and on cell abort; first-token
      id/logit diff for equivalence; document --smoke/--limit gating; BOS-diff tolerance
- [ ] Audit cross-model head scripts (`9_8_research/cross_model_validation/test_*.py`) for the
      same post-W_O slicing and decode-step bugs (Mistral, Pythia, Gemma claims in the v2 draft)
- [ ] Rewrite the Llama head section of the v2 draft
      (`~/github/phd_thesis/preamble/surgeon_dilemma/sections/02_study.tex`) from verified R13 results
- [ ] v2 draft fixes found in R13 design review: `02_study.tex` l.58 says hooks use nnsight (scripts
      use HF hooks); l.152 says "+70.0pp on every pair" (R1: 21/30 pairs at +100pp, 9 at 0)
- [ ] Rebuild arXiv tarball and draft the reply to the reader (Even Heads part)
