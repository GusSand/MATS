# BACKLOG

Claude's queue. Top = next. Add, reorder, and close freely.

- [ ] R13 follow-ups deferred from design review v3 (optional 2–7): assert/record 17/19 token lengths
      and the right-align map; save run.json after isolation/BOS checks and on cell abort; first-token
      id/logit diff for equivalence; document --smoke/--limit gating; BOS-diff tolerance
- [ ] v2 draft, cross-model fixes regardless of reruns (audit docs/reviews/11-22_cross_model_head_scripts_audit.md):
      Pythia "all 16 even heads" (Pythia-160M has 12); "0% on 50 decimal pairs" is a scoring artifact (no
      prompt contains 9.8); Gemma head result is base gemma-2b while the 5% bug rate is the IT model;
      method paragraph 02_study.tex:53-58 describes the post-W_O reshape
- [ ] Rewrite the Llama head section of the v2 draft
      (`~/github/phd_thesis/preamble/surgeon_dilemma/sections/02_study.tex`) from verified R13 results
- [ ] v2 draft fixes found in R13 design review: `02_study.tex` l.58 says hooks use nnsight (scripts
      use HF hooks); l.152 says "+70.0pp on every pair" (R1: 21/30 pairs at +100pp, 9 at 0)
- [ ] Build docs/reports/ (index, RQ page, R13 evidence page) and get it fact-checked
- [ ] Rebuild arXiv tarball and draft the reply to the reader (Even Heads part)
