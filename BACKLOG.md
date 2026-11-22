# BACKLOG

Claude's queue. Top = next. Add, reorder, and close freely.

- [ ] R13 follow-ups deferred from design review v3 (optional 2–7): assert/record 17/19 token lengths
      and the right-align map; save run.json after isolation/BOS checks and on cell abort; first-token
      id/logit diff for equivalence; document --smoke/--limit gating; BOS-diff tolerance
- [x] v2 cross-model fixes (head count, 0/50 artifact, Gemma base/IT, method paragraph): done in rewrite
- [x] Rewrite v2 head sections as four-model table (2026-11-22; abstract, intro, §2–4, discussion,
      limitations, conclusion, appendix, title subtitle). Awaiting Gus's read.
- [ ] Re-review the legacy results v2 still cites (each tagged \todo in the draft), checking for a hook
      left on during generate: negative-result map (diff-of-means 17 conditions, SAE swap, concept-neuron
      ablation incl. Pythia −65pp, steering vectors, full-layer transplant); paraphrase sweep (R2) and
      70 decimal pairs (R4); GPT-2 logit evaluation (R12); appendix SAE/neuron/ACDC sections
- [ ] v2 draft fixes found in R13 design review: `02_study.tex` l.58 says hooks use nnsight (scripts
      use HF hooks); l.152 says "+70.0pp on every pair" (R1: 21/30 pairs at +100pp, 9 at 0)
- [ ] Build docs/reports/ (index, RQ page, R13 evidence page) and get it fact-checked
- [ ] Rebuild arXiv tarball and draft the reply to the reader (Even Heads part)
