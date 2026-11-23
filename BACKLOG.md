# BACKLOG

Claude's queue. Top = next. Add, reorder, and close freely.

- [ ] R17 (stage 1): design review → commit → submit → verify. docs/experiments/11-23_llama8b_localization_map.md
- [ ] Stage 2: head-level search (true heads, attribution patching then exact patches) at the layers R17 triggers
- [ ] Stage 3: Mistral layer-7 head subsets + [/INST]-tail alignment control

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
- [x] v2 nnsight / +70pp fixes: both passages removed in the rewrite
- [ ] Build docs/reports/ (index, RQ page, R13 evidence page) and get it fact-checked
- [ ] Rebuild arXiv tarball and draft the reply to the reader (Even Heads part)
