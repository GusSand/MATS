# INBOX — decisions for MATS's researcher. Append "→ answer:" inline.
# About one screen. Answered items move to docs/inbox_archive.md. Job news goes to the journal; Claude's tasks go to BACKLOG.md.

**Goal.** Which model components cause format-dependent failures such as 9.8 vs 9.11, and which interventions actually repair them?

**Where we stand (verified results only).**
- R13: patching true heads on the prompt only does not repair the answer at Layer 10, not even with
  all 32 heads. 0 of 1,307 head configurations repair it under answer-aligned patching. The paper's
  head-level claims (H\*, parity, 7→8 threshold) came from post-W_O block patching plus decode-step writes.

**Next milestone.** Cross-model head scripts audited; v2 Llama head section rewritten from R13.

## Decisions for you

1. **Retro-review the legacy experiment files?** Installing the review and journal gates flagged
   34 older experiment files (Feb–May 2026, mostly the secure-code steering work) with results but no
   design review, 13 of them with no journal entry. I grandfathered them in
   `docs/review_allowlist.txt` / `docs/journal_allowlist.txt` so R13 could be committed.
   **Recommendation:** leave them grandfathered; that paper already had its own Aug 25 audit. Re-review
   only files whose numbers the v2 Even Heads draft cites. Trade-off: full retro-review costs ~34
   subagent passes and mostly re-checks the other paper.

2. **What to do with the non-Llama head claims?** Audit (docs/reviews/11-22_cross_model_head_scripts_audit.md):
   every Pythia and Gemma head/parity claim used the same post-W_O hook left on during generation (both bugs).
   The Mistral head claims (H22, even-only at ℓ=2,17) have no code or results anywhere; they came from Nam by email.
   **Recommendation:** rerun Pythia-160M and Gemma-2B (IT) with the R13 protocol (small models, under 1 GPU-h
   each, with design review first), and drop the Mistral head claims unless Nam's code turns up.
   Trade-off: dropping Pythia/Gemma instead is faster but leaves Llama as the only model in the head section.

