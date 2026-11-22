# INBOX — decisions for MATS's researcher. Append "→ answer:" inline.
# About one screen. Answered items move to docs/inbox_archive.md. Job news goes to the journal; Claude's tasks go to BACKLOG.md.

**Goal.** Which model components cause format-dependent failures such as 9.8 vs 9.11, and which interventions actually repair them?

**Where we stand (verified results only).**
- R13 (Llama): patching true heads on the prompt only does not repair the answer at layer 10, not even
  with all 32 heads (0 of 1,307 configurations). The paper's head-level claims came from post-W_O block
  patching plus decode-step writes.
- R15 (Mistral): only layer 7 repairs. 0 of 1,024 single heads and 0 of 524 subsets repair; H22 is not
  special. The five-layer repair is not specific (7 of 10 random layer sets also repair).
- R16: the L40S vs A100 gap in R13 (120 vs 83) is not caused by position 0 (keep0 gives 119).
  It is fp16 arithmetic in the receiving run. No R13 verdict changes.

**Next milestone.** Cross-model head scripts audited; v2 Llama head section rewritten from R13.

## Decisions for you

1. **Retro-review the legacy experiment files?** Installing the review and journal gates flagged
   34 older experiment files (Feb–May 2026, mostly the secure-code steering work) with results but no
   design review, 13 of them with no journal entry. I grandfathered them in
   `docs/review_allowlist.txt` / `docs/journal_allowlist.txt` so R13 could be committed.
   **Recommendation:** leave them grandfathered; that paper already had its own Aug 25 audit. Re-review
   only files whose numbers the v2 Even Heads draft cites. Trade-off: full retro-review costs ~34
   subagent passes and mostly re-checks the other paper.

2. **R14: what to do with Pythia, whose reproduction gate failed?** On the A100 the old protocol no longer
   reproduces September's Pythia "all even heads repair" (7/10 references; `all_even` now answers 9.11, and the
   outputs sit on a one-token edge). The training-dynamics replay misses at exactly the two late checkpoints.
   The corrected run finds no layer-6 repair, and the only repairing layer is triggered by ambiguous
   listing outputs. Gemma (base and IT) shows no format-dependent bug at all.
   **Recommendation:** don't waive. Drop all Pythia and Gemma head-level claims from the v2 draft, and state
   that (i) the September Pythia repair does not reproduce under its own protocol on current software and
   (ii) Gemma has no format-dependent bug on our formats. These statements don't need the gate.
   Trade-off: waiving would let us cite the layer-0 "odd > even" numbers, but they rest on ambiguous outputs
   and add nothing the paper needs.

