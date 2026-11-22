# INBOX — decisions for MATS's researcher. Append "→ answer:" inline.
# About one screen. Answered items move to docs/inbox_archive.md. Job news goes to the journal; Claude's tasks go to BACKLOG.md.

**Goal.** Which model components cause format-dependent failures such as 9.8 vs 9.11, and which interventions actually repair them?

**Where we stand (verified results only).**
- Nothing verified yet. Every head-subset result in the Even Heads paper and its v2 draft patched
  128-dim slices of the post-W_O output rather than heads, and kept patching during generation
  (code-reading finding, confirmed by three design reviews; not yet a measured result).

**Next milestone.** Decision 2, then R13 verdicts: does the Layer-10 repair, and H\*, survive true per-head,
prompt-only patching? (docs/experiments/11-21_llama8b_decimal_true_head_patching.md)

## Decisions for you

1. **Retro-review the legacy experiment files?** Installing the review and journal gates flagged
   34 older experiment files (Feb–May 2026, mostly the secure-code steering work) with results but no
   design review, 13 of them with no journal entry. I grandfathered them in
   `docs/review_allowlist.txt` / `docs/journal_allowlist.txt` so R13 could be committed.
   **Recommendation:** leave them grandfathered; that paper already had its own Aug 25 audit. Re-review
   only files whose numbers the v2 Even Heads draft cites. Trade-off: full retro-review costs ~34
   subagent passes and mostly re-checks the other paper.

2. **Waive the R13 reproduction gate?** The old-protocol cell agreed with R8 on 942 of 1,000 subsets;
   the pre-registered threshold was 950. R1 checks passed (199/200, 59/60). All 60 misses went one way
   (correct in March, bug now), and the lost subsets are the weakest repairs (0 of 58 correct without
   decode-step writes, vs 83 of 441 for the rest). R8's code is the same protocol; the hardware differs.
   **Recommendation:** waive, and say so in the paper. The verdicts hang on all-heads and even-16 in
   the corrected cells, which are 0 in both alignments, far from any borderline subset.
   Trade-off: the alternative is rerunning the old protocol on other GPUs (~15 GPU-min each) hoping to
   hit 950, but March's hardware is unrecorded, so a miss there settles nothing.
