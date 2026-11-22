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

**Next milestone.** v2 head section rewritten as a four-model table (Llama, Mistral, Pythia, Gemma) from verified R13–R15 results.

## Decisions for you

(none open)
