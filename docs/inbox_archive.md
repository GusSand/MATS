# INBOX archive

Answered INBOX items, moved here verbatim once applied. Append-only.

2. **Waive the R13 reproduction gate?** The old-protocol cell agreed with R8 on 942 of 1,000 subsets;
   the pre-registered threshold was 950. R1 checks passed (199/200, 59/60). All 60 misses went one way
   (correct in March, bug now), and the lost subsets are the weakest repairs (0 of 58 correct without
   decode-step writes, vs 83 of 441 for the rest). R8's code is the same protocol; the hardware differs.
   **Recommendation:** waive, and say so in the paper. The verdicts hang on all-heads and even-16 in
   the corrected cells, which are 0 in both alignments, far from any borderline subset.
   Trade-off: the alternative is rerunning the old protocol on other GPUs (~15 GPU-min each) hoping to
   hit 950, but March's hardware is unrecorded, so a miss there settles nothing.
   → answer (2026-11-22): waive.
