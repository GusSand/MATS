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

2. **What to do with the non-Llama head claims?** Audit (docs/reviews/11-22_cross_model_head_scripts_audit.md):
   every Pythia and Gemma head/parity claim used the same post-W_O hook left on during generation (both bugs).
   The Mistral head claims (H22, even-only at ℓ=2,17) have no code or results anywhere; they came from Nam by email.
   **Recommendation:** rerun Pythia-160M and Gemma-2B (IT) with the R13 protocol (small models, under 1 GPU-h
   each, with design review first), and drop the Mistral head claims unless Nam's code turns up.
   Trade-off: dropping Pythia/Gemma instead is faster but leaves Llama as the only model in the head section.
   → answer (2026-11-22): rerun.


## Archived 2026-11-22 (legacy retro-review; R14 Pythia gate)

1. **Retro-review the legacy experiment files?** Installing the review and journal gates flagged
   34 older experiment files (Feb–May 2026, mostly the secure-code steering work) with results but no
   design review, 13 of them with no journal entry. I grandfathered them in
   `docs/review_allowlist.txt` / `docs/journal_allowlist.txt` so R13 could be committed.
   **Recommendation:** leave them grandfathered; that paper already had its own Aug 25 audit. Re-review
   only files whose numbers the v2 Even Heads draft cites. Trade-off: full retro-review costs ~34
   subagent passes and mostly re-checks the other paper.

   → answer (2026-11-22): Keep the 34 legacy files grandfathered. Re-review any legacy file whose numbers v2 cites before including those numbers.

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

   → answer (2026-11-22): Keep the reproduction gate in force. Drop the Pythia and Gemma head-level claims from v2, including Pythia’s late-emergence claim. Report that the September Pythia all-even-head repair did not reproduce under the original protocol on the current software stack. Neither Gemma-2B nor Gemma-2B-IT showed the format-dependent error on the prompt formats tested.
