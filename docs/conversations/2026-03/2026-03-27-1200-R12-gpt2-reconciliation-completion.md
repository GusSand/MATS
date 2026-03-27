TITLE: R12 GPT-2 Reconciliation — Completion (Sections 3-6)
DATE: 2026-03-27
PARTICIPANTS: Gus, Claude (Dev role)
SUMMARY: Completed Experiment R12 sections 3-6. Found and fixed critical shared-token bug in logit evaluation that was causing false 100% error rate. True GPT-2 error rate is ~11%, not 0% (Gus) or 100% (Hoang).

INITIAL PROMPT: Resume R12 experiment sections 3-6 from checkpoint (sections 1-2 already complete).

KEY DECISIONS:
- Identified disk full (ENOSPC) as root cause of previous crashes, not OOM or timeout
- Split monolithic experiment into per-section runner scripts to avoid timeout issues
- Found and fixed critical bug: get_logit_difference used shared token IDs (e.g., "1" appears in both "1.8" and "1.11"), causing logit_diff=0.000 for all evaluations
- Added eval_patched_logits helper function for consistent discriminating-token evaluation in patching sections
- Re-ran all sections 3-6 with fixed evaluation

FILES CHANGED:
- 9_8_research/reviewer_experiments/exp_R12_gpt2_reconciliation/run_experiment.py — Fixed get_logit_difference to filter shared token IDs; added eval_patched_logits helper; fixed Section 5 patched eval
- 9_8_research/reviewer_experiments/exp_R12_gpt2_reconciliation/run_section3.py — NEW: Split runner for Section 3
- 9_8_research/reviewer_experiments/exp_R12_gpt2_reconciliation/run_section4.py — NEW: Split runner for Section 4
- 9_8_research/reviewer_experiments/exp_R12_gpt2_reconciliation/run_section5.py — NEW: Split runner for Section 5
- 9_8_research/reviewer_experiments/exp_R12_gpt2_reconciliation/run_section6.py — NEW: Split runner for Section 6
- 9_8_research/reviewer_experiments/exp_R12_gpt2_reconciliation/run_summary.py — NEW: Summary generator
- 9_8_research/reviewer_experiments/exp_R12_gpt2_reconciliation/results_20260327_130729.json — Final results
- 9_8_research/research_journal_even_heads.md — Added R12 journal entry
- docs/experiments/03-27_gpt2_all_decimal_comparison_reconciliation.md — NEW: Detailed experiment report
