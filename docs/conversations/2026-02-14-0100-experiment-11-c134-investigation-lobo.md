TITLE: Experiment 11 - C-134 Transfer Matrix Investigation & Full LOBO
DATE: 2026-02-14
PARTICIPANTS: User, Claude
SUMMARY: Investigated why CWE-134 scored 0% on the transfer matrix diagonal when Exp 8.5 reported 100%. Found root cause (prompt type mismatch + ceiling effect + undertrained vector). Ran full 7-fold LOBO with extended alpha sweep (0-10), finding best alpha=3.0 with modest +4.8pp improvement. Re-ran transfer matrix C-134 row with α=3.0 — diagonal still 0% (outputs garbled, not redirected to secure). Confirmed as legitimate steering failure.

INITIAL PROMPT: We will do experiment 11. [Instructions to investigate C-134 diagonal 0% in transfer matrix, check alphas, check prompt differences vs Exp 8.5, and optionally re-run with proper alpha sweep]

KEY DECISIONS:
- Phase 1 was forensic investigation only (no model runs)
- Root cause identified as prompt mismatch (insecure-variant vs neutral) + ceiling effect + undertrained 2-fold pilot
- Proceeded with Phase 2: Full 7-fold LOBO with alpha 0-10
- Reduced from 10 seeds to 3 seeds and 512 to 256 max tokens for feasible runtime (~8 hours)
- Best alpha found: 3.0 with 74.9% secure (+4.8pp over 70.2% baseline)
- Hard folds (system_log, audit_log) have ~40-47% baselines, barely improve with steering
- Phase 3: Re-ran transfer matrix C-134 row with α=3.0 — diagonal still 0% (all "other"), Py-89 dropped to 62%
- Confirmed C-134 as legitimate steering failure on explicit vulnerability prompts

FILES CHANGED:
- docs/research_journal.md - Added Experiment 11 entry with Phase 1 + Phase 2 + Phase 3 results
- docs/experiments/02-13_llama8b_c134_transfer_matrix_investigation.md - Created detailed experiment report (all 3 phases)
- docs/DATA_INVENTORY.md - Added Experiment 11 data entries
- src/experiments/02-13_c134_full_lobo/run_full_lobo.py - Full LOBO experiment script
- src/experiments/02-13_c134_full_lobo/rerun_c134_transfer_row.py - Transfer matrix row re-run script
- src/experiments/02-13_c134_full_lobo/results/ - 7 per-fold results + 2 aggregate results + transfer row + updated matrix
- src/experiments/02-13_c134_full_lobo/data/activations_cwe134_L31_20260213_221204.npz - Layer 31 activations
- src/experiments/02-10_python_cwe_steering/results/c134_investigation_20260213.json - Phase 1 investigation results
- src/experiments/02-10_python_cwe_steering/results/c134_full_lobo_20260213_222152.json - Copy of LOBO results
- src/experiments/02-10_python_cwe_steering/results/transfer_matrix_updated_20260214_121747.json - Updated transfer matrix
