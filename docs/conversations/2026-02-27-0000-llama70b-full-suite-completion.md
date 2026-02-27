TITLE: Llama-70B Full Suite Completion (Exps 19-21)
DATE: 2026-02-27
PARTICIPANTS: User, Claude
SUMMARY: Completed remaining Llama-70B experiments (CWE-119 LOBO, E2E Pipeline, Logit Lens), documented, committed, pushed, and shut down machine.

INITIAL PROMPT: Continuation from previous session — CWE-119 LOBO had crashed, needed restart and completion of remaining experiments.

KEY DECISIONS:
- Killed zombie GPU process (PID 44122, 44GB VRAM) that caused OOM on resume attempt
- CWE-119 LOBO resumed from fold 3 with narrowed alpha grid [0.0, 1.0, 1.5]
- Launched E2E pipeline and logit lens sequentially after CWE-119 completed
- Documented all results and pushed to GitHub before machine shutdown

FILES CHANGED:
- docs/research_journal.md - Added Experiments 19-21 entries
- docs/experiments/02-26_llama70b_cwe787_cwe89_lobo.md - Updated to include CWE-119, E2E, Logit Lens results
- src/experiments/02-26_llama70b_full_suite/02b_cwe119_lobo_resume.py - Resume script (created in prior session)
- src/experiments/02-26_llama70b_full_suite/results/ - 14 new result files

KEY RESULTS:
- Exp 19 (CWE-119 LOBO): Best α=1.0, 38.4% secure (+38.4pp). Bimodal: gets→fgets folds 82-91%, strcpy→strncpy folds 0%.
- Exp 20 (E2E): CWE-787 100% secure when steered. CWE-119 hurt by cross-CWE interference (-20pp). Overall 87.1%.
- Exp 21 (Logit Lens): Late-layer emergence at L75-79. Both sprintf/snprintf single tokens on 70B.
