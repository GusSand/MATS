TITLE: LOBO Steering α-Sweep Experiment Implementation
DATE: 2026-01-12
PARTICIPANTS: User, Claude (Developer role)
SUMMARY: Implemented and ran Experiment 2 - LOBO Steering α-Sweep to prove steering generalizes across scenario families. Results show 63x improvement in secure code generation.

INITIAL PROMPT: Continue Experiment 2 implementation - LOBO Steering α-Sweep

KEY DECISIONS:
- Used Leave-One-Base-ID-Out (LOBO) cross-validation with 7 folds
- Tested on held-out scenario families (not just paraphrases)
- Alpha grid: {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5}
- 1 generation per prompt per alpha for full grid coverage
- Layer 31 steering (based on prior experiments)

FILES CHANGED:
- src/experiments/01-12_llama8b_cwe787_lobo_steering/experiment_config.py - Created configuration
- src/experiments/01-12_llama8b_cwe787_lobo_steering/lobo_splits.py - Created LOBO cross-validation logic
- src/experiments/01-12_llama8b_cwe787_lobo_steering/run_experiment.py - Created main orchestrator
- src/experiments/01-12_llama8b_cwe787_lobo_steering/resume_experiment.py - Created resume script
- src/experiments/01-12_llama8b_cwe787_lobo_steering/plotting.py - Created figure generation
- docs/research_journal.md - Added Experiment 2 results
- docs/experiments/01-12_llama8b_cwe787_lobo_steering.md - Created detailed report

RESULTS:
- Baseline (α=0.0): 0.6% secure, 92.1% insecure
- Best (α=3.5): 38.2% secure, 21.2% insecure
- Improvement: +37.6 pp secure (63x), -70.9 pp insecure (77% reduction)
- All 7 folds show consistent improvement - proves cross-scenario generalization
- Zero refusals observed

FIGURES GENERATED:
- lobo_alpha_sweep_strict_20260112_211513.pdf/png
- lobo_per_fold_secure_strict_20260112_211513.pdf/png
- lobo_dual_panel_strict_20260112_211513.pdf/png
- (Plus EXPANDED scoring versions)
