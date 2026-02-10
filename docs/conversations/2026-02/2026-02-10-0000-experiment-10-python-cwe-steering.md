TITLE: Experiment 10 — Python CWE Steering & Cross-Language Validation
DATE: 2026-02-10
PARTICIPANTS: User, Claude (Developer role)
SUMMARY: Ran full 7-step experiment pipeline for Python CWE steering (CWE-89, CWE-78, CWE-79). Fixed scorer calibration issues (CWE-89: 42%→1.3% other, CWE-79: 44%→2.6% other). Discovered and fixed critical SteeringGenerator bug (character-based vs token-based prompt stripping). Extracted steering vectors, validated via LOBO (+13.3pp/+7.7pp/+30.3pp), built 6x6 cross-language transfer matrix (3.8x diagonal ratio), achieved 100% probe routing, and +19.0pp E2E improvement on neutral prompts.

INITIAL PROMPT: Please continue the conversation from where we left it off without asking the user any further questions. Continue with the last task that you were asked to work on. (Continuation from prior session that ran Experiment 10 Steps 1-2 and wrote Steps 3-7.)

KEY DECISIONS:
- Fixed CWE-89 scorer with 3 incremental edits (variable-passed queries, f-prefix, triple-quoted f-strings)
- Fixed CWE-79 scorer (triple-quoted f-string HTML detection)
- Fixed SteeringGenerator bug: replaced character-based prompt stripping with token-based
- Re-ran LOBO after bug fix (first run's ~77% "other" rate was caused by the bug)
- Used best α=5.0 for all Python CWEs based on LOBO results
- Fixed f-string syntax error in transfer matrix script (Python 3.11 compatibility)
- Fixed numpy bool_ JSON serialization in probe routing script

FILES CHANGED:
- `src/experiments/02-05_cross_cwe_steering/datasets/cwe89/scoring.py` — 3 scorer fixes
- `src/experiments/02-05_cross_cwe_steering/datasets/cwe79/scoring.py` — 1 scorer fix
- `src/experiments/02-05_cross_cwe_steering/shared/steering_generator.py` — Token-based prompt stripping fix
- `src/experiments/02-10_python_cwe_steering/04_transfer_matrix.py` — f-string syntax fix
- `src/experiments/02-10_python_cwe_steering/05_probe_routing.py` — numpy bool_ fix, save order fix
- `src/experiments/02-10_python_cwe_steering/results/` — All 7 result files created
- `src/experiments/02-10_python_cwe_steering/data/` — Vectors, activations, probe weights
- `docs/research_journal.md` — Added Experiment 10 entry
- `docs/experiments/02-10_llama8b_python_cwe_steering.md` — Created detailed report
- `docs/DATA_INVENTORY.md` — Added Experiment 10 data section
