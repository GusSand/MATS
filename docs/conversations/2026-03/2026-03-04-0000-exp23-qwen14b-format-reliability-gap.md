TITLE: Experiment 23 — Qwen 14B Format-Reliability Gap
DATE: 2026-03-04
PARTICIPANTS: User, Claude
SUMMARY: Ran experiment 22b (Format-Reliability Gap code review) on Qwen2.5-14B-Instruct as experiment 23. Reused existing code from exp22b, added Qwen baselines from prior LOBO experiments. GPT-4o scoring + CWE-119 re-scoring. Results: 100% review accuracy on CWE-787/119/89 with massive knowledge-execution gaps (+97.1pp, +100.0pp, +61.6pp). CWE-134 hardest at 40%.

INITIAL PROMPT: I need you to run experiment 22b but with the Qwen 14B model. Call it experiment 23 and document per claude.md

KEY DECISIONS:
- Reuse existing exp22b code (already had qwen14b in MODEL_CONFIGS)
- Use GPT-4o scoring from the start
- Apply same CWE-119 relaxed re-scoring
- Added Qwen 14B LOBO baselines from Exp 4c (CWE-787: 2.9%), Exp 26 (CWE-119: 0.0%), Exp 16 (CWE-89: 38.4%)

FILES CHANGED:
- src/experiments/03-03_exp22b_format_reliability_gap/exp22b_prompts.py — Added qwen14b baseline code security rates
- src/experiments/03-03_exp22b_format_reliability_gap/results/exp22b_qwen14b_20260304_224313/ — Full results (results.json, results_rescored_119.json, gap_table.csv, SUMMARY.md)
- docs/research_journal.md — Added Experiment 23 entry
- docs/experiments/03-04_qwen14b_format_reliability_gap.md — Detailed experiment report
