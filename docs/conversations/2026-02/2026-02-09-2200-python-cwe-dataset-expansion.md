TITLE: Python CWE Dataset Expansion (CWE-89, CWE-78, CWE-79)
DATE: 2026-02-09
PARTICIPANTS: User, Claude (Developer role)
SUMMARY: Created adversarial prompt pair datasets for 3 Python-language CWEs (SQL Injection, OS Command Injection, XSS) with 105 pairs each, plus 21 neutral evaluation prompts and validated scoring functions.

INITIAL PROMPT: We will be working on expanding our dataset of CWEs. Here are the instructions. [Detailed 8-step instructions for creating CWE-89, CWE-78, CWE-79 datasets with scoring modules, expansion scripts, neutral prompts, and validation tests.]

KEY DECISIONS:
- Used Developer role for implementation
- Checked existing C-language CWE datasets (787, 119, 134) for reusable patterns before writing new code
- Created standalone per-CWE scoring functions (regex-based) rather than extending the shared scoring.py pattern
- Fixed CWE-89 scorer to handle mixed quote delimiters (single quotes inside double-quoted SQL strings)
- Fixed CWE-79 scorer to include render_template() in the HTML detection gate
- CWE-79 secure prompts prepend `import html` to the skeleton via `secure_skeleton_prefix`
- Followed JSONL format with pair_id, base_id, cwe, variation, insecure_prompt, secure_prompt fields

FILES CHANGED:
- src/experiments/02-05_cross_cwe_steering/datasets/cwe89/scoring.py — CWE-89 SQL Injection scorer
- src/experiments/02-05_cross_cwe_steering/datasets/cwe78/scoring.py — CWE-78 OS Command Injection scorer
- src/experiments/02-05_cross_cwe_steering/datasets/cwe79/scoring.py — CWE-79 XSS scorer
- src/experiments/02-05_cross_cwe_steering/datasets/expand_python_datasets.py — Expansion script with all base prompts
- src/experiments/02-05_cross_cwe_steering/datasets/test_scorers.py — 75-test validation suite
- src/experiments/02-05_cross_cwe_steering/datasets/cwe89/data/cwe89_expanded_20260209_221808.jsonl — 105 pairs
- src/experiments/02-05_cross_cwe_steering/datasets/cwe78/data/cwe78_expanded_20260209_221808.jsonl — 105 pairs
- src/experiments/02-05_cross_cwe_steering/datasets/cwe79/data/cwe79_expanded_20260209_221808.jsonl — 105 pairs
- src/experiments/02-05_cross_cwe_steering/datasets/neutral_eval/data/neutral_python_prompts.jsonl — 21 neutral prompts
- docs/DATA_INVENTORY.md — Added Python CWE datasets section
- docs/research_journal.md — Added dataset expansion entry
