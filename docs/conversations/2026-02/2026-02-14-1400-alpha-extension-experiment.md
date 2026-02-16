TITLE: LOBO Alpha Extension Experiment (CWE-89, 78, 79)
DATE: 2026-02-14
PARTICIPANTS: User, Claude (Developer role)
SUMMARY: Ran LOBO cross-validation at higher alphas (6-15) for Python CWEs. Found CWE-89 sweet spot at α=12 (+8.2pp). CWE-78 and CWE-79 show no benefit due to large direction norms causing coherence collapse.

INITIAL PROMPT: new experiment. We will run a LOBO cross validation to higher alphas for CWEs 89, 78, 79. The python CWEs.

KEY DECISIONS:
- Extended alpha grid to {6, 7, 8, 10, 12, 15} while keeping prior {0-5} results
- Ran CWE-89 and CWE-78 together, then killed process when CWE-78 got stuck on gibberish generation at α=15
- Ran CWE-79 separately with exclusive GPU access
- Triple-checked datasets and scorers for CWE-78/79 — confirmed "other" rate is genuine coherence collapse, not scorer bugs
- Identified effective magnitude (norm × α) as key predictor of steering success

FILES CHANGED:
- src/experiments/02-10_python_cwe_steering/03b_lobo_alpha_extension.py — Main alpha extension script
- src/experiments/02-10_python_cwe_steering/03c_lobo_alpha_cwe79_only.py — CWE-79 only variant
- src/experiments/02-10_python_cwe_steering/parse_alpha_extension_output.py — Output log parser
- results/alpha_extension_partial_89_78.json — CWE-89/78 fold data
- results/alpha_extension_results_partial.json — CWE-89/78 aggregated
- results/alpha_extension_full_20260215_015309.json — CWE-79 fold data
- results/alpha_extension_results_20260215_015309.json — CWE-79 summary
- results/alpha_curve_merged_20260215_015309.json — Merged alpha curves
- docs/experiments/02-16_llama8b_python_cwes_alpha_extension.md — Experiment report
- docs/research_journal.md — Updated with TLDR entry
