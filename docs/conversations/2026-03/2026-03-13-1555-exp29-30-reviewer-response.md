TITLE: Experiments 29 & 30 — Reviewer Response Experiments
DATE: 2026-03-13
PARTICIPANTS: User, Claude (Developer role)
SUMMARY: Ran two experiments in parallel to address NeurIPS reviewer concerns. Exp 29 confirmed format-suppression hypothesis via logit lens ablation. Exp 30 expanded static analysis validation to CWE-119 (CodeQL) and CWE-89 (Bandit/Semgrep).

INITIAL PROMPT: We need to do two experiments. Here are the instructions. Perhaps we can do them in parallel? [Detailed specs for Exp 29: Format Ablation Logit Lens and Exp 30: Expanded CodeQL/Static Analysis Validation]

KEY DECISIONS:
- Ran experiments in parallel: Exp 29 (GPU-bound) + Exp 30 (CPU-bound)
- Installed CodeQL v2.16.1 fresh (was not on system)
- Used CodeQL query pack v0.9.0 (v1.5.12 had version mismatch errors)
- Used Semgrep `--config auto` instead of `p/python` (latter missed SQL injection rules)
- Used generate-then-truncate approach for logit lens prefixes
- For neutral prompts, added adversarial/secure comments before function signatures

FILES CHANGED:
- src/experiments/03-13_format_ablation_logit_lens/01_format_ablation_logit_lens.py — Logit lens across 3 conditions × 7 scenarios
- src/experiments/03-13_format_ablation_logit_lens/02_plot_ablation.py — Matplotlib visualization
- src/experiments/03-13_format_ablation_logit_lens/results/ — JSON results + PNG figures
- src/experiments/03-13_expanded_codeql_validation/01_cwe119_codeql_validation.py — CWE-119 CodeQL pipeline
- src/experiments/03-13_expanded_codeql_validation/02_cwe89_bandit_semgrep_validation.py — CWE-89 Bandit/Semgrep pipeline
- src/experiments/03-13_expanded_codeql_validation/results/ — Agreement JSONs
- docs/research_journal.md — Added entries for both experiments
