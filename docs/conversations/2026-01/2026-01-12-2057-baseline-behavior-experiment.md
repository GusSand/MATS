TITLE: Experiment 1 - Baseline Behavior (Base vs Expanded)
DATE: 2026-01-12
PARTICIPANTS: User, Claude (Developer role)
SUMMARY: Implemented and ran Experiment 1 to measure baseline security behavior of Llama-3.1-8B-Instruct on vulnerable prompts. Key finding: 94% insecure code generation rate with zero refusals.

INITIAL PROMPT: Let's start another experiment:

Experiment 1 — Baseline Behavior (Base vs Expanded)

Goal

Show the unsteered model's security behavior and why Expanded is necessary (stability + diversity).

Datasets
- Base: 7 original scenarios (one per base_id).
- Expanded: all variants (e.g., 105 pairs).

Procedure
1. Run generation on vulnerable prompts only for:
   - Base set
   - Expanded set
2. Score with Strict and Expanded scoring.
3. Aggregate:
   - overall secure / insecure / other
   - by base_id
   - by vuln_type (sprintf vs strcat)

Outputs
- Table: Baseline rates on Base vs Expanded
- Bar chart data: secure/insecure/other by base_id (Expanded)
- Include refusal% as a separate line item

KEY DECISIONS:
- Role progression: PM -> Architect -> Developer
- Dual scoring system: STRICT (canonical API) + EXPANDED (alternative safe solutions)
- Refusal detection: No C-code indicators AND refusal language
- Generation counts: Base=10 per prompt, Expanded=3 per prompt
- Bootstrap 95% CIs with 1000 resamples

FILES CHANGED:
- src/experiments/01-12_llama8b_cwe787_baseline_behavior/experiment_config.py - Configuration
- src/experiments/01-12_llama8b_cwe787_baseline_behavior/scoring.py - STRICT/EXPANDED scoring
- src/experiments/01-12_llama8b_cwe787_baseline_behavior/refusal_detection.py - Refusal detection
- src/experiments/01-12_llama8b_cwe787_baseline_behavior/analysis.py - Bootstrap CIs
- src/experiments/01-12_llama8b_cwe787_baseline_behavior/run_experiment.py - Main orchestrator
- src/experiments/01-12_llama8b_cwe787_baseline_behavior/data/experiment1_results_20260112_200647.json - Results
- src/experiments/01-12_llama8b_cwe787_baseline_behavior/data/experiment1_raw_20260112_200647.json - Raw results
- docs/research_journal.md - Added experiment entry
- docs/experiments/01-12_llama8b_cwe787_baseline_behavior.md - Detailed report

RESULTS SUMMARY:
- Base (n=70): 94.3% insecure [88.6-98.6% CI], 0% secure, 0% refusal
- Expanded (n=315): 93.7% insecure [90.8-96.2% CI], 0.3% secure, 0% refusal
- sprintf prompts: 98.7% insecure
- strcat prompts: 81.1% insecure
- Runtime: ~48 minutes total
