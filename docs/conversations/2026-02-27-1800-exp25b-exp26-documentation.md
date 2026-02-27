TITLE: Exp 25b Llama-8B Rerun + Exp 26 Qwen-14B CWE-119 LOBO + Documentation
DATE: 2026-02-27
PARTICIPANTS: User (Developer), Claude (Developer)
SUMMARY: Completed Experiment 25b (functional correctness re-evaluation of Llama-8B with untruncated outputs) and monitored Experiment 26 (Qwen-14B CWE-119 LOBO) to completion. Created all documentation and committed results.

INITIAL PROMPT: Rerun functional correctness for Llama-8B only (Exp 25b): Problem: stored Llama-8B outputs were truncated to 500 chars, causing artificially high INCOMPLETE ratings on steered outputs. Fix: regenerate outputs fresh from the model rather than loading from stored results. Generate 25 steered (α=3.5, layer 31) and 25 baseline (α=0.0) outputs on the same prompts used in Exp 25, storing FULL outputs with no truncation. Then re-evaluate with GPT-4o using same prompt as Exp 25. Save as Exp 25b results alongside existing Exp 25 results. Do not overwrite Exp 25 — keep both for comparison.

KEY DECISIONS:
- Re-extracted activations from model rather than loading cached NPZ (file was missing)
- Ran Llama-8B and Qwen-14B concurrently on same A100 GPU (45GB of 81GB used)
- Used same prompt IDs from Exp 25 for direct comparison
- Generated full untruncated outputs (avg 2432 chars steered, 2171 chars baseline)

FILES CHANGED:
- src/experiments/02-27_functional_correctness/02_exp25b_llama8b_rerun.py - Created: full Exp 25b script (re-extract activations, compute LOBO directions, generate, evaluate with GPT-4o)
- src/experiments/02-27_functional_correctness/results/exp25b_correctness_results_20260227_180052.json - Created: Exp 25b full results
- docs/experiments/02-27_functional_correctness.md - Created then updated: detailed report for Exp 25 + Exp 25b
- docs/experiments/02-27_qwen14b_cwe119_lobo.md - Created: detailed report for Exp 26
- docs/research_journal.md - Updated: added Exp 25b and Exp 26 entries
- src/experiments/02-27_qwen14b_cwe119_lobo/results/* - Created by background script: 7 fold files + 2 summary files

KEY RESULTS:
- Exp 25b: Truncation WAS a confound, but steering penalty is actually worse (-36pp vs -28pp from Exp 25)
  - Steered: 24% functional (was 8% truncated)
  - Baseline: 60% functional (was 36% truncated)
- Exp 26: Complete null result — 0% secure across ALL alphas, ALL folds
  - Direction norms extremely high (173-238 vs ~8 for working models)
  - Folds 4-7: 100% insecure at all alphas
  - Folds 1-3: High refusal rates, steering pushes toward refusal not secure code
