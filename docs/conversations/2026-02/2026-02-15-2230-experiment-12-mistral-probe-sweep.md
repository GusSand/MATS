TITLE: Experiment 12 - Mistral-7B Linear Probe Layer Sweep
DATE: 2026-02-15
PARTICIPANTS: User, Claude (Developer role)
SUMMARY: Ran Experiment 12 to replicate the hierarchical convergence finding from Llama-3.1-8B on Mistral-7B-Instruct-v0.3. Both CWE-787 and CWE-89 show high probe accuracy from early layers but near-zero logit lens probability, confirming the pattern is architecture-general.

INITIAL PROMPT: Run Experiment 12: Mistral-7B Linear Probe Layer Sweep. Goal: Replicate the "hierarchical convergence" finding from Llama-3.1-8B-Instruct on Mistral-7B-Instruct-v0.3.

KEY DECISIONS:
- Used shared ModelLoader and activation collection patterns from existing codebase
- Applied Mistral chat template (not Llama's) via tokenizer.apply_chat_template
- Probed layers [0, 4, 8, 12, 16, 20, 24, 28, 31] with LOBO 7-fold CV
- Tracked P(snprintf) for CWE-787 and P(?) for CWE-89 in logit lens
- CWE-787 dataset uses 'vulnerable'/'secure' keys; CWE-89 uses 'insecure_prompt'/'secure_prompt' — normalized both formats

KEY RESULTS:
- CWE-787: 87.6% accuracy at L0, peak 95.2% at L8, high variance (std 0.08-0.15)
- CWE-89: 95.7% accuracy at L0, 100% at L16, low variance (std 0.00-0.06)
- Logit lens: P(secure token) ≈ 0 at ALL layers for both CWEs
- Vector norms increase monotonically (CWE-787: 0.01→3.80, CWE-89: 0.00→2.08)
- Pattern replicates: early encoding + late emergence confirmed on Mistral

FILES CHANGED:
- src/experiments/02-15_mistral_probe_sweep/01_probe_sweep.py - Main experiment script (NEW)
- src/experiments/02-15_mistral_probe_sweep/results/ - Activations, metadata, and results JSON files (NEW)
- docs/experiments/02-15_mistral7b_cwe787_cwe89_probe_layer_sweep.md - Detailed experiment report (NEW)
- docs/research_journal.md - Added Experiment 12 entry
- docs/DATA_INVENTORY.md - Added Experiment 12 data files
