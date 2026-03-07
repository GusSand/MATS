TITLE: Experiment 28 - Tuned Lens Control for Hierarchical Convergence
DATE: 2026-03-07
PARTICIPANTS: User, Claude (Developer role)
SUMMARY: Ran tuned lens analysis to verify L31 emergence is not representation drift. Trained tuned lens for Llama-3.1-8B-Instruct. Result: Outcome A confirmed - both logit lens and tuned lens show sudden L31 emergence.

INITIAL PROMPT: WE need a new experiment. [Detailed Experiment 28 instructions for tuned lens control analysis]

KEY DECISIONS:
- Used exact same prompts as original logit lens experiment (Exp 01-07)
- Had to train tuned lens locally since no pretrained probes exist on HuggingFace for Llama-3.1-8B-Instruct
- Trained on random token sequences (512 samples) due to HF datasets library compatibility issues
- Trained layer-by-layer with KL divergence objective (5 epochs per layer)
- Added 4 additional prompt pairs from Exp 01-08 for variance estimates

FILES CHANGED:
- src/experiments/03-07_tuned_lens_control/00_train_tuned_lens.py - Training script for tuned lens
- src/experiments/03-07_tuned_lens_control/01_tuned_lens.py - Main experiment script
- src/experiments/03-07_tuned_lens_control/tuned_lens_llama8b/ - Trained tuned lens weights (~2GB)
- src/experiments/03-07_tuned_lens_control/results/ - Experiment results JSON
- docs/experiments/03-07_llama8b_cwe787_tuned_lens_control.md - Detailed experiment report
- docs/research_journal.md - Updated with Exp 28 summary
