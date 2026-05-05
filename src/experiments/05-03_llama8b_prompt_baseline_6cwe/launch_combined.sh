#!/bin/bash
# Launch combined steering+prompting experiment.
# Run AFTER D10 (neutral) finishes and any leftover D11 (secure) is killed.

set -e
cd /home/paperspace/MATS/src/experiments/05-03_llama8b_prompt_baseline_6cwe

# Sanity: smoke test first (3 cond × 1 CWE × 5 prompts × 2 seeds = 30 gens, ~3 min)
echo "Running combined smoke test..."
env/bin/python 01b_run_combined.py \
    --variants adversarial \
    --cwes CWE-89 \
    --n-seeds 2 \
    --batch-size 4 \
    --limit-per-cwe 5 \
    --tag combined_smoke \
    2>&1 | tee results/combined_smoke.log

echo "Smoke test done. Inspect results/combined_smoke.log before full run."
