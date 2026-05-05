#!/bin/bash
# Alpha sweep on combined steering+prompting for 3 CWEs.
# Run AFTER the combined experiment finishes.
#
# Strategy:
# - CWE-787: try lower α (1.0, 2.0) — current α=3.5 over-steers when combined
# - CWE-89:  try higher α (8.0, 10.0, 12.0) — current α=5.0 subsumes prompting
# - CWE-78:  try lower α (2.0, 3.0) — current α=5.0 over-steers when combined
#
# Each (CWE, α) run = 3 conditions × 105 prompts × 10 seeds = 3,150 generations.
# 7 runs total ≈ 9-10h overnight.

set -e
cd /home/paperspace/MATS/src/experiments/05-03_llama8b_prompt_baseline_6cwe

LOG_DIR=results
LOG_FILE="${LOG_DIR}/alpha_sweep_master_$(date +%Y%m%d_%H%M%S).log"

run_one() {
    local cwe=$1
    local alpha=$2
    local tag="sweep_${cwe}_a${alpha}"
    echo "==================================================" | tee -a "$LOG_FILE"
    echo "[$(date)] Launching ${tag}" | tee -a "$LOG_FILE"
    echo "==================================================" | tee -a "$LOG_FILE"
    # Only test combined-with-prompting conditions; steering-alone (baseline_steer)
    # is omitted to save GPU time — we already have it from canonical-α runs and LOBO.
    env/bin/python 01b_run_combined.py \
        --variants adversarial \
        --cwes "$cwe" \
        --alpha-override "$alpha" \
        --conditions sys_verbose_steer usr_verbose_steer \
        --batch-size 8 \
        --tag "$tag" \
        2>&1 | tee -a "$LOG_FILE"
}

# CWE-787: lower alphas (over-steered at canonical α=3.5)
run_one CWE-787 1.0
run_one CWE-787 2.0

# CWE-89: higher alphas (subsumed at canonical α=5.0; paper claim was at α=12)
run_one CWE-89 8.0
run_one CWE-89 10.0
run_one CWE-89 12.0

# CWE-78: lower alphas (over-steered at canonical α=5.0)
run_one CWE-78 2.0
run_one CWE-78 3.0

echo "==================================================" | tee -a "$LOG_FILE"
echo "[$(date)] All sweep runs complete." | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"
