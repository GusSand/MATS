#!/bin/bash
# Reconciliation runs: baseline_steer at Table 2 canonical α for the 3
# full-knowledge CWEs (787, 119, 89). No prompting; chat-template-only;
# bug-fixed harness from 05-03 experiment.
#
# Output: 3 summary JSONs that can be compared directly against Table 2.

set -e
cd /home/paperspace/MATS/src/experiments/05-03_llama8b_prompt_baseline_6cwe

LOG="results/reconciliation_master_$(date +%Y%m%d_%H%M%S).log"

run_one() {
    local cwe=$1
    local alpha=$2
    local tag="recon_${cwe}_a${alpha}"
    echo "==================================================" | tee -a "$LOG"
    echo "[$(date)] ${tag}: baseline_steer at α=${alpha}" | tee -a "$LOG"
    echo "==================================================" | tee -a "$LOG"
    env/bin/python 01b_run_combined.py \
        --variants adversarial \
        --cwes "$cwe" \
        --alpha-override "$alpha" \
        --conditions baseline_steer \
        --batch-size 8 \
        --tag "$tag" \
        2>&1 | tee -a "$LOG"
}

run_one CWE-787 4.0
run_one CWE-119 4.0
run_one CWE-89  12.0

echo "==================================================" | tee -a "$LOG"
echo "[$(date)] Reconciliation complete." | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"
