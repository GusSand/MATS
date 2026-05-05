#!/bin/bash
# Phase A: CWE-787 baseline_steer LOBO sweep at α ∈ {2, 3, 3.5, 4, 5}
# Phase B: CWE-787 usr_verbose_steer LOBO at best α from Phase A
# Phase C: CWE-119 usr_verbose_steer LOBO at α=1.0
#
# Phase B's α is determined dynamically from Phase A's output JSON.

set -e
cd /home/paperspace/MATS/src/experiments/05-03_llama8b_prompt_baseline_6cwe

LOG="results/lobo_phase_abc_$(date +%Y%m%d_%H%M%S).log"

echo "==================================================" | tee -a "$LOG"
echo "[$(date)] Phase A: CWE-787 baseline_steer LOBO α-sweep" | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"
env/bin/python 01d_lobo_combined_sweep.py \
    --cwe CWE-787 \
    --alphas 2.0 3.0 3.5 4.0 5.0 \
    --condition baseline \
    --tag lobo_sweep_787 \
    2>&1 | tee -a "$LOG"

# Find best α from Phase A
PHASE_A_FILE=$(ls -t results/lobo_sweep_787_CWE-787_baseline_*.json | head -1)
BEST_ALPHA=$(python3 -c "
import json
d = json.load(open('$PHASE_A_FILE'))
best = max(d['results_per_alpha'].items(),
           key=lambda kv: kv[1]['aggregate']['strict_secure_rate'])
print(best[0])
")
echo "[$(date)] Phase A best α=$BEST_ALPHA from $PHASE_A_FILE" | tee -a "$LOG"

echo "==================================================" | tee -a "$LOG"
echo "[$(date)] Phase B: CWE-787 usr_verbose_steer LOBO at α=$BEST_ALPHA" | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"
env/bin/python 01d_lobo_combined_sweep.py \
    --cwe CWE-787 \
    --alphas "$BEST_ALPHA" \
    --condition usr_verbose \
    --tag lobo_combined_787 \
    2>&1 | tee -a "$LOG"

echo "==================================================" | tee -a "$LOG"
echo "[$(date)] Phase C: CWE-119 usr_verbose_steer LOBO at α=1.0" | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"
env/bin/python 01d_lobo_combined_sweep.py \
    --cwe CWE-119 \
    --alphas 1.0 \
    --condition usr_verbose \
    --tag lobo_combined_119 \
    2>&1 | tee -a "$LOG"

echo "==================================================" | tee -a "$LOG"
echo "[$(date)] All 3 phases complete." | tee -a "$LOG"
