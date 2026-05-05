#!/bin/bash
# True LOBO reconciliation for CWE-787 and CWE-89 (CWE-119 already done).
set -e
cd /home/paperspace/MATS/src/experiments/05-03_llama8b_prompt_baseline_6cwe

LOG="results/lobo_recon_remaining_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date)] CWE-787 LOBO @ α=4.0" | tee -a "$LOG"
env/bin/python 01c_lobo_recon.py --cwe CWE-787 --alpha 4.0 2>&1 | tee -a "$LOG"

echo "[$(date)] CWE-89 LOBO @ α=12.0" | tee -a "$LOG"
env/bin/python 01c_lobo_recon.py --cwe CWE-89 --alpha 12.0 2>&1 | tee -a "$LOG"

echo "[$(date)] LOBO reconciliation remaining done." | tee -a "$LOG"
