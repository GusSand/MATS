#!/bin/bash
# Wait for current combined run (PID 43848) to finish, then launch alpha sweep.
set -e
cd /home/paperspace/MATS/src/experiments/05-03_llama8b_prompt_baseline_6cwe

WAIT_PID=43848
LOG="results/auto_chain_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date)] Watching PID $WAIT_PID..." | tee -a "$LOG"
while kill -0 $WAIT_PID 2>/dev/null; do
    sleep 60
done
echo "[$(date)] PID $WAIT_PID exited. Launching alpha sweep." | tee -a "$LOG"

# Quick sanity: verify combined summary was written
if ! ls results/combined_adversarial_summary_*.json 2>/dev/null | grep -q .; then
    echo "[$(date)] WARNING: combined summary not found. Aborting sweep." | tee -a "$LOG"
    exit 1
fi

bash launch_alpha_sweep.sh 2>&1 | tee -a "$LOG"
echo "[$(date)] Alpha sweep complete." | tee -a "$LOG"
