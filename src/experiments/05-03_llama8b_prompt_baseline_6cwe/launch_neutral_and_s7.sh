#!/bin/bash
# Launch D10 (neutral) + D11 (S7 secure-variant) sequentially with one model load.
# Run after D9 (primary adversarial) completes.
set -e

cd /home/paperspace/MATS/src/experiments/05-03_llama8b_prompt_baseline_6cwe

# One python process, two variants — saves a model-load cycle.
nohup env/bin/python 01_run_baseline.py \
    --variants neutral secure \
    --batch-size 8 \
    --tag aux \
    > results/aux_stdout.log 2>&1 &

echo "PID: $!"
echo "Monitor with: tail -f results/aux_stdout.log"
