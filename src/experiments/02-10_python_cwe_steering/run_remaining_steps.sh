#!/bin/bash
# Run remaining experiment steps sequentially: 5, 4, 6
# Step 5 is quick (~10 min) - probe routing (fixes JSON serialization)
# Step 4 is the KEY figure (~3-5 hours) - 6x6 transfer matrix
# Step 6 is ~1 hour - E2E pipeline

set -e
cd /home/paperspace/MATS/src/experiments/02-10_python_cwe_steering

echo "=========================================="
echo "Starting remaining steps at $(date)"
echo "=========================================="

echo ""
echo ">>> Step 5: Probe Routing <<<"
echo "Started at $(date)"
python 05_probe_routing.py 2>&1
echo "Step 5 completed at $(date)"

echo ""
echo ">>> Step 4: Transfer Matrix (KEY FIGURE) <<<"
echo "Started at $(date)"
python 04_transfer_matrix.py 2>&1
echo "Step 4 completed at $(date)"

echo ""
echo ">>> Step 6: E2E Pipeline <<<"
echo "Started at $(date)"
python 06_e2e_pipeline.py 2>&1
echo "Step 6 completed at $(date)"

echo ""
echo "=========================================="
echo "All steps completed at $(date)"
echo "=========================================="
