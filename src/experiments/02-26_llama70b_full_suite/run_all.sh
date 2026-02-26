#!/bin/bash
# Run all Llama-70B experiments in sequence.
# Usage: nohup bash run_all.sh > run_all.log 2>&1 &
# Monitor: tail -f run_all.log

set -e  # stop on first failure

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "Llama-70B Full Experiment Suite"
echo "Started: $(date)"
echo "============================================================"

# 1. CWE-89 LOBO (~9h)
echo ""
echo "============================================================"
echo "[1/4] CWE-89 LOBO — Starting at $(date)"
echo "============================================================"
python 01_cwe89_lobo.py
echo "[1/4] CWE-89 LOBO — Finished at $(date)"

# 2. CWE-119 LOBO (~8h)
echo ""
echo "============================================================"
echo "[2/4] CWE-119 LOBO — Starting at $(date)"
echo "============================================================"
python 02_cwe119_lobo.py
echo "[2/4] CWE-119 LOBO — Finished at $(date)"

# 3. E2E Pipeline (~2h)
echo ""
echo "============================================================"
echo "[3/4] E2E Pipeline — Starting at $(date)"
echo "============================================================"
python 03_e2e_pipeline.py
echo "[3/4] E2E Pipeline — Finished at $(date)"

# 4. Logit Lens (~1.5h)
echo ""
echo "============================================================"
echo "[4/4] Logit Lens — Starting at $(date)"
echo "============================================================"
python 04_logit_lens.py
echo "[4/4] Logit Lens — Finished at $(date)"

echo ""
echo "============================================================"
echo "All experiments complete at $(date)"
echo "============================================================"
