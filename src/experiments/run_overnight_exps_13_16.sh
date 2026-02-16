#!/bin/bash
# Master Runner: Experiments 13-16 (Overnight Batch)
# Runs each experiment sequentially, documents results, commits & pushes after each.
#
# Usage: bash src/experiments/run_overnight_exps_13_16.sh 2>&1 | tee overnight_run.log
#
# Expected total runtime: 12-18 hours
#   - Exp 13 (Mistral CWE-89 LOBO): 4-6 hrs
#   - Exp 14 (Mistral CWE-119 LOBO): 4-6 hrs
#   - Exp 15 (Mistral E2E Pipeline): 2-3 hrs
#   - Exp 16 (Qwen CWE-89 LOBO, optional): 4-6 hrs

set +e  # Continue on error (each experiment is independent)
cd /home/paperspace/MATS

echo "============================================================"
echo "OVERNIGHT EXPERIMENT BATCH: Experiments 13-16"
echo "Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"

# Activate virtual environment
source env/bin/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null || true

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXPERIMENT 13: Mistral-7B CWE-89 LOBO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "EXPERIMENT 13: Mistral-7B CWE-89 LOBO"
echo "Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 src/experiments/02-16_mistral_cwe89_lobo/01_run_experiment.py
EXP13_STATUS=$?

if [ $EXP13_STATUS -eq 0 ]; then
    echo "Exp 13 completed successfully. Committing..."
    git add src/experiments/02-16_mistral_cwe89_lobo/
    git commit -m "Add Experiment 13: Mistral-7B CWE-89 (SQL Injection) LOBO cross-validation

Cross-language x cross-architecture steering validation.
Extracts steering vectors at Layer 31 and runs 7-fold LOBO
with alpha sweep [0.0-7.0] and 3 seeds per prompt.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
    git push
    echo "Exp 13 committed and pushed."
else
    echo "WARNING: Exp 13 failed with status $EXP13_STATUS"
fi

echo "Exp 13 finished: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXPERIMENT 14: Mistral-7B CWE-119 LOBO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "EXPERIMENT 14: Mistral-7B CWE-119 LOBO"
echo "Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 src/experiments/02-17_mistral_cwe119_lobo/01_run_experiment.py
EXP14_STATUS=$?

if [ $EXP14_STATUS -eq 0 ]; then
    echo "Exp 14 completed successfully. Committing..."
    git add src/experiments/02-17_mistral_cwe119_lobo/
    git commit -m "Add Experiment 14: Mistral-7B CWE-119 (Buffer Read Overflow) LOBO

Tests whether CWE-119 weakness is architecture-specific or universal.
Includes CWE-787 vs CWE-119 cosine similarity analysis for
representational inseparability evidence.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
    git push
    echo "Exp 14 committed and pushed."
else
    echo "WARNING: Exp 14 failed with status $EXP14_STATUS"
fi

echo "Exp 14 finished: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXPERIMENT 15: Mistral-7B E2E Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "EXPERIMENT 15: Mistral-7B E2E Pipeline"
echo "Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 src/experiments/02-18_mistral_e2e_pipeline/01_run_experiment.py
EXP15_STATUS=$?

if [ $EXP15_STATUS -eq 0 ]; then
    echo "Exp 15 completed successfully. Committing..."
    git add src/experiments/02-18_mistral_e2e_pipeline/
    git commit -m "Add Experiment 15: Mistral-7B E2E probe-gated steering pipeline

Full deployment pipeline validation on second architecture.
Probe routing (buffer_overflow vs injection), steered generation,
and latency benchmarks.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
    git push
    echo "Exp 15 committed and pushed."
else
    echo "WARNING: Exp 15 failed with status $EXP15_STATUS"
fi

echo "Exp 15 finished: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXPERIMENT 16: Qwen-14B CWE-89 LOBO (Optional)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "EXPERIMENT 16 (OPTIONAL): Qwen-14B CWE-89 LOBO"
echo "Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 src/experiments/02-19_qwen14b_cwe89_lobo/01_run_experiment.py
EXP16_STATUS=$?

if [ $EXP16_STATUS -eq 0 ]; then
    echo "Exp 16 completed successfully. Committing..."
    git add src/experiments/02-19_qwen14b_cwe89_lobo/
    git commit -m "Add Experiment 16: Qwen-14B CWE-89 LOBO (third architecture)

Third architecture for cross-language steering validation.
3-way comparison: Llama-8B vs Mistral-7B vs Qwen-14B for CWE-89.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
    git push
    echo "Exp 16 committed and pushed."
else
    echo "WARNING: Exp 16 failed with status $EXP16_STATUS"
fi

echo "Exp 16 finished: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo "============================================================"
echo "OVERNIGHT BATCH COMPLETE"
echo "Finished: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"
echo "Status:"
echo "  Exp 13 (Mistral CWE-89 LOBO):    $([ $EXP13_STATUS -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
echo "  Exp 14 (Mistral CWE-119 LOBO):   $([ $EXP14_STATUS -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
echo "  Exp 15 (Mistral E2E Pipeline):    $([ $EXP15_STATUS -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
echo "  Exp 16 (Qwen CWE-89 LOBO):       $([ $EXP16_STATUS -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
