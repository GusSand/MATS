#!/bin/bash
#SBATCH --job-name=r13_heads
#SBATCH --account=torch_pr_173_tandon_advanced
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/MATS/logs/%j_r13_heads.log

# Experiment file: docs/experiments/11-21_llama8b_decimal_true_head_patching.md
# One job, one GPU, all six cells sequentially (same-hardware comparison).
# Submit from $SCRATCH/MATS/9_8_research/reviewer_experiments/exp_R13_true_head_patching:
#   sbatch --export=ALL,R13_GIT_SHA=<sha> run.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME="${SCRATCH}/MATS/hf_cache"
export HF_HUB_OFFLINE=1            # weights are pre-downloaded via torch-dtn
# The shared cache has the snapshot but no refs/main, so resolve by snapshot path.
# Pinning the revision also records exactly which weights were evaluated.
export R13_MODEL="${HF_HOME}/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
[ -f "${R13_MODEL}/config.json" ] || { echo "FATAL: model snapshot missing: ${R13_MODEL}"; exit 1; }
: "${R13_GIT_SHA:?set R13_GIT_SHA to the committed SHA}"
source "${SCRATCH}/MATS/env/bin/activate"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
python run_experiment.py --out "${SCRATCH}/MATS/results/r13_${SLURM_JOB_ID}"
