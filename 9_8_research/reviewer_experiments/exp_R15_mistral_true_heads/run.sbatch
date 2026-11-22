#!/bin/bash
#SBATCH --job-name=r15_mistral
#SBATCH --account=torch_pr_173_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/MATS/logs/%j_r15_mistral.log

# Experiment file: docs/experiments/11-22_mistral7b_true_head_patching.md
# Submit from $SCRATCH/MATS/9_8_research/reviewer_experiments/exp_R15_mistral_true_heads:
#   sbatch --export=ALL,R15_GIT_SHA=<sha> run.sbatch
# Needs ../exp_R14_cross_model_true_heads/run_experiment.py (imported) synced alongside.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME="${SCRATCH}/MATS/hf_cache"
export HF_HUB_OFFLINE=1            # weights pre-downloaded via torch-dtn (snapshot 63a8b081…)
: "${R15_GIT_SHA:?set R15_GIT_SHA to the committed SHA}"
source "${SCRATCH}/MATS/env/bin/activate"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
python -I run_experiment.py --out "${SCRATCH}/MATS/results/r15_${SLURM_JOB_ID}"
