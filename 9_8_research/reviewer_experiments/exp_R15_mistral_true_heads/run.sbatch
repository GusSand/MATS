#!/bin/bash
#SBATCH --job-name=r15_mistral
#SBATCH --account=torch_pr_173_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/%u/MATS/logs/%j_r15_mistral.log

# Experiment file: docs/experiments/11-22_mistral7b_true_head_patching.md
# Submit from $SCRATCH/MATS/9_8_research/reviewer_experiments/exp_R15_mistral_true_heads:
#   sbatch --export=ALL,R15_GIT_SHA=<sha>,R14_EXPECTED_SHA256=<sha256 of R14 script at that commit> run.sbatch
#   (compute locally: git show <sha>:9_8_research/reviewer_experiments/exp_R14_cross_model_true_heads/run_experiment.py | shasum -a 256)
# Needs ../exp_R14_cross_model_true_heads/run_experiment.py (imported) synced alongside; the job refuses a mismatch.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME="${SCRATCH}/MATS/hf_cache"
export HF_HUB_OFFLINE=1            # weights pre-downloaded via torch-dtn (snapshot 63a8b081…)
: "${R15_GIT_SHA:?set R15_GIT_SHA to the committed SHA}"
: "${R14_EXPECTED_SHA256:?set R14_EXPECTED_SHA256 (see header)}"
source "${SCRATCH}/MATS/env/bin/activate"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
python -I run_experiment.py --out "${SCRATCH}/MATS/results/r15_${SLURM_JOB_ID}"
