#!/bin/bash
#SBATCH --job-name=r18_eleven
#SBATCH --account=torch_pr_173_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/MATS/logs/%j_r18_eleven.log

# Experiment file: docs/experiments/11-23_llama8b_eleven_token.md
# Submit from $SCRATCH/MATS/9_8_research/reviewer_experiments/exp_R18_eleven_token:
#   sbatch --export=ALL,R18_GIT_SHA=<sha> run.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME="${SCRATCH}/MATS/hf_cache"
export HF_HUB_OFFLINE=1
export R18_MODEL="${HF_HOME}/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
export R13_MODEL="${R18_MODEL}"
[ -f "${R18_MODEL}/config.json" ] || { echo "FATAL: model snapshot missing"; exit 1; }
: "${R18_GIT_SHA:?set R18_GIT_SHA}"
source "${SCRATCH}/MATS/env/bin/activate"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
python -I run_experiment.py --out "${SCRATCH}/MATS/results/r18_${SLURM_JOB_ID}"
