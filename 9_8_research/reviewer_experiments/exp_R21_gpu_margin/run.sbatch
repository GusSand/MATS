#!/bin/bash
#SBATCH --job-name=r21_margin
#SBATCH --account=torch_pr_173_tandon_advanced
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/MATS/logs/%j_r21_margin.log

# Experiment file: docs/experiments/11-24_llama8b_gpu_margin_offset.md
# Submit from $SCRATCH/MATS/9_8_research/reviewer_experiments/exp_R21_gpu_margin (the two jobs are independent):
#   sbatch --partition=a100_tandon --nodelist=ga004 --export=ALL,R21_GPU=a100,R21_GIT_SHA=<sha>,R21_SCRIPT_SHA=<sha256> run.sbatch
#   sbatch --partition=l40s_public --nodelist=gl044 --export=ALL,R21_GPU=l40s,R21_GIT_SHA=<sha>,R21_SCRIPT_SHA=<sha256> run.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME="${SCRATCH}/MATS/hf_cache"
export HF_HUB_OFFLINE=1
export R13_MODEL="${HF_HOME}/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
[ -f "${R13_MODEL}/config.json" ] || { echo "FATAL: model snapshot missing"; exit 1; }
: "${R21_GPU:?set R21_GPU}"
: "${R21_GIT_SHA:?set R21_GIT_SHA}"
: "${R21_SCRIPT_SHA:?set R21_SCRIPT_SHA}"
[ "$(sha256sum run_experiment.py | cut -c1-64)" = "${R21_SCRIPT_SHA}" ] || { echo "FATAL: script sha256 mismatch"; exit 2; }
source "${SCRATCH}/MATS/env/bin/activate"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
python -I run_experiment.py --gpu "${R21_GPU}" --out "${SCRATCH}/MATS/results/r21_${R21_GPU}_${SLURM_JOB_ID}"
