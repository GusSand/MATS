#!/bin/bash
#SBATCH --job-name=r20_steer
#SBATCH --account=torch_pr_173_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/MATS/logs/%j_r20_steer.log

# Experiment file: docs/experiments/11-24_llama8b_steering_fix.md
# Submit from $SCRATCH/MATS/9_8_research/reviewer_experiments/exp_R20_steering_fix:
#   sbatch --export=ALL,R20_GIT_SHA=<sha>,R20_SCRIPT_SHA=<sha256 of run_experiment.py at that commit> run.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME="${SCRATCH}/MATS/hf_cache"
export HF_HUB_OFFLINE=1
export R20_MODEL="${HF_HOME}/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
export R13_MODEL="${R20_MODEL}"
[ -f "${R20_MODEL}/config.json" ] || { echo "FATAL: model snapshot missing"; exit 1; }
: "${R20_GIT_SHA:?set R20_GIT_SHA}"
: "${R20_SCRIPT_SHA:?set R20_SCRIPT_SHA to the committed script sha256}"
[ "$(sha256sum run_experiment.py | cut -c1-64)" = "${R20_SCRIPT_SHA}" ] || { echo "FATAL: script sha256 mismatch"; exit 2; }
source "${SCRATCH}/MATS/env/bin/activate"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
python -I run_experiment.py --out "${SCRATCH}/MATS/results/r20_${SLURM_JOB_ID}"
