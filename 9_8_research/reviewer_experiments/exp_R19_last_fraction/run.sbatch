#!/bin/bash
#SBATCH --job-name=r19_lastfrac
#SBATCH --account=torch_pr_173_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/MATS/logs/%j_r19_lastfrac.log

# Experiment file: docs/experiments/11-23_llama8b_last_fraction.md
# Submit from $SCRATCH/MATS/9_8_research/reviewer_experiments/exp_R19_last_fraction:
#   sbatch --export=ALL,R19_GIT_SHA=<sha>,R19_SCRIPT_SHA=<sha256 of run_experiment.py at that commit> run.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME="${SCRATCH}/MATS/hf_cache"
export HF_HUB_OFFLINE=1
export R19_MODEL="${HF_HOME}/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
export R13_MODEL="${R19_MODEL}"
[ -f "${R19_MODEL}/config.json" ] || { echo "FATAL: model snapshot missing"; exit 1; }
: "${R19_GIT_SHA:?set R19_GIT_SHA}"
: "${R19_SCRIPT_SHA:?set R19_SCRIPT_SHA to the committed script sha256}"
[ "$(sha256sum run_experiment.py | cut -c1-64)" = "${R19_SCRIPT_SHA}" ] || { echo "FATAL: script sha256 mismatch"; exit 2; }
source "${SCRATCH}/MATS/env/bin/activate"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
python -I run_experiment.py --out "${SCRATCH}/MATS/results/r19_${SLURM_JOB_ID}"
