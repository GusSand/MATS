#!/bin/bash
#SBATCH --job-name=r16_pos0_a100
#SBATCH --account=torch_pr_173_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ga004
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/MATS/logs/%j_r16_pos0_a100.log

# Experiment file: docs/experiments/11-22_llama8b_position0_gpu_gap.md
# Submit from $SCRATCH/MATS/9_8_research/reviewer_experiments/exp_R16_position0:
#   A=$(sbatch --parsable --export=ALL,R16_GIT_SHA=<sha> run_a100.sbatch)
#   sbatch --dependency=afterok:$A --export=ALL,R16_GIT_SHA=<sha>,R16_DELTA=$SCRATCH/MATS/results/r16_a100_$A/position0.pt run_l40s.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME="${SCRATCH}/MATS/hf_cache"
export HF_HUB_OFFLINE=1
export R13_MODEL="${HF_HOME}/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
[ -f "${R13_MODEL}/config.json" ] || { echo "FATAL: model snapshot missing"; exit 1; }
: "${R16_GIT_SHA:?set R16_GIT_SHA}"
source "${SCRATCH}/MATS/env/bin/activate"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
python -I run_experiment.py --gpu a100 --out "${SCRATCH}/MATS/results/r16_a100_${SLURM_JOB_ID}"
