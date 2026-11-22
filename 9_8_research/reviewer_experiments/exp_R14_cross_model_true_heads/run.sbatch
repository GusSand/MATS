#!/bin/bash
#SBATCH --job-name=r14_xmodel
#SBATCH --account=torch_pr_173_tandon_advanced
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/MATS/logs/%j_r14_xmodel.log

# Experiment file: docs/experiments/11-22_pythia_gemma_true_head_patching.md
# One job, one GPU, models sequentially. Submit from
# $SCRATCH/MATS/9_8_research/reviewer_experiments/exp_R14_cross_model_true_heads:
#   sbatch --export=ALL,R14_GIT_SHA=<sha> run.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME="${SCRATCH}/MATS/hf_cache"
export HF_HUB_OFFLINE=1            # weights pre-downloaded via torch-dtn (refs/main and refs/stepN present)
: "${R14_GIT_SHA:?set R14_GIT_SHA to the committed SHA}"
source "${SCRATCH}/MATS/env/bin/activate"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
OUT="${SCRATCH}/MATS/results/r14_${SLURM_JOB_ID}"
mkdir -p "${OUT}"
# A model whose gate fails exits non-zero; record it and continue with the next model.
for m in pythia160m gemma2b gemma2b_it; do
  python run_experiment.py --model "$m" --out "${OUT}/$m" || echo "MODEL_EXIT_NONZERO $m"
done
python run_experiment.py --model pythia160m --dynamics --main-run "${OUT}/pythia160m/run.json" \
  --out "${OUT}/pythia160m_dynamics" || echo "MODEL_EXIT_NONZERO dynamics"
