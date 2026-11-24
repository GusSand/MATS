#!/bin/bash
#SBATCH --job-name=r22_site
#SBATCH --account=torch_pr_173_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/MATS/logs/%j_r22_site.log

# Experiment file: docs/experiments/11-24_llama8b_site_steering.md
# Submit with scripts/submit.sh (from the repo root): scripts/submit.sh 9_8_research/reviewer_experiments/exp_R22_site_steering R22 <every file below>
# Needs: 9_8_research/reviewer_experiments/exp_R20_steering_fix/run_experiment.py
# Needs: 9_8_research/reviewer_experiments/exp_R19_last_fraction/run_experiment.py
# Needs: 9_8_research/reviewer_experiments/exp_R18_eleven_token/run_experiment.py
# Needs: 9_8_research/reviewer_experiments/exp_R17_localization_map/run_experiment.py
# Needs: 9_8_research/reviewer_experiments/exp_R17_localization_map/results/r17_24213609/run.json

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME="${SCRATCH}/MATS/hf_cache"
export HF_HUB_OFFLINE=1
export R22_MODEL="${HF_HOME}/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
export R13_MODEL="${R22_MODEL}"
[ -f "${R22_MODEL}/config.json" ] || { echo "FATAL: model snapshot missing"; exit 1; }
: "${R22_GIT_SHA:?set R22_GIT_SHA}"
: "${R22_SCRIPT_SHA:?set R22_SCRIPT_SHA to the committed script sha256}"
[ "$(sha256sum run_experiment.py | cut -c1-64)" = "${R22_SCRIPT_SHA}" ] || { echo "FATAL: script sha256 mismatch"; exit 2; }
source "${SCRATCH}/MATS/env/bin/activate"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
python -I run_experiment.py --out "${SCRATCH}/MATS/results/r22_${SLURM_JOB_ID}"
