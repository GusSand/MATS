#!/bin/bash
# submit.sh <exp_dir> <ID, e.g. R22> [imported scripts and reference files...]: preflight, copy (no --delete), compare hashes, sbatch.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
DIR="${1%/}"; ID="$2"; shift 2
python3 scripts/dry_run_check.py "$DIR" "$@"
SHA=$(shasum -a 256 "$DIR/run_experiment.py" | cut -c1-64)
GIT=$(git rev-parse --short HEAD)
ssh -o BatchMode=yes -o ConnectTimeout=6 torch true || { echo "Torch connection is down: ask Gus to run 'torch'"; exit 1; }
rsync -aR "$DIR/run_experiment.py" "$DIR/run.sbatch" "$@" torch-dtn:/scratch/gs157/MATS/
for f in "$DIR/run_experiment.py" "$DIR/run.sbatch" "$@"; do
  a=$(shasum -a 256 "$f" | cut -c1-64); b=$(ssh -o BatchMode=yes torch "sha256sum /scratch/gs157/MATS/$f" | cut -c1-64)
  [ "$a" = "$b" ] || { echo "hash mismatch on Torch: $f"; exit 1; }
done
ssh -o BatchMode=yes torch "cd /scratch/gs157/MATS/$DIR && sbatch --export=ALL,${ID}_GIT_SHA=$GIT,${ID}_SCRIPT_SHA=$SHA run.sbatch"
