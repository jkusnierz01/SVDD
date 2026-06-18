#!/bin/bash
#SBATCH --job-name=precompute_aug_pooled
#SBATCH --account=plgdfsingingpwr-cpu
#SBATCH --partition=plgrid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=logs_precompute_aug_pooled_%j.txt
# Stage 2b: pool augmented features. Run after sonics_preprocess_aug.sh
# Usage: sbatch scripts/bash/precompute_aug_pooled.sh 0

set -euo pipefail
VARIANT="${1:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/big_storage/SVDD}"
cd "${PROJECT_ROOT}"
source .venv/bin/activate

./scripts/preprocessing/run.sh sonics pool-aug "${VARIANT}"
