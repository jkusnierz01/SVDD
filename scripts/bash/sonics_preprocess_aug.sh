#!/bin/bash
#SBATCH -A plgdfsingingpwr-gpu
#SBATCH -p plgrid-gpu-v100
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --output=logs_preprocess_aug_%j.txt
# Stage 1b: offline audio aug + W2V/MERT for Sonics train split (one variant).
# Usage: sbatch scripts/bash/sonics_preprocess_aug.sh 0

module load ffmpeg

set -euo pipefail
VARIANT="${1:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/big_storage/SVDD}"
cd "${PROJECT_ROOT}"
source .venv/bin/activate

echo "Audio aug preprocess variant=${VARIANT}"
./scripts/preprocessing/run.sh sonics features-aug "${VARIANT}"
