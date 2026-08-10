#!/bin/bash
#SBATCH -A plgdfsingingpwr-gpu
#SBATCH -p plgrid-gpu-v100
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --output=logs_extract_benchmark_%j.txt
# Offline fp16 feature extraction for SONICS benchmark subset (~9.3k files).
# Usage: sbatch scripts/bash/extract_benchmark_features.sh

module load ffmpeg

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/big_storage/SVDD}"
INPUT_DIR="${INPUT_DIR:-/net/people/plgrid/plgjedrzejkusnierz/scratch/data/Sonics/benchmark_10pct/all_data_16k_mono}"
OUTPUT_DIR="${OUTPUT_DIR:-/net/people/plgrid/plgjedrzejkusnierz/scratch/data/Sonics/benchmark_10pct/features}"

echo "Project root: $PROJECT_ROOT"
echo "Input dir:    $INPUT_DIR"
echo "Output dir:   $OUTPUT_DIR"
echo ""
echo "Note: Demucs mdx_extra weights must exist under ~/.cache/torch/hub/checkpoints/"
echo "      (4x .th files from mdx_extra bag). Pre-download once if SSL fails on cluster."
echo ""

cd "$PROJECT_ROOT"
source .venv/bin/activate

export TORCH_HOME="${TORCH_HOME:-/net/people/plgrid/plgjedrzejkusnierz/scratch/torch_home}"
mkdir -p "$TORCH_HOME/hub/checkpoints"

python scripts/processing/extract_benchmark_features.py \
    --input-dir  "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda \
    --aug-variant 0 \
    --singgraph-window-batch 8 \
    --skip-existing

echo ""
echo "Job done."
