#!/bin/bash
#SBATCH -A plgdfsingingpwr-gpu
#SBATCH -p plgrid-gpu-v100
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --output=logs_singgraph_%A_%a.txt
#SBATCH --array=0-7
# Step 4: SingGraph fp16 features from pre-separated Demucs stems.
# Usage: sbatch scripts/bash/extract_singgraph_benchmark_features.sh
#
# Example with full RawBoost:
#   sbatch --export=ALL,EXTRA_ARGS="--rawboost-mode full" \
#     scripts/bash/extract_singgraph_benchmark_features.sh
#
# Clean-only example:
#   sbatch --export=ALL,OUTPUT_DIR=.../features_clean,EXTRA_ARGS="--rawboost-mode none --rawboost-scope none" \
#     scripts/bash/extract_singgraph_benchmark_features.sh

module load ffmpeg

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/big_storage/SVDD}"
DEMUCS_DIR="${DEMUCS_DIR:-/net/people/plgrid/plgjedrzejkusnierz/scratch/data/Sonics/benchmark_10pct/demucs_120s}"
OUTPUT_DIR="${OUTPUT_DIR:-/net/people/plgrid/plgjedrzejkusnierz/scratch/data/Sonics/benchmark_10pct/features_train_aug_v0}"
NUM_SHARDS="${NUM_SHARDS:-8}"
EXTRA_ARGS="${EXTRA_ARGS:---rawboost-mode ssi --rawboost-scope train}"

SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"

cd "$PROJECT_ROOT"
source .venv/bin/activate

export TORCH_HOME="${TORCH_HOME:-/net/people/plgrid/plgjedrzejkusnierz/scratch/torch_home}"
mkdir -p "$TORCH_HOME/hub/checkpoints"

echo "=== SingGraph shard ${SHARD_ID}/${NUM_SHARDS} ==="
echo "Node:   ${SLURMD_NODENAME:-local}"
echo "Demucs: ${DEMUCS_DIR}"
echo "Output: ${OUTPUT_DIR}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true

# shellcheck disable=SC2206
python scripts/processing/extract_singgraph_benchmark_features.py \
    --demucs-dir  "$DEMUCS_DIR" \
    --output-dir  "$OUTPUT_DIR" \
    --device      cuda \
    --shard-id    "$SHARD_ID" \
    --num-shards  "$NUM_SHARDS" \
    --skip-existing \
    ${EXTRA_ARGS}

echo "Shard ${SHARD_ID} done."
