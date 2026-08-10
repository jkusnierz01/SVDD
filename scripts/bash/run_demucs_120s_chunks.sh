#!/bin/bash
#SBATCH -A plgdfsingingpwr-gpu
#SBATCH -p plgrid-gpu-v100
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=logs_demucs_120s_%A_%a.txt
#SBATCH --array=0-7
# Output per track: demucs_120s/mdx_extra/{stem}/vocals.wav + no_vocals.wav
# (--skip-existing requires both files; partial old runs are reprocessed)
#
# Usage: sbatch scripts/bash/run_demucs_120s_chunks.sh
#
# Tune NUM_SHARDS to match --array=0-(N-1). With 8 GPUs and ~9322 files,
# each shard ~1165 clips. In-process backend target: ~6-12h total wall time.

module load ffmpeg

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/big_storage/SVDD}"
INPUT_DIR="${INPUT_DIR:-/net/people/plgrid/plgjedrzejkusnierz/scratch/data/Sonics/benchmark_10pct/chunks_120s}"
OUTPUT_DIR="${OUTPUT_DIR:-/net/people/plgrid/plgjedrzejkusnierz/scratch/data/Sonics/benchmark_10pct/demucs_120s}"
NUM_SHARDS="${NUM_SHARDS:-8}"
BACKEND="${BACKEND:-python}"
SHIFTS="${SHIFTS:-0}"
SEGMENT="${SEGMENT:-}"

SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"

cd "$PROJECT_ROOT"
source .venv/bin/activate

export TORCH_HOME="${TORCH_HOME:-/net/people/plgrid/plgjedrzejkusnierz/scratch/torch_home}"
mkdir -p "$TORCH_HOME/hub/checkpoints"

echo "=== Demucs shard ${SHARD_ID}/${NUM_SHARDS} ==="
echo "Node:   ${SLURMD_NODENAME:-local}"
echo "Input:  ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Backend: ${BACKEND} | shifts=${SHIFTS}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true

EXTRA_ARGS=()
if [[ -n "${SEGMENT}" ]]; then
    EXTRA_ARGS+=(--segment "${SEGMENT}")
fi

python scripts/processing/run_demucs_on_120s_chunks.py \
    --input-dir   "$INPUT_DIR" \
    --output-dir  "$OUTPUT_DIR" \
    --backend     "$BACKEND" \
    --device      cuda \
    --fp16 \
    --shifts      "$SHIFTS" \
    --shard-id    "$SHARD_ID" \
    --num-shards  "$NUM_SHARDS" \
    --skip-existing \
    "${EXTRA_ARGS[@]}"

echo "Shard ${SHARD_ID} done."
