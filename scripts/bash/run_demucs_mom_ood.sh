#!/bin/bash
#SBATCH -A plgdfsingingpwr-gpu
#SBATCH -p plgrid-gpu-v100
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=logs_demucs_mom_%A_%a.txt

module load ffmpeg
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/big_storage/SVDD}"
INPUT_DIR="${INPUT_DIR:-/net/people/plgrid/plgjedrzejkusnierz/scratch/data/MoM/chunks_120s}"
OUTPUT_DIR="${OUTPUT_DIR:-/net/people/plgrid/plgjedrzejkusnierz/scratch/data/MoM/demucs_120s}"
NUM_SHARDS="${NUM_SHARDS:-4}"
BACKEND="${BACKEND:-python}"
SHIFTS="${SHIFTS:-0}"
SEGMENT="${SEGMENT:-}"
SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export TORCH_HOME="${TORCH_HOME:-/net/people/plgrid/plgjedrzejkusnierz/scratch/torch_home}"
mkdir -p "$TORCH_HOME/hub/checkpoints"

EXTRA_ARGS=()
if [[ -n "${SEGMENT}" ]]; then
  EXTRA_ARGS+=(--segment "${SEGMENT}")
fi

python scripts/processing/run_demucs_on_120s_chunks.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --backend "$BACKEND" \
  --device cuda \
  --fp16 \
  --shifts "$SHIFTS" \
  --shard-id "$SHARD_ID" \
  --num-shards "$NUM_SHARDS" \
  --skip-existing \
  "${EXTRA_ARGS[@]}"

echo "Shard ${SHARD_ID} done."
