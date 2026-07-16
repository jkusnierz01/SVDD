#!/bin/bash
#SBATCH -A plgdfsingingpwr-gpu
#SBATCH -p plgrid-gpu-v100
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --array=0-3
#SBATCH --output=logs_clam_m6_%A_%a.txt

module load ffmpeg
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/big_storage/SVDD}"
INPUT_DIR="${INPUT_DIR:-/net/people/plgrid/plgjedrzejkusnierz/scratch/data/M6/chunks_120s}"
OUTPUT_DIR="${OUTPUT_DIR:-/net/people/plgrid/plgjedrzejkusnierz/scratch/data/M6/features_clean}"
NUM_SHARDS="${NUM_SHARDS:-4}"
EXTRA_ARGS="${EXTRA_ARGS:---skip-proposed --augment-scope none}"
SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export TORCH_HOME="${TORCH_HOME:-/net/people/plgrid/plgjedrzejkusnierz/scratch/torch_home}"
mkdir -p "$TORCH_HOME/hub/checkpoints"

# shellcheck disable=SC2206
python scripts/processing/extract_proposed_clam_benchmark_features.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda \
  --shard-id "$SHARD_ID" \
  --num-shards "$NUM_SHARDS" \
  --skip-existing \
  ${EXTRA_ARGS}

echo "Shard ${SHARD_ID} done."
