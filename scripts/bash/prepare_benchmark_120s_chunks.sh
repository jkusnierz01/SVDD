#!/bin/bash
#SBATCH -A plgdfsingingpwr-cpu
#SBATCH -p plgrid
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=logs_prepare_120s_chunks_%j.txt
# Step 1: pad/loop all benchmark FLACs to fixed 120s clips.
# Usage: sbatch scripts/bash/prepare_benchmark_120s_chunks.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/big_storage/SVDD}"
INPUT_DIR="${INPUT_DIR:-/net/people/plgrid/plgjedrzejkusnierz/scratch/data/Sonics/benchmark_10pct/all_data_16k_mono}"
OUTPUT_DIR="${OUTPUT_DIR:-/net/people/plgrid/plgjedrzejkusnierz/scratch/data/Sonics/benchmark_10pct/chunks_120s}"

cd "$PROJECT_ROOT"
source .venv/bin/activate

python scripts/processing/prepare_benchmark_120s_chunks.py \
    --input-dir  "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --workers 16 \
    --skip-existing

echo "Done."
