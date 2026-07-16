#!/bin/bash
#SBATCH -A plgdfsingingpwr-cpu
#SBATCH -p plgrid
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=logs_prepare_mom_120s_%j.txt

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/big_storage/SVDD}"
INPUT_ROOT="${INPUT_ROOT:-/net/people/plgrid/plgjedrzejkusnierz/scratch/data/MoM/MoM_16k}"
OUTPUT_DIR="${OUTPUT_DIR:-/net/people/plgrid/plgjedrzejkusnierz/scratch/data/MoM/chunks_120s}"

cd "$PROJECT_ROOT"
source .venv/bin/activate

python scripts/processing/prepare_ood_120s_chunks.py \
  --dataset mom \
  --input-root "$INPUT_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --skip-existing

echo "Done."
