#!/bin/bash
#SBATCH -A plgdfsingingpwr-cpu
#SBATCH -p plgrid
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=logs_align_m6_ood_%j.txt

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/big_storage/SVDD}"
FEATURES_DIR="${FEATURES_DIR:-/net/people/plgrid/plgjedrzejkusnierz/scratch/data/M6/features}"
WORKERS="${WORKERS:-16}"

cd "$PROJECT_ROOT"
source .venv/bin/activate

python scripts/processing/align_clam_benchmark_features.py \
  --features-dir "$FEATURES_DIR" \
  --workers "$WORKERS" \
  --skip-existing

python scripts/processing/align_singgraph_benchmark_features.py \
  --features-dir "$FEATURES_DIR" \
  --workers "$WORKERS" \
  --skip-existing

python scripts/processing/rebuild_benchmark_features_index.py --features-dir "$FEATURES_DIR"

echo "Align fallback done."
