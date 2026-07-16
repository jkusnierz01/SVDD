#!/bin/bash
#SBATCH -A plgdfsingingpwr-cpu
#SBATCH -p plgrid
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=logs_finalize_mamba_pooled_%j.txt
#
# Fix Mamba pooled index only (CLAM/SingGraph already merged).
# Builds aug pooled into separate dir, then merges train-aug into proposed_pooled/index.json.
#
# Usage:
#   sbatch scripts/bash/finalize_mamba_pooled_only.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/big_storage/SVDD}"
BASE="${BASE:-/net/people/plgrid/plgjedrzejkusnierz/scratch/data/Sonics/benchmark_10pct}"
AUG_FEATURES="${AUG_FEATURES:-$BASE/features_train_aug_v0}"
POOLED_DIR="${POOLED_DIR:-$BASE/proposed_pooled}"
AUG_POOLED_DIR="${AUG_POOLED_DIR:-$BASE/proposed_pooled_aug_v0}"

cd "$PROJECT_ROOT"
source .venv/bin/activate

if [ ! -f "$POOLED_DIR/index.clean_only.json" ]; then
  cp "$POOLED_DIR/index.json" "$POOLED_DIR/index.clean_only.json"
fi

python scripts/processing/merge_benchmark_proposed_pooled.py \
  --features-dir "$AUG_FEATURES" \
  --output-dir "$AUG_POOLED_DIR" \
  --skip-existing

python scripts/processing/merge_pooled_index.py \
  --clean "$POOLED_DIR/index.clean_only.json" \
  --aug "$AUG_POOLED_DIR/index.json" \
  --out "$POOLED_DIR/index.json"

python scripts/processing/validate_training_data_counts.py --check-loaders

echo "Done. Mamba index: $POOLED_DIR/index.json"
