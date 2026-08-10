#!/bin/bash
#SBATCH -A plgdfsingingpwr-cpu
#SBATCH -p plgrid
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=01:00:00
#SBATCH --output=logs_finalize_sonics_benchmark_%j.txt
#
# Rebuild + merge SONICS benchmark indexes for training:
#   - CLAM/SingGraph: features_clean/index.json (clean valid/test + train clean + train aug)
#   - Mamba:          proposed_pooled/index.json (clean valid/test + train clean + train aug)
#
# Usage:
#   sbatch scripts/bash/finalize_sonics_benchmark_indices.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/big_storage/SVDD}"
BASE="${BASE:-/net/people/plgrid/plgjedrzejkusnierz/scratch/data/Sonics/benchmark_10pct}"
CLEAN_FEATURES="${CLEAN_FEATURES:-$BASE/features_clean}"
AUG_FEATURES="${AUG_FEATURES:-$BASE/features_train_aug_v0}"
POOLED_DIR="${POOLED_DIR:-$BASE/proposed_pooled}"

cd "$PROJECT_ROOT"
source .venv/bin/activate

echo "=== Rebuild clean feature index ==="
python scripts/processing/rebuild_benchmark_features_index.py \
  --features-dir "$CLEAN_FEATURES" \
  --backup

echo "=== Rebuild train-aug feature index ==="
python scripts/processing/rebuild_benchmark_features_index.py \
  --features-dir "$AUG_FEATURES" \
  --backup

CLEAN_INDEX="$CLEAN_FEATURES/index.json"
AUG_INDEX="$AUG_FEATURES/index.json"
MERGED_FEATURES_INDEX="$CLEAN_FEATURES/index.json"
POOLED_INDEX="$POOLED_DIR/index.json"
MERGED_POOLED_INDEX="$POOLED_DIR/index.json"

echo "=== Backup indexes before merge ==="
cp "$CLEAN_INDEX" "$CLEAN_FEATURES/index.clean_only.json"
cp "$POOLED_INDEX" "$POOLED_DIR/index.clean_only.json"

echo "=== Merge CLAM/SingGraph/Proposed feature index (train aug only) ==="
python scripts/processing/merge_feature_indexes.py \
  --clean "$CLEAN_FEATURES/index.clean_only.json" \
  --aug "$AUG_INDEX" \
  --out "$MERGED_FEATURES_INDEX"

echo "=== Build separate aug pooled dir for Mamba ==="
AUG_POOLED_DIR="${AUG_POOLED_DIR:-$BASE/proposed_pooled_aug_v0}"
python scripts/processing/merge_benchmark_proposed_pooled.py \
  --features-dir "$AUG_FEATURES" \
  --output-dir "$AUG_POOLED_DIR" \
  --skip-existing

echo "=== Merge Mamba pooled index (train aug only) ==="
python scripts/processing/merge_pooled_index.py \
  --clean "$POOLED_DIR/index.clean_only.json" \
  --aug "$AUG_POOLED_DIR/index.json" \
  --out "$MERGED_POOLED_INDEX"

echo "=== Minimal validation ==="
python scripts/processing/validate_benchmark_index_minimal.py \
  --index "$MERGED_FEATURES_INDEX" \
  --forbid-aug-outside-train

python scripts/processing/validate_benchmark_index_minimal.py \
  --index "$MERGED_POOLED_INDEX"

echo "Done."
echo "Train CLAM/SingGraph from: $MERGED_FEATURES_INDEX"
echo "Train Mamba from:          $MERGED_POOLED_INDEX"
echo "Eval clean-only backup:     $CLEAN_FEATURES/index.clean_only.json"
