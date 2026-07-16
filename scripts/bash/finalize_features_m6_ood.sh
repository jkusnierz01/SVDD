#!/bin/bash
#SBATCH -A plgdfsingingpwr-cpu
#SBATCH -p plgrid
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=logs_finalize_m6_%j.txt

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/big_storage/SVDD}"
FEATURES_DIR="${FEATURES_DIR:-/net/people/plgrid/plgjedrzejkusnierz/scratch/data/M6/features_clean}"

cd "$PROJECT_ROOT"
source .venv/bin/activate

python scripts/processing/rebuild_benchmark_features_index.py --features-dir "$FEATURES_DIR"
python scripts/processing/validate_ood_feature_index.py \
  --features-dir "$FEATURES_DIR" \
  --dataset m6 \
  --check-clam \
  --check-singgraph

echo "Finalize done."
