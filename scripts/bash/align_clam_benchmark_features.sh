#!/bin/bash
#SBATCH -A plgdfsingingpwr-cpu
#SBATCH -p plgrid
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --job-name=align_clam
#SBATCH --output=logs_align_clam_%j.txt
# Post-hoc align CLAM MERT time axis to W2V (CPU only, no GPU).
# Usage: sbatch scripts/bash/align_clam_benchmark_features.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/big_storage/SVDD}"
FEATURES_DIR="${FEATURES_DIR:-/net/people/plgrid/plgjedrzejkusnierz/scratch/data/Sonics/benchmark_10pct/features}"
WORKERS="${WORKERS:-16}"

echo "Project root:  $PROJECT_ROOT"
echo "Features dir:  $FEATURES_DIR"
echo "Workers:       $WORKERS"
echo "Node:          ${SLURMD_NODENAME:-local}"
echo ""

cd "$PROJECT_ROOT"
source .venv/bin/activate

python scripts/processing/align_clam_benchmark_features.py \
    --features-dir "$FEATURES_DIR" \
    --workers "$WORKERS" \
    --skip-existing

echo ""
echo "Rebuilding index.json..."
python scripts/processing/rebuild_benchmark_features_index.py \
    --features-dir "$FEATURES_DIR"

echo ""
echo "Job done."
