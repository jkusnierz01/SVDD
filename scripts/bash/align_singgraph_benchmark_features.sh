#!/bin/bash
#SBATCH -A plgdfsingingpwr-cpu
#SBATCH -p plgrid
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --job-name=align_singgraph
#SBATCH --output=logs_align_singgraph_%j.txt
# Post-hoc align SingGraph instrumental (MERT) to vocals (W2V) time axis (CPU only).
# Usage: sbatch scripts/bash/align_singgraph_benchmark_features.sh

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

python scripts/processing/align_singgraph_benchmark_features.py \
    --features-dir "$FEATURES_DIR" \
    --workers "$WORKERS" \
    --skip-existing

echo ""
echo "Rebuilding index.json..."
python scripts/processing/rebuild_benchmark_features_index.py \
    --features-dir "$FEATURES_DIR"

echo ""
echo "Job done."
