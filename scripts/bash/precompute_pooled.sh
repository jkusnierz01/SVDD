#!/bin/bash
#SBATCH --job-name=precompute_pooled
#SBATCH --account=plgdfsingingpwr-cpu
#SBATCH --partition=plgrid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=logs_precompute_pooled.txt
# One-time precompute: concat w2v+mert [6000,2048] -> AvgPool -> [1500,2048] fp16
# Reduces per-epoch I/O from ~4TB to ~500GB (8x improvement)

set -e

INDEX="/net/people/plgrid/plgjedrzejkusnierz/scratch/data/MoM/mom_preprocessed/index.json"
OUTPUT_DIR="/net/people/plgrid/plgjedrzejkusnierz/scratch/data/MoM/preprocessed_pooled_fp16"
WORKERS=16
PROJECT_ROOT="$HOME/big_storage/SVDD"

echo "Project root: $PROJECT_ROOT"
echo "Index:        $INDEX"
echo "Output dir:   $OUTPUT_DIR"
echo "Workers:      $WORKERS"
echo ""

cd "$PROJECT_ROOT"
source .venv/bin/activate

python scripts/processing/precompute_pooled.py \
    --index      "$INDEX" \
    --output_dir "$OUTPUT_DIR" \
    --workers    "$WORKERS"

echo ""
echo "Job done."
