#!/bin/bash
#SBATCH --job-name=download_m6
#SBATCH --account=plgdfsingingpwr-cpu
#SBATCH --partition=plgrid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=04:00:00
#SBATCH --output=logs_download_m6.txt

# Jeśli masz jakiegoś virtualenva lub moduł Pythona, załaduj go tutaj
module load python/3.11

cd $HOME/big_storage/SVDD
source .venv/bin/activate

echo "Starting download process..."
uv run python scripts/download/m6.py
echo "Download finished!"