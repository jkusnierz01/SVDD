#!/bin/bash
#SBATCH --job-name=download_m6
#SBATCH --account=plgdfsingingpwr-cpu
#SBATCH --partition=plgrid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=24:00:00
#SBATCH --output=logs_download_mom.txt

module load python/3.11 ffmpeg

export PYTHONUNBUFFERED=1

cd $HOME/big_storage/SVDD
source .venv/bin/activate

echo "Starting download process..."
uv run python scripts/download/mom.py
echo "Download finished!"