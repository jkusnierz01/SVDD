#!/bin/bash
#SBATCH -A plgdfsingingpwr-gpu
#SBATCH -p plgrid-gpu-v100
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=logs_eval_%j.txt
#
# Usage:
#   sbatch run_eval.sh <experiment>
#
# Examples:
#   sbatch run_eval.sh eval_sonics
#   sbatch run_eval.sh eval_m6
#   sbatch run_eval.sh eval_mom

set -e

EXPERIMENT="${1:?Usage: sbatch run_eval.sh <experiment>}"

module load python/3.11 cuda/12.1 cudnn/8.9.2 ffmpeg

cd $HOME/big_storage/SVDD

source .venv/bin/activate

export WANDB_API_KEY="17832e6bf7fcebf19c988a6a2d3f6d5f706b6ba3"

wandb login

echo "Experiment: $EXPERIMENT"
echo "Start eval..."

HYDRA_FULL_ERROR=1 uv run python src/eval.py +experiments="$EXPERIMENT"

echo "Eval done."