#!/bin/bash
#SBATCH --job-name=walker-hardcore
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/home/babkavi/npfl139/slurm/%x-%j-f.out
#SBATCH --error=/home/babkavi/npfl139/slurm/%x-%j-f.err

set -euo pipefail

PROJECT_DIR="/home/babkavi/npfl139"
VENV_DIR="$PROJECT_DIR/VENV_DIR"

mkdir -p "$PROJECT_DIR/slurm"
cd "$PROJECT_DIR"

source "$VENV_DIR/bin/activate"
export PYTHONUNBUFFERED=1

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

python labs/08/walker_hardcore.py \
    --render_each=0 \
    "$@"
