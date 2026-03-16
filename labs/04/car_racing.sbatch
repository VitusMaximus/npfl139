#!/bin/bash
#SBATCH --job-name=car-racing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/vitek/School/npfl139/slurm/%x-%j.out
#SBATCH --error=/home/vitek/School/npfl139/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR="/home/vitek/School/npfl139"
VENV_DIR="$PROJECT_DIR/VENV_DIR"

mkdir -p "$PROJECT_DIR/slurm"
cd "$PROJECT_DIR"

source "$VENV_DIR/bin/activate"
export PYTHONUNBUFFERED=1

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

python labs/04/car_racing.py \
    --threads "${SLURM_CPUS_PER_TASK:-8}" \
    --num_envs 16 \
    --batch_size 64 \
    --downsample 2 \
    "$@"
