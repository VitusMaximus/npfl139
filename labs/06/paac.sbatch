#!/bin/bash
#SBATCH --job-name=paac
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
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

python -u labs/06/paac.py \
    --threads "${SLURM_CPUS_PER_TASK:-8}" \
    --envs 16 \
    "$@"
