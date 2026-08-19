#!/bin/bash
#SBATCH --job-name=decompose
#SBATCH --output=Logs/decompose_%j.out
#SBATCH --error=Logs/decompose_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1

MODEL="${1:-qwen32}"
SPLIT="${2:-dev}"
LIMIT="${3:-}"

echo "========================================"
echo "Job started: $(date)"
echo "Node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

module purge
module load ALICE/default
module load CUDA/12.4.0
module load Miniconda3/24.7.1-0

source "$SLURM_SUBMIT_DIR/env.sh"

source /easybuild/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
conda activate base
conda activate decompose_env

cd "$REPO_ROOT"

echo "Python: $(which python)"
echo "model=$MODEL split=$SPLIT limit=$LIMIT"

echo "Running decomposition..."
if [ -n "$LIMIT" ]; then
    python -u -m src.decomposition.decompose --model "$MODEL" --split "$SPLIT" --limit "$LIMIT"
else
    python -u -m src.decomposition.decompose --model "$MODEL" --split "$SPLIT"
fi

echo "Job finished: $(date)"