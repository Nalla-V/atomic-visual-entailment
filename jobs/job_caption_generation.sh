#!/bin/bash
#SBATCH --job-name=caption_gen
#SBATCH --output=Logs/caption_gen_%j.out
#SBATCH --error=Logs/caption_gen_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1

LIMIT="${1:-}"

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
conda activate "$ENV_REFINEMENT"

cd "$REPO_ROOT"

echo "limit=$LIMIT"

ARGS=""
if [ -n "$LIMIT" ]; then
    ARGS="--limit $LIMIT"
fi

python -u ablations/refinement/caption_generation.py $ARGS

echo "Job finished: $(date)"