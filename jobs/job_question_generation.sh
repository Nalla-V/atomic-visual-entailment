#!/bin/bash
#SBATCH --job-name=question_gen
#SBATCH --output=Logs/question_gen_%j.out
#SBATCH --error=Logs/question_gen_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100.4g.40gb|A100.3g.40gb"

SPLIT="${1:-dev}"

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
conda activate "$ENV_PREDICT"

cd "$REPO_ROOT"

echo "split=$SPLIT"

python -u ablations/refinement/question_generation.py --split "$SPLIT"

echo "Job finished: $(date)"