#!/bin/bash
#SBATCH --job-name=ground_dino
#SBATCH --output=Logs/ground_dino_%j.out
#SBATCH --error=Logs/ground_dino_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1

SPLIT="${1:?usage: job_grounding_detect.sh SPLIT [EVAL_NAME] [LIMIT]}"
LIMIT="${2:-}"

echo "Job started: $(date)"

module purge
module load ALICE/default
module load CUDA/12.4.0
module load Miniconda3/24.7.1-0

source "$SLURM_SUBMIT_DIR/env.sh"

source /easybuild/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
conda activate base
conda activate "$ENV_GROUNDING"

cd "$REPO_ROOT"

ARGS="--split $SPLIT --eval-name $EVAL_NAME"
if [ -n "$LIMIT" ]; then
    ARGS="$ARGS --limit $LIMIT"
fi

python -u -m src.grounding.detect $ARGS

echo "Job finished: $(date)"