#!/bin/bash
#SBATCH --job-name=ground_phrases
#SBATCH --output=Logs/ground_phrases_%j.out
#SBATCH --error=Logs/ground_phrases_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100.4g.40gb|A100.3g.40gb"

SPLIT="${1:?usage: job_grounding_phrases.sh SPLIT [EVAL_NAME] [LIMIT]}"
LIMIT="${2:-}"

echo "Job started: $(date)"

module purge
module load ALICE/default
module load CUDA/12.4.0
module load Miniconda3/24.7.1-0

source "$SLURM_SUBMIT_DIR/env.sh"

source /easybuild/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
conda activate base
conda activate "$ENV_DECOMPOSE"

cd "$REPO_ROOT"

ARGS="--split $SPLIT --eval-name $EVAL_NAME"
if [ -n "$LIMIT" ]; then
    ARGS="$ARGS --limit $LIMIT"
fi

python -u -m src.grounding.phrases $ARGS

echo "Job finished: $(date)"