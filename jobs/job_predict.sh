#!/bin/bash
#SBATCH --job-name=predict
#SBATCH --output=Logs/predict_%j.out
#SBATCH --error=Logs/predict_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100.4g.40gb|A100.3g.40gb"

VLM="${1:?usage: job_predict.sh VLM SPLIT METHOD [LIMIT]}"
SPLIT="${2:?usage: job_predict.sh VLM SPLIT METHOD [LIMIT]}"
METHOD="${3:?usage: job_predict.sh VLM SPLIT METHOD [LIMIT]}"
LIMIT="${4:-}"

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

echo "Python: $(which python)"
echo "vlm=$VLM split=$SPLIT limit=$LIMIT method=$METHOD"

ARGS="--vlm $VLM --split $SPLIT --method $METHOD"
if [ -n "$LIMIT" ]; then
    ARGS="$ARGS --limit $LIMIT"
fi

python -u -m src.prediction.predict $ARGS

echo "Job finished: $(date)"