#!/bin/bash
#SBATCH --job-name=eval_selector
#SBATCH --output=Logs/eval_selector_%j.out
#SBATCH --error=Logs/eval_selector_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=cpu-short

SPLIT="${1:?usage: job_evaluate_selector.sh SPLIT [TRAIN_RUN_NAME]}"
TRAIN_RUN="${2:-AVE_train_learned_selector_v3_rebuild}"

echo "Job started: $(date)"

module purge
module load ALICE/default
module load Miniconda3/24.7.1-0

source "$SLURM_SUBMIT_DIR/env.sh"

source /easybuild/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
conda activate base
conda activate "$ENV_SELECTION"

cd "$REPO_ROOT"

python -u -m src.selection.evaluate --split "$SPLIT" \
    --train-run-name "$TRAIN_RUN" \
    --output-name "AVE_learned_selection_evaluation_v3_rebuild"

echo "Job finished: $(date)"