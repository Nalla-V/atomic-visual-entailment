#!/bin/bash
#SBATCH --job-name=train_selector
#SBATCH --output=Logs/train_selector_%j.out
#SBATCH --error=Logs/train_selector_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --partition=cpu-short

OUTPUT_NAME="${1:-AVE_train_learned_selector_v3_rebuild}"

echo "========================================"
echo "Job started: $(date)"
echo "Node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

module purge
module load ALICE/default
module load Miniconda3/24.7.1-0

source "$SLURM_SUBMIT_DIR/env.sh"

source /easybuild/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
conda activate base
conda activate "$ENV_SELECTION"

cd "$REPO_ROOT"

echo "Python: $(which python)"
echo "DATA_ROOT: $DATA_ROOT"
echo "Output name: $OUTPUT_NAME"

python -u -m src.selection.train --output-name "$OUTPUT_NAME"

echo "Job finished: $(date)"