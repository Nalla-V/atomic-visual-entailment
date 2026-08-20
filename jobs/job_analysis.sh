#!/bin/bash
#SBATCH --job-name=analysis
#SBATCH --output=Logs/analysis_%j.out
#SBATCH --error=Logs/analysis_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=cpu-short

OUTPUT_NAME="${1:-AVE_train_learned_selector_v3_rebuild}"

echo "Job started: $(date)"

module purge
module load ALICE/default
module load Miniconda3/24.7.1-0

source "$SLURM_SUBMIT_DIR/env.sh"

source /easybuild/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
conda activate base
conda activate "$ENV_SELECTION"

cd "$REPO_ROOT"

python -u -m src.selection.analysis --output-name "$OUTPUT_NAME"

echo "Job finished: $(date)"