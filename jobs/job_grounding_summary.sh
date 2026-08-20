#!/bin/bash
#SBATCH --job-name=ground_summary
#SBATCH --output=Logs/ground_summary_%j.out
#SBATCH --error=Logs/ground_summary_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=cpu-short

SPLIT="${1:?usage: job_grounding_summary.sh SPLIT}"

echo "Job started: $(date)"

module purge
module load ALICE/default
module load Miniconda3/24.7.1-0

source "$SLURM_SUBMIT_DIR/env.sh"

source /easybuild/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
conda activate base
conda activate "$ENV_SELECTION"

cd "$REPO_ROOT"

python -u -m src.grounding.summary --split "$SPLIT"

echo "Job finished: $(date)"