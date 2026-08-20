#!/bin/bash
#SBATCH --job-name=voting
#SBATCH --output=Logs/voting_%j.out
#SBATCH --error=Logs/voting_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=00:30:00
#SBATCH --partition=cpu-short

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

python -u -m src.selection.voting

echo "Job finished: $(date)"