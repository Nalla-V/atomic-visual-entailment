#!/bin/bash
#SBATCH --job-name=results
#SBATCH --output=Logs/results_%j.out
#SBATCH --error=Logs/results_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=cpu-short

SCRIPT="${1:?usage: job_results.sh SCRIPT [ARGS...]}"
shift

echo "Job started: $(date)"

module purge
module load ALICE/default
module load Miniconda3/24.7.1-0

source "$SLURM_SUBMIT_DIR/env.sh"

source /easybuild/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
conda activate base
conda activate "$ENV_SELECTION"

cd "$REPO_ROOT"

echo "Script: $SCRIPT"
echo "Args  : $*"

python -u -m "src.results.$SCRIPT" "$@"

echo "Job finished: $(date)"