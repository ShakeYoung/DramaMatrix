#!/usr/bin/env bash
# Submit from the repository root with:
# mkdir -p logs && sbatch code/run_cluster.sh --project-id Drama_20260307_001 --resume
# Adjust commented Slurm directives for the target cluster before submission.
#SBATCH --job-name=dramamatrix
#SBATCH --output=logs/dramamatrix_%j.log
#SBATCH --error=logs/dramamatrix_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
##SBATCH --partition=<your_partition>
##SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
CONDA_ENV_NAME="${DRAMAMATRIX_CONDA_ENV:-dramamatrix}"

cd "$PROJECT_ROOT"
mkdir -p logs

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda was not found on PATH. Load the cluster's Anaconda/Miniconda module first."
  exit 1
fi

# This works in non-interactive Slurm shells where `conda activate` is otherwise unavailable.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

echo "=========================================================="
echo "Starting DramaMatrix"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "Conda environment: $CONDA_ENV_NAME"
echo "Date: $(date)"
echo "=========================================================="

cd code
python -u main.py "$@"

echo "=========================================================="
echo "Job completed at: $(date)"
echo "=========================================================="
