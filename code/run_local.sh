#!/usr/bin/env bash
# Local launcher for long DramaMatrix runs.
# Example:
# nohup code/run_local.sh --project-id Drama_20260803_001 --resume > logs/dramamatrix_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ENV_NAME="${DRAMAMATRIX_CONDA_ENV:-dramamatrix}"
DRAMAMATRIX_PROXY_URL="${DRAMAMATRIX_PROXY_URL:-http://127.0.0.1:7890}"

export HTTP_PROXY="$DRAMAMATRIX_PROXY_URL"
export HTTPS_PROXY="$DRAMAMATRIX_PROXY_URL"
export ALL_PROXY="$DRAMAMATRIX_PROXY_URL"
export http_proxy="$DRAMAMATRIX_PROXY_URL"
export https_proxy="$DRAMAMATRIX_PROXY_URL"
export all_proxy="$DRAMAMATRIX_PROXY_URL"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,::1}"

cd "$PROJECT_ROOT"
mkdir -p logs

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV_NAME"
fi

echo "=========================================================="
echo "Starting DramaMatrix"
echo "Launcher: local"
echo "Conda environment: ${CONDA_DEFAULT_ENV:-not activated}"
echo "Proxy: $DRAMAMATRIX_PROXY_URL"
echo "Date: $(date)"
echo "=========================================================="

cd code
python -u main.py "$@"

echo "=========================================================="
echo "Job completed at: $(date)"
echo "=========================================================="
