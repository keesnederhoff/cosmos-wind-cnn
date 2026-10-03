#!/bin/bash
#SBATCH --job-name=ba_fit
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=pcmsc
#SBATCH --wckey=reimbursable
#SBATCH --time=01:00:00
#SBATCH --output=ba_fit_%j.log

# Observation-based factor correction (BA): station factors at the 90th
# percentile, class constants, Gaussian factor map, leave-one-out checks.
# Usage: sbatch scripts/cpu_ba_fit.slurm [extra args for ba_fit_factor.py]
set -uo pipefail
export PYTHONUNBUFFERED=1
source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd /home/cnederhoff/cosmos/cosmos-wind-cnn
export COSMOS_VALIDATION_DATA_ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/validation
export COSMOS_VALIDATION_OUTPUT_ROOT=$COSMOS_VALIDATION_DATA_ROOT
python scripts/ba_fit_factor.py "$@"
echo "ba_fit exit: $?"
