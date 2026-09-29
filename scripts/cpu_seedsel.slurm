#!/bin/bash
#SBATCH --job-name=seedsel
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --account=pcmsc
#SBATCH --wckey=reimbursable
#SBATCH --time=04:00:00
#SBATCH --output=seedsel_%j.log

# Seed selection for the 1940-2026 v3 product from station point files.
set -uo pipefail
export PYTHONUNBUFFERED=1
export COSMOS_VALIDATION_DATA_ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/validation
export COSMOS_VALIDATION_OUTPUT_ROOT=$COSMOS_VALIDATION_DATA_ROOT
source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd /home/cnederhoff/cosmos/cosmos-wind-cnn
python scripts/select_seed_points.py
echo "seedsel exit: $?"