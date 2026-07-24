#!/usr/bin/env bash
# SF Bay met-product validation on Caldera (wind-only): ERA5, CONUS404, RTMA-SFbay,
# and the 3 CNN winners vs IEM+NDBC(+CWOP) obs. Runs Era 2 then Era 3.
#   sbatch run_validation_caldera.slurm
#SBATCH --job-name=val_met
#SBATCH --partition=cpu
#SBATCH --account=pcmsc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=24:00:00
#SBATCH --output=val_met_%j.log
set -euo pipefail

source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
export KMP_DUPLICATE_LIB_OK=TRUE

BUNDLE=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/validation
export COSMOS_VALIDATION_DATA_ROOT="$BUNDLE"
export COSMOS_VALIDATION_OUTPUT_ROOT="$BUNDLE/results"
mkdir -p "$COSMOS_VALIDATION_OUTPUT_ROOT"
echo "DATA_ROOT=$COSMOS_VALIDATION_DATA_ROOT  OUTPUT_ROOT=$COSMOS_VALIDATION_OUTPUT_ROOT"

cd /home/cnederhoff/cosmos/cosmos-wind-cnn/validation

echo "================ ERA 2 (2011-2021) ================"
VAL_ERA=2 VAL_MODELS="ERA5,CONUS404,RTMA-SFbay,CNN-allvars,CNN-windonly,CNN-extreme" python run_validation.py

echo "================ ERA 3 (2022-present) ================"
VAL_ERA=3 VAL_MODELS="ERA5,RTMA-SFbay,CNN-allvars,CNN-windonly,CNN-extreme" python run_validation.py

echo "================ ranking / combined skill ================"
python analysis/rank_products.py   || echo "rank_products failed (non-fatal)"
python analysis/combined_skill.py  || echo "combined_skill failed (non-fatal)"
echo "================ validation done ================"
