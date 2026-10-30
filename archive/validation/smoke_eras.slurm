#!/bin/bash
#SBATCH --job-name=smoke_eras
#SBATCH --partition=cpu
#SBATCH --account=pcmsc
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=100G
#SBATCH --time=1:00:00
#SBATCH --output=smoke_eras_%j.log
set -euo pipefail
source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
export KMP_DUPLICATE_LIB_OK=TRUE
BUNDLE=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/validation
export COSMOS_VALIDATION_DATA_ROOT="$BUNDLE"
export COSMOS_VALIDATION_OUTPUT_ROOT="$BUNDLE/results"
cd /home/cnederhoff/cosmos/cosmos-wind-cnn/validation
echo "--- smoke: era E2, one arm + ERA5, throwaway outdir ---"
VAL_ERA=E2 VAL_MODELS="V3-ERAS-s1,ERA5" VAL_WEIGHTS="IEM:1,NDBC:1" \
  VAL_OUTDIR_SUFFIX=_smoke VAL_RANK_DIR=rankings_smoke python run_validation.py
echo "SMOKE DONE"
