#!/usr/bin/env bash
# SF Bay met-product validation on Caldera (wind-only). 2020 era split:
#   Era A 2011-2019  IEM+NDBC              9 products (incl. CONUS404)
#   Era B 2020-2025  IEM+NDBC+USGS(4)      8 products (CONUS404 record ends 2021)
# Products = the 5 bias-corrected CNN twins + CNN-allvars raw (raw/BC control pair)
# + ERA5 / RTMA-SFbay baselines. USGS moorings stay at MEASURED anemometer height
# (1.2 m / 4.93 m vs 10 m model) -- models read high there by construction, so USGS
# is reported as its own group and kept OUT of the headline pooled score.
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

MODELS_A="ERA5,CONUS404,RTMA-SFbay,CNN-allvars,CNN-allvars-BC,CNN-windonly-BC,CNN-extreme-BC,CNN-wave-p2-BC,CNN-wave-p3-BC"
MODELS_B="ERA5,RTMA-SFbay,CNN-allvars,CNN-allvars-BC,CNN-windonly-BC,CNN-extreme-BC,CNN-wave-p2-BC,CNN-wave-p3-BC"

echo "================ ERA A (2011-2019, IEM+NDBC) ================"
VAL_ERA=A VAL_MODELS="$MODELS_A" python run_validation.py

echo "================ ERA B (2020-2025, IEM+NDBC+USGS) ================"
VAL_ERA=B VAL_USGS=1 VAL_MODELS="$MODELS_B" python run_validation.py

echo "================ ranking / combined skill ================"
python analysis/rank_products.py || echo "rank_products failed (non-fatal)"

# Headline: IEM+NDBC only -- USGS excluded so the pooled number is not distorted
# by 4 short records measured at 1.2 m.
VAL_WEIGHTS="IEM:1,NDBC:1" VAL_RANK_DIR=rankings \
    python analysis/combined_skill.py || echo "combined_skill (headline) failed (non-fatal)"

# Second pass with USGS folded in at equal weight, for comparison only.
VAL_WEIGHTS="IEM:1,NDBC:1,USGS:1" VAL_RANK_DIR=rankings_withUSGS \
    python analysis/combined_skill.py || echo "combined_skill (withUSGS) failed (non-fatal)"

echo "================ validation done ================"
