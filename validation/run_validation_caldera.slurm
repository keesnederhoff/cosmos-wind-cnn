#!/usr/bin/env bash
# SF Bay met-product validation on Caldera (wind-only). Runs Era 2 then Era 3.
# Set the two data/output roots below (or pass via --export), then:
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

# --- data/output roots (EDIT to the staged bundle location on Caldera) ------
: "${COSMOS_VALIDATION_DATA_ROOT:=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/validation}"
: "${COSMOS_VALIDATION_OUTPUT_ROOT:=${COSMOS_VALIDATION_DATA_ROOT}/results}"
export COSMOS_VALIDATION_DATA_ROOT COSMOS_VALIDATION_OUTPUT_ROOT
mkdir -p "$COSMOS_VALIDATION_OUTPUT_ROOT"

echo "DATA_ROOT   = $COSMOS_VALIDATION_DATA_ROOT"
echo "OUTPUT_ROOT = $COSMOS_VALIDATION_OUTPUT_ROOT"

cd /home/cnederhoff/cosmos/cosmos-wind-cnn/validation
for ERA in 2 3; do
  echo "================ ERA $ERA ================"
  VAL_ERA="$ERA" python run_validation.py
done
echo "================ validation done ================"
