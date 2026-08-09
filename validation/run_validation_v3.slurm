#!/usr/bin/env bash
# Phase 5B: obs-validation of all 21 v3 campaign arms over the HELD-OUT TEST
# WINDOW (2025-02-06 -> 2025-12-31), against IEM + NDBC stations.
#
# Why this matters more than the grid-point track: grid-point skill is scored
# against RTMA, which is the field the CNN was TRAINED TO IMITATE. Station
# observations are the only truth here that is not the training target.
#
# References alongside the 21 arms: ERA5 (the input, i.e. the skill floor),
# RTMA-SFbay (the training target, i.e. the ceiling), and CNN-wave-p3-BC (the
# v2 production pick) for continuity with the published results.
#
# USGS moorings are reported as their OWN group and kept OUT of the headline
# pooled score: they sit at 1.2-4.93 m anemometer height against a 10 m model
# field, and pooling them previously cost every product 0.03-0.25 skill.
#
#   sbatch [--dependency=afterany:<evaljobs>] run_validation_v3.slurm
#SBATCH --job-name=val_v3
#SBATCH --partition=cpu
#SBATCH --account=pcmsc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=24:00:00
#SBATCH --output=val_v3_%j.log
set -euo pipefail

source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
export KMP_DUPLICATE_LIB_OK=TRUE

BUNDLE=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/validation
export COSMOS_VALIDATION_DATA_ROOT="$BUNDLE"
export COSMOS_VALIDATION_OUTPUT_ROOT="$BUNDLE/results"
mkdir -p "$COSMOS_VALIDATION_OUTPUT_ROOT"

RES=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/sf_bay_rtma_v3/results
FILE=full_record_ERA5_20250206_20260101.nc

# --- Fail loudly if any arm's inference is missing -------------------------
# A product whose file does not resolve is dropped SILENTLY by the engine's path
# audit -- it simply vanishes from the leaderboard. That has bitten this project
# three times, so check here rather than discovering a 17-product run later.
echo "=== inference file audit ==="
MISSING=0
for b in 0 1 2 3 4 5; do
  for s in 1 2 3; do
    p="$RES/qh_P${b}_s${s}/output_inference/$FILE"
    [ -f "$p" ] || { echo "  MISSING qh_P${b}_s${s}"; MISSING=$((MISSING+1)); }
  done
done
for s in 1 2 3; do
  p="$RES/c1_det_P0_s${s}/output_inference/$FILE"
  [ -f "$p" ] || { echo "  MISSING c1_det_P0_s${s}"; MISSING=$((MISSING+1)); }
done
if [ "$MISSING" -gt 0 ]; then
  echo "ABORT: $MISSING of 21 inference files absent -- not running a partial leaderboard."
  exit 1
fi
echo "  all 21 present"

cd /home/cnederhoff/cosmos/cosmos-wind-cnn/validation

echo "================ V3 test window (2025-02-06 -> 2025-12-31) ================"
echo "--- headline: IEM + NDBC pooled ---"
VAL_ERA=V3 VAL_WEIGHTS="IEM:1,NDBC:1" VAL_RANK_DIR=rankings python run_validation.py

echo "--- USGS moorings as a separate group ---"
VAL_ERA=V3 VAL_USGS=1 VAL_ONLY_GROUPS=USGS VAL_OUTDIR_SUFFIX=_usgs \
  python run_validation.py || echo "USGS group failed (non-fatal, coverage may be thin in 2025)"

echo "================ combined skill / ranking ================"
VAL_WEIGHTS="IEM:1,NDBC:1" VAL_RANK_DIR=rankings \
  python analysis/combined_skill.py || echo "combined_skill failed (non-fatal)"

echo "DONE"
