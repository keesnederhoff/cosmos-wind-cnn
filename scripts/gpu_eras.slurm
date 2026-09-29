#!/bin/bash
#SBATCH --job-name=eras
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --account=pcmsc
#SBATCH --wckey=reimbursable
#SBATCH --time=4:00:00
#SBATCH --array=0-80%3
#SBATCH --output=sweep_logs/%x_%a_%A.log

# Three-era inference, 2000-2026, 3 seeds. See era_manifest.txt.
#
# REDUCED QUANTILE SET, and every level is an ACTUAL grid level. The 19-level
# midpoint grid tops out at tau=0.9737 -- there is NO true P99, and asking for
# 0.95/0.99 silently collapses both onto that one level. Requesting real grid
# values keeps the written levels honest about what the model produces.
#
# THREE KNOBS ARE ENV-OVERRIDABLE, and their defaults reproduce the 2026-08-14
# product exactly. Set them to re-run against an extended ERA5 record without
# forking this script:
#   ERAS_MANIFEST  window list (default era_manifest.txt, ends 2026-07-27)
#   ERAS_CONFIG    inference preprocessing config
#   ERAS_CKPT      checkpoint; also becomes the output filename prefix
# The era product is SPEED-TRACK ONLY -- best_speed.pth writes
# speed_full_record_*.nc, which is what validation/config.py globs. There are no
# direction_ year files and the validation never asks for any.
#
# Output location (both opt-in; unset = write into <run>/output_inference/):
#   ERAS_OUT_DIR   write the year file to $ERAS_OUT_DIR/<run>/ instead. Needed
#                  whenever a window is ALREADY ARCHIVED -- same name, and the
#                  archived file would be silently overwritten.
#   ERAS_POINTS=1  station-points mode (1940 seed selection): the year grid goes
#                  to a scratch dir, scripts/extract_station_points.py samples it
#                  at the 45 validation stations into $ERAS_POINTS_DIR, and the
#                  grid is deleted ONLY if extraction (and, where an archived
#                  file of the same window exists, the reproduction check
#                  against it) succeeded. A failed task keeps its grid.

set -euo pipefail
ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay
REPO=/home/cnederhoff/cosmos/cosmos-wind-cnn
MANIFEST=${ERAS_MANIFEST:-$REPO/scripts/era_manifest.txt}

LINE=$(grep -vE '^\s*#|^\s*$' "$MANIFEST" | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
read RN INF_START INF_END <<< "$LINE"
if [ -z "$INF_END" ]; then echo "FATAL: manifest line $SLURM_ARRAY_TASK_ID parsed short: '$LINE'"; exit 1; fi

CKPT=${ERAS_CKPT:-best_speed.pth}
if [ ! -f "$ROOT/sf_bay_rtma_v3/results/$RN/checkpoint/$CKPT" ]; then
  echo "FATAL: no checkpoint for $RN"; exit 1
fi
echo "ERAS idx=$SLURM_ARRAY_TASK_ID run=$RN window=$INF_START..$INF_END"
echo "ERAS manifest=$MANIFEST"

# Replay the FIXED RECIPE -- the archived training.yaml holds file defaults only.
export SWEEP_HEAD=quantile
export SWEEP_QW_EXP=1.0
export SWEEP_GUST_WEIGHT=0.2
export SWEEP_DROPOUT=0.1
export SWEEP_WEIGHT_DECAY=0.001
export SWEEP_ADD_INPUTS=lr_cloud,static_terrain
export SWEEP_BASE_CHANNELS=24
export SWEEP_SEQ_LEN=6
export SWEEP_LR=0.0003
export SWEEP_NONWIND_WEIGHT=0.0

export INFER_CHECKPOINT=$CKPT
export INFER_QUANTILES=0.1316,0.5,0.8158,0.9211,0.9737
# P0-ONLY inference config. The archived one is a superset of all predictor
# blocks and step_inference intersects the time axes of EVERY declared source
# before selecting this arm's five -- its pressure-level files start 2015, so
# without this every year before 2015 aborts on 'No overlapping timesteps'.
export INFER_CONFIG=${ERAS_CONFIG:-$REPO/case_studies/sf_bay_rtma_v3/configs/inference_preprocessing_long.yaml}
echo "ERAS config=$INFER_CONFIG ckpt=$CKPT"

OUT_NAME="${CKPT%.pth}"; OUT_NAME="${OUT_NAME#best_}_"
[ "$CKPT" = "best_model.pth" ] && OUT_NAME=""
OUT_NAME="${OUT_NAME}full_record_ERA5_${INF_START//-/}_${INF_END//-/}.nc"
ARCHIVED="$ROOT/sf_bay_rtma_v3/results/$RN/output_inference/$OUT_NAME"
POINTS=${ERAS_POINTS:-0}
if [ "$POINTS" = "1" ]; then
  PTS_DIR=${ERAS_POINTS_DIR:-$ROOT/sf_bay_rtma_v3/results/v1940_points}
  export INFER_OUT_DIR="$PTS_DIR/_scratch/${RN}_${INF_START//-/}"
elif [ -n "${ERAS_OUT_DIR:-}" ]; then
  export INFER_OUT_DIR="$ERAS_OUT_DIR/$RN"
fi
echo "ERAS out=${INFER_OUT_DIR:-<run>/output_inference} file=$OUT_NAME points=$POINTS"

export COSMOS_DATA_ROOT=$ROOT
export COSMOS_RESULTS_ROOT=$ROOT
export OMP_NUM_THREADS=10
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd /home/cnederhoff/cosmos/cosmos-wind-cnn

srun python scripts/run_training_pipeline.py \
    --case-study case_studies/sf_bay_rtma_v3 \
    --run-name "$RN" \
    --gpus 1 --num-workers 8 \
    --skip-preprocess --skip-train --skip-eval \
    --inference-start "$INF_START" --inference-end "$INF_END"
echo "eras $RN $INF_START exit: $?"

if [ "$POINTS" = "1" ]; then
  GRID="$INFER_OUT_DIR/$OUT_NAME"
  REF_ARGS=()
  [ -f "$ARCHIVED" ] && REF_ARGS=(--ref "$ARCHIVED")
  python scripts/extract_station_points.py --grid "$GRID" \
      --out "$PTS_DIR/$RN/points_${OUT_NAME}" ${REF_ARGS[@]+"${REF_ARGS[@]}"}
  # Reached only if extraction exited 0 (set -e): the grid is disposable now.
  rm -f "$GRID"
  rmdir "$INFER_OUT_DIR" 2>/dev/null || true
  echo "eras points $RN $INF_START done"
fi
