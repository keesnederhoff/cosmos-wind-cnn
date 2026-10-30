#!/bin/bash
#SBATCH --job-name=v4_eras
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
#SBATCH --output=/home/cnederhoff/cosmos/cosmos-wind-cnn/v4_logs/%x_%A_%a.log

# v4 full-record inference, one manifest line per array task. Derived from gpu_eras.slurm.
# The array range, manifest and mode come from the submitter (submit_v4_chain.sh):
#   ERAS_MANIFEST   era_manifest_v4_{points,chosen,bcwin}.txt   (columns: RN START END)
#                   RN may be CHOSEN -> resolved from results/chosen_seed.txt at RUN time
#   ERAS_CKPT       best_speed.pth (default) | best_direction.pth
#   ERAS_POINTS=1   station-points mode (step 1): year grid -> scratch, sampled at the 45
#                   stations by extract_station_points.py, grid deleted ONLY if extraction
#                   succeeded. A failed task keeps its grid.
#   ERAS_OUT_DIR    write the year file to $ERAS_OUT_DIR/<run>/ (step 2 grids)
#   neither         file lands in <run>/output_inference/ (BC val/test windows)
# Written quantile levels: V4_QLEVELS = nearest actual grid levels of the 51-level head.
set -euo pipefail
source /home/cnederhoff/cosmos/cosmos-wind-cnn/scripts/v4_env.sh
MANIFEST=${ERAS_MANIFEST:?ERAS_MANIFEST not set}
CKPT=${ERAS_CKPT:-best_speed.pth}
POINTS=${ERAS_POINTS:-0}
v4_begin "eras_${CKPT%.pth}_${SLURM_ARRAY_TASK_ID}"

LINE=$(grep -vE '^\s*#|^\s*$' "$MANIFEST" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
read RN INF_START INF_END <<< "$LINE"
[ -n "${INF_END:-}" ] || { echo "FATAL: manifest line $SLURM_ARRAY_TASK_ID parsed short: '$LINE'"; exit 1; }
[ "$RN" = CHOSEN ] && RN=$(v4_chosen)

CKDIR=$V4RES/$RN/checkpoint
[ -f "$CKDIR/$CKPT" ] || { echo "FATAL: no $CKPT for $RN in $CKDIR"; exit 1; }
echo "ERAS idx=$SLURM_ARRAY_TASK_ID run=$RN window=$INF_START..$INF_END ckpt=$CKPT points=$POINTS"

# Replay the FIXED RECIPE. dropout / Q are also in the archived training.yaml in v4, but
# replaying them keeps this script correct for a checkpoint trained with a stale yaml.
export SWEEP_HEAD=quantile SWEEP_N_QUANTILES=51 SWEEP_QW_EXP=1.0 SWEEP_GUST_WEIGHT=0.2
export SWEEP_DROPOUT=0.1 SWEEP_WEIGHT_DECAY=0.001 SWEEP_ADD_INPUTS=lr_cloud,static_terrain
export SWEEP_BASE_CHANNELS=24 SWEEP_SEQ_LEN=6 SWEEP_LR=0.0003 SWEEP_NONWIND_WEIGHT=0.0
export INFER_CHECKPOINT=$CKPT
export INFER_QUANTILES=$V4_QLEVELS
export INFER_CONFIG=$REPO/case_studies/sf_bay_rtma_v4/configs/inference_preprocessing_1940.yaml

OUT_NAME="${CKPT%.pth}"; OUT_NAME="${OUT_NAME#best_}_"
[ "$CKPT" = "best_model.pth" ] && OUT_NAME=""
OUT_NAME="${OUT_NAME}full_record_ERA5_${INF_START//-/}_${INF_END//-/}.nc"
if [ "$POINTS" = "1" ]; then
  PTS_DIR=$V4RES/v1940_points
  mkdir -p "$PTS_DIR/$RN"
  if [ -s "$PTS_DIR/$RN/points_$OUT_NAME" ]; then echo "points file exists -> skip"; exit 0; fi
  export INFER_OUT_DIR="$PTS_DIR/_scratch/${RN}_${INF_START//-/}"
elif [ -n "${ERAS_OUT_DIR:-}" ]; then
  export INFER_OUT_DIR="$ERAS_OUT_DIR/$RN"
fi
echo "ERAS out=${INFER_OUT_DIR:-<run>/output_inference} file=$OUT_NAME"
[ "$POINTS" = "1" ] || v4_quota_guard 1.0

export OMP_NUM_THREADS=10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun python scripts/run_training_pipeline.py \
    --case-study case_studies/sf_bay_rtma_v4 --run-name "$RN" \
    --gpus 1 --num-workers 8 \
    --skip-preprocess --skip-train --skip-eval \
    --inference-start "$INF_START" --inference-end "$INF_END"

if [ "$POINTS" = "1" ]; then
  GRID="$INFER_OUT_DIR/$OUT_NAME"
  [ -s "$GRID" ] || { echo "FATAL: inference wrote no $GRID"; exit 1; }
  python scripts/extract_station_points.py --grid "$GRID" --out "$PTS_DIR/$RN/points_$OUT_NAME"
  # reached only if extraction exited 0 (set -e): the grid is disposable now
  rm -f "$GRID"
  rmdir "$INFER_OUT_DIR" 2>/dev/null || true
  echo "points $RN $INF_START done"
else
  OUTF="${INFER_OUT_DIR:-$V4RES/$RN/output_inference}/$OUT_NAME"
  [ -s "$OUTF" ] || { echo "FATAL: expected output missing: $OUTF"; exit 1; }
  ls -la "$OUTF"
fi
