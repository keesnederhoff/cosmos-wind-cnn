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
