#!/bin/bash
#SBATCH --job-name=d3infer
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --account=pcmsc
#SBATCH --time=6:00:00
#SBATCH --array=0-6%3
#SBATCH --output=sweep_logs/%x_%a_%A.log

# Day 3: inference for the fixed shipping recipe on the held-out test window.
# See day3_manifest.txt for the recipe and why each row exists.
#
# --exclusive --mem=0 is NOT optional: 9 of 21 arms in the 2026-07 sweep died of
# GPU co-tenancy OOM on non-exclusive backfill.
#
# The SWEEP_* replay below is mandatory. The archived training.yaml holds file
# defaults, not the per-run overrides, so without it a quantile checkpoint gets
# rebuilt as a deterministic model and load_state_dict fails (or worse, doesn't).

set -euo pipefail
ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay
MANIFEST=/home/cnederhoff/cosmos/cosmos-wind-cnn/day3_manifest.txt

INF_START=2025-02-06
INF_END=2026-01-01

LINE=$(grep -vE '^\s*#|^\s*$' "$MANIFEST" | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
read RN HEAD QW GUSTW DO WD CKPT QSUB ADD <<< "$LINE"
if [ -z "$ADD" ]; then echo "FATAL: manifest line $SLURM_ARRAY_TASK_ID parsed short: '$LINE'"; exit 1; fi

CKPT_PATH=$ROOT/sf_bay_rtma_v3/results/$RN/checkpoint/$CKPT
if [ ! -f "$CKPT_PATH" ]; then echo "FATAL: no checkpoint $CKPT_PATH"; exit 1; fi

echo "D3INFER idx=$SLURM_ARRAY_TASK_ID run=$RN ckpt=$CKPT qsub=$QSUB window=$INF_START..$INF_END"

export SWEEP_HEAD=$HEAD
export SWEEP_QW_EXP=$QW
export SWEEP_GUST_WEIGHT=$GUSTW
export SWEEP_DROPOUT=$DO
export SWEEP_WEIGHT_DECAY=$WD
export SWEEP_ADD_INPUTS=$ADD
export SWEEP_BASE_CHANNELS=24
export SWEEP_SEQ_LEN=6
export SWEEP_LR=0.0003
export SWEEP_NONWIND_WEIGHT=0.0

export INFER_CHECKPOINT=$CKPT
if [ "$QSUB" != "-" ]; then export INFER_QUANTILES=$QSUB; else unset INFER_QUANTILES || true; fi

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
    --skip-preprocess \
    --skip-train \
    --inference-start "$INF_START" --inference-end "$INF_END"
echo "d3infer $RN/$CKPT exit: $?"
