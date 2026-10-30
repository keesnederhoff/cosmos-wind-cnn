#!/bin/bash
#SBATCH --job-name=d5valinf
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --account=pcmsc
#SBATCH --time=6:00:00
#SBATCH --array=0-2%3
#SBATCH --output=sweep_logs/%x_%a_%A.log

# Day 5 prerequisite: VAL-window inference for the three recipe seeds at
# best_speed.pth, so the bias-correction quantile map can be fitted somewhere
# that is neither train nor test.
#
# WHY VAL AND NOT TRAIN: kept from Phase 6b. The map must be calibrated to the
# model's out-of-sample error distribution; the train period is in-sample for
# the weights and misrepresents it.
#
# WHY THIS RUN IS NEEDED AT ALL: the day-3 inference covers ONLY the test
# window, so bc_v3_test.py's xr.align(join='inner') on the val period would
# return ZERO overlapping samples. It refuses below 1000 rather than writing
# all-NaN maps, so this would abort loudly -- but it would still abort.
#
# Three distinct run dirs, so unlike day 3 these can safely run concurrently.

set -euo pipefail
ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay

INF_START=2024-03-14
INF_END=2025-02-06

RUNS=(r1_do010 r1b_do010_s2 r1b_do010_s3)
RN=${RUNS[$SLURM_ARRAY_TASK_ID]}
echo "D5VALINF idx=$SLURM_ARRAY_TASK_ID run=$RN ckpt=best_speed.pth window=$INF_START..$INF_END"

# Replay the FIXED RECIPE. Mandatory: the archived training.yaml holds file
# defaults, not the per-run overrides.
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

export INFER_CHECKPOINT=best_speed.pth

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
    --skip-eval \
    --inference-start "$INF_START" --inference-end "$INF_END"
echo "d5valinf $RN exit: $?"
