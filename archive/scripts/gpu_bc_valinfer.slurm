#!/bin/bash
#SBATCH --job-name=bcinfer
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --account=pcmsc
#SBATCH --time=8:00:00
#SBATCH --array=0-2%2
#SBATCH --output=sweep_logs/%x_%a_%A.log

# Phase 6b: VAL-window inference for the three P0 arms, so a bias-correction
# quantile map can be fitted somewhere that is neither train nor test.
#
# WHY VAL AND NOT TRAIN (a deliberate departure from v2):
#   bias_correct.py fits on the TRAIN period. For v2 that was defensible. Here it
#   is not: these arms select their best checkpoint at epoch 0-1, so the model has
#   barely fitted the train period at all, and its train-period error distribution
#   does not represent its behaviour out of sample. A map fitted there would be
#   calibrated to the wrong error distribution. VAL is out-of-sample for the
#   weights, disjoint from TEST, and 7891 steps is ample for a 200-point per-cell
#   empirical CDF.
#
# WHY IT IS NEEDED AT ALL: the existing v3 inference covers ONLY the test window,
# so bias_correct.py's train-period xr.align(join='inner') would return ZERO
# overlapping samples. np.nanquantile over an empty axis yields NaN maps, and the
# script would write a silently-garbage BC file rather than fail. Do not point the
# stock script at a test-only run.
#
# VAL window = 2024-03-14 09:00 -> 2025-02-06 03:00 (7891 steps, ~13 GB/arm).

set -euo pipefail
ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay

INF_START=2024-03-14
INF_END=2025-02-06

RUNS=(qh_P0_s1 qh_P0_s2 qh_P0_s3)
RN=${RUNS[$SLURM_ARRAY_TASK_ID]}
echo "BCINFER idx=$SLURM_ARRAY_TASK_ID run=$RN window=$INF_START..$INF_END"

# Replay the P0 quantile config. Mandatory: the archived training.yaml holds file
# defaults, not per-run overrides, so without this the checkpoint is rebuilt as a
# deterministic model and loads the wrong head.
export SWEEP_HEAD=quantile
export SWEEP_QW_EXP=2.0
export SWEEP_GUST_WEIGHT=0.2
export SWEEP_ADD_INPUTS=lr_cloud,static_terrain
export SWEEP_BASE_CHANNELS=24
export SWEEP_SEQ_LEN=6
export SWEEP_DROPOUT=0.0
export SWEEP_LR=0.0003
export SWEEP_NONWIND_WEIGHT=0.0

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
echo "bcinfer $RN exit: $?"
