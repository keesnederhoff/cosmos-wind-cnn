#!/bin/bash
#SBATCH --job-name=qheval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --account=pcmsc
#SBATCH --time=8:00:00
#SBATCH --array=0-20%3
#SBATCH --output=sweep_logs/%x_%a_%A.log

# Phase 5A: inference + grid-point evaluation for all 21 campaign arms, on the
# HELD-OUT TEST WINDOW.
#
# Why the test window and not val: every twCRPS number reported during the
# campaign came from the val split, which is exactly what best_model.pth was
# selected on -- so those are optimistically biased. Test (2025-02-06 04:00 ->
# 2025-12-31 23:00, 7892 h) has never been touched.
#
# Why this run exists at all: the campaign's deterministic control reports
# twCRPS on Z-SCORED outputs while the quantile arms report it in physical m/s,
# so the two are not on the same scale and the head comparison is currently
# unanswerable. Scoring from the inference NetCDF puts every arm through one
# code path in physical units.
#
# The SWEEP_* env below is REPLAYED from the manifest and is mandatory: the
# archived training.yaml holds file defaults, not per-run overrides, so without
# it a quantile checkpoint would be rebuilt as a deterministic model.
#
# NOTE the full 19-level quantile grid is written deliberately (INFER_QUANTILES
# unset). CRPS/PIT/coverage need the dense grid -- a reduced set is for the
# shipped product, not for scoring. Costs ~12 GB/arm, ~250 GB total.

set -euo pipefail
ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay
MANIFEST=/home/cnederhoff/cosmos/cosmos-wind-cnn/qhead_manifest.txt

INF_START=2025-02-06
INF_END=2026-01-01

LINE=$(grep -vE '^\s*#|^\s*$' "$MANIFEST" | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
read RN HEAD QW GUSTW ADD <<< "$LINE"
echo "QHEVAL idx=$SLURM_ARRAY_TASK_ID run=$RN head=$HEAD window=$INF_START..$INF_END"

export SWEEP_HEAD=$HEAD
export SWEEP_QW_EXP=$QW
export SWEEP_GUST_WEIGHT=$GUSTW
export SWEEP_ADD_INPUTS=$ADD
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

# Inference + evaluation only; the checkpoint and the shared data store already exist.
srun python scripts/run_training_pipeline.py \
    --case-study case_studies/sf_bay_rtma_v3 \
    --run-name "$RN" \
    --gpus 1 --num-workers 8 \
    --skip-preprocess \
    --skip-train \
    --inference-start "$INF_START" --inference-end "$INF_END"
echo "eval $RN exit: $?"
