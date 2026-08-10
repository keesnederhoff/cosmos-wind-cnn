#!/bin/bash
#SBATCH --job-name=longrec
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --account=pcmsc
#SBATCH --time=8:00:00
#SBATCH --array=0-11%2
#SBATCH --output=sweep_logs/%x_%a_%A.log

# Phase 7: the longest record this model can produce, one calendar year per array
# task, for the seen-vs-unseen era comparison.
#
# WHY 2014 AND NOT 2000: lr_gust (ERA5 wind gust) starts 2013-12-31 19:00, and it
# is the low_res partner of the `gust` variable pair, so it lands in input_vars
# for EVERY arm -- there is no trained checkpoint that can run without it.
# ERA5 10 m u/v then caps the far end at 2025-12-31 23:00. Reaching further back
# means retraining with no gust target.
#
# WHY ONE YEAR PER TASK, IN A FRESH PROCESS: the lazy da.interp graph grows with
# the SPAN of the request, which is what OOM-killed the 87-year attempt at chunk 9.
# Segmenting also parallelises what would otherwise be a ~5 h serial job.
#
# SEGMENT OVERLAP: sequence_length=6 means the first 5 hours of any segment cannot
# be predicted. Each task therefore starts 2 days early and the overlap is dropped
# at concat time -- without it the record would carry a 5-hour hole at every year
# boundary. 2014 is the exception: it can only start at 2013-12-31, which is
# exactly enough history to make 2014-01-01 00:00 the first predicted hour.
#
# OUTPUT SUBSET: P50 (skill) + P90 (peaks). The full 19+9 grid would be ~173 GB
# per arm over 12 years; this is ~35 GB. Calibration work uses the existing
# full-resolution test-window file, not this one.

set -euo pipefail
ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay
RN=${RN:-qh_P0_s1}

YEAR=$((2014 + SLURM_ARRAY_TASK_ID))
if [ "$YEAR" -eq 2014 ]; then
  START=2013-12-31
else
  START=$((YEAR-1))-12-30
fi
END=$((YEAR+1))-01-01
echo "LONGREC task=$SLURM_ARRAY_TASK_ID run=$RN year=$YEAR window=$START..$END"

export INFER_QUANTILES=0.5,0.9

# Replay the P0 quantile config -- mandatory, the archived training.yaml holds
# file defaults rather than per-run overrides.
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
    --skip-preprocess --skip-train --skip-eval \
    --inference-start "$START" --inference-end "$END"
echo "longrec $RN $YEAR exit: $?"
