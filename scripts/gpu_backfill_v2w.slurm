#!/bin/bash
#SBATCH --job-name=bf_v2w
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=200G
#SBATCH --account=pcmsc
#SBATCH --time=4:00:00
#SBATCH --array=0-5%2
#SBATCH --output=sweep_logs/%x_%a_%A.log

set -euo pipefail
ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay
MANIFEST=/home/cnederhoff/cosmos/cosmos-wind-cnn/backfill_v2w_manifest.txt

LINE=$(grep -vE '^\s*#|^\s*$' "$MANIFEST" | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
read RN BC START END <<< "$LINE"
echo "BACKFILL(v2w) idx=$SLURM_ARRAY_TASK_ID run=$RN bc=$BC  $START .. $END"

# v2w_res_bc8 = wind-only + residual ON. The archived training.yaml has the
# FILE defaults (residual_learning:false, nonwind_weight:1.0), so the true
# settings MUST be replayed as SWEEP_* env or the model is built wrong
# (residual off => silently wrong inference output).
export SWEEP_BASE_CHANNELS=$BC
export SWEEP_SEQ_LEN=6
export SWEEP_DROPOUT=0.0
export SWEEP_RESIDUAL=1
export SWEEP_NONWIND_WEIGHT=0.0
export COSMOS_DATA_ROOT=$ROOT
export COSMOS_RESULTS_ROOT=$ROOT
export OMP_NUM_THREADS=18
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd /home/cnederhoff/cosmos/cosmos-wind-cnn

srun python scripts/run_training_pipeline.py \
    --case-study case_studies/sf_bay_rtma \
    --run-name "$RN" \
    --skip-preprocess --skip-train --skip-eval \
    --inference-start "$START" --inference-end "$END" \
    --gpus 1 --num-workers 8
echo "backfill(v2w) $RN $START..$END exit: $?"
