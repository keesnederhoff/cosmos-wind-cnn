#!/bin/bash
#SBATCH --job-name=v2w_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --account=pcmsc
#SBATCH --time=6:00:00
#SBATCH --array=0-7%2
#SBATCH --output=sweep_logs/%x_%a_%A.log

set -euo pipefail
ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay
MANIFEST=/home/cnederhoff/cosmos/cosmos-wind-cnn/v2_arms_wind.txt

LINE=$(grep -vE '^\s*#|^\s*$' "$MANIFEST" | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
read RN BC RES ADD <<< "$LINE"
echo "EVAL arm idx=$SLURM_ARRAY_TASK_ID run=$RN bc=$BC residual=$RES add_inputs=$ADD"

# CRITICAL: step_inference reads the ARCHIVED training.yaml, which still carries
# the file defaults (residual_learning: false, baseline additional_inputs). The
# arm's env MUST be replayed here or inference would build the wrong model --
# loudly for ADD_INPUTS (shape mismatch), SILENTLY for RESIDUAL.
export SWEEP_BASE_CHANNELS=$BC
export SWEEP_SEQ_LEN=6
export SWEEP_DROPOUT=0.0
export SWEEP_RESIDUAL=$RES
export SWEEP_ADD_INPUTS=$ADD
export COSMOS_DATA_ROOT=$ROOT
export COSMOS_RESULTS_ROOT=$ROOT
export OMP_NUM_THREADS=18
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd /home/cnederhoff/cosmos/cosmos-wind-cnn

# 2011-2026 only: the eval window, and it avoids the full-record inference leak.
srun python scripts/run_training_pipeline.py \
    --case-study case_studies/sf_bay_rtma \
    --run-name $RN \
    --skip-preprocess --skip-train \
    --inference-start 2011-01-01 --inference-end 2026-01-01 \
    --gpus 1 --num-workers 8
