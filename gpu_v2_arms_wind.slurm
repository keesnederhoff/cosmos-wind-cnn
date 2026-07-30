#!/bin/bash
#SBATCH --job-name=v2w_arm
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --account=pcmsc
#SBATCH --time=2-00:00:00
#SBATCH --array=0-7%2
#SBATCH --output=sweep_logs/%x_%a_%A.log

set -euo pipefail
ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay
SHARED=$ROOT/sf_bay_rtma/results/v2data/data_processed
MANIFEST=/home/cnederhoff/cosmos/cosmos-wind-cnn/v2_arms_wind.txt

LINE=$(grep -vE '^\s*#|^\s*$' "$MANIFEST" | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
read RN BC RES ADD <<< "$LINE"
echo "ARM idx=$SLURM_ARRAY_TASK_ID run=$RN bc=$BC residual=$RES add_inputs=$ADD"

# Winning hyperparameters held fixed across all arms so the only difference is
# the feature under test.
export SWEEP_BASE_CHANNELS=$BC
export SWEEP_SEQ_LEN=6
export SWEEP_DROPOUT=0.0
export SWEEP_LR=0.0003
export SWEEP_RESIDUAL=$RES
export SWEEP_ADD_INPUTS=$ADD
export SWEEP_NONWIND_WEIGHT=0.0   # GOAL 2: wind-only optimization
export COSMOS_DATA_ROOT=$ROOT
export COSMOS_RESULTS_ROOT=$ROOT
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29900 + SLURM_ARRAY_TASK_ID))
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=10
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Share the one preprocessed dataset across all arms (no data duplication).
# train/val/test.nc are symlinked too: the memmap dataset does not read them, but
# step_evaluate needs them present or it SILENTLY skips evaluation.
RD=$ROOT/sf_bay_rtma/results/$RN
mkdir -p "$RD/data_processed"
for f in memmap normalization_stats.pkl target_grid_reference.nc train.nc val.nc test.nc; do
  ln -sfn "$SHARED/$f" "$RD/data_processed/$f"
done
echo "$RN base_channels=$BC residual=$RES add_inputs=$ADD seq_len=6 dropout=0.0 lr=0.0003" > "$RD/arm_params.txt"

source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd /home/cnederhoff/cosmos/cosmos-wind-cnn
srun python scripts/run_training_pipeline.py \
    --case-study case_studies/sf_bay_rtma \
    --run-name $RN \
    --gpus 2 --num-workers 8 \
    --skip-preprocess --skip-inference --skip-eval
echo "arm $RN exit: $?"
