#!/bin/bash
#SBATCH --job-name=x10
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --account=pcmsc
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3%2
#SBATCH --output=sweep_logs/%x_%a_%A.log

set -euo pipefail
ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay
SHARED=$ROOT/sf_bay_rtma/results/v2data/data_processed
MANIFEST=/home/cnederhoff/cosmos/cosmos-wind-cnn/x10_manifest.txt

LINE=$(grep -vE '^\s*#|^\s*$' "$MANIFEST" | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
read RN BC RES NW DELTA ADD <<< "$LINE"
echo "X10 idx=$SLURM_ARRAY_TASK_ID run=$RN bc=$BC residual=$RES nw=$NW delta=$DELTA add=$ADD"

export SWEEP_BASE_CHANNELS=$BC
export SWEEP_SEQ_LEN=6
export SWEEP_DROPOUT=0.0
export SWEEP_LR=0.0003
export SWEEP_RESIDUAL=$RES
export SWEEP_NONWIND_WEIGHT=$NW
export SWEEP_ADD_INPUTS=$ADD
export SWEEP_EXTREME_WEIGHT=$DELTA
export SWEEP_EXTREME_THRESH=10.0
export COSMOS_DATA_ROOT=$ROOT
export COSMOS_RESULTS_ROOT=$ROOT
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((28000 + SLURM_ARRAY_TASK_ID))
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=10
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RD=$ROOT/sf_bay_rtma/results/$RN
mkdir -p "$RD/data_processed"
for f in memmap normalization_stats.pkl target_grid_reference.nc train.nc val.nc test.nc; do
  ln -sfn "$SHARED/$f" "$RD/data_processed/$f"
done
echo "$RN bc=$BC res=$RES nw=$NW delta=$DELTA add=$ADD sl6 do0.0 lr3e-4" > "$RD/arm_params.txt"

source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd /home/cnederhoff/cosmos/cosmos-wind-cnn
srun python scripts/run_training_pipeline.py \
    --case-study case_studies/sf_bay_rtma \
    --run-name $RN \
    --gpus 2 --num-workers 8 \
    --skip-preprocess \
    --inference-start 2011-01-01 --inference-end 2026-01-01
echo "x10 $RN exit: $?"
