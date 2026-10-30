#!/bin/bash
#SBATCH --job-name=sfb_swrr
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --account=pcmsc
#SBATCH --time=2-00:00:00
#SBATCH --array=2,4,5,6,7,8,9,10,11,12,16,18%2
#SBATCH --output=sweep_logs/%x_%a_%A.log

set -euo pipefail
ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay
SHARED=$ROOT/sf_bay_rtma/results/3737874/data_processed
MANIFEST=/home/cnederhoff/cosmos/cosmos-wind-cnn/sweep_manifest.txt

LINE=$(grep -vE '^\s*#|^\s*$' "$MANIFEST" | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
read RN BC SL DO LR <<< "$LINE"
echo "RERUN idx=$SLURM_ARRAY_TASK_ID run=$RN base_channels=$BC seq_len=$SL dropout=$DO lr=$LR (exclusive node)"

export SWEEP_BASE_CHANNELS=$BC
export SWEEP_SEQ_LEN=$SL
export SWEEP_DROPOUT=$DO
export SWEEP_LR=$LR
export COSMOS_DATA_ROOT=$ROOT
export COSMOS_RESULTS_ROOT=$ROOT
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29700 + SLURM_ARRAY_TASK_ID))
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=10
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RD=$ROOT/sf_bay_rtma/results/$RN
mkdir -p "$RD/data_processed"
ln -sfn "$SHARED/memmap" "$RD/data_processed/memmap"
ln -sfn "$SHARED/normalization_stats.pkl" "$RD/data_processed/normalization_stats.pkl"
ln -sfn "$SHARED/target_grid_reference.nc" "$RD/data_processed/target_grid_reference.nc"
echo "$RN base_channels=$BC seq_len=$SL dropout=$DO lr=$LR" > "$RD/sweep_params.txt"

source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd /home/cnederhoff/cosmos/cosmos-wind-cnn
which python
srun python scripts/run_training_pipeline.py \
    --case-study case_studies/sf_bay_rtma \
    --run-name $RN \
    --gpus 2 --num-workers 8 \
    --skip-preprocess --skip-inference --skip-eval
echo "variant $RN exit: $?"
