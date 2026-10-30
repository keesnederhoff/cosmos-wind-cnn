#!/bin/bash
#SBATCH --job-name=cosmos_win_ev2
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=kees.nederhoff@deltares-usa.us
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=200G
#SBATCH --account=pcmsc
#SBATCH --time=8:00:00
#SBATCH --output=gpu_rtma_winner2_%j.log

set -euo pipefail
export COSMOS_DATA_ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay
export COSMOS_RESULTS_ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay
export SWEEP_BASE_CHANNELS=24
export SWEEP_SEQ_LEN=6
export SWEEP_DROPOUT=0.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=18
RUN_NAME=sw_bc24do0

echo "Winner eval (RTMA overlap window only): $RUN_NAME, inference 2011-01-01..2026-01-01"
source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd /home/cnederhoff/cosmos/cosmos-wind-cnn

srun python scripts/run_training_pipeline.py \
    --case-study case_studies/sf_bay_rtma \
    --run-name $RUN_NAME \
    --skip-preprocess --skip-train \
    --inference-start 2011-01-01 --inference-end 2026-01-01 \
    --gpus 1 --num-workers 8
