#!/bin/bash
#SBATCH --job-name=seedrep_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=18
#SBATCH --account=pcmsc
#SBATCH --time=6:00:00
#SBATCH --array=0-7%2
#SBATCH --output=sweep_logs/%x_%a_%A.log
set -euo pipefail
ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay
MANIFEST=/home/cnederhoff/cosmos/cosmos-wind-cnn/seedrep_manifest.txt
LINE=$(grep -vE '^\s*#|^\s*$' "$MANIFEST" | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
read RN NWW <<< "$LINE"
echo "SEEDREP EVAL idx=$SLURM_ARRAY_TASK_ID run=$RN"
# eval scores WIND skill regardless of training loss; model is bc8/ctrl/8-channel
export SWEEP_BASE_CHANNELS=8
export SWEEP_SEQ_LEN=6
export SWEEP_DROPOUT=0.0
export SWEEP_RESIDUAL=0
export SWEEP_ADD_INPUTS=lr_cloud,static_terrain
export COSMOS_DATA_ROOT=$ROOT
export COSMOS_RESULTS_ROOT=$ROOT
export OMP_NUM_THREADS=18
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd /home/cnederhoff/cosmos/cosmos-wind-cnn
srun python scripts/run_training_pipeline.py \
    --case-study case_studies/sf_bay_rtma --run-name $RN \
    --skip-preprocess --skip-train \
    --inference-start 2011-01-01 --inference-end 2026-01-01 \
    --gpus 1 --num-workers 8
