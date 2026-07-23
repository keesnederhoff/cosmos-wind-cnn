#!/bin/bash
#SBATCH --job-name=sfb_fullrec
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=200G
#SBATCH --account=pcmsc
#SBATCH --time=4:00:00
#SBATCH --array=0-21%6
#SBATCH --output=sweep_logs/%x_%a_%A.log

set -euo pipefail
export COSMOS_DATA_ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay
export COSMOS_RESULTS_ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay
export SWEEP_BASE_CHANNELS=24
export SWEEP_SEQ_LEN=6
export SWEEP_DROPOUT=0.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=18

YS=$((1940 + 4*SLURM_ARRAY_TASK_ID))
YE=$((YS + 4)); if [ "$YE" -gt 2027 ]; then YE=2027; fi
START="${YS}-01-01"; END="${YE}-01-01"
echo "FULLREC segment idx=$SLURM_ARRAY_TASK_ID  $START .. $END"

source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd /home/cnederhoff/cosmos/cosmos-wind-cnn
srun python scripts/run_training_pipeline.py \
    --case-study case_studies/sf_bay_rtma \
    --run-name sw_bc24do0 \
    --skip-preprocess --skip-train --skip-eval \
    --inference-start "$START" --inference-end "$END" \
    --gpus 1 --num-workers 8
