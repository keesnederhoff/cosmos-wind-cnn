#!/bin/bash
#SBATCH --job-name=cosmos_rtma_infer
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=kees.nederhoff@deltares-usa.us
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=0
#SBATCH --account=pcmsc
#SBATCH --time=10:00:00
#SBATCH --output=gpu_rtma_infer_%j.log

set -euo pipefail

export COSMOS_DATA_ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/cosmos-wind-cnn/data
export COSMOS_RESULTS_ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/cosmos-wind-cnn/results
RUN_NAME=3732177
LOG_DIR="${COSMOS_RESULTS_ROOT}/sf_bay_rtma/${RUN_NAME}/logs"
trap 'mkdir -p "$LOG_DIR" && cp "${SLURM_SUBMIT_DIR}/gpu_rtma_infer_${SLURM_JOB_ID}.log" "$LOG_DIR/" 2>/dev/null || true' EXIT

export OMP_NUM_THREADS=18
echo "Resuming inference+eval for run $RUN_NAME (chunked inference, full record)"

source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd /home/cnederhoff/cosmos/cosmos-wind-cnn

srun python scripts/run_training_pipeline.py \
    --case-study case_studies/sf_bay_rtma \
    --run-name $RUN_NAME \
    --skip-preprocess --skip-train \
    --gpus 1 --num-workers 8
