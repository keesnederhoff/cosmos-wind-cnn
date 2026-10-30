#!/bin/bash
#SBATCH --job-name=cosmos_rtma_eval
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --account=pcmsc
#SBATCH --time=04:00:00
#SBATCH --output=cpu_rtma_eval_%j.log
set -uo pipefail
export COSMOS_DATA_ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/cosmos-wind-cnn
export COSMOS_RESULTS_ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/cosmos-wind-cnn
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
RUN_NAME=${1:-3732197}
echo "eval-only for run $RUN_NAME"
source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd /home/cnederhoff/cosmos/cosmos-wind-cnn
python scripts/run_training_pipeline.py \
    --case-study case_studies/sf_bay_rtma \
    --run-name $RUN_NAME \
    --skip-preprocess --skip-train --skip-inference \
    --gpus 0 --eval-points 100
