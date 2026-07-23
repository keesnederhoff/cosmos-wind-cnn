#!/bin/bash
#SBATCH --job-name=v2data_prep
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=300G
#SBATCH --account=pcmsc
#SBATCH --time=12:00:00
#SBATCH --output=sweep_logs/%x_%A.log

set -euo pipefail
ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay
export COSMOS_DATA_ROOT=$ROOT
export COSMOS_RESULTS_ROOT=$ROOT
export OMP_NUM_THREADS=24

echo "v2data preprocess: rebuilding train/val/test.nc + stats WITH the 5 derived statics"
source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd /home/cnederhoff/cosmos/cosmos-wind-cnn

# Preprocess only. Statics have no time dim so they are excluded from the
# common-time intersection -> the train/val/test split is unchanged vs 3737874.
python scripts/run_training_pipeline.py \
    --case-study case_studies/sf_bay_rtma \
    --run-name v2data \
    --skip-train --skip-inference --skip-eval

echo "--- converting to memmap ---"
python scripts/convert_to_memmap.py \
    --data-processed $ROOT/sf_bay_rtma/results/v2data/data_processed
echo "v2data DONE"
