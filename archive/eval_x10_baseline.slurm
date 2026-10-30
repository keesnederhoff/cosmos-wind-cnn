#!/bin/bash
#SBATCH --job-name=x10base
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --account=pcmsc
#SBATCH --time=3:00:00
#SBATCH --output=sweep_logs/%x_%A.log
set -euo pipefail
ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay
export COSMOS_DATA_ROOT=$ROOT COSMOS_RESULTS_ROOT=$ROOT
source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd /home/cnederhoff/cosmos/cosmos-wind-cnn
for RN in os_wo_bc24_base_res_s1 os_wo_bc24_base_res_s2 os_wo_bc24_base_res_s3; do
  echo "==== re-eval $RN (>10 m/s baseline) ===="
  python scripts/run_training_pipeline.py --case-study case_studies/sf_bay_rtma \
     --run-name $RN --skip-preprocess --skip-train --skip-inference --gpus 1 || echo "  $RN eval failed"
done
