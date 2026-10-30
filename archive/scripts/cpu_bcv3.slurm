#!/bin/bash
#SBATCH --job-name=bcv3
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --account=pcmsc
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=sweep_logs/bcv3_%a_%A.log
# Phase 6b: does the post-hoc quantile map still earn its place now that the
# model predicts a distribution? Map is fitted on VAL (these arms select at
# epoch 0-1, so train-period residuals misrepresent out-of-sample behaviour).
set -e
cd /home/cnederhoff/cosmos/cosmos-wind-cnn
source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
RUNS=(qh_P0_s1 qh_P0_s2 qh_P0_s3)
RN=${RUNS[$SLURM_ARRAY_TASK_ID]}
echo "BC v3 test: $RN"
python scripts/bc_v3_test.py --run "$RN"
