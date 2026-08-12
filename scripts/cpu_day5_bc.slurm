#!/bin/bash
#SBATCH --job-name=d5bc
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --account=pcmsc
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=sweep_logs/d5bc_%a_%A.log

# Day 5: does the post-hoc quantile map still earn its place now the recipe is
# properly trained? Phase 6b measured BC buying +0.074 at >10 m/s on the
# epoch-1 arms; that is the baseline this must be reported against.
#
# The day-3 calibration diagnosis makes a falsifiable prediction here. The PIT
# is NOT U-shaped -- so this is not a dispersion failure, which BC could not fix
# -- but is right-skewed in storms, with the nominal 99th percentile sitting at
# the empirical 96th and bias_gt10 = -0.61 m/s. That is a tail LOCATION bias,
# exactly what a quantile map corrects, so the gain should NOT have collapsed.
# If it has, the day-3 diagnosis is wrong and needs revisiting before day 6.
set -e
cd /home/cnederhoff/cosmos/cosmos-wind-cnn
source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
RUNS=(r1_do010 r1b_do010_s2 r1b_do010_s3)
RN=${RUNS[$SLURM_ARRAY_TASK_ID]}
echo "BC day5: $RN (best_speed.pth)"
python scripts/bc_v3_test.py --run "$RN" --prefix speed_
