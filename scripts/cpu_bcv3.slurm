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
cd /home/cnederhoff/cosmos/cosmos-wind-cnn
source ~/.bashrc
conda activate cosmos-wind-cnn 2>/dev/null || true
RUNS=(qh_P0_s1 qh_P0_s2 qh_P0_s3)
RN=${RUNS[$SLURM_ARRAY_TASK_ID]}
echo "BC v3 test: $RN"
python scripts/bc_v3_test.py --run "$RN"
echo "bcv3 $RN exit: $?"
