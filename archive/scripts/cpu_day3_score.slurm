#!/bin/bash
#SBATCH --job-name=d3score
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --account=pcmsc
#SBATCH --time=06:00:00
#SBATCH --output=sweep_logs/d3score_%j.log
set -e
cd /home/cnederhoff/cosmos/cosmos-wind-cnn
source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
python scripts/day3_score.py
