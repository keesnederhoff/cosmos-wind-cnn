#!/bin/bash
#SBATCH --job-name=eracmp
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --account=pcmsc
#SBATCH --time=04:00:00
#SBATCH --output=sweep_logs/eracmp_%j.log
# Phase 7: seen (2020-2025) vs unseen (2014-2019) skill for the long record.
set -e
cd /home/cnederhoff/cosmos/cosmos-wind-cnn
source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
python scripts/era_compare.py --run "${RN:-qh_P0_s1}" --n-points "${NPTS:-200}"
