#!/bin/bash
#SBATCH --job-name=combine40
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --account=pcmsc
#SBATCH --wckey=reimbursable
#SBATCH --time=48:00:00
#SBATCH --output=combine40_%j.log

# One NetCDF per parameter (speed quantiles raw + BC) for the 1940-2026 record.
# Usage: sbatch cpu_combine_full_record.slurm <seed> [extra args]
set -uo pipefail
export PYTHONUNBUFFERED=1
source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd /home/cnederhoff/cosmos/cosmos-wind-cnn
python scripts/combine_full_record.py --seed "$1" "${@:2}"
echo "combine40 exit: $?"