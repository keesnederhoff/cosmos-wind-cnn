#!/bin/bash
#SBATCH --job-name=stage1940
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --account=pcmsc
#SBATCH --wckey=reimbursable
#SBATCH --time=04:00:00
#SBATCH --output=stage1940_%j.log

# Stage the continuous 1940 -> 2026-08-10 ERA5 inference inputs (u, v, cloud,
# filled gust) and prove them against the 2026-08-27 inputs. ~5 GB in RAM per
# variable -> compute node, not a login node.
set -uo pipefail
export PYTHONUNBUFFERED=1
source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd /home/cnederhoff/cosmos/cosmos-wind-cnn
python scripts/stage_era5_1940.py
echo "stage1940 exit: $?"
