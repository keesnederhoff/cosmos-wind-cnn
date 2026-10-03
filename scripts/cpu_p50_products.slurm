#!/bin/bash
#SBATCH --job-name=p50prod
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --account=pcmsc
#SBATCH --wckey=reimbursable
#SBATCH --time=24:00:00
#SBATCH --output=p50prod_%j.log

# P50 (u10, v10, wind_speed) products, raw and BA, for the 1940-2026 record.
# Usage: sbatch scripts/cpu_p50_products.slurm [args for make_p50_products.py]
#        smoke: sbatch --time=01:00:00 scripts/cpu_p50_products.slurm --max-files 1
set -uo pipefail
export PYTHONUNBUFFERED=1
source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd /home/cnederhoff/cosmos/cosmos-wind-cnn
python scripts/make_p50_products.py "$@"
echo "p50prod exit: $?"
