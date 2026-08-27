#!/bin/bash
#SBATCH --job-name=ver_eras
#SBATCH --partition=cpu
#SBATCH --account=pcmsc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=2:00:00
#SBATCH --output=ver_eras_%j.log
set -euo pipefail

# The era files are ~5.3 GB each and reading two of them times out on the login
# node, so the verification runs here instead. Args are passed straight through:
#   sbatch scripts/cpu_verify_eras.slurm --compare
#   sbatch scripts/cpu_verify_eras.slurm --audit

source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
export PYTHONUNBUFFERED=1        # these logs are block-buffered otherwise
cd /home/cnederhoff/cosmos/cosmos-wind-cnn
python scripts/verify_era_output.py "$@"
echo "verify exit: $?"
