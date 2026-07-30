#!/bin/bash
#SBATCH --job-name=cosmos_memmap_conv
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --account=pcmsc
#SBATCH --time=04:00:00
#SBATCH --output=cpu_memmap_conv_%j.log

set -euo pipefail
source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd /home/cnederhoff/cosmos/cosmos-wind-cnn
DP=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/sf_bay_rtma/results/3737874/data_processed
echo "Converting $DP to memmap ..."
python scripts/convert_to_memmap.py --data-processed $DP --splits train val test
echo "conversion exit: $?"
