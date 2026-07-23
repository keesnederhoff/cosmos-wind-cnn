#!/bin/bash
#SBATCH --job-name=cosmos_bench_mm
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --account=pcmsc
#SBATCH --time=00:30:00
#SBATCH --output=cpu_bench_mm_%j.log

set -euo pipefail
source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd /home/cnederhoff/cosmos/cosmos-wind-cnn
DP=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/sf_bay_rtma/results/3737874/data_processed
python scripts/bench_memmap_loader.py --data-processed $DP --batches 200 --num-workers 8 --batch-size 32
echo "bench exit: $?"
