#!/bin/bash
#SBATCH --job-name=biascorr
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --account=pcmsc
#SBATCH --time=04:00:00
#SBATCH --output=biascorr_%j.log

# Post-hoc per-cell quantile-mapping bias correction of a CNN wind product.
# Fits CNN->RTMA quantile maps on the TRAIN period, applies to the full record,
# writes BC_<...>.nc (uppercase so it does NOT hijack the full_record_*.nc eval
# glob), and (--evaluate) scores RAW vs BC vs ERA5 on the held-out TEST period.
# Heavy in RAM (~15 GB fit + full-record apply) -> must run on a compute node,
# NOT a login node. Usage: sbatch scripts/cpu_bias_correct.slurm [RUN_NAME]

set -uo pipefail
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
RUN_NAME=${1:-os_wo_bc24_base_res_s2}
echo "bias-correct run=$RUN_NAME"
source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd /home/cnederhoff/cosmos/cosmos-wind-cnn
python scripts/bias_correct.py --run "$RUN_NAME" --evaluate
echo "biascorr exit: $?"
