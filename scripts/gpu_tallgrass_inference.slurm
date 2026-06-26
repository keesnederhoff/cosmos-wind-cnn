#!/bin/bash
#SBATCH --job-name=cosmos_inference
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=kees.nederhoff@deltares-usa.us
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                    # Single GPU — inference is sequential
#SBATCH --cpus-per-task=18             # 18 cores for DataLoader workers + main
#SBATCH --mem=200G                      # Full record fits comfortably in 200 GB
#SBATCH --account=pcmsc
#SBATCH --time=0-04:00:00              # Full-record inference typically finishes in <2 h
#SBATCH --output=inference_%j.log

set -euo pipefail

# ── Run name of the checkpoint to use ────────────────────────────────────────
# Pass overrides at submission time:
#   sbatch --export=RUN_NAME=3663482,START_DATE=1979-10-01,END_DATE=2021-12-31 gpu_tallgrass_inference.slurm
# Or edit the defaults below.
RUN_NAME=${RUN_NAME:-3663482}
START_DATE=${START_DATE:-1979-10-01}
END_DATE=${END_DATE:-2021-12-31}

# ── Environment ───────────────────────────────────────────────────────────────
source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn

cd /home/cnederhoff/cosmos/cosmos-wind-cnn

# Sanity checks
which python
python -c "import sys; print(sys.executable)"
python -c "import cosmos_wind_cnn; print('import ok:', cosmos_wind_cnn.__file__)"

echo "========================================="
echo "Full-record ERA5 inference"
echo "  run name   : $RUN_NAME"
echo "  period     : $START_DATE → $END_DATE"
echo "  checkpoint : case_studies/sf_bay_conus404/results/${RUN_NAME}/checkpoint/best_model.pth"
echo "  output     : case_studies/sf_bay_conus404/results/${RUN_NAME}/output_inference/full_record.nc"
echo "========================================="

python scripts/inference_full_record.py \
    --case-study  case_studies/sf_bay_conus404   \
    --run-name    "$RUN_NAME"            \
    --start-date  "$START_DATE"          \
    --end-date    "$END_DATE"            \
    --batch-size  64                     \
    --num-workers 8

echo "========================================="
echo "Inference complete — run name: ${RUN_NAME}"
echo "========================================="
