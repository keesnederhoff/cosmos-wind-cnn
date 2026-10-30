#!/bin/bash
#SBATCH --job-name=v4_prep
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --account=pcmsc
#SBATCH --wckey=reimbursable
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/cnederhoff/cosmos/cosmos-wind-cnn/v4_logs/%x_%j.log

# Stage 2: build the ONE shared v4 store (8 recipe channels):
#   preprocess -> train/val/test.nc (water-year split, 48 h gaps) -> memmap
# Derived from cpu_v3_preprocess.slurm; differences: v4 case/run name, wckey (the v3 script
# had none), quota guard, and the env comes from v4_env.sh.
set -euo pipefail
source /home/cnederhoff/cosmos/cosmos-wind-cnn/scripts/v4_env.sh
v4_begin prep
v4_quota_guard 0.5

CASE_STUDY="case_studies/sf_bay_rtma_v4"
RUN_NAME=v4data
RESULTS="$V4RES/$RUN_NAME"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# hard inputs must exist BEFORE any compute
for f in RTMA_SFbay_2p5km_eastward_wind_2011_2026_UTM10_earthrel.nc \
         RTMA_SFbay_2p5km_northward_wind_2011_2026_UTM10_earthrel.nc \
         RTMA_SFbay_2p5km_wind_gust_2016_2026_UTM10.nc \
         ERA5_eastward_wind_1940_2026_UTM.nc ERA5_northward_wind_1940_2026_UTM.nc \
         ERA5_cloud_area_fraction_1940_2026_UTM.nc ERA5_wind_gust_1940_2026_UTM.nc \
         RTMA_SFbay_2p5km_surface_height_static_UTM10.nc; do
  [ -s "$V4/raw_data/$f" ] || { echo "FATAL: missing $V4/raw_data/$f"; exit 1; }
done

if [ -s "$RESULTS/data_processed/memmap/test/meta.json" ]; then
  echo "memmap store already complete -> skipping rebuild (delete $RESULTS to force)"
else
  echo "=================== PREPROCESS ==================="
  srun python scripts/run_training_pipeline.py --case-study "$CASE_STUDY" --run-name "$RUN_NAME" \
      --gpus 0 --skip-train --skip-inference --skip-eval
  echo "=================== MEMMAP CONVERT ==================="
  srun python scripts/convert_to_memmap.py --data-processed "$RESULTS/data_processed" \
      --splits train val test
fi
ls -la "$RESULTS/data_processed/"
du -sh "$RESULTS/data_processed/memmap" || true
