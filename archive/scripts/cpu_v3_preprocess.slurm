#!/bin/bash
#SBATCH --job-name=v3_prep
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --account=pcmsc
#SBATCH --time=1-00:00:00
#SBATCH --output=v3_prep_%j.log

# Build the ONE shared preprocessed store for the v3 predictor-block campaign:
#   preprocess (all 30 file_dict channels) -> train/val/test.nc -> memmap
#
# Every arm symlinks this single store and selects its channels with
# SWEEP_ADD_INPUTS, so all 18 runs train on bit-identical data and the ablation
# stays controlled. This mirrors the v2data pattern, which was verified to
# reproduce identical splits and bit-identical normalization stats.

set -euo pipefail

# NOTE the stock scripts/cpu_tallgrass.slurm carries a STALE data root
# (.../cnn_wind_sfbay/cosmos-wind-cnn), which resolves to a nonexistent
# raw_data dir. The correct root is one level up -- get_data_dir() reads
# <root>/<case_name>/raw_data.
export COSMOS_DATA_ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay
export COSMOS_RESULTS_ROOT=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay

CASE_STUDY="case_studies/sf_bay_rtma_v3"
RUN_NAME="${RUN_NAME:-v3data}"
RESULTS="${COSMOS_RESULTS_ROOT}/sf_bay_rtma_v3/results/${RUN_NAME}"
LOG_DIR="${RESULTS}/logs"

trap 'mkdir -p "$LOG_DIR" && cp "${SLURM_SUBMIT_DIR}/v3_prep_${SLURM_JOB_ID}.log" "$LOG_DIR/" 2>/dev/null || true' EXIT

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Node        : $SLURM_NODELIST"
echo "CPUs        : $SLURM_CPUS_PER_TASK"
echo "Case study  : $CASE_STUDY"
echo "Run name    : $RUN_NAME"
echo "Data root   : $COSMOS_DATA_ROOT"
echo "Results     : $RESULTS"

source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
cd /home/cnederhoff/cosmos/cosmos-wind-cnn

python -c "import cosmos_wind_cnn; print('import ok:', cosmos_wind_cnn.__file__)"

# ── Step 1: preprocess only ───────────────────────────────────────────────────
echo ""
echo "=================== PREPROCESS ==================="
srun python scripts/run_training_pipeline.py \
    --case-study "$CASE_STUDY" \
    --run-name "$RUN_NAME" \
    --gpus 0 \
    --skip-train \
    --skip-inference \
    --skip-eval

# ── Step 2: convert splits to memmap ──────────────────────────────────────────
# Uncompressed + memory-mapped is what makes training viable: random access to
# the zlib-compressed NetCDF ran ~17 batches/HOUR, the memmap path ~22 batch/s
# warm. It also shares ONE copy per node through the page cache instead of a
# private copy per DDP rank.
echo ""
echo "=================== MEMMAP CONVERT ==================="
srun python scripts/convert_to_memmap.py \
    --data-processed "${RESULTS}/data_processed" \
    --splits train val test

echo ""
echo "=================== SUMMARY ==================="
ls -la "${RESULTS}/data_processed/" || true
du -sh "${RESULTS}/data_processed/memmap" 2>/dev/null || true
for s in train val test; do
  echo "--- $s ---"
  cat "${RESULTS}/data_processed/memmap/${s}/meta.json" 2>/dev/null || true
  echo ""
done
echo "DONE"
