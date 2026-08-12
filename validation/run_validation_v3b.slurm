#!/usr/bin/env bash
# Day 4: obs-validation of the fixed shipping recipe over the HELD-OUT TEST
# WINDOW (2025-02-06 -> 2025-12-31), against IEM + NDBC stations.
#
# WHY THIS RUN DECIDES THE PROJECT. Grid-point skill is scored against RTMA --
# the field the CNN was trained to imitate -- so it cannot arbitrate between a
# model that is genuinely better and one that imitates RTMA more closely.
# Stations are the only truth here that is not the training target, and they are
# where the contradiction this whole wrap-up exists to resolve first appeared:
# the deterministic control led the obs board at 0.514/0.511/0.484 and beat RTMA
# itself (0.483), while the best quantile arm managed only 0.469.
#
# Day 3 has already shown the recipe beats both C1 and the qh_P0 incumbent on
# gridded all-hours skill (0.2317 median vs 0.1686 and 0.1775). If it also takes
# the obs board, the contradiction dissolves and the quantile head ships. If C1
# still leads on stations, the contradiction was real and survives a properly
# trained quantile head -- which is itself the headline result, and the day-7
# recommendation becomes the deterministic model.
#
# 28 products: the 21 campaign arms, the 3 recipe seeds (best_speed.pth, ep 7),
# the 3 direction-checkpoint seeds (best_direction.pth, ep 27) and the one
# selection-rule comparator (best_model.pth, ep 10).
#
# USGS moorings stay OUT of the headline pooled score and are reported as their
# own group: they sit at 1.2-4.93 m anemometer height against a 10 m model field,
# and pooling them previously cost every product 0.03-0.25 skill.
#
#   sbatch run_validation_v3b.slurm
#SBATCH --job-name=val_v3b
#SBATCH --partition=cpu
#SBATCH --account=pcmsc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=24:00:00
#SBATCH --output=val_v3b_%j.log
set -euo pipefail

source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
export KMP_DUPLICATE_LIB_OK=TRUE

BUNDLE=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/validation
export COSMOS_VALIDATION_DATA_ROOT="$BUNDLE"
export COSMOS_VALIDATION_OUTPUT_ROOT="$BUNDLE/results"
mkdir -p "$COSMOS_VALIDATION_OUTPUT_ROOT"

cd /home/cnederhoff/cosmos/cosmos-wind-cnn/validation

# --- Fail loudly if any registered arm's inference is missing ---------------
# A product whose file does not resolve is dropped SILENTLY by the engine's path
# audit -- it simply vanishes from the leaderboard. That has bitten this project
# three times. The audit is driven from config.py itself rather than a hardcoded
# arm list, so it cannot drift from what is actually registered.
echo "=== inference file audit ==="
python - <<'PY'
import sys
sys.path.insert(0, '.')
import config
miss = [m for m in config.V3_MODELS if not config.MODELS[m]['u_file'].exists()]
print(f"  registered: {len(config.V3_MODELS)}  resolved: {len(config.V3_MODELS)-len(miss)}")
if miss:
    for m in miss:
        print(f"  MISSING {m} -> {config.MODELS[m]['u_file']}")
    raise SystemExit(f"ABORT: {len(miss)} inference files absent -- "
                     "not running a partial leaderboard.")
print("  all present")
PY

echo "================ V3 test window (2025-02-06 -> 2025-12-31) ================"
echo "--- headline: IEM + NDBC pooled ---"
VAL_ERA=V3 VAL_WEIGHTS="IEM:1,NDBC:1" VAL_RANK_DIR=rankings_day4 \
  VAL_OUTDIR_SUFFIX=_day4 python run_validation.py

echo "--- USGS moorings as a separate group ---"
VAL_ERA=V3 VAL_USGS=1 VAL_ONLY_GROUPS=USGS VAL_OUTDIR_SUFFIX=_day4_usgs \
  python run_validation.py || echo "USGS group failed (non-fatal, coverage may be thin in 2025)"

echo "================ combined skill / ranking ================"
VAL_WEIGHTS="IEM:1,NDBC:1" VAL_RANK_DIR=rankings_day4 \
  python analysis/combined_skill.py || echo "combined_skill failed (non-fatal)"

echo "DONE"
