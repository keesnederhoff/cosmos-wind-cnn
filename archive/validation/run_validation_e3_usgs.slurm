#!/bin/bash
#SBATCH --job-name=val_e3usgs
#SBATCH --partition=cpu
#SBATCH --account=pcmsc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=8:00:00
#SBATCH --output=val_e3usgs_%j.log
set -euo pipefail

# Era E3 scored against ALL THREE station groups in ONE extraction, so the
# with-USGS and without-USGS pooled scores are a controlled comparison rather
# than two separate runs that could differ for other reasons. combined_skill
# then runs twice over the SAME csv; its `source.isin(WEIGHTS)` filter is what
# drops USGS for the second pass, so nothing but the weighting changes.
#
# "Equal weight to all" is applied at CATEGORY level (IEM:1,NDBC:1,USGS:1),
# which is what VAL_WEIGHTS controls. Note what that means: 4 moorings carry the
# same collective weight as 20 IEM stations, so each mooring gets roughly 5x the
# influence of an individual land station. The pooled score weights by sample
# size WITHIN a category, which is why EMC_MW101 (n=168, about a week of data,
# skill -0.38) cannot distort the result the way it does a station-mean.

source /home/cnederhoff/miniforge3/etc/profile.d/conda.sh
conda activate cosmos_wind_cnn
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1

BUNDLE=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/validation
export COSMOS_VALIDATION_DATA_ROOT="$BUNDLE"
export COSMOS_VALIDATION_OUTPUT_ROOT="$BUNDLE/results"
cd /home/cnederhoff/cosmos/cosmos-wind-cnn/validation

echo "======== E3 2020-2026, ALL groups (IEM + NDBC + USGS) ========"
# VAL_USGS=1 adds the moorings to the station set; NO VAL_ONLY_GROUPS, so they
# are scored ALONGSIDE the land stations rather than instead of them.
VAL_ERA=E3 VAL_USGS=1 VAL_OUTDIR_SUFFIX=_withusgs \
  VAL_WEIGHTS="IEM:1,NDBC:1,USGS:1" VAL_RANK_DIR=rankings_e3_all \
  python run_validation.py

D=obsE3_2020-2026__withusgs
# combined_skill writes its combined_*.png into the ERA dir under fixed
# filenames, and both passes share that dir -- so the second pass would silently
# overwrite the first pass's figures. Park each pass's output before the next
# runs; only the rankings CSVs go to distinct RANK_DIRs on their own.
OUT="$COSMOS_VALIDATION_OUTPUT_ROOT/$D"

echo "======== pooled WITH usgs (equal category weights) ========"
VAL_WEIGHTS="IEM:1,NDBC:1,USGS:1" VAL_RANK_DIR=rankings_e3_withusgs \
  VAL_ERA_DIRS="Obs E3 with USGS=$D" python analysis/combined_skill.py
mkdir -p "$OUT/combined_withusgs"
mv "$OUT"/combined_*.png "$OUT/combined_withusgs"/ 2>/dev/null || true
echo "  parked $(ls -1 "$OUT/combined_withusgs" | wc -l) figures in combined_withusgs/"

echo "======== pooled WITHOUT usgs (same extraction, control) ========"
VAL_WEIGHTS="IEM:1,NDBC:1" VAL_RANK_DIR=rankings_e3_nousgs \
  VAL_ERA_DIRS="Obs E3 no USGS=$D" python analysis/combined_skill.py
mkdir -p "$OUT/combined_nousgs"
mv "$OUT"/combined_*.png "$OUT/combined_nousgs"/ 2>/dev/null || true
echo "  parked $(ls -1 "$OUT/combined_nousgs" | wc -l) figures in combined_nousgs/"

echo "DONE"
