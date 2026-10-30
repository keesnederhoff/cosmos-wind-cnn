#!/usr/bin/env bash
#SBATCH --job-name=v4_cleanup
#SBATCH --account=pcmsc
#SBATCH --wckey=reimbursable
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/home/cnederhoff/cosmos/cosmos-wind-cnn/v4_logs/cleanup_%j.log
#
# Space cleanup after the v4 chain. Dry run by default; pass 'execute' to delete:
#   sbatch scripts/cleanup_v4_followup.sh            # manifest + pre-flight, deletes nothing
#   sbatch scripts/cleanup_v4_followup.sh execute    # same checks, then deletes
# Modelled on ~/caldera_cleanup.sh: explicit KEEP list, abort if a keep path is missing.
set -uo pipefail
MODE="${1:-dryrun}"
REPO=/home/cnederhoff/cosmos/cosmos-wind-cnn
BASE=/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay
V2=$BASE/sf_bay_rtma/results
LONG=$BASE/sf_bay_rtma_long/results
V3=$BASE/sf_bay_rtma_v3/results
V4=$BASE/sf_bay_rtma_v4/results
STAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST=$REPO/v4_logs/cleanup_manifest_${MODE}_$STAMP.txt

echo "============ MODE: $MODE  $(date) ============"

KEEP=(
  "$BASE/sf_bay_rtma/raw_data"          # v3 and long raw_data are symlinks to this
  "$BASE/sf_bay_conus404"
  "$BASE/sf_bay_observed_data"
  "$BASE/validation"
  "$V2/v2data"
  "$V2/os_av_bc24_terr_res_s2"
  "$V2/os_wo_bc24_base_res_s2"
  "$V2/x10_wo_bc24_res_d1_s1"
  "$V2/x10_wo_bc24_res_d1_s2"
  "$V2/wv_wo_bc24_res_p2_w10_s1"        # registered in validation/config.py
  "$V2/wv_wo_bc24_res_p3_w10_s2"        # registered in validation/config.py
  "$V3/v3data/data_processed/normalization_stats.pkl"   # run folders link here; v3 inference needs it
  "$V3/v3data/data_processed/target_grid_reference.nc"
  "$V3/v1940_product"
  "$V3/v1940_points/station_table.nc"   # read by the v4 scripts
  "$V3/r1_do010"
  "$V3/r1b_do010_s2"
  "$V3/r1b_do010_s3"
  "$V4/v1940_product"
  "$V4/v1940_points"
  "$V4/v1940_grids"
  "$V4/v4data"
  "$V4/v4_s1"
  "$V4/v4_s2"
  "$V4/v4_s3"
  "$BASE/sf_bay_rtma_v4/raw_data"
)

# ---- whole-directory targets
VD=$V3/v3data/data_processed
TARGETS=("$V3/v1940_grids" "$VD/memmap" "$VD/train.nc" "$VD/val.nc" "$VD/test.nc")
for d in "$V2"/*/; do
  d=${d%/}
  case " ${KEEP[*]} " in *" $d "*) ;; *) TARGETS+=("$d") ;; esac
done
for d in "$LONG"/*/; do TARGETS+=("${d%/}"); done

# ---- sweep folders: only the inference NetCDFs go, checkpoints/logs/scores stay
SWEEPS=()
for d in "$V3"/qh_* "$V3"/c1_det_* "$V3"/qw* "$V3"/r0* "$V3"/r2* "$V3"/r3* "$V3"/r4*; do
  [ -d "$d/output_inference" ] && SWEEPS+=("$d/output_inference")
done

# ---- pre-flight -----------------------------------------------------------------
fail=0
echo "--- keep paths"
for k in "${KEEP[@]}"; do
  [ -e "$k" ] || { echo "!! MISSING KEEP: $k"; fail=1; }
done
echo "--- a target must never be, contain, or sit inside a keep path"
for t in "${TARGETS[@]}"; do
  case "$t" in "$BASE"/*) ;; *) echo "!! OUTSIDE BASE: $t"; fail=1 ;; esac
  for k in "${KEEP[@]}"; do
    case "$k/" in "$t"/*) echo "!! TARGET HOLDS KEEP: $t > $k"; fail=1 ;; esac
    case "$t/" in "$k"/*) echo "!! TARGET INSIDE KEEP: $t < $k"; fail=1 ;; esac
  done
done
echo "--- running jobs of mine"
RUNNING=$(squeue -u "$USER" -h -o '%i %j' | grep -v "^${SLURM_JOB_ID:-none} " || true)
if [ -n "$RUNNING" ]; then echo "$RUNNING"; echo "!! jobs are running"; fail=1; else echo "none"; fi

echo "--- symlinks anywhere under BASE that point into a target"
find "$BASE" -maxdepth 6 -type l -printf '%p -> %l\n' 2>/dev/null > "$MANIFEST.links"
echo "symlinks scanned: $(wc -l < "$MANIFEST.links")"
for t in "${TARGETS[@]}"; do
  # Run-folder data_processed links to the v3 training store dangle by design (retraining only).
  hits=$(grep -F -- "-> $t" "$MANIFEST.links" | grep -v '/results/[^/]*/data_processed/' || true)
  if [ -n "$hits" ]; then echo "!! LINK INTO TARGET: $hits"; fail=1; fi
done
echo "run-folder links that will dangle: $(grep -cE -- "-> $VD/(memmap|train.nc|val.nc|test.nc)" "$MANIFEST.links")"
echo "broken symlinks before: $(find "$BASE" -maxdepth 6 -xtype l 2>/dev/null | wc -l)"

echo "--- repo references to targets (review, not fatal)"
for t in "${TARGETS[@]}" "${SWEEPS[@]}"; do
  b=${t#"$BASE"/}
  n=$(basename "$(dirname "$b")")/$(basename "$b")
  hits=$(grep -rlF --include='*.py' --include='*.sh' --include='*.slurm' --include='*.yaml' -- "$n" \
         "$REPO/scripts" "$REPO/validation" "$REPO/case_studies" 2>/dev/null | sed "s|$REPO/||" | tr '\n' ' ')
  [ -n "$hits" ] && echo "  $n: $hits"
done
echo "  run names in validation/config.py:"
for d in "$V2"/*/ "$LONG"/*/; do
  n=$(basename "$d")
  c=$(grep -cF -- "$n" "$REPO/validation/config.py")
  [ "$c" != 0 ] && echo "    $n: $c lines"
done

echo "--- v3 product must be intact before its grids go"
source "$REPO/scripts/v4_env.sh"
python - "$V3/v1940_product" <<'PY'
import glob, sys
import numpy as np
import xarray as xr
bad = 0
for f in sorted(glob.glob(sys.argv[1] + '/*.nc')):
    with xr.open_dataset(f) as d:
        if 'time' not in d.dims:
            print('  static', f.rsplit('/', 1)[1])
            continue
        t = d['time'].values
    gaps = int((np.diff(t).astype('timedelta64[h]').astype(int) != 1).sum())
    print(f'  {f.rsplit("/", 1)[1]}: {t.size} h {str(t[0])[:13]} -> {str(t[-1])[:13]} breaks {gaps}')
    bad += gaps > 0
sys.exit(1 if bad else 0)
PY
[ $? = 0 ] || { echo "!! v3 product check failed"; fail=1; }

# ---- manifest ------------------------------------------------------------------
echo "--- manifest $MANIFEST"
: > "$MANIFEST"
total=0
for t in "${TARGETS[@]}"; do
  kb=$(du -sk "$t" 2>/dev/null | cut -f1)
  total=$((total + ${kb:-0}))
  printf '%12d KB  DIR  %s\n' "${kb:-0}" "$t" >> "$MANIFEST"
done
for s in "${SWEEPS[@]}"; do
  kb=$(find "$s" -maxdepth 1 -type f -name '*.nc' -printf '%k\n' | awk '{a+=$1} END {print a+0}')
  total=$((total + kb))
  printf '%12d KB  NC   %s/*.nc\n' "$kb" "$s" >> "$MANIFEST"
done
sort -n "$MANIFEST"
echo "TOTAL: $((total / 1024 / 1024)) GB in ${#TARGETS[@]} folders + ${#SWEEPS[@]} sweep inference folders"
lfs quota -g pcmsc /caldera/projects/usgs/hazards/pcmsc 2>/dev/null | sed -n 3,4p

if [ "$fail" != 0 ]; then echo "PRE-FLIGHT FAILED: nothing deleted"; exit 1; fi
echo "PRE-FLIGHT OK"
if [ "$MODE" != execute ]; then echo "dry run: nothing deleted"; exit 0; fi

# ---- delete --------------------------------------------------------------------
for t in "${TARGETS[@]}"; do
  echo "deleting $t"
  find "$t" -depth -delete
done
for s in "${SWEEPS[@]}"; do
  echo "deleting $s/*.nc"
  find "$s" -maxdepth 1 -type f -name '*.nc' -delete
done

echo "--- post"
for k in "${KEEP[@]}"; do [ -e "$k" ] || echo "!! KEEP GONE: $k"; done
echo "broken symlinks after: $(find "$BASE" -maxdepth 6 -xtype l 2>/dev/null | wc -l)"
lfs quota -g pcmsc /caldera/projects/usgs/hazards/pcmsc 2>/dev/null | sed -n 3,4p
echo "DONE $(date)"
