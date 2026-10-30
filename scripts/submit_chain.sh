#!/bin/bash
# Submit the WHOLE pipeline as ONE Slurm dependency DAG. After this returns, no further
# interaction is needed: every edge is --dependency=afterok with --kill-on-invalid-dep=yes,
# so a failed gate cancels everything downstream instead of leaving it queued.
#
#   bash scripts/submit_chain.sh --dry          # lint + print the DAG, submit nothing
#   bash scripts/submit_chain.sh                # lint + submit
#   START_STAGE=points bash scripts/submit_chain.sh   # resume: earlier stages assumed done
# Stages in order: rotate prep train points seedsel infer combine ba products val
# Cancel everything: bash scripts/cancel_chain.sh
# Status: tail -f <caldera>/sf_bay_rtma/results/chain_status.log ; result: RESULTS.md + DONE
set -euo pipefail
cd /home/cnederhoff/cosmos/cosmos-wind-cnn
source scripts/env.sh
# env.sh exports its helper functions for `bash -c` stages; do NOT let them ride along
# in sbatch's environment (each job sources env.sh itself).
export -nf chosen_seed quota_guard stage_begin
DRY=0; [ "${1:-}" = "--dry" ] && DRY=1
unset ERAS_OUT_DIR ERAS_POINTS ERAS_CKPT ERAS_MANIFEST
mkdir -p logs
START=${START_STAGE:-rotate}
ORDER=(rotate prep train points seedsel infer combine ba products val)

# ------------------------------------------------------------------------------ LINT
echo "=== LINT ==="
FAIL=0
SBATCH_FILES="scripts/cpu_rotate.slurm scripts/cpu_preprocess.slurm scripts/gpu_train.slurm scripts/gpu_eras.slurm scripts/cpu_run.slurm validation/run_validation.slurm validation/run_validation_peryear.slurm"
for f in $SBATCH_FILES scripts/env.sh scripts/submit_chain.sh scripts/cancel_chain.sh; do
  [ -f "$f" ] || { echo "  FAIL missing $f"; FAIL=1; continue; }
  bash -n "$f" || { echo "  FAIL bash -n $f"; FAIL=1; }
  if grep -q $'\r' "$f"; then echo "  FAIL CRLF in $f (a CRLF launcher submits zero jobs silently)"; FAIL=1; fi
done
for f in $SBATCH_FILES; do
  grep -q -- '--wckey=reimbursable' "$f" || { echo "  FAIL no wckey in $f"; FAIL=1; }
  grep -q -- '--account=pcmsc' "$f"      || { echo "  FAIL no account in $f"; FAIL=1; }
done
for f in scripts/rotate_rtma_earthrel.py scripts/gate.py scripts/report.py scripts/quantile_reliability.py \
         scripts/make_ba_quantiles.py scripts/make_p50_products.py scripts/combine_full_record.py \
         scripts/ba_fit_factor.py scripts/select_seed_points.py scripts/extract_station_points.py \
         scripts/run_training_pipeline.py validation/run_validation.py validation/config.py \
         src/cosmos_wind_cnn/data/preprocessing.py; do
  python -m py_compile "$f" || { echo "  FAIL py_compile $f"; FAIL=1; }
  if grep -q $'\r' "$f"; then echo "  WARN CRLF in $f"; fi
done
n_pts=$(grep -vE '^\s*#|^\s*$' scripts/era_manifest_points.txt | wc -l)
n_chosen=$(grep -vE '^\s*#|^\s*$' scripts/era_manifest_chosen.txt | wc -l)
n_bc=$(grep -vE '^\s*#|^\s*$' scripts/era_manifest_bcwin.txt | wc -l)
echo "  manifests: points=$n_pts (261) chosen=$n_chosen (87) bcwin=$n_bc (2)"
[ "$n_pts" = 261 ] && [ "$n_chosen" = 87 ] && [ "$n_bc" = 2 ] || { echo "  FAIL manifest counts"; FAIL=1; }
for f in preprocessing training inference_preprocessing_1940; do
  [ -f case_studies/sf_bay_rtma/configs/$f.yaml ] || { echo "  FAIL missing config $f"; FAIL=1; continue; }
done
python scripts/rotate_rtma_earthrel.py --help >/dev/null || { echo "  FAIL rotate --help"; FAIL=1; }
for f in ERA5_eastward_wind_1940_2026_UTM.nc ERA5_northward_wind_1940_2026_UTM.nc ERA5_cloud_area_fraction_1940_2026_UTM.nc \
         ERA5_wind_gust_1940_2026_UTM.nc RTMA_SFbay_2p5km_wind_gust_2016_2026_UTM10.nc \
         RTMA_SFbay_2p5km_eastward_wind_2011_2026_UTM10.nc RTMA_SFbay_2p5km_northward_wind_2011_2026_UTM10.nc \
         RTMA_SFbay_2p5km_surface_height_static_UTM10.nc; do
  [ -s "$CASE_DIR/raw_data/$f" ] || { echo "  FAIL missing raw input $f"; FAIL=1; }
done
quota_guard 1.5 || { echo "  FAIL quota"; FAIL=1; }
squeue -u "$USER" -h | head -5 | sed 's/^/  queue: /' || true
[ "$FAIL" = 0 ] || { echo "LINT FAILED -- nothing submitted"; exit 1; }
echo "LINT OK"

# ----------------------------------------------------------------------------- SUBMIT
JOBS=chain_jobids.txt
[ "$DRY" = 1 ] || : > "$JOBS"
N=0
sub() {  # sub <label> <sbatch args...>  -> echoes job id
  local label=$1; shift
  if [ "$DRY" = 1 ]; then N=$((N+1)); echo "  [dry] $label: sbatch $*" >&2; echo "9999$N"; return; fi
  local id; id=$(sbatch --parsable "$@" | cut -d';' -f1)
  echo "$label $id" >> "$JOBS"; echo "  submitted $label = $id" >&2; echo "$id"
}
dep() {  # dep <id...> -> afterok flags over the non-empty ids
  local ids=""; for i in "$@"; do [ -n "$i" ] && ids="$ids:$i"; done
  [ -n "$ids" ] && echo "--dependency=afterok$ids --kill-on-invalid-dep=yes" || echo ""
}
idx() { local i=0; for s in "${ORDER[@]}"; do [ "$s" = "$1" ] && echo $i; i=$((i+1)); done; }
want() { [ "$(idx "$1")" -ge "$(idx "$START")" ]; }
CPU="scripts/cpu_run.slurm"
J_rot=""; J_prep=""; J_gprep=""; J_train=""; J_gtrain=""; J_pts=""; J_gpts=""; J_sel=""
J_bc=""; J_grid=""; J_dir=""; J_ggrid=""; J_gdir=""; J_comb=""; J_ba=""; J_baq=""; J_p50=""; J_gprod=""
J_rel=""; J_val=""; J_ys=""

echo "=== SUBMIT (start stage: $START, dry=$DRY) ==="
# non-gating side job: per-year RTMA/ERA5 skill (no dependencies)
want rotate && J_ys=$(sub yearskill validation/run_validation_peryear.slurm)

want rotate && J_rot=$(sub rotate scripts/cpu_rotate.slurm)
if want prep; then
  J_prep=$(sub prep $(dep "$J_rot") scripts/cpu_preprocess.slurm)
  J_gprep=$(sub gate_prep $(dep "$J_prep") --job-name=gate_prep --cpus-per-task=8 --mem=96G --time=02:00:00 \
            $CPU gate_prep python scripts/gate.py prep)
fi
if want train; then
  J_train=$(sub train $(dep "$J_gprep") scripts/gpu_train.slurm)
  J_gtrain=$(sub gate_train $(dep "$J_train") --job-name=gate_train --cpus-per-task=4 --mem=32G --time=01:00:00 \
             $CPU gate_train python scripts/gate.py train)
fi
if want points; then
  J_pts=$(sub points $(dep "$J_gtrain") --array=0-260%3 \
          --export=ALL,ERAS_MANIFEST=$REPO/scripts/era_manifest_points.txt,ERAS_POINTS=1 scripts/gpu_eras.slurm)
  J_gpts=$(sub gate_points $(dep "$J_pts") --job-name=gate_pts --cpus-per-task=8 --mem=64G --time=02:00:00 \
           $CPU gate_points python scripts/gate.py points)
fi
if want seedsel; then
  J_sel=$(sub seedsel $(dep "$J_gpts") --job-name=cnn_seedsel --cpus-per-task=16 --mem=128G --time=04:00:00 \
          $CPU seedsel python scripts/select_seed_points.py)
fi
if want infer; then
  J_bc=$(sub bc_windows $(dep "$J_sel") --array=0-1 \
         --export=ALL,ERAS_MANIFEST=$REPO/scripts/era_manifest_bcwin.txt scripts/gpu_eras.slurm)
  J_grid=$(sub grids $(dep "$J_sel") --array=0-86%3 \
           --export=ALL,ERAS_MANIFEST=$REPO/scripts/era_manifest_chosen.txt,ERAS_OUT_DIR=$RES/v1940_grids scripts/gpu_eras.slurm)
  J_dir=$(sub dirtrack $(dep "$J_sel") --array=0-86%3 \
          --export=ALL,ERAS_MANIFEST=$REPO/scripts/era_manifest_chosen.txt,ERAS_OUT_DIR=$RES/v1940_grids,ERAS_CKPT=best_direction.pth scripts/gpu_eras.slurm)
  J_ggrid=$(sub gate_grids $(dep "$J_grid") --job-name=gate_grid --cpus-per-task=8 --mem=64G --time=02:00:00 \
            $CPU gate_grids python scripts/gate.py grids)
  J_gdir=$(sub gate_dirtrack $(dep "$J_dir") --job-name=gate_dir --cpus-per-task=8 --mem=64G --time=02:00:00 \
           $CPU gate_dirtrack python scripts/gate.py dirtrack)
fi
if want combine; then
  J_comb=$(sub combine $(dep "$J_ggrid" "$J_bc") --job-name=cnn_combine --cpus-per-task=16 --mem=128G --time=48:00:00 \
           $CPU combine bash -c 'quota_guard 1.0 && python scripts/combine_full_record.py --seed "$(chosen_seed)"')
fi
if want ba; then
  J_ba=$(sub ba_fit $(dep "$J_comb") --job-name=cnn_ba_fit --cpus-per-task=4 --mem=48G --time=02:00:00 \
         $CPU ba_fit python scripts/ba_fit_factor.py)
  J_baq=$(sub ba_quantiles $(dep "$J_ba") --job-name=cnn_ba_q --cpus-per-task=8 --mem=64G --time=08:00:00 \
          $CPU ba_quantiles bash -c 'quota_guard 0.5 && python scripts/make_ba_quantiles.py')
fi
if want products; then
  J_p50=$(sub p50 $(dep "$J_ba" "$J_gdir") --job-name=cnn_p50 --cpus-per-task=8 --mem=96G --time=24:00:00 \
          $CPU p50 bash -c 'quota_guard 0.5 && python scripts/make_p50_products.py --dir-dir "$RES/v1940_grids/$(chosen_seed)"')
  J_gprod=$(sub gate_products $(dep "$J_p50" "$J_baq") --job-name=gate_prod --cpus-per-task=8 --mem=96G --time=02:00:00 \
            $CPU gate_products python scripts/gate.py products)
fi
if want val; then
  J_rel=$(sub reliability $(dep "$J_gprod") --job-name=cnn_reliab --cpus-per-task=16 --mem=128G --time=04:00:00 \
          $CPU reliability python scripts/quantile_reliability.py)
  J_val=$(sub val $(dep "$J_gprod") validation/run_validation.slurm)
  # report runs afterany: it must describe a failed chain just as well as a finished one
  ANY=""; for i in "$J_val" "$J_rel" "$J_ys"; do [ -n "$i" ] && ANY="$ANY:$i"; done
  J_rep=$(sub report ${ANY:+--dependency=afterany$ANY} --job-name=report --cpus-per-task=2 --mem=16G --time=00:30:00 \
          $CPU report python scripts/report.py)
fi
echo "=== DAG submitted. Jobs listed in $JOBS ==="
[ "$DRY" = 1 ] || squeue -u "$USER" -o "%.10i %.14j %.9T %.20E" | head -60
