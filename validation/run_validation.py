"""
SF Bay meteorological product validation — single era-aware driver.

Set ERA below and run. Each era runs the products that exist in that window
against ALL stations (IEM + NDBC + CWOP; + USGS moorings if enabled in config).
Per-station figures are made for the quality groups; CWOP is stats-only.
Outputs land in config.OUTPUT_ROOT / <era outdir>.

Run:  python run_validation.py
"""
from pathlib import Path
import validate_met_models as V
import config
import os

# === CONFIGURATION =========================================================
# Every knob below is overridable from the environment so the same file drives
# an interactive Windows run and the Caldera slurm harness without being edited.
ERA = os.environ.get('VAL_ERA', '2')   # '1' 1990-2010 | '2' 2011-2021 | '3' 2022-present
                                       # 'A' 2011-2019 | 'B' 2020-2025  (2020 split; USGS moorings all land in B)
# None = all stations; else restrict to these obs groups, e.g. VAL_ONLY_GROUPS=USGS
ONLY_GROUPS = [g.strip() for g in os.environ.get('VAL_ONLY_GROUPS', '').split(',') if g.strip()] or None
# VAL_STATIONS=AAMC1,WT_MW101 restricts to named stations (finer than ONLY_GROUPS) --
# used for focused figure runs so the full-network results are not recomputed.
ONLY_STATIONS = [s.strip() for s in os.environ.get('VAL_STATIONS', '').split(',') if s.strip()] or None
# VAL_SPATIAL=1 turns on the per-station peak-event wind-field maps (off by default:
# they are skipped entirely in a normal overview run).
MAKE_SPATIAL = os.environ.get('VAL_SPATIAL', '0').lower() in ('1', 'true', 'yes')
# VAL_OUTDIR_SUFFIX appends to the era output dir so a focused run cannot overwrite
# the published results.
OUTDIR_SUFFIX = os.environ.get('VAL_OUTDIR_SUFFIX', '')

ERAS = {
    '1': (['ERA5', 'AORC', 'CNN', 'CNN-RTMA-20260625'],
          ('1990-01-01', '2011-01-01'), 'era1_1990-2010'),
    '2': (['ERA5', 'RTMA', 'AORC', 'CNN', 'CNN-RTMA-20260625', 'CNN-allvars', 'CNN-windonly'],
          ('2011-01-01', '2022-01-01'), 'era2_2011-2021'),
    '3': (['ERA5', 'RTMA', 'AORC', 'CNN', 'CNN-RTMA-20260625', 'CNN-allvars', 'CNN-windonly'],
          ('2022-01-01', '2027-01-01'), 'era3_2022-present'),
    # ---- 2020 split (2026-07-30). All four USGS moorings start after
    # 2020-01-22, so era B carries every USGS record. Product set is the five
    # bias-corrected CNN twins + CNN-allvars raw as the one raw/BC control pair.
    'A': (['ERA5', 'CONUS404', 'RTMA-SFbay', 'CNN-allvars', 'CNN-allvars-BC',
           'CNN-windonly-BC', 'CNN-extreme-BC', 'CNN-wave-p2-BC', 'CNN-wave-p3-BC'],
          ('2011-01-01', '2020-01-01'), 'eraA_2011-2019'),
    # CONUS404 ends 2021 -> excluded from B rather than scored on a partial window.
    'B': (['ERA5', 'RTMA-SFbay', 'CNN-allvars', 'CNN-allvars-BC',
           'CNN-windonly-BC', 'CNN-extreme-BC', 'CNN-wave-p2-BC', 'CNN-wave-p3-BC'],
          ('2020-01-01', '2026-01-01'), 'eraB_2020-2025'),
}

# VAL_VARIABLES=wind,temperature,... ; default wind-only -- the scalar ERA5/CONUS404
# sources are not staged in the Caldera bundle, so asking for them there fails.
VARIABLES         = [v.strip() for v in os.environ.get('VAL_VARIABLES', 'wind').split(',') if v.strip()]
MAKE_SPATIAL_MAPS = MAKE_SPATIAL   # per-station peak-event wind-field maps (VAL_SPATIAL=1)
CWOP_PLOT_SAMPLE  = 0       # CWOP stats-only (per-station figures for a sample if >0)
# ===========================================================================

# Three-era obs track (2026-08-14). The shipping recipe scored against REAL
# observations across the two eras it never saw and the one it trained on:
#   E1 2000-2010  pre-RTMA. No gridded truth exists here at all -- stations are
#                 the only verification possible, which is the whole point.
#   E2 2011-2019  RTMA exists but was never trained on. The generalisation test.
#   E3 2020-2026  the training era (train 2020-01-01..2024-03-13, val ..2025-02-05,
#                 test ..2025-12-31). Skill here is an UPPER BOUND, not a forecast
#                 of skill elsewhere -- read it against E1/E2, never on its own.
# CONUS404 ends 2021 so it references E1/E2 only; RTMA starts 2011 so it
# references E2/E3 only. Neither is dropped silently -- each era names its own.
# USGS moorings all start after 2020-01-22 and so can only appear in E3; run
# them as their own group (VAL_GROUPS=usgs), never pooled into the IEM+NDBC
# headline, because their anemometers sit at 1.2-4.9 m rather than 10 m.
# E3's END IS THE RECORD'S END, and it moves whenever ERA5 is extended. It was
# '2026-07-27' for the 2026-08-14 product; the 2026-08-27 ERA5 drop pushed the
# CNN record to 2026-08-10T01, so a stale end here would silently drop the whole
# extension and make the re-run invisible in the scores -- the same class of
# defect as the hardcoded ERA_DIRS in combined_skill.py. Override with
# VAL_ERA_END to reproduce an older window exactly (e.g. VAL_ERA_END=2026-07-27
# for a like-for-like comparison against the 2026-08-14 baseline).
#
# NOTE the references do not all reach the new end: RTMA truth stops at
# 2026-07-30T18, so in the last ~10 days the CNN is scored against stations with
# no RTMA column beside it. That is a coverage difference, not a bias -- each
# model is matched to observations independently -- but do not read the tail as
# a CNN-vs-RTMA result.

def _product_present(cfg):
    """True if a MODELS entry's primary data exists (single/multi-file or box)."""
    p = cfg.get('u_file') or cfg.get('data_dir')
    return p is not None and Path(p).exists()


_E3_END = os.environ.get('VAL_ERA_END', '2026-08-11')
_ERA_TR = {
    # 2026-08-29 products-comparison: full reference field per era, each product
    # only in eras it (mostly) covers. Sup3rWind (2007-2013) is PARTIAL in both
    # E1 and E2; HRRR (Oct 2014+) partial in E2. UCLA/CONUS404/NOW-23 end
    # 2020/2021/2022 -> excluded from E3 rather than scored on <=2 of 6.6 yr.
    'E1': (('2000-01-01', '2011-01-01'), 'obsE1_2000-2010',
           ['ERA5', 'CONUS404', 'UCLA', 'NOW-23', 'Sup3rWind']),
    'E2': (('2011-01-01', '2020-01-01'), 'obsE2_2011-2019',
           ['ERA5', 'CONUS404', 'RTMA-SFbay', 'HRRR', 'UCLA', 'NOW-23', 'Sup3rWind']),
    'E3': (('2020-01-01', _E3_END), 'obsE3_2020-2026',
           ['ERA5', 'RTMA-SFbay', 'HRRR']),
    # Full-record eras (2026-09-29, the 1940-2026 product). ERA5 is the only
    # gridded reference before 1979; CONUS404 starts 1979, so it enters E0 only.
    # Station counts thin out fast: ~6 IEM in 1940-59, 10 in 1960-79, no NDBC
    # before 1980 -- quote every pre-2000 number with its station count.
    'Em2': (('1940-01-01', '1960-01-01'), 'obsEm2_1940-1959', ['ERA5']),
    'Em1': (('1960-01-01', '1980-01-01'), 'obsEm1_1960-1979', ['ERA5']),
    'E0':  (('1980-01-01', '2000-01-01'), 'obsE0_1980-1999', ['ERA5', 'CONUS404']),
}
# VAL_ERA_SET=1940 scores the 1940-2026 product (config.V3_1940_MODELS). Over
# E1-E3 it also carries the archived same-seed arm V3-ERAS-s3, which the new
# product reproduces bit-for-bit -- any score gap between the two is a red flag.
# Unset keeps the 2026-08 era runs exactly as they were.
_ERA_SET = os.environ.get('VAL_ERA_SET', '')
if ERA in _ERA_TR:
    from config import V3_ERA_MODELS as _VE, V3_1940_MODELS as _V40
    tr, outdir, _refs = _ERA_TR[ERA]
    _pre2000 = ERA in ('Em2', 'Em1', 'E0')
    if _ERA_SET == 'ba':
        # P50 raw + observation-based BA products (2026-10-02); refs unchanged
        from config import P50_MODELS as _PBA
        _cnn = list(_PBA)
    elif _ERA_SET == '1940':
        _cnn = list(_V40) + ([] if _pre2000 else ['V3-ERAS-s3'])
    elif _pre2000:
        _cnn = list(_V40)          # the V3-ERAS arms start in 2000
    elif config.MODELS['CNN-quantile-v3']['data_dir'].exists():
        _cnn = ['CNN-quantile-v3']  # Windows data home (2026-08-29 products comparison)
    else:
        _cnn = list(_VE)           # Caldera: the three recipe seeds
    # Reference products that are not staged in this environment (HRRR, UCLA,
    # NOW-23, Sup3rWind live on the Windows side only) are dropped here, loudly,
    # instead of failing the path audit one by one.
    _missing = [m for m in _refs if not _product_present(config.MODELS[m])]
    if _missing:
        print(f"references not staged here, skipped: {_missing}", flush=True)
    models = _cnn + [m for m in _refs if m not in _missing]
elif ERA == 'V3':
    # v3 held-out test window. Product list is built from config.V3_MODELS so it
    # cannot drift from what config.py actually defines, plus ERA5 and RTMA as
    # references and the v2 production pick for continuity.
    from config import V3_MODELS as _V3
    models = list(_V3) + ['ERA5', 'RTMA-SFbay', 'CNN-wave-p3-BC']
    tr = ('2025-02-06', '2026-01-01')
    outdir = 'v3_test_2025'
else:
    models, tr, outdir = ERAS[ERA]
_vm = os.environ.get('VAL_MODELS')
if _vm:
    models = [m.strip() for m in _vm.split(',') if m.strip()]
V.MODELS_TO_RUN     = models
V.VARIABLES         = VARIABLES
V.TIME_RANGE        = tr

# USGS-focused run: restrict the station set and use a distinct output dir so
# the existing full-network Era-2 results are not overwritten.
if ONLY_GROUPS:
    outdir = f"{outdir}_{'_'.join(ONLY_GROUPS)}"
    V.STATIONS_TO_RUN = [s for s, c in V.STATIONS.items() if c['group'] in ONLY_GROUPS]

if ONLY_STATIONS:
    missing = [s for s in ONLY_STATIONS if s not in V.STATIONS]
    if missing:
        raise SystemExit(f"VAL_STATIONS: unknown station(s) {missing}")
    V.STATIONS_TO_RUN = [s for s in V.STATIONS_TO_RUN if s in ONLY_STATIONS] \
        if ONLY_GROUPS else list(ONLY_STATIONS)

if OUTDIR_SUFFIX:
    outdir = f"{outdir}_{OUTDIR_SUFFIX}"

# Land boundary for the spatial maps. config.LDB_FILE (deltabay.ldb) is not present in
# the Caldera bundle, so fall back to contouring the RTMA land-sea mask, which lives on
# the same 2.5 km UTM-10 grid as the model output.
if MAKE_SPATIAL_MAPS:
    try:
        V.LDB_POLYGONS_OVERRIDE = V.coastline_from_landsea(config.LANDSEA_FILE)
        print(f"coastline: {len(V.LDB_POLYGONS_OVERRIDE)} polylines from land-sea mask")
    except Exception as _e:
        print(f"coastline unavailable ({type(_e).__name__}: {_e}) -- maps without land boundary")

V.OUTPUT_DIR        = config.OUTPUT_ROOT / outdir
V.MAKE_SPATIAL_MAPS = MAKE_SPATIAL_MAPS
V.CWOP_PLOT_SAMPLE_N = CWOP_PLOT_SAMPLE

groups_lbl = '+'.join(ONLY_GROUPS) if ONLY_GROUPS else 'ALL'
print(f"=== Era {ERA} [{groups_lbl}]: {len(models)} models x "
      f"{len(V.STATIONS_TO_RUN)} stations, {tr} -> {V.OUTPUT_DIR} ===", flush=True)
V.main()
