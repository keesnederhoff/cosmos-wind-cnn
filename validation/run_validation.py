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
MAKE_SPATIAL_MAPS = False   # slow cartopy peak maps; True for final figures
CWOP_PLOT_SAMPLE  = 0       # CWOP stats-only (per-station figures for a sample if >0)
# ===========================================================================

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

V.OUTPUT_DIR        = config.OUTPUT_ROOT / outdir
V.MAKE_SPATIAL_MAPS = MAKE_SPATIAL_MAPS
V.CWOP_PLOT_SAMPLE_N = CWOP_PLOT_SAMPLE

groups_lbl = '+'.join(ONLY_GROUPS) if ONLY_GROUPS else 'ALL'
print(f"=== Era {ERA} [{groups_lbl}]: {len(models)} models x "
      f"{len(V.STATIONS_TO_RUN)} stations, {tr} -> {V.OUTPUT_DIR} ===", flush=True)
V.main()
