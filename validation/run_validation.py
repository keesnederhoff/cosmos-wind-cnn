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
ERA = os.environ.get('VAL_ERA', '2')   # '1' 1990-2010 | '2' 2011-2021 | '3' 2022-present
ONLY_GROUPS = None   # None = all stations; else restrict to these obs groups

ERAS = {
    '1': (['NOW-23', 'Sup3rWind', 'ERA5', 'CONUS404', 'UCLA', 'WRF_CalNev', 'CNN'],
          ('1990-01-01', '2011-01-01'), 'era1_1990-2010'),
    '2': (['NOW-23', 'Sup3rWind', 'RTMA', 'ERA5', 'HRRR', 'CONUS404', 'UCLA',
           'WRF_CalNev', 'CNN', 'CNN-RTMA-20260625',
           'CNN-allvars', 'CNN-windonly', 'CNN-extreme'],
          ('2011-01-01', '2022-01-01'), 'era2_2011-2021'),
    '3': (['RTMA', 'HRRR', 'ERA5', 'CNN', 'CNN-RTMA-20260625', 'NOW-23',
           'CNN-allvars', 'CNN-windonly', 'CNN-extreme'],
          ('2022-01-01', '2027-01-01'), 'era3_2022-present'),
}

VARIABLES         = ['wind']   # wind-only validation (user choice 2026-07-23)
MAKE_SPATIAL_MAPS = False   # slow cartopy peak maps; True for final figures
CWOP_PLOT_SAMPLE  = 0       # CWOP stats-only (per-station figures for a sample if >0)
# ===========================================================================

models, tr, outdir = ERAS[ERA]
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
