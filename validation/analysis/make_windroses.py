"""Generate the obs-vs-model wind-rose comparison figures only (the rest of the
validation outputs already exist). Reuses the verified batched extraction; loops
the 3 eras x their models x the figure stations and calls plot_wind_rose.
Set ONLY_ERA env (1|2|3) to do a single era; default = all three."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root on path
import config
import numpy as np
import pandas as pd
import validate_met_models as V

if not V.HAS_WINDROSE:
    print("ERROR: windrose still not importable in this env."); sys.exit(1)

ALL_ST = ['46026', 'AAMC1', 'WT_MW101', 'EMC_MW101', 'SFO', 'CCR', 'G0049', 'F0247']
ERAS = {
    '1': (['NOW-23', 'Sup3rWind', 'ERA5', 'CONUS404', 'UCLA', 'WRF_CalNev', 'CNN'],
          ('1990-01-01', '2011-01-01'), 'era1_1990-2010'),
    '2': (['NOW-23', 'Sup3rWind', 'RTMA', 'ERA5', 'HRRR', 'CONUS404', 'UCLA', 'WRF_CalNev', 'CNN'],
          ('2011-01-01', '2022-01-01'), 'era2_2011-2021'),
    '3': (['RTMA', 'HRRR', 'ERA5', 'CNN', 'NOW-23'],
          ('2022-01-01', '2027-01-01'), 'era3_2022-present'),
}
BASE = config.OUTPUT_ROOT


def load_model_dispatch(name, cfg):
    if cfg.get('kind') == 'point_product':
        return V.load_point_product(name, cfg)
    if cfg.get('kind') == 'box':
        return V.load_model_box(name, cfg)
    if cfg.get('crs') == 'latlon_2d':
        return V.load_model_wrf_calnev(name, cfg)
    if 'data_dir' in cfg and cfg.get('crs') == 'lcc':
        return V.load_model_ucla(name, cfg)
    if 'data_dir' in cfg and cfg.get('crs', 'utm10n') == 'utm10n':
        return V.load_model_utm_multifile(name, cfg)
    return V.load_model(name, cfg)


def do_era(era):
    models, tr, outdir = ERAS[era]
    out_root = BASE / outdir
    tslice = (pd.Timestamp(tr[0]), pd.Timestamp(tr[1]))
    print(f"\n===== Era {era}: {outdir}  {tr} =====", flush=True)

    # Load stations (time-filtered) exactly like main()
    station_data = {}
    for sid in ALL_ST:
        cfg = V.STATIONS.get(sid)
        if cfg is None:
            continue
        data = V.load_station(sid, cfg)
        if data is None:
            continue
        mask = (data['time'] >= tslice[0]) & (data['time'] <= tslice[1])
        for k in ['time', 'u10', 'v10', 'speed10', 'dir_deg', 'air_temp_C']:
            data[k] = data[k][mask]
        station_data[sid] = data
    if not station_data:
        print("  no stations in window"); return

    cwop_loaded = [s for s in station_data if V.STATIONS.get(s, {}).get('group') == 'CWOP']
    cwop_sample = set(sorted(cwop_loaded,
                            key=lambda s: int(np.isfinite(station_data[s]['speed10']).sum()),
                            reverse=True)[:V.CWOP_PLOT_SAMPLE_N])

    for model_name in models:
        cfg = V.MODELS.get(model_name)
        if cfg is None:
            continue
        md = load_model_dispatch(model_name, cfg)
        if md is None:
            print(f"  skip {model_name} (load failed)"); continue
        all_ts = V.extract_model_all_stations(md, cfg, station_data,
                                              method=V.INTERPOLATION_METHOD)
        for sid, obs in station_data.items():
            grp = V.STATIONS.get(sid, {}).get('group', 'OTHER')
            make_plots = (grp in V.PLOT_FIGURES_FOR_GROUPS) or (sid in cwop_sample)
            if not make_plots:
                continue
            mts = all_ts[sid]
            spd_m = V._clip_to_physical_bounds(mts['speed10'], 'Wind Speed [m/s]')
            spd_o = V._clip_to_physical_bounds(obs['speed10'], 'Wind Speed [m/s]')
            dir_m = V._clip_to_physical_bounds(mts['dir_deg'], 'Wind Direction [deg]')
            dir_o = V._clip_to_physical_bounds(obs['dir_deg'], 'Wind Direction [deg]')
            spd_m_all, spd_o_all, t_spd = V.match_timeseries(spd_m, mts['time'], spd_o, obs['time'])
            dir_m_all, dir_o_all, t_dir = V.match_timeseries(dir_m, mts['time'], dir_o, obs['time'])
            if spd_m_all is None or dir_m_all is None:
                continue
            t_common = np.intersect1d(t_spd, t_dir)
            if len(t_common) <= 50:
                continue
            ms = np.isin(t_spd, t_common); mdir = np.isin(t_dir, t_common)
            mout = out_root / grp / model_name
            mout.mkdir(parents=True, exist_ok=True)
            V.plot_wind_rose(spd_m_all[ms], dir_m_all[mdir],
                             spd_o_all[ms], dir_o_all[mdir],
                             sid, model_name, mout)
            print(f"    rose: {model_name} x {sid}", flush=True)
        for d in md.get('datasets', []):
            try: d.close()
            except Exception: pass


only = os.environ.get('ONLY_ERA')
for e in ([only] if only else ['1', '2', '3']):
    do_era(e)
print("\nWIND ROSES DONE.")
