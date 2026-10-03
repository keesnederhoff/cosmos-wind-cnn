"""Combined cross-category multi-model comparison + Taylor diagrams.

Combines the source categories (USGS / NDBC / IEM / CWOP) into ONE score per
model per variable, using category WEIGHTS. The combined skill is a *pooled*
score (ratio of weighted sums of MSE and variance), NOT a mean of per-station
Murphy skills -- that keeps noisy low-variance CWOP stations from blowing the
average up.

  within category : pool stations by sample size n  (proper error pooling)
  across category : apply WEIGHTS  (equal => each category counts the same;
                    USGS up-weighted because it forces the Bay model directly)

Variables:
  Wind Speed / U10 / V10 -> pooled Murphy skill (+ Taylor): obs variance defined.
  Wind Direction         -> CIRCULAR statistics: Murphy skill / std ratio are not
                            meaningful, so we combine circular RMSE (deg) + circular
                            correlation instead (lower RMSE = better).

Reads the existing per-era validation_statistics.csv -- no model re-extraction.
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root on path
import config

# === CONFIGURATION =========================================================
# Category weights, overridable from the environment:
#     VAL_WEIGHTS="IEM:1,NDBC:1"      -> pool IEM+NDBC only (USGS reported separately)
#     VAL_WEIGHTS="IEM:1,NDBC:1,USGS:1"
# A category absent from WEIGHTS is dropped from the pooled score entirely.
# NOTE the default weights every category equally and drops CWOP entirely
# (2026-08-28). Pass VAL_WEIGHTS to reproduce an older run's weighting exactly,
# e.g. VAL_WEIGHTS="USGS:2,NDBC:1,IEM:1,CWOP:0.5" for the pre-2026-08-28 default.
_DEFAULT_WEIGHTS = {'IEM': 1.0, 'NDBC': 1.0, 'USGS': 1.0}
_wenv = os.environ.get('VAL_WEIGHTS', '').strip()
WEIGHTS = ({kv.split(':')[0].strip(): float(kv.split(':')[1])
            for kv in _wenv.split(',') if ':' in kv} if _wenv else dict(_DEFAULT_WEIGHTS))
# Output subdirectory for the rankings CSV (so a second pass can write elsewhere).
RANK_DIR = os.environ.get('VAL_RANK_DIR', 'rankings')
BASE = config.OUTPUT_ROOT
ERA_DIRS = {
    'Era 1  1990-2010': 'era1_1990-2010',
    'Era 2  2011-2021': 'era2_2011-2021',
    'Era 3  2022-present': 'era3_2022-present',
    'Era A  2011-2019': 'eraA_2011-2019',
    'Era B  2020-2025': 'eraB_2020-2025',
    # Three-era obs track (2026-08-14). E1 predates RTMA entirely.
    'Obs E1  2000-2010': 'obsE1_2000-2010',
    'Obs E2  2011-2019': 'obsE2_2011-2019',
    'Obs E3  2020-2026': 'obsE3_2020-2026',
}
# This dict being hardcoded is exactly how a campaign's ranking silently goes
# missing: a new era writes its per-station CSV, nothing here names the
# directory, and the combined pass produces no row for it without erroring.
# VAL_ERA_DIRS lets a run declare its own eras, e.g.
#     VAL_ERA_DIRS="Obs E3=obsE3_2020-2026_USGS__usgs"
_edenv = os.environ.get('VAL_ERA_DIRS', '').strip()
if _edenv:
    ERA_DIRS = {kv.split('=', 1)[0].strip(): kv.split('=', 1)[1].strip()
                for kv in _edenv.split(',') if '=' in kv}
# Murphy-skill variables (have a defined observed variance) and their file keys.
SKILL_VARS = [('Wind Speed [m/s]', 'speed'),
              ('Wind U10 [m/s]', 'u10'),
              ('Wind V10 [m/s]', 'v10'),
              ('Air Temperature [C]', 'temp'),
              ('Air Pressure [hPa]', 'pressure'),
              ('Dew Point [C]', 'dewpoint'),
              ('Relative Humidity [%]', 'rh'),
              ('Solar Radiation [W/m2]', 'radiation'),
              ('Precipitation [mm/hr]', 'precip')]
DIR_VAR = 'Wind Direction [deg]'

MODEL_COLORS = {
    'ERA5': 'tab:blue', 'HRRR': 'tab:red', 'CONUS404': 'tab:orange',
    'CNN': 'tab:green', 'UCLA': 'tab:purple', 'WRF_CalNev': 'tab:brown',
    'NOW-23': 'tab:pink', 'Sup3rWind': 'tab:olive', 'RTMA': 'tab:cyan',
    'CONUS404-downscaled': 'tab:gray', 'CONUS404-downscaled-100m': 'gold',
}
_FALLBACK_COLORS = ['black', 'magenta', 'teal', 'navy', 'crimson', 'darkgreen']

MODEL_MARKERS = {
    'CNN-quantile-v3': 'o', 'ERA5': 's', 'CONUS404': 'D', 'RTMA-SFbay': '^',
    'RTMA': '^', 'AORC': 'v', 'CNN': 'P',
}
_FALLBACK_MARKERS = ['X', '*', 'h', '8', 'p', '<', '>']
# ===========================================================================


def model_color_map(models):
    cmap = {}
    for i, m in enumerate(sorted(m for m in set(models) if m not in MODEL_COLORS)):
        cmap[m] = _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)]
    for m in models:
        if m in MODEL_COLORS:
            cmap[m] = MODEL_COLORS[m]
    return cmap


def model_marker_map(models):
    mmap = {}
    for i, m in enumerate(sorted(m for m in set(models) if m not in MODEL_MARKERS)):
        mmap[m] = _FALLBACK_MARKERS[i % len(_FALLBACK_MARKERS)]
    for m in models:
        if m in MODEL_MARKERS:
            mmap[m] = MODEL_MARKERS[m]
    return mmap


# ---- Murphy-skill variables (speed / u10 / v10) ---------------------------
def _nw_mean(vals, n):
    """Sample-size-weighted mean that ignores non-finite entries."""
    v = np.asarray(vals, dtype=float)
    m = np.isfinite(v)
    return float(np.sum(n[m] * v[m]) / n[m].sum()) if m.any() and n[m].sum() > 0 else np.nan


def _catavg(cats, w, key):
    """Weighted average across categories, skipping categories whose value is NaN."""
    ks = [c for c in cats if np.isfinite(cats[c][key])]
    if not ks:
        return np.nan
    tot = sum(w[c] for c in ks)
    return sum(w[c] * cats[c][key] for c in ks) / tot if tot > 0 else np.nan


def _cat_stats(g):
    n = g['n'].values.astype(float); W = n.sum()
    mse = np.sum(n * g['rmse'].values ** 2) / W
    cmse = np.sum(n * (g['rmse'].values ** 2 - g['bias'].values ** 2)) / W  # centered (bias-removed)
    var = np.sum(n * g['obs_std'].values ** 2) / W
    bias = np.sum(n * g['bias'].values) / W
    rz = np.sum(n * np.arctanh(np.clip(g['corr'].values, -0.999, 0.999))) / W
    stdr = np.sum(n * (g['model_std'].values / g['obs_std'].values)) / W
    # Energy-weighted Murphy skills (obs**q weighting; q=2 stress, q=3 energy).
    # These mirror the wave_exp p2/p3 training loss. They are NOT poolable the way
    # MSE/variance are -- the weighted sums are not recoverable from rmse/bias --
    # so carry the sample-size-weighted station MEAN and label it as such.
    ew1 = _nw_mean(g['skill_ew_u1'].values, n) if 'skill_ew_u1' in g else np.nan
    ew = _nw_mean(g['skill_ew'].values, n) if 'skill_ew' in g else np.nan
    ew3 = _nw_mean(g['skill_ew_u3'].values, n) if 'skill_ew_u3' in g else np.nan
    return dict(mse=mse, cmse=cmse, var=var, bias=bias, rz=rz, stdr=stdr,
                corr=np.tanh(rz), ew1=ew1, ew=ew, ew3=ew3,
                skill_c=(1.0 - mse / var if var > 0 else np.nan), n_sta=len(g))


def combine_skill(df_m):
    cats = {c: _cat_stats(g) for c, g in df_m.groupby('source') if c in WEIGHTS}
    if not cats:
        return None
    w = {c: WEIGHTS[c] for c in cats}; Wt = sum(w.values())
    num_mse = sum(w[c] * cats[c]['mse'] for c in cats)
    num_cmse = sum(w[c] * cats[c]['cmse'] for c in cats)   # per-station bias removed
    num_var = sum(w[c] * cats[c]['var'] for c in cats)
    return dict(
        skill=(1.0 - num_mse / num_var if num_var > 0 else np.nan),
        skill_dm=(1.0 - num_cmse / num_var if num_var > 0 else np.nan),  # bias-removed
        skill_catmean=sum(w[c] * cats[c]['skill_c'] for c in cats) / Wt,
        rmse=np.sqrt(num_mse / Wt),
        bias=sum(w[c] * cats[c]['bias'] for c in cats) / Wt,
        corr=np.tanh(sum(w[c] * cats[c]['rz'] for c in cats) / Wt),
        std_ratio=sum(w[c] * cats[c]['stdr'] for c in cats) / Wt,
        skill_ew_u1_mean=_catavg(cats, w, 'ew1'),  # station mean, q=1 (linear)
        skill_ew_mean=_catavg(cats, w, 'ew'),      # station mean, q=2 (stress ~U^2)
        skill_ew_u3_mean=_catavg(cats, w, 'ew3'),  # station mean, q=3 (energy ~U^3)
        cats='+'.join(f"{c}({cats[c]['n_sta']})" for c in sorted(cats)),
        cats_detail=cats)


# ---- Direction (circular) -------------------------------------------------
def _cat_stats_dir(g):
    n = g['n'].values.astype(float); W = n.sum()
    mse = np.sum(n * g['rmse'].values ** 2) / W          # rmse is circular (deg)
    mae = np.sum(n * g['mae'].values) / W
    bias = np.sum(n * g['bias'].values) / W
    corr = np.sum(n * g['corr'].values) / W              # circular corr
    return dict(mse=mse, mae=mae, bias=bias, corr=corr, n_sta=len(g))


def combine_dir(df_m):
    cats = {c: _cat_stats_dir(g) for c, g in df_m.groupby('source') if c in WEIGHTS}
    if not cats:
        return None
    w = {c: WEIGHTS[c] for c in cats}; Wt = sum(w.values())
    return dict(
        rmse=np.sqrt(sum(w[c] * cats[c]['mse'] for c in cats) / Wt),   # deg
        mae=sum(w[c] * cats[c]['mae'] for c in cats) / Wt,
        bias=sum(w[c] * cats[c]['bias'] for c in cats) / Wt,
        corr=sum(w[c] * cats[c]['corr'] for c in cats) / Wt,
        cats='+'.join(f"{c}({cats[c]['n_sta']})" for c in sorted(cats)))


def taylor(ax, models_xy, colors, markers, station_points=None, rmax=2.3):
    """Taylor diagram: per-station markers (small, transparent) plus one bold
    pooled/weighted marker per model, with centered-RMS-difference arcs about
    the (r=1, nstd=1) reference point.

    models_xy   : {model: (corr, std_ratio)} -- the bold marker position,
                  already pooled/weighted by combine_skill()/combine_dir().
    station_points : optional list of (model, corr, std_ratio) -- one row per
                  raw station observation, drawn as small alpha=0.32 markers
                  underneath the bold marker. None/[] draws no per-station layer.
    """
    ax.set_thetamin(0); ax.set_thetamax(90)
    ax.set_theta_zero_location('E'); ax.set_theta_direction(1)
    ax.set_rmax(rmax)

    # normalized-std-dev rings (constant radius, centered at origin)
    rticks = [r for r in [0.5, 1.0, 1.5, 2.0] if r <= rmax]
    for r in rticks:
        ax.plot(np.linspace(0, np.pi / 2, 100), [r] * 100, color='0.85', lw=0.6, zorder=0)
    ax.set_rticks(rticks); ax.set_rlabel_position(95)

    # correlation gridlines
    for Rc in [0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99]:
        ax.plot([np.arccos(Rc)] * 2, [0, rmax], color='0.85', lw=0.6, zorder=0)
        ax.text(np.arccos(Rc), rmax * 1.02, f'{Rc:g}', fontsize=7, color='0.4', ha='center')

    # centered RMS-difference arcs about the (r=1, nstd=1) reference point
    th = np.linspace(0, np.pi / 2, 400)
    for rms in [x for x in (0.5, 1.0, 1.5, 2.0) if x < rmax]:
        xx = 1.0 + rms * np.cos(th)
        yy = rms * np.sin(th)
        rr = np.hypot(xx, yy)
        tt = np.arctan2(yy, xx)
        keep = rr <= rmax
        ax.plot(tt[keep], rr[keep], ':', color='0.72', lw=0.8, zorder=1)
        if keep.any():
            idx = np.where(keep)[0][-1]
            ax.text(tt[idx], rr[idx], f'{rms:g}', color='0.55', fontsize=7,
                    ha='center', va='bottom', bbox=dict(fc='white', ec='none', pad=0.4))

    # dashed reference arc at nstd = 1, and the obs/gauge reference point
    ax.plot(np.linspace(0, np.pi / 2, 200), np.ones(200), '--', color='0.5', lw=0.9, zorder=2)
    ax.plot(0, 1.0, 'k*', ms=16, zorder=8)

    # per-station small markers (underneath the bold marker)
    for m, r, s in (station_points or []):
        ax.plot(np.arccos(np.clip(r, -1, 1)), s, markers.get(m, 'o'),
                color=colors.get(m, 'gray'), ms=3.2, alpha=0.32, mew=0, zorder=4)

    # bold pooled/weighted marker per model
    for m, (R, s) in models_xy.items():
        ax.plot(np.arccos(np.clip(R, -1, 1)), s, markers.get(m, 'o'), ms=11,
                color=colors.get(m, 'gray'), mec='white', mew=1.1, zorder=7)

    ax.text(np.pi / 4, rmax * 1.12, 'correlation', fontsize=9, ha='center', color='0.3')
    ax.set_xlabel('normalized standard deviation', fontsize=9)


def _bar(models, vals, colors, ylabel, title, fpath, baseline=0.0):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(models)), vals, color=[colors[m] for m in models],
           edgecolor='k', lw=0.5)
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=30, ha='right')
    if baseline is not None:
        ax.axhline(baseline, color='k', lw=0.8)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig(fpath, dpi=150); plt.close(fig)


if __name__ == '__main__':
    rows_all = []
    for label, d in ERA_DIRS.items():
        fp = BASE / d / 'validation_statistics.csv'
        if not fp.exists():
            print(f"{label}: no CSV"); continue
        raw = pd.read_csv(fp)
        raw = raw[(~raw['station'].astype(str).str.contains('MEAN')) & raw['source'].isin(WEIGHTS)]

        print("\n" + "=" * 84)
        print(f"{label}   weights={WEIGHTS}")
        print("=" * 84)

        # ---- Murphy-skill variables ----
        for var, key in SKILL_VARS:
            df = raw[raw['variable'] == var]
            df = df[(df['obs_std'] > 0.05) & (df['n'] >= 50)
                    & np.isfinite(df['rmse']) & np.isfinite(df['corr'])
                    & np.isfinite(df['obs_std']) & np.isfinite(df['model_std'])]
            if df.empty:
                continue
            res = {m: combine_skill(g) for m, g in df.groupby('model')}
            res = {m: c for m, c in res.items() if c}
            res = dict(sorted(res.items(), key=lambda kv: kv[1]['skill'], reverse=True))
            print(f"\n  [{var}]  (pooled Murphy skill; skill_dm = bias-removed; ew* = station-mean energy-weighted)")
            print(f"  {'model':<16} {'skill':>7} {'skill_dm':>8} {'ew_q1':>7} {'ew_q2':>7} {'ew_q3':>7} {'rmse':>6} {'bias':>6} {'corr':>6} {'std*':>6}  categories")
            for m, c in res.items():
                print(f"  {m:<16} {c['skill']:>7.3f} {c['skill_dm']:>8.3f} "
                      f"{c['skill_ew_u1_mean']:>7.3f} "
                      f"{c['skill_ew_mean']:>7.3f} {c['skill_ew_u3_mean']:>7.3f} "
                      f"{c['rmse']:>6.2f} {c['bias']:>+6.2f} "
                      f"{c['corr']:>6.3f} {c['std_ratio']:>6.2f}  {c['cats']}")
                rows_all.append({'era': label, 'variable': var, 'model': m, **c})
            colors = model_color_map(list(res)); markers = model_marker_map(list(res)); models = list(res)
            _bar(models, [res[m]['skill'] for m in models], colors,
                 'combined Murphy skill (pooled, weighted)',
                 f'{label} — combined {var} skill\nweights {WEIGHTS}',
                 BASE / d / f'combined_skill_{key}.png')

            station_points = [(m, r, s) for m, r, s in
                              zip(df['model'], df['corr'], df['model_std'] / df['obs_std'])]

            fig = plt.figure(figsize=(8, 8)); ax = fig.add_subplot(111, polar=True)
            taylor(ax, {m: (res[m]['corr'], res[m]['std_ratio']) for m in models}, colors, markers,
                   station_points=station_points)
            h = [Line2D([0], [0], marker='*', color='k', ms=13, ls='None', label='Obs (ref)')]
            h += [Line2D([0], [0], marker=markers[m], color='w', markerfacecolor=colors[m], ms=10,
                         ls='None', label=m) for m in models]
            ax.legend(handles=h, loc='upper right', bbox_to_anchor=(1.32, 1.05), fontsize=9)
            ax.set_title(f'{label} — combined Taylor: {var}\nweights {WEIGHTS}\n'
                         f'small = per-station, bold = pooled/weighted', fontsize=10, pad=28)
            fig.tight_layout(); fig.savefig(BASE / d / f'combined_taylor_{key}.png', dpi=150); plt.close(fig)

        # ---- Wind direction (circular) ----
        dfd = raw[raw['variable'] == DIR_VAR]
        dfd = dfd[(dfd['n'] >= 50) & np.isfinite(dfd['rmse']) & np.isfinite(dfd['corr'])]
        if not dfd.empty:
            resd = {m: combine_dir(g) for m, g in dfd.groupby('model')}
            resd = {m: c for m, c in resd.items() if c}
            resd = dict(sorted(resd.items(), key=lambda kv: kv[1]['rmse']))   # lower=better
            print(f"\n  [{DIR_VAR}]  (circular; ranked by RMSE, lower=better)")
            print(f"  {'model':<12} {'rmse[deg]':>9} {'mae[deg]':>9} {'bias[deg]':>9} {'circ_corr':>9}  categories")
            for m, c in resd.items():
                print(f"  {m:<12} {c['rmse']:>9.1f} {c['mae']:>9.1f} {c['bias']:>+9.1f} "
                      f"{c['corr']:>9.3f}  {c['cats']}")
                rows_all.append({'era': label, 'variable': DIR_VAR, 'model': m,
                                 'rmse': c['rmse'], 'mae': c['mae'], 'bias': c['bias'],
                                 'corr': c['corr'], 'cats': c['cats']})
            colors = model_color_map(list(resd)); models = list(resd)
            _bar(models, [resd[m]['rmse'] for m in models], colors,
                 'combined circular RMSE [deg] (lower = better)',
                 f'{label} — combined wind-direction RMSE\nweights {WEIGHTS}',
                 BASE / d / 'combined_dir_rmse.png', baseline=None)
        print(f"  -> figures saved in {d}")

    if rows_all:
        _rank = config.OUTPUT_ROOT / RANK_DIR
        _rank.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_all).to_csv(_rank / 'combined_skill_weighted.csv', index=False)
        print(f"\nWrote {_rank / 'combined_skill_weighted.csv'}  (weights={WEIGHTS})")
    print("\nDONE.")
