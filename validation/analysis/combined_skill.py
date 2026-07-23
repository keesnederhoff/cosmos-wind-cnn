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
WEIGHTS = {'USGS': 2.0, 'NDBC': 1.0, 'IEM': 1.0, 'CWOP': 0.5}  # category weights
BASE = config.OUTPUT_ROOT
ERA_DIRS = {
    'Era 1  1990-2010': 'era1_1990-2010',
    'Era 2  2011-2021': 'era2_2011-2021',
    'Era 3  2022-present': 'era3_2022-present',
}
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
# ===========================================================================


def model_color_map(models):
    cmap = {}
    for i, m in enumerate(sorted(m for m in set(models) if m not in MODEL_COLORS)):
        cmap[m] = _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)]
    for m in models:
        if m in MODEL_COLORS:
            cmap[m] = MODEL_COLORS[m]
    return cmap


# ---- Murphy-skill variables (speed / u10 / v10) ---------------------------
def _cat_stats(g):
    n = g['n'].values.astype(float); W = n.sum()
    mse = np.sum(n * g['rmse'].values ** 2) / W
    cmse = np.sum(n * (g['rmse'].values ** 2 - g['bias'].values ** 2)) / W  # centered (bias-removed)
    var = np.sum(n * g['obs_std'].values ** 2) / W
    bias = np.sum(n * g['bias'].values) / W
    rz = np.sum(n * np.arctanh(np.clip(g['corr'].values, -0.999, 0.999))) / W
    stdr = np.sum(n * (g['model_std'].values / g['obs_std'].values)) / W
    return dict(mse=mse, cmse=cmse, var=var, bias=bias, rz=rz, stdr=stdr,
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
        cats='+'.join(f"{c}({cats[c]['n_sta']})" for c in sorted(cats)))


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


def taylor(ax, models_xy, colors):
    for r in [0.5, 1.0, 1.5, 2.0]:
        ax.plot(np.linspace(0, np.pi / 2, 100), [r] * 100, color='0.85', lw=0.6, zorder=0)
    for Rc in [0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99]:
        ax.plot([np.arccos(Rc)] * 2, [0, 2.2], color='0.85', lw=0.6, zorder=0)
        ax.text(np.arccos(Rc), 2.25, f'{Rc:g}', fontsize=7, color='0.4', ha='center')
    ax.plot(0, 1.0, 'k*', ms=16, zorder=5)
    for m, (R, s) in models_xy.items():
        ax.plot(np.arccos(np.clip(R, -1, 1)), s, 'o', ms=11,
                color=colors.get(m, 'gray'), mec='white', mew=0.6, zorder=4)
    ax.set_thetamin(0); ax.set_thetamax(90)
    ax.set_rmax(2.3); ax.set_rticks([0.5, 1.0, 1.5, 2.0]); ax.set_rlabel_position(95)
    ax.text(np.pi / 4, 2.55, 'correlation', fontsize=9, ha='center', color='0.3')
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
        print(f"\n  [{var}]  (pooled Murphy skill; skill_dm = bias-removed)")
        print(f"  {'model':<12} {'skill':>7} {'skill_dm':>8} {'rmse':>6} {'bias':>6} {'corr':>6} {'std*':>6}  categories")
        for m, c in res.items():
            print(f"  {m:<12} {c['skill']:>7.3f} {c['skill_dm']:>8.3f} {c['rmse']:>6.2f} {c['bias']:>+6.2f} "
                  f"{c['corr']:>6.3f} {c['std_ratio']:>6.2f}  {c['cats']}")
            rows_all.append({'era': label, 'variable': var, 'model': m, **c})
        colors = model_color_map(list(res)); models = list(res)
        _bar(models, [res[m]['skill'] for m in models], colors,
             'combined Murphy skill (pooled, weighted)',
             f'{label} — combined {var} skill\nweights {WEIGHTS}',
             BASE / d / f'combined_skill_{key}.png')
        fig = plt.figure(figsize=(8, 8)); ax = fig.add_subplot(111, polar=True)
        taylor(ax, {m: (res[m]['corr'], res[m]['std_ratio']) for m in models}, colors)
        h = [Line2D([0], [0], marker='*', color='k', ms=13, ls='None', label='Obs (ref)')]
        h += [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[m], ms=10,
                     ls='None', label=m) for m in models]
        ax.legend(handles=h, loc='upper right', bbox_to_anchor=(1.32, 1.05), fontsize=9)
        ax.set_title(f'{label} — combined Taylor: {var}\nweights {WEIGHTS}', fontsize=10, pad=24)
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
    (config.OUTPUT_ROOT / 'rankings').mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_all).to_csv(config.OUTPUT_ROOT / 'rankings' / 'combined_skill_weighted.csv', index=False)
    print(f"\nWrote {config.OUTPUT_ROOT / 'rankings' / 'combined_skill_weighted.csv'}")
print("\nDONE.")
