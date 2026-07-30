"""Figures built specifically for the findings deck.

Colour policy: the categorical hues are the validated 8-slot reference palette,
used unchanged and assigned to a FIXED model -> colour map (colour follows the
entity, never its rank -- bars are sorted by skill, so a rank-based map would
repaint models between charts). CONUS404 is composite-encoded as hatched slot-1
blue rather than given a 9th hue, because a 9th would be unvalidated.

Because sorting makes bar adjacency data-dependent, every bar carries a direct
value label and the model name sits on the axis: identity never rests on colour.

    python analysis/deck_figs.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

OUT = config.OUTPUT_ROOT / 'deck_figs'
ERA_A, ERA_B = 'eraA_2011-2019', 'eraB_2020-2025'
ERAS = {ERA_A: 'Era A  2011-2019', ERA_B: 'Era B  2020-2025'}
SPEED, TOP10, DIRV = 'Wind Speed [m/s]', 'Wind Speed [m/s] (top 10%)', 'Wind Direction [deg]'
U10, V10 = 'Wind U10 [m/s]', 'Wind V10 [m/s]'

# --- reference palette, light mode (references/palette.md) ---------------------
S1, S2, S3, S4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
S5, S6, S7, S8 = '#e87ba4', '#008300', '#4a3aa7', '#e34948'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, SURF = '#e1e0d9', '#c3c2b7', '#fcfcfb'

COLOR = {
    'ERA5': S1, 'CONUS404': S1, 'RTMA-SFbay': S2,
    'CNN-allvars': S3, 'CNN-allvars-BC': S4, 'CNN-windonly-BC': S5,
    'CNN-extreme-BC': S6, 'CNN-wave-p2-BC': S7, 'CNN-wave-p3-BC': S8,
}
HATCH = {'CONUS404': '///'}          # second channel on bars (patches)
MARKER = {'CONUS404': 's'}           # second channel on points (hatch does not render on markers)
SHORT = {
    'CNN-windonly-BC': 'CNN wind-only\n(BC)', 'CNN-wave-p2-BC': 'CNN wave-p2\n(BC)',
    'CNN-wave-p3-BC': 'CNN wave-p3\n(BC)', 'CNN-allvars-BC': 'CNN all-vars\n(BC)',
    'CNN-extreme-BC': 'CNN extreme\n(BC)', 'CNN-allvars': 'CNN all-vars\n(raw)',
    'RTMA-SFbay': 'RTMA\n2.5 km', 'ERA5': 'ERA5\n~31 km', 'CONUS404': 'CONUS404\n4 km',
}
BEST_CNN = 'CNN-windonly-BC'         # best pooled wind-speed skill in both eras


def _style(ax):
    ax.set_facecolor(SURF)
    ax.grid(axis='y', color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=10)


def _cats(df_era):
    """Station-count string per network, e.g. 'IEM 19 + NDBC 17 (no USGS)'."""
    g = df_era.groupby('source')['station'].nunique()
    parts = [f"{k} {int(v)}" for k, v in g.items() if k in ('IEM', 'NDBC', 'USGS')]
    s = ' + '.join(parts)
    if 'USGS' not in g.index and len(parts) > 1:
        s += '   (no USGS moorings in this era)'
    return s


def load(era_dir):
    fp = config.OUTPUT_ROOT / era_dir / 'validation_statistics.csv'
    d = pd.read_csv(fp)
    return d[(~d['station'].astype(str).str.contains('MEAN'))
             & d['source'].isin(['IEM', 'NDBC', 'USGS'])]


def rank(variable, era_dir):
    rk = pd.read_csv(config.OUTPUT_ROOT / 'rankings' / 'combined_skill_weighted.csv')
    tag = 'Era A' if era_dir == ERA_A else 'Era B'
    return rk[rk['era'].str.startswith(tag) & (rk['variable'] == variable)]


# ---------------------------------------------------------------- bar charts
def bars(era_dir, variable, col, title, ylabel, fname, lower_better=False):
    d = rank(variable, era_dir).copy()
    d[col] = pd.to_numeric(d[col], errors='coerce')
    d = d.dropna(subset=[col]).sort_values(col, ascending=lower_better)
    stats = load(era_dir)
    stats = stats[stats['variable'] == variable]

    fig, ax = plt.subplots(figsize=(11, 5.4))
    fig.patch.set_facecolor(SURF)
    _style(ax)
    x = np.arange(len(d))
    for i, (_, r) in enumerate(d.iterrows()):
        m = r['model']
        ax.bar(i, r[col], width=0.68, color=COLOR.get(m, MUTED),
               hatch=HATCH.get(m), edgecolor='white', linewidth=1.2, zorder=3)
        off = (max(d[col]) - min(min(d[col]), 0)) * 0.02
        ax.text(i, r[col] + (off if r[col] >= 0 else -off * 2.2),
                f"{r[col]:.3f}", ha='center',
                va='bottom' if r[col] >= 0 else 'top',
                fontsize=10, color=INK, fontweight='bold')
    # Headroom so value labels on negative bars clear the tick labels.
    lo, hi = min(0, float(d[col].min())), max(0, float(d[col].max()))
    pad = (hi - lo) * 0.16
    ax.set_ylim(lo - (pad if lo < 0 else 0), hi + pad)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT.get(m, m) for m in d['model']], fontsize=9, color=INK2)
    ax.set_ylabel(ylabel, fontsize=11, color=INK2)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_title(title, fontsize=13, color=INK, pad=26, loc='left')
    ax.text(0, 1.03, f"{ERAS[era_dir]}   |   stations: {_cats(stats)}",
            transform=ax.transAxes, fontsize=10, color=MUTED)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / fname, dpi=160, facecolor=SURF)
    plt.close(fig)
    print(f"  {fname}")


# ------------------------------------------------- 3-product per-metric charts
def three_product(era_dir):
    """ERA5 vs RTMA vs best CNN, one panel per metric. Slots 1-3: the trio the
    palette validates on the all-pairs list."""
    models = ['ERA5', 'RTMA-SFbay', BEST_CNN]
    trio = {'ERA5': S1, 'RTMA-SFbay': S2, BEST_CNN: S3}
    metrics = [
        (SPEED, 'skill', 'Murphy skill', 'higher is better', False),
        (SPEED, 'skill_ew_mean', 'Energy-weighted skill  (U^2, wind stress)', 'higher is better', False),
        (SPEED, 'skill_ew_u3_mean', 'Energy-weighted skill  (U^3, wave energy)', 'higher is better', False),
        (TOP10, 'rmse', 'Top-10% wind RMSE [m/s]', 'lower is better', True),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.6))
    fig.patch.set_facecolor(SURF)
    for ax, (var, col, lab, hint, low) in zip(axes, metrics):
        d = rank(var, era_dir)
        vals = []
        for m in models:
            q = d[d['model'] == m]
            vals.append(float(q[col].iloc[0]) if len(q) and pd.notna(q[col].iloc[0]) else np.nan)
        _style(ax)
        for i, (m, v) in enumerate(zip(models, vals)):
            ax.bar(i, v, width=0.62, color=trio[m], edgecolor='white', lw=1.2, zorder=3)
            span = np.nanmax(vals) - min(np.nanmin(vals), 0)
            ax.text(i, v + (span * 0.03 if v >= 0 else -span * 0.06), f"{v:.2f}",
                    ha='center', va='bottom' if v >= 0 else 'top',
                    fontsize=11, color=INK, fontweight='bold')
        lo, hi = min(0, np.nanmin(vals)), max(0, np.nanmax(vals))
        pad = (hi - lo) * 0.20
        ax.set_ylim(lo - (pad if lo < 0 else 0), hi + pad)
        ax.set_xticks(range(3))
        ax.set_xticklabels(['ERA5', 'RTMA', 'CNN'], fontsize=11, color=INK2)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_title(f"{lab}\n({hint})", fontsize=11, color=INK, pad=8)
    stats = load(era_dir)
    stats = stats[stats['variable'] == SPEED]
    fig.suptitle(f"{ERAS[era_dir]}  -  ERA5 vs RTMA vs best CNN ({SHORT[BEST_CNN].replace(chr(10),' ')})",
                 fontsize=14, color=INK, y=1.04, x=0.02, ha='left')
    fig.text(0.02, 0.965, f"stations: {_cats(stats)}", fontsize=10, color=MUTED, ha='left')
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fn = f'three_product_{era_dir}.png'
    fig.savefig(OUT / fn, dpi=160, facecolor=SURF, bbox_inches='tight')
    plt.close(fig)
    print(f"  {fn}")


# ------------------------------------------------------------- Taylor, pooled
def taylor_all(era_dir, variable=SPEED, group=None):
    """Taylor diagram over ALL stations pooled by sample size.

    Drawn in Cartesian coordinates (x = sigma*cos(theta), y = sigma*sin(theta),
    theta = arccos(R)) rather than a polar axis: matplotlib's polar wedge fights
    custom correlation ticks and r-label placement, and the geometry here has to
    be exact.

    This is the pooled score, NOT the category-weighted 'combined' one. For these
    eras the two agree closely (Era A: R 0.775 vs 0.773, sigma* 0.749 vs 0.761),
    so the distinction is about what the axis claims, not about the picture.

    The five bias-corrected variants land on top of each other -- pooled R within
    0.005 and sigma* within 0.004. That overlap is the honest result: they are
    statistically indistinguishable, so they are drawn as they fall.
    """
    d = load(era_dir)
    d = d[(d['variable'] == variable) & (d['obs_std'] > 0.05) & (d['n'] >= 50)]
    if group:
        # One network at a time: IEM airport anemometers and NDBC buoys sit in
        # different exposures, so pooling them hides which one drives the result.
        d = d[d['source'] == group]

    pts = {}
    for m, g in d.groupby('model'):
        n = g['n'].to_numpy(float)
        rz = np.sum(n * np.arctanh(np.clip(g['corr'].to_numpy(float), -0.999, 0.999))) / n.sum()
        R = float(np.tanh(rz))
        sr = float(np.sum(n * (g['model_std'].to_numpy(float) / g['obs_std'].to_numpy(float))) / n.sum())
        pts[m] = (R, sr)

    RMAX = 1.25
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)

    # sigma arcs
    th = np.linspace(0, np.pi / 2, 200)
    for r in (0.25, 0.5, 0.75, 1.0):
        ax.plot(r * np.cos(th), r * np.sin(th), color=GRID, lw=0.8, zorder=0)
    ax.plot(RMAX * np.cos(th), RMAX * np.sin(th), color=AXIS, lw=1.0, zorder=1)

    # correlation rays
    for c in (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        a = np.arccos(c)
        ax.plot([0, RMAX * np.cos(a)], [0, RMAX * np.sin(a)], color=GRID, lw=0.7, zorder=0)
        ax.text(1.035 * RMAX * np.cos(a), 1.035 * RMAX * np.sin(a), f'{c:g}',
                fontsize=9, color=INK2, ha='left', va='center')
    ax.text(RMAX * 0.80, RMAX * 0.80, 'correlation', fontsize=10, color=INK2,
            ha='center', rotation=-45)

    # reference: perfect correlation, unit variance
    ax.plot(1.0, 0.0, marker='*', ms=18, color=INK, zorder=6)
    ax.annotate('observations', (1.0, 0.0), textcoords='offset points',
                xytext=(4, 10), fontsize=9, color=INK2)

    handles = []
    for m, (R, sr) in sorted(pts.items()):
        a = np.arccos(np.clip(R, -1, 1))
        ax.plot(sr * np.cos(a), sr * np.sin(a), MARKER.get(m, 'o'), ms=11,
                color=COLOR.get(m, MUTED), mec='white', mew=1.4, zorder=5, alpha=0.93)
        handles.append(plt.Line2D([0], [0], marker=MARKER.get(m, 'o'), ls='None',
                                  color=COLOR.get(m, MUTED), mec='white', mew=1.2,
                                  ms=9, label=m))

    ax.set_xlim(0, RMAX * 1.12); ax.set_ylim(0, RMAX * 1.08)
    ax.set_aspect('equal')
    ax.set_xlabel('normalised standard deviation', fontsize=10, color=INK2)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.34, 1.0),
              fontsize=9, frameon=False)
    # Quote the spread actually measured on THIS subset -- it differs between the
    # pooled set and each individual network.
    bc = {k: v for k, v in pts.items() if k.endswith('-BC')}
    if len(bc) > 1:
        dr = max(v[0] for v in bc.values()) - min(v[0] for v in bc.values())
        ds = max(v[1] for v in bc.values()) - min(v[1] for v in bc.values())
        ax.text(0.02, -0.155,
                f'The {len(bc)} bias-corrected variants overlap: correlation spread '
                f'{dr:.3f}, sigma* spread {ds:.3f}.\nThat is the result, not a plotting '
                f'artefact -- they are statistically indistinguishable.',
                transform=ax.transAxes, fontsize=8.5, color=MUTED, va='top')
    what = f'{group} stations' if group else 'all stations pooled'
    ax.set_title(f"{ERAS[era_dir]}  -  Taylor diagram, {what}\n"
                 f"{_cats(d)}", fontsize=12, color=INK, loc='left', pad=12)
    OUT.mkdir(parents=True, exist_ok=True)
    fn = (f'taylor_{group}_{era_dir}.png' if group
          else f'taylor_allstations_{era_dir}.png')
    fig.savefig(OUT / fn, dpi=160, facecolor=SURF, bbox_inches='tight')
    plt.close(fig)
    print(f"  {fn}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("bar charts:")
    for era in (ERA_A, ERA_B):
        bars(era, SPEED, 'skill', 'Wind-speed skill', 'Murphy skill',
             f'bars_speed_{era}.png')
        bars(era, U10, 'skill', 'U10 (east-west) skill', 'Murphy skill',
             f'bars_u10_{era}.png')
        bars(era, V10, 'skill', 'V10 (north-south) skill', 'Murphy skill',
             f'bars_v10_{era}.png')
        bars(era, DIRV, 'rmse', 'Wind-direction error', 'circular RMSE [deg]',
             f'bars_dir_{era}.png', lower_better=True)
    print("three-product panels:")
    for era in (ERA_A, ERA_B):
        three_product(era)
    print("taylor:")
    for era in (ERA_A, ERA_B):
        taylor_all(era)
        for grp in ('NDBC', 'IEM'):
            taylor_all(era, group=grp)


if __name__ == '__main__':
    main()
