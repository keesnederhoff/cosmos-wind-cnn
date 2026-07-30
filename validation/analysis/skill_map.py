"""Station skill maps: where each product wins and loses across the Bay.

One panel per product, stations plotted at their real coordinates and coloured by
Murphy skill (diverging about the cross-product median so differences are visible --
an absolute 0-1 scale washes them out). Marker size encodes sample count, marker
shape encodes network so the USGS moorings stay identifiable.

Coordinates come from reference/station_inventory.csv (lat/lon/name/area); skills from
each era's validation_statistics.csv. Nothing is re-scored -- this only re-plots.
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

VAR = 'Wind Speed [m/s]'
ERA_DIRS = {'Era A  2011-2019': 'eraA_2011-2019', 'Era B  2020-2025': 'eraB_2020-2025'}
MARKERS = {'IEM': 'o', 'NDBC': 's', 'USGS': 'D'}
# Panel order: baseline, target, then the CNN products.
ORDER = ['ERA5', 'CONUS404', 'RTMA-SFbay', 'CNN-allvars', 'CNN-allvars-BC',
         'CNN-windonly-BC', 'CNN-extreme-BC', 'CNN-wave-p2-BC', 'CNN-wave-p3-BC']


def load_coords():
    """Station lat/lon for every network.

    Source is the loaded registry, NOT reference/station_inventory.csv -- that file
    only lists the 20 IEM stations, so it silently drops every NDBC and USGS site
    (Alameda included). The registry resolves coordinates from the archive NetCDFs
    and KNOWN_STATION_COORDINATES, covering all 45.
    """
    os.environ.setdefault('VAL_USGS', '1')   # registry must include the moorings
    import validate_met_models as V
    rows = [{'station_id': sid, 'lat': cfg['lat'], 'lon': cfg['lon']}
            for sid, cfg in V.STATIONS.items()
            if cfg.get('lat') is not None and cfg.get('lon') is not None]
    df = pd.DataFrame(rows).set_index('station_id')
    # Friendly names where the inventory has them (IEM only); harmless if absent.
    inv_fp = config.REFERENCE_DIR / 'station_inventory.csv'
    if inv_fp.exists():
        inv = pd.read_csv(inv_fp).set_index('station_id')
        for col in ('name', 'area'):
            if col in inv.columns:
                df[col] = inv[col]
    return df


def coastline_lonlat():
    """Coastline for context, converted from the UTM-10 land-sea mask to lon/lat."""
    try:
        import validate_met_models as V
        polys = V.coastline_from_landsea(config.LANDSEA_FILE)
    except Exception as e:
        print(f"  (no coastline: {type(e).__name__}: {e})")
        return []
    out = []
    for x, y in polys:
        lon, lat = utm10_to_lonlat(np.asarray(x), np.asarray(y))
        out.append((lon, lat))
    return out


def utm10_to_lonlat(x, y):
    """UTM zone 10N -> lon/lat (WGS84), inverse transverse Mercator."""
    k0, a, f = 0.9996, 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    e1 = (1 - np.sqrt(1 - e2)) / (1 + np.sqrt(1 - e2))
    m = (y - 0.0) / k0
    mu = m / (a * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))
    phi1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * np.sin(2 * mu)
            + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * np.sin(4 * mu)
            + (151 * e1 ** 3 / 96) * np.sin(6 * mu))
    ep2 = e2 / (1 - e2)
    c1 = ep2 * np.cos(phi1) ** 2
    t1 = np.tan(phi1) ** 2
    n1 = a / np.sqrt(1 - e2 * np.sin(phi1) ** 2)
    r1 = a * (1 - e2) / (1 - e2 * np.sin(phi1) ** 2) ** 1.5
    d = (x - 500000.0) / (n1 * k0)
    lat = phi1 - (n1 * np.tan(phi1) / r1) * (
        d ** 2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * ep2) * d ** 4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * ep2 - 3 * c1 ** 2) * d ** 6 / 720)
    lon = (d - (1 + 2 * t1 + c1) * d ** 3 / 6
           + (5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * ep2 + 24 * t1 ** 2) * d ** 5 / 120) / np.cos(phi1)
    return np.degrees(lon) - 123.0, np.degrees(lat)


THREE = ['ERA5', 'CNN-windonly-BC', 'RTMA-SFbay']   # input -> downscaled -> target
THREE_LABEL = {'ERA5': 'ERA5  ~31 km', 'CNN-windonly-BC': 'CNN  2.5 km',
               'RTMA-SFbay': 'RTMA  2.5 km'}


def three_panel(label, era_dir, coords, coast, out_root):
    """ERA5 / CNN / RTMA only -- the win-lose comparison that matters.

    Shares one diverging colour scale across the three panels so the panels are
    comparable; centred on the across-panel median rather than 0 so the spread is
    visible (an absolute 0-1 scale makes all three look alike).
    """
    fp = config.OUTPUT_ROOT / era_dir / 'validation_statistics.csv'
    if not fp.exists():
        print(f"{label}: no CSV, skipping")
        return
    df = pd.read_csv(fp)
    df = df[(df['variable'] == VAR) & (~df['station'].astype(str).str.contains('MEAN'))
            & df['source'].isin(MARKERS) & df['model'].isin(THREE)]
    if df.empty:
        return
    vals = pd.to_numeric(df['skill'], errors='coerce').dropna().to_numpy()
    ctr = float(np.median(vals))
    half = float(max(0.15, np.percentile(np.abs(vals - ctr), 90)))
    vmin, vmax = ctr - half, ctr + half

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.4), squeeze=False)
    sc = None
    for ax, m in zip(axes.ravel(), THREE):
        sub = df[df['model'] == m]
        for src, mk in MARKERS.items():
            s2 = sub[sub['source'] == src]
            if s2.empty:
                continue
            j = s2.join(coords, on='station', how='inner')
            if j.empty:
                continue
            n = j['n'].astype(float)
            sc = ax.scatter(j['lon'], j['lat'], c=j['skill'].astype(float),
                            s=26 + 70 * np.sqrt(n / n.max()), marker=mk,
                            cmap='RdYlBu_r', vmin=vmin, vmax=vmax,
                            edgecolor='k', linewidth=0.5, zorder=3)
        for lon, lat in coast:
            ax.plot(lon, lat, color='0.55', lw=0.5, zorder=1)
        med = pd.to_numeric(sub['skill'], errors='coerce').median()
        ax.set_title(f"{THREE_LABEL.get(m, m)}\nmedian skill {med:+.3f}", fontsize=12)
        ax.set_xlim(-122.75, -121.6); ax.set_ylim(37.2, 38.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect(1 / np.cos(np.radians(37.85)))
    if sc is not None:
        cb = fig.colorbar(sc, ax=axes, shrink=0.75, pad=0.015)
        cb.set_label('Murphy skill vs observations (wind speed)')
    handles = [plt.Line2D([0], [0], marker=mk, ls='None', color='0.3',
                          markeredgecolor='k', label=src)
               for src, mk in MARKERS.items()
               if not df[df['source'] == src].empty]
    axes.ravel()[0].legend(handles=handles, loc='upper left', fontsize=8,
                           title='network', title_fontsize=8)
    fig.suptitle(f'{label}  -  wind-speed skill by station   '
                 f'(marker size ~ sample count)', fontsize=14, x=0.02, ha='left')
    fp_out = out_root / f'skill_map_3panel_{era_dir}.png'
    fig.savefig(fp_out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {fp_out}")


def main():
    out_root = config.OUTPUT_ROOT / 'skill_maps'
    out_root.mkdir(parents=True, exist_ok=True)
    coords = load_coords()
    coast = coastline_lonlat()

    for label, d in ERA_DIRS.items():
        fp = config.OUTPUT_ROOT / d / 'validation_statistics.csv'
        if not fp.exists():
            print(f"{label}: no CSV, skipping")
            continue
        df = pd.read_csv(fp)
        df = df[(df['variable'] == VAR) & (~df['station'].astype(str).str.contains('MEAN'))
                & df['source'].isin(MARKERS)]
        models = [m for m in ORDER if m in set(df['model'])]
        if not models:
            continue

        # Diverging scale centred on the all-product median keeps the between-product
        # differences readable; a fixed 0-1 scale makes every panel look the same.
        vals = df['skill'].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        ctr = float(np.median(vals))
        half = float(max(0.15, np.percentile(np.abs(vals - ctr), 90)))
        vmin, vmax = ctr - half, ctr + half

        ncol = 3
        nrow = int(np.ceil(len(models) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.3 * nrow),
                                 squeeze=False)
        sc = None
        for ax, m in zip(axes.ravel(), models):
            sub = df[df['model'] == m]
            for src, mk in MARKERS.items():
                s2 = sub[sub['source'] == src]
                if s2.empty:
                    continue
                j = s2.join(coords, on='station', how='inner')
                if j.empty:
                    continue
                sc = ax.scatter(j['lon'], j['lat'], c=j['skill'].astype(float),
                                s=18 + 42 * np.sqrt(j['n'].astype(float) / j['n'].astype(float).max()),
                                marker=mk, cmap='RdYlBu_r', vmin=vmin, vmax=vmax,
                                edgecolor='k', linewidth=0.4, zorder=3)
            for lon, lat in coast:
                ax.plot(lon, lat, color='0.55', lw=0.5, zorder=1)
            med = pd.to_numeric(sub['skill'], errors='coerce').median()
            ax.set_title(f"{m}\nmedian skill {med:+.3f}", fontsize=9)
            ax.set_xlim(-122.75, -121.6); ax.set_ylim(37.2, 38.5)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect(1 / np.cos(np.radians(37.85)))
        for ax in axes.ravel()[len(models):]:
            ax.axis('off')
        if sc is not None:
            cb = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.02)
            cb.set_label('Murphy skill vs observations (wind speed)')
        handles = [plt.Line2D([0], [0], marker=mk, ls='None', color='0.3',
                              markeredgecolor='k', label=src)
                   for src, mk in MARKERS.items()]
        axes.ravel()[0].legend(handles=handles, loc='upper left', fontsize=7,
                               title='network', title_fontsize=7)
        fig.suptitle(f'{label} - wind-speed skill by station '
                     f'(marker size ~ sample count)', fontsize=12)
        fp_out = out_root / f'skill_map_{d}.png'
        fig.savefig(fp_out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"wrote {fp_out}  ({len(models)} products)")

        three_panel(label, d, coords, coast, out_root)


if __name__ == '__main__':
    main()
