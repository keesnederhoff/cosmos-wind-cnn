"""
figstyle.py - Kees Nederhoff's matplotlib house style for coastal model figures.

Import and call apply_style() before plotting. The helpers cover the corrections
that otherwise get requested by hand on nearly every figure: kilometre axes,
scale bar, north arrow, symmetric limits on change fields, muted basemaps, and
the right export settings per purpose.

Dependencies: matplotlib, numpy. cmocean is used when present and silently
falls back to matplotlib colormaps when not.

Usage
-----
    from figstyle import apply_style, km_axes, add_scalebar, add_north_arrow, save_figure

    apply_style()
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    ax.pcolormesh(x, y, dz, **change_kwargs(dz, vmax=0.1))
    km_axes(ax)
    add_scalebar(ax, 2000)
    add_north_arrow(ax)
    save_figure(fig, "bed_change", purpose="publication")
"""

from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_FONT = 11          # axis tick labels; everything else scales off this
LABEL_FONT = 12         # axis labels
PANEL_FONT = 12         # (a), (b) panel tags
LEGEND_FONT = 10

DPI = {                 # export resolution per purpose
    "quicklook": 150,
    "report": 300,
    "publication": 600,
}

# Colormap per physical field. Change fields must be diverging so the sign reads.
CMAPS = {
    "bed_level":   ("cmo.deep_r", "terrain"),
    "bed_change":  ("cmo.balance", "RdBu_r"),
    "depth":       ("cmo.deep", "Blues"),
    "hs":          ("cmo.amp", "magma"),
    "velocity":    ("cmo.speed", "viridis"),
    "salinity":    ("cmo.haline", "viridis"),
    "temperature": ("cmo.thermal", "inferno"),
    "generic":     ("viridis", "viridis"),
}

# Three maximally distinct colours for categorical flood-extent overlays.
CATEGORICAL = ["#1f77b4", "#d62728", "#2ca02c"]   # blue, red, green


# ============================================================================
# STYLE
# ============================================================================

def apply_style() -> None:
    """Set rcParams. Call once, before creating figures."""
    mpl.rcParams.update({
        "font.size":          BASE_FONT,
        "axes.labelsize":     LABEL_FONT,
        "axes.titlesize":     PANEL_FONT,
        "xtick.labelsize":    BASE_FONT,
        "ytick.labelsize":    BASE_FONT,
        "legend.fontsize":    LEGEND_FONT,
        "figure.titlesize":   LABEL_FONT,   # suptitle stays small; prefer none at all
        "axes.grid":          False,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "figure.constrained_layout.use": True,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype":       42,           # editable text in Illustrator, not outlines
        "ps.fonttype":        42,
    })


def get_cmap(field: str = "generic"):
    """Colormap for a named field, preferring cmocean when installed."""
    preferred, fallback = CMAPS.get(field, CMAPS["generic"])
    try:
        import cmocean  # noqa: F401  (registers the cmo.* names)
        return plt.get_cmap(preferred)
    except Exception:
        return plt.get_cmap(fallback)


# ============================================================================
# AXES HELPERS
# ============================================================================

def km_axes(ax, xlabel: str = "Easting (km)", ylabel: str = "Northing (km)") -> None:
    """Relabel projected metre axes as kilometres without rescaling the data.

    Keeps the underlying coordinates in metres so overlays and picking still
    work; only the tick labels change.
    """
    fmt = mpl.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def fill_frame(ax, x, y, margin: float = 0.02) -> None:
    """Set limits to the data extent plus a small margin, instead of autoscale padding."""
    x, y = np.asarray(x), np.asarray(y)
    dx = (np.nanmax(x) - np.nanmin(x)) * margin
    dy = (np.nanmax(y) - np.nanmin(y)) * margin
    ax.set_xlim(np.nanmin(x) - dx, np.nanmax(x) + dx)
    ax.set_ylim(np.nanmin(y) - dy, np.nanmax(y) + dy)
    ax.set_aspect("equal")


def diverging_limits(data, percentile: float = 99.0) -> tuple:
    """Symmetric (vmin, vmax) about zero for a change field.

    Uses a percentile rather than the max so a handful of extreme cells do not
    flatten the whole map to mid-colour.
    """
    v = float(np.nanpercentile(np.abs(np.asarray(data, dtype=float)), percentile))
    return (-v, v)


def change_kwargs(data, vmax: Optional[float] = None, percentile: float = 99.0) -> dict:
    """pcolormesh/imshow kwargs for a change field: diverging map, centred on zero."""
    if vmax is None:
        vmin, vmax = diverging_limits(data, percentile)
    else:
        vmin = -abs(vmax)
        vmax = abs(vmax)
    return dict(cmap=get_cmap("bed_change"), vmin=vmin, vmax=vmax)


# ============================================================================
# MAP FURNITURE
# ============================================================================

def add_scalebar(ax, length_m: float, location: str = "lower right",
                 pad_frac: float = 0.06, height_frac: float = 0.012,
                 label: Optional[str] = None) -> None:
    """Draw a filled scale bar in data (metre) coordinates."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    w, h = x1 - x0, y1 - y0

    bx = x1 - pad_frac * w - length_m if "right" in location else x0 + pad_frac * w
    by = y0 + pad_frac * h if "lower" in location else y1 - pad_frac * h

    ax.add_patch(Rectangle((bx, by), length_m, height_frac * h,
                           facecolor="black", edgecolor="black",
                           zorder=10, clip_on=False))
    if label is None:
        label = f"{length_m / 1000:g} km" if length_m >= 1000 else f"{length_m:g} m"
    ax.text(bx + length_m / 2, by + height_frac * h * 1.6, label,
            ha="center", va="bottom", fontsize=BASE_FONT, zorder=10)


def add_north_arrow(ax, location: str = "upper right",
                    pad_frac: float = 0.06, size_frac: float = 0.06) -> None:
    """Draw a simple N arrow. Assumes a north-up projected CRS."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    w, h = x1 - x0, y1 - y0

    ax_x = x1 - pad_frac * w if "right" in location else x0 + pad_frac * w
    ax_y = y1 - pad_frac * h if "upper" in location else y0 + pad_frac * h
    length = size_frac * h

    ax.annotate("", xy=(ax_x, ax_y), xytext=(ax_x, ax_y - length),
                arrowprops=dict(facecolor="black", edgecolor="black",
                                width=2, headwidth=9, headlength=9),
                zorder=10, annotation_clip=False)
    ax.text(ax_x, ax_y + 0.01 * h, "N", ha="center", va="bottom",
            fontsize=BASE_FONT + 1, fontweight="bold", zorder=10)


def mute_basemap(artist, alpha: float = 0.45, zorder: int = 0) -> None:
    """Push a basemap or geoimage behind the data and desaturate it.

    Full-strength basemaps compete with the data; this is requested by hand
    otherwise.
    """
    artist.set_alpha(alpha)
    artist.set_zorder(zorder)


def strip_attribution(ax) -> None:
    """Remove contextily/OSM credit text drawn into the axes.

    Attribution belongs in the caption for anything published; in the corner it
    is clutter.
    """
    for t in list(ax.texts):
        s = t.get_text().lower()
        if any(k in s for k in ("openstreetmap", "contextily", "(c) ", "©", "tiles")):
            t.remove()


def panel_tags(axes: Sequence, start: int = 0, loc: tuple = (0.02, 0.97),
               skip_blank: bool = True) -> None:
    """Label panels (a), (b), ... inside the axes rather than as titles.

    Blank placeholder panels (axis turned off, kept so the grid shape stays
    fixed across figures) are skipped by default but still consume a letter, so
    a given scenario keeps the same tag in every figure. Pass skip_blank=False
    to letter them contiguously instead.
    """
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        if skip_blank and not ax.axison:
            continue
        ax.text(*loc, f"({chr(ord('a') + start + i)})", transform=ax.transAxes,
                ha="left", va="top", fontsize=PANEL_FONT, fontweight="bold", zorder=11)


# ============================================================================
# EXPORT
# ============================================================================

def save_figure(fig, name: str, outdir: Path = Path("."),
                purpose: str = "quicklook", rasterize_data: bool = True) -> list:
    """Save at the resolution the purpose needs; add vector PDF for publication.

    Returns the paths written.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dpi = DPI.get(purpose, DPI["quicklook"])
    written = []

    if rasterize_data and purpose == "publication":
        # Keep text and lines vector; rasterise heavy meshes so the PDF stays small.
        for ax in fig.get_axes():
            for coll in ax.collections:
                coll.set_rasterized(True)

    png = outdir / f"{name}.png"
    fig.savefig(png, dpi=dpi)
    written.append(png)

    if purpose == "publication":
        pdf = outdir / f"{name}.pdf"
        fig.savefig(pdf, dpi=dpi)
        written.append(pdf)

    return written
