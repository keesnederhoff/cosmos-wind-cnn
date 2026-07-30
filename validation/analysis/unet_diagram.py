"""Architecture diagram of the 3D U-Net used for ERA5 -> RTMA downscaling.

Drawn from the actual configuration rather than a generic U-Net sketch:
  - input  : 8 channels (ERA5 u/v + temp/dew/pressure/rain + cloud + static terrain),
             interpolated onto the RTMA 2.5 km grid, 6 hourly steps of context
  - grid   : 162 x 123, encoder pools (1,2,2) so the TIME axis is never pooled
  - width  : base_channels 24, doubling each level
  - output : 2 channels (hr_u, hr_v) at the centre step, residual on the input wind

    python analysis/unet_diagram.py
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

S1, S2, S3, S4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, SURF = '#e1e0d9', '#c3c2b7', '#fcfcfb'

# Read off the trained checkpoint (os_wo_bc24_base_res_s2) and models/unet3d.py:
# 4 encoder levels enc1..enc4 at base*1,2,4,8 with base_channels=24, a base*16
# bottleneck, MaxPool3d((1,2,2)) so only H,W halve, and a 1x1 Conv2d head.
# (level, channels, HxW label)
ENC = [(0, 24, '123x162'), (1, 48, '61x81'), (2, 96, '30x40'), (3, 192, '15x20')]
BOT = (4, 384, '7x10')


def box(ax, x, y, w, h, color, label, sub=None, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.02',
                                fc=color, ec='white', lw=1.4, alpha=alpha, zorder=3))
    ax.text(x + w / 2, y + h / 2 + (0.018 if sub else 0), label, ha='center', va='center',
            fontsize=9.5, color='white', fontweight='bold', zorder=4)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.030, sub, ha='center', va='center',
                fontsize=8, color='white', alpha=0.95, zorder=4)


def arrow(ax, p0, p1, color=MUTED, style='-|>', lw=1.4, ls='-'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, color=color, lw=lw,
                                 linestyle=ls, mutation_scale=13,
                                 shrinkA=2, shrinkB=2, zorder=2))


def main():
    fig, ax = plt.subplots(figsize=(13.2, 6.2))
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    BW, BH = 0.075, 0.072
    ytop = 0.74
    dy = 0.138

    # ---- input -----------------------------------------------------------
    box(ax, 0.005, ytop, 0.085, BH, S1, 'ERA5', '8 ch | 6 steps')
    ax.text(0.048, ytop - 0.044,
            'u, v, temp, dewpoint,\npressure, rain, cloud,\nstatic terrain\n\ninterpolated onto the\nRTMA 2.5 km grid',
            ha='center', va='top', fontsize=7.4, color=INK2)

    # ---- encoder ---------------------------------------------------------
    xs_enc, xs_dec = [], []
    for i, (lvl, ch, hw) in enumerate(ENC):
        x = 0.098 + i * 0.085
        y = ytop - lvl * dy
        box(ax, x, y, BW, BH, S1, f'{ch} ch', hw)
        xs_enc.append((x, y))
        if i == 0:
            arrow(ax, (0.090, ytop + BH / 2), (x, y + BH / 2))
        else:
            px, py = xs_enc[i - 1]
            arrow(ax, (px + BW / 2, py), (x + BW / 2, y + BH))

    # ---- bottleneck ------------------------------------------------------
    lvl, ch, hw = BOT
    xb, yb = 0.098 + 4 * 0.085, ytop - lvl * dy
    box(ax, xb, yb, BW, BH, S4, f'{ch} ch', hw)
    px, py = xs_enc[-1]
    arrow(ax, (px + BW / 2, py), (xb + BW / 2, yb + BH))
    ax.text(xb + BW / 2, yb - 0.045, 'bottleneck', ha='center', fontsize=8, color=MUTED)

    # ---- decoder + skips -------------------------------------------------
    for i, (lvl, ch, hw) in enumerate(reversed(ENC)):
        x = xb + 0.085 * (i + 1)
        y = ytop - lvl * dy
        box(ax, x, y, BW, BH, S3, f'{ch} ch', hw)
        xs_dec.append((x, y))
        prev = (xb, yb) if i == 0 else xs_dec[i - 1]
        arrow(ax, (prev[0] + BW / 2, prev[1] + BH), (x + BW / 2, y))
        ex, ey = [e for e in xs_enc if abs(e[1] - y) < 1e-9][0]
        arrow(ax, (ex + BW, ey + BH / 2), (x, y + BH / 2), color=S2, ls='--', lw=1.3)

    # ---- output ----------------------------------------------------------
    xo = xs_dec[-1][0] + 0.085
    box(ax, xo, ytop, 0.092, BH, S2, 'output', '6 ch -> u, v')
    arrow(ax, (xs_dec[-1][0] + BW, ytop + BH / 2), (xo, ytop + BH / 2))
    ax.text(xo + 0.046, ytop - 0.044,
            'centre step of the\n6-step window\n2.5 km, 123x162',
            ha='center', va='top', fontsize=7.6, color=INK2)

    # residual connection
    arrow(ax, (0.048, ytop + BH + 0.012), (xo + 0.046, ytop + BH + 0.012),
          color=S2, ls=':', lw=1.6, style='-|>')
    ax.text((0.048 + xo + 0.046) / 2, ytop + BH + 0.028,
            'residual: the network learns the CORRECTION to the interpolated ERA5 wind, not the wind itself',
            ha='center', fontsize=8.4, color=S2, style='italic')

    # legend / notes
    notes = (
        'Pooling is MaxPool3d(1, 2, 2) -- only height and width halve, so the 6-step '
        'time context survives intact to the bottleneck.\n'
        'Orange dashed = skip connections, re-injecting encoder detail at matching '
        'resolution.  base_channels = 24, doubling per level.\n'
        'The head predicts all 6 variable pairs; the wind-only loss weights just u and v, '
        'so the other four channels are untrained by-products.\n'
        'Loss variants differ ONLY in what the loss weights: all-variables, wind-only, '
        'a hard >10 m/s term, or smooth wave-energy weighting (U^2 / U^3).')
    ax.text(0.012, 0.075, notes, fontsize=8.6, color=INK2, va='top', linespacing=1.7)

    ax.text(0.012, 0.965, '3D U-Net:  ERA5 (~31 km)  ->  RTMA resolution (2.5 km)',
            fontsize=14, color=INK, fontweight='bold', va='top')

    out = config.OUTPUT_ROOT / 'deck_figs'
    out.mkdir(parents=True, exist_ok=True)
    fp = out / 'unet_architecture.png'
    fig.savefig(fp, dpi=170, facecolor=SURF, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {fp}")


if __name__ == '__main__':
    main()
