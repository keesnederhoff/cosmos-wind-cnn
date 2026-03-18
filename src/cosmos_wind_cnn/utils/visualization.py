"""
Visualization utilities for evaluation and analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_sample_predictions(preds, targets, inputs, output_vars, coords, save_dir, n_samples=5):
    """Plot sample predictions vs targets for all variables."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    indices = np.random.choice(len(preds), size=min(n_samples, len(preds)), replace=False)

    # Extract coordinates
    x_coords = coords.get('x')
    y_coords = coords.get('y')
    has_coords = x_coords is not None and y_coords is not None

    if has_coords:
        extent = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]
    else:
        extent = None

    # Find indices for each variable type
    wind_u_idx = next((i for i, var in enumerate(output_vars) if '_u' in var.lower()), None)
    wind_v_idx = next((i for i, var in enumerate(output_vars) if '_v' in var.lower()), None)
    temp_idx = next((i for i, var in enumerate(output_vars) if 'temp' in var.lower()), None)
    pressure_idx = next((i for i, var in enumerate(output_vars) if 'pressure' in var.lower()), None)
    radiation_idx = next((i for i, var in enumerate(output_vars) if 'radiation' in var.lower()), None)

    print(f"Variable indices - wind_u: {wind_u_idx}, wind_v: {wind_v_idx}, "
          f"temp: {temp_idx}, pressure: {pressure_idx}, radiation: {radiation_idx}")
    print(f"Output vars: {output_vars}")

    # Track which variable indices have dedicated plot handlers
    plotted_indices = set()
    if wind_u_idx is not None:
        plotted_indices.add(wind_u_idx)
    if wind_v_idx is not None:
        plotted_indices.add(wind_v_idx)
    if temp_idx is not None:
        plotted_indices.add(temp_idx)
    if pressure_idx is not None:
        plotted_indices.add(pressure_idx)
    if radiation_idx is not None:
        plotted_indices.add(radiation_idx)

    for idx in indices:
        pred = preds[idx]
        target = targets[idx]
        input_sample = inputs[idx]

        # Wind fields (3x3 grid)
        if wind_u_idx is not None and wind_v_idx is not None:
            _plot_wind_sample(input_sample, target, pred, wind_u_idx, wind_v_idx,
                              has_coords, extent, save_dir, idx)

        # Temperature
        if temp_idx is not None:
            _plot_scalar_sample(input_sample, target, pred, temp_idx, 'Temperature',
                                'RdYlBu_r', 'K', has_coords, extent, save_dir, idx)

        # Pressure
        if pressure_idx is not None:
            _plot_scalar_sample(input_sample, target, pred, pressure_idx, 'Air Pressure',
                                'viridis', 'Pa', has_coords, extent, save_dir, idx)

        # Radiation
        if radiation_idx is not None:
            _plot_scalar_sample(input_sample, target, pred, radiation_idx, 'Solar Radiation',
                                'YlOrRd', 'W/m2', has_coords, extent, save_dir, idx,
                                vmin_override=0)

        # Any remaining variables not covered by the dedicated handlers above
        for var_i, var_name in enumerate(output_vars):
            if var_i not in plotted_indices:
                _plot_scalar_sample(input_sample, target, pred, var_i, var_name,
                                    'viridis', '', has_coords, extent, save_dir, idx)


def _plot_wind_sample(input_sample, target, pred, u_idx, v_idx,
                      has_coords, extent, save_dir, sample_idx):
    """Plot wind U, V, and speed for a single sample."""
    try:
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f'Wind Fields - Sample {sample_idx}', fontsize=16, y=0.995)

        input_speed = np.sqrt(input_sample[u_idx] ** 2 + input_sample[v_idx] ** 2)
        target_speed = np.sqrt(target[u_idx] ** 2 + target[v_idx] ** 2)
        pred_speed = np.sqrt(pred[u_idx] ** 2 + pred[v_idx] ** 2)

        imshow_kw = dict(extent=extent, origin='lower', aspect='auto')

        # U component row
        for col, (data, title) in enumerate([
            (input_sample[u_idx], 'Input U (ERA5)'),
            (target[u_idx], 'Target U (CONUS404)'),
            (pred[u_idx], 'Predicted U'),
        ]):
            im = axes[0, col].imshow(data, cmap='RdBu_r', vmin=-15, vmax=15, **imshow_kw)
            axes[0, col].set_title(title)
            plt.colorbar(im, ax=axes[0, col], label='m/s')

        # V component row
        for col, (data, title) in enumerate([
            (input_sample[v_idx], 'Input V (ERA5)'),
            (target[v_idx], 'Target V (CONUS404)'),
            (pred[v_idx], 'Predicted V'),
        ]):
            im = axes[1, col].imshow(data, cmap='RdBu_r', vmin=-15, vmax=15, **imshow_kw)
            axes[1, col].set_title(title)
            plt.colorbar(im, ax=axes[1, col], label='m/s')

        # Speed row
        vmax_speed = max(input_speed.max(), target_speed.max(), pred_speed.max())
        for col, (data, title) in enumerate([
            (input_speed, 'Input Speed (ERA5)'),
            (target_speed, 'Target Speed (CONUS404)'),
            (pred_speed, 'Predicted Speed'),
        ]):
            im = axes[2, col].imshow(data, cmap='viridis', vmin=0, vmax=vmax_speed, **imshow_kw)
            axes[2, col].set_title(title)
            plt.colorbar(im, ax=axes[2, col], label='m/s')

        axes[0, 0].set_ylabel('U Component', fontsize=12)
        axes[1, 0].set_ylabel('V Component', fontsize=12)
        axes[2, 0].set_ylabel('Wind Speed', fontsize=12)

        plt.tight_layout()
        plt.savefig(save_dir / f'wind_sample_{sample_idx}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved wind_sample_{sample_idx}.png")
    except Exception as e:
        print(f"  ERROR creating wind plot: {e}")
        import traceback
        traceback.print_exc()


def _plot_scalar_sample(input_sample, target, pred, var_idx, var_label,
                        cmap, units, has_coords, extent, save_dir, sample_idx,
                        vmin_override=None):
    """Plot a scalar variable (temperature, pressure, radiation) for a single sample."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'{var_label} - Sample {sample_idx}', fontsize=16)

    if vmin_override is not None:
        vmin = vmin_override
    else:
        vmin = min(input_sample[var_idx].min(), target[var_idx].min(), pred[var_idx].min())
    vmax = max(input_sample[var_idx].max(), target[var_idx].max(), pred[var_idx].max())

    imshow_kw = dict(cmap=cmap, vmin=vmin, vmax=vmax, extent=extent,
                     origin='lower', aspect='auto')

    for col, (data, title) in enumerate([
        (input_sample[var_idx], 'Input (ERA5)'),
        (target[var_idx], 'Target (CONUS404)'),
        (pred[var_idx], 'Predicted'),
    ]):
        im = axes[col].imshow(data, **imshow_kw)
        axes[col].set_title(title)
        plt.colorbar(im, ax=axes[col], label=units)

    plt.tight_layout()
    fname = var_label.lower().replace(' ', '_')
    plt.savefig(save_dir / f'{fname}_sample_{sample_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_distribution(preds, targets, output_vars, save_dir):
    """Plot error distributions with RMSE and skill scores for all variables."""
    save_dir = Path(save_dir)

    n_vars = len(output_vars)
    n_cols = min(3, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_vars == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_vars > 1 else axes

    for i, var_name in enumerate(output_vars):
        # Cast to float64 immediately to prevent float32 overflow when squaring large values
        error = (preds[:, i] - targets[:, i]).flatten().double().numpy()
        target_flat = targets[:, i].flatten().double().numpy()
        pred_flat = preds[:, i].flatten().double().numpy()

        # Mask out NaN/Inf (from masked pixels or denormalization overflow)
        mask = np.isfinite(error) & np.isfinite(target_flat) & np.isfinite(pred_flat)
        n_invalid = (~mask).sum()
        error = error[mask]
        target_flat = target_flat[mask]
        pred_flat = pred_flat[mask]

        ax = axes[i] if n_vars > 1 else axes

        if len(error) == 0:
            ax.set_title(f'{var_name}\n(all values invalid — check data masking)', fontsize=10)
            ax.axis('off')
            continue

        rmse = np.sqrt(np.mean(error ** 2))
        mae = np.mean(np.abs(error))
        bias = np.mean(error)
        ss_res = np.sum(error ** 2)
        ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        corr = np.corrcoef(target_flat, pred_flat)[0, 1] if len(target_flat) > 1 else np.nan
        target_std = np.std(target_flat)
        scatter_index = rmse / target_std if target_std != 0 else np.nan

        # Clip histogram to 1st–99th percentile so extreme outliers don't collapse the x-axis
        p1, p99 = np.percentile(error, [1, 99])
        error_clipped = error[(error >= p1) & (error <= p99)]
        n_outliers = len(error) - len(error_clipped)

        ax.hist(error_clipped, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
        bias_label = f'Bias: {bias:.3f}' if np.isfinite(bias) else f'Bias: {bias}'
        ax.axvline(np.clip(bias, p1, p99), color='orange', linestyle='--', linewidth=2,
                   label=bias_label)
        title_parts = []
        if n_invalid > 0:
            title_parts.append(f'{n_invalid:,} invalid pixels masked')
        if n_outliers > 0:
            title_parts.append(f'{n_outliers:,} outliers outside p1–p99')
        if title_parts:
            ax.set_title(f'{var_name} ({"; ".join(title_parts)})', fontsize=9, color='darkorange')

        textstr = (f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR2: {r_squared:.4f}\n'
                   f'Corr: {corr:.4f}\nSI: {scatter_index:.4f}\nN: {len(error):,}')
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        if n_invalid == 0:
            ax.set_title(f'{var_name} Error Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(save_dir / 'error_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Scatter plots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_vars == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_vars > 1 else axes

    for i, var_name in enumerate(output_vars):
        # Cast to float64 to prevent overflow when squaring large values
        target_flat = targets[:, i].flatten().double().numpy()
        pred_flat = preds[:, i].flatten().double().numpy()

        # Mask out NaN/Inf
        mask = np.isfinite(target_flat) & np.isfinite(pred_flat)
        target_flat = target_flat[mask]
        pred_flat = pred_flat[mask]

        ax = axes[i] if n_vars > 1 else axes

        if len(target_flat) == 0:
            ax.set_title(f'{var_name}\n(all values invalid)', fontsize=10)
            ax.axis('off')
            continue

        rmse = np.sqrt(np.mean((pred_flat - target_flat) ** 2))
        ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
        r_squared = 1 - (np.sum((pred_flat - target_flat) ** 2) / ss_tot) if ss_tot != 0 else np.nan
        target_std = np.std(target_flat)
        scatter_index = rmse / target_std if target_std != 0 else np.nan

        ax.hexbin(target_flat, pred_flat, gridsize=50, cmap='Blues', mincnt=1)

        min_val = min(target_flat.min(), pred_flat.min())
        max_val = max(target_flat.max(), pred_flat.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 line')

        textstr = (f'RMSE: {rmse:.4f}\nR2: {r_squared:.4f}\n'
                   f'SI: {scatter_index:.4f}\nN: {len(target_flat):,}')
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_title(f'{var_name} Predictions vs Targets', fontsize=12, fontweight='bold')
        ax.set_xlabel('Target')
        ax.set_ylabel('Predicted')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(save_dir / 'scatter_plots.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_normalization_stats(stats: dict, save_dir):
    """
    Plot mean ± std and min/max range for every variable in the normalization
    stats dict.  Saves two horizontal bar charts side by side.

    Args:
        stats:    dict returned by NetCDFPreprocessor.calculate_and_save_stats
                  {var_name: {'mean': float, 'std': float, 'min': float, 'max': float}}
        save_dir: directory where 'normalization_stats.png' will be written
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    var_names = list(stats.keys())
    means  = np.array([stats[v]['mean'] for v in var_names])
    stds   = np.array([stats[v]['std']  for v in var_names])
    mins   = np.array([stats[v]['min']  for v in var_names])
    maxs   = np.array([stats[v]['max']  for v in var_names])
    ranges = maxs - mins

    n = len(var_names)
    y = np.arange(n)
    height = max(5, n * 0.55)

    fig, axes = plt.subplots(1, 3, figsize=(18, height))

    # ── Panel 1: Mean ± Std ──────────────────────────────────────────────────
    ax = axes[0]
    bars = ax.barh(y, means, xerr=stds, align='center', height=0.6,
                   color='steelblue', alpha=0.8,
                   error_kw=dict(ecolor='black', capsize=4, linewidth=1.2))
    ax.set_yticks(y)
    ax.set_yticklabels(var_names, fontsize=10)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Value (raw units)', fontsize=10)
    ax.set_title('Mean ± 1σ\n(training set)', fontweight='bold', fontsize=11)
    ax.grid(True, axis='x', alpha=0.3)
    # Annotate mean values
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_width() + s + 0.01 * abs(ax.get_xlim()[1] - ax.get_xlim()[0]),
                bar.get_y() + bar.get_height() / 2,
                f'{m:.3g}', va='center', ha='left', fontsize=8, color='dimgray')

    # ── Panel 2: Standard Deviation ─────────────────────────────────────────
    ax = axes[1]
    bars = ax.barh(y, stds, align='center', height=0.6, color='coral', alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(var_names, fontsize=10)
    ax.set_xlabel('Standard deviation (raw units)', fontsize=10)
    ax.set_title('Standard Deviation\n(training set)', fontweight='bold', fontsize=11)
    ax.grid(True, axis='x', alpha=0.3)
    for bar, s in zip(bars, stds):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f'{s:.3g}', va='center', ha='left', fontsize=8, color='dimgray')

    # ── Panel 3: Min / Max range ─────────────────────────────────────────────
    ax = axes[2]
    # Horizontal range bars (broken barh from min to max)
    ax.barh(y, ranges, left=mins, align='center', height=0.6,
            color='mediumseagreen', alpha=0.75)
    # Mark mean as a vertical tick inside the range
    ax.scatter(means, y, color='black', zorder=5, s=30, label='Mean')
    ax.set_yticks(y)
    ax.set_yticklabels(var_names, fontsize=10)
    ax.set_xlabel('Value (raw units)', fontsize=10)
    ax.set_title('Min – Max range\n(▪ = mean)', fontweight='bold', fontsize=11)
    ax.grid(True, axis='x', alpha=0.3)
    for i_v, (lo, hi) in enumerate(zip(mins, maxs)):
        ax.text(hi, i_v, f'  {hi:.3g}', va='center', ha='left', fontsize=7, color='dimgray')
        ax.text(lo, i_v, f'{lo:.3g}  ', va='center', ha='right', fontsize=7, color='dimgray')

    plt.suptitle('Normalization statistics — computed on training split',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    out_path = save_dir / 'normalization_stats.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")


def plot_spatial_stats(train_ds, save_dir):
    """
    For each variable in the training dataset, compute the time-mean and
    time-standard-deviation and save them as spatial map PNGs.

    One file per variable is written to  <save_dir>/spatial_stats/<var>.png,
    showing:
      left panel  — time-mean field
      right panel — temporal standard deviation field

    Args:
        train_ds: xarray.Dataset (may be dask-backed / lazy)
        save_dir: root output directory (e.g. data/processed/)
    """
    out_dir = Path(save_dir) / 'spatial_stats'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Spatial coordinates (UTM metres if available)
    x_coords = train_ds['x'].values if 'x' in train_ds.coords else None
    y_coords = train_ds['y'].values if 'y' in train_ds.coords else None
    has_coords = x_coords is not None and y_coords is not None

    var_names = list(train_ds.data_vars)
    print(f"  Saving spatial stats to {out_dir}/")

    for var_name in var_names:
        print(f"    {var_name} ...", end=" ", flush=True)
        da = train_ds[var_name]

        # Compute (triggers dask if lazy)
        mean_field = da.mean(dim='time').values   # shape (y, x)
        std_field  = da.std(dim='time').values    # shape (y, x)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        panels = [
            (axes[0], mean_field, 'Time Mean',          'RdBu_r'),
            (axes[1], std_field,  'Temporal Std Dev',   'plasma'),
        ]

        for ax, field, title, cmap in panels:
            n_nan = int(np.isnan(field).sum())

            if has_coords:
                im = ax.pcolormesh(x_coords, y_coords, field,
                                   cmap=cmap, shading='auto')
                ax.set_xlabel('Easting (m)', fontsize=9)
                ax.set_ylabel('Northing (m)', fontsize=9)
                ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
            else:
                im = ax.imshow(field, cmap=cmap, origin='lower', aspect='auto')
                ax.set_xlabel('column index', fontsize=9)
                ax.set_ylabel('row index', fontsize=9)

            plt.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
            ax.set_aspect('equal', adjustable='box')

            nan_note = f'\n({n_nan:,} NaN pixels)' if n_nan > 0 else ''
            ax.set_title(f'{title}{nan_note}', fontsize=11, fontweight='bold')

        plt.suptitle(f'{var_name}  —  spatial statistics (training split)',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(out_dir / f'{var_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("done")

    print(f"  All spatial stats saved to {out_dir}/")
