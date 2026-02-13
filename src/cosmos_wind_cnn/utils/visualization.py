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
        error = (preds[:, i] - targets[:, i]).flatten().numpy()
        target_flat = targets[:, i].flatten().numpy()
        pred_flat = preds[:, i].flatten().numpy()

        rmse = np.sqrt(np.mean(error ** 2))
        mae = np.mean(np.abs(error))
        bias = np.mean(error)
        ss_res = np.sum(error ** 2)
        ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        corr = np.corrcoef(target_flat, pred_flat)[0, 1]
        target_std = np.std(target_flat)
        scatter_index = rmse / target_std if target_std != 0 else np.nan

        ax = axes[i] if n_vars > 1 else axes
        ax.hist(error, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
        ax.axvline(bias, color='orange', linestyle='--', linewidth=2, label=f'Bias: {bias:.3f}')

        textstr = (f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR2: {r_squared:.4f}\n'
                   f'Corr: {corr:.4f}\nSI: {scatter_index:.4f}')
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

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
        target_flat = targets[:, i].flatten().numpy()
        pred_flat = preds[:, i].flatten().numpy()

        rmse = np.sqrt(np.mean((pred_flat - target_flat) ** 2))
        r_squared = 1 - (np.sum((pred_flat - target_flat) ** 2) /
                         np.sum((target_flat - np.mean(target_flat)) ** 2))
        target_std = np.std(target_flat)
        scatter_index = rmse / target_std if target_std != 0 else np.nan

        ax = axes[i] if n_vars > 1 else axes
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
