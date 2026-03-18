"""
Evaluation metrics for wind prediction
"""

import torch
import numpy as np


def _finite_wind_mask(pred, target):
    """
    Boolean mask (batch, H, W): pixels where BOTH U and V channels
    are finite in BOTH pred and target.
    """
    return (torch.isfinite(pred[:, 0]) & torch.isfinite(pred[:, 1]) &
            torch.isfinite(target[:, 0]) & torch.isfinite(target[:, 1]))


def calculate_rmse(pred, target):
    """Root Mean Square Error — skips non-finite pixels, uses float64."""
    mask = torch.isfinite(pred) & torch.isfinite(target)
    if not mask.any():
        return float('nan')
    diff = (pred[mask] - target[mask]).double()
    return torch.sqrt(torch.mean(diff ** 2)).item()


def calculate_mae(pred, target):
    """Mean Absolute Error — skips non-finite pixels, uses float64."""
    mask = torch.isfinite(pred) & torch.isfinite(target)
    if not mask.any():
        return float('nan')
    diff = (pred[mask] - target[mask]).double()
    return torch.mean(torch.abs(diff)).item()


def calculate_wind_speed_metrics(pred, target):
    """
    Calculate metrics for wind speed.

    Args:
        pred:   (batch, 2, H, W) — predicted [u, v]
        target: (batch, 2, H, W) — target [u, v]

    Returns:
        dict with speed_rmse, speed_mae, speed_bias
    """
    mask = _finite_wind_mask(pred, target)  # (batch, H, W)
    if not mask.any():
        return {'speed_rmse': float('nan'), 'speed_mae': float('nan'), 'speed_bias': float('nan')}

    # Cast to float64 before squaring to prevent float32 overflow
    p = pred.double()
    t = target.double()
    pred_speed = torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)   # (batch, H, W)
    target_speed = torch.sqrt(t[:, 0] ** 2 + t[:, 1] ** 2)

    ps = pred_speed[mask]
    ts = target_speed[mask]

    return {
        'speed_rmse': torch.sqrt(torch.mean((ps - ts) ** 2)).item(),
        'speed_mae':  torch.mean(torch.abs(ps - ts)).item(),
        'speed_bias': torch.mean(ps - ts).item(),
    }


def calculate_direction_metrics(pred, target):
    """
    Calculate metrics for wind direction.

    Args:
        pred:   (batch, 2, H, W) — predicted [u, v]
        target: (batch, 2, H, W) — target [u, v]

    Returns:
        dict with direction_mae (in degrees)
    """
    mask = _finite_wind_mask(pred, target)  # (batch, H, W)
    if not mask.any():
        return {'direction_mae': float('nan')}

    # Extract valid U/V pairs
    pred_u = pred[:, 0][mask]
    pred_v = pred[:, 1][mask]
    target_u = target[:, 0][mask]
    target_v = target[:, 1][mask]

    pred_angle   = torch.atan2(pred_v,   pred_u)
    target_angle = torch.atan2(target_v, target_u)

    angle_diff = pred_angle - target_angle
    angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))

    direction_mae = torch.mean(torch.abs(angle_diff)).item() * 180 / np.pi
    return {'direction_mae': direction_mae}


def calculate_all_metrics(pred, target):
    """
    Calculate all metrics.

    Args:
        pred:   (batch, 2, H, W)
        target: (batch, 2, H, W)

    Returns:
        dict with all metrics
    """
    n_nonfinite_pred   = (~torch.isfinite(pred)).sum().item()
    n_nonfinite_target = (~torch.isfinite(target)).sum().item()
    if n_nonfinite_pred > 0 or n_nonfinite_target > 0:
        total = pred.numel()
        print(f"  [metrics] non-finite values — pred: {n_nonfinite_pred:,}/{total:,}, "
              f"target: {n_nonfinite_target:,}/{total:,} (masked before computing metrics)")

    metrics = {}
    metrics['rmse'] = calculate_rmse(pred, target)
    metrics['mae']  = calculate_mae(pred, target)
    metrics.update(calculate_wind_speed_metrics(pred, target))
    metrics.update(calculate_direction_metrics(pred, target))

    return metrics
