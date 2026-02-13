"""
Evaluation metrics for wind prediction
"""

import torch
import numpy as np


def calculate_rmse(pred, target):
    """Root Mean Square Error"""
    return torch.sqrt(torch.mean((pred - target)**2)).item()


def calculate_mae(pred, target):
    """Mean Absolute Error"""
    return torch.mean(torch.abs(pred - target)).item()


def calculate_wind_speed_metrics(pred, target):
    """
    Calculate metrics for wind speed
    
    Args:
        pred: (batch, 2, H, W) - predicted [u, v]
        target: (batch, 2, H, W) - target [u, v]
    
    Returns:
        dict with speed_rmse, speed_mae, speed_bias
    """
    pred_speed = torch.sqrt(pred[:, 0]**2 + pred[:, 1]**2)
    target_speed = torch.sqrt(target[:, 0]**2 + target[:, 1]**2)
    
    speed_rmse = torch.sqrt(torch.mean((pred_speed - target_speed)**2)).item()
    speed_mae = torch.mean(torch.abs(pred_speed - target_speed)).item()
    speed_bias = torch.mean(pred_speed - target_speed).item()
    
    return {
        'speed_rmse': speed_rmse,
        'speed_mae': speed_mae,
        'speed_bias': speed_bias
    }


def calculate_direction_metrics(pred, target):
    """
    Calculate metrics for wind direction
    
    Args:
        pred: (batch, 2, H, W) - predicted [u, v]
        target: (batch, 2, H, W) - target [u, v]
    
    Returns:
        dict with direction_mae (in degrees)
    """
    # Calculate angles
    pred_angle = torch.atan2(pred[:, 1], pred[:, 0])
    target_angle = torch.atan2(target[:, 1], target[:, 0])
    
    # Angular difference (accounting for wrapping)
    angle_diff = pred_angle - target_angle
    angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
    
    # Convert to degrees
    direction_mae = torch.mean(torch.abs(angle_diff)).item() * 180 / np.pi
    
    return {
        'direction_mae': direction_mae
    }


def calculate_all_metrics(pred, target):
    """
    Calculate all metrics
    
    Args:
        pred: (batch, 2, H, W)
        target: (batch, 2, H, W)
    
    Returns:
        dict with all metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['rmse'] = calculate_rmse(pred, target)
    metrics['mae'] = calculate_mae(pred, target)
    
    # Wind-specific metrics
    speed_metrics = calculate_wind_speed_metrics(pred, target)
    direction_metrics = calculate_direction_metrics(pred, target)
    
    metrics.update(speed_metrics)
    metrics.update(direction_metrics)
    
    return metrics
