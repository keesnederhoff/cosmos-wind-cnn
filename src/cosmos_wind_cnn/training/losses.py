"""
Custom loss functions for wind prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WindLoss(nn.Module):
    """
    Multi-component loss for wind prediction
    Combines:
    - MSE on wind components (u, v)
    - MAE on wind speed
    - Cosine similarity on wind direction
    """
    
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.3):
        """
        Args:
            alpha: Weight for component MSE loss
            beta: Weight for speed MAE loss
            gamma: Weight for direction cosine loss
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def forward(self, pred, target):
        """
        Args:
            pred: (batch, 2, H, W) - predicted [u, v]
            target: (batch, 2, H, W) - target [u, v]
        
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary of individual loss components
        """
        # 1. MSE on wind components
        component_loss = F.mse_loss(pred, target)
        
        # 2. Wind speed MAE
        pred_speed = torch.sqrt(pred[:, 0]**2 + pred[:, 1]**2 + 1e-8)
        target_speed = torch.sqrt(target[:, 0]**2 + target[:, 1]**2 + 1e-8)
        speed_loss = F.l1_loss(pred_speed, target_speed)
        
        # 3. Direction loss (cosine similarity)
        # Normalize wind vectors
        pred_norm = pred / (pred_speed.unsqueeze(1) + 1e-8)
        target_norm = target / (target_speed.unsqueeze(1) + 1e-8)
        
        # Dot product between normalized vectors
        cos_sim = (pred_norm[:, 0] * target_norm[:, 0] + 
                   pred_norm[:, 1] * target_norm[:, 1])
        direction_loss = 1 - cos_sim.mean()
        
        # Combined loss
        total_loss = (self.alpha * component_loss + 
                     self.beta * speed_loss +
                     self.gamma * direction_loss)
        
        # Return loss components for logging
        loss_dict = {
            'component_loss': component_loss.item(),
            'speed_loss': speed_loss.item(),
            'direction_loss': direction_loss.item()
        }
        
        return total_loss, loss_dict


class CombinedLoss(nn.Module):
    """
    Combined loss for multiple variable types:
    - Wind pairs (u, v) use WindLoss
    - Other variables (temperature, etc.) use MSE
    """
    
    def __init__(self, wind_pair_indices=None, alpha=1.0, beta=0.5, gamma=0.3):
        """
        Args:
            wind_pair_indices: List of tuples [(u_idx, v_idx), ...] indicating which 
                             output channels are wind pairs
            alpha, beta, gamma: Weights for WindLoss components
        """
        super().__init__()
        self.wind_pair_indices = wind_pair_indices or []
        self.wind_loss = WindLoss(alpha, beta, gamma)
        self.mse_loss = nn.MSELoss()
        
        # Track which output channels are part of wind pairs
        self.wind_channels = set()
        for u_idx, v_idx in self.wind_pair_indices:
            self.wind_channels.add(u_idx)
            self.wind_channels.add(v_idx)
        
        # Non-wind channels
        self.non_wind_channels = []
    
    def forward(self, pred, target):
        """
        Args:
            pred: (batch, n_channels, H, W) - predicted outputs
            target: (batch, n_channels, H, W) - target outputs
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of loss components
        """
        n_channels = pred.shape[1]
        
        # Identify non-wind channels
        if not self.non_wind_channels:
            self.non_wind_channels = [i for i in range(n_channels) if i not in self.wind_channels]
        
        total_loss = 0.0
        loss_dict = {}
        n_components = 0
        
        # Calculate wind losses for each pair
        for pair_idx, (u_idx, v_idx) in enumerate(self.wind_pair_indices):
            if u_idx < n_channels and v_idx < n_channels:
                wind_pred = pred[:, [u_idx, v_idx], :, :]
                wind_target = target[:, [u_idx, v_idx], :, :]
                
                wind_loss, wind_components = self.wind_loss(wind_pred, wind_target)
                total_loss += wind_loss
                n_components += 1
                
                # Log with pair index
                for key, val in wind_components.items():
                    loss_dict[f'wind_pair_{pair_idx}_{key}'] = val
        
        # Calculate MSE for non-wind variables
        if self.non_wind_channels:
            non_wind_pred = pred[:, self.non_wind_channels, :, :]
            non_wind_target = target[:, self.non_wind_channels, :, :]
            
            non_wind_loss = self.mse_loss(non_wind_pred, non_wind_target)
            total_loss += non_wind_loss
            n_components += 1
            
            loss_dict['non_wind_mse'] = non_wind_loss.item()
        
        # Average over components
        if n_components > 0:
            total_loss = total_loss / n_components
        
        # Add total loss to dict
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


class MSELoss(nn.Module):
    """
    Simple MSE loss wrapper for compatibility
    """
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        loss = self.mse(pred, target)
        loss_dict = {'mse_loss': loss.item()}
        return loss, loss_dict
