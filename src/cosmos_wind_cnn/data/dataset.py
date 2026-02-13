"""
PyTorch Dataset classes for wind prediction
"""

import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import pickle
from pathlib import Path
from typing import List


class WindDataset3D(Dataset):
    """
    Dataset for 3D U-Net wind prediction
    Handles pre-processed NetCDF data
    """
    
    def __init__(
        self,
        netcdf_path: str,
        stats_path: str,
        input_vars: List[str],
        output_vars: List[str],
        sequence_length: int = 6,
        forecast_horizon: int = 1,
        stride: int = 1
    ):
        """
        Args:
            netcdf_path: Path to processed NetCDF file
            stats_path: Path to normalization statistics pickle
            input_vars: List of input variable names
            output_vars: List of output variable names
            sequence_length: Number of timesteps in input sequence
            forecast_horizon: How many steps ahead to predict
            stride: Stride between samples
        """
        self.netcdf_path = netcdf_path
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        
        # Load data
        print(f"Loading data from {netcdf_path}")
        self.data = xr.open_dataset(netcdf_path)
        
        # Load normalization statistics
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)
        
        # Get dimensions
        self.n_times = len(self.data.time)
        self.height = len(self.data.latitude)
        self.width = len(self.data.longitude)
        
        # Calculate valid indices
        self.valid_indices = self._get_valid_indices()
        
        print(f"Dataset initialized:")
        print(f"  Samples: {len(self.valid_indices)}")
        print(f"  Input shape: ({sequence_length}, {len(input_vars)}, {self.height}, {self.width})")
        print(f"  Output shape: ({len(output_vars)}, {self.height}, {self.width})")
    
    def _get_valid_indices(self):
        """Get valid starting indices for sequences"""
        max_idx = self.n_times - self.sequence_length - self.forecast_horizon
        return list(range(0, max_idx, self.stride))
    
    def normalize(self, data: np.ndarray, var_name: str) -> np.ndarray:
        """Normalize data using pre-computed statistics"""
        mean = self.stats[var_name]['mean']
        std = self.stats[var_name]['std']
        return (data - mean) / (std + 1e-8)
    
    def denormalize(self, data: np.ndarray, var_name: str) -> np.ndarray:
        """Denormalize data back to original scale"""
        mean = self.stats[var_name]['mean']
        std = self.stats[var_name]['std']
        return data * (std + 1e-8) + mean
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """
        Returns:
            input: (sequence_length, n_input_vars, height, width)
            target: (n_output_vars, height, width)
        """
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length
        target_idx = end_idx + self.forecast_horizon - 1
        
        # Extract input sequence
        input_data = []
        for var in self.input_vars:
            var_data = self.data[var].isel(
                time=slice(start_idx, end_idx)
            ).values
            var_data = self.normalize(var_data, var)
            input_data.append(var_data)
        
        # Stack: (seq_len, n_vars, height, width)
        input_tensor = np.stack(input_data, axis=1)
        input_tensor = torch.FloatTensor(input_tensor)
        
        # Extract target
        target_data = []
        for var in self.output_vars:
            var_data = self.data[var].isel(time=target_idx).values
            var_data = self.normalize(var_data, var)
            target_data.append(var_data)
        
        # Stack: (n_output_vars, height, width)
        target_tensor = np.stack(target_data, axis=0)
        target_tensor = torch.FloatTensor(target_tensor)
        
        return input_tensor, target_tensor


class WindDatasetInMemory(Dataset):
    """
    Faster version that loads all data into memory
    Use this if you have enough RAM
    """
    
    def __init__(
        self,
        netcdf_path: str,
        stats_path: str,
        input_vars: List[str],
        output_vars: List[str],
        sequence_length: int = 6,
        forecast_horizon: int = 1,
        stride: int = 1
    ):
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        
        # Load normalization statistics
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)
        
        # Load all data into memory
        print(f"Loading data from {netcdf_path} into memory...")
        data = xr.open_dataset(netcdf_path)
        
        self.data_array = {}
        for var in input_vars + output_vars:
            self.data_array[var] = data[var].values
        
        self.n_times = len(data.time)
        self.height = data.dims['latitude']
        self.width = data.dims['longitude']
        
        data.close()
        
        # Calculate valid indices
        self.valid_indices = self._get_valid_indices()
        
        print(f"Dataset loaded into memory:")
        print(f"  Samples: {len(self.valid_indices)}")
        print(f"  Input shape: ({sequence_length}, {len(input_vars)}, {self.height}, {self.width})")
        print(f"  Output shape: ({len(output_vars)}, {self.height}, {self.width})")
    
    def _get_valid_indices(self):
        """Get valid starting indices for sequences"""
        max_idx = self.n_times - self.sequence_length - self.forecast_horizon
        return list(range(0, max_idx, self.stride))
    
    def normalize(self, data: np.ndarray, var_name: str) -> np.ndarray:
        """Normalize data using pre-computed statistics"""
        mean = self.stats[var_name]['mean']
        std = self.stats[var_name]['std']
        return (data - mean) / (std + 1e-8)
    
    def denormalize(self, data: np.ndarray, var_name: str) -> np.ndarray:
        """Denormalize data back to original scale"""
        mean = self.stats[var_name]['mean']
        std = self.stats[var_name]['std']
        return data * (std + 1e-8) + mean
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length
        target_idx = end_idx + self.forecast_horizon - 1
        
        # Extract from in-memory arrays
        input_data = []
        for var in self.input_vars:
            var_data = self.data_array[var][start_idx:end_idx]
            var_data = self.normalize(var_data, var)
            input_data.append(var_data)
        
        input_tensor = torch.FloatTensor(np.stack(input_data, axis=1))
        
        target_data = []
        for var in self.output_vars:
            var_data = self.data_array[var][target_idx]
            var_data = self.normalize(var_data, var)
            target_data.append(var_data)
        
        target_tensor = torch.FloatTensor(np.stack(target_data, axis=0))
        
        return input_tensor, target_tensor
