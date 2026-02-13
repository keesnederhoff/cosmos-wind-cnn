"""
Configuration parsing utilities
"""

import yaml
from pathlib import Path
from typing import Dict, List, Tuple


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_variable_config(config: dict) -> Tuple[List[str], List[str], List[Tuple[int, int]]]:
    """
    Parse variable_pairs and additional_inputs from config.

    Returns:
        input_vars: list of input variable names
        output_vars: list of output variable names (high-res targets)
        wind_pair_indices: list of (u_idx, v_idx) tuples for wind loss calculation
    """
    input_vars = []
    output_vars = []
    wind_var_indices = {}  # var_name -> index in output_vars

    # Process variable pairs
    for var_name, pair in config['variable_pairs'].items():
        high_res = pair['high_res']
        low_res = pair['low_res']

        # Low-res goes in as input
        input_vars.append(low_res)
        # High-res is the target output
        output_idx = len(output_vars)
        output_vars.append(high_res)

        # Track wind variables by their index in output_vars
        if 'wind' in var_name or var_name in ['u', 'v']:
            wind_var_indices[var_name] = output_idx

    # Add additional input-only variables
    if 'additional_inputs' in config and config['additional_inputs']:
        additional = config['additional_inputs']
        if isinstance(additional, str):
            input_vars.append(additional)
        elif isinstance(additional, list):
            input_vars.extend(additional)

    # Group wind pairs as (u_idx, v_idx) for loss calculation
    wind_pair_indices = []
    u_indices = [(name, idx) for name, idx in wind_var_indices.items() if 'u' in name]
    v_indices = [(name, idx) for name, idx in wind_var_indices.items() if 'v' in name]

    if u_indices and v_indices:
        wind_pair_indices = [(u_idx, v_idx) for (_, u_idx), (_, v_idx)
                             in zip(u_indices, v_indices)]

    return input_vars, output_vars, wind_pair_indices
