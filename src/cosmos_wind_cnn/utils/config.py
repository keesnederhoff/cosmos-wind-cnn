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
    wind_pairs = []

    # Process variable pairs
    for var_name, pair in config['variable_pairs'].items():
        high_res = pair['high_res']
        low_res = pair['low_res']

        # Low-res goes in as input
        input_vars.append(low_res)
        # High-res is the target output
        output_vars.append(high_res)

        # Track wind pairs for special wind loss
        if 'wind' in var_name or var_name in ['u', 'v']:
            wind_pairs.append((var_name, high_res))

    # Add additional input-only variables
    if 'additional_inputs' in config and config['additional_inputs']:
        additional = config['additional_inputs']
        if isinstance(additional, str):
            input_vars.append(additional)
        elif isinstance(additional, list):
            input_vars.extend(additional)

    # Group wind pairs as (u_idx, v_idx) for loss calculation
    wind_pair_indices = []
    output_u_indices = [i for i, (name, _) in enumerate(wind_pairs) if 'u' in name]
    output_v_indices = [i for i, (name, _) in enumerate(wind_pairs) if 'v' in name]

    if output_u_indices and output_v_indices:
        wind_pair_indices = list(zip(output_u_indices, output_v_indices))

    return input_vars, output_vars, wind_pair_indices
