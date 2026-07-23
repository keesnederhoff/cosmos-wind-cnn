"""
Configuration parsing utilities
"""

import os
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_run_dirs(case_dir, run_name: str) -> dict:
    """
    Return the canonical directory layout for a single run.

    All run artefacts live under  ``<COSMOS_RESULTS_ROOT>/<case_name>/results/<run_name>/``.

    Parameters
    ----------
    case_dir : str or Path
        Root of the case study (e.g. ``case_studies/sf_bay_conus404``).
    run_name : str
        Unique identifier for the run (SLURM job ID, experiment tag, …).

    Returns
    -------
    dict with keys:
        run_root         – results/<run_name>/
        checkpoint       – results/<run_name>/checkpoint/
        data_processed   – results/<run_name>/data_processed/
        logs             – results/<run_name>/logs/
        output_inference – results/<run_name>/output_inference/
        output_evaluation– results/<run_name>/output_evaluation/
    """
    case_dir = Path(case_dir)
    _results_root = os.environ.get('COSMOS_RESULTS_ROOT')
    if not _results_root:
        raise RuntimeError(
            "COSMOS_RESULTS_ROOT is not set. Point it at your results storage base "
            "(run outputs go to <COSMOS_RESULTS_ROOT>/<case_name>/results/<run_name>/).\n"
            "  Windows:  set COSMOS_RESULTS_ROOT=G:\\03-downscaling_meteo_cnn\n"
            "  Linux:    export COSMOS_RESULTS_ROOT=/path/to/storage"
        )
    run_root = Path(_results_root) / case_dir.name / 'results' / str(run_name)
    return {
        'run_root':          run_root,
        'checkpoint':        run_root / 'checkpoint',
        'data_processed':    run_root / 'data_processed',
        'logs':              run_root / 'logs',
        'output_inference':  run_root / 'output_inference',
        'output_evaluation': run_root / 'output_evaluation',
    }


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


def env_bool(value: str) -> bool:
    """Parse an env-var string as a boolean.

    Needed because `bool("0")` is True in Python, so a plain bool cast in the
    SWEEP_* override table would make `SWEEP_RESIDUAL=0` silently ENABLE
    residual mode.
    """
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def env_list(value: str) -> List[str]:
    """Parse a comma-separated env-var string into a list of names."""
    return [p.strip() for p in str(value).split(',') if p.strip()]


def residual_channel_map(config: dict, input_vars: List[str],
                         output_vars: List[str]) -> List[int]:
    """Index into `input_vars` of the low-res counterpart of each output var.

    Residual learning adds the (already interpolated) low-res field back onto the
    network output, so each high-res target needs the position of its paired
    low-res input channel. Derived explicitly from `variable_pairs` rather than
    assuming the identity mapping that `parse_variable_config` happens to produce
    today, so a reordering of variable_pairs can never silently mis-pair channels.
    """
    lr_for_hr = {pair['high_res']: pair['low_res']
                 for pair in config['variable_pairs'].values()}
    idx = []
    for hr in output_vars:
        if hr not in lr_for_hr:
            raise KeyError(
                f"residual_learning: output '{hr}' has no low_res counterpart in "
                f"variable_pairs; residual mode requires every output to be paired."
            )
        lr = lr_for_hr[hr]
        if lr not in input_vars:
            raise KeyError(
                f"residual_learning: low_res '{lr}' (paired with '{hr}') is not in "
                f"input_vars {input_vars}"
            )
        idx.append(input_vars.index(lr))
    return idx


def residual_affine(stats: dict, config: dict, output_vars: List[str]):
    """Per-output-channel (scale, shift) taking a normalized low-res input into
    the normalized high-res target space.

    Inputs and targets are each z-scored with their OWN mean/std, so the residual
    skip is NOT an identity in normalized space. With the dataset's convention
    (see WindDatasetMemmap.normalize/denormalize, which use `std + 1e-8`)::

        lr_phys = lr_norm * (sd_lr + 1e-8) + mu_lr
        hr_norm = (hr_phys - mu_hr) / (sd_hr + 1e-8)

    substituting hr_phys ~= lr_phys gives ``hr_norm ~= scale * lr_norm + shift``::

        scale = (sd_lr + 1e-8) / (sd_hr + 1e-8)
        shift = (mu_lr - mu_hr) / (sd_hr + 1e-8)

    Omitting this rescale still trains without error but learns a biased offset,
    so it is the one silent-failure mode of residual mode.

    Returns (scale, shift) float32 arrays of length len(output_vars).
    """
    lr_for_hr = {pair['high_res']: pair['low_res']
                 for pair in config['variable_pairs'].values()}
    scale, shift = [], []
    for hr in output_vars:
        lr = lr_for_hr[hr]
        for name in (hr, lr):
            if name not in stats:
                raise KeyError(
                    f"residual_learning: '{name}' missing from normalization stats"
                )
        sd_hr = stats[hr]['std'] + 1e-8
        sd_lr = stats[lr]['std'] + 1e-8
        scale.append(sd_lr / sd_hr)
        shift.append((stats[lr]['mean'] - stats[hr]['mean']) / sd_hr)
    return (np.asarray(scale, dtype=np.float32),
            np.asarray(shift, dtype=np.float32))


def classify_file_keys(file_dict, target_prefix: str = 'hr_',
                       input_prefix: str = 'lr_'):
    """
    Partition file_dict keys into (target, input, other) by prefix, preserving order.

    target keys define the high-resolution reference grid; input keys are the coarse
    fields interpolated onto it. Anything matching neither prefix is returned as 'other'.
    """
    target_keys = [k for k in file_dict if k.startswith(target_prefix)]
    input_keys = [k for k in file_dict if k.startswith(input_prefix)]
    other_keys = [k for k in file_dict
                  if k not in target_keys and k not in input_keys]
    return target_keys, input_keys, other_keys


# Units keyed by variable-name suffix (prefix-agnostic: works for hr_*, lr_*, ...)
_UNIT_BY_SUFFIX = {
    'air_temp': 'K', 'dew_temp': 'K', 'pressure': 'Pa',
    'solar': 'W m**-2', 'thermal': 'W m**-2', 'rain': 'mm hr**-1',
    'u': 'm s**-1', 'v': 'm s**-1',
}


def var_units_for(var_names):
    """Map each variable name to a unit string by matching its suffix.

    Longer suffixes are matched first so 'air_temp' is not shadowed by 'temp'-style
    fragments. Names with no known suffix are omitted from the result.
    """
    suffixes = sorted(_UNIT_BY_SUFFIX, key=len, reverse=True)
    units = {}
    for name in var_names:
        for suffix in suffixes:
            if name == suffix or name.endswith('_' + suffix):
                units[name] = _UNIT_BY_SUFFIX[suffix]
                break
    return units


def wind_var_names(variable_pairs):
    """Return (u_target, v_target, u_input, v_input) from a training config's
    variable_pairs, or None if a u/v wind pair is not present.

    Recognises pair names 'wind_u'/'u' and 'wind_v'/'v'.
    """
    out = {}
    for pair_name, pair in variable_pairs.items():
        if pair_name in ('wind_u', 'u'):
            out['u_target'] = pair['high_res']
            out['u_input'] = pair['low_res']
        elif pair_name in ('wind_v', 'v'):
            out['v_target'] = pair['high_res']
            out['v_input'] = pair['low_res']
    if all(k in out for k in ('u_target', 'v_target', 'u_input', 'v_input')):
        return out['u_target'], out['v_target'], out['u_input'], out['v_input']
    return None


def get_data_dir(case_dir):
    """Directory holding a case study's raw NetCDF inputs (shared across runs).

    Read from ``<COSMOS_DATA_ROOT>/<case_name>/raw_data``. Data lives OUTSIDE the
    repo, so COSMOS_DATA_ROOT must be set; raises RuntimeError if it is not.
    """
    case_dir = Path(case_dir)
    root = os.environ.get('COSMOS_DATA_ROOT')
    if not root:
        raise RuntimeError(
            "COSMOS_DATA_ROOT is not set. Point it at your raw-data storage base "
            "(raw inputs are read from <COSMOS_DATA_ROOT>/<case_name>/raw_data/).\n"
            "  Windows:  set COSMOS_DATA_ROOT=G:\\03-downscaling_meteo_cnn\n"
            "  Linux:    export COSMOS_DATA_ROOT=/path/to/storage"
        )
    return Path(root) / case_dir.name / 'raw_data'
