"""Memory-mapped dataset for fast, RAM-shared 3D U-Net training.

Reads the uncompressed raw float32 arrays produced by
``scripts/convert_to_memmap.py`` (one ``<var>.dat`` per variable per split,
each shaped ``(T, H, W)``), plus ``meta.json`` and ``nan_at_time.npy``.

Why this exists (the durable fix for the training OOM + I/O stalls):
  * ``WindDatasetInMemory`` loads a *private full copy* of the dataset into
    every DDP rank -> N x replication -> host-RAM OOM as soon as channels,
    timesteps, or GPUs grow.
  * ``WindDataset3D`` (lazy NetCDF) avoids the RAM blow-up but reads the
    zlib-compressed NetCDF with random access, so every __getitem__
    re-decompresses chunks over Lustre -> catastrophically slow + periodic
    HDF5/Lustre stalls.

``np.memmap`` on an *uncompressed* file fixes both: all ranks mmap the same
file, so the OS page cache holds a *single shared copy* of the hot pages per
node (no per-rank replication, all GPUs usable), and reads are plain page
faults with no decompression. When the data fits in RAM it is effectively
in-memory speed after warmup; when it does not, it degrades to the working
set instead of OOMing. Valid-window indices come from a precomputed
``nan_at_time`` array, so there is no init-time NaN scan (that scan, combined
with xarray ``cache=True``, was the original OOM trigger).

Drop-in for ``WindDataset3D``: identical constructor signature and identical
``(input, target)`` tensor shapes. ``netcdf_path`` is used only to locate the
sibling ``memmap/<split>/`` directory (split = file stem, e.g. ``train``).
"""

import json
import pickle
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


class WindDatasetMemmap(Dataset):
    def __init__(
        self,
        netcdf_path: str,
        stats_path: str,
        input_vars: List[str],
        output_vars: List[str],
        sequence_length: int = 6,
        forecast_horizon: int = 0,
        stride: int = 1,
        verbose: bool = True,
    ):
        split = Path(netcdf_path).stem  # train / val / test
        mdir = Path(netcdf_path).parent / "memmap" / split
        meta_path = mdir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"memmap data not found at {mdir}. "
                f"Run scripts/convert_to_memmap.py on the data_processed dir first."
            )

        meta = json.loads(meta_path.read_text())
        self.mdir = mdir
        self.dtype = meta["dtype"]
        self.n_times = int(meta["T"])
        self.height = int(meta["H"])
        self.width = int(meta["W"])
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        self.verbose = verbose

        missing = [v for v in set(input_vars) | set(output_vars)
                   if v not in meta["vars"]]
        if missing:
            raise KeyError(f"variables missing from memmap {mdir}: {missing}")

        # Read-only memmaps. Inherited across DataLoader worker forks safely;
        # pages are shared across all workers and DDP ranks via the page cache.
        self._mm = {}
        for var in set(input_vars) | set(output_vars):
            self._mm[var] = np.memmap(
                mdir / (var + ".dat"), dtype=self.dtype, mode="r",
                shape=(self.n_times, self.height, self.width),
            )

        with open(stats_path, "rb") as f:
            self.stats = pickle.load(f)

        nan_at_time = np.load(mdir / "nan_at_time.npy")
        self.valid_indices = self._get_valid_indices(nan_at_time)

        if verbose:
            print("Memmap dataset initialized:")
            print(f"  Samples: {len(self.valid_indices)}")
            print(f"  Input shape: ({sequence_length}, {len(input_vars)}, "
                  f"{self.height}, {self.width})")
            print(f"  Output shape: ({len(output_vars)}, {self.height}, {self.width})")

    def _get_valid_indices(self, nan_at_time: np.ndarray):
        """Drop any window that contains a NaN timestep — same rule as
        WindDataset3D, but from a precomputed per-timestep NaN flag (no scan)."""
        max_idx = self.n_times - self.sequence_length - self.forecast_horizon
        all_indices = list(range(0, max_idx, self.stride))
        window = self.sequence_length + max(self.forecast_horizon, 1)
        valid = [idx for idx in all_indices
                 if not nan_at_time[idx: idx + window].any()]
        n_dropped = len(all_indices) - len(valid)
        if n_dropped > 0 and self.verbose:
            print(f"  Dropped {n_dropped:,} / {len(all_indices):,} samples "
                  f"({100 * n_dropped / len(all_indices):.1f}%) — NaN pixels in window")
        return valid

    def normalize(self, data: np.ndarray, var_name: str) -> np.ndarray:
        mean = self.stats[var_name]["mean"]
        std = self.stats[var_name]["std"]
        return (data - mean) / (std + 1e-8)

    def denormalize(self, data: np.ndarray, var_name: str) -> np.ndarray:
        mean = self.stats[var_name]["mean"]
        std = self.stats[var_name]["std"]
        return data * (std + 1e-8) + mean

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length
        target_idx = end_idx + self.forecast_horizon - 1

        input_data = []
        for var in self.input_vars:
            arr = np.asarray(self._mm[var][start_idx:end_idx], dtype=np.float32)
            input_data.append(self.normalize(arr, var))
        # (seq_len, n_input_vars, H, W)
        input_tensor = torch.from_numpy(np.stack(input_data, axis=1)).float()

        target_data = []
        for var in self.output_vars:
            arr = np.asarray(self._mm[var][target_idx], dtype=np.float32)
            target_data.append(self.normalize(arr, var))
        # (n_output_vars, H, W)
        target_tensor = torch.from_numpy(np.stack(target_data, axis=0)).float()

        return input_tensor, target_tensor
