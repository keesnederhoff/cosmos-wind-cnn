#!/usr/bin/env python
"""Microbenchmark: DataLoader throughput from WindDatasetMemmap (no GPU).

Answers the goal-critical question: can the memmap data path feed batches
fast enough that 4-GPU training is GPU-bound (so it converges well within the
2-day wall), rather than data-bound like the compressed-NetCDF path?

Runs two passes over the SAME first N batches (shuffle off):
  cold  = first touch, pages faulted in from Lustre
  warm  = re-read, served from OS page cache (== steady-state, since the
          137 GB dataset fits in the 385 GB node RAM after epoch 1)

Reports batches/s, samples/s, and a projected single-process epoch time.
Real training splits the epoch across 4 DDP ranks, so per-rank load is ~1/4
of the batches and 4 ranks run concurrently -> effective epoch time is far
lower than the single-process projection printed here.
"""
import argparse
import time

from torch.utils.data import DataLoader

from cosmos_wind_cnn.utils.config import load_config, parse_variable_config
from cosmos_wind_cnn.data.dataset_memmap import WindDatasetMemmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-processed", required=True)
    ap.add_argument("--case", default="case_studies/sf_bay_rtma")
    ap.add_argument("--batches", type=int, default=300)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()

    cfg = load_config(a.case + "/configs/training.yaml")
    iv, ov, _ = parse_variable_config(cfg)
    ds = WindDatasetMemmap(
        a.data_processed + "/train.nc",
        a.data_processed + "/normalization_stats.pkl",
        iv, ov, cfg["sequence_length"], cfg["forecast_horizon"],
        cfg.get("train_stride", 1), verbose=True,
    )
    total_batches = (len(ds) + a.batch_size - 1) // a.batch_size
    print(f"dataset: {len(ds)} samples, {total_batches} batches/epoch "
          f"(bs={a.batch_size}, workers={a.num_workers})", flush=True)

    for label in ["cold", "warm"]:
        dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                        num_workers=a.num_workers, pin_memory=False,
                        persistent_workers=False)
        it = iter(dl)
        next(it)  # spin up workers (not timed)
        t0 = time.time()
        n = 0
        for _ in range(a.batches):
            try:
                next(it)
            except StopIteration:
                break
            n += 1
        dt = time.time() - t0
        bps = n / dt if dt > 0 else 0.0
        sps = bps * a.batch_size
        epoch_min = (total_batches / bps / 60.0) if bps > 0 else float("inf")
        print(f"[{label}] {n} batches in {dt:.1f}s -> {bps:.2f} batch/s, "
              f"{sps:.0f} samples/s | single-proc epoch ~{epoch_min:.1f} min "
              f"(4-GPU ~{epoch_min / 4.0:.1f} min)", flush=True)
        del it, dl

    print("BENCH DONE", flush=True)


if __name__ == "__main__":
    main()
