# sf_bay_rtma hyperparameter sweep

Launched 2026-06-30. 21 train-only variants exploring `base_channels`,
`sequence_length`, `dropout_rate`, `learning_rate` around the baseline
(`base_channels=16, sequence_length=6, dropout=0.1, lr=3e-4` = job 3761035).

## How it runs (autonomous — no laptop/session needed)
- Slurm **job array** `sfb_sweep` (`gpu_sweep_array.slurm`, `--array=0-20%2`):
  Slurm keeps **2 runs going at a time** and auto-launches the next as GPUs free.
  Each task = 2 GPUs, non-exclusive (backfills onto spare GPUs), 48h wall.
- Variants are listed in `sweep_manifest.txt` (one line per run:
  `run_name base_channels seq_len dropout lr`).
- Per-variant values are injected via env vars (`SWEEP_BASE_CHANNELS`,
  `SWEEP_SEQ_LEN`, `SWEEP_DROPOUT`, `SWEEP_LR`) read by `scripts/train.py`
  (and `step_inference`). Each run **symlinks the shared read-only memmap**
  (`.../results/3737874/data_processed/memmap`) — no data duplication.
- Runs are **train-only** (`--skip-inference --skip-eval`); compared by
  validation loss. Outputs land in
  `/caldera/projects/usgs/hazards/pcmsc/cosmos/cnn_wind_sfbay/sf_bay_rtma/results/<run_name>/`
  (checkpoint/best_model.pth, logs/, sweep_params.txt). Array logs in `sweep_logs/`.

## Check progress / results
    cd /home/cnederhoff/cosmos/cosmos-wind-cnn
    conda activate cosmos_wind_cnn          # source miniforge first
    python scripts/sweep_collect.py         # table sorted by best val_loss
    squeue -u cnederhoff                    # queue state (sfb_sweep_<id>)
    sacct -j <arrayjobid> --format=JobID,JobName,State,Elapsed,ExitCode

## After the sweep: full eval on the winner(s)
Pick the lowest-val_loss variant, then run the FULL pipeline (inference +
evaluate over the record) on it, reusing its trained checkpoint:
    sbatch <a full-pipeline slurm> with --run-name <winner> --skip-preprocess \
        --skip-train  (and the matching SWEEP_* env exports so the model
        rebuilds at the right base_channels/seq_len/dropout)
Then read wind-speed RMSE vs ERA5 + skill from output_evaluation/.

## Control
- Cancel the whole sweep:        `scancel <arrayjobid>`
- Cancel one variant:            `scancel <arrayjobid>_<index>`
- Change concurrency (e.g. to 3): `scontrol update arraytaskthrottle=3 jobid=<arrayjobid>`
- Add variants: append lines to `sweep_manifest.txt` and resubmit with a wider
  `--array` range.
