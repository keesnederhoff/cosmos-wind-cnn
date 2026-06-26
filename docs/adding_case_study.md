# Adding a New Case Study

## Storage

The repo holds **only** `configs/` + `README.md` for each case study. Raw data and run outputs are external, controlled by two env vars that must be set before running anything (the code raises a clear error if either is unset):

```bat
:: Windows — point both at your storage drive
set COSMOS_DATA_ROOT=G:\03-downscaling_meteo_cnn
set COSMOS_RESULTS_ROOT=G:\03-downscaling_meteo_cnn
```
```bash
# Linux/HPC: already exported by the Tallgrass SLURM scripts (caldera base)
```

With those set:
- Raw inputs go to `$COSMOS_DATA_ROOT/my_study/raw_data/` (locally `G:\03-downscaling_meteo_cnn\my_study\raw_data\`)
- Per-run outputs land at `$COSMOS_RESULTS_ROOT/my_study/results/<run_name>/` (locally `G:\03-downscaling_meteo_cnn\my_study\results\<run_name>\`)

`case_studies/_template/` is the canonical example — copy it and follow the steps below.

## 1. Create the directory structure

Copy the template:

```bash
cp -r case_studies/_template case_studies/my_study
```

This gives you:
```
case_studies/my_study/
├── configs/
│   ├── preprocessing.yaml
│   ├── training.yaml
│   └── inference_preprocessing.yaml
└── README.md
```

When you run the pipeline, a `results/<run_name>/` directory is created automatically under `$COSMOS_RESULTS_ROOT/my_study/results/` with subdirectories for checkpoint, processed data, logs, inference output, and evaluation output.

## 2. Prepare the data

Follow the pipeline in [data_preparation.md](data_preparation.md):

1. Convert your high-resolution (HR) target data to your target UTM grid
2. Interpolate your low-resolution (LR) input data to the same grid
3. Place all NetCDF files in `$COSMOS_DATA_ROOT/my_study/raw_data/`

Requirements:
- One NetCDF file per variable
- All files on the same spatial grid (same x/y or lat/lon coordinates)
- Overlapping time coordinates

## 3. Configure preprocessing

Edit `case_studies/my_study/configs/preprocessing.yaml`:

```yaml
file_dict:
  hr_u: 'your_high_res_u_wind.nc'
  lr_u: 'your_low_res_u_wind.nc'
  hr_v: 'your_high_res_v_wind.nc'
  lr_v: 'your_low_res_v_wind.nc'
  # ... add all variable pairs

train_ratio: 0.7
val_ratio: 0.15
test_ratio: 0.15
```

The variable keys (e.g., `hr_u`, `lr_u`) must match the keys used in the training config's `variable_pairs`.
All keys use the `hr_` prefix for the high-resolution target and `lr_` for the low-resolution input —
this convention is dataset-agnostic (CONUS404, RTMA, ERA5, CMIP6, or any other source).

### Optional: gappy data products

If your HR data product has missing hours (e.g. RTMA, with ~1% missing), set:

```yaml
regular_time_grid: true       # reindex onto a complete hourly axis, NaN-filling gaps
```

This reindexes onto a complete hourly axis before splitting. Missing timestamps become NaN rows,
which the dataset's NaN-window dropping then excludes, so no sequence window silently spans a time gap.
See `case_studies/sf_bay_rtma` for a worked example (RTMA 2.5 km HR target / ERA5 LR input).

## 4. Configure training

Edit `case_studies/my_study/configs/training.yaml`:

- Set `hr_source` and `lr_source` to record the actual datasets used (provenance labels, e.g.
  `hr_source: CONUS404` / `lr_source: ERA5`). These do not affect model behavior but are
  archived with the run for reproducibility.
- Update `variable_pairs` to match your preprocessing keys, mapping `high_res:` / `low_res:`
  values to the `hr_` / `lr_` keys defined in your preprocessing config:
  ```yaml
  hr_source: CONUS404
  lr_source: ERA5
  variable_pairs:
    wind_u:
      high_res: 'hr_u'
      low_res:  'lr_u'
    wind_v:
      high_res: 'hr_v'
      low_res:  'lr_v'
    # ... all pairs
  ```
- Adjust `batch_size` based on your GPU memory and grid size
- Consider reducing `base_channels` if GPU memory is limited
- Tune `sequence_length` based on the temporal autocorrelation of your data

## 5. Configure inference preprocessing

Edit `case_studies/my_study/configs/inference_preprocessing.yaml`:

- Map each input variable to its source file and variable name
- Set physical bounds (e.g., wind speed limits, temperature range)
- Specify interpolation method and compression level

This config is used when running inference on data that isn't already on the target grid (e.g., raw ERA5 for a different time period, or CMIP6 data).

## 6. Run the pipeline

### Recommended: full pipeline

```bash
python scripts/run_training_pipeline.py \
    --case-study case_studies/my_study \
    --run-name first_run \
    --gpus 4
```

This runs all 5 steps: preprocess → train → archive configs → inference → evaluate. Skip individual steps with `--skip-preprocess`, `--skip-train`, `--skip-inference`, `--skip-eval`.

### Alternative: individual scripts

```bash
python scripts/preprocess_training.py --case-study case_studies/my_study --run-name first_run
python scripts/train.py --case-study case_studies/my_study --run-name first_run
python scripts/evaluate.py --case-study case_studies/my_study --run-name first_run
```

### Standalone inference (new data)

```bash
python scripts/run_inference.py \
    --case-study case_studies/my_study \
    --run-name first_run \
    --start-date 2024-01-01 \
    --end-date 2026-12-31
```

All outputs are saved under `$COSMOS_RESULTS_ROOT/my_study/results/<run_name>/`.

## 7. Update the case study README

Edit `case_studies/my_study/README.md` with domain-specific details:
- Geographic extent and coordinate system
- Data sources and time period
- Any special considerations (topography, coastal effects, etc.)

## Tips

- Start with the same hyperparameters as SF Bay and tune from there
- Use TensorBoard to compare training curves: `tensorboard --logdir $COSMOS_RESULTS_ROOT/my_study/results/<run_name>/logs`
- The validation notebooks can be parameterized to work with any case study
