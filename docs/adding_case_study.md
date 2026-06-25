# Adding a New Case Study

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
├── data/
│   └── raw/.gitkeep
└── README.md
```

When you run the pipeline, a `results/<run_name>/` directory is created automatically with subdirectories for checkpoint, processed data, logs, inference output, and evaluation output.

## 2. Prepare the data

Follow the pipeline in [data_preparation.md](data_preparation.md):

1. Convert CONUS404 data to your target UTM grid
2. Interpolate ERA5 data to the same grid
3. Place all NetCDF files in `case_studies/my_study/data/raw/`

Requirements:
- One NetCDF file per variable
- All files on the same spatial grid (same x/y or lat/lon coordinates)
- Overlapping time coordinates

## 3. Configure preprocessing

Edit `case_studies/my_study/configs/preprocessing.yaml`:

```yaml
file_dict:
  conus404_u: 'your_high_res_u_wind.nc'
  era5_u: 'your_low_res_u_wind.nc'
  conus404_v: 'your_high_res_v_wind.nc'
  era5_v: 'your_low_res_v_wind.nc'
  # ... add all variable pairs

train_ratio: 0.7
val_ratio: 0.15
test_ratio: 0.15
```

The variable keys (e.g., `conus404_u`, `era5_u`) must match the keys used in the training config's `variable_pairs`.

### Optional: non-CONUS404 targets and gappy products

The pipeline defaults to `conus404_` (target) and `era5_` (input) key prefixes. To use a
different high-resolution target product, set the prefixes explicitly in
`preprocessing.yaml`:

```yaml
target_prefix: 'rtma_'        # high-resolution reference grid
input_prefix: 'era5_'         # coarse input
regular_time_grid: true       # reindex onto a complete hourly axis, NaN-filling gaps
```

`regular_time_grid` is needed for products with missing hours (e.g. RTMA): missing
timestamps become NaN rows, which the dataset's NaN-window dropping then excludes, so no
sequence window silently spans a time gap. See `case_studies/sf_bay_rtma` for a worked
example (RTMA 2.5 km target).

## 4. Configure training

Edit `case_studies/my_study/configs/training.yaml`:

- Update `variable_pairs` to match your preprocessing keys
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

All outputs are saved under `case_studies/my_study/results/<run_name>/`.

## 7. Update the case study README

Edit `case_studies/my_study/README.md` with domain-specific details:
- Geographic extent and coordinate system
- Data sources and time period
- Any special considerations (topography, coastal effects, etc.)

## Tips

- Start with the same hyperparameters as SF Bay and tune from there
- Use TensorBoard to compare training curves: `tensorboard --logdir case_studies/my_study/results/<run_name>/logs`
- The validation notebooks can be parameterized to work with any case study
