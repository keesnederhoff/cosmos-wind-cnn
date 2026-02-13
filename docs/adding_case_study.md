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
│   └── training.yaml
├── data/
│   ├── raw/.gitkeep
│   └── processed/.gitkeep
├── checkpoints/.gitkeep
├── logs/.gitkeep
├── outputs/.gitkeep
└── README.md
```

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

## 4. Configure training

Edit `case_studies/my_study/configs/training.yaml`:

- Update `variable_pairs` to match your preprocessing keys
- Adjust `batch_size` based on your GPU memory and grid size
- Consider reducing `base_channels` if GPU memory is limited
- Tune `sequence_length` based on the temporal autocorrelation of your data

## 5. Run the pipeline

```bash
# Preprocess
python scripts/preprocess.py --case-study case_studies/my_study

# Train
python scripts/train.py --case-study case_studies/my_study

# Evaluate
python scripts/evaluate.py --case-study case_studies/my_study

# Inference on new data
python scripts/inference.py --case-study case_studies/my_study
```

## 6. Update the case study README

Edit `case_studies/my_study/README.md` with domain-specific details:
- Geographic extent and coordinate system
- Data sources and time period
- Any special considerations (topography, coastal effects, etc.)

## Tips

- Start with the same hyperparameters as SF Bay and tune from there
- Use TensorBoard to compare training curves across case studies
- The validation notebooks can be parameterized to work with any case study
