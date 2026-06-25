# SF Bay RTMA Case Study

Statistical downscaling of meteorological variables for the San Francisco Bay
region, using RTMA as the high-resolution training target (sister study to the
CONUS404-based `sf_bay`).

## Data

- **High-resolution target:** RTMA at 2.5 km (SF Bay domain), UTM Zone 10N, hourly
- **Low-resolution input:** ERA5 at ~31 km, interpolated to the same UTM10N grid
- **Training period:** 2011-2026 (RTMA availability; ~1% missing hours, mostly 2013)
- **Inference period:** 1940-2026 (full ERA5 record; pre-2011 extrapolates)
- **Grid:** 162 (x) x 123 (y) @ 2.5 km

## Variables (6 pairs)

| Variable | RTMA source | ERA5 source |
|----------|-------------|-------------|
| Eastward wind (U) | eastward_wind | u10 |
| Northward wind (V) | northward_wind | v10 |
| Air temperature | air_temperature | t2m |
| Dew point temperature | dew_point_temperature | d2m |
| Air pressure (MSL) | air_pressure_fixed_height | msl |
| Precipitation | precipitation | tp |

No shortwave/longwave radiation (RTMA has none). RTMA precipitation skill for
extremes is uncertain — validate `rtma_rain` against gauges before using it for
compound-flood forcing.

## Usage

```bash
python scripts/run_training_pipeline.py \
    --case-study case_studies/sf_bay_rtma \
    --run-name first_run \
    --gpus 4
```

## Deploy on Tallgrass (GPU, 4x V100)

1. **Stage the raw data** into `data/raw/` (6 RTMA targets from `M:` + 6 ERA5 inputs;
   ~60 GB, skips already-copied files):

   ```bash
   conda run -n cosmos_wind_cnn python case_studies/sf_bay_rtma/stage_data.py
   ```

2. **Sync to Tallgrass** (rsync the case study; raw data is git-ignored so it must be
   transferred explicitly), then submit the existing pipeline SLURM with the case study
   overridden (no edit to the SLURM file needed):

   ```bash
   sbatch --export=ALL,CASE_STUDY=case_studies/sf_bay_rtma scripts/gpu_tallgrass.slurm
   ```

   The full pipeline (preprocess → train DDP → archive → inference → evaluate) writes to
   `case_studies/sf_bay_rtma/results/<job_id>/`. A CPU-only run uses
   `cpu_tallgrass.slurm` the same way. After pulling new code on HPC, `pip install -e .`.

See `docs/adding_case_study.md` for the full workflow and
`docs/superpowers/specs/2026-06-22-rtma-sf-bay-downscaling-design.md` for the design.
