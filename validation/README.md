# SF Bay Meteorological Product Validation

Point-observation validation & ranking of gridded wind (and met) products over SF Bay —
which product best reproduces observed winds, and is therefore the most defensible forcing
for the SF Bay Community Model over 1940–present. Relocated from
`g:\01_meteorlogical_analysis_sfbay\` (see `docs/2026-07-23-validation-relocation-design.md`).

## Layout
- `config.py` — single source of truth (products, obs, station scope, run options).
- `validate_met_models.py` — the engine.
- `run_validation.py` — era-aware driver (set `ERA`, run).
- `analysis/` — ranking, combined skill, wind roses, comparison slides.
- `reference/` — station inventory + land boundary.
- `stage_validation_data.py` — build the data bundle from raw sources.

## Environment
Runs in the `cosmos_wind_cnn` conda env (base env fails on some 2-D grids). On Windows:
```
conda activate cosmos_wind_cnn
set KMP_DUPLICATE_LIB_OK=TRUE
set COSMOS_VALIDATION_DATA_ROOT=g:\01_meteorlogical_analysis_sfbay
set COSMOS_VALIDATION_OUTPUT_ROOT=g:\01_meteorlogical_analysis_sfbay\results
```
On Caldera the SLURM launcher exports the Linux equivalents.

The `COSMOS_VALIDATION_DATA_ROOT` bundle is self-contained:
- `modeled_data\` — `era5`, `rtma`, `aorc`, `cnn_fullrecord`, `os_av_bc24_terr_res_s2`, `os_wo_bc24_base_res_s2`
- `observed_data\` — IEM / NDBC / CWOP archive NetCDFs + the ERO20 Grizzly Bay USGS mooring
- `reference\` — station inventory + `deltabay.ldb` land boundary

## Run
1. Build the bundle once:  `python stage_validation_data.py`  (preview with `--dry-run`).
2. Edit `ERA` in `run_validation.py`, then:  `python run_validation.py`.
3. Rank / pool:  `python analysis/rank_products.py`, `python analysis/combined_skill.py`.

## Active products
Active set: **ERA5, RTMA, AORC, CNN** — where CNN spans four variants: CNN (CONUS404
truth), CNN-RTMA, CNN-allvars, CNN-windonly.

| Era | Window | Products |
|---|---|---|
| 1 | 1990–2010 | ERA5, AORC, CNN, CNN-RTMA |
| 2 | 2011–2021 | ERA5, RTMA, AORC, CNN, CNN-RTMA, CNN-allvars, CNN-windonly |
| 3 | 2022–present | ERA5, RTMA, AORC, CNN, CNN-RTMA, CNN-allvars, CNN-windonly |

> Secondary products (HRRR, CONUS404, UCLA, WRF_CalNev, NOW-23, Sup3rWind, USGS
> moorings) were dropped from the active config; re-add if needed.

## CNN file map (`modeled_data\`)
| Product | Bundle file |
|---|---|
| CNN (CONUS404) | `cnn_fullrecord\CNN_conus404_full_record_ERA5_19400101_20270101.nc` |
| CNN-RTMA | `cnn_fullrecord\CNN_rtma_full_record_ERA5_19400101_20270101.nc` |
| CNN-allvars | `os_av_bc24_terr_res_s2\full_record_ERA5_20110101_20260101.nc` |
| CNN-windonly | `os_wo_bc24_base_res_s2\full_record_ERA5_20110101_20260101.nc` |

## Caveats
- Anemometer height: IEM/NDBC/CWOP treated at 10 m (log-correction is a no-op); the ERO20
  USGS mooring is kept at measured height, compared directly to 10 m model output. Documented,
  not silently corrected.
- AORC 10 m wind masks the open Pacific (NaN) and switches provenance at 2018 (NLDAS-2 →
  URMA); offshore NDBC buoys drop out of AORC's stats.
