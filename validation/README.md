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
Runs in the `cosmos_wind_cnn` conda env (base env fails on the HRRR 2-D grid). On Windows:
```
conda activate cosmos_wind_cnn
set KMP_DUPLICATE_LIB_OK=TRUE
set COSMOS_VALIDATION_DATA_ROOT=G:\03-downscaling_meteo_cnn\validation
set COSMOS_VALIDATION_OUTPUT_ROOT=G:\03-downscaling_meteo_cnn\validation\results
```
On Caldera the SLURM launcher exports the Linux equivalents.

## Run
1. Build the bundle once:  `python stage_validation_data.py`  (preview with `--dry-run`).
2. Edit `ERA` in `run_validation.py`, then:  `python run_validation.py`.
3. Rank / pool:  `python analysis/rank_products.py`, `python analysis/combined_skill.py`.

## Product × era matrix
| Era | Window | Products |
|---|---|---|
| 1 | 1990–2010 | NOW-23, Sup3rWind, ERA5, CONUS404, UCLA, WRF_CalNev, CNN |
| 2 | 2011–2021 | + RTMA, HRRR, CNN-RTMA |
| 3 | 2022–present | RTMA, HRRR, ERA5, CNN, CNN-RTMA, NOW-23 |

## CNN file rename map (staging)
| Product | Bundle file |
|---|---|
| CNN (CONUS404) | `cnn/cnn_conus404.nc` |
| CNN-RTMA | `cnn/cnn_rtma.nc` |
| CNN-allvars | `cnn/cnn_allvars.nc` |
| CNN-windonly | `cnn/cnn_windonly.nc` |

## Caveats
- Anemometer height: IEM/NDBC/CWOP treated at 10 m (log-correction is a no-op); USGS moorings
  kept at measured height, compared directly to 10 m model output. Documented, not silently corrected.
- `CONUS404-downscaled` / `-100m` are wired in `config.MODELS` but excluded from the bundle
  (in no era's product list); they skip-clean if absent.
