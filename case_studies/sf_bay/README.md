# SF Bay Case Study

Statistical downscaling of meteorological variables for the San Francisco Bay region.

## Data

- **High-resolution:** CONUS404 at 4 km (SFbay domain), UTM Zone 10N
- **Low-resolution:** ERA5 at ~31 km, interpolated to the same UTM10N grid
- **Time period:** Water Years 2020-2021
- **Domain extent:** UTM10N x=[425-596 km], y=[4092-4257 km]

## Variables

| Variable | CONUS404 source | ERA5 source |
|----------|----------------|-------------|
| Eastward wind (U) | U10 | u10 |
| Northward wind (V) | V10 | v10 |
| Dew point temperature | TD2 | d2m |
| Air pressure (MSL) | PSFC (converted) | msl |
| Solar radiation | ACSWDNB (converted) | ssr |
| Cloud cover (input only) | - | tcc |

## Usage

```bash
python scripts/preprocess.py --case-study case_studies/sf_bay
python scripts/train.py --case-study case_studies/sf_bay
python scripts/evaluate.py --case-study case_studies/sf_bay
python scripts/inference.py --case-study case_studies/sf_bay
```
