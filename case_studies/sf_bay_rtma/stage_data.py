"""Stage full RTMA target files (from M:) + ERA5 input files (from sf_bay_conus404) into data/raw.

One-off helper. Run from anywhere:
    conda run -n cosmos_wind_cnn python case_studies/sf_bay_rtma/stage_data.py

Copies ~60 GB of RTMA + the (smaller) ERA5 inputs. Skips files already staged with a
matching size, so it is safe to re-run / resume. After staging, rsync the whole
case_studies/sf_bay_rtma directory to Tallgrass.
"""
import shutil
from pathlib import Path

from cosmos_wind_cnn.utils.config import get_data_dir

RTMA_SRC = Path("m:/emeryville_crescent/03_model_setup/meteo")
CASE_DIR = Path(__file__).resolve().parent
ERA5_SRC = get_data_dir(CASE_DIR.parent / "sf_bay_conus404")
DEST = get_data_dir(CASE_DIR)
DEST.mkdir(parents=True, exist_ok=True)

RTMA_FILES = [
    "RTMA_SFbay_2p5km_eastward_wind_2011_2026_UTM10.nc",
    "RTMA_SFbay_2p5km_northward_wind_2011_2026_UTM10.nc",
    "RTMA_SFbay_2p5km_air_temperature_2011_2026_UTM10.nc",
    "RTMA_SFbay_2p5km_dew_point_temperature_2011_2026_UTM10.nc",
    "RTMA_SFbay_2p5km_air_pressure_fixed_height_2011_2026_UTM10.nc",
    "RTMA_SFbay_2p5km_precipitation_2011_2026_UTM10.nc",
    "RTMA_SFbay_2p5km_surface_height_static_UTM10.nc",   # static terrain input
]
ERA5_FILES = [
    "ERA5_eastward_wind_1940_2026_UTM.nc",
    "ERA5_northward_wind_1940_2026_UTM.nc",
    "ERA5_air_temperature_1940_2026_UTM.nc",
    "ERA5_dew_point_temperature_1940_2026_UTM.nc",
    "ERA5_air_pressure_fixed_height_1940_2026_UTM.nc",
    "ERA5_precipitation_1940_2026_UTM.nc",
    "ERA5_cloud_area_fraction_1940_2026_UTM.nc",          # cloud cover input
]


def copy_all(src_dir, files):
    for name in files:
        src, dst = src_dir / name, DEST / name
        if not src.exists():
            raise FileNotFoundError(src)
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            print(f"  [skip] {name} (already staged)")
            continue
        size_gb = src.stat().st_size / 1e9
        print(f"  [copy] {name} ({size_gb:.1f} GB) ...", flush=True)
        shutil.copy2(src, dst)


print("Staging RTMA targets...")
copy_all(RTMA_SRC, RTMA_FILES)
print("Staging ERA5 inputs...")
copy_all(ERA5_SRC, ERA5_FILES)
print("Done. Files in:", DEST)
