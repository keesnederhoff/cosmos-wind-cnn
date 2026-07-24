"""
Assemble the SF Bay validation data bundle from its scattered source drives
into the canonical $COSMOS_VALIDATION_DATA_ROOT layout.

Produces the local bundle AND defines the exact set shipped to Caldera.
The full bundle is ~490 GB (CNN full-record files 29-86 GB each, UCLA 82 GB,
AORC 81 GB), so copies use robocopy (multithreaded, resumable) on Windows.

Preview:              python stage_validation_data.py --dry-run
Full copy (~490 GB): python stage_validation_data.py
Subset (fast):       python stage_validation_data.py --products=era5,rtma,obs,reference

# === CONFIGURATION ===  (source drives; edit if the raw data moves)
"""
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --- source roots (Windows raw-data homes; source of truth stays here) ------
SFBAY_OBS   = Path(r"d:\data\meteo\SFBay\data")
EMERYVILLE  = Path(r"m:\emeryville_crescent")
CNN_OUT     = Path(r"G:\03-downscaling_meteo_cnn")
PROJECT_OLD = Path(r"g:\01_meteorlogical_analysis_sfbay")
LDB_SRC     = Path(r"f:\Alameda\03_modelsetup\_inputs\inputnoah\deltabay.ldb")


@dataclass
class Entry:
    src: Path
    glob: str
    dest: str                       # canonical subdir name
    rename: dict = field(default_factory=dict)   # {source_name: dest_name}


MANIFEST = [
    # obs
    Entry(SFBAY_OBS, "pws_sfbay_waterfront_*.nc", "obs"),
    Entry(SFBAY_OBS, "ERO20_GrizzlyBay_meteorological.nc", "obs"),
    # moorings
    Entry(EMERYVILLE / "01_data" / "whales_tale", "DMP23MW*.nc", "moorings"),
    Entry(EMERYVILLE / "01_data" / "emc_data", "EMC26MW101met.nc", "moorings"),
    # reference
    Entry(PROJECT_OLD / "reference", "station_inventory.*", "reference"),
    Entry(LDB_SRC.parent, LDB_SRC.name, "reference"),
    # reanalysis / hi-res products
    Entry(CNN_OUT / "sf_bay_conus404" / "raw_data", "ERA5_*_UTM.nc", "era5"),
    Entry(EMERYVILLE / "04_model_runs" / "meteo", "HRRR_WY2015-WY2026_*.nc", "hrrr"),
    Entry(EMERYVILLE / "03_model_setup" / "meteo", "CONUS404_SFbay_4km_*.nc", "conus404"),
    Entry(EMERYVILLE / "04_model_runs" / "meteo" / "rtma_gee_grid", "RTMA_grid_2p5km_*.nc", "rtma"),
    Entry(EMERYVILLE / "01_data" / "other_meteo_data" / "now23", "now23_ca_bayarea_box_*.nc", "now23"),
    Entry(EMERYVILLE / "01_data" / "other_meteo_data" / "sup3rwind", "sup3rwind_bayarea_box_*.nc", "sup3rwind"),
    Entry(EMERYVILLE / "01_data" / "other_meteo_data" / "data" / "ucla_reanalysis", "era5_reanalysis_1hr_*.nc", "ucla_reanalysis"),
    Entry(EMERYVILLE / "01_data" / "other_meteo_data" / "data" / "wrf_calnev", "wrfout_d02_V1_*_bayarea.nc", "wrf_calnev"),
    Entry(PROJECT_OLD / "data" / "aorc", "AORC_SFbay_800m_*.nc", "aorc"),
    # cnn (rename to distinct names — sources share full_record_ERA5_*.nc)
    Entry(CNN_OUT / "sf_bay_conus404" / "results" / "3679830" / "output_inference",
          "full_record_ERA5_19400101_20270101.nc", "cnn",
          rename={"full_record_ERA5_19400101_20270101.nc": "cnn_conus404.nc"}),
    Entry(CNN_OUT / "sf_bay_rtma" / "results" / "3732177" / "output_inference",
          "full_record_ERA5_19400101_20270101.nc", "cnn",
          rename={"full_record_ERA5_19400101_20270101.nc": "cnn_rtma.nc"}),
    Entry(PROJECT_OLD / "data" / "os_av_bc24_terr_res_s2",
          "full_record_ERA5_20110101_20260101.nc", "cnn",
          rename={"full_record_ERA5_20110101_20260101.nc": "cnn_allvars.nc"}),
    Entry(PROJECT_OLD / "data" / "os_wo_bc24_base_res_s2",
          "full_record_ERA5_20110101_20260101.nc", "cnn",
          rename={"full_record_ERA5_20110101_20260101.nc": "cnn_windonly.nc"}),
    Entry(PROJECT_OLD / "data" / "x10_wo_bc24_res_d1_s2",
          "full_record_ERA5_20110101_20260101.nc", "cnn",
          rename={"full_record_ERA5_20110101_20260101.nc": "cnn_extreme.nc"}),
]


def _robocopy(src_dir: Path, dst_dir: Path, file_pattern: str) -> None:
    """Bulk copy via robocopy — multithreaded, resumable, network-friendly.
    robocopy exit codes 0-7 are success; 8+ is a real error."""
    cp = subprocess.run(
        ["robocopy", str(src_dir), str(dst_dir), file_pattern,
         "/MT:16", "/Z", "/R:2", "/W:5", "/NFL", "/NDL", "/NP", "/NJH", "/NJS"],
        capture_output=True, text=True,
    )
    if cp.returncode >= 8:
        raise RuntimeError(
            f"robocopy failed ({cp.returncode}) for {src_dir}\\{file_pattern}\n"
            f"{cp.stdout}\n{cp.stderr}"
        )


def stage(dest_root: Path, dry_run: bool = False, products=None) -> None:
    """Stage the bundle into dest_root.

    products=None -> every MANIFEST entry; else an iterable of canonical dest
    names (e.g. ['era5','rtma','obs','reference']) to stage a subset.
    """
    if dry_run:
        print("DRY-RUN — no files will be copied")
    total_files = total_bytes = 0
    for e in MANIFEST:
        if products is not None and e.dest not in products:
            continue
        dst_dir = dest_root / e.dest
        if not e.src.exists():
            print(f"  SKIP (source absent): {e.src}")
            continue
        matches = sorted(e.src.glob(e.glob))
        if not matches:
            print(f"  SKIP (no match): {e.src / e.glob}")
            continue
        total_files += len(matches)
        total_bytes += sum(f.stat().st_size for f in matches)
        if dry_run:
            for f in matches:
                out_name = e.rename.get(f.name, f.name)
                print(f"  DRY-RUN {f}  ->  {dst_dir / out_name}")
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        if e.rename:
            # single-file entries needing a distinct name (robocopy can't rename)
            for f in matches:
                target = dst_dir / e.rename.get(f.name, f.name)
                shutil.copy2(f, target)
                print(f"  staged {f.name}  ->  {e.dest}/{target.name}")
        elif os.name == "nt":
            _robocopy(e.src, dst_dir, e.glob)
            print(f"  staged {len(matches)} file(s)  ->  {e.dest}/  (robocopy)")
        else:  # Caldera / Linux
            for f in matches:
                shutil.copy2(f, dst_dir / f.name)
            print(f"  staged {len(matches)} file(s)  ->  {e.dest}/")
    print(f"\n{'planned' if dry_run else 'staged'}: {total_files} files, "
          f"{total_bytes / 1024**3:.1f} GB")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    products = None   # default: everything;  --products=era5,rtma,obs,reference for a subset
    for a in sys.argv:
        if a.startswith("--products="):
            products = [p.strip() for p in a.split("=", 1)[1].split(",") if p.strip()]
    root = Path(os.environ.get("COSMOS_VALIDATION_DATA_ROOT",
                               r"G:\03-downscaling_meteo_cnn\validation"))
    sel = "ALL products" if products is None else ", ".join(products)
    print(f"{'DRY-RUN: ' if dry else ''}staging {sel} -> {root}")
    stage(root, dry_run=dry, products=products)
    print("done.")
