import importlib
import os
import sys
from pathlib import Path

import pytest


def _fresh_config(data_root, out_root):
    """Import config with roots pointed at the given dirs (fresh each call)."""
    os.environ["COSMOS_VALIDATION_DATA_ROOT"] = str(data_root)
    os.environ["COSMOS_VALIDATION_OUTPUT_ROOT"] = str(out_root)
    sys.modules.pop("config", None)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # validation/
    return importlib.import_module("config")


def test_roots_resolve_from_env(tmp_path):
    cfg = _fresh_config(tmp_path / "data", tmp_path / "out")
    assert cfg.DATA_ROOT == tmp_path / "data"
    assert cfg.OUTPUT_ROOT == tmp_path / "out"


def test_missing_env_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("COSMOS_VALIDATION_DATA_ROOT", raising=False)
    monkeypatch.delenv("COSMOS_VALIDATION_OUTPUT_ROOT", raising=False)
    sys.modules.pop("config", None)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    with pytest.raises(RuntimeError, match="COSMOS_VALIDATION_DATA_ROOT"):
        importlib.import_module("config")


def test_product_paths_anchor_under_data_root(tmp_path):
    cfg = _fresh_config(tmp_path / "data", tmp_path / "out")
    dr = tmp_path / "data"
    assert cfg.PWS_DIR == dr / "observed_data"
    assert cfg.MODELS["ERA5"]["u_file"] == dr / "modeled_data" / "era5" / "ERA5_eastward_wind_1940_2026_UTM.nc"
    assert cfg.MODELS["RTMA"]["data_dir"] == dr / "modeled_data" / "rtma"
    assert cfg.MODELS["AORC"]["u_file"].parent == dr / "modeled_data" / "aorc"
    assert cfg.MODELS["CNN"]["u_file"] == dr / "modeled_data" / "cnn_fullrecord" / "CNN_conus404_full_record_ERA5_19400101_20270101.nc"
    assert cfg.MODELS["CNN-RTMA-20260625"]["u_file"] == dr / "modeled_data" / "cnn_fullrecord" / "CNN_rtma_full_record_ERA5_19400101_20270101.nc"
    assert cfg.MODELS["CNN-allvars"]["u_file"] == dr / "modeled_data" / "os_av_bc24_terr_res_s2" / "full_record_ERA5_20110101_20260101.nc"
    assert cfg.LDB_FILE == dr / "reference" / "deltabay.ldb"
    assert cfg.USGS_MOORINGS["ERO20_GRZ"]["file_path"].parent == dr / "observed_data"
    for dropped in ("HRRR", "CONUS404", "UCLA", "WRF_CalNev", "NOW-23", "Sup3rWind"):
        assert dropped not in cfg.MODELS


def test_no_hardcoded_path_literals_in_source():
    import re
    src = (Path(__file__).resolve().parents[1] / "config.py").read_text(encoding="utf-8")
    # Flag actual path literals like Path(r"m:\...") — NOT drive letters that appear
    # in the RuntimeError help text (e.g. "set VAR=G:\...").
    hits = re.findall(r'Path\(\s*r?["\'][A-Za-z]:', src)
    assert not hits, f"hardcoded drive-path literal(s) still in config.py: {hits}"
