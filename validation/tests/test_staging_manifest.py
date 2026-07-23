import importlib
import sys
from pathlib import Path

VALIDATION = Path(__file__).resolve().parents[1]
CANONICAL = {
    "obs", "moorings", "reference", "era5", "hrrr", "conus404", "rtma",
    "now23", "sup3rwind", "ucla_reanalysis", "wrf_calnev", "cnn", "aorc",
}


def _mod():
    sys.path.insert(0, str(VALIDATION))
    sys.modules.pop("stage_validation_data", None)
    return importlib.import_module("stage_validation_data")


def test_manifest_dest_subdirs_are_canonical():
    m = _mod()
    for entry in m.MANIFEST:
        assert entry.dest in CANONICAL, f"non-canonical dest {entry.dest!r}"


def test_manifest_covers_every_engine_product():
    m = _mod()
    dests = {e.dest for e in m.MANIFEST}
    # every product dir the engine reads must be represented
    for need in ("era5", "hrrr", "conus404", "rtma", "now23", "sup3rwind",
                 "ucla_reanalysis", "wrf_calnev", "cnn", "aorc", "obs", "moorings", "reference"):
        assert need in dests, f"MANIFEST missing {need}"


def test_cnn_entries_have_rename_map():
    m = _mod()
    cnn_entries = [e for e in m.MANIFEST if e.dest == "cnn"]
    assert cnn_entries, "no cnn entries"
    targets = set()
    for e in cnn_entries:
        assert e.rename, f"cnn entry {e.glob} needs a rename map"
        targets.update(e.rename.values())
    assert {"cnn_conus404.nc", "cnn_rtma.nc", "cnn_allvars.nc", "cnn_windonly.nc"} <= targets


def test_dry_run_touches_nothing(tmp_path, monkeypatch, capsys):
    m = _mod()
    dest_root = tmp_path / "bundle"
    m.stage(dest_root, dry_run=True)
    out = capsys.readouterr().out
    assert "DRY-RUN" in out
    assert not dest_root.exists() or not any(dest_root.rglob("*.nc"))
