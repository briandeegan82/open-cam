"""Measured USGS splib07a NIR material chart (spectra/usgs_nir)."""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import nir_materials  # noqa: E402

CHART = REPO / nir_materials.OUT_REL


def _load() -> dict[str, np.ndarray]:
    return {p.stem.split("_", 1)[1]: np.loadtxt(p, delimiter=",") for p in sorted(CHART.glob("*.csv"))}


def test_chart_files_match_member_table():
    names = [p.stem for p in sorted(CHART.glob("*.csv"))]
    assert names == [f"{i:02d}_{n}" for i, n in enumerate(nir_materials.NIR_CHART, start=1)]
    assert len(names) == 24


def test_curves_cover_380_to_1000_nm_and_are_physical():
    for name, d in _load().items():
        wl, r = d[:, 0], d[:, 1]
        assert wl[0] <= 380 and wl[-1] >= 1000, name
        assert np.all(np.diff(wl) == 1.0), name
        assert np.all((r > 0.0) & (r < 1.05)), name
        assert np.max(np.abs(np.diff(r))) < 0.02, name  # no splice steps / bad channels


def test_measured_nir_is_not_a_730nm_hold():
    """Leaves and dyed fabrics brighten past the red edge; carbon blacks and the reference stay flat."""
    c = _load()

    def r(name, lam):
        return float(np.interp(lam, c[name][:, 0], c[name][:, 1]))

    for leaf in ("oak_leaf_fresh", "aspen_leaf_green", "pine_needles"):
        assert r(leaf, 850) > 3 * r(leaf, 650), leaf
    assert r("tarp_green", 850) > 4 * r("tarp_green", 730)
    assert r("nylon_green", 850) > 1.5 * r("nylon_green", 730)
    for flat in ("hdpe_black", "brick_grey", "spectralon_white"):
        assert abs(r(flat, 940) - r(flat, 730)) < 0.03, flat
    assert r("spectralon_white", 550) > 0.97


def test_patch_dir_renders_the_chart(tmp_path):
    out = tmp_path / "gen"
    cmd = [sys.executable, str(REPO / "tools" / "build_colorchecker_scene.py"), "--out-dir", str(out)]
    cmd += ["--patch-dir", nir_materials.OUT_REL, "--lambda-max", "1100", "--pbrt-lambda-max", "1100"]
    subprocess.run(cmd, check=True, cwd=REPO, capture_output=True)
    ref = np.load(out / "spectral_reference_1nm.npz")
    assert ref["reflectance"].shape[0] == 24
    i850 = int(np.argmin(np.abs(ref["wavelength_nm"] - 850)))
    oak = np.loadtxt(CHART / "01_oak_leaf_fresh.csv", delimiter=",")
    assert ref["reflectance"][0, i850] == pytest.approx(np.interp(850, oak[:, 0], oak[:, 1]), abs=1e-4)
    meta = json.loads((out / "colorchecker_manifest.json").read_text())
    assert "patches" in json.dumps(meta)


def test_default_patch_dir_is_still_xrite(tmp_path):
    out = tmp_path / "gen"
    cmd = [sys.executable, str(REPO / "tools" / "build_colorchecker_scene.py"), "--out-dir", str(out)]
    subprocess.run(cmd, check=True, cwd=REPO, capture_output=True)
    ref = np.load(out / "spectral_reference_1nm.npz")
    white = np.loadtxt(REPO / "spectra" / "xrite" / "19_white_005D.csv", delimiter=",")
    i = int(np.argmin(np.abs(ref["wavelength_nm"] - 550)))
    assert ref["reflectance"][18, i] == pytest.approx(np.interp(550, white[:, 0], white[:, 1]), abs=1e-4)


@pytest.mark.skipif(
    not (nir_materials.cache_dir(REPO) / "usgs" / "splib07a_ASCII.zip").is_file(),
    reason="USGS archive not in the cache (python tools/nir_materials.py downloads it)",
)
def test_committed_csvs_regenerate_from_archive(tmp_path):
    z = nir_materials.cache_dir(REPO) / "usgs" / "splib07a_ASCII.zip"
    dest = nir_materials.cache_dir(tmp_path) / "usgs"
    dest.mkdir(parents=True)
    (dest / z.name).symlink_to(z)
    assert zipfile.is_zipfile(z)
    for p in nir_materials.write_chart(tmp_path):
        assert p.read_text() == (CHART / p.name).read_text(), p.name
