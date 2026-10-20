"""Winter / snow options of the highway builder (tools/highway_snow.py)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import build_highway_scene as hw  # noqa: E402
import highway_snow as snow  # noqa: E402

BASE = ["--proxy-cars", "--allow-missing-assets", "--xres", "64", "--yres", "36", "--pixelsamples", "1"]


def _build(tmp_path: Path, *extra: str) -> tuple[str, dict]:
    hw.main([*BASE, "--out-dir", str(tmp_path), *extra])
    return (tmp_path / "highway.pbrt").read_text(), json.loads((tmp_path / "highway_manifest.json").read_text())


def test_albedo_physics():
    wl = np.array([500.0, 1030.0])
    fine, coarse = snow.snow_albedo(wl, 0.1), snow.snow_albedo(wl, 1.0)
    assert fine[0] > 0.98 and 0.75 < fine[1] < 0.9  # Wiscombe & Warren 1980: ~0.8 at 1.03 um for r = 50 um
    assert np.all(coarse < fine) and coarse[1] < 0.6  # bigger grains absorb more, mostly in the NIR
    assert (
        snow.snow_albedo(wl, 0.25, "heavy")[0] < snow.snow_albedo(wl, 0.25, "light")[0] < snow.snow_albedo(wl, 0.25)[0]
    )


def test_snowfall_visibility():
    v1 = 2.996 / snow.snowfall_extinction(1.0, 0.1)
    assert 500 < v1 < 2000  # ~1 km visibility at 1 mm/h (Rasmussen et al. 1999 order of magnitude)
    assert snow.snowfall_extinction(3.0, 0.1) > snow.snowfall_extinction(1.0, 0.1) > 0
    assert snow.snowfall_extinction(0.0, 0.1) == 0.0


def test_default_unchanged(tmp_path):
    scene, m = _build(tmp_path)
    assert "snow" not in m and "snow:" not in scene and not (tmp_path / "snowfall.pbrt").exists()


def test_verges(tmp_path):
    scene, m = _build(tmp_path, "--snow", "verges")
    assert 'MakeNamedMaterial "grass" "string type" "diffuse" "spectrum reflectance" "spd/snow_verge.spd"' in scene
    assert 'NamedMaterial "snow:verge"' in scene and 'NamedMaterial "snow:road"' not in scene
    assert m["snow"]["surfaces"]["verge"]["albedo_450_650nm"] > 0.95 and "banks" in m["snow"]
    assert scene.index('MakeNamedMaterial "snow:verge"') > scene.index("WorldBegin")


def test_slush_sets_wet_and_leaves_tracks(tmp_path):
    scene, m = _build(tmp_path, "--snow", "slush", "--no-snowbanks")
    assert m["road"]["wet"]["level"] == "wet" and m["snow"]["notes"] and "banks" not in m["snow"]
    assert '"snow:road" "string type" "coateddiffuse"' in scene
    geo = {"paved": [(0.0, 10.0)], "lane_centres": [1.8, 5.4, 9.0]}
    rects = snow.road_rects(geo, "slush", np.random.default_rng(0), 0.0, 3.0)
    covered = sum((b - a) for a, b, za, zb in rects if za == 0.0)
    assert 10.0 - 6 * 1.3 * snow.TRACK_W < covered < 10.0 - 6 * 0.75 * snow.TRACK_W + 1e-9
    for c in (1.8 + 0.85, 5.4 - 0.85):  # track centres stay clear
        assert not any(a < c < b for a, b, za, zb in rects if za == 0.0)


def test_packed_and_flakes(tmp_path):
    scene, m = _build(tmp_path, "--snow", "packed", "--snowfall-mmh", "0.5", "--snowfall-range-m", "6", "--fov", "40")
    assert m["snow"]["road"]["cover"].startswith("whole") and 'Include "snowfall.pbrt"' in scene
    f = m["snow"]["snowfall"]
    assert f["flakes"] > 1000 and f["equivalent_visibility_m"] > 1000
    rows = (tmp_path / "snowfall.pbrt").read_text().splitlines()
    assert len(rows) == f["flakes"]
    z = np.array([float(r.split()[4]) for r in rows])
    assert z.min() >= 0.5 and z.max() <= 6.0


def test_rejects():
    with pytest.raises(SystemExit):
        hw.main([*BASE, "--snow", "packed", "--road-wetness", "wet"])
    with pytest.raises(SystemExit):
        hw.main([*BASE, "--snowfall-mmh", "-1"])
