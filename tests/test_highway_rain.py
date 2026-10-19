"""Falling rain option of the highway builder (tools/highway_rain.py)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import build_highway_scene as hw  # noqa: E402
import highway_rain as rain  # noqa: E402

BASE = ["--proxy-cars", "--allow-missing-assets", "--xres", "64", "--yres", "36", "--pixelsamples", "1"]


def _build(tmp_path: Path, *extra: str) -> tuple[str, dict]:
    hw.main([*BASE, "--out-dir", str(tmp_path), *extra])
    return (tmp_path / "highway.pbrt").read_text(), json.loads((tmp_path / "highway_manifest.json").read_text())


def test_physics():
    v = rain.terminal_velocity(np.array([1.0, 2.0, 4.0]))
    assert np.allclose(v, [4.03, 6.49, 8.83], atol=0.15)  # Gunn & Kinzer 1949 measurements
    assert rain.marshall_palmer(1.0) == (8000.0, 4.1)
    b2, b10 = rain.rain_extinction(2.0), rain.rain_extinction(10.0)
    assert 0 < b2 < b10 and 1000 < 2.996 / b10 < 4000 and rain.rain_extinction(0.0) == 0.0


def test_default_unchanged(tmp_path):
    scene, m = _build(tmp_path)
    assert "rain" not in m and "rain:water" not in scene and "shutterclose" not in scene
    assert not (tmp_path / "rain.pbrt").exists()


def test_rain_scene(tmp_path):
    scene, m = _build(tmp_path, "--rain-mmh", "5", "--rain-range-m", "4", "--fov", "40")
    r = m["rain"]
    assert m["road"]["wet"]["level"] == "wet" and len(r["notes"]) == 2
    assert f'"float shutterclose" [{rain.DEFAULT_EXPOSURE_S:g}]' in scene.replace("0.004]", "0.004]")
    assert 'Include "rain.pbrt"' in scene and scene.index("rain:water") > scene.index("WorldBegin")
    rows = (tmp_path / "rain.pbrt").read_text().splitlines()
    assert len(rows) == r["drops"] > 100
    tok = [row.split() for row in rows]
    z = np.array([float(t[4]) for t in tok])
    fall = np.array([float(t[9]) for t in tok])
    assert z.min() >= 0.5 and z.max() <= 4.0
    assert np.all(fall < 0) and np.all(-fall <= 9.65 * rain.DEFAULT_EXPOSURE_S)


def test_explicit_exposure_and_wetness_kept(tmp_path):
    _, m = _build(
        tmp_path, "--rain-mmh", "2", "--rain-range-m", "3", "--exposure-s", "0.01", "--road-wetness", "flooded"
    )
    assert m["rain"]["exposure_s"] == 0.01 and m["rain"]["notes"] == [] and m["road"]["wet"]["level"] == "flooded"


def test_rejects():
    with pytest.raises(SystemExit):
        hw.main([*BASE, "--rain-mmh", "-1"])
    with pytest.raises(SystemExit):
        hw.main([*BASE, "--rain-mmh", "200", "--rain-range-m", "60"])
