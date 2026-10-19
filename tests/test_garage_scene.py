"""Parking garage / surround-view fisheye builder (tools/build_garage_scene.py)."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import build_garage_scene as garage  # noqa: E402

BASE = ["--proxy-cars", "--allow-missing-assets", "--xres", "64", "--yres", "48", "--pixelsamples", "1"]


def _build(tmp_path: Path, *extra: str) -> tuple[str, dict]:
    garage.main([*BASE, "--out-dir", str(tmp_path), *extra])
    return (tmp_path / "garage.pbrt").read_text(), json.loads((tmp_path / "garage_manifest.json").read_text())


def test_default_fisheye_mixed_lighting(tmp_path):
    scene, m = _build(tmp_path)
    assert 'Camera "realistic"' in scene and "fisheye.10mm.dat" in scene
    assert m["camera"]["view"] == "front" and m["camera"]["film_diagonal_mm"] == 34.0
    kinds = {lm["kind"] for lm in m["lighting"]["luminaires"]}
    assert kinds == {"led", "fluorescent"}
    assert "lamp_fl_f2.spd" in scene and "lamp_led_batten.spd" in scene
    assert 0 < m["lighting"]["luminaires_on"] < len(m["lighting"]["luminaires"])
    assert "Sky" not in scene and "infinite" not in scene  # enclosed: no daylight
    assert 'Integrator "volpath"' in scene and "Scale -1" not in scene


def test_luminaire_photometry():
    flux, length, width = garage.LUMINAIRE["fluorescent"]
    lum = flux / (math.pi * length * width)
    assert 8_000.0 < lum < 20_000.0  # T8 tube luminance order
    # mean horizontal illuminance of the lit fittings over the deck, ~ EN 12464-1 parking level
    rows = len(garage.luminaire_rows("mixed"))
    n = len(np.arange(garage.Z_BACK + 3.0, garage.Z_END - 2.0, garage.LUM_PITCH))
    area = 2 * garage.WALL_X * (garage.Z_END - garage.Z_BACK)
    e = 0.5 * n * (garage.LUMINAIRE["led"][0] + 2 * flux) / area  # 0.5 = utilisation factor
    assert 20.0 < e < 150.0 and rows == 3


def test_bays_ground_truth(tmp_path):
    _, m = _build(tmp_path, "--occupancy", "1.0")
    assert len(m["bays"]) == 2 * garage.N_BAYS and all(b["occupied"] for b in m["bays"])
    assert len(m["cars"]) == len(m["bays"])
    for b in m["bays"]:
        xs = [c[0] for c in b["corners_xz"]]
        zs = [c[1] for c in b["corners_xz"]]
        assert max(xs) - min(xs) == pytest.approx(garage.BAY_D) and max(zs) - min(zs) == pytest.approx(garage.BAY_W)
        assert min(abs(x) for x in xs) == pytest.approx(garage.AISLE_W / 2)
    _, m0 = _build(tmp_path, "--occupancy", "0", "--lights-off-fraction", "0", "--lighting", "led")
    assert m0["cars"] == [] and m0["lighting"]["luminaires_on"] == len(m0["lighting"]["luminaires"])
    assert {lm["kind"] for lm in m0["lighting"]["luminaires"]} == {"led"}


@pytest.mark.parametrize("view", list(garage.VIEWS))
def test_views_outside_ego_body(view):
    eye, target = garage.camera_pose(view)
    inside = abs(eye[0]) < garage.EGO_W / 2 and abs(eye[2] - garage.EGO_Z) < garage.EGO_L / 2
    assert not inside and target[1] < eye[1]  # mounted on the body surface, pitched down


def test_pinhole_and_bad_fraction(tmp_path):
    scene, _ = _build(tmp_path, "--camera", "pinhole", "--fov", "100", "--view", "left")
    assert 'Camera "perspective" "float fov" [100]' in scene
    with pytest.raises(SystemExit):
        garage.parse_args(["--lights-off-fraction", "1.5"])
