"""Rural night road builder (tools/build_rural_scene.py) and the opt-in high beam (highway_night)."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import build_rural_scene as rural  # noqa: E402
import highway_night as night  # noqa: E402

BASE = ["--proxy-cars", "--allow-missing-assets", "--xres", "64", "--yres", "36", "--pixelsamples", "1"]


def _build(tmp_path: Path, *extra: str) -> tuple[str, dict]:
    rural.main([*BASE, "--out-dir", str(tmp_path), *extra])
    return (tmp_path / "rural.pbrt").read_text(), json.loads((tmp_path / "rural_manifest.json").read_text())


def test_high_beam_meets_r112_and_keeps_low_beam():
    hv = night.high_beam_cd(np.array([0.0]), np.array([0.0]))[0]
    assert hv >= 30_000.0  # >= 48 lux at 25 m on H-V
    assert 2 * hv <= 430_000.0  # R48 cap on the sum of driving beams
    n = 256
    low, high = night.headlamp_map(n, "led"), night.headlamp_map(n, "led", "high")
    assert np.all(high >= low) and high.max() > 2 * low.max()


def test_default_scene_unlit_glare_posts_deer(tmp_path):
    scene, m = _build(tmp_path)
    assert m["lighting"]["street_lighting"] == "none" and "streetlight" not in scene
    assert "highbeam_" in scene and "lowbeam_" in scene  # oncoming high, ego low
    assert m["cars"][0]["beam"] == "high" and m["lighting"]["ego_beam"] == "low"
    assert scene.index("highbeam_") > scene.index("lowbeam_")  # ego lamps first, then the oncoming car
    assert m["delineators"]["count"] == len(rural.post_positions()) > 20
    assert 'MakeNamedMaterial "sheet_white" "string type" "retroreflective"' in scene
    assert 'MakeNamedMaterial "deer_eye" "string type" "retroreflective"' in scene
    d = m["deer"]
    assert d["kind"] == "roe" and d["bbox_world"]["min"][2] < 55.0 < d["bbox_world"]["max"][2]
    for e in d["eyes_world"]:  # head forward = -x when crossing right-to-left
        assert e[0] < d["position_world"][0]
        assert d["bbox_world"]["min"][0] <= e[0] <= d["bbox_world"]["max"][0]
    assert "Scale -1" not in scene and 'Integrator "volpath"' in scene


def test_options(tmp_path):
    scene, m = _build(tmp_path, "--oncoming", "none", "--deer", "none", "--ego-beam", "high", "--road-wetness", "wet")
    assert m["cars"] == [] and m["deer"] is None and "oncoming" not in m["lighting"]
    assert m["road"]["wet"]["level"] == "wet"
    assert m["lighting"]["ego"][0]["flux_lm_per_lamp"] > 1000.0 and "highbeam_" in scene and "lowbeam_" not in scene


def test_deer_heading_away(tmp_path):
    lines, meta = rural.deer_lines("red", 1.0, 40.0, -90.0)
    for e in meta["eyes_world"]:
        assert e[2] > 40.0  # facing away along +z
    assert meta["bbox_world"]["max"][1] == pytest.approx(0.62 * 1.2 + 0.66 * 1.2 / 0.7)
    assert any(ln.startswith("Rotate -90") for ln in lines)
    assert not math.isnan(meta["bbox_world"]["min"][0])


def test_rejects_day_and_streetlights():
    with pytest.raises(SystemExit):
        rural.parse_args(["--time-of-day", "day"])
    with pytest.raises(SystemExit):
        rural.parse_args(["--streetlights", "hps"])
