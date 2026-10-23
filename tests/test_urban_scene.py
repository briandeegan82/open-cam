"""Urban intersection builder (tools/build_urban_scene.py): layout, emitters, reuse of the highway modules."""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import build_urban_scene as urban  # noqa: E402
from render_time_slices import scene_name  # noqa: E402

BASE = ["--proxy-cars", "--allow-missing-assets", "--xres", "64", "--yres", "36", "--pixelsamples", "1"]


def _build(tmp_path: Path, *extra: str) -> tuple[str, dict]:
    urban.main([*BASE, "--out-dir", str(tmp_path), *extra])
    return (tmp_path / "urban.pbrt").read_text(), json.loads((tmp_path / "urban_manifest.json").read_text())


def test_night_wet_default_has_signals_shops_streetlights(tmp_path):
    scene, m = _build(tmp_path)
    assert m["lighting"]["time_of_day"] == "night"
    assert m["road"]["wet"]["level"] == "wet"
    kinds = [e["kind"] for e in m["emitters"]]
    assert kinds.count("traffic_signal") == 5 and kinds.count("shop_led_sign") == 4
    ego = [e for e in m["emitters"] if e["kind"] == "traffic_signal" and e["faces"] == "-z"]
    assert {e["aspect"] for e in ego} == {"red"}
    shops = [e for e in m["emitters"] if e["kind"] == "shop_led_sign"]
    assert all(e["pwm"]["frequency_hz"] == 100.0 for e in shops)
    assert all(e["pwm"] is None for e in m["emitters"] if e["kind"] == "traffic_signal")  # DC by default
    for e in m["emitters"]:
        assert f'Include "{e["include"]}"' in scene
        on = (tmp_path / e["states"]["on"]).read_text()
        off = (tmp_path / e["states"]["off"]).read_text()
        assert "AreaLightSource" in on and "AreaLightSource" not in off
    assert m["lighting"]["streetlights"]["columns"] > 4
    assert m["facades"]["windows_lit"] > 0
    assert "Scale -1" not in scene and 'Integrator "volpath"' in scene
    assert scene_name(tmp_path) == "urban"


def test_signal_luminance_matches_intensity(tmp_path):
    _, m = _build(tmp_path, "--signal-state", "green", "--signal-pwm-hz", "90", "--signal-duty", "0.25")
    sig = [e for e in m["emitters"] if e["kind"] == "traffic_signal" and e["faces"] == "-z"]
    assert {e["aspect"] for e in sig} == {"green"}
    area = 3.141592653589793 * (urban.ASPECT_D / 2) ** 2
    for e in sig:  # time-averaged luminance x lens area = nominal intensity
        assert e["led_peak_luminance_cd_m2"] * 0.25 * area == pytest.approx(urban.SIGNAL_CD["green"])
        assert e["pwm"]["frequency_hz"] == 90.0


def test_day_dry_no_window_emitters(tmp_path):
    scene, m = _build(tmp_path, "--time-of-day", "day", "--road-wetness", "dry")
    assert m["facades"]["windows_lit"] == 0 and m["road"]["wet"] is None
    assert 'LightSource "distant"' in scene and "lamp_window" not in scene.split("WorldBegin")[1].split("window_lit")[0]


def test_vru_on_near_zebra_and_signs_beyond_junction(tmp_path):
    _, m = _build(tmp_path, "--vru", "pedestrian", "--sign-set", "stvo")
    zebra = [r for r in urban.zebra_rects() if r[3] < urban.Z_INT]
    z0, z1 = min(r[2] for r in zebra), max(r[3] for r in zebra)
    assert z0 <= m["vru"]["distance_m"] <= z1
    for s in m.get("signs", {}).get("signs", []):
        assert s["distance_m"] > urban.Z_INT + urban.HALF


def test_rejects_highway_only_options(tmp_path):
    with pytest.raises(SystemExit):
        urban.parse_args(["--puddles"])
    with pytest.raises(SystemExit):
        urban.parse_args(["--lit-windows", "2"])


def test_regularize_is_opt_in(tmp_path):
    for d, a in (("plain", ()), ("reg", ("--regularize",))):
        (tmp_path / d).mkdir()
        scene = _build(tmp_path / d, *a)[0]
        assert ('"bool regularize" true' in scene) == bool(a)
