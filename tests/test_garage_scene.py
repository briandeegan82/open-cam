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
    lamps = scene + "".join(f.read_text() for f in (tmp_path / "emitters").glob("*.pbrt"))
    assert "lamp_fl_f2.spd" in lamps and "lamp_led_batten.spd" in lamps
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


def test_mains_ripple_waveform():
    from highway_incar import PWM, MainsRipple, waveform_from_dict

    r = MainsRipple(100.0, 0.35, phase_s=1e-3)
    lv = [r.level(t) for t in np.linspace(0, 0.01, 2001)]
    assert max(lv) == pytest.approx(1.0) and (max(lv) - min(lv)) / (max(lv) + min(lv)) == pytest.approx(0.35, abs=1e-4)
    assert r.on_fraction(0.003, 0.013) == pytest.approx(r.duty) and r.duty == pytest.approx(np.mean(lv), abs=1e-3)
    steps = {k: v for k, v in r.state_levels().items() if k.startswith("s")}
    assert len(steps) == 16 and np.mean(list(steps.values())) == pytest.approx(r.duty)
    assert r.state_levels()[r.state(1e-3)] == pytest.approx(min(steps.values()))  # trough at phase_s
    assert len(r.edges(0.0, 0.01)) == 16
    assert waveform_from_dict(r.as_dict()) == r and waveform_from_dict(PWM(90, 0.2).as_dict()) == PWM(90, 0.2)


def test_mains_flicker_emitters(tmp_path):
    scene, m = _build(tmp_path, "--exposure-s", "0.004")
    em = m["emitters"]
    assert [e["mains_phase"] for e in em] == ["L1", "L2", "L3"] and len({e["pwm"]["phase_s"] for e in em}) == 3
    for e in em:
        assert e["pwm"]["type"] == "mains" and e["pwm"]["frequency_hz"] == 100.0
        assert f'Include "{e["include"]}"' in scene and {"on", "off", "s0", "s15"} <= set(e["states"])
        assert all((tmp_path / f).is_file() for f in e["states"].values())
    pf = {e["kind"]: e["pwm"]["percent_flicker"] for e in em}
    assert pf == {"luminaire_led": 0.10, "luminaire_fluorescent": 0.35}
    nominal = {lm["kind"]: lm["luminance_cd_m2"] for lm in m["lighting"]["luminaires"]}
    led = next(e for e in em if e["kind"] == "luminaire_led")
    assert led["peak_luminance_cd_m2"] * (1 - 0.2 / 1.1 / 2) == pytest.approx(nominal["led"])
    assert m["exposure"]["integration_time_s"] == 0.004 and m["lighting"]["mains_flicker"]["mains_hz"] == 50.0
    off_scene, m0 = _build(tmp_path / "off", "--mains-flicker", "off")
    assert m0["emitters"] == [] and "lamp_fl_f2.spd" in off_scene and m0["lighting"]["mains_flicker"] is None


def test_mains_flicker_time_slices(tmp_path):
    import render_time_slices as rts

    scene, m = _build(tmp_path, "--exposure-s", "0.002", "--mains-hz", "60")
    assert rts.scene_name(tmp_path) == "garage" and rts.scene_is_static(m, scene)
    ids, frac = rts.linear_plan(m["exposure"], m["emitters"], 48)
    assert frac.shape == (3, 48) and 0.0 < frac.min() < frac.max() <= 1.0
    assert not np.allclose(frac[0], frac[1])  # different phases ripple out of step
    _, m1 = _build(tmp_path / "one", "--exposure-s", "0.002", "--mains-hz", "60", "--mains-phases", "1")
    assert {e["mains_phase"] for e in m1["emitters"]} == {"L1"} and len(
        {e["pwm"]["phase_s"] for e in m1["emitters"]}
    ) == 1
    sl = rts.plan_slices(m["exposure"], m["emitters"], 48, bands=4, spp=16)
    assert {st for s in sl for st in s.states.values()} <= {f"s{k}" for k in range(16)}
    for y0 in {s.y0 for s in sl}:
        assert sum(s.weight for s in sl if s.y0 == y0) == pytest.approx(1.0)
    text = rts.slice_scene_text(scene, sl[0], m["emitters"])
    assert m["emitters"][0]["states"][sl[0].states[ids[0]]] in text
