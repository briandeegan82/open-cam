"""Tunnel HDR scene: builder geometry / photometry / ROIs, plus a pbrt-gated radiance-ratio check."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import build_tunnel_scene as bts  # noqa: E402

PBRT = REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"
PBRT_TIMEOUT_S = int(os.environ.get("OPENCAM_PBRT_TIMEOUT", "600"))


def _build(tmp_path: Path, *extra: str) -> dict:
    return bts.build(bts.parse_args(["--out-dir", str(tmp_path), "--pixelsamples", "4", *extra]))


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict:
    return _build(tmp_path_factory.mktemp("tunnel"))


def test_both_views_written(built):
    assert set(built) == {"outside_in", "inside_out"}
    for b in built.values():
        txt = b["scene"].read_text()
        assert txt.count('LightSource "distant"') == 1 and txt.count('LightSource "infinite"') == 1
        assert "# Tunnel lining" in txt and txt.count("# Headwall at z") == 2 and "# Hillside" in txt
        assert "# Target car" in txt
        assert b["manifest"].is_file()


def test_luminaire_count(built):
    m = built["outside_in"]["meta"]
    lum = m["luminaires"]
    txt = built["outside_in"]["scene"].read_text()
    assert txt.count('AreaLightSource "diffuse"') == lum["count"] == sum(lum["count_by_zone"].values())
    th = m["photometry"]["threshold_length_m"]
    # two rows; threshold spacing over both threshold halves
    n_th = lum["count_by_zone"]["threshold_1"] + lum["count_by_zone"]["threshold_2"]
    assert n_th == 2 * math.ceil((th - 1.25) / lum["spacing_m"]["threshold_1"])
    assert all(n > 0 for n in lum["count_by_zone"].values())
    assert min(lum["luminance_cd_m2"].values()) > 0


def test_stepped_zone_profile(built):
    ph = built["outside_in"]["meta"]["photometry"]
    zones, l_th, l_in = ph["zones"], ph["threshold_road_luminance_cd_m2"], ph["interior_road_luminance_cd_m2"]
    assert ph["zone_profile"] == "stepped"
    # default L_th = 75: 0.05 / 0.02 L_th fall below L_in = 6, so one transition step remains
    assert [z["zone"] for z in zones] == ["threshold_1", "threshold_2", "transition_1", "interior", "exit_1", "exit_2"]
    lum = [z["road_luminance_cd_m2"] for z in zones]
    assert lum == pytest.approx([l_th, 0.5 * l_th, 0.15 * l_th, l_in, 3 * l_in, 5 * l_in])
    tun = built["outside_in"]["meta"]["tunnel"]
    assert zones[0]["z_start_m"] == tun["portal_entrance_z"] and zones[-1]["z_end_m"] == tun["portal_exit_z"]
    for a, b in zip(zones, zones[1:]):
        assert a["z_end_m"] == b["z_start_m"]
    assert built["inside_out"]["meta"]["target"]["zone"].startswith("exit_2")


def test_zone_profile_transition_steps_and_two_step():
    sp = {"threshold": 3.0, "transition": 4.5, "interior": 9.0, "exit": 4.5}
    zones = bts.zone_profile(40.0, 840.0, 250.0, 6.0, 80.0, (72.0, 89.0, 133.0), (30.0, 30.0), sp)
    # 0.15 / 0.05 L_th = 37.5 / 12.5 > L_in; 0.02 L_th = 5 <= L_in becomes interior
    assert [z["zone"] for z in zones][2:5] == ["transition_1", "transition_2", "interior"]
    assert zones[3]["road_luminance"] == pytest.approx(12.5) and zones[3]["spacing"] == 4.5
    two = bts.zone_profile(40.0, 440.0, 75.0, 6.0, 80.0, (72.0,), (30.0,), sp, "two-step")
    assert [(z["zone"], z["z0"], z["z1"]) for z in two] == [("threshold", 40.0, 120.0), ("interior", 120.0, 440.0)]
    with pytest.raises(SystemExit):
        bts.zone_profile(40.0, 240.0, 75.0, 6.0, 80.0, (72.0, 89.0), (30.0, 30.0), sp)


def test_two_step_profile_keeps_legacy_scene(tmp_path):
    b = _build(tmp_path, "--zone-profile", "two-step")
    m = b["inside_out"]["meta"]
    assert [z["zone"] for z in m["photometry"]["zones"]] == ["threshold", "interior"]
    assert m["dynamic_range"]["dark_roi"] == "interior_road"
    assert any("no transition or exit-zone lighting" in a for a in m["approximations"])


def test_headwall_hole_matches_tube_profile():
    p, t = bts.headwall(50.0)
    assert np.allclose(p[:, 2], 50.0) and p[:, 1].min() >= 0.0 and p[:, 1].max() <= bts.HEADWALL_H + 1e-9
    inner = p[0::2]  # polar strip: hole boundary on even vertices
    assert np.isclose(inner[:, 0].min(), bts.WALL_X[0]) and np.isclose(inner[:, 0].max(), bts.WALL_X[1])
    assert np.isclose(inner[:, 1].max(), bts.CROWN_H, atol=0.02)
    # total headwall area = rectangle - tube cross-section
    tri = p[t]
    u, v = tri[:, 1, :2] - tri[:, 0, :2], tri[:, 2, :2] - tri[:, 0, :2]
    area = 0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]).sum()
    prof = np.array(bts.tube_profile())
    hole = 0.5 * abs(np.dot(prof[:, 0], np.roll(prof[:, 1], -1)) - np.dot(prof[:, 1], np.roll(prof[:, 0], -1)))
    assert area == pytest.approx(2 * bts.HEADWALL_HALF_W * bts.HEADWALL_H - hole, rel=2e-3)


def test_photometry_and_dynamic_range(built):
    for b in built.values():
        m = b["meta"]
        ph, dr = m["photometry"], m["dynamic_range"]
        assert ph["threshold_road_luminance_cd_m2"] == pytest.approx(ph["k"] * ph["l20_cd_m2"])
        lux = m["lighting"]["reference_illuminance_lux"]
        assert 1e4 < lux < 1.2e5
        assert ph["exterior_road_luminance_cd_m2"] == pytest.approx(
            ph["road_luminous_reflectance"]["daylight"] * lux / math.pi
        )
        rois = m["rois"]
        ratio = rois[dr["bright_roi"]]["predicted_luminance_cd_m2"] / rois[dr["dark_roi"]]["predicted_luminance_cd_m2"]
        assert dr["db"] == pytest.approx(20 * math.log10(ratio))
    assert built["inside_out"]["meta"]["dynamic_range"]["db"] > built["outside_in"]["meta"]["dynamic_range"]["db"] > 20


def test_lamp_solve_hits_design_luminance(built):
    """Direct road luminance at every zone centre, summed over all zones' lamps, equals the design value."""
    m = built["outside_in"]["meta"]
    zones = [
        {"zone": z["zone"], "z0": z["z_start_m"], "z1": z["z_end_m"], "spacing": z["luminaire_spacing_m"]}
        for z in m["photometry"]["zones"]
    ]
    lamps = bts.luminaire_positions(zones)
    assert len(lamps) == m["luminaires"]["count"]
    lamp_l = m["luminaires"]["luminance_cd_m2"]
    rho = m["photometry"]["road_luminous_reflectance"]["lamp"]
    for zn, design in zip(zones, m["photometry"]["zones"]):
        pts = bts.zone_road_points(zn)
        e = sum(bts.direct_road_illuminance(pts, lamps, j).mean() * lamp_l[j] for j in lamp_l)
        assert rho * e / math.pi == pytest.approx(design["road_luminance_cd_m2"], rel=1e-9), zn["zone"]


def test_rois_inside_frame_and_nondegenerate(built):
    need = {"outside_in": {"exterior_road", "threshold_road", "threshold_wall", "target", "headwall"},
            "inside_out": {"exit_zone_road", "interior_wall", "exit_opening", "exit_road", "target"}}  # fmt: skip
    for view, b in built.items():
        m = b["meta"]
        W, H = m["film"]["xres"], m["film"]["yres"]
        have = {k for k, r in m["rois"].items() if r["roi_xyxy"]}
        assert need[view] <= have, view
        for r in m["rois"].values():
            if r["roi_xyxy"]:
                x0, y0, x1, y1 = r["roi_xyxy"]
                assert 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H
        assert m["target"]["roi"] == "target"


def test_inside_out_dark_roi_follows_zone(built, tmp_path):
    """Default camera (60 m) looks over the exit zone; from 100 m out it sees interior road too."""
    io = built["inside_out"]["meta"]
    assert io["dynamic_range"]["dark_roi"] == "exit_zone_road"
    assert io["rois"]["exit_zone_road"]["predicted_luminance_cd_m2"] == pytest.approx(
        3 * io["photometry"]["interior_road_luminance_cd_m2"]
    )
    far = _build(tmp_path, "--view", "inside_out", "--exit-distance", "100")["inside_out"]["meta"]
    assert far["dynamic_range"]["dark_roi"] == "interior_road" and "exit_zone_road" in far["rois"]
    assert far["dynamic_range"]["db"] == pytest.approx(io["dynamic_range"]["db"] + 20 * math.log10(3))
    with pytest.raises(SystemExit):
        _build(tmp_path, "--view", "inside_out", "--exit-distance", "85")


def test_projector_mirror_convention():
    """World +x is image-right for a camera looking down +z (verified against pbrt renders)."""
    pr = bts.Projector(np.zeros(3), np.array([0.0, 0.0, 1.0]), 40.0, 300, 200)
    assert pr(np.array([0.0, 0.0, 10.0]))[0] == pytest.approx([150.0, 100.0])
    q = pr(np.array([1.0, 1.0, 10.0]))[0]
    assert q[0] > 150 and q[1] < 100


def test_realistic_camera_manifest(tmp_path):
    b = _build(tmp_path, "--view", "outside_in", "--camera", "realistic", "--no-lens-dispersion")["outside_in"]
    assert b["meta"]["camera"]["type"] == "realistic"
    assert 'Camera "realistic"' in b["scene"].read_text()


@pytest.mark.skipif(not PBRT.is_file() and os.environ.get("OPENCAM_REQUIRE_PBRT") != "1", reason="pbrt not built")
def test_pbrt_radiance_ratio_matches_manifest(tmp_path):
    """Render small RGB images; the measured bright/dark luminance ratio matches the manifest."""
    if not PBRT.is_file():
        pytest.fail("OPENCAM_REQUIRE_PBRT=1 but pbrt is not built")
    from exr_multispectral import linear_rgb_from_exr

    built = _build(tmp_path, "--film", "rgb", "--xres", "480", "--yres", "320", "--pixelsamples", "64")
    (tmp_path / "out").mkdir(exist_ok=True)
    for view, b in built.items():
        subprocess.run([str(PBRT), "--quiet", b["scene"].name], cwd=tmp_path, check=True, timeout=PBRT_TIMEOUT_S)
        m = b["meta"]
        Y = linear_rgb_from_exr(tmp_path / m["film_output"]) @ np.array([0.2126, 0.7152, 0.0722])
        lum = {}
        for k in (m["dynamic_range"]["bright_roi"], m["dynamic_range"]["dark_roi"]):
            x0, y0, x1, y1 = m["rois"][k]["roi_xyxy"]
            lum[k] = float(Y[y0:y1, x0:x1].mean())
            # absolute: EXR luminance is cd/m^2. The prediction is direct light only; lining
            # interreflections and daylight entering the portal add up to ~70 % (docs/TUNNEL_SCENES.md).
            pred = m["rois"][k]["predicted_luminance_cd_m2"]
            assert pred * 0.9 < lum[k] < pred * 1.8, (view, k, lum[k], pred)
        db = 20 * math.log10(lum[m["dynamic_range"]["bright_roi"]] / lum[m["dynamic_range"]["dark_roi"]])
        assert db == pytest.approx(m["dynamic_range"]["db"], abs=4.5), view
    shutil.rmtree(tmp_path / "out")


@pytest.fixture(scope="module")
def built_rtable(tmp_path_factory) -> dict:
    return _build(tmp_path_factory.mktemp("tunnel_rt"), "--road-reflection", "rtable")


def test_rtable_scene_and_manifest(built_rtable):
    for b in built_rtable.values():
        text = b["scene"].read_text()
        assert '"string type" "roadrtable"' in text and "PLACEHOLDER, not CIE data" in text
        rr = b["meta"]["photometry"]["road_reflection"]
        assert rr["model"] == "rtable" and rr["table"]["placeholder"] is True
        assert rr["tint_luminous_factor"]["lamp"] == pytest.approx(1.0, abs=0.02)
        assert any("PLACEHOLDER" in a for a in b["meta"]["approximations"])


def test_rtable_lamp_solve_hits_design_luminance(built_rtable):
    """r-table road luminance (1 deg observer along +z) at every zone centre equals the design value."""
    import road_rtable as rt

    m = built_rtable["outside_in"]["meta"]
    rr = m["photometry"]["road_reflection"]
    table = rt.load(rr["table_file"])
    zones = [
        {"zone": z["zone"], "z0": z["z_start_m"], "z1": z["z_end_m"], "spacing": z["luminaire_spacing_m"]}
        for z in m["photometry"]["zones"]
    ]
    lamps = bts.luminaire_positions(zones)
    lamp_l = m["luminaires"]["luminance_cd_m2"]
    k = rr["tint_luminous_factor"]["lamp"]
    for zn, design in zip(zones, m["photometry"]["zones"]):
        pts = bts.zone_road_points(zn)
        lum = k * sum(bts.direct_road_luminance(pts, lamps, j, table).mean() * lamp_l[j] for j in lamp_l)
        assert lum == pytest.approx(design["road_luminance_cd_m2"], rel=1e-9), zn["zone"]


def test_direct_road_luminance_lambertian_table_matches_illuminance():
    """A constant-q table (q = rho / pi) reproduces the Lambertian rho E / pi."""
    import road_rtable as rt

    t = np.linspace(0.0, 12.0, 4801)  # dense: r = q cos^3 is interpolated linearly in tan(gamma)
    q = 0.08 / math.pi
    table = rt.RTable("lambert", t, [0.0, 180.0], np.repeat((q / (1 + t * t) ** 1.5)[:, None], 2, axis=1))
    zones = bts.zone_profile(
        80.0,
        420.0,
        120.0,
        6.0,
        80.0,
        (72.0,),
        (60.0,),
        {k: 9.0 for k in ("threshold", "transition", "interior", "exit")},
        "stepped",
    )
    lamps = bts.luminaire_positions(zones)
    pts = bts.zone_road_points(zones[0])
    e = bts.direct_road_illuminance(pts, lamps, zones[0]["zone"])
    lum = bts.direct_road_luminance(pts, lamps, zones[0]["zone"], table)
    # lamps beyond tan(gamma) = 12 are clamped to the last row (a few 1e-3 of the far-field light)
    np.testing.assert_allclose(lum, 0.08 * e / math.pi, rtol=5e-3)


@pytest.mark.skipif(not PBRT.is_file() and os.environ.get("OPENCAM_REQUIRE_PBRT") != "1", reason="pbrt not built")
def test_pbrt_rtable_tunnel_renders(tmp_path):
    """r-table road (patched pbrt roadrtable material): rendered ROI luminances near the 1 deg design values."""
    if not PBRT.is_file():
        pytest.fail("OPENCAM_REQUIRE_PBRT=1 but pbrt is not built")
    from exr_multispectral import linear_rgb_from_exr

    built = _build(
        tmp_path,
        "--road-reflection",
        "rtable",
        "--film",
        "rgb",
        "--xres",
        "480",
        "--yres",
        "320",
        "--pixelsamples",
        "64",
    )
    (tmp_path / "out").mkdir(exist_ok=True)
    for view, b in built.items():
        subprocess.run([str(PBRT), "--quiet", b["scene"].name], cwd=tmp_path, check=True, timeout=PBRT_TIMEOUT_S)
        m = b["meta"]
        Y = linear_rgb_from_exr(tmp_path / m["film_output"]) @ np.array([0.2126, 0.7152, 0.0722])
        for k in (m["dynamic_range"]["bright_roi"], m["dynamic_range"]["dark_roi"]):
            x0, y0, x1, y1 = m["rois"][k]["roi_xyxy"]
            lum, pred = float(Y[y0:y1, x0:x1].mean()), m["rois"][k]["predicted_luminance_cd_m2"]
            # Sunlit exterior road: ~1.0x (outside-in) / ~0.85x (seen from inside). Lamp-lit roads add
            # lining interreflections and daylight; the glossy placeholder table reflects the daylight
            # entering at the exit portal strongly toward the camera (inside-out exit zone ~3.8x).
            lo, hi = (0.7, 1.5) if k == m["dynamic_range"]["bright_roi"] else (0.9, 5.0)
            assert pred * lo < lum < pred * hi, (view, k, lum, pred)
    shutil.rmtree(tmp_path / "out")
