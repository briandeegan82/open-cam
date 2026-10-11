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
    assert txt.count('AreaLightSource "diffuse"') == lum["count"] == lum["count_threshold"] + lum["count_interior"]
    th = m["photometry"]["threshold_length_m"]
    # two rows; threshold spacing over the threshold zone, interior spacing for the rest
    assert lum["count_threshold"] == 2 * math.ceil((th - 1.25) / lum["spacing_m"]["threshold"])
    assert lum["luminance_cd_m2"]["threshold"] > lum["luminance_cd_m2"]["interior"] > 0


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
    m = built["outside_in"]["meta"]
    lamps = bts.luminaire_positions(
        m["tunnel"]["portal_entrance_z"], m["tunnel"]["portal_exit_z"], m["photometry"]["threshold_length_m"], 3.0, 9.0
    )
    zc = m["tunnel"]["portal_entrance_z"] + 0.5 * m["photometry"]["threshold_length_m"]
    pts = np.array([(x, 0.0, zc + dz) for x in np.linspace(0.5, bts.ROAD_W - 0.5, 7) for dz in np.linspace(-6, 6, 13)])
    e = bts.direct_road_illuminance(pts, lamps, "threshold").mean() * m["luminaires"]["luminance_cd_m2"]["threshold"]
    rho = m["photometry"]["road_luminous_reflectance"]["lamp"]
    assert rho * e / math.pi == pytest.approx(m["photometry"]["threshold_road_luminance_cd_m2"], rel=1e-9)


def test_rois_inside_frame_and_nondegenerate(built):
    need = {"outside_in": {"exterior_road", "threshold_road", "threshold_wall", "target", "headwall"},
            "inside_out": {"interior_road", "interior_wall", "exit_opening", "exit_road", "target"}}  # fmt: skip
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
        subprocess.run([str(PBRT), "--quiet", b["scene"].name], cwd=tmp_path, check=True)
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
