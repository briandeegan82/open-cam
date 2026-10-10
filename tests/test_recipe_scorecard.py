"""Recipe scorecard: recipe discovery, optics grouping, world-space ROIs and a pbrt smoke run."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import run_recipe_scorecard as sc  # noqa: E402


def test_discovers_every_recipe():
    names = sc.discover_recipes()
    assert names == sorted(p.stem for p in (REPO / "config" / "camera_recipes").glob("*.yaml"))
    assert "default" in names
    with pytest.raises(SystemExit):
        sc.discover_recipes(["no_such_recipe"])


def test_groups_share_optics_and_cover_all_recipes():
    names = sc.discover_recipes()
    groups = sc.group_recipes(names)
    assert sorted(r for g in groups.values() for r in g["recipes"]) == names
    assert len(groups) < len(names)
    assert {g["optics"]["camera"] for g in groups.values()} <= {"perspective", "thinlens", "realistic"}
    hdr_family = {"default", "default_hdr_dcg", "default_hdr_split_pixel", "default_hdr_lofic", "default_hdr_3exp"}
    keys = {k for k, g in groups.items() if hdr_family & set(g["recipes"])}
    assert len(keys) == 1  # same optics -> one render reused by all five sensor variants


def test_realistic_fov_from_focal_length():
    o = {"camera": "realistic", "efl_mm": 50.0}
    fov = sc.vertical_fov(o, 960, 640)
    h = 35.0 * 640 / np.hypot(960, 640)
    assert fov == pytest.approx(2 * np.degrees(np.arctan(h / 100.0)))
    assert sc.vertical_fov({"camera": "perspective", "fov": 45.0}, 960, 640) == 45.0


def test_framing_keeps_target_plane_in_focus():
    args = sc.builder_args("diorama", {"camera": "thinlens", "fov": 20.0, "lens_radius": 0.04}, 360, 240, 4)
    a = dict(zip(args[::2], args[1::2], strict=False))
    d, z = float(a["--cam-dist"]), sc.SCENES["diorama"][3]
    assert float(a["--focus-distance"]) == pytest.approx(d - z, rel=1e-5)
    assert float(a["--thinlens-focal-distance"]) == pytest.approx(d - z, rel=1e-5)
    # narrower FOV -> camera moves back so the cards fill the same fraction of the frame
    assert (d - z) * np.tan(np.radians(10)) == pytest.approx((1.6 - z) * np.tan(np.radians(20)), rel=1e-4)


def test_world_roi_selects_pixels_inside_rectangle():
    yy, xx = np.mgrid[0:50, 0:80].astype(float)
    P = np.stack([xx / 10.0, (50 - yy) / 10.0, np.zeros_like(xx)], axis=-1)
    roi = sc.world_roi(P, [2.0, 1.0, 4.0, 3.0], 0.0, margin=0.0)
    x0, y0, x1, y1 = roi
    assert P[y0, x0, 0] >= 2.0 and P[y1 - 1, x1 - 1, 0] <= 4.0
    assert P[y1 - 1, x0, 1] >= 1.0 and P[y0, x0, 1] <= 3.0
    assert sc.world_roi(P, [2.0, 1.0, 4.0, 3.0], 5.0) is None


def test_guide_sites_pick_green_of_bayer():
    lay = sc.cm.resolve_layout({"layout": "GBRG"})
    e = np.arange(16.0).reshape(4, 4)
    sites, th, tw = sc.guide_sites(e, lay)
    assert (th, tw) == (2, 2)
    assert np.array_equal(sites, e[0::2, 0::2])


@pytest.mark.skipif(
    not sc.PBRT.is_file() and not os.environ.get("OPENCAM_REQUIRE_PBRT"), reason="pbrt binary not built"
)
def test_smoke_run_writes_report(tmp_path):
    rc = sc.main(
        ["--recipes", "default", "--scenes", "skin", "flare", "--xres", "120", "--yres", "80",
         "--pixelsamples", "4", "--out-dir", str(tmp_path)]
    )  # fmt: skip
    assert rc == 0
    data = json.loads((tmp_path / "scorecard.json").read_text())
    row = data["rows"][0]
    assert row["recipe"] == "default" and np.isfinite(row["skin_de00_mean"])
    assert "| default |" in (tmp_path / "scorecard.md").read_text()
    assert (tmp_path / "scorecard.csv").read_text().startswith("recipe,optics,cfa,hdr")


def test_binned_layout_collapses_quad_bayer_to_bayer():
    quad = sc.cm.resolve_layout({"layout": "QUAD_BAYER"})
    lay, b = sc.binned_layout(quad, (320, 480), (640, 960))
    assert b == 2 and lay.tile == (("R", "G"), ("G", "B"))
    assert sc.binned_layout(quad, (640, 960), (640, 960)) == (quad, 1)


def _const_mosaic(layout, values, shape=(24, 24)):
    th, tw = len(layout.tile), len(layout.tile[0])
    e = np.zeros(shape)
    for r in range(th):
        for c in range(tw):
            e[r::th, c::tw] = values[layout.tile[r][c]]
    return e


def test_white_signal_meters_on_brightest_channel():
    rgbw = sc.cm.resolve_layout({"layout": "RGBW"})
    e = _const_mosaic(rgbw, {"R": 1000.0, "G": 1200.0, "W": 3000.0, "B": 1100.0})
    assert sc.white_signal_e(e, rgbw, [4, 4, 20, 20]) == pytest.approx(3000.0, rel=1e-3)
    with pytest.raises(ValueError, match="empty"):
        sc.white_signal_e(e, rgbw, [10, 10, 10, 20])


def test_meter_signal_includes_card_highlights():
    bayer = sc.cm.resolve_layout({"layout": "RGGB"})
    e = _const_mosaic(bayer, {"R": 1000.0, "G": 1000.0, "B": 1000.0}, (48, 48))
    e[24:, 24:] *= 2.0
    assert sc.meter_signal_e(e, bayer, [2, 2, 20, 20]) == pytest.approx(1000.0, rel=1e-3)
    assert sc.meter_signal_e(e, bayer, [2, 2, 20, 20], [[26, 26, 46, 46]]) == pytest.approx(2000.0, rel=1e-3)


def test_despeckle_removes_fireflies_but_keeps_edge():
    crop = np.where(np.arange(40)[None, :] < 20, 0.1, 1.0) * np.ones((40, 1))
    hot = crop.copy()
    hot[5, 30] = hot[17, 25] = 8.0
    np.testing.assert_allclose(sc.despeckle(hot), crop)
    np.testing.assert_array_equal(sc.despeckle(crop), crop)
