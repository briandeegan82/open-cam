"""Flare metrics, flare scene builder and runner."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import build_flare_test_scene as fs
import numpy as np
import pytest
import run_flare_test as runner
from iqlab import flare
from scipy import ndimage
from synthetic_data import REPO

PBRT = Path(os.environ.get("OPENCAM_PBRT", REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"))


def _hole_field(glare: float, shape=(200, 300)) -> np.ndarray:
    img = np.full(shape, 100.0)
    for cy, cx in ((100, 150), (40, 60), (160, 240)):
        img[cy - 10 : cy + 10, cx - 10 : cx + 10] = 0.0
    return img + glare * 100.0


@pytest.mark.parametrize("glare", [0.0, 0.01, 0.05])
def test_black_hole_glare_recovers_uniform_veil(glare):
    holes = flare.black_hole_glare(_hole_field(glare))
    assert len(holes) == 3
    for h in holes:
        assert h.glare_percent == pytest.approx(100 * glare / (1 + glare), abs=1e-6)


def test_black_hole_glare_sees_spread():
    img = ndimage.gaussian_filter(_hole_field(0.0), 4.0)
    assert all(h.glare_percent > 0 for h in flare.black_hole_glare(img))


def test_point_source_flare_ghost_and_stray():
    img = np.zeros((128, 128))
    img[62:66, 30:34] = 1000.0
    img[90:96, 100:106] = 2.0  # ghost
    f = flare.point_source_flare(img, exclude_radius_px=8)
    assert (f.x, f.y) == pytest.approx((31.5, 63.5))
    assert f.ghost_peak_relative == pytest.approx(0.002)
    assert f.stray_fraction == pytest.approx(72.0 / (16000 + 72))
    assert f.radial_profile[0] > 0.5


def test_scene_builder(tmp_path):
    args = fs.build_parser().parse_args(["--out-dir", str(tmp_path), "--camera", "realistic", "--field-steps", "4"])
    m = fs.write_scenes(args)
    assert len(m["scenes"]) == 5
    vg = (tmp_path / "veiling_glare.pbrt").read_text()
    assert 'Camera "realistic"' in vg and vg.count("Material") == 1 + len(fs.HOLE_POSITIONS)
    assert json.loads((tmp_path / "flare_manifest.json").read_text())["scenes"][-1]["field_deg"] == 30.0


def test_ghost_sweep_coating_ordering(tmp_path):
    rows = runner.ghost_sweep("config/lenses/wide_22mm.dat", ["uncoated", "mgf2"], [0.0, 20.0], n_grid=24)
    by = {(r["coating"], r["field_deg"]): r["total_ghost_fraction"] for r in rows}
    for a in (0.0, 20.0):
        assert 0 < by[("mgf2", a)] < by[("uncoated", a)]


def test_cli_scores_npy(tmp_path):
    np.save(tmp_path / "vg.npy", _hole_field(0.02))
    pt = np.zeros((64, 64))
    pt[30:33, 30:33] = 10.0
    np.save(tmp_path / "pt.npy", pt)
    rc = runner.main(
        ["--veiling-exr", str(tmp_path / "vg.npy"), "--point-exr", str(tmp_path / "pt.npy"), "--out-dir", str(tmp_path)]
    )
    assert rc == 0
    rep = json.loads((tmp_path / "flare_results.json").read_text())
    assert len(rep["veiling_glare"][str(tmp_path / "vg.npy")]) == 3


@pytest.mark.skipif(not PBRT.is_file(), reason="pbrt binary not built")
def test_pinhole_render_has_negligible_veiling_glare(tmp_path):
    from exr_multispectral import linear_rgb_from_exr

    args = ["--out-dir", str(tmp_path), "--camera", "perspective", "--film", "rgb", "--targets", "veiling_glare"]
    fs.write_scenes(fs.build_parser().parse_args([*args, "--xres", "240", "--yres", "160", "--pixelsamples", "4"]))
    subprocess.run([str(PBRT), "--quiet", "veiling_glare.pbrt"], cwd=tmp_path, check=True, timeout=300)
    holes = flare.black_hole_glare(linear_rgb_from_exr(tmp_path / "veiling_glare_rgb.exr"))
    assert len(holes) == len(fs.HOLE_POSITIONS)
    assert max(h.glare_percent for h in holes) < 0.5
