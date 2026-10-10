"""Skin-tone spectra, chart builder and scorer."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import build_skin_tone_chart as sc
import numpy as np
import pytest
import run_skin_tone_test as runner
from colour_science import WHITE_D65, load_illuminant, xyz_to_srgb_linear
from iqlab import skin
from synthetic_data import REPO

PBRT = Path(os.environ.get("OPENCAM_PBRT", REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"))
WL = np.arange(380.0, 781.0, 5.0)


def _labs(ill="D65"):
    _, spd = load_illuminant(REPO, ill, WL)
    return skin.reflectance_lab(WL, np.stack([s["reflectance"] for s in skin.skin_tone_set(REPO, WL)]), spd)


def test_melanin_series_spans_ita_and_darkens_blue_most():
    lab = _labs()[: len(skin.DEFAULT_TAU)]
    ita = skin.ita_deg(lab)
    assert np.all(np.diff(ita) < 0) and ita[0] > 55 and ita[-1] < -30
    assert np.all(np.diff(lab[:, 0]) < 0)
    base = skin.skin_tone_set(REPO, WL)[1]["reflectance"]
    r = skin.melanin_reflectance(WL, base, 1.0) / base
    assert r[WL == 450][0] < r[WL == 650][0]


def test_colorchecker_skin_reference_values():
    lab = _labs()
    # X-Rite published D50 values are ~(65.7, 18.1, 17.8) light / (37.5, 14.4, 14.9) dark; D65 is close
    assert lab[-2] == pytest.approx([65.7, 18.1, 17.8], abs=4)
    assert lab[-1] == pytest.approx([37.5, 14.4, 14.9], abs=4)


def test_skin_metrics_known_shift():
    ref = np.array([[60.0, 15.0, 20.0]])
    m = skin.skin_metrics(np.array([[60.0, 20.0, 15.0]]), ref)
    assert m["delta_C"][0] == pytest.approx(0.0, abs=1e-9)
    assert m["delta_hue_deg"][0] == pytest.approx(np.degrees(np.arctan2(15, 20) - np.arctan2(20, 15)))
    assert skin.skin_metrics(ref, ref)["delta_e00"][0] == pytest.approx(0.0, abs=1e-9)
    assert skin.ita_deg(np.array([50.0, 0.0, 10.0])) == pytest.approx(0.0)


def _synthetic_render(chart, ill, gain=3.0, tint=(1.0, 1.0, 1.0)):
    img = np.zeros((chart["yres"], chart["xres"], 3))

    for p in chart["patches"]:
        x0, y0, x1, y1 = p["roi_xyxy"]
        img[y0:y1, x0:x1] = (
            xyz_to_srgb_linear(skin.lab_to_xyz(np.array(p["reference_lab"][ill]), WHITE_D65)) * gain * np.array(tint)
        )
    return img


def test_builder_and_scorer_roundtrip(tmp_path):
    chart = sc.write_chart(sc.build_parser().parse_args(["--out-dir", str(tmp_path), "--illuminants", "D65", "A"]))
    assert len(chart["patches"]) == 12 and (tmp_path / "skin_tones_A.pbrt").is_file()
    assert (tmp_path / "spd" / "illuminant_A.spd").is_file()
    res = runner.score(chart, _synthetic_render(chart, "A"), "A")
    assert res["max_delta_e00"] < 1e-6  # exposure gain drops out via the white patch
    bad = runner.score(chart, _synthetic_render(chart, "A", tint=(1.1, 1.0, 0.9)), "A")
    assert bad["mean_delta_e00"] > 1.0 and all(r["delta_hue_deg"] != 0 for r in bad["patches"])


@pytest.mark.skipif(not PBRT.is_file(), reason="pbrt binary not built")
# pbrt RGB film vs 5 nm CIE reference: <1 dE00 under D65; under A chroma reads up to ~2.2 dE00 high on dark tones
@pytest.mark.parametrize(("ill", "tol"), [("D65", 1.5), ("A", 3.0)])
def test_pbrt_render_matches_spectral_reference(tmp_path, ill, tol):
    from exr_multispectral import linear_rgb_from_exr

    args = ["--out-dir", str(tmp_path), "--illuminants", ill, "--film", "rgb", "--xres", "240", "--yres", "160"]
    chart = sc.write_chart(sc.build_parser().parse_args([*args, "--pixelsamples", "16"]))
    subprocess.run([str(PBRT), "--quiet", chart["scenes"][ill]["scene"]], cwd=tmp_path, check=True, timeout=300)
    res = runner.score(chart, linear_rgb_from_exr(tmp_path / chart["scenes"][ill]["film_output"]), ill)
    assert res["max_delta_e00"] < tol
