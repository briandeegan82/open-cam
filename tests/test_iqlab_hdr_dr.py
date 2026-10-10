"""HDR dynamic-range chart builder and runner (tools/build_hdr_dr_chart.py, tools/run_hdr_dr_test.py)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import build_hdr_dr_chart as chart_mod
import numpy as np
import pytest
import run_hdr_dr_test as runner
from synthetic_data import REPO

PBRT = Path(os.environ.get("OPENCAM_PBRT", REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"))


def test_levels_span_and_contrast():
    lv = chart_mod.chart_levels(21, 120.0)
    assert lv[-1] == pytest.approx(1.0)
    assert 20 * np.log10(lv[-1] / lv[0]) == pytest.approx(120.0)
    k = chart_mod.contrast_ratio(0.2)
    assert (k - 1) / (k + 1) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        chart_mod.contrast_ratio(1.0)


def test_chart_files_and_rois(tmp_path):
    chart = chart_mod.write_chart(chart_mod.build_parser().parse_args(["--out-dir", str(tmp_path), "--film", "rgb"]))
    scene = (tmp_path / "hdr_dr_chart.pbrt").read_text()
    assert scene.count("AreaLightSource") == 2 * chart["n_levels"]
    assert json.loads((tmp_path / "chart.json").read_text())["n_levels"] == 21
    boxes = []
    for p in chart["patches"]:
        for half in ("low", "high"):
            x0, y0, x1, y1 = p[half]["roi_xyxy"]
            assert 0 <= x0 < x1 <= chart["xres"] and 0 <= y0 < y1 <= chart["yres"]
            boxes.append((x0, y0, x1, y1))
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]
    first = chart["patches"][0]
    assert first["low"]["roi_xyxy"][0] > chart["xres"] / 2  # world -x is raster right (pbrt LookAt)
    assert first["high"]["roi_xyxy"][0] < first["low"]["roi_xyxy"][0]


@pytest.fixture(scope="module")
def model_results():
    chart = runner.default_chart()
    return {r: runner.run_model(r, chart, n_px=2304) for r in ("default", "default_hdr_dcg", "default_hdr_lofic")}


def test_model_dr_matches_theory_and_hdr_beats_linear(model_results):
    for res in model_results.values():
        assert res["dr_snr1_to_saturation_db"] == pytest.approx(res["theory_dr_snr1_db"], abs=1.0)
        assert res["dr_snr10_to_saturation_db"] == pytest.approx(res["theory_dr_snr10_db"], abs=1.0)
        # chart-limited DR loses at most one 6 dB step plus the 1/k half-patch offset
        assert res["theory_dr_snr1_db"] - 10.0 < res["dr_snr1_db"] <= res["dr_snr1_to_saturation_db"] + 0.1
        assert res["dr_snr10_db"] < res["dr_snr1_db"]
    assert model_results["default_hdr_dcg"]["dr_snr1_db"] > model_results["default"]["dr_snr1_db"]
    assert model_results["default_hdr_lofic"]["dr_snr1_db"] > model_results["default"]["dr_snr1_db"] + 10
    assert model_results["default_hdr_dcg"]["architecture"] == "dcg"


def test_model_cdp_high_when_bright_low_in_noise(model_results):
    rows = model_results["default"]["patches"]
    ok = [r for r in rows if r["saturated_fraction"] == 0]
    assert ok[-1]["cdp"] > 0.95
    assert rows[0]["cdp"] < 0.3
    assert rows[-1]["saturated_fraction"] > 0.5 or rows[-1]["mean_e"] < model_results["default"]["saturation_e"]


def test_measured_mode_from_synthetic_frames(tmp_path):
    chart = runner.default_chart(xres=480, yres=240)
    rng = np.random.default_rng(0)
    clean = np.zeros((240, 480))
    for p in chart["patches"]:
        for half in ("low", "high"):
            x0, y0, x1, y1 = p[half]["roi_xyxy"]
            clean[y0:y1, x0:x1] = 1e5 * p[half]["relative_luminance"] / chart["contrast_ratio"]
    frames = [clean + rng.normal(0, 2.0, clean.shape) for _ in range(2)]
    res = runner.run_measured(chart, frames)
    # SNR=1 at 2 e- read noise, max unsaturated = brightest low patch
    expected = 20 * np.log10(1e5 / chart["contrast_ratio"] / 2.0)
    assert res["dr_snr1_db"] == pytest.approx(expected, abs=1.5)
    for i, f in enumerate(frames):
        np.savez(tmp_path / f"f{i}.npz", hdr_e=f)
    (tmp_path / "chart.json").write_text(json.dumps(chart))
    out = tmp_path / "out"
    rc = runner.main(
        ["--chart", str(tmp_path / "chart.json"), "--hdr-npz", str(tmp_path / "f0.npz"), str(tmp_path / "f1.npz")]
        + ["--out-dir", str(out), "--figure"]
    )
    assert rc == 0 and (out / "snr_cdp.png").is_file() and (out / "measured_patches.csv").is_file()
    with pytest.raises(ValueError):
        runner.run_measured(chart, [clean[:10]])


def test_cli_model_mode(tmp_path):
    assert runner.main(["--recipes", "default_hdr_3exp", "--pixels-per-patch", "1024", "--out-dir", str(tmp_path)]) == 0
    data = json.loads((tmp_path / "hdr_dr_results.json").read_text())
    assert data["results"][0]["architecture"] == "multi_exposure"


@pytest.mark.skipif(not PBRT.is_file(), reason="pbrt binary not built")
def test_pbrt_render_matches_chart_rois(tmp_path):
    from exr_multispectral import linear_rgb_from_exr

    args = ["--out-dir", str(tmp_path), "--film", "rgb", "--xres", "240", "--yres", "120", "--pixelsamples", "4"]
    chart = chart_mod.write_chart(chart_mod.build_parser().parse_args([*args, "--span-db", "40"]))
    subprocess.run([str(PBRT), "--quiet", "hdr_dr_chart.pbrt"], cwd=tmp_path, check=True, timeout=300)
    g = linear_rgb_from_exr(tmp_path / chart["film_output"])[..., 1]
    for p in chart["patches"]:
        for half in ("low", "high"):
            x0, y0, x1, y1 = p[half]["roi_xyxy"]
            expect = chart["peak_radiance"] * p[half]["relative_luminance"] / chart["contrast_ratio"]
            assert g[y0:y1, x0:x1].mean() == pytest.approx(expect, rel=0.02)
