"""Procedural IQ-lab diorama builder."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import build_iq_diorama as dio
import numpy as np
import pytest
from synthetic_data import REPO

PBRT = Path(os.environ.get("OPENCAM_PBRT", REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"))


def test_procedural_textures():
    star = dio.siemens_star(256, spokes=36)
    assert star.min() == pytest.approx(0.05) and star.max() == pytest.approx(0.8)
    edge = dio.slanted_edge(128, 5.0)
    assert 0.1 <= edge.min() < edge.max() <= 0.8 and edge[:, :10].mean() < edge[:, -10:].mean()


def test_builder_outputs_and_rois(tmp_path):
    args = dio.build_parser().parse_args(["--out-dir", str(tmp_path), "--xres", "720", "--yres", "480"])
    meta = dio.write_diorama(args)
    scene = (tmp_path / "iq_diorama.pbrt").read_text()
    assert scene.count("AreaLightSource") == 3 and scene.count('"sphere"') == 5
    assert len(meta["colorchecker"]) == 24 and len(meta["skin"]) == 12
    assert set(meta["cards"]) == {"dead_leaves", "siemens_star", "slanted_edge"}
    for name in meta["cards"]:
        assert (tmp_path / "textures" / f"{name}.png").is_file()
    rois = [p["roi_xyxy"] for p in meta["colorchecker"] + meta["skin"]] + [
        c["roi_xyxy"] for c in meta["cards"].values()
    ]
    for x0, y0, x1, y1 in rois + [meta["window"]["roi_xyxy"], meta["shadow_reference"]["roi_xyxy"]]:
        assert 0 <= x0 < x1 <= 720 and 0 <= y0 < y1 <= 480
    assert json.loads((tmp_path / "diorama.json").read_text())["film_output"] == "iq_diorama_spectral.exr"


@pytest.mark.skipif(not PBRT.is_file(), reason="pbrt binary not built")
def test_render_rois_hit_the_right_objects(tmp_path):
    from exr_multispectral import linear_rgb_from_exr

    args = [
        "--out-dir",
        str(tmp_path),
        "--film",
        "rgb",
        "--xres",
        "360",
        "--yres",
        "240",
        "--pixelsamples",
        "8",
        "--maxdepth",
        "3",
    ]
    meta = dio.write_diorama(dio.build_parser().parse_args(args))
    subprocess.run([str(PBRT), "--quiet", "iq_diorama.pbrt"], cwd=tmp_path, check=True, timeout=600)
    img = linear_rgb_from_exr(tmp_path / meta["film_output"]).mean(axis=2)

    def roi(r):
        x0, y0, x1, y1 = r
        return img[y0:y1, x0:x1]

    cc = [float(np.median(roi(p["roi_xyxy"]))) for p in meta["colorchecker"]]
    assert cc[18] > 10 * cc[23]  # ColorChecker white vs black
    assert np.all(np.diff(cc[18:24]) < 0)  # neutral row descends
    window = float(np.median(roi(meta["window"]["roi_xyxy"])))
    assert window > 100 * float(np.median(roi(meta["shadow_reference"]["roi_xyxy"])))
    assert (
        roi(meta["cards"]["dead_leaves"]["roi_xyxy"]).std() > 0.1 * roi(meta["cards"]["dead_leaves"]["roi_xyxy"]).mean()
    )


def test_sppm_integrator_line(tmp_path):
    args = dio.build_parser().parse_args(
        ["--out-dir", str(tmp_path), "--integrator", "sppm", "--sppm-photons", "5000", "--sppm-radius", "0.03"]
    )
    dio.write_diorama(args)
    scene = (tmp_path / "iq_diorama.pbrt").read_text()
    assert (
        'Integrator "sppm" "integer maxdepth" [6] "integer photonsperiteration" [5000] "float radius" [0.03]' in scene
    )
    default = dio.build_parser().parse_args(["--out-dir", str(tmp_path / "d")])
    assert dio.integrator_line(default).startswith('Integrator "volpath"')
    with pytest.raises(ValueError):
        dio.integrator_line(dio.build_parser().parse_args(["--integrator", "sppm", "--sppm-photons", "0"]))


def _spectral_exr(path):
    import re

    import Imath
    import OpenEXR

    exr = OpenEXR.InputFile(str(path))
    dw = exr.header()["dataWindow"]
    w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    chans = {c: np.frombuffer(exr.channel(c, pt), np.float32).reshape(h, w) for c in exr.header()["channels"]}
    buckets = sorted((c for c in chans if c.startswith("S0.")), key=lambda c: float(re.findall(r"[\d.]+", c[3:])[0]))
    return chans, buckets


@pytest.mark.skipif(not PBRT.is_file(), reason="pbrt binary not built")
def test_sppm_spectral_film_matches_volpath(tmp_path):
    """Spectral SPPM (pbrt patch 0009) writes the volpath EXR layout and agrees per bucket."""
    out = {}
    for name, extra in (
        ("volpath", ["--pixelsamples", "64"]),
        ("sppm", ["--integrator", "sppm", "--pixelsamples", "256", "--sppm-photons", "20000"]),
    ):
        d = tmp_path / name
        args = ["--out-dir", str(d), "--xres", "60", "--yres", "40", "--spectral-nbuckets", "8", *extra]
        meta = dio.write_diorama(dio.build_parser().parse_args(args))
        subprocess.run([str(PBRT), "--quiet", "--seed", "1", "iq_diorama.pbrt"], cwd=d, check=True, timeout=600)
        out[name] = _spectral_exr(d / meta["film_output"])
    (vol, vb), (spm, sb) = out["volpath"], out["sppm"]
    assert sb == vb and len(sb) == 8
    assert all(np.isfinite(spm[c]).all() for c in spm)
    ratio = np.array([spm[c].mean() / vol[c].mean() for c in vb])
    assert np.all(np.abs(ratio - 1) < 0.08), ratio
    # RGB channels (sensor RGB -> output RGB) agree with the buckets' integral as in volpath
    rgb = np.array([spm[c].mean() / vol[c].mean() for c in "RGB"])
    assert np.all(np.abs(rgb - 1) < 0.08), rgb


def test_glass_roughness_is_opt_in(tmp_path):
    def glass_line(*extra):
        args = dio.build_parser().parse_args(["--out-dir", str(tmp_path), *extra])
        dio.write_diorama(args)
        return next(ln for ln in (tmp_path / "iq_diorama.pbrt").read_text().splitlines() if '"dielectric"' in ln)

    assert "roughness" not in glass_line()
    assert '"float roughness" [0.15]' in glass_line("--glass-roughness", "0.15")
    with pytest.raises(ValueError):
        dio.write_diorama(dio.build_parser().parse_args(["--out-dir", str(tmp_path), "--glass-roughness", "-0.1"]))
