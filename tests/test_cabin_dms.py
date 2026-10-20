import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import build_cabin_scene as cabin  # noqa: E402
import nir_skin  # noqa: E402


def test_absorbers_and_skin_model_bounded():
    wl = np.arange(400.0, 1101.0, 10.0)
    assert nir_skin.water_mua(np.array([970.0]))[0] > 5 * nir_skin.water_mua(np.array([850.0]))[0]
    assert nir_skin.blood_mua(np.array([420.0]))[0] > 100 * nir_skin.blood_mua(np.array([800.0]))[0]
    r = [nir_skin.skin_reflectance(wl, f) for f in nir_skin.MELANIN.values()]
    assert all(np.all((x > 0) & (x < 1)) for x in r)
    assert all(np.all(a[:10] >= b[:10]) for a, b in zip(r, r[1:]))  # more melanin -> darker (blue)


def test_measured_extension_matches_join_and_has_water_dip():
    tone, fitz = cabin.SKIN["light"]
    d = np.loadtxt(cabin.REPO / "spectra/xrite" / f"{tone}.csv", delimiter=",")
    wl = np.array([600.0, 730.0, 850.0, 940.0, 970.0])
    r = nir_skin.extend_measured(wl, d[:, 0], d[:, 1], f_mel=nir_skin.MELANIN[fitz])
    assert r[1] == pytest.approx(np.interp(730.0, d[:, 0], d[:, 1]))
    assert r[4] < r[3] < r[2]
    assert (
        cabin.skin_spectrum("dark")[np.searchsorted(cabin.WL, 940)]
        < cabin.skin_spectrum("light")[np.searchsorted(cabin.WL, 940)]
    )


def test_builder_writes_radiometric_led_and_gaze_truth(tmp_path):
    cabin.main(["--out-dir", str(tmp_path), "--gaze-yaw-deg", "10", "--glasses"])
    txt = (tmp_path / "cabin.pbrt").read_text()
    m = json.loads((tmp_path / "cabin_manifest.json").read_text())
    assert 'AreaLightSource "diffuse"' in txt and '"bool photometric" false' in txt
    assert '"float lambdamax" [1100]' in txt and '"float eta" [1.376]' in txt
    assert m["led"]["peak_nm"] == 940 and len(m["driver"]["eyes"]) == 2
    for e in m["driver"]["eyes"]:
        d = np.subtract(e["pupil_centre"], e["eye_centre"])
        assert np.linalg.norm(d) == pytest.approx(cabin.EYE_R, rel=1e-3)
        assert d[0] > 0  # yaw +10 deg turns the gaze towards +x


def test_glint_prediction_projects_near_pupil_and_follows_gaze(tmp_path):
    m0 = cabin.build(cabin.parse_args(["--out-dir", str(tmp_path / "a")]))
    m1 = cabin.build(cabin.parse_args(["--out-dir", str(tmp_path / "b"), "--gaze-yaw-deg", "15"]))
    for e0, e1 in zip(m0["driver"]["eyes"], m1["driver"]["eyes"]):
        g0 = e0["glint"]
        assert g0["on_cornea"] and g0["fresnel"] == pytest.approx(((1.376 - 1) / 2.376) ** 2, rel=0.01)
        assert np.linalg.norm(np.subtract(g0["point"], e0["cornea_centre"])) == pytest.approx(cabin.CORNEA_R, abs=2e-6)
        assert np.linalg.norm(np.subtract(g0["pixel_xy"], e0["pupil_pixel_xy"])) < 4  # looking at the camera
        # the glint stays near-fixed while the pupil moves with the gaze (PCCR principle)
        shift_pupil = np.subtract(e1["pupil_pixel_xy"], e0["pupil_pixel_xy"])
        shift_glint = np.subtract(e1["glint"]["pixel_xy"], g0["pixel_xy"])
        assert abs(shift_pupil[0]) > 1.5 * abs(shift_glint[0])


PBRT = Path(os.environ.get("OPENCAM_PBRT", cabin.REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"))
CIE_Y_INTEGRAL = 106.856895  # pbrt's spectral film scales radiance by it


def _render_cornea(td: Path, alpha: float, spp: int = 256):
    """Isolated cornea sphere (black core) 0.65 m from the camera, 2.5 mm Lambertian LED 40 mm off-axis,
    flat 500-600 nm spectrum (works on the stock 830 nm build). Returns (pixel sum of L*dOmega, analytic)."""
    res, fov, d, r_led, w_sr = 48, 3.0, 0.65, 2.5e-3, 0.5
    rough = f'"float roughness" [{alpha}] "bool remaproughness" false' if alpha else ""
    (td / "g.pbrt").write_text(
        f"""LookAt 0 0 0 0 0 1 0 1 0
Camera "perspective" "float fov" [{fov}]
Sampler "zsobol" "integer pixelsamples" [{spp}]
Integrator "volpath" "integer maxdepth" [4]
PixelFilter "box" "float xradius" [0.5] "float yradius" [0.5]
Film "spectral" "string filename" ["g.exr"] "integer xresolution" [{res}] "integer yresolution" [{res}]
  "bool savefp16" false "integer nbuckets" [1] "float lambdamin" [500] "float lambdamax" [600]
WorldBegin
AttributeBegin Translate 0.04 0 0
AreaLightSource "diffuse" "spectrum L" [499 0 500 1 600 1 601 0] "bool photometric" false
  "float scale" [{w_sr / (100 * math.pi * r_led**2):.6g}]
Material "diffuse" "rgb reflectance" [0 0 0]
Shape "disk" "float radius" [{r_led}]
AttributeEnd
AttributeBegin Translate 0 0 {d}
Material "dielectric" "float eta" [{cabin.CORNEA_ETA}] {rough}
Shape "sphere" "float radius" [{cabin.CORNEA_R}]
Material "diffuse" "rgb reflectance" [0 0 0]
Shape "sphere" "float radius" [{cabin.CORNEA_R * 0.99}]
AttributeEnd
"""
    )
    r = subprocess.run([str(PBRT), "--quiet", "g.pbrt"], cwd=td, capture_output=True, timeout=600)
    if r.returncode != 0:
        pytest.skip(f"pbrt lacks patch 0015 or failed: {r.stderr.decode()[-200:]}")
    from exr_multispectral import spectral_buckets_from_exr

    img, _ = spectral_buckets_from_exr(td / "g.exr")
    w = 2 * math.tan(math.radians(fov / 2)) / res
    yy, xx = np.mgrid[0:res, 0:res]
    omega = w * w / (1 + ((xx + 0.5 - res / 2) * w) ** 2 + ((yy + 0.5 - res / 2) * w) ** 2) ** 1.5
    pbrt_e = float((img[..., 0] * omega).sum()) * 100 / CIE_Y_INTEGRAL  # band-integrated W/m^2
    g = cabin.corneal_glint(
        np.array([0.0, 0.0, d]),
        np.array([0.0, 0.0, -1.0]),
        np.zeros(3),
        np.array([0.04, 0, 0]),
        np.array([0.0, 0.0, 1.0]),
        w_sr,
    )
    return pbrt_e, g["camera_irradiance_w_m2"]


@pytest.mark.skipif(not PBRT.is_file(), reason="pbrt not built")
def test_rough_cornea_glint_power_matches_mirror_sphere(tmp_path):
    for alpha in (0.02, 0.05):
        got, want = _render_cornea(tmp_path, alpha)
        assert got == pytest.approx(want, rel=0.05), alpha
