"""NIR / no-IRCF spectral model (tools/nir_spectra.py) and the radiometric-light scene hooks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import nir_spectra as ns  # noqa: E402
from pbrt_spectral_exr_to_electrons import scene_radiometry_from_manifest  # noqa: E402


def test_alpha_matches_green_2008_table() -> None:
    # Green (2008) tabulates k = 6.37e-4 at 1000 nm -> alpha = 4 pi k / lambda = 80 /cm.
    lines = (REPO / ns.SI_NK_CSV).read_text().splitlines()
    d = np.loadtxt([ln for ln in lines if ln[:1].isdigit()], delimiter=",")
    k1000 = float(d[d[:, 0] == 1000.0, 2][0])
    alpha = ns.silicon_alpha_per_um(np.array([1000.0]))[0]
    assert alpha == pytest.approx(4 * np.pi * k1000 / 1.0, rel=1e-9)
    assert alpha * 1e4 == pytest.approx(4 * np.pi * k1000 / 1e-4, rel=1e-9)


def test_silicon_qe_beer_lambert_limits() -> None:
    lam = np.array([450.0, 850.0, 940.0, 1100.0])
    thin = ns.silicon_qe(lam, thickness_um=3.0)
    thick = ns.silicon_qe(lam, thickness_um=1e4)
    alpha = ns.silicon_alpha_per_um(lam)
    np.testing.assert_allclose(thin, 1 - np.exp(-alpha * 3.0), rtol=1e-12)
    assert thin[0] > 0.99  # blue absorbed in the first ~0.5 um
    assert thin[1] > thin[2] > thin[3]  # absorption length grows into the NIR
    assert thick[1] > 0.99
    # Bare-Si Fresnel reflectance ~33% at 850 nm (n ~ 3.65).
    n, k = ns.silicon_nk(np.array([850.0]))
    r = ((n - 1) ** 2 + k**2) / ((n + 1) ** 2 + k**2)
    assert 0.3 < r[0] < 0.35
    np.testing.assert_allclose(ns.silicon_qe(lam[1:2], thickness_um=3.0, reflectance="fresnel"), (1 - r) * thin[1])


def test_extended_qe_continuous_and_transparent_in_nir() -> None:
    g = ns.GRID_NM
    si = ns.silicon_qe(g)
    vis_wl = np.arange(380.0, 701.0, 1.0)
    vis_qe = 0.4 * np.exp(-0.5 * ((vis_wl - 530.0) / 40.0) ** 2)  # green-like
    ext = ns.extend_qe_to_nir(vis_wl, vis_qe, si)
    j = int(np.argmin(abs(g - 650.0)))
    assert ext[j] == pytest.approx(np.interp(650.0, vis_wl, vis_qe), rel=1e-6)
    assert np.all(ext[g >= 800.0] == pytest.approx(si[g >= 800.0]))  # T = 1 beyond 800 nm


def test_led_spd_fwhm() -> None:
    g = np.arange(700.0, 1001.0, 0.1)
    spd = ns.gaussian_led_spd(g, 850.0, 30.0)
    above = g[spd >= 0.5]
    assert above[-1] - above[0] == pytest.approx(30.0, abs=0.2)
    assert g[np.argmax(spd)] == pytest.approx(850.0)


def test_committed_curves_match_generator(tmp_path: Path) -> None:
    for sub in ("spectra/QE/interpolated", "spectra/silicon"):
        (tmp_path / sub).mkdir(parents=True)
        for f in (REPO / sub).glob("*.csv"):
            (tmp_path / sub / f.name).write_text(f.read_text())
    subprocess.run([sys.executable, str(REPO / "tools/nir_spectra.py"), "--repo-root", str(tmp_path)], check=True)
    for rel in ("spectra/QE/nir/QE_red_noircf.csv", "spectra/illuminant/nir/LED_940nm.csv"):
        assert (tmp_path / rel).read_text() == (REPO / rel).read_text()


def test_radiometric_light_flag(tmp_path: Path) -> None:
    def build(*extra: str) -> tuple[str, dict]:
        out = tmp_path / ("r" if extra else "p")
        args = ["--out-dir", str(out), "--film", "spectral", "--film-output", str(out / "x.exr"), *extra]
        subprocess.run([sys.executable, str(REPO / "tools/build_colorchecker_scene.py"), *args], check=True)
        m = json.loads((out / "colorchecker_manifest.json").read_text())
        return (out / "colorchecker.pbrt").read_text(), m

    pbrt_p, man_p = build()
    assert '"bool photometric"' not in pbrt_p
    assert "chart_illuminance_exr_lux" in scene_radiometry_from_manifest(man_p)
    pbrt_r, man_r = build(
        "--radiometric-light",
        "--illuminant",
        str(REPO / "spectra/illuminant/nir/LED_850nm.csv"),
        "--lambda-max",
        "1100",
        "--spectral-lambda-max",
        "1100",
        "--pbrt-lambda-max",
        "1100",
    )
    assert '"bool photometric" false' in pbrt_r
    assert "chart_illuminance_exr_lux" not in scene_radiometry_from_manifest(man_r)
    with pytest.raises(subprocess.CalledProcessError):
        build("--spectral-lambda-max", "1100")  # stock pbrt range ends at 830 nm


def test_nir_patch_is_opt_in() -> None:
    patch = (REPO / "third_party/patches/0002-nir-lambda-max-radiometric-lights.patch").read_text()
    assert "#define PBRT_LAMBDA_MAX_NM 830" in patch
    assert patch.count('GetOneBool("photometric", true)') == 4
