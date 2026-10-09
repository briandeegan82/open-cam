"""Validation of tools/lens_diffraction_psf.py against analytic results.

Airy pattern / circular-pupil MTF: Born & Wolf (1999) 8.5.2; Goodman (2005) eq. 6-32.
Defocus OPD on the reference sphere: W = -dz (1 - cos u') (Welford 1986, ch. 6).
"""

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import fftconvolve

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import lens_diffraction_psf as ldp  # noqa: E402
from lens_prescription import load_lens_file, paraxial_focal_lengths_mm, traced_f_number  # noqa: E402

DGAUSS = REPO / "config" / "lenses" / "dgauss.50mm.dat"


def _ideal_pupil(f_number: float, n: int = 161) -> ldp.FieldPupil:
    """Aberration-free uniform circular pupil of working f-number N (s_max = 1/(2N))."""
    smax = 0.5 / f_number
    g = np.linspace(-smax, smax, n)
    sx, sy = np.meshgrid(g, g)
    inside = np.hypot(sx, sy) <= smax
    ang = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    s = np.vstack([np.column_stack([sx[inside], sy[inside]]), smax * np.column_stack([np.cos(ang), np.sin(ang)])])
    m = s.shape[0]
    return ldp.FieldPupil(
        image_point=np.zeros(3),
        image_height_mm=0.0,
        object_height_mm=0.0,
        s=s,
        W_mm=np.zeros(m),
        amp=np.ones(m),
        stop_uv=s / smax,
        spot_mm=np.zeros((1, 2)),
        spot_weight=np.ones(1),
        sz_chief=1.0,
        pupil_area_rel_stop=1.0,
    )


@pytest.fixture(scope="module")
def dgauss_f2():
    lens = ldp.LensSystem.from_file(DGAUSS, 25.0)
    t = ldp.pbrt_film_distance_mm(lens, 3200.0, 42.6)
    z_film = float(lens.z_vertex[-1]) + t
    return lens, z_film, z_film - 3200.0


@pytest.mark.parametrize("lam_nm,f_number", [(550.0, 2.0), (450.0, 5.6), (650.0, 11.0)])
def test_ideal_pupil_matches_airy_and_analytic_mtf(lam_nm, f_number):
    fp = _ideal_pupil(f_number)
    os_ = ldp.oversampling_for(fp, lam_nm, 1.0)
    dx = 1.0 / os_
    n = os_ * (2 * math.ceil(48 * lam_nm * 1e-3 * f_number / dx / os_) + 1)  # >= 96 pupil samples
    psf = ldp.fine_psf(fp, lam_nm, dx, n, aberrations=False)
    c = n // 2
    r_um = np.arange(n - c) * dx
    prof = psf[c, c:] / psf[c, c]
    airy = ldp.airy_psf(r_um, lam_nm, f_number)
    first_zero = 1.22 * lam_nm * 1e-3 * f_number
    sel = r_um < 3 * first_zero
    assert np.max(np.abs(prof[sel] - airy[sel])) < 0.01
    f, mx, my = ldp.mtf_from_psf(psf, dx)
    ref = ldp.diffraction_mtf(f, lam_nm, f_number)
    assert np.max(np.abs(mx - ref)) < 0.01
    assert np.max(np.abs(my - ref)) < 0.01


def test_pbrt_focus_matches_paraxial_back_focus():
    lens = ldp.LensSystem.from_file(DGAUSS)
    _efl, bfd = paraxial_focal_lengths_mm(load_lens_file(DGAUSS))
    assert ldp.pbrt_film_distance_mm(lens, 1e9, 35.0) == pytest.approx(bfd, abs=1e-3)


def test_on_axis_working_f_number_matches_traced_f_number():
    # Small stop: transverse aberration is negligible, so the reference-sphere pupil extent equals
    # the marginal-ray angle used by traced_f_number (at f/2 they differ by spherical aberration).
    lens = ldp.LensSystem.from_file(DGAUSS, 4.3)
    z_film = float(lens.z_vertex[-1]) + ldp.pbrt_film_distance_mm(lens, 1e9, 35.0)
    fp = ldp.trace_field_pupil(lens, z_film - 1e7, z_film, 0.0)
    assert fp.working_f_number == pytest.approx(traced_f_number(str(DGAUSS), 4.3), rel=0.005)


def test_film_shift_adds_reference_sphere_defocus(dgauss_f2):
    from scipy.interpolate import LinearNDInterpolator

    lens, z_film, z_obj = dgauss_f2
    dz = 0.02
    a = ldp.trace_field_pupil(lens, z_obj, z_film, 0.0)
    b = ldp.trace_field_pupil(lens, z_obj, z_film + dz, 0.0)
    r = np.hypot(a.s[:, 0], a.s[:, 1])
    sel = r < 0.2
    dw = LinearNDInterpolator(b.s, b.W_mm)(a.s[sel]) - a.W_mm[sel]
    pred = -dz * (1.0 - np.sqrt(1.0 - r[sel] ** 2))
    ok = np.isfinite(dw)
    assert np.max(np.abs(dw[ok] - pred[ok])) < 0.01 * np.max(np.abs(pred))


def test_wave_mtf_approaches_geometric_limit_for_large_defocus(dgauss_f2):
    lens, z_film, z_obj = dgauss_f2
    fp = ldp.trace_field_pupil(lens, z_obj, z_film + 0.3, 0.0)
    os_ = ldp.oversampling_for(fp, 550.0, 2.0)
    psf = ldp.fine_psf(fp, 550.0, 2.0 / os_, 512)
    f, mx, _ = ldp.mtf_from_psf(psf, 2.0 / os_)
    geo = ldp.geometric_otf(fp, f, 0.0)
    for fi in (5.0, 10.0):
        i = int(np.argmin(np.abs(f - fi)))
        assert mx[i] == pytest.approx(geo[i], abs=0.02)


def test_off_axis_pupil_is_vignetted(dgauss_f2):
    lens, z_film, z_obj = dgauss_f2
    on = ldp.trace_field_pupil(lens, z_obj, z_film, 0.0)
    off = ldp.trace_field_pupil(lens, z_obj, z_film, 16.0)
    assert on.pupil_area_rel_stop > 0.99
    assert off.pupil_area_rel_stop < 0.6
    assert off.image_point[1] == pytest.approx(-16.0, abs=1e-5)


def test_pixel_kernel_preserves_in_band_otf():
    fp = _ideal_pupil(8.0)
    pitch, lam, k = 3.0, 550.0, 41
    ker = ldp.pixel_kernel(fp, lam, pitch, k, aberrations=False)
    assert ker.sum() == pytest.approx(1.0)
    otf = np.abs(np.fft.fft2(np.fft.ifftshift(ker)))
    f = np.fft.fftfreq(k, d=pitch * 1e-3)[: k // 2 + 1]
    ref = ldp.diffraction_mtf(f, lam, 8.0)
    assert np.max(np.abs(otf[0, : k // 2 + 1] - ref)) < 0.003


def test_spatially_varying_reduces_to_convolution_and_conserves_energy():
    rng = np.random.default_rng(0)
    img = np.zeros((60, 80))
    img[10:50, 10:70] = rng.random((40, 60))
    k1 = np.outer(np.hanning(7), np.hanning(7))
    k1 /= k1.sum()
    k2 = np.zeros((7, 7))
    k2[3, 1:6] = 0.2
    cx = np.array([20.0, 60.0])
    cy = np.array([15.0, 45.0])
    wx = ldp._hat_weights(80, 0, cx)
    wy = ldp._hat_weights(60, 0, cy)
    assert np.allclose(wx.sum(0), 1.0) and np.allclose(wy.sum(0), 1.0)
    same = {(j, i): k1 for j in range(2) for i in range(2)}
    out = ldp.apply_spatially_varying(img, same, wy, wx, pad=4)
    ref = fftconvolve(img, k1, mode="same")
    assert np.max(np.abs(out - ref)) < 1e-5
    mixed = {(0, 0): k1, (0, 1): k2, (1, 0): k2, (1, 1): k1}
    out2 = ldp.apply_spatially_varying(img, mixed, wy, wx, pad=4)
    assert out2.sum() == pytest.approx(img.sum(), rel=1e-5)


def test_crop_offset_selects_same_kernels(dgauss_f2):
    lens, z_film, z_obj = dgauss_f2
    model = ldp.LensPsfModel(lens, z_film, z_obj, pitch_um=20.0, film_res=(64, 48), tile_grid=(3, 3))
    k_full = model.tile_kernel(*[c[0] for c in model.tile_centres_px()], 550.0, 7)
    assert k_full.shape == (7, 7) and k_full.sum() == pytest.approx(1.0)
    w_crop = ldp._hat_weights(16, 40, model.tile_centres_px()[0])
    assert w_crop[0].sum() == 0.0  # crop at x>=40 is far from the first tile column


def test_disabled_post_psf_leaves_exr_unchanged(tmp_path):
    from exr_multispectral import write_separate_channels_exr

    exr = tmp_path / "in.exr"
    rng = np.random.default_rng(1)
    write_separate_channels_exr(exr, {"S0.550nm": rng.random((8, 8)).astype(np.float32)})
    before = exr.read_bytes()
    cfg = tmp_path / "optics.yaml"
    cfg.write_text("post_psf:\n  enabled: false\n  mode: lens_diffraction\n")
    subprocess.run(
        [sys.executable, str(REPO / "tools" / "apply_spectral_psf.py"), "--exr-in", str(exr), "--config", str(cfg)],
        check=True,
        cwd=REPO,
    )
    assert exr.read_bytes() == before


def test_lens_diffraction_mode_runs_on_spectral_exr(tmp_path):
    from exr_multispectral import read_separate_exr_channels, write_separate_channels_exr

    exr = tmp_path / "in.exr"
    img = np.zeros((96, 96), dtype=np.float32)
    img[48, 48] = 1.0
    write_separate_channels_exr(exr, {"S0.450nm": img, "S0.650nm": img})
    cfg = tmp_path / "optics.yaml"
    cfg.write_text(
        "post_psf:\n  enabled: true\n  mode: lens_diffraction\n  lens_diffraction:\n"
        f"    lensfile: {DGAUSS}\n    aperture_diameter_mm: 2.4\n    focus_distance_m: 3.2\n"
        "    pixel_pitch_um: 3.0\n    tile_grid: [2, 2]\n    pupil_samples: 33\n"
    )
    out = tmp_path / "out.exr"
    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "apply_spectral_psf.py"),
            "--exr-in",
            str(exr),
            "--exr-out",
            str(out),
            "--config",
            str(cfg),
        ],
        check=True,
        cwd=REPO,
    )
    ch = read_separate_exr_channels(out)
    for name in ("S0.450nm", "S0.650nm"):
        assert ch[name].sum() == pytest.approx(1.0, rel=1e-4)
        assert ch[name][48, 48] < 1.0
    # Longer wavelength -> wider Airy core -> lower peak.
    assert ch["S0.650nm"][48, 48] < ch["S0.450nm"][48, 48]
    assert math.isfinite(float(ch["S0.650nm"].max()))
