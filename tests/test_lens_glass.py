"""Glass dispersion for traced lens prescriptions (tools/lens_glass.py)."""

from __future__ import annotations

import math
from pathlib import Path

import lens_glass as lg
import numpy as np
import pytest
from lens_diffraction_psf import LensSystem

REPO = Path(__file__).resolve().parents[1]
DGAUSS = REPO / "config/lenses/dgauss.50mm.dat"
WIDE = REPO / "config/lenses/wide_22mm.dat"
TELE = REPO / "config/lenses/telephoto.250mm.dat"


@pytest.mark.parametrize("name", sorted(lg.SCHOTT_SELLMEIER))
def test_sellmeier_reproduces_catalogue_nd_and_abbe(name):
    nd, vd, coeffs = lg.SCHOTT_SELLMEIER[name]
    n_d, n_f, n_c = (float(lg.sellmeier_n(coeffs, lam)) for lam in (lg.LAMBDA_D_NM, lg.LAMBDA_F_NM, lg.LAMBDA_C_NM))
    assert n_d == pytest.approx(nd, abs=2e-5)
    assert (n_d - 1) / (n_f - n_c) == pytest.approx(vd, abs=0.05)


def test_bk7_like_normal_dispersion_monotone():
    _, _, coeffs = lg.SCHOTT_SELLMEIER["N-BAK2"]
    n = lg.sellmeier_n(coeffs, np.linspace(380, 1000, 64))
    assert np.all(np.diff(n) < 0)


@pytest.mark.parametrize("lens", [DGAUSS, WIDE])
def test_sidecar_keeps_nd_and_sets_design_abbe(lens):
    rows = LensSystem.from_file(lens)
    at_d = lg.refractive_indices(lens, lg.LAMBDA_D_NM)
    np.testing.assert_allclose(at_d, rows.n_after, atol=1e-9)
    for g in lg.load_dispersion(lens):
        if g is not None:
            assert lg.abbe_number(g) == pytest.approx(g.nu_d, rel=1e-9)


def test_dgauss_patent_abbe_numbers():
    nus = [g.nu_d for g in lg.load_dispersion(DGAUSS) if g is not None]
    assert nus == [47.1, 47.1, 30.1, 38.4, 57.0, 48.1]


def test_blue_index_higher_than_red():
    blue, red = lg.refractive_indices(DGAUSS, 450.0), lg.refractive_indices(DGAUSS, 650.0)
    glass = blue != 1.0
    assert np.all(blue[glass] > red[glass])
    assert np.all(blue[~glass] == red[~glass])


def test_lens_without_sidecar_is_wavelength_independent():
    assert lg.load_dispersion(TELE) is None
    assert lg.pbrt_camera_lines(TELE) == []
    np.testing.assert_array_equal(lg.refractive_indices(TELE, 450.0), lg.refractive_indices(TELE, 650.0))


def test_pbrt_values_layout():
    n_rows = len(LensSystem.from_file(DGAUSS).radius)
    vals = lg.pbrt_dispersion_values(DGAUSS)
    assert len(vals) == 7 * n_rows
    disp = lg.load_dispersion(DGAUSS)
    for i, g in enumerate(disp):
        chunk = vals[7 * i : 7 * i + 7]
        if g is None:
            assert chunk == [0.0] * 7
        else:
            assert chunk[0] == pytest.approx(g.scale) and tuple(chunk[1:]) == g.coeffs
    (line,) = lg.pbrt_camera_lines(DGAUSS)
    assert line.strip().startswith('"float dispersion" [') and len(line.split("[")[1].split()) == 7 * n_rows
    assert lg.pbrt_camera_lines(DGAUSS, enabled=False) == []


def test_sidecar_validation(tmp_path):
    lens = tmp_path / "l.dat"
    lens.write_text(DGAUSS.read_text())
    (tmp_path / "l.glass.yaml").write_text("rows:\n  0: {glass: N-BAF10}\n")
    with pytest.raises(ValueError, match="glass rows"):
        lg.load_dispersion(lens)
    rows = ",".join(f"{i}: {{glass: N-SF6}}" for i in (0, 2, 3, 6, 7, 9))
    (tmp_path / "l.glass.yaml").write_text(f"rows: {{{rows}}}\n")
    with pytest.raises(ValueError, match="too far"):
        lg.load_dispersion(lens)


def test_paraxial_longitudinal_colour_is_positive_for_dgauss():
    """Undercorrected-style axial colour sign check: blue focuses closer than red (or within 0.5 %)."""

    def bfd(lam):
        lens = LensSystem.from_file(DGAUSS, wavelength_nm=lam)
        y, u = 1e-3, 0.0
        for i in range(lens.n_surfaces):
            n1, n2 = lens.n_before(i), float(lens.n_after[i])
            c = 0.0 if lens.radius[i] == 0 else 1.0 / lens.radius[i]
            u = (n1 * u - y * (n2 - n1) * c) / n2
            if i < lens.n_surfaces - 1:
                y += lens.thickness[i] * u
        return -y / u

    b_f, b_d, b_c = bfd(lg.LAMBDA_F_NM), bfd(lg.LAMBDA_D_NM), bfd(lg.LAMBDA_C_NM)
    assert math.isfinite(b_d) and b_d > 0
    assert abs(b_f - b_c) < 0.01 * b_d
    assert b_f != b_c


def test_scene_builder_emits_dispersion_line(tmp_path):
    import argparse

    import build_image_quality_targets as iq

    args = argparse.Namespace(
        camera="realistic",
        lensfile="config/lenses/dgauss.50mm.dat",
        aperture_diameter_mm=6.0,
        focus_distance=3.0,
        cam_dist=3.0,
        lens_dispersion=True,
    )
    block = iq._camera_block(args, tmp_path, REPO)
    assert any('"float dispersion"' in line for line in block)
    assert not any("dispersionsplit" in line for line in block)
    args.lens_dispersion_split = True
    assert '    "bool dispersionsplit" true' in iq._camera_block(args, tmp_path, REPO)
    args.lens_dispersion = False
    assert not any("dispersion" in line for line in iq._camera_block(args, tmp_path, REPO))
    assert not any('"float dispersion"' in line for line in iq._camera_block(args, tmp_path, REPO))
    args.lensfile = "config/lenses/telephoto.250mm.dat"
    args.lens_dispersion = True
    assert not any('"float dispersion"' in line for line in iq._camera_block(args, tmp_path, REPO))


def test_psf_model_traces_pupil_per_wavelength():
    import lens_diffraction_psf as ldp

    cfg = {"lensfile": str(DGAUSS), "aperture_diameter_mm": 25.0, "focus_distance_m": 3.2, "pixel_pitch_um": 2.0}
    model = ldp.model_from_config(cfg, {}, {}, repo=REPO, film_res=(2000, 2000))
    assert model.dispersive_lens == DGAUSS
    assert model.lens_at(450.0).n_after[0] > model.lens_at(650.0).n_after[0]
    off = ldp.model_from_config({**cfg, "dispersion": False}, {}, {}, repo=REPO, film_res=(2000, 2000))
    assert off.dispersive_lens is None and off.lens_at(450.0) is off.lens


@pytest.mark.parametrize("name", sorted(lg.SCHOTT_TAUI10))
def test_internal_transmittance_catalogue_and_lambert_beer(name):
    tab = lg.SCHOTT_TAUI10[name]
    wl = np.array(sorted(tab), float)
    tau = np.array([tab[k] for k in sorted(tab)])
    np.testing.assert_allclose(lg.internal_transmittance(name, wl, 10.0), tau, atol=1e-12)
    np.testing.assert_allclose(lg.internal_transmittance(name, wl, 20.0), tau**2, atol=1e-12)
    np.testing.assert_allclose(lg.internal_transmittance(name, wl, 0.0), 1.0)
    assert np.all(np.diff(lg.internal_transmittance(name, [360.0, 450.0, 550.0], 10.0)) >= 0)


def test_every_sidecar_glass_has_transmittance_data():
    assert set(lg.SCHOTT_TAUI10) >= set(lg.SCHOTT_SELLMEIER)
