import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import lens_ghosts as lg  # noqa: E402
import lens_scatter as ls  # noqa: E402

DGAUSS = REPO / "config" / "lenses" / "dgauss.50mm.dat"


@pytest.fixture(scope="module")
def lens():
    return lg.load_lens(DGAUSS, aperture_diameter_mm=6.194, focus_distance_m=3.0)


def test_abg_shape_normalised():
    for g in (2.0, 2.5):
        x = np.linspace(0, 1, 200001)
        y = 2 * math.pi * x * ls.abg_shape(x, g, 0.01)
        assert np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)) == pytest.approx(1.0, rel=1e-3)


def test_tis_rayleigh_rice():
    s = ls.SurfaceScatter(0, 0.5, 1.0, 1.0)
    assert float(s.tis(500.0, 2.0)) == pytest.approx((2 * math.pi * 2.0 * 0.5 / 500.0) ** 2)
    assert float(s.tis(400.0, 2.0) / s.tis(800.0, 2.0)) == pytest.approx(4.0)


def test_k_matches_paraxial_film_shift(lens):
    """A small direction change after surface i, traced paraxially, lands k_i * dbeta away on film."""
    for s in ls.surface_scatter(lens):
        to_surf, m = ls._after_matrices(lens, s.index)
        n_i = lens.surfaces[s.index].n_after
        assert s.k_mm == pytest.approx(abs(m @ np.array([0.0, n_i * 1e-4]))[0] / 1e-4)
        assert 0 < s.beta_max <= 1


def test_last_surface_scatter_no_truncation_and_short_lever(lens):
    sc = ls.surface_scatter(lens)
    last = sc[-1]
    assert last.beta_max == 1.0
    assert last.k_mm == pytest.approx(lens.z_film - lens.surfaces[last.index].z, rel=1e-9)


def test_kernel_energy(lens):
    kern, tis, rep = ls.scatter_kernel(lens, (401, 401), 0.01, {"roughness_nm": 2.0})
    assert tis == pytest.approx(sum(r["tis"] for r in rep))
    reach = sum(r["tis"] * r["reaches_film"] for r in rep)
    # The g = 2 tail is logarithmic, so much lands beyond the +-4 mm grid: bound by the inscribed disc.
    norm1 = ls.abg_norm(2.0, ls.DEFAULTS["abg_shoulder"], 1.0)
    disc = sum(
        r["tis"] * ls.abg_norm(2.0, ls.DEFAULTS["abg_shoulder"], min(4.0 / r["k_mm"], r["beta_max"])) / norm1
        for r in rep
    )
    assert 0.98 * disc < kern.sum() <= 1.001 * reach


def test_black_hole_glare_matches_analytic_g2(lens):
    """Uniform field with a central black hole: glare at the centre = sum TIS_i x ABg integral outside the hole."""
    pix, half, r_hole = 0.02, 300, 20
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1]
    img = (np.hypot(yy, xx) > r_hole).astype(float)
    cfg = {"roughness_nm": 3.0, "reference_nm": 550.0}
    out, info = ls.apply_scatter(img[:, :, None], [550.0], lens, pix, cfg)
    glare = out[half, half, 0]
    g, sh = 2.0, ls.DEFAULTS["abg_shoulder"]
    b_field = half * pix  # the field is finite: integrate the ring between the hole and the field edge
    expect = 0.0
    for r in info["surfaces"]:
        k, bmax = r["k_mm"], r["beta_max"]
        lo, hi = min(r_hole * pix / k, bmax), min(b_field / k, bmax)
        expect += r["tis"] * math.pi * math.log((sh**g + hi**2) / (sh**g + lo**2)) / ls.abg_norm(g, sh, 1.0)
    assert glare == pytest.approx(expect, rel=0.1)
    assert glare > 0


def test_scatter_bluer(lens):
    img = np.zeros((101, 101, 2))
    img[50, 50] = 1.0
    out, _ = ls.apply_scatter(img, [450.0, 650.0], lens, 0.01)
    halo = out.sum((0, 1)) - out[50, 50]
    assert halo[0] / halo[1] == pytest.approx((650 / 450) ** 2, rel=0.05)
