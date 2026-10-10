import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import si_diffusion as sd  # noqa: E402


def test_absorption_depth_matches_green_2008():
    depth = 1.0 / sd.absorption_coefficient_per_um([450.0, 550.0, 650.0, 850.0])
    assert 0.3 < depth[0] < 0.5 and 1.2 < depth[1] < 2.0 and 3.0 < depth[2] < 4.0 and 15 < depth[3] < 25
    assert np.all(np.diff(depth) > 0)


def test_poisson_rect_strip_matches_cauchy_marginal():
    h, a = 0.7, 0.5
    strip = sd.poisson_rect(-a, a, -1e6, 1e6, h)
    assert strip == pytest.approx(2.0 / math.pi * math.atan(a / h), rel=1e-6)
    assert sd.poisson_rect(-1e6, 1e6, -1e6, 1e6, h) == pytest.approx(1.0, abs=1e-6)


def test_zero_height_is_delta_and_kernels_conserve_charge():
    k0 = sd.pixel_kernel(0.0, 1.0, 2, 4)
    assert k0[2, 2] == 1.0 and k0.sum() == 1.0
    # Reflecting far side: every carrier reaches the junction, so a wide kernel holds all of it.
    wide = sd.pixel_kernel(1.0, 1.5, 12, 2)
    assert wide.sum() == pytest.approx(1.0, abs=1e-2)
    k = sd.wavelength_kernel(600.0, 1.4)
    assert k.sum() == pytest.approx(1.0) and np.allclose(k, k.T) and np.allclose(k, k[::-1, ::-1])


def test_fsi_red_spreads_more_bsi_blue_spreads_more():
    fsi = {"architecture": "fsi", "epi_thickness_um": 5.0, "depletion_depth_um": 0.5}
    bsi = {"architecture": "bsi", "epi_thickness_um": 3.0, "depletion_depth_um": 1.0}
    nn = sd.nearest_neighbour_fraction
    assert nn(sd.wavelength_kernel(650, 1.4, fsi)) > 2 * nn(sd.wavelength_kernel(450, 1.4, fsi))
    assert nn(sd.wavelength_kernel(450, 1.4, bsi)) > nn(sd.wavelength_kernel(650, 1.4, bsi))

    def leak(k):
        return 1.0 - k[k.shape[0] // 2, k.shape[1] // 2]

    assert leak(sd.wavelength_kernel(650, 1.0, fsi)) > leak(sd.wavelength_kernel(650, 3.0, fsi))


def test_channel_kernel_is_spectral_mixture_and_apply_conserves_charge():
    lam = np.array([450.0, 650.0])
    ks = sd.channel_kernels(lam, np.array([[1.0, 0.0], [0.0, 1.0]]), 1.4)
    assert np.allclose(ks[0], sd.wavelength_kernel(450.0, 1.4))
    mix = sd.channel_kernels(lam, np.array([[1.0], [3.0]]), 1.4)[0]
    assert np.allclose(mix, 0.25 * ks[0] + 0.75 * ks[1])
    idx = np.tile(np.array([[0, 1], [1, 0]]), (8, 8))
    raw = np.zeros((16, 16))
    raw[8, 8] = 1000.0
    out = sd.apply_site_kernels(raw, idx, ks)
    assert out.sum() == pytest.approx(1000.0, rel=1e-5)
    assert out[8, 9] == pytest.approx(1000.0 * ks[0][4, 5], rel=1e-5)


def test_apply_silicon_diffusion_bayer_routes_by_source_and_needs_spectra():
    import apply_emva_noise as aen
    import cfa_mosaic as cm

    lay = cm.resolve_layout({"layout": "RGGB"})
    # [K, C]: R sees only 650 nm, G only 550 nm, B only 450 nm.
    meta = {"channel_spectra": {"wavelength_nm": [450.0, 550.0, 650.0], "electrons": [[0, 0, 1], [0, 1, 0], [1, 0, 0]]}}
    raw = np.zeros((16, 16))
    raw[8, 8] = 1000.0  # an R site (RGGB, even row/col)
    out = aen.apply_silicon_diffusion(raw, lay, {"architecture": "fsi"}, meta, 1.4)
    k_red = sd.wavelength_kernel(650.0, 1.4, {"architecture": "fsi"})
    assert out.sum() == pytest.approx(1000.0, rel=1e-5)
    assert out[8, 9] == pytest.approx(1000.0 * k_red[4, 5], rel=1e-5)
    with pytest.raises(ValueError, match="spectral EXR"):
        aen.apply_silicon_diffusion(raw, lay, {}, {}, 1.4)
