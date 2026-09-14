import numpy as np

from opencam_gui.core import optics_engine as oe


def test_rgb_center_wavelengths():
    centers = oe.rgb_center_wavelengths_nm()
    assert centers["R"] > centers["G"] > centers["B"]


def test_chromatic_sigma_matches_tool_module_directly():
    m = oe._psf_module()
    expected = m.psf_sigma_chromatic(550.0, 2.8, 1.4, 0.5)
    got = oe.chromatic_sigma_px(550.0, 2.8, 1.4, 0.5)
    assert got == expected


def test_compute_psf_kernel_chromatic_gaussian_peak_normalised():
    result = oe.compute_psf_kernel(
        mode="chromatic_gaussian", wavelength_nm=550.0, f_number=2.8,
        pixel_pitch_um=1.4, sigma_geometric_px=0.5, size=33,
    )
    assert result.kernel.shape == (33, 33)
    assert np.isclose(result.kernel.max(), 1.0)
    assert result.rho0_px == 0.0


def test_compute_psf_kernel_airy_disk_has_rho0():
    result = oe.compute_psf_kernel(
        mode="airy_disk", wavelength_nm=550.0, f_number=8.0,
        pixel_pitch_um=1.4, sigma_geometric_px=0.0, size=65,
    )
    assert result.rho0_px > 0.0
    assert np.isclose(result.kernel.max(), 1.0)


def test_larger_f_number_widens_the_psf():
    narrow = oe.compute_psf_kernel(
        mode="chromatic_gaussian", wavelength_nm=550.0, f_number=1.8,
        pixel_pitch_um=1.4, sigma_geometric_px=0.0, size=65,
    )
    wide = oe.compute_psf_kernel(
        mode="chromatic_gaussian", wavelength_nm=550.0, f_number=16.0,
        pixel_pitch_um=1.4, sigma_geometric_px=0.0, size=65,
    )
    assert wide.sigma_total_px > narrow.sigma_total_px


def test_radial_profile_is_monotonic_decreasing_for_gaussian():
    result = oe.compute_psf_kernel(
        mode="chromatic_gaussian", wavelength_nm=550.0, f_number=4.0,
        pixel_pitch_um=1.4, sigma_geometric_px=1.0, size=65,
    )
    radii, profile = oe.radial_profile(result.kernel)
    # Compare near the peak, not the far tail: a small-sigma Gaussian underflows
    # to exactly 0.0 well before the kernel edge, which would make any two far
    # radii spuriously "equal" instead of strictly decreasing.
    assert profile[0] > profile[3] > profile[8]


def test_lateral_ca_zero_coefficient_is_a_noop():
    chart = oe.radial_test_chart(64, 6)
    rgb = oe.lateral_ca_rgb_preview(chart, 0.0, 550.0)
    assert np.allclose(rgb[..., 0], rgb[..., 1], atol=1e-6)
    assert np.allclose(rgb[..., 1], rgb[..., 2], atol=1e-6)


def test_lateral_ca_nonzero_coefficient_splits_channels():
    chart = oe.radial_test_chart(64, 6)
    rgb = oe.lateral_ca_rgb_preview(chart, 0.03, 550.0)
    assert not np.allclose(rgb[..., 0], rgb[..., 2], atol=1e-6)
