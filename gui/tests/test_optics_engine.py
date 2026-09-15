import pytest
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


# =====================================================================
# Stray light
# =====================================================================
def _spike_angles_deg(n_blades: int, size: int = 256, radius: float = 60.0) -> list[float]:
    """Angles of the starburst arms, found as circular local maxima in log intensity."""
    from scipy.ndimage import map_coordinates

    kernel = oe.aperture_diffraction_kernel(n_blades, size)
    c = size / 2.0
    n_ang = 2880
    ang = np.linspace(0.0, 2.0 * np.pi, n_ang, endpoint=False)
    prof = map_coordinates(
        np.log10(kernel + 1e-20),
        [c + radius * np.sin(ang), c + radius * np.cos(ang)],
        order=1,
        mode="nearest",
    )
    peaks = (prof > np.roll(prof, 1)) & (prof >= np.roll(prof, -1)) & (prof > np.median(prof) + 1.0)

    # Rasterising the polygon mask splits each arm into a narrow pair, so merge
    # maxima that sit within 10 degrees of one another.
    merged: list[int] = []
    for i in np.flatnonzero(peaks):
        if not merged or (i - merged[-1]) * 360.0 / n_ang > 10.0:
            merged.append(int(i))
    if len(merged) > 1 and (merged[0] + n_ang - merged[-1]) * 360.0 / n_ang <= 10.0:
        merged.pop()
    return [i * 360.0 / n_ang for i in merged]


def test_stray_light_disabled_is_a_noop():
    img = oe.stray_light_test_image(64)
    out = oe.apply_stray_light(img, oe.stray_light_config(enabled=False, veiling_glare_fraction=0.2))
    assert np.allclose(out, img)


def test_veiling_glare_destroys_contrast_without_blurring():
    """Veiling glare adds a constant, so the black-to-white *difference* survives
    while their sum grows -- contrast falls with no loss of sharpness."""
    clean = oe.stray_light_test_image(128)
    strayed = oe.apply_stray_light(clean, oe.stray_light_config(veiling_glare_fraction=0.2))
    ec = oe.edge_contrast(clean, strayed)

    assert ec.strayed_contrast < ec.clean_contrast
    assert ec.contrast_loss_percent > 20.0
    # The step is still a step: the jump across the edge is unchanged.
    assert np.ptp(ec.strayed) == pytest.approx(0.8 * np.ptp(ec.clean), rel=1e-3)


def test_more_veiling_glare_costs_more_contrast():
    clean = oe.stray_light_test_image(96)
    losses = [
        oe.edge_contrast(
            clean, oe.apply_stray_light(clean, oe.stray_light_config(veiling_glare_fraction=v))
        ).contrast_loss_percent
        for v in (0.0, 0.05, 0.10, 0.20)
    ]
    assert losses == sorted(losses)
    # apply_stray_light round-trips through float32, so "no loss" is not bit-exact.
    assert losses[0] == pytest.approx(0.0, abs=1e-5)


def test_halo_spreads_light_around_the_bright_source():
    clean = oe.stray_light_test_image(96)
    strayed = oe.apply_stray_light(
        clean, oe.stray_light_config(halo_sigma_pixels=12.0, halo_strength=0.2)
    )
    # Scatter is additive: it lifts the dark surround near the source.
    assert strayed.sum() > clean.sum()
    assert strayed[5, 5] > clean[5, 5]


def test_ghost_is_a_rotated_copy_of_the_scene():
    clean = oe.stray_light_test_image(96)
    strength = 0.1
    strayed = oe.apply_stray_light(
        clean, oe.stray_light_config(ghost_enabled=True, ghost_strength=strength)
    )
    np.testing.assert_allclose(strayed, clean + strength * np.rot90(clean, k=2), rtol=1e-5)


def test_even_blade_counts_give_n_spikes():
    """Opposing blade pairs are parallel, so their perpendicular streaks are
    collinear and merge -- 8 blades give 8 arms, not 16."""
    for n in (8, 12):
        assert len(_spike_angles_deg(n)) == n


def test_odd_blade_counts_give_2n_spikes():
    for n in (5, 7, 9):
        assert len(_spike_angles_deg(n)) == 2 * n


def test_spikes_are_evenly_spaced():
    angles = _spike_angles_deg(7)
    spacing = np.diff(angles)
    assert np.allclose(spacing, 360.0 / 14.0, atol=1.0)


def test_starburst_kernel_is_peak_normalised_and_centred():
    kernel = oe.aperture_diffraction_kernel(6, 128)
    assert kernel.max() == pytest.approx(1.0)
    assert np.unravel_index(int(np.argmax(kernel)), kernel.shape) == (64, 64)


def test_tone_mapping_keeps_order_and_bounds():
    img = oe.stray_light_test_image(48)
    toned = oe.tone_for_display(img)
    assert toned.min() >= 0.0 and toned.max() <= 1.0
    # A brighter scene value must never map to a darker display value.
    flat_in, flat_out = img.ravel(), toned.ravel()
    order = np.argsort(flat_in)
    assert np.all(np.diff(flat_out[order]) >= -1e-9)


def test_test_image_spans_a_large_dynamic_range():
    img = oe.stray_light_test_image(64)
    assert img.max() / img[img > 0].min() > 1000.0
