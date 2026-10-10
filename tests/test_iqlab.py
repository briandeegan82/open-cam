"""Known-answer tests for the tools/iqlab image-quality metrics."""

from __future__ import annotations

import numpy as np
import pytest
from iqlab import cpiq, dead_leaves, geometry, p2020, snr
from scipy import ndimage
from scipy.stats import norm

RNG = np.random.default_rng(1234)


def test_patch_snr_matches_injected_noise():
    patch = 400.0 + RNG.normal(0, 20.0, (200, 200))
    s = snr.patch_stats(patch)
    assert s.mean == pytest.approx(400.0, rel=1e-3)
    assert s.snr == pytest.approx(20.0, rel=0.02)
    assert s.snr_db == pytest.approx(26.02, abs=0.1)


def test_patch_snr_ignores_illumination_ramp_and_counts_saturation():
    yy, xx = np.mgrid[0:100, 0:100]
    patch = 1000.0 + 3.0 * xx + 2.0 * yy + RNG.normal(0, 10.0, (100, 100))
    assert snr.patch_stats(patch).std == pytest.approx(10.0, rel=0.05)
    assert snr.patch_stats(patch, detrend=False).std > 50
    clipped = np.minimum(patch, 1200.0)
    assert snr.patch_stats(clipped, saturation=1200.0).saturated_fraction > 0.3


def test_temporal_stats_reject_fixed_pattern():
    fpn = RNG.normal(0, 30.0, (64, 64))
    frames = 500.0 + fpn + RNG.normal(0, 5.0, (16, 64, 64))
    assert snr.temporal_patch_stats(frames).std == pytest.approx(5.0, rel=0.05)


def test_dynamic_range_of_shot_read_limited_pixel():
    read, full_well = 3.0, 10_000.0
    signal = np.geomspace(0.1, full_well, 200)
    curve = snr.shot_read_snr(signal, read)
    # SNR = 1 where S^2 = S + read^2 -> S = (1 + sqrt(1 + 4 read^2)) / 2
    s1 = (1 + np.sqrt(1 + 4 * read**2)) / 2
    dr = snr.dynamic_range(signal, curve, snr_threshold=1.0)
    assert dr.min_signal == pytest.approx(s1, rel=1e-3)
    assert dr.db == pytest.approx(20 * np.log10(full_well / s1), abs=0.01)
    dr10 = snr.dynamic_range(signal, curve, snr_threshold=10.0)
    assert dr10.min_signal == pytest.approx((100 + np.sqrt(100**2 + 4 * 100 * read**2)) / 2, rel=1e-3)
    assert dr10.stops < dr.stops


def test_dynamic_range_excludes_saturated_patches():
    signal = np.array([1.0, 10.0, 100.0, 1000.0, 1000.0])
    sat = np.array([0, 0, 0, 0, 1.0])
    dr = snr.dynamic_range(signal * [1, 1, 1, 1, 5], np.array([0.5, 2, 9, 30, 30]), saturated_fraction=sat)
    assert dr.max_signal == 1000.0
    assert snr.snr_threshold_signal(signal, np.full(5, 0.1), 1.0) != snr.snr_threshold_signal(signal, signal, 1.0)


def test_cdp_limits_and_gaussian_prediction():
    dark = np.full(1000, 100.0)
    bright = np.full(1000, 300.0)
    assert p2020.contrast_detection_probability(dark, bright) == 1.0
    assert p2020.contrast_detection_probability(bright, dark) == 0.0
    sigma = 40.0
    dark = 100.0 + RNG.normal(0, sigma, 200_000)
    bright = 300.0 + RNG.normal(0, sigma, 200_000)
    cdp = p2020.contrast_detection_probability(dark, bright, epsilon=0.5)
    # Monte-Carlo reference from the same Gaussian model
    d = 100.0 + RNG.normal(0, sigma, 400_000)
    b = 300.0 + RNG.normal(0, sigma, 400_000)
    c = (b - d) / (b + d)
    ref = np.mean(np.abs(c - 0.5) <= 0.25)
    assert cdp == pytest.approx(ref, abs=0.01)
    noisier = p2020.contrast_detection_probability(
        100.0 + RNG.normal(0, 4 * sigma, 50_000), 300.0 + RNG.normal(0, 4 * sigma, 50_000)
    )
    assert noisier < cdp


def test_cdp_with_scene_contrast_penalises_compression():
    dark = np.full(100, 100.0)
    bright = np.full(100, 150.0)  # rendered contrast 0.2, scene contrast 0.8
    assert p2020.contrast_detection_probability(dark, bright, nominal_contrast=0.8) == 0.0
    assert p2020.cdp_vs_level([dark, bright, bright * 2], contrast_pairs=[(0, 1), (1, 2)]).tolist() == [1.0, 1.0]


def test_colour_separation_mahalanobis_and_probability():
    a = RNG.normal(0, 1, (20_000, 3))
    b = RNG.normal(0, 1, (20_000, 3)) + np.array([2.0, 0.0, 0.0])
    sep = p2020.colour_separation(a, b)
    assert sep.euclidean == pytest.approx(2.0, abs=0.03)
    assert sep.mahalanobis == pytest.approx(2.0, abs=0.03)
    assert sep.separation_probability == pytest.approx(norm.cdf(1.0), abs=0.005)
    assert sep.empirical_accuracy == pytest.approx(norm.cdf(1.0), abs=0.01)
    # same mean shift but along a high-variance axis is less separable
    cov = np.diag([16.0, 1.0, 1.0])
    a2 = RNG.multivariate_normal(np.zeros(3), cov, 20_000)
    b2 = RNG.multivariate_normal([2.0, 0, 0], cov, 20_000)
    assert p2020.colour_separation(a2, b2).mahalanobis == pytest.approx(0.5, abs=0.02)


def test_csf_and_acutance_properties():
    nu = np.linspace(0, 60, 6001)
    peak = nu[np.argmax(cpiq.csf_cpiq(nu))]
    assert peak == pytest.approx(0.8 / 0.2, abs=0.02)  # d/dnu(nu^c e^-b nu) = 0 -> nu = c / b
    view = cpiq.viewing_condition("uhd_24in_0p6m", 2160)
    f = np.linspace(0, 0.5, 256)
    assert cpiq.acutance(f, np.ones_like(f), view) == pytest.approx(1.0)
    a_sharp = cpiq.acutance(f, np.exp(-2 * (np.pi * 0.5 * f) ** 2), view)
    a_soft = cpiq.acutance(f, np.exp(-2 * (np.pi * 2.0 * f) ** 2), view)
    assert 1.0 > a_sharp > a_soft > 0.0
    # same image viewed from further away looks sharper
    far = cpiq.ViewingCondition(2160, 0.2989, 2.0)
    assert cpiq.acutance(f, np.exp(-2 * (np.pi * 2.0 * f) ** 2), far) > a_soft
    with pytest.raises(ValueError):
        cpiq.viewing_condition("nope", 100)


def test_visual_noise_monotonic_and_zero_for_flat():
    view = cpiq.viewing_condition("monitor_100pct", 256)
    flat = np.full((128, 128, 3), 0.5)
    assert cpiq.visual_noise(flat + RNG.normal(0, 1e-9, flat.shape), view) == pytest.approx(0.0, abs=1e-6)
    lum = 0.5 + RNG.normal(0, 0.02, (128, 128, 1)) * np.ones(3)
    lum_hi = 0.5 + RNG.normal(0, 0.04, (128, 128, 1)) * np.ones(3)
    assert 0 < cpiq.visual_noise(lum, view) < cpiq.visual_noise(lum_hi, view)
    # coarse (low-frequency) noise is more visible than the same-variance fine-grain noise at this viewing
    fine = RNG.normal(0, 1, (128, 128))
    coarse = ndimage.zoom(RNG.normal(0, 1, (16, 16)), 8, order=1)
    fine, coarse = (0.5 + 0.03 * n / n.std() for n in (fine, coarse))
    vn_fine = cpiq.visual_noise(fine[..., None] * np.ones(3), view)
    vn_coarse = cpiq.visual_noise(coarse[..., None] * np.ones(3), view)
    assert vn_coarse > vn_fine
    assert cpiq.visual_noise(fine[..., None] * np.ones(3), view, filter_csf=False) > vn_fine


def test_chroma_level_and_colour_uniformity():
    ref = np.array([[50, 40, 20], [60, -30, 50], [40, 10, -45.0]])
    meas = ref * [1, 0.8, 0.8]
    assert cpiq.chroma_level(meas, ref) == pytest.approx(80.0)
    assert cpiq.chroma_level(meas, ref, indices=[0]) == pytest.approx(80.0)
    flat = np.full((90, 120, 3), 0.6)
    assert cpiq.colour_uniformity(flat).max_delta_e00 == pytest.approx(0.0, abs=1e-6)
    yy, xx = np.mgrid[0:90, 0:120]
    tinted = flat.copy()
    tinted[..., 0] *= 1 + 0.15 * np.hypot((xx - 60) / 60, (yy - 45) / 45)  # red colour shading at the corners
    cu = cpiq.colour_uniformity(tinted, grid=(9, 12))
    assert cu.max_delta_uv > 0.005
    assert cu.max_delta_e00 > 2.0
    assert cpiq.lab_from_srgb(np.ones((1, 3)))[0, 0] == pytest.approx(100.0, abs=0.01)


def test_texture_mtf_recovers_gaussian_blur():
    ideal = dead_leaves.dead_leaves(256, r_min=1.0, r_max=60.0, seed=3)
    sigma = 1.2
    captured = ndimage.gaussian_filter(ideal, sigma, mode="wrap")
    f, m = dead_leaves.texture_mtf(captured, ideal)
    expected = np.exp(-2 * (np.pi * sigma * f) ** 2)
    sel = (f > 0.02) & (f < 0.25)
    np.testing.assert_allclose(m[sel], expected[sel], atol=0.06)
    noise = RNG.normal(0, 0.02, ideal.shape)
    _, m_noisy = dead_leaves.texture_mtf(captured + noise, ideal, noise_patch=noise)
    np.testing.assert_allclose(m_noisy[sel], expected[sel], atol=0.08)
    view = cpiq.viewing_condition("monitor_100pct", 256)
    assert dead_leaves.texture_acutance(captured, ideal, view) < dead_leaves.texture_acutance(ideal, ideal, view)
    assert 0.2 < ideal.mean() < 0.8 and ideal.min() >= 0.25


@pytest.mark.parametrize("k1", [0.0, -0.03, 0.03])
def test_grid_distortion_recovers_radial_k1(k1):
    shape = (481, 641)
    img = geometry.dot_grid_image(shape, pitch=40.0, radius=6.0, k1=k1)
    pts = geometry.dot_centroids(img)
    assert len(pts) > 150
    gd = geometry.grid_distortion(pts, shape)
    rn = np.hypot((shape[1] - 1) / 2, (shape[0] - 1) / 2)
    expected = k1 * (gd.field_radius / rn) ** 2 * 100.0
    np.testing.assert_allclose(gd.radial_percent, expected, atol=0.12)
    if k1 == 0.0:
        assert gd.max_local_percent < 0.1


def test_lateral_chromatic_displacement_recovers_shift():
    g = geometry.dot_grid_image((240, 320), pitch=32.0, radius=5.0)
    rgb = np.stack([ndimage.shift(g, (0, 1.5), mode="nearest"), g, ndimage.shift(g, (-0.5, 0), mode="nearest")], -1)
    lcd = geometry.lateral_chromatic_displacement(rgb)
    np.testing.assert_allclose(np.nanmedian(lcd.red_minus_green, axis=0), [1.5, 0.0], atol=0.05)
    np.testing.assert_allclose(np.nanmedian(lcd.blue_minus_green, axis=0), [0.0, -0.5], atol=0.05)
    assert lcd.max_px == pytest.approx(1.5, abs=0.1)
