"""Tests for tools/sfr_analysis.py.

The central claim is that the slanted-edge method recovers a blur it was never
told about, so most of these tests blur a synthetic edge by a known amount and
check the measured MTF against closed-form theory.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import sfr_analysis as sfr  # noqa: E402

THEORY_FREQ = np.linspace(0.0, 2.0, 4001)


def _blurred_edge(sigma_px: float, size: int = 192, angle_deg: float = 5.0) -> np.ndarray:
    return gaussian_filter(sfr.synthetic_slanted_edge(size, size, angle_deg), sigma_px)


class TestSyntheticTargets(unittest.TestCase):
    def test_edge_spans_the_requested_levels(self):
        roi = sfr.synthetic_slanted_edge(64, 64, 5.0, dark=0.05, bright=0.95)
        self.assertAlmostEqual(float(roi.max()), 0.95, places=6)
        self.assertAlmostEqual(float(roi.min()), 0.05, places=6)

    def test_area_sampling_produces_intermediate_edge_pixels(self):
        """Partial pixel coverage is what carries the sub-pixel phase information."""
        roi = sfr.synthetic_slanted_edge(64, 64, 5.0, dark=0.0, bright=1.0)
        interior = roi[(roi > 0.05) & (roi < 0.95)]
        self.assertGreater(interior.size, 0)

    def test_a_vertical_edge_has_no_phase_diversity(self):
        """Every row crosses at the same sub-pixel offset, which is exactly why
        ISO 12233 requires a slant."""
        roi = sfr.synthetic_slanted_edge(64, 64, angle_deg=0.0)
        positions = sfr.row_edge_positions(roi)
        self.assertAlmostEqual(float(np.nanstd(positions)), 0.0, places=9)

    def test_siemens_star_frequency_rises_toward_the_centre(self):
        f_inner = sfr.star_frequency_cy_per_px(10.0, 72)
        f_outer = sfr.star_frequency_cy_per_px(100.0, 72)
        self.assertGreater(f_inner, f_outer)

    def test_star_nyquist_radius_is_where_frequency_equals_one_half(self):
        r = sfr.star_nyquist_radius_px(72)
        self.assertAlmostEqual(float(sfr.star_frequency_cy_per_px(r, 72)), 0.5, places=9)

    def test_siemens_star_is_bounded_and_square(self):
        star = sfr.siemens_star(128, 72)
        self.assertEqual(star.shape, (128, 128))
        self.assertGreaterEqual(float(star.min()), 0.0)
        self.assertLessEqual(float(star.max()), 1.0)


class TestEdgeDetection(unittest.TestCase):
    def test_recovers_the_edge_angle(self):
        for angle in (2.0, 5.0, 8.0, -5.0):
            roi = sfr.synthetic_slanted_edge(160, 160, angle)
            self.assertAlmostEqual(sfr.find_edge_angle(roi), angle, places=1)

    def test_angle_survives_blur(self):
        roi = _blurred_edge(2.0)
        self.assertAlmostEqual(sfr.find_edge_angle(roi), 5.0, places=1)

    def test_row_positions_advance_linearly_down_the_edge(self):
        """Individual rows jitter by the supersampling step on an unblurred edge,
        so the meaningful check is the fitted slope, not each difference."""
        roi = sfr.synthetic_slanted_edge(128, 128, 5.0)
        positions = sfr.row_edge_positions(roi)
        slope = np.polyfit(np.arange(positions.size), positions, 1)[0]
        self.assertAlmostEqual(slope, math.tan(math.radians(5.0)), places=4)

    def test_row_positions_sweep_through_every_sub_pixel_phase(self):
        """The whole method rests on this: over 128 rows a 5-degree edge visits
        the full range of sub-pixel offsets."""
        positions = sfr.row_edge_positions(sfr.synthetic_slanted_edge(128, 128, 5.0))
        phases = np.mod(positions, 1.0)
        self.assertGreater(float(phases.max() - phases.min()), 0.9)

    def test_a_blank_roi_is_rejected(self):
        with self.assertRaises(ValueError):
            sfr.find_edge_angle(np.ones((32, 32)))

    def test_a_bright_border_does_not_drag_the_edge_location(self):
        """A zero-padded convolution leaves a step at the ROI border. A whole-row
        derivative centroid chases it and lands tens of pixels off the real edge,
        which then builds the ESF around the wrong origin."""
        roi = _blurred_edge(1.5, size=128)
        clean = float(np.nanmean(sfr.row_edge_positions(roi)))

        bordered = roi.copy()
        bordered[:, :2] = 0.0
        bordered[:, -2:] = 0.0
        self.assertAlmostEqual(float(np.nanmean(sfr.row_edge_positions(bordered))), clean, delta=0.5)


class TestEsfAndLsf(unittest.TestCase):
    def test_esf_is_oversampled_four_times(self):
        roi = _blurred_edge(1.5)
        _, _, bin_width = sfr.edge_spread_function(roi)
        self.assertAlmostEqual(bin_width, 0.25)

    def test_esf_is_monotonic_from_bright_to_dark(self):
        roi = _blurred_edge(1.5)
        _, esf, _ = sfr.edge_spread_function(roi)
        # Sampled left-to-right the edge runs bright -> dark.
        self.assertGreater(float(esf[: esf.size // 4].mean()), float(esf[-esf.size // 4 :].mean()))

    def test_esf_has_no_gaps(self):
        _, esf, _ = sfr.edge_spread_function(_blurred_edge(1.0))
        self.assertTrue(np.all(np.isfinite(esf)))

    def test_lsf_peaks_at_the_edge(self):
        position, esf, _ = sfr.edge_spread_function(_blurred_edge(1.5))
        lsf = sfr.line_spread_function(esf, window=False)
        peak_position = position[int(np.argmax(np.abs(lsf)))]
        self.assertAlmostEqual(peak_position, 0.0, delta=0.5)

    def test_lsf_width_grows_with_blur(self):
        widths = []
        for sigma in (1.0, 2.0, 3.0):
            position, esf, _ = sfr.edge_spread_function(_blurred_edge(sigma))
            lsf = np.abs(sfr.line_spread_function(esf, window=False))
            lsf = lsf / lsf.sum()
            centre = float((position * lsf).sum())
            widths.append(math.sqrt(float((lsf * (position - centre) ** 2).sum())))
        self.assertEqual(widths, sorted(widths))


class TestMtf(unittest.TestCase):
    def test_mtf_starts_at_unity(self):
        result = sfr.slanted_edge_sfr(_blurred_edge(1.5))
        self.assertAlmostEqual(float(result.mtf[0]), 1.0, places=6)

    def test_mtf_decreases_with_frequency(self):
        result = sfr.slanted_edge_sfr(_blurred_edge(2.0))
        in_band = result.mtf[result.frequency_cy_per_px <= 0.5]
        self.assertTrue(np.all(np.diff(in_band) < 0.02))

    def test_more_blur_means_lower_mtf50(self):
        scores = [sfr.slanted_edge_sfr(_blurred_edge(s)).mtf50_cy_per_px for s in (1.0, 2.0, 3.0)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_measured_mtf50_matches_the_cascaded_system_theory(self):
        """The edge is area-sampled, so the measurement legitimately includes the
        pixel aperture as well as the lens blur -- and matches their product."""
        for sigma in (1.0, 1.5, 2.0, 3.0):
            measured = sfr.slanted_edge_sfr(_blurred_edge(sigma)).mtf50_cy_per_px
            theory = sfr.mtf50(
                THEORY_FREQ,
                sfr.system_mtf(sfr.gaussian_mtf(THEORY_FREQ, sigma), sfr.pixel_aperture_mtf(THEORY_FREQ)),
            )
            self.assertAlmostEqual(measured / theory, 1.0, delta=0.02, msg=f"sigma={sigma}")

    def test_measured_mtf_curve_tracks_theory_across_the_band(self):
        sigma = 2.0
        result = sfr.slanted_edge_sfr(_blurred_edge(sigma))
        in_band = result.frequency_cy_per_px <= 0.5
        theory = sfr.system_mtf(
            sfr.gaussian_mtf(result.frequency_cy_per_px[in_band], sigma),
            sfr.pixel_aperture_mtf(result.frequency_cy_per_px[in_band]),
        )
        np.testing.assert_allclose(result.mtf[in_band], theory, atol=0.02)

    def test_recovered_sigma_matches_the_blur_applied(self):
        for sigma in (1.5, 2.0, 3.0):
            result = sfr.slanted_edge_sfr(_blurred_edge(sigma))
            # Undo the pixel aperture's contribution before inverting the Gaussian.
            self.assertAlmostEqual(sfr.gaussian_sigma_from_mtf50(result.mtf50_cy_per_px) / sigma, 1.0, delta=0.05)

    def test_mtf10_is_beyond_mtf50(self):
        result = sfr.slanted_edge_sfr(_blurred_edge(2.0))
        self.assertGreater(result.mtf10_cy_per_px, result.mtf50_cy_per_px)

    def test_frequency_at_mtf_is_nan_when_the_level_is_never_reached(self):
        freq = np.linspace(0.0, 0.5, 50)
        self.assertTrue(math.isnan(sfr.frequency_at_mtf(freq, np.ones_like(freq), 0.5)))

    def test_differentiation_correction_raises_the_high_frequency_response(self):
        position, esf, bin_width = sfr.edge_spread_function(_blurred_edge(2.0))
        lsf = sfr.line_spread_function(esf)
        _, raw = sfr.mtf_from_lsf(lsf, bin_width, correct_differentiation=False)
        _, corrected = sfr.mtf_from_lsf(lsf, bin_width, correct_differentiation=True)
        self.assertTrue(np.all(corrected[1:] >= raw[1:] - 1e-12))

    def test_a_sharper_edge_resolves_more_at_nyquist(self):
        sharp = sfr.slanted_edge_sfr(_blurred_edge(0.8)).mtf_at_nyquist
        soft = sfr.slanted_edge_sfr(_blurred_edge(2.5)).mtf_at_nyquist
        self.assertGreater(sharp, soft)


class TestTheoreticalMtfs(unittest.TestCase):
    def test_gaussian_mtf_round_trips_through_mtf50(self):
        for sigma in (0.5, 1.0, 2.0, 4.0):
            m50 = sfr.mtf50(THEORY_FREQ, sfr.gaussian_mtf(THEORY_FREQ, sigma))
            self.assertAlmostEqual(sfr.gaussian_sigma_from_mtf50(m50), sigma, places=3)

    def test_diffraction_mtf_falls_to_zero_at_the_cutoff(self):
        cutoff = sfr.diffraction_cutoff_cy_per_px(8.0, 550.0, 4.3)
        self.assertAlmostEqual(float(sfr.diffraction_mtf(np.array([cutoff]), 8.0, 550.0, 4.3)[0]), 0.0, places=9)

    def test_diffraction_mtf_starts_at_unity(self):
        self.assertAlmostEqual(float(sfr.diffraction_mtf(np.array([0.0]), 5.6, 550.0, 4.3)[0]), 1.0, places=9)

    def test_diffraction_cutoff_follows_one_over_lambda_n(self):
        """Stopping down two stops halves the cutoff frequency."""
        a = sfr.diffraction_cutoff_cy_per_px(4.0, 550.0, 4.3)
        b = sfr.diffraction_cutoff_cy_per_px(8.0, 550.0, 4.3)
        self.assertAlmostEqual(a / b, 2.0, places=9)

    def test_small_pixels_push_the_cutoff_below_nyquist_sooner(self):
        big = sfr.diffraction_cutoff_cy_per_px(11.0, 550.0, 5.94)
        small = sfr.diffraction_cutoff_cy_per_px(11.0, 550.0, 1.4)
        self.assertGreater(big, small)

    def test_f22_on_a_phone_pixel_cannot_reach_nyquist(self):
        """The reason phone cameras do not stop down: diffraction alone puts the
        cutoff below the sensor's own Nyquist limit."""
        self.assertLess(sfr.diffraction_cutoff_cy_per_px(22.0, 550.0, 1.4), 0.5)

    def test_pixel_aperture_loses_a_third_of_contrast_at_nyquist(self):
        value = float(sfr.pixel_aperture_mtf(np.array([0.5]))[0])
        self.assertAlmostEqual(value, 2.0 / math.pi, places=6)

    def test_pixel_aperture_is_transparent_at_dc(self):
        self.assertAlmostEqual(float(sfr.pixel_aperture_mtf(np.array([0.0]))[0]), 1.0, places=9)

    def test_system_mtf_is_the_product_of_its_stages(self):
        a = sfr.gaussian_mtf(THEORY_FREQ, 1.0)
        b = sfr.pixel_aperture_mtf(THEORY_FREQ)
        np.testing.assert_allclose(sfr.system_mtf(a, b), a * b, rtol=1e-12)

    def test_cascading_stages_can_only_lose_contrast(self):
        a = sfr.gaussian_mtf(THEORY_FREQ, 1.0)
        b = sfr.pixel_aperture_mtf(THEORY_FREQ)
        self.assertTrue(np.all(sfr.system_mtf(a, b) <= a + 1e-12))

    def test_cycles_per_mm_conversion(self):
        self.assertAlmostEqual(sfr.cycles_per_mm(0.5, 5.0), 100.0, places=9)


def _noisy_edge(angle_deg: float, sigma_px: float, noise: float, seed: int, shape=(240, 80)) -> np.ndarray:
    """The paper's ROI geometry (240 x 80 px) with Gaussian blur and additive noise."""
    edge = sfr.synthetic_slanted_edge(shape[0], shape[1], angle_deg, dark=0.1, bright=0.9)
    rng = np.random.default_rng(seed)
    return gaussian_filter(edge, sigma_px, mode="nearest") + rng.normal(0.0, noise, edge.shape)


def _theory_mtf50(sigma_px: float) -> float:
    return sfr.mtf50(
        THEORY_FREQ, sfr.system_mtf(sfr.gaussian_mtf(THEORY_FREQ, sigma_px), sfr.pixel_aperture_mtf(THEORY_FREQ))
    )


def _legacy_angle(roi: np.ndarray) -> float:
    pos = sfr.row_edge_positions_legacy(roi)
    return math.degrees(math.atan(np.polyfit(np.arange(roi.shape[0]), pos, 1)[0]))


class TestNoisyEdges(unittest.TestCase):
    """ISO 12233-style edge location must survive render / sensor noise."""

    ANGLES = (-8.0, -5.0, 2.0, 5.0)

    def test_angle_is_recovered_under_noise(self):
        for angle in self.ANGLES:
            for seed in range(3):
                roi = _noisy_edge(angle, 1.5, 0.03, seed)
                self.assertAlmostEqual(sfr.find_edge_angle(roi), angle, delta=0.08, msg=f"{angle=} {seed=}")

    def test_mtf50_matches_analytic_gaussian_under_noise(self):
        for sigma in (1.0, 2.0):
            theory = _theory_mtf50(sigma)
            ratios = [sfr.slanted_edge_sfr(_noisy_edge(-5.0, sigma, 0.02, s)).mtf50_cy_per_px / theory for s in range(5)]
            self.assertAlmostEqual(float(np.mean(ratios)), 1.0, delta=0.03, msg=f"{sigma=} {ratios=}")
            self.assertLess(max(abs(r - 1.0) for r in ratios), 0.06, msg=f"{sigma=} {ratios=}")

    def test_legacy_abs_centroid_is_biased_and_the_fix_is_not(self):
        """Regression for the pbrt-render bug: |derivative| centroid -> ~half the slant."""
        roi = _noisy_edge(-5.0, 1.5, 0.03, 0)
        self.assertGreater(_legacy_angle(roi), -3.5)
        self.assertAlmostEqual(sfr.find_edge_angle(roi), -5.0, delta=0.08)

    def test_outlier_rows_are_rejected(self):
        roi = _noisy_edge(5.0, 1.5, 0.01, 0)
        roi[::17, 5] += 50.0  # hot pixels far from the edge
        fit = sfr.fit_edge(roi)
        self.assertAlmostEqual(fit.angle_deg, 5.0, delta=0.08)

    def test_shading_ramp_does_not_bias_the_edge(self):
        roi = _noisy_edge(-5.0, 1.5, 0.01, 0)
        ramp = np.linspace(1.15, 0.85, roi.shape[1])[None, :]
        self.assertAlmostEqual(sfr.find_edge_angle(roi * ramp), -5.0, delta=0.08)

    def test_flatfield_undoes_a_multiplicative_ramp(self):
        sigma = 1.5
        roi = _noisy_edge(-5.0, sigma, 0.0, 0) * np.linspace(1.1, 0.9, 80)[None, :]
        plain = sfr.slanted_edge_sfr(roi).mtf50_cy_per_px
        flat = sfr.slanted_edge_sfr(roi, flatfield=True).mtf50_cy_per_px
        theory = _theory_mtf50(sigma)
        self.assertLess(abs(flat / theory - 1.0), abs(plain / theory - 1.0))
        self.assertAlmostEqual(flat / theory, 1.0, delta=0.03)

    def test_flatfield_is_a_no_op_without_shading(self):
        roi = _blurred_edge(1.5)
        a = sfr.slanted_edge_sfr(roi).mtf50_cy_per_px
        b = sfr.slanted_edge_sfr(roi, flatfield=True).mtf50_cy_per_px
        self.assertAlmostEqual(a, b, delta=0.002)

    def test_polarity_does_not_matter(self):
        roi = _noisy_edge(5.0, 1.5, 0.02, 1)
        self.assertAlmostEqual(sfr.find_edge_angle(roi), sfr.find_edge_angle(1.0 - roi), delta=1e-9)

    def test_quadratic_fit_follows_a_curved_edge(self):
        rows, cols = np.mgrid[0:240, 0:80].astype(float)
        centre = 40.0 + 0.08 * (rows - 120.0) + 2e-4 * (rows - 120.0) ** 2
        roi = gaussian_filter((cols > centre).astype(float), 1.0)
        fit = sfr.fit_edge(roi, fit_order=2)
        self.assertAlmostEqual(float(fit.coefficients[0]), 2e-4, delta=2e-5)

    def test_whole_phase_row_count(self):
        self.assertEqual(sfr._whole_phase_rows(240, 0.0), 240)
        n = sfr._whole_phase_rows(240, math.tan(math.radians(5.0)))
        self.assertAlmostEqual(n * math.tan(math.radians(5.0)), round(n * math.tan(math.radians(5.0))), delta=0.05)


if __name__ == "__main__":
    unittest.main()
