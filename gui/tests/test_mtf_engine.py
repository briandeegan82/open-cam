"""Tests for the resolution / MTF adapter.

The ISO 12233 machinery itself is covered by ``tests/test_sfr_analysis.py``.
What matters here is the round trip the demo actually performs: blur an edge
with the pipeline's own PSF functions, measure it back with the slanted-edge
method, and get the blur you put in. If that closes, every number the demo
shows is a measurement rather than a re-statement of its own input.
"""

from __future__ import annotations

import numpy as np
import pytest

from opencam_gui.core import mtf_engine as me
from opencam_gui.core.repo import import_tool
from opencam_gui.topics.mtf.scenarios import SCENARIOS

PITCH_UM = 4.3


def _psf():
    return import_tool("apply_spectral_psf")


def _gaussian_edge(sigma_px, size=192, angle_deg=5.0):
    """An edge blurred by a Gaussian of known sigma, via the repo's own blur."""
    sfr = import_tool("sfr_analysis")
    margin = max(16, size // 4)
    big = sfr.synthetic_slanted_edge(size + 2 * margin, size + 2 * margin, angle_deg)
    blurred = np.asarray(
        _psf().separable_gaussian_blur_2d(big.astype(np.float32), sigma_px), dtype=np.float64)
    return np.ascontiguousarray(blurred[margin:margin + size, margin:margin + size])


class TestSyntheticEdge:
    def test_the_roi_is_square_and_bounded(self):
        roi = me.synthetic_edge(size=128)
        assert roi.shape == (128, 128)
        assert -0.01 <= roi.min() and roi.max() <= 1.01

    def test_the_edge_is_slanted_rather_than_axis_aligned(self):
        """ISO 12233 needs a slant: it is what supplies the sub-pixel phases the
        oversampled edge profile is built from."""
        roi = me.synthetic_edge(size=192, angle_deg=5.0)
        assert me.measure(roi, PITCH_UM).angle_deg == pytest.approx(5.0, abs=0.5)

    def test_no_second_edge_hides_at_the_border(self):
        """Blurring in place would zero-pad and leave an artificial step at the
        ROI edge, which the SFR would happily measure instead of the real one."""
        roi = me.synthetic_edge(size=192, sigma_geometric_px=3.0)
        assert roi[:, 0].mean() == pytest.approx(roi[:, 4].mean(), abs=0.02)
        assert roi[:, -1].mean() == pytest.approx(roi[:, -5].mean(), abs=0.02)

    def test_both_psf_modes_produce_a_measurable_edge(self):
        for mode in ("chromatic_gaussian", "airy_disk"):
            m = me.measure(me.synthetic_edge(mode=mode), PITCH_UM)
            assert 0.0 < m.mtf50_cy_per_px < 2.0

    def test_a_sharp_lens_can_out_resolve_the_pixel(self):
        """The slanted edge measures the *pre-sampling* MTF: the slant supplies
        sub-pixel phases, so the method sees past Nyquist. Contrast still sitting
        above 50 percent at 0.5 cy/px is not a measurement error, it is the
        condition for aliasing -- which is what the aliasing tab then shows."""
        m = me.measure(me.synthetic_edge(sigma_geometric_px=0.1, f_number=2.0), PITCH_UM)
        assert m.mtf50_cy_per_px > me.NYQUIST_CY_PER_PX
        assert m.mtf_at_nyquist > 0.5


class TestGaussianRoundTrip:
    """Blur by a known sigma, measure, and recover it. A Gaussian's MTF is
    exp(-2 pi^2 sigma^2 f^2), so MTF50 sits at sqrt(ln 2 / 2) / (pi sigma)."""

    @staticmethod
    def _expected_mtf50(sigma_px):
        return np.sqrt(np.log(2.0) / 2.0) / (np.pi * sigma_px)

    @pytest.mark.parametrize("sigma", [0.8, 1.2, 1.8, 2.5])
    def test_mtf50_recovers_the_sigma_that_went_in(self, sigma):
        measured = me.measure(_gaussian_edge(sigma), PITCH_UM).mtf50_cy_per_px
        assert measured == pytest.approx(self._expected_mtf50(sigma), rel=0.08)

    def test_the_whole_curve_matches_the_analytic_gaussian_mtf(self):
        sigma = 1.5
        m = me.measure(_gaussian_edge(sigma, size=256), PITCH_UM)
        band = m.frequency_cy_per_px <= 0.4
        expected = np.exp(-2.0 * np.pi**2 * sigma**2 * m.frequency_cy_per_px[band] ** 2)
        assert np.max(np.abs(m.mtf[band] - expected)) < 0.05

    def test_more_blur_is_less_resolution(self):
        sharp = me.measure(_gaussian_edge(0.6), PITCH_UM)
        soft = me.measure(_gaussian_edge(2.4), PITCH_UM)
        assert soft.mtf50_cy_per_px < sharp.mtf50_cy_per_px
        assert soft.mtf_at_nyquist < sharp.mtf_at_nyquist

    def test_the_measurement_does_not_depend_on_the_slant_angle(self):
        """The angle is a sampling trick, not a property of the system."""
        values = [me.measure(_gaussian_edge(1.2, angle_deg=a), PITCH_UM).mtf50_cy_per_px
                  for a in (3.0, 5.0, 8.0, 12.0)]
        assert np.ptp(values) < 0.05 * np.mean(values)

    def test_a_bigger_roi_does_not_move_the_answer(self):
        a = me.measure(_gaussian_edge(1.2, size=128), PITCH_UM).mtf50_cy_per_px
        b = me.measure(_gaussian_edge(1.2, size=256), PITCH_UM).mtf50_cy_per_px
        assert a == pytest.approx(b, rel=0.05)


class TestMeasurementShape:
    def test_the_esf_lsf_and_mtf_are_all_populated(self):
        m = me.measure(me.synthetic_edge(), PITCH_UM)
        assert m.esf.size == m.esf_position_px.size > 0
        assert m.lsf.size == m.lsf_position_px.size > 0
        assert m.mtf.size == m.frequency_cy_per_px.size > 0

    def test_the_mtf_starts_at_unity_and_never_exceeds_it(self):
        m = me.measure(me.synthetic_edge(), PITCH_UM)
        assert m.mtf[0] == pytest.approx(1.0, abs=1e-6)
        assert m.mtf.max() <= 1.0 + 1e-6

    def test_the_lsf_peaks_near_the_middle_of_its_support(self):
        """The LSF is the derivative of the edge profile, so it should be centred
        on the edge rather than pinned against one end."""
        m = me.measure(me.synthetic_edge(size=192), PITCH_UM)
        peak = m.lsf_position_px[int(np.argmax(np.abs(m.lsf)))]
        span = np.ptp(m.lsf_position_px)
        assert abs(peak - np.mean(m.lsf_position_px)) < 0.1 * span

    def test_mtf10_sits_above_mtf50(self):
        m = me.measure(me.synthetic_edge(), PITCH_UM)
        assert m.mtf10_cy_per_px > m.mtf50_cy_per_px

    def test_cycles_per_mm_is_cycles_per_pixel_scaled_by_the_pitch(self):
        """The same optics on a finer pixel is the same lens: cycles/mm is a
        property of the lens, cycles/px of the pair."""
        m = me.measure(me.synthetic_edge(), PITCH_UM)
        assert m.mtf50_cy_per_mm == pytest.approx(
            m.mtf50_cy_per_px * 1000.0 / PITCH_UM, rel=1e-9)


class TestDiffractionTheory:
    def test_the_cutoff_follows_one_over_lambda_n(self):
        """The hard limit of an aberration-free lens, and the only part of an MTF
        curve that is pure physics."""
        pitch_mm = PITCH_UM / 1000.0
        for f_number in (4.0, 8.0, 16.0):
            curves = me.theory_curves(f_number=f_number, pixel_pitch_um=PITCH_UM)
            cutoff_cy_per_mm = curves.diffraction_cutoff_cy_per_px / pitch_mm
            assert cutoff_cy_per_mm == pytest.approx(1.0 / (550e-6 * f_number), rel=1e-6)

    def test_stopping_down_lowers_the_cutoff(self):
        wide = me.theory_curves(f_number=2.8, pixel_pitch_um=PITCH_UM)
        narrow = me.theory_curves(f_number=16.0, pixel_pitch_um=PITCH_UM)
        assert narrow.diffraction_cutoff_cy_per_px < wide.diffraction_cutoff_cy_per_px

    def test_the_diffraction_curve_is_monotone_and_reaches_zero_at_the_cutoff(self):
        curves = me.theory_curves(f_number=16.0, pixel_pitch_um=PITCH_UM)
        assert np.all(np.diff(curves.diffraction) <= 1e-12)
        past = curves.frequency_cy_per_px >= curves.diffraction_cutoff_cy_per_px
        assert np.all(curves.diffraction[past] < 1e-9)

    def test_the_pixel_aperture_first_nulls_at_one_cycle_per_pixel(self):
        """A square pixel integrates over its own width, so its MTF is a sinc
        with its first zero where one cycle fits in one pixel."""
        curves = me.theory_curves(f_number=4.0, pixel_pitch_um=PITCH_UM)
        at_one = np.interp(1.0, curves.frequency_cy_per_px, curves.pixel_aperture)
        assert at_one == pytest.approx(0.0, abs=1e-6)
        at_nyquist = np.interp(0.5, curves.frequency_cy_per_px, curves.pixel_aperture)
        assert at_nyquist == pytest.approx(2.0 / np.pi, rel=1e-3)

    def test_the_system_is_the_product_of_its_stages(self):
        curves = me.theory_curves(f_number=8.0, pixel_pitch_um=PITCH_UM)
        assert np.allclose(curves.system, curves.diffraction * curves.pixel_aperture)

    def test_the_system_is_never_better_than_its_worst_stage(self):
        curves = me.theory_curves(f_number=8.0, pixel_pitch_um=PITCH_UM)
        assert np.all(curves.system <= curves.diffraction + 1e-12)
        assert np.all(curves.system <= curves.pixel_aperture + 1e-12)

    def test_a_measured_airy_edge_tracks_the_diffraction_theory(self):
        """The point of the overlay: measurement and theory on the same axes, and
        they agree."""
        roi = me.synthetic_edge(size=256, mode="airy_disk", f_number=16.0,
                                pixel_pitch_um=PITCH_UM, sigma_geometric_px=0.0)
        measured = me.measure(roi, PITCH_UM)
        curves = me.theory_curves(f_number=16.0, pixel_pitch_um=PITCH_UM)

        band = (measured.frequency_cy_per_px > 0.02) & (measured.frequency_cy_per_px < 0.3)
        theory = np.interp(measured.frequency_cy_per_px[band],
                           curves.frequency_cy_per_px, curves.diffraction)
        assert np.max(np.abs(measured.mtf[band] - theory)) < 0.1

    def test_tiny_pixels_put_the_diffraction_cutoff_below_nyquist(self):
        """The phone-camera scenario: at f/11 on a 1.4 um pixel the lens gives up
        before the sensor does, so stopping down can only cost resolution."""
        curves = me.theory_curves(f_number=11.0, pixel_pitch_um=1.4)
        assert curves.diffraction_cutoff_cy_per_px < me.NYQUIST_CY_PER_PX


class TestAliasing:
    def test_sampling_coarsely_shrinks_the_image_by_the_factor(self):
        preview = me.aliasing_preview(size=256, downsample=4)
        assert preview.reference.shape == (256, 256)
        assert preview.sampled.shape == (64, 64)

    def test_the_star_aliases_without_a_prefilter(self):
        """Inside the Nyquist radius the spokes are finer than the grid, and a
        point sampler folds them back rather than losing them."""
        preview = me.aliasing_preview(size=256, downsample=4, prefilter_sigma_px=0.0)
        assert _centre_contrast(preview) > 0.25

    def test_a_prefilter_removes_the_moire_before_it_is_sampled(self):
        raw = me.aliasing_preview(size=256, downsample=4, prefilter_sigma_px=0.0)
        filtered = me.aliasing_preview(size=256, downsample=4, prefilter_sigma_px=3.0)
        assert _centre_contrast(filtered) < 0.5 * _centre_contrast(raw)

    def test_the_prefilter_leaves_the_coarse_spokes_alone(self):
        """The trade is local: an OLPF costs detail near Nyquist, not everywhere."""
        raw = me.aliasing_preview(size=256, downsample=4, prefilter_sigma_px=3.0)
        edge = raw.sampled[:8, :]
        assert np.ptp(edge) > 0.5

    def test_the_reference_is_never_the_prefiltered_image(self):
        """The 'before' picture has to stay the honest one for the comparison to
        mean anything."""
        a = me.aliasing_preview(size=128, prefilter_sigma_px=0.0)
        b = me.aliasing_preview(size=128, prefilter_sigma_px=3.0)
        assert np.array_equal(a.reference, b.reference)

    def test_more_spokes_bring_the_nyquist_radius_further_out(self):
        few = me.aliasing_preview(size=256, spokes=36)
        many = me.aliasing_preview(size=256, spokes=144)
        assert many.nyquist_radius_px > few.nyquist_radius_px

    def test_upsampling_is_nearest_neighbour_so_the_samples_stay_visible(self):
        arr = np.array([[0.0, 1.0], [1.0, 0.0]])
        up = me.upsample_nearest(arr, 3)
        assert up.shape == (6, 6)
        assert set(np.unique(up)) == {0.0, 1.0}


def _centre_contrast(preview):
    """Local contrast inside the Nyquist radius, where aliasing lives."""
    sampled = preview.sampled
    r = max(int(preview.nyquist_radius_px / preview.downsample), 4)
    c = sampled.shape[0] // 2
    patch = sampled[max(c - r, 0):c + r, max(c - r, 0):c + r]
    return float(np.abs(np.diff(patch, axis=1)).mean())


class TestRenderedSource:
    def test_missing_renders_are_reported_rather_than_raising(self):
        """A student without a PBRT build still gets the synthetic path."""
        assert isinstance(me.rendered_edge_candidates(), list)


class TestScenarios:
    @pytest.mark.parametrize("sid", list(SCENARIOS))
    def test_every_scenario_measures_something_sensible(self, sid):
        sc = SCENARIOS[sid]
        roi = me.synthetic_edge(mode=sc.mode, f_number=sc.f_number,
                                pixel_pitch_um=sc.pixel_pitch_um,
                                sigma_geometric_px=sc.sigma_geometric_px,
                                angle_deg=sc.edge_angle_deg)
        m = me.measure(roi, sc.pixel_pitch_um)
        assert 0.0 < m.mtf50_cy_per_px < 2.0
        assert np.all(np.isfinite(m.mtf))
        assert m.mtf50_cy_per_px < m.mtf10_cy_per_px

    def test_the_soft_lens_really_is_softer_than_the_reference(self):
        def mtf50(sid):
            sc = SCENARIOS[sid]
            return me.measure(me.synthetic_edge(
                mode=sc.mode, f_number=sc.f_number, pixel_pitch_um=sc.pixel_pitch_um,
                sigma_geometric_px=sc.sigma_geometric_px), sc.pixel_pitch_um).mtf50_cy_per_px

        assert mtf50("soft_lens") < 0.5 * mtf50("sharp_reference")

    def test_the_f16_scenario_is_actually_diffraction_limited(self):
        """Its claim is that the cutoff is set by the aperture, not the aberration
        term, so the measured MTF50 should land near the diffraction prediction."""
        sc = SCENARIOS["diffraction_limited_f16"]
        m = me.measure(me.synthetic_edge(
            size=256, mode=sc.mode, f_number=sc.f_number,
            pixel_pitch_um=sc.pixel_pitch_um,
            sigma_geometric_px=sc.sigma_geometric_px), sc.pixel_pitch_um)
        cutoff = me.theory_curves(
            f_number=sc.f_number,
            pixel_pitch_um=sc.pixel_pitch_um).diffraction_cutoff_cy_per_px
        # The diffraction MTF crosses 0.5 at roughly 0.4 of its cutoff.
        assert m.mtf50_cy_per_px == pytest.approx(0.4 * cutoff, rel=0.25)

    def test_the_olpf_scenario_differs_from_the_unfiltered_one(self):
        raw, olpf = SCENARIOS["aliasing_no_filter"], SCENARIOS["aliasing_with_olpf"]
        assert olpf.prefilter_sigma_px > raw.prefilter_sigma_px == 0.0

    def test_scenarios_only_name_real_psf_modes(self):
        for sc in SCENARIOS.values():
            assert sc.mode in ("chromatic_gaussian", "airy_disk"), sc.id
