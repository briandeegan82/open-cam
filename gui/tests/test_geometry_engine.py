"""Tests for the imaging-geometry adapter.

Closed-form optics is covered by ``tests/test_imaging_geometry.py`` at the repo
root. These tests check that the adapter assembles plot-ready arrays correctly
and that the teaching claims the lecture scenarios make actually hold.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from opencam_gui.core import geometry_engine as ge
from opencam_gui.topics.geometry.scenarios import SCENARIOS


def _summary(**overrides):
    kwargs = dict(
        focal_length_mm=50.0,
        f_number=5.6,
        focus_distance_mm=3000.0,
        format_id="full_frame",
        pixel_pitch_um=5.94,
    )
    kwargs.update(overrides)
    return ge.summarize(**kwargs)


class TestSummary:
    def test_matches_a_published_depth_of_field_calculation(self):
        """50 mm f/5.6 at 3 m on full frame: 2.50 m to 3.74 m, hyperfocal 14.9 m."""
        s = _summary()
        assert s.dof_near_mm / 1000.0 == pytest.approx(2.50, abs=0.02)
        assert s.dof_far_mm / 1000.0 == pytest.approx(3.74, abs=0.02)
        assert s.hyperfocal_mm / 1000.0 == pytest.approx(14.9, abs=0.1)

    def test_full_frame_50mm_is_the_reference_normal_lens(self):
        s = _summary()
        assert s.crop_factor == pytest.approx(1.0, abs=1e-9)
        assert s.equivalent_focal_length_mm == pytest.approx(50.0, abs=1e-9)
        assert s.fov_diagonal_deg == pytest.approx(46.8, abs=0.8)

    def test_pixel_coc_gives_a_shallower_depth_of_field_than_the_print_criterion(self):
        loose = _summary(use_pixel_coc=False)
        strict = _summary(use_pixel_coc=True)
        assert strict.coc_mm < loose.coc_mm
        assert strict.dof_total_mm < loose.dof_total_mm

    def test_aperture_diameter_is_focal_length_over_f_number(self):
        s = _summary(focal_length_mm=85.0, f_number=1.8)
        assert s.aperture_diameter_mm == pytest.approx(85.0 / 1.8)

    def test_unknown_sensor_format_is_rejected(self):
        with pytest.raises(ValueError):
            _summary(format_id="imax")


class TestEquivalence:
    """Crop factor scales the focal length and the depth of field, not the f-number."""

    def test_same_framing_on_two_formats(self):
        mft = _summary(format_id="micro_four_thirds", focal_length_mm=25.0, f_number=2.8, focus_distance_mm=2000.0)
        ff = _summary(format_id="full_frame", focal_length_mm=50.0, f_number=2.8, focus_distance_mm=2000.0)
        assert mft.fov_diagonal_deg == pytest.approx(ff.fov_diagonal_deg, abs=1.0)

    def test_half_the_physical_aperture(self):
        mft = _summary(format_id="micro_four_thirds", focal_length_mm=25.0, f_number=2.8)
        ff = _summary(format_id="full_frame", focal_length_mm=50.0, f_number=2.8)
        assert mft.aperture_diameter_mm == pytest.approx(ff.aperture_diameter_mm / 2.0)

    def test_roughly_twice_the_depth_of_field(self):
        mft = _summary(format_id="micro_four_thirds", focal_length_mm=25.0, f_number=2.8, focus_distance_mm=2000.0)
        ff = _summary(format_id="full_frame", focal_length_mm=50.0, f_number=2.8, focus_distance_mm=2000.0)
        assert mft.dof_total_mm / ff.dof_total_mm == pytest.approx(2.0, rel=0.1)


class TestRayDiagram:
    def test_every_construction_ray_starts_at_the_object_and_ends_at_the_image(self):
        d = ge.ray_diagram(focal_length_mm=50.0, object_distance_mm=200.0, object_height_mm=20.0)
        assert len(d.rays) == 3
        x_obj, y_obj = d.object_arrow[0][0], d.object_arrow[1][1]
        x_img, y_img = d.image_arrow[0][0], d.image_arrow[1][1]
        for xs, ys in d.rays:
            assert xs[0] == pytest.approx(x_obj)
            assert ys[0] == pytest.approx(y_obj)
            assert xs[-1] == pytest.approx(x_img)
            assert ys[-1] == pytest.approx(y_img)

    def test_the_image_is_inverted_relative_to_the_object(self):
        d = ge.ray_diagram(focal_length_mm=50.0, object_distance_mm=200.0, object_height_mm=20.0)
        assert d.object_arrow[1][1] * d.image_arrow[1][1] < 0

    def test_life_size_macro_draws_a_symmetric_diagram(self):
        d = ge.ray_diagram(focal_length_mm=100.0, object_distance_mm=200.0, object_height_mm=20.0)
        assert abs(d.image_arrow[1][1]) == pytest.approx(abs(d.object_arrow[1][1]))

    def test_object_inside_the_focal_length_is_flagged_as_virtual(self):
        d = ge.ray_diagram(focal_length_mm=50.0, object_distance_mm=30.0)
        assert d.image_is_virtual

    def test_a_distant_object_still_produces_a_finite_drawable_diagram(self):
        d = ge.ray_diagram(focal_length_mm=50.0, object_distance_mm=50_000.0)
        assert not d.image_is_virtual
        for xs, ys in d.rays:
            assert all(math.isfinite(v) for v in xs + ys)


class TestFieldOfView:
    def test_fov_falls_monotonically_as_focal_length_grows(self):
        _, fov = ge.fov_vs_focal_length(sensor_width_mm=36.0, sensor_height_mm=24.0)
        assert np.all(np.diff(fov) < 0)

    def test_footprint_is_a_closed_rectangle(self):
        xs, ys = ge.subject_footprint(
            focal_length_mm=50.0, sensor_width_mm=36.0, sensor_height_mm=24.0, distance_mm=3000.0
        )
        assert (xs[0], ys[0]) == (xs[-1], ys[-1])
        assert len(xs) == len(ys) == 5

    def test_footprint_has_the_sensor_aspect_ratio(self):
        xs, ys = ge.subject_footprint(
            focal_length_mm=50.0, sensor_width_mm=36.0, sensor_height_mm=24.0, distance_mm=3000.0
        )
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        assert width / height == pytest.approx(36.0 / 24.0, rel=1e-6)


class TestDepthOfFieldSweeps:
    def test_near_limit_is_always_below_the_far_limit(self):
        sweep = ge.dof_vs_focus_distance(
            focal_length_mm=50.0,
            f_number=5.6,
            coc_mm=0.03,
            min_distance_mm=200.0,
            max_distance_mm=50_000.0,
        )
        assert np.all(sweep.near_m <= sweep.far_m)

    def test_far_limit_runs_away_past_the_hyperfocal_distance(self):
        sweep = ge.dof_vs_focus_distance(
            focal_length_mm=50.0,
            f_number=5.6,
            coc_mm=0.03,
            min_distance_mm=200.0,
            max_distance_mm=200_000.0,
        )
        beyond = sweep.far_m[sweep.focus_distance_m > sweep.hyperfocal_m * 1.2]
        assert beyond.size and np.all(beyond > 100.0)

    def test_blur_dips_to_zero_at_the_plane_of_focus(self):
        d_m, blur = ge.blur_vs_object_distance(
            focal_length_mm=50.0,
            f_number=2.8,
            focus_distance_mm=3000.0,
            min_distance_mm=500.0,
            max_distance_mm=20_000.0,
        )
        assert blur.min() == pytest.approx(0.0, abs=1.0)
        assert d_m[int(np.argmin(blur))] == pytest.approx(3.0, rel=0.05)

    def test_aperture_tradeoff_has_an_interior_optimum(self):
        trade = ge.aperture_tradeoff(focal_length_mm=50.0, focus_distance_mm=3000.0, object_distance_mm=2500.0)
        assert trade.f_numbers[0] < trade.optimum_f_number < trade.f_numbers[-1]

    def test_defocus_falls_and_diffraction_rises_with_f_number(self):
        trade = ge.aperture_tradeoff(focal_length_mm=50.0, focus_distance_mm=3000.0, object_distance_mm=2500.0)
        assert np.all(np.diff(trade.defocus_blur_um) < 0)
        assert np.all(np.diff(trade.diffraction_blur_um) > 0)

    def test_total_is_the_quadrature_sum(self):
        trade = ge.aperture_tradeoff(focal_length_mm=50.0, focus_distance_mm=3000.0, object_distance_mm=2500.0)
        np.testing.assert_allclose(
            trade.total_blur_um,
            np.hypot(trade.defocus_blur_um, trade.diffraction_blur_um),
            rtol=1e-12,
        )


class TestVignettingAndDistortion:
    def test_relative_illumination_is_normalised_to_the_corner(self):
        r, ri = ge.relative_illumination(focal_length_mm=35.0, sensor_diagonal_mm=43.27)
        assert r[0] == pytest.approx(0.0)
        assert r[-1] == pytest.approx(1.0)
        assert ri[0] == pytest.approx(1.0)

    def test_vignette_preview_is_brightest_in_the_middle(self):
        img = ge.vignetting_preview(focal_length_mm=16.0, sensor_width_mm=36.0, sensor_height_mm=24.0, size=41)
        assert img.shape == (41, 41)
        assert img[20, 20] == pytest.approx(img.max())
        assert img[0, 0] < img[20, 20]

    def test_distortion_grid_returns_json_friendly_lists(self):
        lines = ge.distortion_grid(k1=-0.1, k2=0.0)
        assert lines
        for xs, ys in lines:
            assert isinstance(xs, list) and isinstance(ys, list)
            assert len(xs) == len(ys)

    def test_distortion_coefficients_come_from_the_lens_block(self):
        model = {"lens": {"distortion_k1": -0.2, "distortion_k2": 0.05}}
        assert ge.distortion_coefficients(model) == (-0.2, 0.05, 0.0, 0.0)
        assert ge.distortion_coefficients({}) == (0.0, 0.0, 0.0, 0.0)


class TestScenarios:
    @pytest.mark.parametrize("sid", list(SCENARIOS))
    def test_every_scenario_summarises_to_finite_numbers(self, sid):
        sc = SCENARIOS[sid]
        s = ge.summarize(
            focal_length_mm=sc.focal_length_mm,
            f_number=sc.f_number,
            focus_distance_mm=sc.focus_distance_mm,
            format_id=sc.format_id,
            pixel_pitch_um=sc.pixel_pitch_um,
            use_pixel_coc=sc.use_pixel_coc,
            k1=sc.distortion_k1,
            k2=sc.distortion_k2,
        )
        assert math.isfinite(s.fov_diagonal_deg)
        assert math.isfinite(s.hyperfocal_mm)
        assert math.isfinite(s.dof_near_mm)
        assert s.dof_near_mm > 0

    def test_macro_scenario_really_is_life_size(self):
        sc = SCENARIOS["macro_life_size"]
        s = ge.summarize(
            focal_length_mm=sc.focal_length_mm,
            f_number=sc.f_number,
            focus_distance_mm=sc.focus_distance_mm,
            format_id=sc.format_id,
            pixel_pitch_um=sc.pixel_pitch_um,
        )
        assert abs(s.magnification) == pytest.approx(1.0, abs=1e-6)
        assert s.bellows_factor == pytest.approx(4.0, abs=1e-6)
        assert s.effective_f_number == pytest.approx(2.0 * sc.f_number, abs=1e-6)

    def test_phone_scenario_is_a_26mm_equivalent(self):
        sc = SCENARIOS["phone_wide"]
        s = ge.summarize(
            focal_length_mm=sc.focal_length_mm,
            f_number=sc.f_number,
            focus_distance_mm=sc.focus_distance_mm,
            format_id=sc.format_id,
            pixel_pitch_um=sc.pixel_pitch_um,
        )
        assert s.equivalent_focal_length_mm == pytest.approx(26.0, abs=2.0)

    def test_landscape_scenario_reaches_infinity(self):
        sc = SCENARIOS["landscape_hyperfocal"]
        s = ge.summarize(
            focal_length_mm=sc.focal_length_mm,
            f_number=sc.f_number,
            focus_distance_mm=sc.focus_distance_mm,
            format_id=sc.format_id,
            pixel_pitch_um=sc.pixel_pitch_um,
        )
        assert not math.isfinite(s.dof_far_mm)

    def test_portrait_scenario_has_a_depth_of_field_of_a_few_centimetres(self):
        sc = SCENARIOS["portrait_shallow_dof"]
        s = ge.summarize(
            focal_length_mm=sc.focal_length_mm,
            f_number=sc.f_number,
            focus_distance_mm=sc.focus_distance_mm,
            format_id=sc.format_id,
            pixel_pitch_um=sc.pixel_pitch_um,
        )
        assert 10.0 < s.dof_total_mm < 100.0

    def test_barrel_scenario_bows_outward(self):
        sc = SCENARIOS["barrel_distortion"]
        assert sc.distortion_k1 < 0
        s = ge.summarize(
            focal_length_mm=sc.focal_length_mm,
            f_number=sc.f_number,
            focus_distance_mm=sc.focus_distance_mm,
            format_id=sc.format_id,
            pixel_pitch_um=sc.pixel_pitch_um,
            k1=sc.distortion_k1,
            k2=sc.distortion_k2,
        )
        assert s.distortion_percent < 0
