"""Tests for tools/imaging_geometry.py against textbook closed-form results."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import imaging_geometry as ig  # noqa: E402


class TestThinLens(unittest.TestCase):
    def test_object_at_infinity_images_at_the_focal_plane(self):
        self.assertAlmostEqual(ig.thin_lens_image_distance_mm(50.0, math.inf), 50.0)
        self.assertAlmostEqual(ig.magnification(50.0, math.inf), 0.0)

    def test_object_at_twice_focal_length_images_1_to_1(self):
        s_i = ig.thin_lens_image_distance_mm(50.0, 100.0)
        self.assertAlmostEqual(s_i, 100.0)
        self.assertAlmostEqual(ig.magnification(50.0, 100.0), -1.0)

    def test_conjugate_relation_holds(self):
        f, s_o = 35.0, 1200.0
        s_i = ig.thin_lens_image_distance_mm(f, s_o)
        self.assertAlmostEqual(1.0 / s_o + 1.0 / s_i, 1.0 / f, places=12)

    def test_object_inside_the_focal_length_forms_no_real_image(self):
        self.assertEqual(ig.thin_lens_image_distance_mm(50.0, 40.0), math.inf)

    def test_object_distance_for_magnification_round_trips(self):
        for m in (0.1, 0.5, 1.0, 2.0):
            s_o = ig.object_distance_for_magnification_mm(100.0, m)
            self.assertAlmostEqual(abs(ig.magnification(100.0, s_o)), m, places=9)

    def test_life_size_macro_costs_two_stops(self):
        self.assertAlmostEqual(ig.bellows_exposure_factor(1.0), 4.0)
        self.assertAlmostEqual(ig.effective_f_number(2.8, 1.0), 5.6)

    def test_bellows_factor_is_negligible_at_infinity(self):
        self.assertAlmostEqual(ig.bellows_exposure_factor(0.0), 1.0)


class TestFieldOfView(unittest.TestCase):
    def test_50mm_on_full_frame_matches_the_published_figures(self):
        fov = ig.field_of_view(50.0, 36.0, 24.0)
        self.assertAlmostEqual(fov.horizontal_deg, 39.6, places=1)
        self.assertAlmostEqual(fov.vertical_deg, 27.0, places=1)
        self.assertAlmostEqual(fov.diagonal_deg, 46.8, places=1)

    def test_fov_uses_the_infinity_formula_when_focused_at_infinity(self):
        f, w = 24.0, 36.0
        expected = math.degrees(2.0 * math.atan(w / (2.0 * f)))
        self.assertAlmostEqual(ig.field_of_view(f, w, 24.0).horizontal_deg, expected, places=12)

    def test_close_focus_narrows_the_field(self):
        far = ig.field_of_view(100.0, 36.0, 24.0, object_distance_mm=math.inf)
        near = ig.field_of_view(100.0, 36.0, 24.0, object_distance_mm=200.0)
        self.assertLess(near.horizontal_deg, far.horizontal_deg)

    def test_subject_extent_grows_linearly_with_distance(self):
        fov = ig.field_of_view(50.0, 36.0, 24.0).horizontal_deg
        near = ig.subject_extent_mm(fov, 1000.0)
        far = ig.subject_extent_mm(fov, 2000.0)
        self.assertAlmostEqual(far / near, 2.0, places=9)


class TestCropFactor(unittest.TestCase):
    def test_full_frame_has_unit_crop_factor(self):
        ff = ig.sensor_format("full_frame")
        self.assertAlmostEqual(ff.crop_factor, 1.0, places=9)

    def test_aps_c_crop_factor_is_about_1_5(self):
        self.assertAlmostEqual(ig.sensor_format("aps_c").crop_factor, 1.53, places=2)

    def test_micro_four_thirds_crop_factor_is_about_2(self):
        self.assertAlmostEqual(ig.sensor_format("micro_four_thirds").crop_factor, 2.0, places=1)

    def test_equivalent_focal_length_preserves_field_of_view(self):
        """The whole point of '35 mm equivalent': same angle on both formats."""
        apsc = ig.sensor_format("aps_c")
        ff = ig.sensor_format("full_frame")
        f_apsc = 35.0
        f_equiv = ig.equivalent_focal_length_mm(f_apsc, apsc.diagonal_mm)
        self.assertAlmostEqual(
            ig.field_of_view(f_apsc, apsc.width_mm, apsc.height_mm).diagonal_deg,
            ig.field_of_view(f_equiv, ff.width_mm, ff.height_mm).diagonal_deg,
            places=6,
        )

    def test_formats_are_ordered_smallest_first(self):
        diagonals = [f.diagonal_mm for f in ig.SENSOR_FORMATS.values()]
        self.assertEqual(diagonals, sorted(diagonals))

    def test_unknown_format_is_rejected(self):
        with self.assertRaises(ValueError):
            ig.sensor_format("super_8")

    def test_megapixels_from_pitch(self):
        ff = ig.sensor_format("full_frame")
        self.assertAlmostEqual(ff.megapixels(5.94), 24.5, places=1)


class TestDepthOfField(unittest.TestCase):
    def test_full_frame_coc_is_the_familiar_0_03mm(self):
        coc = ig.circle_of_confusion_mm(ig.sensor_format("full_frame").diagonal_mm)
        self.assertAlmostEqual(coc, 0.030, places=3)

    def test_hyperfocal_matches_the_closed_form(self):
        f, N, c = 50.0, 8.0, 0.03
        self.assertAlmostEqual(ig.hyperfocal_distance_mm(f, N, c), f * f / (N * c) + f, places=9)

    def test_focusing_at_hyperfocal_gives_far_limit_infinity_and_near_limit_half(self):
        H = ig.hyperfocal_distance_mm(50.0, 8.0, 0.03)
        dof = ig.depth_of_field(50.0, 8.0, H, 0.03)
        self.assertTrue(dof.is_infinite)
        self.assertAlmostEqual(dof.near_mm, H / 2.0, places=4)

    def test_stopping_down_deepens_depth_of_field(self):
        wide = ig.depth_of_field(50.0, 1.4, 2000.0, 0.03)
        stopped = ig.depth_of_field(50.0, 11.0, 2000.0, 0.03)
        self.assertGreater(stopped.total_mm, wide.total_mm)

    def test_longer_lens_shallows_depth_of_field_at_the_same_distance(self):
        wide = ig.depth_of_field(24.0, 2.8, 3000.0, 0.03)
        tele = ig.depth_of_field(200.0, 2.8, 3000.0, 0.03)
        self.assertLess(tele.total_mm, wide.total_mm)

    def test_depth_of_field_is_asymmetric_with_more_behind_the_subject(self):
        dof = ig.depth_of_field(50.0, 5.6, 3000.0, 0.03)
        self.assertGreater(dof.behind_mm, dof.in_front_mm)

    def test_dof_limits_are_where_defocus_blur_equals_the_coc(self):
        """The near/far limits are a threshold on blur-disc diameter, so the
        blur at each limit must come back to exactly the CoC."""
        f, N, s, coc = 50.0, 4.0, 2000.0, 0.03
        dof = ig.depth_of_field(f, N, s, coc)
        for limit in (dof.near_mm, dof.far_mm):
            self.assertAlmostEqual(
                ig.defocus_blur_diameter_mm(f, N, s, limit), coc, places=6
            )

    def test_blur_is_zero_at_the_plane_of_focus(self):
        self.assertAlmostEqual(ig.defocus_blur_diameter_mm(50.0, 2.0, 1500.0, 1500.0), 0.0)

    def test_pixel_coc_is_stricter_than_the_print_criterion(self):
        ff = ig.sensor_format("full_frame")
        self.assertLess(
            ig.pixel_circle_of_confusion_mm(5.94), ig.circle_of_confusion_mm(ff.diagonal_mm)
        )

    def test_airy_disk_grows_with_f_number(self):
        self.assertLess(ig.airy_disk_diameter_mm(2.8), ig.airy_disk_diameter_mm(16.0))

    def test_diffraction_limited_f_number_is_self_consistent(self):
        coc = 0.03
        N = ig.diffraction_limited_f_number(coc)
        self.assertAlmostEqual(ig.airy_disk_diameter_mm(N), coc, places=9)


class TestRelativeIllumination(unittest.TestCase):
    def test_on_axis_illumination_is_unity(self):
        r, ri = ig.relative_illumination_profile(50.0, 43.27)
        self.assertAlmostEqual(float(ri[0]), 1.0, places=12)

    def test_falloff_is_monotonic_toward_the_corner(self):
        _, ri = ig.relative_illumination_profile(35.0, 43.27)
        self.assertTrue(np.all(np.diff(ri) <= 0))

    def test_matches_cos4_of_the_chief_ray_angle(self):
        f, r_mm = 50.0, 21.6
        ri = float(ig.relative_illumination_cos4(r_mm, f))
        self.assertAlmostEqual(ri, math.cos(math.atan2(r_mm, f)) ** 4, places=12)

    def test_wide_lenses_lose_more_corner_light_than_long_ones(self):
        diag = ig.sensor_format("full_frame").diagonal_mm
        self.assertGreater(ig.corner_falloff_stops(16.0, diag), ig.corner_falloff_stops(85.0, diag))

    def test_corner_falloff_is_reported_in_stops(self):
        diag = ig.sensor_format("full_frame").diagonal_mm
        _, ri = ig.relative_illumination_profile(24.0, diag, n_samples=2)
        self.assertAlmostEqual(
            ig.corner_falloff_stops(24.0, diag), -math.log2(float(ri[-1])), places=12
        )


class TestDistortion(unittest.TestCase):
    def test_zero_coefficients_are_the_identity(self):
        x = np.linspace(-1.0, 1.0, 11)
        y = np.linspace(-1.0, 1.0, 11)
        xd, yd = ig.brown_conrady_distort(x, y, 0.0, 0.0)
        np.testing.assert_allclose(xd, x)
        np.testing.assert_allclose(yd, y)

    def test_negative_k1_is_barrel_and_pulls_points_inward(self):
        xd, _ = ig.brown_conrady_distort(np.array([1.0]), np.array([0.0]), -0.1, 0.0)
        self.assertLess(float(xd[0]), 1.0)

    def test_positive_k1_is_pincushion_and_pushes_points_outward(self):
        xd, _ = ig.brown_conrady_distort(np.array([1.0]), np.array([0.0]), 0.1, 0.0)
        self.assertGreater(float(xd[0]), 1.0)

    def test_distort_then_undistort_round_trips(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(-0.8, 0.8, 200)
        y = rng.uniform(-0.8, 0.8, 200)
        xd, yd = ig.brown_conrady_distort(x, y, -0.12, 0.03, 0.001, -0.002)
        xu, yu = ig.brown_conrady_undistort(xd, yd, -0.12, 0.03, 0.001, -0.002)
        np.testing.assert_allclose(xu, x, atol=1e-7)
        np.testing.assert_allclose(yu, y, atol=1e-7)

    def test_undistort_matches_the_sensor_forward_iteration(self):
        """spectral_sensor_forward.py inlines this exact fixed-point loop; if the
        two ever diverge the demo would be teaching the wrong coefficients."""
        k1, k2, p1, p2 = -0.15, 0.04, 0.002, -0.001
        rng = np.random.default_rng(1)
        xn = rng.uniform(-0.9, 0.9, 64)
        yn = rng.uniform(-0.9, 0.9, 64)

        xu, yu = xn.copy(), yn.copy()
        for _ in range(10):
            r2 = xu**2 + yu**2
            rad = 1.0 + k1 * r2 + k2 * r2**2
            xu = (xn - (2.0 * p1 * xu * yu + p2 * (r2 + 2.0 * xu**2))) / rad
            yu = (yn - (p1 * (r2 + 2.0 * yu**2) + 2.0 * p2 * xu * yu)) / rad

        gx, gy = ig.brown_conrady_undistort(xn, yn, k1, k2, p1, p2, iterations=10)
        np.testing.assert_allclose(gx, xu, rtol=0, atol=0)
        np.testing.assert_allclose(gy, yu, rtol=0, atol=0)

    def test_grid_is_straight_only_when_undistorted(self):
        straight = ig.distortion_grid(0.0, 0.0, n_lines=5, n_samples=21)
        # Each horizontal line must have constant y when there is no distortion.
        for _, ys in straight[:5]:
            self.assertAlmostEqual(float(np.ptp(ys)), 0.0, places=12)

        bowed = ig.distortion_grid(-0.25, 0.0, n_lines=5, n_samples=21)
        self.assertGreater(max(float(np.ptp(ys)) for _, ys in bowed[:5]), 1e-3)

    def test_grid_returns_a_polyline_per_line_in_both_directions(self):
        lines = ig.distortion_grid(-0.1, 0.0, n_lines=7, n_samples=31)
        self.assertEqual(len(lines), 14)
        for xs, ys in lines:
            self.assertEqual(len(xs), 31)
            self.assertEqual(len(ys), 31)

    def test_radial_distortion_percent_sign_follows_k1(self):
        self.assertLess(ig.radial_distortion_percent(-0.05, 0.0), 0.0)
        self.assertGreater(ig.radial_distortion_percent(0.05, 0.0), 0.0)
        self.assertAlmostEqual(ig.radial_distortion_percent(-0.05, 0.0), -5.0, places=9)

    def test_coefficients_read_from_a_lens_config_block(self):
        self.assertEqual(
            ig.distortion_coefficients({"distortion_k1": -0.1, "distortion_k2": 0.02}),
            (-0.1, 0.02, 0.0, 0.0),
        )

    def test_missing_or_null_coefficients_default_to_zero(self):
        self.assertEqual(ig.distortion_coefficients({}), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(ig.distortion_coefficients({"distortion_k1": None}), (0.0, 0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
