"""Tests for tools/colour_science.py.

Colour code fails quietly -- a wrong matrix still produces a plausible picture --
so these check against external ground truth wherever one exists: published
illuminant chromaticities, the X-Rite ColorChecker's own sRGB values, and the
CIEDE2000 reference vectors from Sharma, Wu & Dalal (2005).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import colour_science as cs  # noqa: E402

WL = np.arange(380.0, 731.0, 1.0)


class TestObserver(unittest.TestCase):
    def test_cmf_resamples_to_the_requested_grid(self):
        cmf = cs.cmf_on_grid(WL)
        self.assertEqual(cmf.shape, (3, WL.size))

    def test_the_luminous_efficiency_function_peaks_at_555_nm(self):
        """ybar is V(lambda) by construction, and V peaks at 555 nm."""
        cmf = cs.cmf_on_grid(WL)
        self.assertAlmostEqual(float(WL[np.argmax(cmf[1])]), 555.0, delta=1.0)
        self.assertAlmostEqual(float(cmf[1].max()), 1.0, places=3)

    def test_cmfs_are_zero_outside_the_visible_range(self):
        cmf = cs.cmf_on_grid(np.array([200.0, 300.0, 900.0, 1200.0]))
        self.assertTrue(np.allclose(cmf, 0.0))

    def test_equal_energy_white_sits_at_the_e_illuminant_point(self):
        """A flat spectrum must land at x = y = 1/3. If the CMFs were mis-scaled
        relative to each other this is the first thing to move."""
        x, y, z = cs.tristimulus(WL, np.ones_like(WL))
        xy = cs.xy_chromaticity(np.array([[x, y, z]]))[0]
        self.assertAlmostEqual(float(xy[0]), 1.0 / 3.0, places=2)
        self.assertAlmostEqual(float(xy[1]), 1.0 / 3.0, places=2)


class TestIlluminants(unittest.TestCase):
    def test_all_eighteen_illuminants_are_present(self):
        self.assertEqual(len(cs.list_illuminant_ids(REPO)), 18)

    def test_chromaticities_match_the_published_values(self):
        for name, x, y in (("D65", 0.3127, 0.3290), ("D50", 0.3457, 0.3585),
                           ("A", 0.4476, 0.4074), ("D75", 0.2990, 0.3149)):
            _, spd = cs.load_illuminant(REPO, name, WL)
            xy = cs.xy_chromaticity(cs.white_point_xyz(WL, spd)[None, :])[0]
            self.assertAlmostEqual(float(xy[0]), x, delta=0.003, msg=name)
            self.assertAlmostEqual(float(xy[1]), y, delta=0.003, msg=name)

    def test_correlated_colour_temperatures_match_their_names(self):
        for name, cct in (("D65", 6504), ("D50", 5003), ("A", 2856), ("D75", 7504)):
            _, spd = cs.load_illuminant(REPO, name, WL)
            xy = cs.xy_chromaticity(cs.white_point_xyz(WL, spd)[None, :])[0]
            got = cs.correlated_colour_temperature(xy)
            self.assertAlmostEqual(got, cct, delta=0.01 * cct, msg=name)

    def test_an_unknown_illuminant_lists_the_real_ones(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            cs.load_illuminant(REPO, "sunshine")
        self.assertIn("D65", str(ctx.exception))

    def test_white_point_is_normalised_to_unit_luminance(self):
        _, spd = cs.load_illuminant(REPO, "D65", WL)
        self.assertAlmostEqual(float(cs.white_point_xyz(WL, spd)[1]), 1.0, places=9)


class TestColorChecker(unittest.TestCase):
    def test_loads_all_twenty_four_patches(self):
        chart = cs.load_colorchecker(REPO, WL)
        self.assertEqual(chart.reflectance.shape, (24, WL.size))
        self.assertEqual(len(chart.names), 24)

    def test_reflectance_is_physically_bounded(self):
        chart = cs.load_colorchecker(REPO, WL)
        self.assertGreaterEqual(float(chart.reflectance.min()), 0.0)
        self.assertLessEqual(float(chart.reflectance.max()), 1.0)

    def test_the_neutral_ladder_descends_in_luminance(self):
        """Patches 19-24 are the grey ramp, white through black."""
        chart = cs.load_colorchecker(REPO, WL)
        _, spd = cs.load_illuminant(REPO, "D65", WL)
        y = cs.xyz_from_spectra(WL, chart.reflectance, spd)[cs.COLORCHECKER_NEUTRAL_SLICE, 1]
        self.assertTrue(np.all(np.diff(y) < 0), msg=str(y))

    def test_rendering_under_d65_matches_the_published_srgb_values(self):
        """End-to-end: reflectance, illuminant, observer, XYZ, sRGB primaries and
        the transfer function all have to be right for this to land."""
        chart = cs.load_colorchecker(REPO, WL)
        _, spd = cs.load_illuminant(REPO, "D65", WL)
        srgb = np.clip(cs.xyz_to_srgb_linear(
            cs.xyz_from_spectra(WL, chart.reflectance, spd)), 0.0, 1.0) ** (1 / 2.2)

        published = np.array([
            [115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67],
            [133, 128, 177], [103, 189, 170], [214, 126, 44], [80, 91, 166],
            [193, 90, 99], [94, 60, 108], [157, 188, 64], [224, 163, 46],
            [56, 61, 150], [70, 148, 73], [175, 54, 60], [231, 199, 31],
            [187, 86, 149], [8, 133, 161], [243, 243, 242], [200, 200, 200],
            [160, 160, 160], [122, 122, 121], [85, 85, 85], [52, 52, 52],
        ], dtype=np.float64)

        error = np.abs(srgb * 255.0 - published)
        self.assertLess(float(error.mean()), 4.0)
        self.assertLess(float(error.max()), 16.0)


class TestSpectralIntegration(unittest.TestCase):
    def test_mismatched_grids_are_rejected_rather_than_broadcast(self):
        with self.assertRaises(ValueError):
            cs.integrate_spectra(WL, np.ones((2, WL.size)), np.ones((3, WL.size - 1)))

    def test_radiance_is_reflectance_times_illuminant(self):
        refl = np.array([[0.5, 0.25]])
        illum = np.array([2.0, 4.0])
        self.assertTrue(np.allclose(cs.radiance_spectra(refl, illum), [[1.0, 1.0]]))

    def test_a_perfect_white_diffuser_has_unit_luminance(self):
        _, spd = cs.load_illuminant(REPO, "D65", WL)
        xyz = cs.xyz_from_spectra(WL, np.ones((1, WL.size)), spd)
        self.assertAlmostEqual(float(xyz[0, 1]), 1.0, places=6)

    def test_doubling_the_light_doubles_the_response(self):
        _, spd = cs.load_illuminant(REPO, "D65", WL)
        refl = cs.load_colorchecker(REPO, WL).reflectance
        qe = cs.cmf_on_grid(WL)
        single = cs.camera_rgb_from_spectra(WL, refl, spd, qe, normalise=False)
        double = cs.camera_rgb_from_spectra(WL, refl, 2.0 * spd, qe, normalise=False)
        self.assertTrue(np.allclose(double, 2.0 * single))

    def test_normalising_puts_a_white_diffuser_at_one_in_every_channel(self):
        _, spd = cs.load_illuminant(REPO, "A", WL)
        qe = cs.cmf_on_grid(WL)
        rgb = cs.camera_rgb_from_spectra(WL, np.ones((1, WL.size)), spd, qe)
        self.assertTrue(np.allclose(rgb, 1.0, atol=1e-9))


class TestLutherCondition(unittest.TestCase):
    def test_a_camera_that_is_the_observer_is_perfectly_colorimetric(self):
        self.assertAlmostEqual(cs.luther_condition_error(WL, cs.cmf_on_grid(WL)), 0.0, places=9)

    def test_any_linear_mix_of_the_cmfs_also_satisfies_it(self):
        """The condition is on the span of the curves, not the curves themselves --
        which is why a CCM can exist at all."""
        mix = np.array([[1.0, 0.2, 0.0], [0.0, 1.0, 0.1], [0.1, 0.0, 1.0]])
        self.assertAlmostEqual(
            cs.luther_condition_error(WL, mix @ cs.cmf_on_grid(WL)), 0.0, places=9)

    def test_a_real_camera_misses_it(self):
        qe = cs.load_qe_rgb(REPO, {
            "red": "spectra/QE/interpolated/QE_red.csv",
            "green": "spectra/QE/interpolated/QE_green.csv",
            "blue": "spectra/QE/interpolated/QE_blue.csv",
        }, WL)
        self.assertGreater(cs.luther_condition_error(WL, qe), 0.05)


class TestColourSpaces(unittest.TestCase):
    def test_xyz_and_srgb_matrices_are_inverses(self):
        rgb = np.array([[0.2, 0.5, 0.9], [1.0, 1.0, 1.0], [0.0, 0.3, 0.1]])
        self.assertTrue(np.allclose(cs.xyz_to_srgb_linear(cs.srgb_linear_to_xyz(rgb)), rgb))

    def test_srgb_white_is_the_d65_white_point(self):
        xyz = cs.srgb_linear_to_xyz(np.array([[1.0, 1.0, 1.0]]))[0]
        self.assertTrue(np.allclose(xyz, cs.WHITE_D65, atol=1e-4), msg=str(xyz))

    def test_lab_of_the_white_point_is_pure_lightness(self):
        lab = cs.xyz_to_lab(cs.WHITE_D65[None, :])[0]
        self.assertAlmostEqual(float(lab[0]), 100.0, places=6)
        self.assertAlmostEqual(float(lab[1]), 0.0, places=6)
        self.assertAlmostEqual(float(lab[2]), 0.0, places=6)

    def test_lab_of_black_is_zero(self):
        lab = cs.xyz_to_lab(np.zeros((1, 3)))[0]
        self.assertAlmostEqual(float(lab[0]), 0.0, places=9)

    def test_middle_grey_is_about_lightness_fifty(self):
        lab = cs.xyz_to_lab(0.184 * cs.WHITE_D65[None, :])[0]
        self.assertAlmostEqual(float(lab[0]), 50.0, delta=0.5)

    def test_chromatic_adaptation_maps_one_white_onto_the_other(self):
        _, a = cs.load_illuminant(REPO, "A", WL)
        white_a = cs.white_point_xyz(WL, a)
        adapted = cs.chromatic_adaptation_matrix(white_a, cs.WHITE_D65) @ white_a
        self.assertTrue(np.allclose(adapted, cs.WHITE_D65, atol=1e-6), msg=str(adapted))

    def test_adapting_a_white_to_itself_is_the_identity(self):
        m = cs.chromatic_adaptation_matrix(cs.WHITE_D65, cs.WHITE_D65)
        self.assertTrue(np.allclose(m, np.eye(3), atol=1e-9))

    def test_chromaticity_ignores_intensity(self):
        xyz = np.array([[0.2, 0.3, 0.5]])
        self.assertTrue(np.allclose(cs.xy_chromaticity(xyz), cs.xy_chromaticity(7.0 * xyz)))


class TestColourDifference(unittest.TestCase):
    #: Sharma, Wu & Dalal (2005), the reference implementation's test vectors.
    SHARMA = [
        ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485), 2.0425),
        ((50.0000, 3.1571, -77.2803), (50.0000, 0.0000, -82.7485), 2.8615),
        ((50.0000, 2.8361, -74.0200), (50.0000, 0.0000, -82.7485), 3.4412),
        ((50.0000, -1.3802, -84.2814), (50.0000, 0.0000, -82.7485), 1.0000),
        ((50.0000, -1.1848, -84.8006), (50.0000, 0.0000, -82.7485), 1.0000),
        ((50.0000, -0.9009, -85.5211), (50.0000, 0.0000, -82.7485), 1.0000),
        ((50.0000, 0.0000, 0.0000), (50.0000, -1.0000, 2.0000), 2.3669),
        ((50.0000, -1.0000, 2.0000), (50.0000, 0.0000, 0.0000), 2.3669),
        ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0009), 7.1792),
        ((50.0000, 2.5000, 0.0000), (50.0000, 0.0000, -2.5000), 4.3065),
        ((50.0000, 2.5000, 0.0000), (73.0000, 25.0000, -18.0000), 27.1492),
        ((50.0000, 2.5000, 0.0000), (50.0000, 3.1736, 0.5854), 1.0000),
        ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
        ((63.0109, -31.0961, -5.8663), (62.8187, -29.7946, -4.0864), 1.2630),
        ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
        ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
    ]

    def test_matches_the_ciede2000_reference_vectors(self):
        a = np.array([t[0] for t in self.SHARMA])
        b = np.array([t[1] for t in self.SHARMA])
        want = np.array([t[2] for t in self.SHARMA])
        self.assertTrue(np.allclose(cs.delta_e_2000(a, b), want, atol=1e-4))

    def test_is_symmetric(self):
        a = np.array([t[0] for t in self.SHARMA])
        b = np.array([t[1] for t in self.SHARMA])
        self.assertTrue(np.allclose(cs.delta_e_2000(a, b), cs.delta_e_2000(b, a)))

    def test_identical_colours_have_zero_difference(self):
        a = np.array([[50.0, 2.5, -3.0]])
        self.assertAlmostEqual(float(cs.delta_e_2000(a, a)[0]), 0.0, places=9)
        self.assertAlmostEqual(float(cs.delta_e_76(a, a)[0]), 0.0, places=9)

    def test_delta_e_76_is_plain_euclidean_distance(self):
        a, b = np.array([[50.0, 0.0, 0.0]]), np.array([[53.0, 4.0, 0.0]])
        self.assertAlmostEqual(float(cs.delta_e_76(a, b)[0]), 5.0, places=9)

    def test_the_two_metrics_disagree_on_saturated_blues(self):
        """The case CIEDE2000 was introduced to fix: CIE76 badly overstates it."""
        a, b = np.array([[50.0, 2.6772, -79.7751]]), np.array([[50.0, 0.0, -82.7485]])
        self.assertGreater(float(cs.delta_e_76(a, b)[0]), 1.8 * float(cs.delta_e_2000(a, b)[0]))


if __name__ == "__main__":
    unittest.main()
