"""Tests for the photometric exposure chain in tools/sensor_radiometry.py.

The exposure triangle is only a teaching aid if the numbers behind it are real,
so these check the chain against the rules photographers actually use: sunny 16,
reciprocity, and the stop as a factor of two.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import sensor_radiometry as sr  # noqa: E402

PIXEL = dict(pixel_pitch_um=4.3, quantum_efficiency=0.6)


class TestCameraEquation(unittest.TestCase):
    def test_illuminance_falls_as_the_square_of_the_f_number(self):
        a = sr.image_plane_illuminance_lux(1000.0, 4.0)
        b = sr.image_plane_illuminance_lux(1000.0, 8.0)
        self.assertAlmostEqual(float(a / b), 4.0, places=9)

    def test_one_stop_of_aperture_halves_the_light(self):
        wide = sr.image_plane_illuminance_lux(1000.0, 4.0)
        stopped = sr.image_plane_illuminance_lux(1000.0, 4.0 * math.sqrt(2.0))
        self.assertAlmostEqual(float(wide / stopped), 2.0, places=9)

    def test_matches_the_closed_form(self):
        got = sr.image_plane_illuminance_lux(500.0, 2.8, transmission=0.85)
        want = (math.pi / 4.0) * 0.85 * 500.0 / (2.8 ** 2)
        self.assertAlmostEqual(float(got), want, places=9)

    def test_vignetting_scales_illuminance_directly(self):
        centre = sr.image_plane_illuminance_lux(1000.0, 4.0)
        corner = sr.image_plane_illuminance_lux(1000.0, 4.0, relative_illumination=0.5)
        self.assertAlmostEqual(float(corner / centre), 0.5, places=9)

    def test_a_non_positive_aperture_is_rejected(self):
        with self.assertRaises(ValueError):
            sr.image_plane_illuminance_lux(1000.0, 0.0)


class TestPhotonsAndElectrons(unittest.TestCase):
    def test_photon_rate_is_linear_in_illuminance_and_area(self):
        base = sr.photons_per_second_per_pixel(100.0, 1e-11)
        self.assertAlmostEqual(
            float(sr.photons_per_second_per_pixel(200.0, 1e-11)), 2.0 * float(base), places=6)
        self.assertAlmostEqual(
            float(sr.photons_per_second_per_pixel(100.0, 2e-11)), 2.0 * float(base), places=6)

    def test_bluer_photons_are_more_energetic_so_fewer_arrive(self):
        red = sr.photons_per_second_per_pixel(100.0, 1e-11, wavelength_nm=650.0)
        blue = sr.photons_per_second_per_pixel(100.0, 1e-11, wavelength_nm=450.0)
        self.assertGreater(float(red), float(blue))
        self.assertAlmostEqual(float(red / blue), 650.0 / 450.0, places=9)

    def test_electrons_are_linear_in_time_and_quantum_efficiency(self):
        base = sr.electrons_from_exposure(500.0, 4.0, 0.01, **PIXEL)
        doubled = sr.electrons_from_exposure(500.0, 4.0, 0.02, **PIXEL)
        self.assertAlmostEqual(float(doubled), 2.0 * float(base), places=6)

        half_qe = sr.electrons_from_exposure(
            500.0, 4.0, 0.01, pixel_pitch_um=4.3, quantum_efficiency=0.3)
        self.assertAlmostEqual(float(half_qe), 0.5 * float(base), places=6)

    def test_bigger_pixels_collect_as_the_square_of_the_pitch(self):
        small = sr.electrons_from_exposure(
            500.0, 4.0, 0.01, pixel_pitch_um=2.0, quantum_efficiency=0.6)
        big = sr.electrons_from_exposure(
            500.0, 4.0, 0.01, pixel_pitch_um=4.0, quantum_efficiency=0.6)
        self.assertAlmostEqual(float(big / small), 4.0, places=6)

    def test_reciprocity(self):
        """Opening one stop and halving the time must land on the same electrons."""
        a = sr.electrons_from_exposure(500.0, 4.0, 1 / 250.0, **PIXEL)
        b = sr.electrons_from_exposure(500.0, 2.0, 1 / 1000.0, **PIXEL)
        self.assertAlmostEqual(float(a), float(b), places=6)

    def test_a_plausible_daylight_exposure_lands_in_a_plausible_well(self):
        """Sunny 16 on a 4.3 um pixel should fill a typical well, not overflow it
        by orders of magnitude -- a units error anywhere shows up here."""
        e = float(sr.electrons_from_exposure(4000.0, 16.0, 1 / 125.0, **PIXEL))
        self.assertGreater(e, 1_000.0)
        self.assertLess(e, 60_000.0)


class TestExposureValue(unittest.TestCase):
    def test_ev_matches_the_definition(self):
        self.assertAlmostEqual(float(sr.exposure_value(4.0, 1 / 16.0)), 8.0, places=9)
        self.assertAlmostEqual(float(sr.exposure_value(1.0, 1.0)), 0.0, places=9)

    def test_one_stop_of_either_control_is_one_ev(self):
        base = float(sr.exposure_value(4.0, 1 / 250.0))
        self.assertAlmostEqual(float(sr.exposure_value(4.0, 1 / 500.0)), base + 1.0, places=9)
        self.assertAlmostEqual(
            float(sr.exposure_value(4.0 * math.sqrt(2.0), 1 / 250.0)), base + 1.0, places=9)

    def test_sunny_16_is_about_ev_15(self):
        self.assertAlmostEqual(float(sr.exposure_value(16.0, 1 / 128.0)), 15.0, places=6)

    def test_ev100_round_trips_through_luminance(self):
        for luminance in (0.5, 40.0, 4000.0):
            back = sr.luminance_from_ev100(sr.ev100_from_luminance(luminance))
            self.assertAlmostEqual(float(back), luminance, places=6)

    def test_sunny_16_meters_close_to_the_sunny_16_setting(self):
        """A 4000 cd/m2 sunlit subject should meter within a third of a stop of
        the f/16 rule -- which is the whole reason the rule is memorable."""
        scene = float(sr.ev100_from_luminance(4000.0))
        setting = float(sr.exposure_value(16.0, 1 / 125.0))
        self.assertLess(abs(scene - setting), 0.34)

    def test_solving_for_either_control_returns_to_the_same_ev(self):
        ev = 12.0
        t = float(sr.shutter_for_exposure_value(5.6, ev))
        self.assertAlmostEqual(float(sr.exposure_value(5.6, t)), ev, places=9)
        n = float(sr.f_number_for_exposure_value(1 / 250.0, ev))
        self.assertAlmostEqual(float(sr.exposure_value(n, 1 / 250.0)), ev, places=9)

    def test_ev_solvers_are_vectorised(self):
        f = np.array([2.0, 4.0, 8.0])
        t = sr.shutter_for_exposure_value(f, 12.0)
        self.assertEqual(t.shape, f.shape)
        self.assertTrue(np.allclose(sr.exposure_value(f, t), 12.0))

    def test_non_positive_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            sr.exposure_value(4.0, 0.0)
        with self.assertRaises(ValueError):
            sr.ev100_from_luminance(0.0)


if __name__ == "__main__":
    unittest.main()
