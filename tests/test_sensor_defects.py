"""Tests for the sensor-defect models in tools/apply_emva_noise.py.

These cover the models the pipeline applies inside ``main()`` and the exposure
demo drives directly: ISO gain, kTC reset noise, row/column FPN, 1/f flicker and
ADC INL/DNL.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import apply_emva_noise as aen  # noqa: E402

K_B_J = 1.380649e-23
Q_E = 1.602176634e-19


class TestIsoGain(unittest.TestCase):
    def test_unity_gain_changes_nothing(self):
        self.assertEqual(aen.iso_scaled_conversion(0.8, 10000.0, 1.0), (0.8, 10000.0))

    def test_gain_divides_both_the_conversion_gain_and_the_well(self):
        k, well = aen.iso_scaled_conversion(0.8, 10000.0, 4.0)
        self.assertAlmostEqual(k, 0.2, places=12)
        self.assertAlmostEqual(well, 2500.0, places=9)

    def test_dn_per_electron_rises_exactly_with_gain(self):
        """The point of ISO: the same electrons land on a higher code."""
        base_k, _ = aen.iso_scaled_conversion(0.8, 10000.0, 1.0)
        pushed_k, _ = aen.iso_scaled_conversion(0.8, 10000.0, 8.0)
        self.assertAlmostEqual((100.0 / pushed_k) / (100.0 / base_k), 8.0, places=9)

    def test_non_positive_gain_is_rejected(self):
        with self.assertRaises(ValueError):
            aen.iso_scaled_conversion(0.8, 10000.0, 0.0)

    def test_amplifier_noise_adds_in_quadrature_above_unity_gain(self):
        got = aen.amplifier_read_noise_e(3.0, 0.5, 4.0)
        self.assertAlmostEqual(got, float(np.sqrt(9.0 + 4.0)), places=9)

    def test_amplifier_noise_is_inert_at_or_below_unity_gain(self):
        self.assertEqual(aen.amplifier_read_noise_e(3.0, 0.5, 1.0), 3.0)
        self.assertEqual(aen.amplifier_read_noise_e(3.0, 0.0, 16.0), 3.0)


class TestKtcNoise(unittest.TestCase):
    def test_matches_sqrt_ktc_over_q(self):
        sigma = aen.ktc_sigma_e(temperature_c=20.0, K_e_per_DN=0.8, bit_depth=12, node_capacitance_fF=5.0)
        want = float(np.sqrt(K_B_J * 293.15 * 5e-15) / Q_E)
        self.assertAlmostEqual(sigma, want, places=9)

    def test_scales_as_the_square_root_of_absolute_temperature(self):
        cold = aen.ktc_sigma_e(temperature_c=-23.15, K_e_per_DN=0.8, bit_depth=12, node_capacitance_fF=5.0)
        hot = aen.ktc_sigma_e(temperature_c=76.85, K_e_per_DN=0.8, bit_depth=12, node_capacitance_fF=5.0)
        self.assertAlmostEqual(hot / cold, float(np.sqrt(350.0 / 250.0)), places=9)

    def test_scales_as_the_square_root_of_capacitance(self):
        small = aen.ktc_sigma_e(temperature_c=20.0, K_e_per_DN=0.8, bit_depth=12, node_capacitance_fF=2.0)
        big = aen.ktc_sigma_e(temperature_c=20.0, K_e_per_DN=0.8, bit_depth=12, node_capacitance_fF=8.0)
        self.assertAlmostEqual(big / small, 2.0, places=9)

    def test_capacitance_derived_from_conversion_gain_when_not_measured(self):
        sigma = aen.ktc_sigma_e(temperature_c=20.0, K_e_per_DN=0.8, bit_depth=12, vref_V=1.8)
        C = 0.8 * Q_E * 4095.0 / 1.8
        self.assertAlmostEqual(sigma, float(np.sqrt(K_B_J * 293.15 * C) / Q_E), places=9)

    def test_a_realistic_sense_node_gives_a_handful_of_electrons(self):
        """A few fF at room temperature is tens of electrons -- large enough that
        cancelling it with CDS is not optional."""
        sigma = aen.ktc_sigma_e(temperature_c=20.0, K_e_per_DN=0.8, bit_depth=12, node_capacitance_fF=3.0)
        self.assertGreater(sigma, 5.0)
        self.assertLess(sigma, 100.0)


class TestRowColumnFpn(unittest.TestCase):
    def test_offsets_broadcast_against_a_mono_frame(self):
        row, col = aen.row_column_fpn_offsets((8, 5), 3.0, 4.0, np.random.default_rng(0))
        self.assertEqual(row.shape, (8, 1))
        self.assertEqual(col.shape, (1, 5))
        self.assertEqual((np.zeros((8, 5)) + row + col).shape, (8, 5))

    def test_offsets_broadcast_against_a_multi_channel_frame(self):
        row, col = aen.row_column_fpn_offsets((8, 5, 3), 3.0, 4.0, np.random.default_rng(0))
        self.assertEqual(row.shape, (8, 1, 1))
        self.assertEqual(col.shape, (1, 5, 1))
        self.assertEqual((np.zeros((8, 5, 3)) + row + col).shape, (8, 5, 3))

    def test_a_row_offset_is_constant_along_its_row(self):
        """That is the whole signature: it survives averaging across the row."""
        row, _ = aen.row_column_fpn_offsets((64, 64), 5.0, 0.0, np.random.default_rng(1))
        frame = np.zeros((64, 64)) + row
        self.assertTrue(np.allclose(frame.std(axis=1), 0.0))
        self.assertGreater(frame.mean(axis=1).std(), 1.0)

    def test_zero_sigma_produces_no_pattern(self):
        row, col = aen.row_column_fpn_offsets((8, 5), 0.0, 0.0, np.random.default_rng(0))
        self.assertTrue(np.all(row == 0))
        self.assertTrue(np.all(col == 0))

    def test_the_requested_spread_is_delivered(self):
        row, _ = aen.row_column_fpn_offsets((20000, 4), 7.0, 0.0, np.random.default_rng(2))
        self.assertAlmostEqual(float(row.std()), 7.0, delta=0.2)

    def test_the_draw_is_reproducible_from_the_seed(self):
        a = aen.row_column_fpn_offsets((16, 16), 3.0, 3.0, np.random.default_rng(7))
        b = aen.row_column_fpn_offsets((16, 16), 3.0, 3.0, np.random.default_rng(7))
        self.assertTrue(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))


class TestFlickerNoise(unittest.TestCase):
    def test_returns_one_offset_per_row_at_the_requested_spread(self):
        pink = aen.flicker_row_offsets(256, 4.0, np.random.default_rng(0))
        self.assertEqual(pink.shape, (256,))
        self.assertAlmostEqual(float(pink.std()), 4.0, places=4)

    def test_zero_sigma_is_silent(self):
        self.assertTrue(np.all(aen.flicker_row_offsets(64, 0.0, np.random.default_rng(0)) == 0))

    def test_neighbouring_rows_are_correlated_unlike_white_noise(self):
        """Smooth banding rather than row-to-row hash. The margin is modest
        because 1/f is only mildly smooth -- it is not a 1/f^2 random walk."""
        rng = np.random.default_rng(3)
        pink = aen.flicker_row_offsets(4096, 1.0, rng)
        white = rng.normal(0.0, 1.0, size=4096)
        self.assertLess(float(np.abs(np.diff(pink)).mean()), 0.6 * float(np.abs(np.diff(white)).mean()))

    def test_the_power_spectrum_falls_as_one_over_f(self):
        """The defining property. Fitting log power against log frequency should
        recover a slope of -1, not 0 (white) or -2 (a random walk)."""
        power = np.zeros(2049)
        for seed in range(24):  # average periodograms down to a fittable curve
            pink = aen.flicker_row_offsets(4096, 1.0, np.random.default_rng(seed))
            power += np.abs(np.fft.rfft(pink)) ** 2
        freq = np.fft.rfftfreq(4096)
        band = slice(1, 1024)
        slope = np.polyfit(np.log(freq[band]), np.log(power[band]), 1)[0]
        self.assertAlmostEqual(float(slope), -1.0, delta=0.15)

    def test_power_per_bin_is_far_higher_at_low_frequency(self):
        pink = aen.flicker_row_offsets(4096, 1.0, np.random.default_rng(4))
        power = np.abs(np.fft.rfft(pink)) ** 2
        n = power.size
        self.assertGreater(power[1 : n // 16].mean(), 20.0 * power[n // 2 :].mean())

    def test_a_row_count_that_is_not_a_power_of_two_still_works(self):
        pink = aen.flicker_row_offsets(1000, 2.0, np.random.default_rng(5))
        self.assertEqual(pink.shape, (1000,))
        self.assertTrue(np.all(np.isfinite(pink)))


class TestAdcNonlinearity(unittest.TestCase):
    def test_inl_vanishes_at_both_ends_and_bows_in_the_middle(self):
        ramp = np.linspace(0.0, 4095.0, 512)
        out = aen.apply_adc_inl(ramp.copy(), black_dn=0.0, max_dn=4095.0, quadratic_fraction=0.03)
        dev = out - ramp
        self.assertAlmostEqual(float(dev[0]), 0.0, places=6)
        self.assertAlmostEqual(float(dev[-1]), 0.0, places=6)
        self.assertAlmostEqual(float(dev.max()), 0.03 * 4095.0 / 4.0, delta=1.0)

    def test_zero_inl_is_a_no_op(self):
        ramp = np.linspace(0.0, 4095.0, 64)
        self.assertTrue(
            np.array_equal(aen.apply_adc_inl(ramp.copy(), black_dn=0.0, max_dn=4095.0, quadratic_fraction=0.0), ramp)
        )

    def test_inl_never_pushes_a_code_out_of_range(self):
        ramp = np.linspace(0.0, 4095.0, 512)
        out = aen.apply_adc_inl(ramp, black_dn=0.0, max_dn=4095.0, quadratic_fraction=0.5)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 4095.0)

    def test_dnl_table_covers_every_code(self):
        table = aen.adc_dnl_table(4095.0, 0.5, np.random.default_rng(0))
        self.assertEqual(table.size, 4096)
        self.assertAlmostEqual(float(table.std()), 0.5, delta=0.03)

    def test_dnl_is_a_lookup_so_the_same_code_always_lands_the_same_way(self):
        """This is what separates DNL from read noise: averaging frames will not
        remove it."""
        table = aen.adc_dnl_table(4095.0, 1.0, np.random.default_rng(1))
        repeated = np.full(32, 2000.0)
        out = aen.apply_adc_dnl(repeated, table, 4095.0)
        self.assertAlmostEqual(float(out.std()), 0.0, places=9)
        self.assertAlmostEqual(float(out[0]), 2000.0 + float(table[2000]), places=5)

    def test_different_codes_get_different_offsets(self):
        table = aen.adc_dnl_table(4095.0, 1.0, np.random.default_rng(2))
        codes = np.arange(0.0, 4096.0)
        out = aen.apply_adc_dnl(codes.copy(), table, 4095.0)
        self.assertGreater(float((out - codes).std()), 0.5)

    def test_dnl_respects_the_code_range(self):
        table = aen.adc_dnl_table(4095.0, 20.0, np.random.default_rng(3))
        out = aen.apply_adc_dnl(np.array([0.0, 4095.0]), table, 4095.0)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 4095.0)


class TestBlooming(unittest.TestCase):
    def test_a_saturated_pixel_spills_into_its_neighbours(self):
        frame = np.zeros((9, 9), dtype=np.float32)
        frame[4, 4] = 5000.0
        out = aen.apply_blooming(frame, full_well_e=1000.0, spread_fraction=0.5)
        self.assertAlmostEqual(float(out[4, 4]), 1000.0, places=3)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            self.assertGreater(float(out[4 + dy, 4 + dx]), 0.0)
        self.assertAlmostEqual(float(out[4 + 1, 4 + 1]), 0.0, places=3)

    def test_nothing_ends_above_full_well(self):
        frame = np.full((16, 16), 9000.0, dtype=np.float32)
        out = aen.apply_blooming(frame, full_well_e=1000.0)
        self.assertLessEqual(float(out.max()), 1000.0 + 1e-3)

    def test_an_unsaturated_frame_is_left_alone(self):
        frame = np.full((8, 8), 500.0, dtype=np.float32)
        out = aen.apply_blooming(frame, full_well_e=1000.0)
        self.assertTrue(np.allclose(out, frame))

    def test_wider_spread_moves_more_charge(self):
        frame = np.zeros((9, 9), dtype=np.float32)
        frame[4, 4] = 5000.0
        narrow = aen.apply_blooming(frame, 1000.0, spread_fraction=0.2)
        wide = aen.apply_blooming(frame, 1000.0, spread_fraction=0.9)
        self.assertGreater(float(wide[3, 4]), float(narrow[3, 4]))

    def test_multi_channel_input_is_rejected(self):
        with self.assertRaises(ValueError):
            aen.apply_blooming(np.zeros((4, 4, 3), dtype=np.float32), 1000.0)


if __name__ == "__main__":
    unittest.main()
