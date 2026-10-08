from __future__ import annotations

import unittest

import numpy as np

from tools.emva_theory import (
    compare_config_to_datasheet,
    dark_floor_clip_mean_var_dn,
    dsnu_offset_map,
    emva1288_dsnu,
    emva1288_prnu,
    emva1288_spatial_stats,
    mean_dn_linear,
    monte_carlo_temporal_dn_stats,
    photon_transfer_curve_checks,
    prnu_gain_map,
    simulate_uniform_stack,
    spatial_std_electrons,
    temporal_variance_dn_squared,
)
from tools.validate_emva_model import _effective_mean_tol_dn


class TestEmvaTheory(unittest.TestCase):
    def test_dark_folded_matches_monte_carlo(self) -> None:
        K, sigma, black = 3.35, 2.0, 64.0
        pm, pv = dark_floor_clip_mean_var_dn(sigma, K, black)
        mm, mv = monte_carlo_temporal_dn_stats(
            0.0,
            sigma,
            K,
            black,
            use_poisson=True,
            full_well_e=None,
            n_trials=50_000,
            seed=42,
        )
        self.assertAlmostEqual(mm, pm, delta=0.06)
        self.assertAlmostEqual(mv, pv, delta=0.02)

    def test_ptc_high_signal_variance_poisson(self) -> None:
        K, sigma, black = 3.35, 2.0, 64.0
        mu = 5000.0
        pred = temporal_variance_dn_squared(mu, sigma, K, use_poisson=True)
        _, mc = monte_carlo_temporal_dn_stats(
            mu,
            sigma,
            K,
            black,
            use_poisson=True,
            full_well_e=None,
            n_trials=40_000,
            seed=7,
        )
        self.assertLess(abs(mc - pred) / pred, 0.04)

    def test_mean_dn_linear_mid_signal(self) -> None:
        K, black = 3.35, 64.0
        mu = 1000.0
        mm, _ = monte_carlo_temporal_dn_stats(
            mu,
            2.0,
            K,
            black,
            use_poisson=True,
            full_well_e=None,
            n_trials=30_000,
            seed=11,
        )
        self.assertAlmostEqual(mm, mean_dn_linear(mu, K, black), delta=0.08)

    def test_photon_transfer_curve_runs(self) -> None:
        rows = photon_transfer_curve_checks(
            np.array([0.0, 500.0, 2000.0], dtype=np.float64),
            2.0,
            3.35,
            64.0,
            use_poisson=True,
            full_well_e=None,
            n_trials=8000,
            seed=0,
            variance_rtol=0.1,
            mean_atol=0.2,
        )
        self.assertTrue(all(r["mean_ok"] and r["var_ok"] for r in rows))

    def test_effective_mean_tolerance_uses_statistical_floor(self) -> None:
        tol = _effective_mean_tol_dn(0.1, pred_var_dn=900.0, n_trials=100)
        # sqrt(900/100)=3, 3sigma = 9 dominates fixed atol 0.1
        self.assertAlmostEqual(tol, 9.0, places=6)

    def test_datasheet_compare_supports_dn_per_e_gain(self) -> None:
        # datasheet gives DN/e, but config uses e/DN
        r = compare_config_to_datasheet(
            K_cfg=0.5,
            sigma_cfg=2.0,
            fw_cfg=5000.0,
            black_cfg=64.0,
            K_ds=2.0,
            sigma_ds=2.0,
            fw_ds=5000.0,
            black_ds=64.0,
            rtol=1e-6,
            gain_convention_ds="dn_per_e",
        )
        self.assertTrue(r["all_ok"])

    def test_datasheet_compare_scales_black_level_with_bit_depth(self) -> None:
        # datasheet black at 10-bit should scale by 4 at 12-bit
        r = compare_config_to_datasheet(
            K_cfg=1.0,
            sigma_cfg=2.0,
            fw_cfg=5000.0,
            black_cfg=64.0,
            K_ds=1.0,
            sigma_ds=2.0,
            fw_ds=5000.0,
            black_ds=16.0,
            rtol=1e-6,
            bit_depth_cfg=12,
            bit_depth_ds=10,
        )
        self.assertTrue(r["all_ok"])


class TestEmva1288DsnuPrnu(unittest.TestCase):
    def test_spatial_std_combines_additive_dsnu_and_multiplicative_prnu(self) -> None:
        mu = np.array([0.0, 1000.0, 4000.0])
        std = spatial_std_electrons(mu, dsnu_std_e=3.0, prnu_std_fraction=0.02)
        np.testing.assert_allclose(std[0], 3.0)
        np.testing.assert_allclose(std[1], np.sqrt(3.0**2 + (0.02 * 1000.0) ** 2))
        np.testing.assert_allclose(std[2], np.sqrt(3.0**2 + (0.02 * 4000.0) ** 2))

    def test_temporal_correction_removes_read_noise_bias(self) -> None:
        rng = np.random.default_rng(0)
        h = w = 48
        prnu = prnu_gain_map((h, w), 0.0, rng)
        dsnu = dsnu_offset_map((h, w), dsnu_std_e=0.0, dark_mean_e=0.0, rng=rng)
        sigma_d, k, n_frames = 4.0, 1.0, 40
        # Uniform field well above the dark floor so read noise is not clipped.
        stack = simulate_uniform_stack(
            mu_e=200.0,
            n_frames=n_frames,
            prnu_map=prnu,
            dsnu_map=dsnu,
            dark_mean_e=0.0,
            sigma_d_e=sigma_d,
            K_e_per_DN=k,
            black_level_DN=0.0,
            full_well_e=10000.0,
            use_poisson=False,
            seed=1,
        )
        stats = emva1288_spatial_stats(stack)
        expected_residual = sigma_d / np.sqrt(n_frames)
        self.assertAlmostEqual(stats.spatial_std_mean_image_dn, expected_residual, delta=0.12)
        # After EMVA correction the leftover "DSNU" should collapse toward 0.
        self.assertLess(stats.corrected_spatial_std_dn, 0.25 * expected_residual)

    def test_emva1288_recovers_injected_gaussian_dsnu_and_prnu(self) -> None:
        rng = np.random.default_rng(2)
        h = w = 64
        prnu_true, dsnu_true = 0.025, 6.0
        prnu = prnu_gain_map((h, w), prnu_true, rng)
        dsnu = dsnu_offset_map(
            (h, w),
            dsnu_std_e=dsnu_true,
            dark_mean_e=0.0,
            rng=rng,
            model="gaussian",
        )
        k = 2.0
        dark = simulate_uniform_stack(
            mu_e=0.0,
            n_frames=80,
            prnu_map=prnu,
            dsnu_map=dsnu,
            dark_mean_e=0.0,
            sigma_d_e=1.0,
            K_e_per_DN=k,
            black_level_DN=16.0,
            full_well_e=8000.0,
            use_poisson=True,
            seed=3,
        )
        bright = simulate_uniform_stack(
            mu_e=2000.0,
            n_frames=80,
            prnu_map=prnu,
            dsnu_map=dsnu,
            dark_mean_e=0.0,
            sigma_d_e=1.0,
            K_e_per_DN=k,
            black_level_DN=16.0,
            full_well_e=8000.0,
            use_poisson=True,
            seed=4,
        )
        dsnu_m = emva1288_dsnu(dark, k)
        prnu_m = emva1288_prnu(dark, bright)
        self.assertLess(abs(dsnu_m.dsnu_e - dsnu_true) / dsnu_true, 0.12)
        self.assertLess(abs(prnu_m.prnu_fraction - prnu_true) / prnu_true, 0.15)

    def test_lognormal_dsnu_is_zero_mean_and_matches_std_when_well_conditioned(self) -> None:
        rng = np.random.default_rng(5)
        dark_mean, dsnu_std = 20.0, 5.0
        offset = dsnu_offset_map(
            (256, 256),
            dsnu_std_e=dsnu_std,
            dark_mean_e=dark_mean,
            rng=rng,
            model="lognormal",
        )
        self.assertLess(abs(float(offset.mean())), 0.4)
        self.assertAlmostEqual(float(offset.std(ddof=1)), dsnu_std, delta=0.6)


if __name__ == "__main__":
    unittest.main()
