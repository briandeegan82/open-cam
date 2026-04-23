from __future__ import annotations

import unittest

import numpy as np

from tools.emva_theory import (
    compare_config_to_datasheet,
    dark_floor_clip_mean_var_dn,
    mean_dn_linear,
    monte_carlo_temporal_dn_stats,
    photon_transfer_curve_checks,
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


if __name__ == "__main__":
    unittest.main()
