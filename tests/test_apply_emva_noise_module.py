"""Module tests: run ``apply_emva_noise.main()`` end to end on synthetic EXRs."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from synthetic_data import (
    SPECTRAL_LAMBDAS_NM,
    colour_bars,
    run_tool_main,
    write_flat_qe,
    write_gaussian_qe,
    write_rgb_exr,
    write_spectral_exr,
    write_yaml,
)

import apply_emva_noise  # noqa: E402  (synthetic_data puts tools/ on sys.path)
from exr_multispectral import trapezoid_weights_nm  # noqa: E402
from sensor_radiometry import C_LIGHT, H_PLANCK  # noqa: E402

H, W = 24, 32
BIT_DEPTH = 12
MAX_DN = (1 << BIT_DEPTH) - 1


class _EmvaCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.qe = write_gaussian_qe(self.tmp)
        self.rgb_exr = write_rgb_exr(self.tmp / "scene_rgb.exr", colour_bars(H, W))
        self.raw_out = self.tmp / "noisy.raw16"
        self.cfg_path = self.tmp / "noise.yaml"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _config(
        self,
        *,
        exr: Path | None = None,
        bayer: dict | None = None,
        processing: dict | None = None,
        emva: dict | None = None,
        adc: dict | None = None,
    ) -> dict:
        cfg = {
            "sensor": {
                "pixel_pitch_um": 3.0,
                "f_number": 2.8,
                "integration_time_s": 0.01,
                "fill_factor": 1.0,
                "quantum_efficiency": self.qe,
            },
            "emva": {
                "overall_system_gain_K_e_per_DN": 0.5,
                "sigma_d_e": 2.0,
                "dsnu_std_e": 0.3,
                "prnu_std_fraction": 0.005,
                "black_level_DN": 64.0,
                **(emva or {}),
            },
            "adc": {"full_well_e": 10000.0, "bit_depth": BIT_DEPTH, **(adc or {})},
            "processing": {
                "linear_exr_mode": "rgb",
                "exposure_scale_e_per_unit": 1000.0,
                "preview_white_balance": {"enabled": False},
                "preview_color_correction": {"enabled": False},
                **(processing or {}),
            },
            "bayer": bayer or {"enabled": False},
            "output": {"linear_rgb_in": str(exr or self.rgb_exr), "raw_out": str(self.raw_out)},
        }
        write_yaml(self.cfg_path, cfg)
        return cfg

    def _run(self, *extra: str, seed: int = 0) -> tuple[dict, np.ndarray]:
        run_tool_main(apply_emva_noise.main, ["--config", str(self.cfg_path), "--seed", str(seed), *extra])
        raw = np.fromfile(self.raw_out, dtype=np.uint16).reshape(H, W)
        stats = json.loads((self.tmp / "noisy_png" / "run_stats.json").read_text())
        return stats, raw


class TestRgbPath(_EmvaCase):
    def test_rgb_run_writes_raw_previews_and_stats(self) -> None:
        self._config()
        stats, raw = self._run()
        self.assertEqual(stats["signal_source"], "linear_exr")
        self.assertEqual(stats["linear_exr_mode"], "rgb")
        self.assertFalse(stats["bayer_enabled"])
        self.assertIsNone(stats["demosaic"])
        self.assertEqual(stats["bit_depth"], BIT_DEPTH)
        self.assertTrue(np.all(raw >= 0) and np.all(raw <= MAX_DN))
        self.assertGreater(float(raw.mean()), 64.0)
        self.assertTrue(any((self.tmp / "noisy_png").glob("*.png")))
        qe_rel = stats["qe_relative_rgb"]
        self.assertAlmostEqual(max(qe_rel), 1.0, places=6)

    def test_same_seed_is_bit_exact_and_new_seed_changes_temporal_noise(self) -> None:
        self._config()
        _, a = self._run(seed=7)
        _, b = self._run(seed=7)
        _, c = self._run(seed=8)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_hard_clipping_saturates_at_adc_max_and_soft_clipping_does_not(self) -> None:
        self._config(adc={"clipping": "hard"})
        stats, raw = self._run("--auto-exposure", "--target-fullwell-fraction", "0.5")
        self.assertTrue(stats["adc_clipping"])
        self.assertEqual(int(raw.max()), MAX_DN)
        self._config(adc={"clipping": "soft"})
        stats, raw = self._run("--auto-exposure", "--target-fullwell-fraction", "0.5")
        self.assertFalse(stats["adc_clipping"])
        self.assertGreater(int(raw.max()), MAX_DN)

    def test_invalid_clipping_value_is_rejected(self) -> None:
        self._config(adc={"clipping": "sometimes"})
        with self.assertRaisesRegex(ValueError, "adc.clipping"):
            self._run()

    def test_preview_wb_and_ccm_run_against_exr_reference(self) -> None:
        self._config(
            processing={
                "preview_white_balance": {"enabled": True, "method": "gray_world"},
                "preview_color_correction": {"enabled": True, "method": "lstsq_exr_reference"},
            }
        )
        stats, _ = self._run()
        self.assertEqual(stats["preview_color_correction_source"], "lstsq_exr_reference")
        ccm = np.asarray(stats["preview_color_correction_matrix_3x3"])
        np.testing.assert_allclose(ccm.sum(axis=0), 1.0, atol=1e-5)


class TestBayerPath(_EmvaCase):
    def test_stats_report_the_demosaic_algorithm_that_ran(self) -> None:
        for alg in ("bilinear", "malvar"):
            with self.subTest(alg=alg):
                self._config(bayer={"enabled": True, "pattern": "GRBG", "demosaic": alg})
                stats, raw = self._run()
                self.assertEqual(stats["demosaic"], alg)
                self.assertEqual(stats["bayer_pattern"], "GRBG")
                self.assertEqual(raw.shape, (H, W))

    def test_mosaic_carries_per_site_colour(self) -> None:
        # Red bar (cols 8..15): R sites must read brighter than B sites under an RGGB mosaic.
        self._config(
            bayer={"enabled": True, "pattern": "RGGB", "demosaic": False},
            emva={"sigma_d_e": 0.0, "dsnu_std_e": 0.0, "prnu_std_fraction": 0.0},
        )
        stats, raw = self._run()
        self.assertIsNone(stats["demosaic"])
        red_bar = raw[:, 8:16].astype(np.float64)
        self.assertGreater(red_bar[0::2, 0::2].mean(), 1.5 * red_bar[1::2, 1::2].mean())

    def test_optional_sensor_effects_run_and_defect_map_persists(self) -> None:
        defect_map = self.tmp / "defects.npz"
        self._config(
            bayer={
                "enabled": True,
                "pattern": "RGGB",
                "demosaic": "malvar",
                "spatial_crosstalk": {"enabled": True, "sigma_pixels": 0.4},
            },
            emva={
                "blooming": {"enabled": True, "spread_fraction": 0.5},
                "defect_pixels": {
                    "enabled": True,
                    "hot_pixel_rate": 0.02,
                    "stuck_high_rate": 0.01,
                    "persistent_map_npz": str(defect_map),
                },
            },
        )
        first, raw_a = self._run(seed=1)
        self.assertTrue(first["blooming_enabled"])
        self.assertTrue(defect_map.is_file())
        second, _ = self._run(seed=2)
        for key in ("hot_pixel_count", "stuck_high_count", "stuck_low_count"):
            self.assertEqual(first["defect_pixels"][key], second["defect_pixels"][key])
        self.assertGreater(first["defect_pixels"]["hot_pixel_count"], 0)
        stuck = np.load(defect_map)["stuck_high_mask"].astype(bool)
        self.assertTrue(np.all(raw_a[stuck] > 0))


class TestIntegrateQePath(_EmvaCase):
    def _spectral_exr(self) -> tuple[Path, np.ndarray]:
        lam = SPECTRAL_LAMBDAS_NM
        planes = np.broadcast_to(np.linspace(0.2, 1.0, W, dtype=np.float32)[None, :, None], (H, W, lam.size))
        planes = (planes * np.ones(lam.size, dtype=np.float32) * 2000.0).astype(np.float32)
        return write_spectral_exr(self.tmp / "scene_spectral.exr", planes, lam), planes

    def test_integrate_qe_uses_spectral_planes(self) -> None:
        exr, _ = self._spectral_exr()
        self._config(exr=exr, processing={"linear_exr_mode": "integrate_qe", "exposure_scale_e_per_unit": 1.0})
        stats, _ = self._run()
        self.assertEqual(stats["signal_source"], "linear_exr_integrate_qe")
        self.assertGreater(stats["signal_e_mean_rgb"][1], 0.0)

    def test_integrate_qe_falls_back_to_rgb_for_rgb_only_exr(self) -> None:
        self._config(processing={"linear_exr_mode": "integrate_qe"})
        stats, _ = self._run()
        self.assertEqual(stats["signal_source"], "linear_exr_rgb_fallback")

    def test_spectral_integration_matches_closed_form(self) -> None:
        exr, planes = self._spectral_exr()
        qe = write_flat_qe(self.tmp, 0.5)
        sensor = {"pixel_pitch_um": 3.0, "f_number": 2.8, "integration_time_s": 0.01, "fill_factor": 0.8}
        cal = {"optics_transmittance": 0.9, "irradiance_scale_W_m2nm_per_unit": 1e-3}
        got = apply_emva_noise.integrate_exr_spectral_qe(exr, self.tmp, qe, sensor, cal)
        lam = SPECTRAL_LAMBDAS_NM.astype(np.float64)
        irr = planes.astype(np.float64) * np.pi / (4 * 2.8**2) * 0.9 * 1e-3
        photons = irr * lam * 1e-9 / (H_PLANCK * C_LIGHT)
        expected = np.sum(photons * 0.5 * trapezoid_weights_nm(lam), axis=2) * 0.01 * 0.8 * (3e-6) ** 2
        np.testing.assert_allclose(got, np.repeat(expected[:, :, None], 3, axis=2), rtol=1e-5)


class TestPreviewCcmHelpers(unittest.TestCase):
    def test_identity_scene_gives_neutral_preserving_ccm(self) -> None:
        rng = np.random.default_rng(0)
        ref = rng.random((32, 32, 3)).astype(np.float32)
        dn = ref * 1000.0 + 64.0
        for method in ("diag_exr_reference", "lstsq_exr_reference"):
            ccm = apply_emva_noise.fit_preview_ccm(dn, ref, 64.0, method)
            np.testing.assert_allclose(ccm.sum(axis=0), 1.0, atol=1e-5)
            np.testing.assert_allclose(ccm, np.eye(3), atol=1e-3)

    def test_ccm_application_keeps_the_black_pedestal(self) -> None:
        dn = np.full((2, 2, 3), 64.0)
        out = apply_emva_noise.apply_preview_ccm_dn(dn, 64.0, np.array([[2.0, 0, 0], [0, 1, 0], [-1, 0, 1]]))
        np.testing.assert_allclose(out, 64.0)


if __name__ == "__main__":
    unittest.main()
