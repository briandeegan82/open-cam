"""HDR pixel architectures (tools/hdr_pixel.py): analytic SNR vs Monte Carlo, merge, PWL, hook."""

from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import apply_emva_noise  # noqa: E402  (synthetic_data puts tools/ on sys.path)
import hdr_pixel as hp
import numpy as np
import validate_hdr_model as vhm
from camera_model import load_camera_model, noise_config_from_camera_model
from emva_theory import temporal_variance_electrons_squared
from synthetic_data import colour_bars, run_tool_main, write_gaussian_qe, write_rgb_exr, write_yaml

REPO = Path(__file__).resolve().parent.parent
H, W = 24, 32


def _noise_cfg(recipe: str) -> dict:
    cm = load_camera_model(REPO / "config" / "camera_recipes" / f"{recipe}.yaml")
    return noise_config_from_camera_model(cm, "", "")


def _arch(recipe: str, mutate=None) -> hp.HdrArchitecture:
    cfg = copy.deepcopy(_noise_cfg(recipe))
    if mutate:
        mutate(cfg["hdr"])
    return hp.build_architecture(cfg["hdr"], hp.base_from_noise_config(cfg))


class TestTheoryVsMonteCarlo(unittest.TestCase):
    def test_example_models_agree_with_monte_carlo(self) -> None:
        for recipe in vhm.DEFAULT_RECIPES:
            with self.subTest(recipe=recipe):
                r = vhm.validate_recipe(recipe, trials=20000, seed=0, levels=12)
                self.assertLess(r["max_rel_err_snr_temporal"], 0.035)
                self.assertLess(r["max_rel_err_snr_total"], 0.035)
                self.assertLess(r["max_mean_bias_sem"], 5.0)
                self.assertEqual(len(r["dips"]), len(r["transitions_e"]) - 1)
                for d in r["dips"]:
                    self.assertAlmostEqual(d["mc_dip_dB"], d["theory_dip_dB"], delta=0.5)

    def test_dcg_switch_reduces_to_single_gain_emva_snr(self) -> None:
        a = _arch("default_hdr_dcg", lambda h: h["dcg"].update(readout="switch"))
        self.assertEqual(a.method, "select")
        dark = a.collector("pd").dark_e
        t_hcg = dict(a.transitions_e())["hcg"]
        for name, mus in (("hcg", [5.0, 100.0, 0.95 * t_hcg]), ("lcg", [1.05 * t_hcg, 20000.0])):
            r = next(x for x in a.readouts if x.name == name)
            mu = np.array(mus)
            var = [temporal_variance_electrons_squared(m, r.sigma_e, use_poisson=True, mu_dark_e=dark) for m in mus]
            expected = mu / np.sqrt(np.asarray(var) + r.K_e_per_DN**2 / 12.0)
            np.testing.assert_allclose(hp.theory_snr(a, mu)["snr"], expected, rtol=1e-12)

    def test_dual_readout_merge_is_minimum_variance(self) -> None:
        a = _arch("default_hdr_dcg")
        mu = np.array([1000.0])
        C = hp.estimate_covariance(a, mu, [1.0, 1.0])[0]
        var = hp.theory_snr(a, mu)["var_e2"][0]
        self.assertAlmostEqual(var, 1.0 / np.sum(np.linalg.inv(C)), places=9)
        self.assertLess(var, np.min(np.diag(C)))

    def test_split_pixel_dip_matches_small_photodiode_closed_form(self) -> None:
        a = _arch("default_hdr_split_pixel")
        spd = a.collector("spd")
        r = next(x for x in a.readouts if x.collector == "spd")
        mu = 1.05 * dict(a.transitions_e())["lpd"]
        q = spd.response * mu
        expected = q / math.sqrt(q + spd.dark_e + r.sigma_e**2 + r.K_e_per_DN**2 / 12.0)
        self.assertAlmostEqual(hp.theory_snr(a, np.array([mu]), include_compander=False)["snr"][0], expected, places=9)
        below = hp.theory_snr(a, np.array([0.95 * dict(a.transitions_e())["lpd"]]))["snr"][0]
        self.assertGreater(20 * math.log10(below / expected), 10.0)

    def test_lofic_lcg_includes_ktc_of_combined_capacitance(self) -> None:
        a = _arch("default_hdr_lofic")
        lcg = next(x for x in a.readouts if x.name == "lcg")
        ktc = math.sqrt(1.380649e-23 * 293.15 * 19.2e-15) / 1.602176634e-19
        self.assertAlmostEqual(ktc, 55.0, delta=0.1)
        self.assertAlmostEqual(lcg.sigma_e, math.hypot(15.0, ktc), places=9)
        self.assertTrue(lcg.reads_overflow)
        self.assertEqual(a.saturation_e(lcg), 120000.0)
        self.assertEqual(a.saturation_e(next(x for x in a.readouts if x.name == "hcg")), 8000.0)


class TestArchitectureDetails(unittest.TestCase):
    def test_multi_exposure_is_sequential_and_accepts_capture_inputs(self) -> None:
        a = _arch("default_hdr_3exp")
        np.testing.assert_allclose([c.t_int_s for c in a.collectors], [0.01, 0.000625, 0.0000390625])
        np.testing.assert_allclose([c.t_start_s for c in a.collectors], [0.0, 0.01, 0.010625])
        sig = np.full(4000, 1000.0)
        dn = hp.simulate_captures(a, sig, np.random.default_rng(0))
        dn0 = hp.simulate_captures(a, sig, np.random.default_rng(0), capture_signal_e={"exp1": np.zeros(4000)})
        self.assertAlmostEqual(float(np.mean(dn["exp1"])), 64 + 62.5 / 2.5, delta=0.5)
        self.assertLess(float(np.mean(dn0["exp1"])), 64.5)

    def test_split_pixel_spectral_ratio_map_follows_cfa(self) -> None:
        a = _arch(
            "default_hdr_split_pixel",
            lambda h: h["split_pixel"]["small"].update(sensitivity_ratio_rgb=[0.02, 0.025, 0.03]),
        )
        m = hp.response_map(a.collector("spd"), (4, 4), "RGGB")
        self.assertEqual((m[0, 0], m[0, 1], m[1, 0], m[1, 1]), (0.02, 0.025, 0.025, 0.03))
        self.assertEqual(hp.response_map(a.collector("lpd"), (4, 4), "RGGB"), 1.0)

    def test_pwl_auto_knees_roundtrip_and_snr_cost(self) -> None:
        a = _arch("default_hdr_3exp")
        c = a.compander
        self.assertEqual((c.knees_in[0], c.knees_out[0]), (0.0, 0.0))
        self.assertTrue(np.all(np.diff(c.knees_in) > 0) and np.all(np.diff(c.knees_out) > 0))
        self.assertAlmostEqual(c.knees_out[-1] + c.black_DN, 4095.0)
        x = np.concatenate([[-20.0, -1.0], np.geomspace(0.5, c.knees_in[-1], 2000)])
        err = np.abs(hp.decompand(hp.compand(x, c), c) - x)
        step = np.sqrt(12.0 * hp.compand_quantisation_var_dn(np.maximum(x, 0), c))
        self.assertTrue(np.all(err <= 0.5 * step + 1e-9))
        mu = np.geomspace(10.0, 0.95 * a.max_reference_e, 400)
        loss = 20 * np.log10(hp.theory_snr(a, mu, include_compander=False)["snr"] / hp.theory_snr(a, mu)["snr"])
        self.assertLess(float(np.nanmax(loss)), 1.0)

    def test_invalid_configs_raise(self) -> None:
        base = hp.base_from_noise_config(_noise_cfg("default_hdr_dcg"))
        with self.assertRaises(ValueError):
            hp.build_architecture({"architecture": "quad_bayer"}, base)
        with self.assertRaises(ValueError):
            hp.build_architecture({"architecture": "dcg", "merge": {"threshold_fraction": 0.0}}, base)
        with self.assertRaises(ValueError):
            hp.build_architecture({"architecture": "lofic", "lofic": {}}, base)
        with self.assertRaises(ValueError):
            hp.parse_capture_args(["no_equals_sign"])

    def test_camera_recipe_threads_hdr_section(self) -> None:
        self.assertEqual(_noise_cfg("default")["hdr"], {})
        self.assertFalse(hp.hdr_enabled(_noise_cfg("default")))
        for recipe in vhm.DEFAULT_RECIPES:
            self.assertTrue(hp.hdr_enabled(_noise_cfg(recipe)), recipe)

    def test_illuminance_conversion_scales_with_pixel_area_and_time(self) -> None:
        cfg = _noise_cfg("default_hdr_dcg")
        e1 = vhm.electrons_per_lux(cfg)
        cfg2 = copy.deepcopy(cfg)
        cfg2["sensor"]["pixel_pitch_um"] *= 2
        cfg2["sensor"]["integration_time_s"] *= 3
        self.assertAlmostEqual(vhm.electrons_per_lux(cfg2) / e1, 12.0, places=9)
        self.assertGreater(e1, 0.0)


class TestPipelineHook(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.qe = write_gaussian_qe(self.tmp)
        self.exr = write_rgb_exr(self.tmp / "scene.exr", colour_bars(H, W))
        self.raw_out = self.tmp / "noisy.raw16"
        self.cfg_path = self.tmp / "noise.yaml"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, hdr: dict | None, *, bayer: bool = False, scale: float = 1000.0) -> None:
        cfg = {
            "sensor": {"pixel_pitch_um": 3.0, "integration_time_s": 0.01, "quantum_efficiency": self.qe},
            "emva": {
                "overall_system_gain_K_e_per_DN": 2.5,
                "sigma_d_e": 2.0,
                "dsnu_std_e": 0.3,
                "prnu_std_fraction": 0.005,
                "black_level_DN": 64.0,
                "dark_current_e_per_s": 1.0,
                "spatial_noise_seed": 7,
            },
            "adc": {"full_well_e": 10000.0, "bit_depth": 12},
            "processing": {
                "linear_exr_mode": "rgb",
                "exposure_scale_e_per_unit": scale,
                "preview_white_balance": {"enabled": False},
                "preview_color_correction": {"enabled": False},
            },
            "bayer": {"enabled": True, "pattern": "RGGB", "demosaic": False} if bayer else {"enabled": False},
            "output": {"linear_rgb_in": str(self.exr), "raw_out": str(self.raw_out)},
        }
        if hdr is not None:
            cfg["hdr"] = hdr
        write_yaml(self.cfg_path, cfg)

    def _run(self, *extra: str) -> tuple[dict, bytes, dict[str, bytes]]:
        run_tool_main(apply_emva_noise.main, ["--config", str(self.cfg_path), "--seed", "3", *extra])
        stats = json.loads((self.tmp / "noisy_png" / "run_stats.json").read_text())
        pngs = {p.name: p.read_bytes() for p in sorted((self.tmp / "noisy_png").glob("*.png"))}
        return stats, self.raw_out.read_bytes(), pngs

    def test_disabled_hdr_is_bit_identical_to_no_hdr_block(self) -> None:
        for bayer in (False, True):
            with self.subTest(bayer=bayer):
                self._write(None, bayer=bayer)
                s0, raw0, png0 = self._run()
                self._write({"enabled": False, "architecture": "dcg", "dcg": {"hcg": {"K_e_per_DN": 0.5}}}, bayer=bayer)
                s1, raw1, png1 = self._run()
                self.assertEqual(raw0, raw1)
                self.assertEqual(png0, png1)
                self.assertEqual(s0, s1)
                self.assertNotIn("hdr", s1)
                self.assertFalse((self.tmp / "noisy_hdr").exists())

    def test_dcg_writes_per_capture_raws_and_linear_hdr(self) -> None:
        hdr = {
            "enabled": True,
            "architecture": "dcg",
            "dcg": {
                "photodiode_full_well_e": 25000,
                "hcg": {"K_e_per_DN": 0.7, "sigma_e": 1.5, "full_well_e": 2800},
                "lcg": {"K_e_per_DN": 6.25, "sigma_e": 8.0, "full_well_e": 25000},
            },
        }
        self._write(hdr, bayer=True, scale=15000.0)
        stats, raw, _ = self._run()
        out = self.tmp / "noisy_hdr"
        for name in ("hcg", "lcg"):
            self.assertEqual((out / f"{name}.raw16").stat().st_size, H * W * 2)
        lin = np.load(out / "hdr_linear.npz")
        self.assertEqual(lin["hdr_e"].shape, (H, W))
        meta = json.loads((out / "hdr_metadata.json").read_text())
        self.assertEqual(meta["architecture"], "dcg")
        self.assertEqual(stats["hdr"]["architecture"], "dcg")
        np.testing.assert_array_equal(
            np.frombuffer(raw, "<u2").reshape(H, W), np.clip(np.rint(lin["hdr_dn"]), 0, 65535).astype(np.uint16)
        )
        self.assertAlmostEqual(stats["hdr"]["hdr_e_mean"] / stats["signal_e_mean_mono"], 1.0, delta=0.02)
        hcg = np.fromfile(out / "hcg.raw16", "<u2")
        self.assertGreater(int(hcg.max()), 2800 / 0.7 + 64 - 20)  # HCG swing saturates on bright bars

    def test_multi_exposure_capture_electrons_and_compand(self) -> None:
        npz = self.tmp / "exp1.npz"
        np.savez(npz, electrons_rgb=np.zeros((H, W, 3), np.float32))
        hdr = {
            "enabled": True,
            "architecture": "multi_exposure",
            "multi_exposure": {"exposure_ratios": [1.0, 0.0625]},
            "compand": {"enabled": True, "bit_depth": 12},
        }
        self._write(hdr, bayer=True, scale=40000.0)
        stats, raw, _ = self._run("--hdr-capture-electrons", f"exp1={npz}")
        self.assertLessEqual(int(np.frombuffer(raw, "<u2").max()), 4095)
        self.assertEqual(stats["hdr"]["outputs"]["raw_out_content"], "companded_code")
        self.assertIn("exp1", stats["hdr"]["capture_inputs"])
        exp1 = np.fromfile(self.tmp / "noisy_hdr" / "exp1.raw16", "<u2")
        self.assertLess(float(exp1.mean()), 65.0)
        lin = np.load(self.tmp / "noisy_hdr" / "hdr_linear.npz")
        self.assertIn("companded_code", lin.files)


if __name__ == "__main__":
    unittest.main()
