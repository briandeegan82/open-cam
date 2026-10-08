"""Module tests for spectral_sensor_forward (analytic mode), exr_multispectral and pipeline_shell_env."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import exr_multispectral as em
import numpy as np
import pipeline_shell_env
import spectral_sensor_forward as ssf
from sensor_radiometry import C_LIGHT, H_PLANCK
from synthetic_data import REPO, run_tool_main, write_flat_qe, write_yaml

XRES, YRES = 60, 40
_trapz = getattr(np, "trapezoid", None) or np.trapz


class TestSpectralSensorForwardAnalytic(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)
        gen = self.repo / "scenes" / "generated"
        gen.mkdir(parents=True)
        self.wl = np.arange(400.0, 701.0, 1.0)
        self.refl = np.linspace(0.05, 0.9, 24)
        np.savez(
            gen / "spectral_reference_1nm.npz",
            wavelength_nm=self.wl,
            illuminant=np.full_like(self.wl, 100.0),
            reflectance=np.repeat(self.refl[:, None], self.wl.size, axis=1),
        )
        self.manifest = {
            "geometry": {"patch_width": 0.45, "patch_height": 0.45, "gap": 0.04, "board_size": [2.9, 1.92]},
            "camera": {"type": "pinhole", "cam_dist": 4.25, "fov_deg": 45.0},
            "film": {"xresolution": XRES, "yresolution": YRES},
            # Light straight along the chart normal → cosθ = 1.
            "lighting": {"distant": {"from": [0.0, 0.0, 3.0], "to": [0.0, 0.0, 0.0]}},
        }
        self._write_manifest()
        self.qe = write_flat_qe(self.repo, 0.5)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_manifest(self) -> None:
        (self.repo / "scenes" / "generated" / "colorchecker_manifest.json").write_text(json.dumps(self.manifest))

    def _run(self, model: dict | None = None, lens: dict | None = None, extra: list[str] | None = None) -> dict:
        camera = {
            "sensor": {
                "pixel_pitch_um": 2.0,
                "f_number": 2.0,
                "integration_time_s": 0.01,
                "fill_factor": 1.0,
                "quantum_efficiency": self.qe,
            },
            "lens": lens or {},
            "noise": {},
            "cfa": {"enabled": False},
            "sensor_forward": {
                "model": model
                or {
                    "calibration": {
                        "mode": "photon_counting",
                        "irradiance_scale_W_m2nm_per_unit": 1e-3,
                        "target_illuminance_lux": 500.0,
                    },
                },
            },
        }
        cfg = write_yaml(self.repo / "camera.yaml", camera)
        run_tool_main(ssf.main, ["--repo-root", str(self.repo), "--camera-model-config", str(cfg), *(extra or [])])
        return dict(np.load(self.repo / "out" / "sensor_forward_electrons.npz"))

    def test_photon_counting_matches_closed_form(self) -> None:
        out = self._run()
        self.assertAlmostEqual(float(out["illuminance_input_lux"]) * float(out["illuminance_scale"]), 500.0, places=6)
        irr = 100.0 * 1e-3 * float(out["illuminance_scale"])
        photons = _trapz(irr * self.wl * 1e-9 / (H_PLANCK * C_LIGHT) * 0.5, self.wl)
        # Lambertian radiance E·R/π times the thin-lens factor π/(4N²).
        geom = 0.01 * (2e-6) ** 2 / (4.0 * 2.0**2)
        expected = self.refl * photons * geom
        np.testing.assert_allclose(out["patch_electrons_rgb"], np.repeat(expected[:, None], 3, axis=1), rtol=1e-6)
        # Every patch value is rasterised into the image.
        g = out["electrons_rgb"][..., 1]
        for e in expected:
            self.assertTrue(np.any(np.isclose(g, e, rtol=1e-5)), e)

    def test_integration_time_override_is_linear(self) -> None:
        base = self._run()["patch_electrons_rgb"]
        doubled = self._run(extra=["--integration-time-s", "0.02"])["patch_electrons_rgb"]
        np.testing.assert_allclose(doubled, 2.0 * base, rtol=1e-6)

    def test_target_lux_cli_override(self) -> None:
        base = self._run()["patch_electrons_rgb"]
        out = self._run(extra=["--target-illuminance-lux", "250"])
        np.testing.assert_allclose(out["patch_electrons_rgb"], 0.5 * base, rtol=1e-6)
        self.assertEqual(float(out["illuminance_target_lux"]), 250.0)

    def test_legacy_mode_uses_electrons_scale(self) -> None:
        model = {"calibration": {"mode": "legacy", "use_aperture_factor": False}, "electrons_scale": 1e10}
        out = self._run(model)
        expected = self.refl * _trapz(np.full_like(self.wl, 100.0 * 0.5), self.wl) * 0.01 * (2e-6) ** 2 * 1e10
        np.testing.assert_allclose(out["patch_electrons_rgb"][:, 0], expected, rtol=1e-6)
        self.assertEqual(str(out["calibration_mode"]), "legacy")

    def test_optics_and_spatial_transmission(self) -> None:
        base = self._run()
        model = {
            "calibration": {
                "mode": "photon_counting",
                "target_illuminance_lux": 500.0,
                "optics_transmittance": 0.5,
                "optics_transmittance_spatial": {"enabled": True, "edge_factor": 0.5},
            },
            "include_surround": True,
        }
        out = self._run(model)
        np.testing.assert_allclose(out["patch_electrons_rgb"], 0.5 * base["patch_electrons_rgb"], rtol=1e-6)
        self.assertTrue(json.loads(str(out["optics_transmittance_spatial"]))["enabled"])
        ratio = out["electrons_rgb"][..., 1] / np.where(
            base["electrons_rgb"][..., 1] > 0, base["electrons_rgb"][..., 1], 1
        )
        self.assertLess(float(ratio[YRES // 2, 2]), float(ratio[YRES // 2, XRES // 2]))

    def test_cos4_vignetting_and_distortion_keep_patch_values(self) -> None:
        base = self._run()
        model = {
            "calibration": {"mode": "photon_counting", "target_illuminance_lux": 500.0},
            "vignetting_cos4": True,
        }
        out = self._run(model, lens={"distortion_k1": -0.1})
        np.testing.assert_allclose(out["patch_electrons_rgb"], base["patch_electrons_rgb"], rtol=1e-6)
        self.assertTrue(bool(out["vignetting_cos4"]))
        self.assertFalse(np.array_equal(out["electrons_rgb"], base["electrons_rgb"]))
        self.assertLessEqual(float(out["electrons_rgb"].max()), float(base["electrons_rgb"].max()) * (1 + 1e-6))

    def test_tilted_light_applies_cosine(self) -> None:
        base = self._run()["patch_electrons_rgb"]
        self.manifest["lighting"]["distant"]["from"] = [0.0, 3.0, 3.0]
        self._write_manifest()
        out = self._run()
        self.assertAlmostEqual(float(out["cos_theta"]), np.cos(np.pi / 4), places=9)
        np.testing.assert_allclose(out["patch_electrons_rgb"], base * np.cos(np.pi / 4), rtol=1e-6)

    def test_realistic_camera_rejected(self) -> None:
        self.manifest["camera"]["type"] = "realistic"
        self._write_manifest()
        with self.assertRaisesRegex(RuntimeError, "realistic camera"):
            self._run()

    def test_wrong_patch_count_rejected(self) -> None:
        gen = self.repo / "scenes" / "generated"
        np.savez(
            gen / "spectral_reference_1nm.npz",
            wavelength_nm=self.wl,
            illuminant=self.wl,
            reflectance=np.ones((3, self.wl.size)),
        )
        with self.assertRaisesRegex(RuntimeError, "24 patch"):
            self._run()


class TestSpatialTransmissionMap(unittest.TestCase):
    def test_disabled_is_unity(self) -> None:
        m, meta = ssf.build_spatial_transmission_map(
            4, 5, {}, repo=REPO, wavelength_nm=np.arange(3.0), qe_rgb=np.ones((3, 3))
        )
        np.testing.assert_array_equal(m, 1.0)
        self.assertFalse(meta["enabled"])

    def test_radial_profile(self) -> None:
        m, _ = ssf.build_spatial_transmission_map(
            9,
            9,
            {"enabled": True, "edge_factor": 0.4, "exponent": 2.0},
            repo=REPO,
            wavelength_nm=np.arange(3.0),
            qe_rgb=np.ones((3, 3)),
        )
        self.assertAlmostEqual(float(m[4, 4, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(m[0, 0, 0]), 0.4, places=6)
        self.assertTrue(np.all(np.diff(m[4, 4:, 1]) < 0))

    def test_spectral_edge_factors_are_qe_weighted(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "edge.csv"
            p.write_text("400,0.2\n700,0.8\n")
            wl = np.array([400.0, 700.0])
            qe = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
            _, meta = ssf.build_spatial_transmission_map(
                3,
                3,
                {"enabled": True, "spectral_edge_factors_csv": "edge.csv"},
                repo=Path(d),
                wavelength_nm=wl,
                qe_rgb=qe,
            )
        np.testing.assert_allclose(meta["edge_factor_rgb"], [0.2, 0.5, 0.8])

    def test_unknown_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ssf.build_spatial_transmission_map(
                2,
                2,
                {"enabled": True, "mode": "cos4"},
                repo=REPO,
                wavelength_nm=np.arange(3.0),
                qe_rgb=np.ones((3, 3)),
            )


class TestExrMultispectral(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_parse_wavelength(self) -> None:
        self.assertEqual(em.parse_s0_wavelength_nm("S0.555nm"), 555.0)
        self.assertEqual(em.parse_s0_wavelength_nm("S0.555,5nm"), 555.5)
        self.assertIsNone(em.parse_s0_wavelength_nm("R"))
        self.assertIsNone(em.parse_s0_wavelength_nm("S1.555nm"))

    def test_roundtrip_and_sorted_buckets(self) -> None:
        a = np.arange(12, dtype=np.float32).reshape(3, 4)
        ch = {"R": a, "G": a + 1, "B": a + 2, "S0.600nm": a * 3, "S0.450nm": a * 2}
        path = self.tmp / "s.exr"
        em.write_separate_channels_exr(path, ch)
        back = em.read_separate_exr_channels(path)
        for k, v in ch.items():
            np.testing.assert_array_equal(back[k], v)
        planes, lam = em.spectral_buckets_from_exr(path)
        np.testing.assert_array_equal(lam, [450.0, 600.0])
        np.testing.assert_array_equal(planes[..., 0], a * 2)
        np.testing.assert_array_equal(em.linear_rgb_from_exr(path)[..., 2], a + 2)

    def test_errors(self) -> None:
        with self.assertRaises(ValueError):
            em.write_separate_channels_exr(self.tmp / "x.exr", {})
        with self.assertRaises(ValueError):
            em.write_separate_channels_exr(self.tmp / "x.exr", {"R": np.zeros((2, 2, 2), np.float32)})
        with self.assertRaises(ValueError):
            em.write_separate_channels_exr(self.tmp / "x.exr", {"R": np.zeros((2, 2)), "G": np.zeros((2, 3))})
        path = self.tmp / "rgb.exr"
        em.write_separate_channels_exr(path, {"Y": np.zeros((2, 2), np.float32)})
        with self.assertRaisesRegex(ValueError, "no S0"):
            em.spectral_buckets_from_exr(path)
        with self.assertRaisesRegex(ValueError, "no R,G,B"):
            em.linear_rgb_from_exr(path)

    def test_trapezoid_weights(self) -> None:
        np.testing.assert_array_equal(em.trapezoid_weights_nm(np.array([500.0])), [1.0])
        np.testing.assert_array_equal(em.trapezoid_weights_nm(np.array([400.0, 410.0])), [5.0, 5.0])
        lam = np.array([400.0, 410.0, 430.0, 460.0])
        w = em.trapezoid_weights_nm(lam)
        np.testing.assert_array_equal(w, [5.0, 15.0, 25.0, 15.0])
        y = 2.0 * lam + 1.0
        self.assertAlmostEqual(float(np.sum(w * y)), float(_trapz(y, lam)))


class TestPipelineShellEnv(unittest.TestCase):
    def test_bash_and_env0_formats(self) -> None:
        bash = run_tool_main(pipeline_shell_env.main, [str(REPO), "config/pipeline.yaml"])
        lines = dict(line.removeprefix("export ").split("=", 1) for line in bash.strip().splitlines())
        self.assertEqual(lines["PBRT_REL"], "third_party/pbrt-v4/build/pbrt")
        self.assertIn("XRES", lines)
        env0 = run_tool_main(pipeline_shell_env.main, [str(REPO), "config/pipeline.yaml", "--format", "env0"])
        pairs = dict(item.split("=", 1) for item in env0.split("\0") if item)
        self.assertEqual(set(pairs), set(lines))
        self.assertEqual(pairs["XRES"], lines["XRES"])

    def test_usage_errors(self) -> None:
        for argv in ([], [str(REPO), "config/pipeline.yaml", "--format", "zsh"], [str(REPO), "missing.yaml"]):
            with self.assertRaises(SystemExit) as cm:
                run_tool_main(pipeline_shell_env.main, argv)
            self.assertEqual(cm.exception.code, 2)

    def test_lensfile_canonicalisation(self) -> None:
        self.assertEqual(
            pipeline_shell_env.canonicalize_lensfile_rel(pipeline_shell_env.LEGACY_REALISTIC_LENSFILE),
            pipeline_shell_env.DEFAULT_REALISTIC_LENSFILE,
        )
        self.assertEqual(pipeline_shell_env.canonicalize_lensfile_rel("x.dat"), "x.dat")


class TestValidateEmvaModel(unittest.TestCase):
    def setUp(self) -> None:
        import yaml

        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.base = yaml.safe_load((REPO / "config" / "camera_models" / "default.yaml").read_text())

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, camera: dict, *extra: str) -> tuple[int, dict]:
        import validate_emva_model

        cfg = write_yaml(self.tmp / "camera.yaml", camera)
        out = self.tmp / "report.json"
        with self.assertRaises(SystemExit) as cm:
            run_tool_main(
                validate_emva_model.main,
                ["--repo-root", str(self.tmp), "--camera-model-config", str(cfg), "--json-out", str(out), *extra],
            )
        return cm.exception.code, json.loads(out.read_text())

    def test_research_policy_exit_code_follows_report(self) -> None:
        code, report = self._run(self.base)
        self.assertEqual(code, 0 if report["all_ok"] else 1)
        self.assertTrue(report["ptc_all_ok"])
        self.assertTrue(report["dark_frame"]["var_ok"] and report["dark_frame"]["mean_ok"])
        self.assertGreaterEqual(len(report["photon_transfer_curve"]), 2)
        model = report["model"]
        self.assertAlmostEqual(model["K_effective_e_per_DN"], model["K_base_e_per_DN"] / model["iso_gain_factor"])

    def test_strict_policy_rejects_inferred_tier(self) -> None:
        camera = dict(self.base)
        camera["source"] = {**(camera.get("source") or {}), "calibration_tier": "inferred"}
        code, report = self._run(camera, "--calibration-tier-policy", "strict")
        self.assertEqual(code, 1)
        self.assertFalse(report["calibration_ok"])
        self.assertIn("strict_policy_requires_measured_tier", report["calibration_failure_reasons"])

    def test_strict_calibration_flags_heuristic_parameters(self) -> None:
        camera = dict(self.base)
        camera["source"] = {**(camera.get("source") or {}), "emva_param_method": "heuristic_scaling"}
        code, report = self._run(camera, "--strict-calibration")
        self.assertEqual(code, 1)
        self.assertIn("heuristic_emva_parameters", report["calibration_failure_reasons"])

    def test_missing_config_exits_2(self) -> None:
        import validate_emva_model

        with self.assertRaises(SystemExit) as cm:
            run_tool_main(validate_emva_model.main, ["--camera-model-config", str(self.tmp / "nope.yaml")])
        self.assertEqual(cm.exception.code, 2)


class TestValidateDemosaicLinear(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.noise = write_yaml(
            self.tmp / "noise.yaml",
            {
                "bayer": {"enabled": True, "pattern": "RGGB"},
                "adc": {"bit_depth": 16, "full_well_e": 1e6},
                "emva": {"overall_system_gain_K_e_per_DN": 1.0, "black_level_DN": 0.0},
            },
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, electrons: np.ndarray, *extra: str, noise: Path | None = None) -> dict:
        import validate_demosaic_linear

        npz = self.tmp / "e.npz"
        np.savez(npz, electrons_rgb=electrons.astype(np.float32))
        out = self.tmp / "metrics.json"
        run_tool_main(
            validate_demosaic_linear.main,
            [
                "--repo-root",
                str(self.tmp),
                "--config",
                str(noise or self.noise),
                "--electrons-npz",
                str(npz),
                "--json-out",
                str(out),
                *extra,
            ],
        )
        return json.loads(out.read_text())

    def test_linear_ramp_is_reconstructed_exactly(self) -> None:
        yy, xx = np.mgrid[0:32, 0:48].astype(np.float64)
        ramp = np.stack([100 + 3 * xx + 2 * yy, 200 + xx, 300 + 5 * yy], axis=2)
        m = self._run(ramp)
        self.assertLess(m["max_abs_dn"], 1e-3)
        self.assertEqual(m["eval_shape"], [28, 44, 3])
        self.assertNotIn("masked", m)

    def test_high_frequency_scene_has_error(self) -> None:
        checker = (np.indices((32, 48)).sum(axis=0) % 2)[..., None] * np.array([1000.0, 500.0, 250.0])
        m = self._run(checker, "--crop", "0")
        self.assertGreater(m["rmse_dn"], 100.0)
        self.assertEqual(len(m["mae_dn_rgb"]), 3)

    def test_patch_mask_metrics(self) -> None:
        manifest = {
            "geometry": {"patch_width": 0.45, "patch_height": 0.45, "gap": 0.04, "board_size": [2.9, 1.92]},
            "camera": {"cam_dist": 4.25, "fov_deg": 45.0},
        }
        mpath = self.tmp / "manifest.json"
        mpath.write_text(json.dumps(manifest))
        rng = np.random.default_rng(0)
        m = self._run(rng.uniform(100, 1000, (40, 60, 3)), "--manifest", str(mpath))
        self.assertIn("masked", m)
        self.assertGreater(m["masked"]["pixel_count"], 0)
        manifest["camera"].pop("fov_deg")
        mpath.write_text(json.dumps(manifest))
        self.assertNotIn("masked", self._run(rng.uniform(100, 1000, (40, 60, 3)), "--manifest", str(mpath)))

    def test_bayer_disabled_rejected(self) -> None:
        noise = write_yaml(self.tmp / "off.yaml", {"bayer": {"enabled": False}, "adc": {"full_well_e": 1e4}})
        with self.assertRaisesRegex(RuntimeError, "bayer.enabled"):
            self._run(np.ones((8, 8, 3)), noise=noise)


if __name__ == "__main__":
    unittest.main()
