"""The integrate_qe path (apply_emva_noise) and pbrt_spectral_exr_to_electrons share one chain."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import apply_emva_noise
import numpy as np
import pbrt_spectral_exr_to_electrons as pbrt_tool
from spectral_sensor_forward import illuminance_lux_from_irradiance
from synthetic_data import (
    SPECTRAL_LAMBDAS_NM,
    run_tool_main,
    write_curve,
    write_gaussian_qe,
    write_spectral_exr,
    write_yaml,
)

H, W = 12, 16


class TestSharedRadiometry(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        lam = SPECTRAL_LAMBDAS_NM
        ramp = np.linspace(0.5, 1.5, W, dtype=np.float32)[None, :, None]
        spd = np.linspace(1.0, 2.0, lam.size, dtype=np.float32)[None, None, :]
        self.planes = np.broadcast_to(ramp * spd * 10.0, (H, W, lam.size)).astype(np.float32)
        self.exr = write_spectral_exr(self.tmp / "scene.exr", self.planes, lam)
        self.qe = write_gaussian_qe(self.tmp)
        write_curve(self.tmp / "tau.csv", np.arange(380.0, 781.0, 10.0), np.linspace(0.95, 0.7, 41))
        write_curve(self.tmp / "illum.csv", np.arange(380.0, 781.0, 10.0), np.full(41, 100.0))
        (self.tmp / "manifest.json").write_text(json.dumps({"film": {"xresolution": W, "yresolution": H}}))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _camera(self, calibration: dict, pbrt: dict | None = None, *, spatial: bool = True) -> dict:
        model = {
            "calibration": {"mode": "photon_counting", **calibration},
            "pbrt_spectral_exr": pbrt or {},
        }
        if spatial:
            # Lives under sensor_forward.model in the shipped camera models.
            model["optics_transmittance_spatial"] = {"enabled": True, "edge_factor": 0.6, "exponent": 2.0}
        return {
            "sensor": {
                "pixel_pitch_um": 3.0,
                "f_number": 2.8,
                "integration_time_s": 0.01,
                "fill_factor": 0.9,
                "quantum_efficiency": self.qe,
            },
            # 50 mm focused at 0.1 m → m = 1, (1+m)² = 4.
            "lens": {"focal_length_mm": 50.0, "realistic_focus_distance": 0.1},
            "noise": {},
            "cfa": {"enabled": False},
            "sensor_forward": {"model": model},
        }

    def _both_paths(self, camera: dict, manifest: str = "manifest.json") -> tuple[np.ndarray, np.ndarray, dict]:
        cfg_path = write_yaml(self.tmp / "camera.yaml", camera)
        out = self.tmp / "e.npz"
        run_tool_main(
            pbrt_tool.main,
            [
                "--repo-root",
                str(self.tmp),
                "--exr",
                str(self.exr),
                "--camera-model-config",
                str(cfg_path),
                "--scene-manifest-json",
                str(self.tmp / manifest),
                "--out",
                str(out),
            ],
        )
        npz = np.load(out)
        model = camera["sensor_forward"]["model"]
        e_iq = apply_emva_noise.integrate_exr_spectral_qe(
            self.exr,
            self.tmp,
            self.qe,
            camera["sensor"],
            dict(model["calibration"]),
            lens_cfg=camera["lens"],
            model_cfg=model,
            scene=apply_emva_noise._scene_radiometry(self.tmp / manifest, self.tmp),
        )
        return npz["electrons_rgb"], e_iq, dict(npz)

    def _scene_manifest(self, camera_type: str, light_scale: float = 2.0) -> str:
        name = f"manifest_{camera_type}.json"
        manifest = {
            "film": {"xresolution": W, "yresolution": H},
            "camera": {"type": camera_type, "cam_dist": 4.0, "focus_distance": 0.1},
            "lighting": {"distant": {"from": [0, 0, 3], "to": [0, 0, 0], "scale": light_scale}},
        }
        (self.tmp / name).write_text(json.dumps(manifest))
        return name

    def test_paths_agree_with_every_factor_active(self) -> None:
        camera = self._camera({"optics_transmittance_csv": "tau.csv", "irradiance_scale_W_m2nm_per_unit": 2e-3})
        e_pbrt, e_iq, npz = self._both_paths(camera)
        np.testing.assert_allclose(e_iq, e_pbrt, rtol=1e-6)
        self.assertAlmostEqual(float(npz["magnification_factor"]), 4.0, places=6)
        self.assertEqual(str(npz["optics_transmittance_mode"]), "spectral_csv")
        self.assertTrue(json.loads(str(npz["optics_transmittance_spatial"]))["enabled"])
        # Model-level vignetting map is applied: corners darker than centre column.
        self.assertLess(e_iq[0, 0, 1] / e_iq[H // 2, 0, 1], 0.9)

    def test_magnification_and_spatial_map_scale_signal(self) -> None:
        base = self._camera({}, spatial=False)
        base["lens"] = {}
        _, e_far, _ = self._both_paths(base)
        _, e_macro, _ = self._both_paths(self._camera({}, spatial=False))
        np.testing.assert_allclose(e_macro, e_far / 4.0, rtol=1e-6)

    def test_illuminant_csv_takes_precedence_over_autocalibration(self) -> None:
        cal = {"target_illuminance_lux": 500.0, "illuminant_override_csv": "illum.csv"}
        _, csv_only, _ = self._both_paths(self._camera(cal, spatial=False))
        e_pbrt, e_iq, npz = self._both_paths(
            self._camera(cal, {"radiometric_autocalibration": "mean_photopic_lux"}, spatial=False)
        )
        np.testing.assert_allclose(e_pbrt, csv_only, rtol=1e-6)
        np.testing.assert_allclose(e_iq, csv_only, rtol=1e-6)
        self.assertEqual(float(npz["exr_radiometric_autocalibration_scale"]), 1.0)

    def test_autocalibration_alone_hits_target_mean_illuminance(self) -> None:
        camera = self._camera(
            {"target_illuminance_lux": 500.0, "optics_transmittance": 0.8},
            {"radiometric_autocalibration": "mean_photopic_lux"},
            spatial=False,
        )
        e_pbrt, e_iq, npz = self._both_paths(camera)
        np.testing.assert_allclose(e_iq, e_pbrt, rtol=1e-6)
        irr = float(npz["radiance_to_irradiance"]) * 0.8 * float(npz["exr_radiometric_autocalibration_scale"])
        mean_e = self.planes.astype(np.float64).mean(axis=(0, 1)) * irr * float(npz["photometry_calibration_scale"])
        self.assertAlmostEqual(illuminance_lux_from_irradiance(SPECTRAL_LAMBDAS_NM, mean_e), 500.0, places=3)

    def test_target_lux_without_illuminant_or_autocal_is_rejected(self) -> None:
        camera = self._camera({"target_illuminance_lux": 500.0}, spatial=False)
        with self.assertRaisesRegex(RuntimeError, "illuminant_override_csv"):
            apply_emva_noise.integrate_exr_spectral_qe(
                self.exr,
                self.tmp,
                self.qe,
                camera["sensor"],
                camera["sensor_forward"]["model"]["calibration"],
            )

    def test_scene_manifest_calibrates_chart_lux_in_both_paths(self) -> None:
        cal = {"target_illuminance_lux": 500.0, "illuminant_override_csv": "illum.csv"}
        camera = self._camera(cal, {"radiometric_autocalibration": "mean_photopic_lux"}, spatial=False)
        e_pbrt, e_iq, npz = self._both_paths(camera, self._scene_manifest("pinhole"))
        np.testing.assert_allclose(e_iq, e_pbrt, rtol=1e-6)
        self.assertEqual(str(npz["photometric_calibration"]), "scene_chart_lux")
        expected = 500.0 / (683.0 * pbrt_tool.PBRT_CIE_Y_INTEGRAL * 2.0)
        self.assertAlmostEqual(float(npz["photometry_calibration_scale"]), expected, places=12)
        # Focus distance comes from the scene (0.1 m → m = 1), not lens.realistic_focus_distance.
        camera["lens"] = {"focal_length_mm": 50.0, "realistic_focus_distance": 1000.0}
        e_pbrt_far_lens, e_iq_far_lens, _ = self._both_paths(camera, self._scene_manifest("pinhole"))
        np.testing.assert_allclose(e_iq_far_lens, e_pbrt, rtol=1e-6)
        np.testing.assert_allclose(e_pbrt_far_lens, e_pbrt, rtol=1e-6)

    def test_realistic_scene_skips_thin_lens_factor_in_both_paths(self) -> None:
        cal = {"target_illuminance_lux": 500.0}
        pinhole_pbrt, _, pin = self._both_paths(self._camera(cal, spatial=False), self._scene_manifest("pinhole"))
        e_pbrt, e_iq, npz = self._both_paths(self._camera(cal, spatial=False), self._scene_manifest("realistic"))
        np.testing.assert_allclose(e_iq, e_pbrt, rtol=1e-6)
        self.assertEqual(float(npz["radiance_to_irradiance"]), 1.0)
        rad_to_e = float(pin["radiance_to_irradiance"]) / float(pin["magnification_factor"])
        np.testing.assert_allclose(pinhole_pbrt, e_pbrt * float(pin["radiance_to_irradiance"]), rtol=1e-6)
        self.assertLess(rad_to_e, 1.0)

    def test_missing_scene_manifest_falls_back(self) -> None:
        self.assertIsNone(apply_emva_noise._scene_radiometry(None, self.tmp))
        self.assertIsNone(apply_emva_noise._scene_radiometry(Path("nope.json"), self.tmp))


if __name__ == "__main__":
    unittest.main()
