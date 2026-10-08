"""Module tests: run the PSF, PBRT-EXR-to-electrons and pipeline tools end to end on synthetic inputs."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import apply_spectral_psf  # noqa: E402  (synthetic_data puts tools/ on sys.path)
import numpy as np
import pbrt_spectral_exr_to_electrons  # noqa: E402
import run_pipeline  # noqa: E402
import yaml
from exr_multispectral import read_separate_exr_channels, trapezoid_weights_nm  # noqa: E402
from sensor_radiometry import C_LIGHT, H_PLANCK  # noqa: E402
from synthetic_data import (
    REPO,
    SPECTRAL_LAMBDAS_NM,
    run_tool_main,
    spectral_channel_name,
    write_flat_qe,
    write_spectral_exr,
    write_yaml,
)


class _TmpDirCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()


class TestApplySpectralPsfModule(_TmpDirCase):
    LAMBDAS = np.array([450.0, 550.0, 650.0])
    SIZE = 65

    def setUp(self) -> None:
        super().setUp()
        planes = np.zeros((self.SIZE, self.SIZE, self.LAMBDAS.size), dtype=np.float32)
        c = self.SIZE // 2
        planes[c, c, :] = 1.0
        rgb = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.float32)
        rgb[c, c, :] = 1.0
        self.exr_in = write_spectral_exr(self.tmp / "in.exr", planes, self.LAMBDAS, rgb=rgb)
        self.exr_out = self.tmp / "out.exr"

    def _run(self, post_psf: dict) -> dict[str, np.ndarray]:
        cfg = write_yaml(self.tmp / "optics.yaml", {"post_psf": post_psf})
        run_tool_main(
            apply_spectral_psf.main,
            ["--config", str(cfg), "--exr-in", str(self.exr_in), "--exr-out", str(self.exr_out)],
        )
        return read_separate_exr_channels(self.exr_out)

    def _spectral(self, chans: dict[str, np.ndarray]) -> list[np.ndarray]:
        return [chans[spectral_channel_name(lam)] for lam in self.LAMBDAS]

    def test_gaussian_blurs_every_channel_identically_and_conserves_energy(self) -> None:
        chans = self._run({"enabled": True, "mode": "gaussian", "sigma_pixels": 1.5})
        self.assertEqual(set(chans), set(read_separate_exr_channels(self.exr_in)))
        ref = chans["G"]
        for name, arr in chans.items():
            np.testing.assert_allclose(arr, ref, atol=1e-7, err_msg=name)
            self.assertAlmostEqual(float(arr.sum()), 1.0, places=4)
        self.assertLess(float(ref.max()), 0.2)

    def test_chromatic_gaussian_blurs_longer_wavelengths_more(self) -> None:
        chans = self._run({"enabled": True, "mode": "chromatic_gaussian", "f_number": 8.0, "pixel_pitch_um": 2.0})
        peaks = [float(a.max()) for a in self._spectral(chans)]
        self.assertGreater(peaks[0], peaks[1])
        self.assertGreater(peaks[1], peaks[2])
        for arr in self._spectral(chans):
            self.assertAlmostEqual(float(arr.sum()), 1.0, places=4)

    def test_airy_disk_widens_with_wavelength(self) -> None:
        chans = self._run({"enabled": True, "mode": "airy_disk", "f_number": 4.0, "pixel_pitch_um": 2.0})
        peaks = [float(a.max()) for a in self._spectral(chans)]
        self.assertGreater(peaks[0], peaks[2])
        for arr in self._spectral(chans):
            self.assertAlmostEqual(float(arr.sum()), 1.0, places=3)

    def test_disabled_psf_exits_without_writing(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self._run({"enabled": False, "mode": "gaussian", "sigma_pixels": 1.0})
        self.assertEqual(ctx.exception.code, 0)
        self.assertFalse(self.exr_out.exists())

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._run({"enabled": True, "mode": "bokeh"})


class TestPbrtSpectralExrToElectronsModule(_TmpDirCase):
    H, W = 6, 8
    F_NUMBER = 2.0
    PITCH_UM = 2.0
    T_INT = 0.02
    QE = 0.5
    IRR_SCALE = 1e-3
    OPTICS_T = 0.9

    def setUp(self) -> None:
        super().setUp()
        lam = SPECTRAL_LAMBDAS_NM
        ramp = np.linspace(0.5, 1.5, self.W, dtype=np.float32)[None, :, None]
        spectrum = (1.0 + (lam - 400.0) / 300.0).astype(np.float32)[None, None, :]
        self.planes = np.broadcast_to(ramp * spectrum, (self.H, self.W, lam.size)).astype(np.float32) * 100.0
        self.exr = write_spectral_exr(self.tmp / "render.exr", self.planes, lam)
        sensor = {
            "pixel_pitch_um": self.PITCH_UM,
            "f_number": self.F_NUMBER,
            "integration_time_s": self.T_INT,
            "fill_factor": 1.0,
            "quantum_efficiency": write_flat_qe(self.tmp, self.QE),
        }
        self.noise_cfg = write_yaml(self.tmp / "noise.yaml", {"sensor": sensor})
        self.sensor_cfg = write_yaml(
            self.tmp / "sensor_forward.yaml",
            {
                "model": {
                    "calibration": {
                        "mode": "photon_counting",
                        "irradiance_scale_W_m2nm_per_unit": self.IRR_SCALE,
                        "optics_transmittance": self.OPTICS_T,
                    },
                    "pbrt_spectral_exr": {"radiance_to_irradiance": "thin_lens"},
                }
            },
        )
        self.manifest = self.tmp / "manifest.json"
        self.manifest.write_text(json.dumps({"film": {"xresolution": self.W, "yresolution": self.H}}))
        self.out = self.tmp / "electrons.npz"

    def _run(self, *extra: str) -> np.lib.npyio.NpzFile:
        run_tool_main(
            pbrt_spectral_exr_to_electrons.main,
            [
                "--exr",
                str(self.exr),
                "--sensor-config",
                str(self.sensor_cfg),
                "--noise-config",
                str(self.noise_cfg),
                "--scene-manifest-json",
                str(self.manifest),
                "--out",
                str(self.out),
                *extra,
            ],
        )
        return np.load(self.out)

    def _expected(self, t_int: float) -> np.ndarray:
        lam = SPECTRAL_LAMBDAS_NM.astype(np.float64)
        irr = self.planes.astype(np.float64) * (np.pi / (4.0 * self.F_NUMBER**2)) * self.OPTICS_T * self.IRR_SCALE
        photons = irr * (lam * 1e-9) / (H_PLANCK * C_LIGHT)
        per_px = np.sum(photons * self.QE * trapezoid_weights_nm(lam), axis=2) * t_int * (self.PITCH_UM * 1e-6) ** 2
        return np.repeat(per_px[:, :, None], 3, axis=2)

    def test_electrons_match_closed_form_photon_count(self) -> None:
        data = self._run()
        e = data["electrons_rgb"]
        self.assertEqual(e.shape, (self.H, self.W, 3))
        self.assertEqual(e.dtype, np.float32)
        np.testing.assert_allclose(e, self._expected(self.T_INT), rtol=1e-5)
        self.assertTrue(self.out.with_suffix(".png").is_file())

    def test_integration_time_override_scales_linearly(self) -> None:
        base = self._run()["electrons_rgb"].copy()
        doubled = self._run("--integration-time-s", str(2 * self.T_INT))["electrons_rgb"]
        np.testing.assert_allclose(doubled, 2.0 * base, rtol=1e-6)

    def test_resolution_mismatch_with_manifest_is_rejected(self) -> None:
        self.manifest.write_text(json.dumps({"film": {"xresolution": self.W + 1, "yresolution": self.H}}))
        with self.assertRaisesRegex(ValueError, "does not match manifest"):
            self._run()


class TestRunPipelineDryRun(_TmpDirCase):
    def test_dry_run_records_every_stage_without_executing(self) -> None:
        cfg = yaml.safe_load((REPO / "config" / "pipeline.yaml").read_text())
        cfg["paths"]["out_dir"] = str(self.tmp)
        cfg_path = write_yaml(self.tmp / "pipeline.yaml", cfg)
        stdout = run_tool_main(run_pipeline.main, ["--config", str(cfg_path), "--dry-run", "--name", "unittest"])
        manifests = sorted(self.tmp.glob("run_unittest_*.json"))
        self.assertEqual(len(manifests), 1)
        manifest = json.loads(manifests[0].read_text())
        self.assertTrue(manifest["dry_run"])
        cmds = manifest["commands"]
        self.assertGreaterEqual(len(cmds), 3)
        for entry in cmds:
            self.assertTrue(entry["dry_run"])
            self.assertIsNone(entry["returncode"])
        scripts = [Path(c["cmd"][1]).name for c in cmds if len(c["cmd"]) > 1]
        self.assertIn("apply_emva_noise.py", scripts)
        self.assertEqual(stdout.count("$ "), len(cmds))


if __name__ == "__main__":
    unittest.main()
