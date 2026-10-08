"""Full chain on a real pbrt render: scene → pbrt → electrons → EMVA noise + Bayer → demosaic check.

The exposure is chosen so the brightest patch sits at ~60% of the ADC range, then the raw
is checked against the EMVA1288 model: per-pixel residuals ``DN − (black + e/K)`` normalised
by the predicted temporal + fixed-pattern noise must be ~N(0, 1).
Skipped without the pbrt binary (``OPENCAM_REQUIRE_PBRT=1`` makes that an error).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from camera_model import load_camera_model, noise_config_from_camera_model
from synthetic_data import REPO
from test_pbrt_e2e import PBRT

XRES, YRES = 96, 64
CAMERA_MODEL = REPO / "config" / "camera_models" / "iphone_8.yaml"
TOOLS = REPO / "tools"


def _run(*args: str) -> None:
    subprocess.run([sys.executable, *args], check=True, capture_output=True, cwd=REPO)


@unittest.skipUnless(PBRT.is_file(), f"pbrt binary not built ({PBRT}); see docs/BUILD_PBRT.txt")
class TestPbrtFullPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = cls.tmp = Path(cls._tmp.name)
        exr, scene = tmp / "cc.exr", tmp / "scene"
        manifest = scene / "colorchecker_manifest.json"
        _run(
            str(TOOLS / "build_colorchecker_scene.py"),
            *("--out-dir", str(scene), "--film", "spectral", "--film-output", str(exr)),
            *("--xres", str(XRES), "--yres", str(YRES), "--pixelsamples", "16"),
            *("--spectral-nbuckets", "16", "--spectral-lambda-min", "400", "--spectral-lambda-max", "700"),
        )
        subprocess.run([str(PBRT), "--quiet", "--seed", "1", str(scene / "colorchecker.pbrt")], check=True)

        camera_model = load_camera_model(CAMERA_MODEL)
        emva, adc = camera_model["noise"]["emva"], camera_model["noise"]["adc"]
        cls.K = float(emva["overall_system_gain_K_e_per_DN"])
        cls.black = float(emva["black_level_DN"])
        cls.white_dn = 2 ** int(adc["bit_depth"]) - 1
        cls.emva = emva

        def electrons(t_s: float) -> Path:
            out = tmp / f"e_{t_s:.6g}.npz"
            _run(
                str(TOOLS / "pbrt_spectral_exr_to_electrons.py"),
                *("--exr", str(exr), "--camera-model-config", str(CAMERA_MODEL)),
                *("--scene-manifest-json", str(manifest), "--out", str(out), "--integration-time-s", str(t_s)),
            )
            return out

        e_probe = np.load(electrons(0.1))["electrons_rgb"]
        cls.t_s = 0.1 * 0.6 * (cls.white_dn - cls.black) * cls.K / float(np.percentile(e_probe, 99.5))
        cls.e_npz = electrons(cls.t_s)
        cls.e_clean = np.load(cls.e_npz)["electrons_rgb"].astype(np.float64)

        cls.noise_cfg = tmp / "noise.yaml"
        cls.noise_cfg.write_text(
            yaml.safe_dump(noise_config_from_camera_model(camera_model, str(exr), str(tmp / "noisy.raw16")))
        )
        cls.raw = cls._noise(seed=0)
        cls.stats = json.loads((tmp / "noisy_png" / "run_stats.json").read_text())
        cls.pattern = cls.stats["bayer_pattern"]

        _run(
            str(TOOLS / "validate_demosaic_linear.py"),
            *("--config", str(cls.noise_cfg), "--electrons-npz", str(cls.e_npz)),
            *("--manifest", str(manifest), "--json-out", str(tmp / "demosaic.json")),
        )
        cls.demosaic = json.loads((tmp / "demosaic.json").read_text())

    @classmethod
    def _noise(cls, seed: int) -> np.ndarray:
        _run(
            str(TOOLS / "apply_emva_noise.py"),
            *("--config", str(cls.noise_cfg), "--electrons-npz", str(cls.e_npz)),
            *("--seed", str(seed), "--integration-time-s", str(cls.t_s)),
        )
        return np.fromfile(cls.tmp / "noisy.raw16", dtype="<u2").reshape(YRES, XRES)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _clean_mosaic_e(self) -> np.ndarray:
        idx = {"R": 0, "G": 1, "B": 2}
        out = np.empty((YRES, XRES))
        for k, ch in enumerate(self.pattern):
            dy, dx = divmod(k, 2)
            out[dy::2, dx::2] = self.e_clean[dy::2, dx::2, idx[ch]]
        return out

    def test_raw_layout_and_metadata(self) -> None:
        self.assertEqual(self.raw.shape, (YRES, XRES))
        self.assertTrue(self.stats["bayer_enabled"])
        self.assertEqual(self.stats["K_effective_e_per_DN"], self.K)
        self.assertLessEqual(int(self.raw.max()), self.white_dn)
        clipped = np.mean(self.raw >= self.white_dn)
        self.assertLess(clipped, 0.01, "exposure should keep the chart below ADC clipping")

    def test_raw_matches_emva_noise_model(self) -> None:
        e = self._clean_mosaic_e()
        emva = self.emva
        dark_e = float(emva.get("dark_current_e_per_s", 0.0)) * self.t_s
        mu_e = e + dark_e
        var_e = (
            mu_e
            + float(emva["sigma_d_e"]) ** 2
            + (float(emva.get("prnu_std_fraction", 0.0)) * e) ** 2
            + float(emva.get("dsnu_std_e", 0.0)) ** 2
        )
        var_dn = var_e / self.K**2 + 1.0 / 12.0
        raw = self.raw.astype(np.float64)
        valid = (raw > 0) & (raw < self.white_dn)
        z = ((raw - (self.black + mu_e / self.K)) / np.sqrt(var_dn))[valid]
        self.assertLess(abs(float(z.mean())), 0.1, "gain or black level off")
        self.assertAlmostEqual(float(z.var()), 1.0, delta=0.15, msg="noise variance off EMVA prediction")

    def test_seed_determinism(self) -> None:
        np.testing.assert_array_equal(self._noise(seed=0), self.raw)
        self.assertFalse(np.array_equal(self._noise(seed=1), self.raw))
        self._noise(seed=0)

    def test_demosaic_fidelity_on_patch_interiors(self) -> None:
        masked = self.demosaic["masked"]
        self.assertGreater(masked["pixel_count"], 1000)
        self.assertLess(masked["rmse_dn"], 0.02 * (self.white_dn - self.black))


if __name__ == "__main__":
    unittest.main()
