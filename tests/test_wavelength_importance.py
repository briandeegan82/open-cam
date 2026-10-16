"""Sensor-response-weighted wavelength sampling (pbrt patch 0012, tools/wavelength_importance.py)."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import wavelength_importance as wi
import yaml
from exr_multispectral import parse_s0_wavelength_nm, read_separate_exr_channels
from synthetic_data import REPO

PBRT = Path(os.environ.get("OPENCAM_PBRT", REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"))
SONY = REPO / "config/camera_models/sony_alpha_6100.yaml"


def _qe() -> dict:
    return wi.sensor_qe_config(yaml.safe_load(SONY.read_text()))


class TestBucketWeights(unittest.TestCase):
    def test_weights_positive_mean_one_and_follow_response(self):
        w = wi.bucket_weights(_qe(), 360.0, 830.0, 47, 0.25)
        self.assertTrue(np.all(w > 0))
        self.assertAlmostEqual(float(w.mean()), 1.0, places=9)
        self.assertAlmostEqual(float(w.min()), 0.25, places=6)  # IR-cut buckets keep the uniform share
        lam = 360.0 + 10.0 * (np.arange(47) + 0.5)
        self.assertGreater(float(w[(lam > 450) & (lam < 620)].mean()), 1.5)
        with self.assertRaises(ValueError):
            wi.bucket_weights(_qe(), 360.0, 830.0, 47, 0.0)

    def test_scene_edit(self):
        scene = 'Film "spectral" "integer nbuckets" [ 4 ] "float lambdamin" [ 400 ] "float lambdamax" [ 700 ]\n\nWorldBegin\n'
        self.assertEqual(wi.film_params(scene), (400.0, 700.0, 4))
        out = wi.apply_to_scene(scene, np.array([0.5, 1.5, 1.5, 0.5]))
        self.assertIn('Film "spectral" "float bucketweights" [ 0.5 1.5 1.5 0.5 ]', out)
        with self.assertRaises(ValueError):
            wi.apply_to_scene(out, np.ones(4))


@unittest.skipUnless(PBRT.is_file(), f"pbrt binary not built ({PBRT}); see docs/BUILD_PBRT.txt")
class TestPbrtBucketWeights(unittest.TestCase):
    def _render(self, d: Path, scene: str, out: str) -> np.ndarray:
        subprocess.run(
            [str(PBRT), "--quiet", "--seed", "1", "--spp", "256", "--outfile", out, scene],
            cwd=d,
            check=True,
            timeout=600,
        )
        ch = read_separate_exr_channels(d / out)
        names = sorted((k for k in ch if parse_s0_wavelength_nm(k) is not None), key=parse_s0_wavelength_nm)
        return np.array([float(np.asarray(ch[k], np.float64).mean()) for k in names])

    def test_bucket_means_unbiased(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "scene"
            cmd = [sys.executable, str(REPO / "tools/build_colorchecker_scene.py"), "--out-dir", str(d)]
            cmd += ["--camera", "pinhole", "--film", "spectral", "--xres", "48", "--yres", "32"]
            cmd += ["--spectral-nbuckets", "12", "--spectral-lambda-min", "400", "--spectral-lambda-max", "700"]
            subprocess.run(cmd, check=True, timeout=600, capture_output=True)
            scene = next(d.glob("*.pbrt"))
            txt = scene.read_text()
            w = wi.bucket_weights(_qe(), *wi.film_params(txt), 0.25)
            (d / "weighted.pbrt").write_text(wi.apply_to_scene(txt, w))
            uni = self._render(d, scene.name, "u.exr")
            imp = self._render(d, "weighted.pbrt", "w.exr")
            np.testing.assert_allclose(imp / uni, 1.0, atol=0.03)
            (d / "bad.pbrt").write_text(wi.apply_to_scene(txt, np.r_[0.0, np.ones(11)]))
            bad = subprocess.run([str(PBRT), "--quiet", "bad.pbrt"], cwd=d, capture_output=True, text=True, timeout=600)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("bucketweights", bad.stderr + bad.stdout)


if __name__ == "__main__":
    unittest.main()
