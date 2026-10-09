"""Fast analytic and pbrt accurate modes agree on absolute electrons at the same chart lux.

Renders the ColorChecker with pbrt-v4 through a pinhole and through the traced 50 mm
double-Gauss (Nikon Z6) and 22 mm wide (iPhone 8) lenses, converts each with
``pbrt_spectral_exr_to_electrons`` and compares the central patches with
``spectral_sensor_forward`` (no ray tracing), whose f-number is traced from the same lens file.
Skipped without pbrt.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pbrt_spectral_exr_to_electrons as pbrt_tool
import spectral_sensor_forward as analytic_tool
from synthetic_data import REPO, run_tool_main
from test_pbrt_e2e import PBRT, XRES, YRES, _patch_centres_px

RECIPES = REPO / "config" / "camera_recipes"
# recipe -> (lens file, pbrt stop diameter mm, focal length mm); matches the recipe's lens model.
REALISTIC = {
    "nikon_z6": ("config/lenses/dgauss.50mm.dat", 25.0, 50.0),
    "iphone_8": ("config/lenses/wide_22mm.dat", 8.756, 22.0),
}
CENTRAL_PATCHES = [8, 9, 14, 15]


def _common(recipe: str) -> list[str]:
    return [
        "--camera-model-config",
        str(RECIPES / f"{recipe}.yaml"),
        "--target-illuminance-lux",
        "1000",
        "--integration-time-s",
        "0.01",
    ]


if not PBRT.is_file() and os.environ.get("OPENCAM_REQUIRE_PBRT") == "1":
    raise RuntimeError(f"OPENCAM_REQUIRE_PBRT=1 but pbrt binary not found at {PBRT}")


def _render(tmp: Path, tag: str, extra: list[str]) -> Path:
    out = tmp / tag
    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_colorchecker_scene.py"),
            "--out-dir",
            str(out),
            "--film",
            "spectral",
            "--film-output",
            str(out / "cc.exr"),
            "--xres",
            str(XRES),
            "--yres",
            str(YRES),
            "--pixelsamples",
            "64",
            # Recipe calibration uses illuminant_override_csv D65, so light the scene with D65 too.
            "--illuminant",
            str(REPO / "spectra" / "illuminant" / "interpolated" / "D65.csv"),
            # Full 360-830 nm range so blue QE below 400 nm is rendered as in the analytic model.
            "--spectral-nbuckets",
            "47",
            "--spectral-lambda-min",
            "360",
            "--spectral-lambda-max",
            "830",
            *extra,
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run([str(PBRT), "--quiet", "--seed", "1", str(out / "colorchecker.pbrt")], check=True)
    return out


def _central(e: np.ndarray, centres: list[tuple[int, int]]) -> np.ndarray:
    return np.array(
        [e[r - 1 : r + 2, c - 1 : c + 2].mean(axis=(0, 1)) for r, c in (centres[i] for i in CENTRAL_PATCHES)]
    )


@unittest.skipUnless(PBRT.is_file(), f"pbrt binary not built ({PBRT}); see docs/BUILD_PBRT.txt")
class TestChartLuxAgreement(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import json

        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        scenes = {"pinhole": _render(tmp, "pinhole", [])}
        # Same framing as the 35° pinhole at 4.25 m on pbrt's 35 mm-diagonal 3:2 film.
        short_mm = 35.0 * 2.0 / math.sqrt(13.0)
        for recipe, (lensfile, aperture, focal) in REALISTIC.items():
            dist = 4.25 * math.tan(math.radians(17.5)) / ((short_mm / 2.0) / focal)
            scenes[recipe] = _render(
                tmp,
                recipe,
                ["--camera", "realistic", "--lensfile", lensfile, "--aperture-diameter-mm", f"{aperture}"]
                + ["--cam-dist", f"{dist}", "--focus-distance", f"{dist}"],
            )
        pinhole = scenes["pinhole"]
        cls.centres = _patch_centres_px(json.loads((pinhole / "colorchecker_manifest.json").read_text()))
        cls.analytic = {}
        for recipe in REALISTIC:
            run_tool_main(
                analytic_tool.main,
                [
                    *_common(recipe),
                    "--scene-manifest-json",
                    str(pinhole / "colorchecker_manifest.json"),
                    "--spectral-reference-npz",
                    str(pinhole / "spectral_reference_1nm.npz"),
                    "--out",
                    str(tmp / f"analytic_{recipe}.npz"),
                ],
            )
            cls.analytic[recipe] = _central(np.load(tmp / f"analytic_{recipe}.npz")["electrons_rgb"], cls.centres)
        cls.pbrt = {}
        for name, scene in scenes.items():
            recipe = "nikon_z6" if name == "pinhole" else name
            out = tmp / f"{name}.npz"
            run_tool_main(
                pbrt_tool.main,
                [
                    "--exr",
                    str(scene / "cc.exr"),
                    *_common(recipe),
                    "--scene-manifest-json",
                    str(scene / "colorchecker_manifest.json"),
                    "--out",
                    str(out),
                ],
            )
            cls.pbrt[name] = dict(np.load(out))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_pinhole_render_matches_analytic_electrons(self) -> None:
        npz = self.pbrt["pinhole"]
        self.assertEqual(str(npz["photometric_calibration"]), "scene_chart_lux")
        ratio = _central(npz["electrons_rgb"], self.centres) / self.analytic["nikon_z6"]
        np.testing.assert_allclose(ratio, 1.0, atol=0.06)

    def test_realistic_lens_matches_analytic_with_traced_f_number(self) -> None:
        for recipe in REALISTIC:
            with self.subTest(recipe=recipe):
                npz = self.pbrt[recipe]
                self.assertEqual(str(npz["radiance_to_irradiance_mode"]), "pbrt_film_irradiance")
                self.assertEqual(float(npz["radiance_to_irradiance"]), 1.0)
                # Near-axis patches: only vignetting and the off-axis pupil remain (focal/aperture gave 0.77 for iPhone 8).
                ratio = _central(npz["electrons_rgb"], self.centres) / self.analytic[recipe]
                self.assertTrue(np.all((ratio > 0.9) & (ratio < 1.05)), ratio)


if __name__ == "__main__":
    unittest.main()
