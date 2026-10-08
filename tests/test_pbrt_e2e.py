"""End-to-end: build the ColorChecker scene, render it with pbrt-v4 SpectralFilm, convert to electrons.

Skipped when the pbrt binary is missing (set ``OPENCAM_PBRT`` to override the path, or
``OPENCAM_REQUIRE_PBRT=1`` to fail instead of skipping, as CI does).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pbrt_spectral_exr_to_electrons as pbrt_tool
from camera_model import load_camera_model
from exr_multispectral import spectral_buckets_from_exr
from synthetic_data import REPO, run_tool_main

PBRT = Path(os.environ.get("OPENCAM_PBRT", REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"))
XRES, YRES, NBUCKETS = 96, 64, 16
CAMERA_MODEL = REPO / "config" / "camera_recipes" / "default.yaml"

if not PBRT.is_file() and os.environ.get("OPENCAM_REQUIRE_PBRT") == "1":
    raise RuntimeError(f"OPENCAM_REQUIRE_PBRT=1 but pbrt binary not found at {PBRT}")


def _patch_centres_px(manifest: dict) -> list[tuple[int, int]]:
    """Pixel (row, col) of each patch centre, same world→pixel mapping as spectral_sensor_forward."""
    g, cam = manifest["geometry"], manifest["camera"]
    pw, ph, gap = g["patch_width"], g["patch_height"], g["gap"]
    board_w, board_h = g["board_size"]
    tan_half = np.tan(np.deg2rad(cam["fov_deg"]) * 0.5)
    d, aspect = cam["cam_dist"], XRES / YRES
    out = []
    for row in range(4):
        for col in range(6):
            xw = -(-board_w / 2.0 + col * (pw + gap) + pw / 2.0)
            yw = board_h / 2.0 - row * (ph + gap) - ph / 2.0
            x_ndc = -xw / (d * tan_half * aspect)
            y_ndc = yw / (d * tan_half)
            out.append((round((1.0 - y_ndc) / 2.0 * YRES - 0.5), round((x_ndc + 1.0) / 2.0 * XRES - 0.5)))
    return out


@unittest.skipUnless(PBRT.is_file(), f"pbrt binary not built ({PBRT}); see docs/BUILD_PBRT.txt")
class TestPbrtEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.exr = tmp / "cc.exr"
        subprocess.run(
            [
                sys.executable,
                str(REPO / "tools" / "build_colorchecker_scene.py"),
                "--out-dir",
                str(tmp / "scene"),
                "--film",
                "spectral",
                "--film-output",
                str(cls.exr),
                "--xres",
                str(XRES),
                "--yres",
                str(YRES),
                "--pixelsamples",
                "16",
                "--spectral-nbuckets",
                str(NBUCKETS),
                "--spectral-lambda-min",
                "400",
                "--spectral-lambda-max",
                "700",
            ],
            check=True,
            capture_output=True,
        )
        cls.manifest_path = tmp / "scene" / "colorchecker_manifest.json"
        cls.manifest = json.loads(cls.manifest_path.read_text())
        cls.reference = dict(np.load(tmp / "scene" / "spectral_reference_1nm.npz"))
        subprocess.run([str(PBRT), "--quiet", "--seed", "1", str(tmp / "scene" / "colorchecker.pbrt")], check=True)
        out = tmp / "electrons.npz"
        run_tool_main(
            pbrt_tool.main,
            [
                "--exr",
                str(cls.exr),
                "--camera-model-config",
                str(CAMERA_MODEL),
                "--scene-manifest-json",
                str(cls.manifest_path),
                "--out",
                str(out),
            ],
        )
        cls.npz = dict(np.load(out))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_spectral_exr_layout(self) -> None:
        planes, lam = spectral_buckets_from_exr(self.exr)
        self.assertEqual(planes.shape, (YRES, XRES, NBUCKETS))
        self.assertTrue(np.all(np.diff(lam) > 0))
        self.assertTrue(lam[0] > 400.0 and lam[-1] < 700.0)
        self.assertTrue(np.all(np.isfinite(planes)) and np.all(planes >= 0.0))

    def test_electrons_are_physical(self) -> None:
        e = self.npz["electrons_rgb"]
        self.assertEqual(e.shape, (YRES, XRES, 3))
        self.assertTrue(np.all(np.isfinite(e)) and np.all(e >= 0.0))
        self.assertGreater(float(e.mean()), 0.0)

    def test_patch_responses_match_spectral_reference(self) -> None:
        """Per-patch electrons ∝ ∫ E·R_i·QE_c·λ dλ: one scale factor for all 24 patches."""
        e = self.npz["electrons_rgb"].astype(np.float64)
        rendered = np.array(
            [e[r - 1 : r + 2, c - 1 : c + 2].mean(axis=(0, 1)) for r, c in _patch_centres_px(self.manifest)]
        )

        wl = self.reference["wavelength_nm"]
        band = (wl >= 400.0) & (wl <= 700.0)
        wl = wl[band]
        sensor = load_camera_model(CAMERA_MODEL)["sensor"]
        qe = pbrt_tool.qe_stack_on_lambdas(REPO, sensor["quantum_efficiency"], wl)
        spd = self.reference["illuminant"][band] * self.reference["reflectance"][:, band]
        expected = np.einsum("pk,ck,k->pc", spd, qe, wl)

        ratio = rendered / expected
        rel_spread = ratio.std(axis=0) / ratio.mean(axis=0)
        self.assertTrue(np.all(rel_spread < 0.05), f"per-channel patch ratio spread {rel_spread}")
        neutral = rendered[18:24, 1]
        self.assertTrue(np.all(np.diff(neutral) < 0), f"neutral row not monotonic: {neutral}")


@unittest.skipUnless(PBRT.is_file(), f"pbrt binary not built ({PBRT}); see docs/BUILD_PBRT.txt")
class TestPbrtHighwayHaze(unittest.TestCase):
    """The manifest's road illuminance under fog matches what pbrt's volpath delivers to the road."""

    def _probe(self, tmp: Path, name: str, *haze: str) -> tuple[float, dict]:
        """Radiance of an infinite Lambertian ground (albedo = the MC model's) under the scene's lights."""
        import re

        import yaml
        from highway_sky import LUMA, read_rgb_exr

        m = yaml.safe_load((REPO / "config" / "highway_assets.yaml").read_text())
        m["cache_dir"] = str(tmp / "empty_cache")
        (tmp / "assets.yaml").write_text(yaml.safe_dump(m))
        scene = tmp / name
        cmd = [sys.executable, str(REPO / "tools" / "build_highway_scene.py"), "--out-dir", str(scene)]
        cmd += ["--asset-manifest", str(tmp / "assets.yaml"), "--allow-missing-assets", *haze]
        subprocess.run(cmd, check=True, capture_output=True)
        txt = (scene / "highway.pbrt").read_text()
        head = txt[: txt.index("AttributeEnd", txt.index('LightSource "infinite"')) + len("AttributeEnd")]
        exr = tmp / f"{name}.exr"
        head = re.sub(r"^LookAt .*$", "LookAt 0 0.05 0  0 0 0  0 0 1", head, flags=re.M)
        head = re.sub(r"^Camera .*$", 'Camera "perspective" "float fov" [20]', head, flags=re.M)
        head = re.sub(r'"string filename" \[".*?"\]', f'"string filename" ["{exr}"]', head, count=1)
        head = re.sub(r'"integer xresolution" \[\d+\]', '"integer xresolution" [8]', head)
        head = re.sub(r'"integer yresolution" \[\d+\]', '"integer yresolution" [8]', head)
        head = re.sub(r'"integer pixelsamples" \[\d+\]', '"integer pixelsamples" [2048]', head)
        rho = json.loads((scene / "highway_manifest.json").read_text())
        albedo = (rho["atmosphere"] or {}).get("road_illuminance", {}).get("ground_albedo", 0.15)
        w = 150_000
        (scene / "probe.pbrt").write_text(
            head
            + f'\nMaterial "diffuse" "spectrum reflectance" [300 {albedo} 900 {albedo}]\n'
            + f'Shape "trianglemesh" "point3 P" [{-w} 0 {-w} {w} 0 {-w} {-w} 0 {w} {w} 0 {w}]'
            + ' "integer indices" [0 1 2 2 1 3]\n'
        )
        subprocess.run([str(PBRT), "--quiet", "--seed", "1", str(scene / "probe.pbrt")], check=True)
        return float((read_rgb_exr(exr) @ LUMA).mean()), rho

    def test_fog_road_illuminance_matches_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            l_clear, plain = self._probe(tmp, "plain")
            l_fog, fog = self._probe(tmp, "fog", "--haze", "fog")
        rendered = l_fog / l_clear
        expected = fog["lighting"]["reference_illuminance_lux"] / plain["lighting"]["reference_illuminance_lux"]
        self.assertLess(expected, 0.85)
        self.assertAlmostEqual(rendered, expected, delta=0.05 * expected)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(PBRT.is_file(), f"pbrt binary not built ({PBRT}); see docs/BUILD_PBRT.txt")
class TestPbrtHighwaySmoke(unittest.TestCase):
    """Highway builder -> pbrt -> electrons, offline (proxy cars, analytic sky, no textures)."""

    def test_highway_render_and_electrons(self) -> None:
        import yaml

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            m = yaml.safe_load((REPO / "config" / "highway_assets.yaml").read_text())
            m["cache_dir"] = str(tmp / "empty_cache")
            (tmp / "assets.yaml").write_text(yaml.safe_dump(m))
            exr = tmp / "hw.exr"
            subprocess.run(
                [
                    sys.executable,
                    str(REPO / "tools" / "build_highway_scene.py"),
                    "--out-dir",
                    str(tmp / "scene"),
                    "--asset-manifest",
                    str(tmp / "assets.yaml"),
                    "--allow-missing-assets",
                    "--film-output",
                    str(exr),
                    "--xres",
                    "64",
                    "--yres",
                    "36",
                    "--pixelsamples",
                    "4",
                    "--spectral-lambda-min",
                    "400",
                    "--spectral-lambda-max",
                    "700",
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run([str(PBRT), "--quiet", "--seed", "1", str(tmp / "scene" / "highway.pbrt")], check=True)
            L, _ = spectral_buckets_from_exr(exr)
            self.assertEqual(L.shape[:2], (36, 64))
            self.assertTrue(np.isfinite(L).all())
            self.assertGreater(float(L[:12].mean()), float(L[-8:].mean()))  # bright sky above dark asphalt
            out = tmp / "electrons.npz"
            run_tool_main(
                pbrt_tool.main,
                [
                    "--exr",
                    str(exr),
                    "--camera-model-config",
                    str(CAMERA_MODEL),
                    "--scene-manifest-json",
                    str(tmp / "scene" / "highway_manifest.json"),
                    "--out",
                    str(out),
                    "--integration-time-s",
                    "0.0001",
                ],
            )
            npz = dict(np.load(out))
            e = next(v for k, v in npz.items() if np.asarray(v).ndim >= 2)
            self.assertTrue(np.isfinite(e).all())
            self.assertGreater(float(np.mean(e)), 0.0)
