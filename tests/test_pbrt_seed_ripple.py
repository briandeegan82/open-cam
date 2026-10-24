"""pbrt ``--seed`` reaches the sampler, and spectral-film buckets one stratum apart are decorrelated."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from exr_multispectral import spectral_buckets_from_exr  # noqa: E402

PBRT = Path(os.environ.get("OPENCAM_PBRT", REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"))
PBRT_TIMEOUT_S = int(os.environ.get("OPENCAM_PBRT_TIMEOUT", "600"))

SCENE = """LookAt 0 0 5  0 0 0  0 1 0
Camera "perspective" "float fov" [30]
Sampler "zsobol" "integer pixelsamples" [16]
Film "spectral" "integer xresolution" [48] "integer yresolution" [48] "integer nbuckets" [16]
    "float lambdamin" [400] "float lambdamax" [720] "string filename" "out.exr" {film}
WorldBegin
LightSource "distant" "point3 from" [1 1 2] "point3 to" [0 0 0] "blackbody L" [5500]
Material "diffuse"
Shape "sphere" "float radius" [1.2]
"""


CHECKER_SCENE = """LookAt 0 3 6  0 0 0  0 1 0
Camera "perspective" "float fov" [30]
Sampler "zsobol" "integer pixelsamples" [64]
Film "spectral" "integer xresolution" [64] "integer yresolution" [64] "integer nbuckets" [16]
    "float lambdamin" [400] "float lambdamax" [720] "string filename" "out.exr" {film}
WorldBegin
AttributeBegin
Translate 1.5 2.5 1
LightSource "point" "blackbody I" [3000] "float scale" [20]
AttributeEnd
Texture "chk" "spectrum" "checkerboard" "float uscale" [40] "float vscale" [40]
    "rgb tex1" [0.9 0.9 0.9] "rgb tex2" [0.05 0.05 0.05]
Material "diffuse" "texture reflectance" "chk"
Shape "bilinearmesh" "point3 P" [-4 0 -4 4 0 -4 -4 0 4 4 0 4] "point2 uv" [0 0 1 0 0 1 1 1]
"""


def test_builders_do_not_pin_the_pbrt_seed():
    # An in-scene ``Option "seed"`` overrides pbrt's --seed (pbrt-v4 scene.cpp), which made --seed a no-op.
    for path in sorted((REPO / "tools").glob("build_*.py")):
        assert not re.search(r"Option \\?\"seed\\?\"", path.read_text()), path.name


@unittest.skipUnless(PBRT.is_file(), f"pbrt binary not built ({PBRT})")
class TestPbrtSeedAndRipple(unittest.TestCase):
    def _render(self, tmp: Path, seed: int, film: str = "") -> np.ndarray:
        (tmp / "s.pbrt").write_text(SCENE.format(film=film))
        subprocess.run(
            [str(PBRT), "--quiet", "--seed", str(seed), "s.pbrt"], cwd=tmp, check=True, timeout=PBRT_TIMEOUT_S
        )
        cube, _ = spectral_buckets_from_exr(tmp / "out.exr")
        return np.asarray(cube, dtype=np.float64)

    def test_seed_changes_render(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = self._render(Path(td), 1), self._render(Path(td), 2)
        self.assertGreater(np.abs(a - b).max(), 0.0)

    def test_buckets_one_stratum_apart_are_independent(self):
        # 16 buckets, 4 strata: with stock pbrt sampling buckets k and k + 4 share every path, so their
        # Monte Carlo noise (difference of two seeds) is fully correlated.
        def lag4_noise_corr(film: str) -> float:
            with tempfile.TemporaryDirectory() as td:
                a, b = self._render(Path(td), 1, film), self._render(Path(td), 2, film)
            n = (a - b)[a.mean(axis=2) > 0.2 * a.mean(axis=2).max()]
            return float(np.mean([np.corrcoef(n[:, k], n[:, k + 4])[0, 1] for k in range(12)]))

        self.assertGreater(lag4_noise_corr('"bool jitterwavelengths" false'), 0.9)
        self.assertLess(abs(lag4_noise_corr("")), 0.2)
        self.assertLess(abs(lag4_noise_corr('"string wavelengthjitter" "sample"')), 0.2)

    def test_pixel_rotation_keeps_buckets_stratified(self):
        # Patch 0023: with one wavelength offset per pixel, which bucket a sample lands in is set by
        # its first wavelength sample alone, so each bucket keeps the sampler's stratification. A
        # fresh offset per sample (patch 0006, "sample") loses it. Fixed seeds, so deterministic.
        scene = CHECKER_SCENE

        def bucket_noise(film: str) -> float:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                cubes = []
                for seed in (1, 2):
                    (tmp / "s.pbrt").write_text(scene.format(film=film))
                    subprocess.run(
                        [str(PBRT), "--quiet", "--seed", str(seed), "s.pbrt"],
                        cwd=tmp,
                        check=True,
                        timeout=PBRT_TIMEOUT_S,
                    )
                    cube, _ = spectral_buckets_from_exr(tmp / "out.exr")
                    cubes.append(np.asarray(cube, dtype=np.float64))
            a, b = cubes
            lit = a.mean(axis=2) > 0.05 * a.mean(axis=2).max()
            n, mu = (a - b)[lit], ((a + b) / 2)[lit]
            return float(np.sqrt((n**2).mean(axis=0) / 2).mean() / mu.mean())

        pixel, sample = bucket_noise(""), bucket_noise('"string wavelengthjitter" "sample"')
        self.assertLess(pixel, 0.99 * sample)
