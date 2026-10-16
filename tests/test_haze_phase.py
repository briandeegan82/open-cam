"""Rayleigh + Henyey-Greenstein haze phase mixture (pbrt patch 0011, highway_atmosphere --haze-phase)."""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import highway_atmosphere as atm
import numpy as np
from exr_multispectral import read_separate_exr_channels
from synthetic_data import REPO

PBRT = Path(os.environ.get("OPENCAM_PBRT", REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"))


def _hg(c: float, g: float) -> float:
    return (1 - g * g) / (4 * math.pi * (1 + g * g - 2 * g * c) ** 1.5)


class TestPhaseParams(unittest.TestCase):
    def test_default_is_effective_single_lobe(self):
        p = atm.PRESETS["clear"]
        self.assertEqual(atm.phase_params(p), (atm.effective_g(p), 0.0))
        self.assertFalse(any("rayleighfraction" in ln for ln in atm.pbrt_medium_lines(p, "a.spd", "s.spd")))

    def test_mixed_uses_aerosol_g_and_molecular_share(self):
        for name in ("clear", "fog"):
            p = atm.PRESETS[name]
            m = atm.phase_params(atm.replace(p, phase="mixed"))
            _, ss, _ = atm.coefficients(p, 550.0)
            self.assertEqual(m[0], p.g)
            self.assertAlmostEqual(m[1], atm.RAYLEIGH_550_PER_M / float(ss), places=12)
            # Same first moment as the effective lobe (Rayleigh has zero mean cosine).
            self.assertAlmostEqual((1 - m[1]) * m[0], atm.effective_g(p), places=12)
        lines = atm.pbrt_medium_lines(atm.replace(atm.PRESETS["clear"], phase="mixed"), "a.spd", "s.spd")
        self.assertTrue(any('"float rayleighfraction"' in ln for ln in lines))

    def test_rayleigh_sampler_matches_distribution(self):
        rng = np.random.default_rng(1)
        mu = atm._sample_rayleigh(np.ones(400_000), rng)  # incoming along +z: samples are cos(theta)
        hist, edges = np.histogram(mu, bins=20, range=(-1, 1), density=True)
        c = 0.5 * (edges[1:] + edges[:-1])
        np.testing.assert_allclose(hist, 3 / 8 * (1 + c * c), rtol=0.03)

    def test_cli_flag(self):
        import build_highway_scene as hw

        p = atm.params_from_args(hw.parse_args(["--haze", "clear", "--haze-phase", "mixed"]))
        self.assertEqual(p.phase, "mixed")


@unittest.skipUnless(PBRT.is_file(), f"pbrt binary not built ({PBRT}); see docs/BUILD_PBRT.txt")
class TestPbrtPhaseMixture(unittest.TestCase):
    def _render(self, tmp: Path, name: str, light: str, medium: str, maxdepth: int, spp: int) -> float:
        (tmp / f"{name}.pbrt").write_text(
            f"""LookAt 0 0 5  0 0 0  0 1 0
Camera "orthographic" "float screenwindow" [-0.5 0.5 -0.5 0.5]
Sampler "independent" "integer pixelsamples" [{spp}]
Film "rgb" "integer xresolution" [16] "integer yresolution" [16] "string filename" ["{name}.exr"]
Integrator "volpath" "integer maxdepth" [{maxdepth}]
WorldBegin
{light}
MakeNamedMedium "m" "string type" "homogeneous" "rgb sigma_a" [0 0 0] {medium}
AttributeBegin
MediumInterface "m" ""
Material "interface"
Shape "sphere" "float radius" 1
AttributeEnd
"""
        )
        subprocess.run([str(PBRT), "--quiet", "--seed", "3", f"{name}.pbrt"], cwd=tmp, check=True, timeout=600)
        ch = read_separate_exr_channels(tmp / f"{name}.exr")
        return float(np.mean([np.asarray(ch[k]).mean() for k in ("R", "G", "B")]))

    def test_single_scatter_angular_ratio_and_furnace(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            thin = '"rgb sigma_s" [0.01 0.01 0.01]'
            for th in (30.0, 90.0):
                t = math.radians(th)
                light = (
                    f'LightSource "distant" "point3 from" [{math.sin(t):.6f} 0 {-math.cos(t):.6f}]'
                    ' "point3 to" [0 0 0] "rgb L" [1 1 1]'
                )
                iso = self._render(tmp, f"iso{th:.0f}", light, thin + ' "float g" [0]', 1, 64)
                mix = self._render(
                    tmp, f"mix{th:.0f}", light, thin + ' "float g" [0.6] "float rayleighfraction" [0.5]', 1, 64
                )
                c = math.cos(t)
                expect = 4 * math.pi * (0.5 * 3 / (16 * math.pi) * (1 + c * c) + 0.5 * _hg(c, 0.6))
                self.assertAlmostEqual(mix / iso, expect, delta=0.01 * expect)
            furnace = self._render(
                tmp,
                "furnace",
                'LightSource "infinite" "rgb L" [1 1 1]',
                '"rgb sigma_s" [3 3 3] "float g" [0.6] "float rayleighfraction" [0.5]',
                200,
                32,
            )
            self.assertAlmostEqual(furnace, 1.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
