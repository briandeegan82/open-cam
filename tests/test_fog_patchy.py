"""Opt-in fog banks / patchy fog (--fog-bank, --fog-patchiness; pbrt patch 0014 horizontal modulation)."""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import build_highway_scene as hw
import highway_atmosphere as atm
import numpy as np
from exr_multispectral import read_separate_exr_channels
from synthetic_data import REPO

PBRT = Path(os.environ.get("OPENCAM_PBRT", REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"))


def _params(*extra: str) -> atm.HazeParams:
    return atm.params_from_args(hw.parse_args(["--haze", "fog", *extra]))


class TestModulation(unittest.TestCase):
    def test_uniform_by_default(self):
        p = _params()
        self.assertIsNone(atm.horizontal_modulation(p))
        self.assertFalse(any("modulation" in ln for ln in atm.pbrt_medium_lines(p, "a", "s")))

    def test_bank_edges(self):
        p = _params("--fog-bank", "100:500", "--fog-bank-ramp-m", "40", "--fog-bank-outside", "0.1")
        x, z, m = atm.horizontal_modulation(p)
        col = m[:, x.size // 2]
        f = lambda zz: float(np.interp(zz, z, col))  # noqa: E731
        self.assertAlmostEqual(f(100.0), 0.1 + 0.9 * 0.5, delta=0.01)
        self.assertAlmostEqual(f(80.0), 0.1 + 0.9 * 0.1, delta=0.01)  # 10-90 % over the ramp
        self.assertAlmostEqual(f(120.0), 0.1 + 0.9 * 0.9, delta=0.01)
        self.assertAlmostEqual(f(300.0), 1.0, delta=1e-3)
        self.assertAlmostEqual(f(-100.0), 0.1, delta=1e-3)
        self.assertAlmostEqual(f(1000.0), 0.1, delta=1e-3)
        self.assertTrue(np.allclose(m, m[:, :1]))  # a bank varies along the road only

    def test_open_ended_bank_and_patches(self):
        p = _params("--fog-bank", "50:inf", "--fog-patchiness", "0.6", "--fog-patch-seed", "3")
        x, z, m = atm.horizontal_modulation(p)
        self.assertTrue(m.min() >= 0.0 and m.max() <= 1.0)
        far = m[z > 300]
        self.assertGreater(far.std(), 0.05)  # patches vary across the road
        self.assertGreater(far.mean(), 0.55)
        again = atm.horizontal_modulation(p)[2]
        np.testing.assert_array_equal(m, again)
        lines = atm.pbrt_medium_lines(p, "a", "s")
        self.assertTrue(any('"float modulation"' in ln for ln in lines))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            _params("--fog-bank", "200:100")
        with self.assertRaises(ValueError):
            _params("--fog-patchiness", "1.5")


@unittest.skipUnless(PBRT.is_file(), f"pbrt binary not built ({PBRT}); see docs/BUILD_PBRT.txt")
class TestPbrtModulation(unittest.TestCase):
    def test_transmittance_through_bank(self):
        # Absorbing grid, camera at z = 0 looking at an emitter at z = 300 through m(z) nodes
        # [0.2, 1, 1, 0.2, 0.2] at z = 0..400: tau = sigma * 220 m (vs sigma * 300 m without).
        sigma = 0.003
        mod = " ".join(["0.2 0.2", "1 1", "1 1", "0.2 0.2", "0.2 0.2"])
        scene = f"""MakeNamedMedium "m" "string type" "uniformgrid" "rgb sigma_a" [{sigma} {sigma} {sigma}]
    "rgb sigma_s" [0 0 0] "integer nx" [2] "integer ny" [2] "integer nz" [2]
    "point3 p0" [-50 -50 -1] "point3 p1" [50 50 400] "float density" [1 1 1 1 1 1 1 1]
    "integer modnx" [2] "integer modnz" [5] "point2 modp0" [-50 0] "point2 modp1" [50 400]
    "float modulation" [{mod}]
MediumInterface "m" "m"
LookAt 0 0 0  0 0 1  0 1 0
Camera "orthographic" "float screenwindow" [-1 1 -1 1]
Sampler "independent" "integer pixelsamples" [1024]
Film "rgb" "integer xresolution" [4] "integer yresolution" [4] "string filename" ["t.exr"]
Integrator "volpath" "integer maxdepth" [1]
WorldBegin
AttributeBegin
MediumInterface "m" "m"
AreaLightSource "diffuse" "rgb L" [1 1 1] "bool twosided" true
Shape "trianglemesh" "point3 P" [-20 -20 300  20 -20 300  20 20 300  -20 20 300] "integer indices" [0 1 2 0 2 3]
AttributeEnd
"""
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "t.pbrt").write_text(scene)
            subprocess.run([str(PBRT), "--quiet", "--seed", "5", "t.pbrt"], cwd=d, check=True, timeout=300)
            ch = read_separate_exr_channels(Path(d) / "t.exr")
            t = float(np.mean([np.asarray(ch[k]).mean() for k in ("R", "G", "B")]))
        self.assertAlmostEqual(t, math.exp(-sigma * 220.0), delta=0.02)
        self.assertGreater(abs(t - math.exp(-sigma * 300.0)), 0.08)


if __name__ == "__main__":
    unittest.main()
