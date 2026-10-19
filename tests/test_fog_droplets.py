"""Opt-in Mie droplet fog (tools/fog_mie.py, --fog-droplets, pbrt patch 0013 tabulated phase)."""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import fog_mie
import highway_atmosphere as atm
import numpy as np
from exr_multispectral import read_separate_exr_channels
from synthetic_data import REPO

PBRT = Path(os.environ.get("OPENCAM_PBRT", REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"))


class TestMie(unittest.TestCase):
    def test_bohren_huffman_reference(self):
        # BHMIE test case (Bohren & Huffman 1983, app. A): m = 1.55, x = 5.213 -> Q_ext = Q_sca = 3.105.
        qe, qs = fog_mie.efficiencies(1.55 + 0j, np.array([5.213]))
        self.assertAlmostEqual(float(qe[0]), 3.105, delta=2e-3)
        self.assertAlmostEqual(float(qs[0]), float(qe[0]), places=9)

    def test_extinction_paradox(self):
        qe, _ = fog_mie.efficiencies(1.33 + 0j, np.array([1000.0]))
        self.assertAlmostEqual(float(qe[0]), 2.0, delta=0.03)

    def test_phase_table_normalised_and_forward_peaked(self):
        for name in fog_mie.DROPLETS:
            e, v = fog_mie.phase_table(name)
            self.assertAlmostEqual(2 * math.pi * float((v * np.diff(e)).sum()), 1.0, places=9)
            g = fog_mie.asymmetry(e, v)
            self.assertTrue(0.8 < g < 0.9, (name, g))
            # Mie fog: far more back scatter and less side scatter than HG with the same g.
            hg = lambda c, g=g: (1 - g * g) / (4 * math.pi * (1 + g * g - 2 * g * c) ** 1.5)  # noqa: E731
            self.assertGreater(v[0], 5 * hg(-1.0))
            self.assertLess(v[np.searchsorted(e, 0.0)], 0.5 * hg(0.0))

    def test_sampler_matches_table(self):
        e, v = fog_mie.phase_table("radiation")
        c = fog_mie.sample_cos(e, v, np.random.default_rng(2).random(400_000))
        coarse = np.array([-1.0, -0.5, 0.0, 0.5, 0.9, 0.99, 1.0])
        hist = np.histogram(c, bins=coarse)[0] / c.size
        cdf = np.concatenate([[0.0], np.cumsum(v * np.diff(e))]) * 2 * math.pi
        expect = np.diff(np.interp(coarse, e, cdf))
        np.testing.assert_allclose(hist, expect, rtol=0.03, atol=2e-4)

    def test_droplet_extinction_is_grey(self):
        rel, alb = fog_mie.extinction_shape("radiation", np.array([400.0, 550.0, 1000.0]))
        self.assertAlmostEqual(float(rel[1]), 1.0, places=9)
        self.assertTrue(np.all(np.abs(rel - 1) < 0.05))
        self.assertTrue(np.all(alb > 0.999))  # water absorbs weakly near 950 nm


class TestDropletParams(unittest.TestCase):
    def test_cli_and_medium_lines(self):
        import build_highway_scene as hw

        p = atm.params_from_args(hw.parse_args(["--haze", "fog", "--fog-droplets", "advection"]))
        self.assertEqual(p.droplets, "advection")
        sa, ss, se = atm.coefficients(p, 550.0)
        self.assertAlmostEqual(float(se), atm.extinction_550(p), delta=1e-6 * float(se))
        lines = atm.pbrt_medium_lines(p, "a.spd", "s.spd")
        self.assertTrue(any('"float phasetheta"' in ln for ln in lines))
        self.assertTrue(any('"float phasevalues"' in ln for ln in lines))
        default = atm.pbrt_medium_lines(atm.PRESETS["fog"], "a.spd", "s.spd")
        self.assertFalse(any("phasetheta" in ln for ln in default))

    def test_slab_mc_runs_with_table(self):
        p = atm.replace(atm.PRESETS["fog"], droplets="radiation")
        rng = np.random.default_rng(0)
        tot, direct = atm.slab_flux_mc(p, 550.0, np.ones(20_000), 0.1, rng)
        self.assertTrue(0 < direct < tot < 1.2, (tot, direct))


@unittest.skipUnless(PBRT.is_file(), f"pbrt binary not built ({PBRT}); see docs/BUILD_PBRT.txt")
class TestPbrtPhaseTable(unittest.TestCase):
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

    def test_single_scatter_angles_and_furnace(self):
        th, vals = fog_mie.pbrt_table("radiation")
        table = '"float phasetheta" [ {} ] "float phasevalues" [ {} ]'.format(
            " ".join(f"{v:.9g}" for v in th), " ".join(f"{v:.7g}" for v in vals)
        )
        e, v = fog_mie.phase_table("radiation")
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            thin = '"rgb sigma_s" [0.01 0.01 0.01]'
            for deg in (30.0, 90.0, 170.0):
                t = math.radians(deg)
                # Light travels from "from" to the origin; the camera looks down -z: scattering angle = deg.
                light = (
                    f'LightSource "distant" "point3 from" [{math.sin(t):.6f} 0 {-math.cos(t):.6f}]'
                    ' "point3 to" [0 0 0] "rgb L" [1 1 1]'
                )
                iso = self._render(tmp, f"iso{deg:.0f}", light, thin + ' "float g" [0]', 1, 64)
                tab = self._render(tmp, f"tab{deg:.0f}", light, f"{thin} {table}", 1, 64)
                expect = 4 * math.pi * float(v[np.searchsorted(e, math.cos(t)) - 1])
                self.assertAlmostEqual(tab / iso, expect, delta=0.01 * expect)
            furnace = self._render(
                tmp, "furnace", 'LightSource "infinite" "rgb L" [1 1 1]', f'"rgb sigma_s" [3 3 3] {table}', 200, 32
            )
            self.assertAlmostEqual(furnace, 1.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
