"""Road r-table loader/evaluator (tools/road_rtable.py) and the patched pbrt ``roadrtable`` material."""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import road_rtable as rt
from exr_multispectral import read_separate_exr_channels
from synthetic_data import REPO

PBRT = Path(os.environ.get("OPENCAM_PBRT", REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"))


def _small_table() -> rt.RTable:
    return rt.RTable(
        "t", [0.0, 1.0, 4.0], [0.0, 90.0, 180.0], [[0.01, 0.01, 0.01], [0.02, 0.01, 0.005], [0.004, 0.001, 0.0]]
    )


class TestRTable(unittest.TestCase):
    def test_placeholder_file_is_flagged_and_matches_generator(self) -> None:
        tb = rt.load(rt.PLACEHOLDER)
        self.assertTrue(tb.placeholder)
        self.assertIn("NOT a CIE table", tb.source)
        gen = rt.placeholder_table()
        np.testing.assert_allclose(tb.r, gen.r, rtol=5e-4, atol=1e-7)
        self.assertEqual(tb.r.shape, (29, 20))

    def test_q_at_knots_and_interpolation(self) -> None:
        tb = _small_table()
        for i, t in enumerate(tb.tan_gamma):
            for j, b in enumerate(tb.beta_deg):
                self.assertAlmostEqual(float(tb.q(t, b)), tb.r[i, j] * (1 + t * t) ** 1.5, places=12)
        # bilinear in r, then / cos^3
        self.assertAlmostEqual(float(tb.q(0.5, 45.0)), 0.5 * (0.01 + 0.015) * 1.25**1.5, places=12)
        # clamped beyond the table
        self.assertAlmostEqual(float(tb.q(10.0, 200.0)), float(tb.q(4.0, 180.0)))

    def test_q0_and_s1(self) -> None:
        flat = rt.RTable("flat", [0.0, 12.0], [0.0, 180.0], np.full((2, 2), 0.02))
        # constant r: q = r (1 + t^2)^1.5, Q0 = solid-angle mean
        g = np.linspace(0, math.atan(12), 200001)
        ref = np.trapezoid(0.02 / np.cos(g) ** 3 * np.sin(g), g) / np.trapezoid(np.sin(g), g)
        self.assertAlmostEqual(flat.q0() / ref, 1.0, places=3)
        tb = rt.load(rt.PLACEHOLDER)
        self.assertAlmostEqual(tb.s1(), tb.r[tb.tan_gamma.tolist().index(2.0), 0] / tb.r[0, 0], places=9)

    def test_brdf_blend_and_energy(self) -> None:
        tb = rt.load(rt.PLACEHOLDER)
        ci = np.cos(np.radians(60))
        self.assertAlmostEqual(
            float(tb.brdf(ci, math.sin(math.radians(1)), -1.0)), float(tb.q(math.tan(math.radians(60)), 0))
        )
        self.assertAlmostEqual(float(tb.brdf(ci, math.sin(math.radians(30)), -1.0)), tb.q0())
        self.assertEqual(float(tb.brdf(-0.1, 0.5, 1.0)), 0.0)
        for a in (1.0, 5.0, 45.0, 89.0):
            self.assertLess(tb.albedo(a), 1.0)
            self.assertGreater(tb.albedo(a), 0.0)

    def test_csv_round_trip_and_validation(self) -> None:
        tb = _small_table()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.csv"
            rt.write_csv(tb, p)
            back = rt.load(p)
            np.testing.assert_allclose(back.r, tb.r)
            np.testing.assert_allclose(back.beta_deg, tb.beta_deg)
            self.assertFalse(back.placeholder)
            p.write_text("# name: bad\n0,0,10\n0,1,2\n")
            with self.assertRaises(ValueError):
                rt.load(p)
        with self.assertRaises(ValueError):
            rt.RTable("bad", [0, 1], [0, 0], np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            rt.RTable("bad", [0, 1], [0, 10], -np.ones((2, 2)))

    def test_material_lines(self) -> None:
        tb = rt.load(rt.PLACEHOLDER)
        lines = rt.material_lines("asphalt", tb, '"spectrum tint" "spd/a.spd"')
        text = "\n".join(lines)
        self.assertIn("PLACEHOLDER, not CIE data", lines[0])
        self.assertIn('"string type" "roadrtable"', text)
        self.assertEqual(len(lines[-1].split("[")[1].split()), tb.r.size)

    def test_tint_spectrum_luminous_reflectance_is_one(self) -> None:
        from colour_science import cmf_on_grid

        wl = np.arange(380, 781, 5.0)
        tint = rt.tint_spectrum(0.05 + 0.1 * (wl - 380) / 400, wl)
        y = cmf_on_grid(wl)[1]
        self.assertAlmostEqual(float(np.sum(tint * y) / np.sum(y)), 1.0, places=12)


def _pbrt_has_rtable() -> bool:
    if not PBRT.is_file():
        return False
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.pbrt"
        p.write_text(
            'Film "rgb" "integer xresolution" [1] "integer yresolution" [1] "string filename" "t.exr"\n'
            'Camera "perspective"\nWorldBegin\n'
            'Material "roadrtable" "float tangamma" [0 1] "float beta" [0 180] "float r" [1 1 1 1]\n'
        )
        return subprocess.run([str(PBRT), "--quiet", str(p)], capture_output=True, cwd=td).returncode == 0


PBRT_RTABLE = _pbrt_has_rtable()
if os.environ.get("OPENCAM_REQUIRE_PBRT") == "1" and not PBRT_RTABLE:
    raise RuntimeError(f"OPENCAM_REQUIRE_PBRT=1 but {PBRT} lacks the roadrtable patch (tools/build_pbrt.sh)")


@unittest.skipUnless(PBRT_RTABLE, f"patched pbrt not built ({PBRT}); see docs/BUILD_PBRT.txt")
class TestPbrtRoadRTable(unittest.TestCase):
    """Rendered road / white-Lambertian ratio = pi q against the Python evaluator."""

    def _ratio(self, material: str, cam: np.ndarray, light: np.ndarray) -> float:
        out = []
        for mat in (material, 'Material "diffuse" "spectrum reflectance" [360 1 830 1]'):
            with tempfile.TemporaryDirectory() as td:
                s = [
                    f"LookAt {' '.join(map(str, cam))} 0 0 0 0 0 1",
                    'Camera "perspective" "float fov" [0.5]',
                    'Sampler "independent" "integer pixelsamples" [4]',
                    'Integrator "path" "integer maxdepth" [1]',
                    'Film "rgb" "integer xresolution" [16] "integer yresolution" [16] "string filename" "o.exr"',
                    "WorldBegin",
                    f'LightSource "point" "point3 from" [{" ".join(map(str, light))}] "spectrum I" [360 1 830 1]',
                    mat,
                    'Shape "bilinearmesh" "point3 P" [-50 -50 0 50 -50 0 -50 50 0 50 50 0]',
                ]
                (Path(td) / "s.pbrt").write_text("\n".join(s) + "\n")
                subprocess.run([str(PBRT), "--quiet", "s.pbrt"], check=True, cwd=td, capture_output=True)
                ch = read_separate_exr_channels(Path(td) / "o.exr")
                out.append(float(np.mean(ch["G"])))
        return out[0] / out[1]

    def test_table_lookup_matches_python(self) -> None:
        tb = rt.load(rt.PLACEHOLDER)
        mat = "\n".join(rt.material_lines("road", tb, "", 89.0, 90.0)[1:]) + '\nNamedMaterial "road"'
        elev = math.radians(20.0)
        cam = np.array([0.0, -30 * math.cos(elev), 30 * math.sin(elev)])
        for gamma, phi in ((40.0, 90.0), (65.0, 90.0), (65.0, 30.0), (30.0, -90.0)):
            g, f = math.radians(gamma), math.radians(phi)
            light = 5 * np.array([math.sin(g) * math.cos(f), math.sin(g) * math.sin(f), math.cos(g)])
            cos_phi = float(np.dot(light[:2], cam[:2]) / (np.linalg.norm(light[:2]) * np.linalg.norm(cam[:2])))
            q = float(tb.brdf(math.cos(g), math.sin(elev), cos_phi, 89.0, 90.0))
            with self.subTest(gamma=gamma, phi=phi):
                self.assertAlmostEqual(self._ratio(mat, cam, light) / (math.pi * q), 1.0, delta=0.02)

    def test_high_observer_blends_to_q0(self) -> None:
        tb = rt.load(rt.PLACEHOLDER)
        mat = "\n".join(rt.material_lines("road", tb, "")[1:]) + '\nNamedMaterial "road"'
        cam = np.array([0.0, -20.0, 20.0])  # 45 deg > observationmax
        light = np.array([3.0, 4.0, 2.0])
        self.assertAlmostEqual(self._ratio(mat, cam, light) / (math.pi * tb.q0()), 1.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
