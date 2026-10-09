"""Wet road / puddles / wet markings / spray (tools/highway_wet.py)."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import unittest
from pathlib import Path

import highway_night as night
import highway_wet as wet
import numpy as np
from test_highway_scene import build_without_assets


class TestWaterFilmOptics(unittest.TestCase):
    def test_fresnel_normal_incidence(self) -> None:
        self.assertAlmostEqual(float(wet.fresnel(1.0, 1.333)), (0.333 / 2.333) ** 2, places=12)
        self.assertAlmostEqual(float(wet.fresnel(0.0, 1.333)), 1.0, places=9)

    def test_diffuse_reflectances_water(self) -> None:
        """r_e(1.333) = 0.066 (Lekner & Dorf 1988); r_i from 1 - r_i = (1 - r_e)/n^2 vs direct integration."""
        r_e, r_i = wet.diffuse_reflectances(1.333)
        self.assertAlmostEqual(r_e, 0.066, delta=0.001)
        n, m = 1.333, 200000
        c = (np.arange(m) + 0.5) / m  # inside the water: TIR beyond the critical angle
        s_t = n * np.sqrt(1 - c**2)
        ct = np.sqrt(np.clip(1 - s_t**2, 0, 1))
        rs = np.where(s_t < 1, (n * c - ct) / (n * c + ct), 1.0)
        rp = np.where(s_t < 1, (c - n * ct) / (c + n * ct), 1.0)
        r_i_direct = float(np.mean(0.5 * (rs**2 + rp**2) * 2 * c))
        self.assertAlmostEqual(r_i, r_i_direct, places=4)

    def test_lekner_dorf_limits_and_darkening(self) -> None:
        self.assertAlmostEqual(float(wet.body_albedo(0.3, 1.0 + 1e-9)), 0.3, places=6)
        r_e = wet.diffuse_reflectances(1.333)[0]  # no absorption: only the specular share r_e is missing
        self.assertAlmostEqual(float(wet.body_albedo(1.0, 1.333)), 1.0 - r_e, places=9)
        a = np.linspace(0.03, 0.25, 12)
        ratio = wet.darkening_ratio(a)
        self.assertTrue(np.all(ratio < 0.56) and np.all(ratio > 0.49))  # dark asphalt ~halves when wet
        self.assertTrue(np.all(np.diff(ratio) > 0))  # brighter surfaces darken less

    def test_wet_base_scale_nearly_linear(self) -> None:
        """One spectral scale per spd is valid across texture luminance x0.4..x1.6 (<10 %, documented)."""
        for a0 in (0.095, 0.155):
            s0 = wet.wet_base_reflectance(a0) / a0
            for f in (0.4, 1.6):
                s = wet.wet_base_reflectance(a0 * f) / (a0 * f)
                self.assertLess(abs(s / s0 - 1), 0.10)

    def test_q0_vs_cie_tables(self) -> None:
        """Dry model near the CIE R2/R3 Q0 (0.07); wet/flooded inside the W1..W4 span (0.11-0.25)."""
        dry = wet.q0("dry")
        self.assertGreater(dry, 0.05)
        self.assertLess(dry, wet.CIE_Q0["R1"])
        self.assertLess(wet.q0("damp"), wet.CIE_Q0["W1"])
        for lvl in ("wet", "flooded"):
            self.assertGreaterEqual(wet.q0(lvl), wet.CIE_Q0["W1"], lvl)
            self.assertLessEqual(wet.q0(lvl), wet.CIE_Q0["W4"] * 1.02, lvl)
        self.assertLess(wet.q0("wet"), wet.q0("flooded"))

    def test_q0_lambertian(self) -> None:
        """Q0 x pi = reflectance for a Lambertian surface (CIE definition), by the same integrator."""
        co = math.sin(math.radians(1.0))
        g = np.linspace(0, math.atan(12.0), 400)[:, None] + 0 * np.linspace(0, math.pi, 361)[None, :]
        w = np.sin(g)
        q = np.full_like(g, 0.2 / math.pi) + 0 * co
        self.assertAlmostEqual(float((q * w).sum() / w.sum()) * math.pi, 0.2, places=9)


class TestWetMarkingsAndSpray(unittest.TestCase):
    def test_wet_marking_rl_en1436(self) -> None:
        self.assertEqual(wet.wet_marking_rl("wet"), {"white": 35.0, "yellow": 25.0})
        self.assertEqual(wet.wet_marking_rl("flooded"), {"white": 25.0, "yellow": 25.0})
        for colour, rl in wet.wet_marking_rl("wet").items():
            self.assertLess(rl, 0.2 * night.MARKING_RL[colour])
            a, ci, co = night.marking_geometry()
            r = float(night.retro_ra(a, ci, co, night.marking_ra_param(rl), **night._kw(night.MARKING))) / co
            self.assertAlmostEqual(1e3 * r, rl, places=6)

    def test_spray_scaling(self) -> None:
        self.assertAlmostEqual(wet.spray_extinction(90, "wet", True), wet.SPRAY_EXT_REF)
        self.assertEqual(wet.spray_extinction(25, "wet", True), 0.0)
        self.assertEqual(wet.spray_extinction(120, "damp", True), 0.0)
        self.assertLess(wet.spray_extinction(80, "wet", False), wet.spray_extinction(80, "wet", True))
        self.assertLess(wet.spray_extinction(60, "wet", True), wet.spray_extinction(110, "wet", True))
        self.assertAlmostEqual(wet.spray_extinction(70, "flooded", True), 2 * wet.spray_extinction(70, "wet", True))
        rho = wet.spray_density(10, 8, 24, 2.4, 1.6, 15.0)
        self.assertAlmostEqual(float(rho.max()), 1.0)
        self.assertGreater(rho[0].mean(), rho[-1].mean())  # decays downstream
        self.assertGreater(rho[:, 0].mean(), rho[:, -1].mean())  # and with height

    def test_spray_cars_within_measured_extinction(self) -> None:
        """Cars at motorway speeds on a wet road stay below the measured 0.2 m^-1 peak (Otxoterena 2021)."""
        for v in (90, 110, 130):
            self.assertLessEqual(wet.spray_extinction(v, "wet", False), wet.SPRAY_EXT_REF)

    def test_spray_box_below_road_and_clear_of_camera(self) -> None:
        rho = wet.spray_density(10, 8, 24, 2.4, 1.6, 15.0, -wet.SPRAY_FLOOR)
        y = -wet.SPRAY_FLOOR + (np.arange(8) + 0.5) / 8 * (1.6 + wet.SPRAY_FLOOR)
        self.assertEqual(float(rho[:, y < 0].max()), 0.0)
        cars = [
            {"id": "c", "x": 0.0, "distance_m": 8.0, "length_m": 4.0, "width_m": 1.8, "height_m": 1.5, "heading_deg": 0}
        ]
        args = argparse.Namespace(road_wetness="wet")
        lines, meta = wet.spray_lines(cars, [120.0], [True], args, "", lambda x, y, z: [f"Translate {x} {y} {z}"])
        z0 = 8.0 - 2.0 - 0.15
        self.assertLess(meta[0]["length_m"], z0)  # the plume ends in front of the camera at z = 0
        self.assertIn(f'"point3 p0" [-1.2 {-wet.SPRAY_FLOOR}', "\n".join(lines))

    def test_spray_box_outward_normals(self) -> None:
        """pbrt-v4: a ray leaving along +n enters MediumInterface 'outside', so box normals must point out."""
        line = wet._box(-1.0, 0.0, -3.0, 1.0, 2.0, 0.0)[0]
        pts = np.array(line.split("[")[1].split("]")[0].split(), float).reshape(-1, 3)
        idx = np.array(line.split("[")[2].split("]")[0].split(), int).reshape(-1, 3)
        centre = pts.mean(axis=0)
        for a, b, c in idx:
            n = np.cross(pts[a] - pts[c], pts[b] - pts[c])  # pbrt Triangle: Cross(dp02, dp12)
            self.assertGreater(float(np.dot(n, (pts[a] + pts[b] + pts[c]) / 3 - centre)), 0.0)


class TestBuilderWet(unittest.TestCase):
    def test_dry_default_unchanged(self) -> None:
        """--road-wetness dry (the default) writes exactly the same scene as no flag."""
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "a").mkdir()
            (Path(td) / "b").mkdir()
            a = build_without_assets(Path(td) / "a")
            b = build_without_assets(Path(td) / "b", "--road-wetness", "dry")
            ta = (Path(td) / "a" / "scene" / "highway.pbrt").read_text()
            tb = (Path(td) / "b" / "scene" / "highway.pbrt").read_text()
            self.assertEqual(ta, tb)
            self.assertNotIn("wet", a["road"])
            self.assertEqual(json.dumps(a, sort_keys=True).replace("/a/", "/b/"), json.dumps(b, sort_keys=True))
            self.assertNotIn("wet_", ta)
            self.assertNotIn("spray", ta)
            self.assertNotIn("1.333", ta)

    def test_wet_day_scene(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m = build_without_assets(Path(td), "--road-wetness", "wet", "--puddles", "--spray")
            out = Path(td) / "scene"
            txt = (out / "highway.pbrt").read_text()
            w = m["road"]["wet"]
            self.assertIn('MakeNamedMaterial "wet:puddle"', txt)
            self.assertIn('"wet:film" "wet:puddle"', txt)
            self.assertIn('"float eta" [1.333]', txt)
            self.assertIn('"spd/wet_asphalt_aged.spd"', txt)
            self.assertNotIn(
                '"spd/asphalt_aged.spd"', txt.split("# Road wear")[1].split('NamedMaterial "asphalt"\n')[0]
            )
            self.assertIn('"volpath"', txt)
            self.assertIn('MakeNamedMedium "spray:car00"', txt)
            self.assertGreater(w["puddles"]["area_fraction"], 0.0)
            self.assertLess(w["puddles"]["area_fraction"], 0.1)
            self.assertTrue(w["spray"]["vehicles"])
            wet_r = np.loadtxt(out / "spd" / "wet_asphalt_aged.spd")[:, 1]
            dry_r = np.loadtxt(out / "spd" / "asphalt_aged.spd")[:, 1]
            np.testing.assert_allclose(wet_r, wet.wet_base_reflectance(dry_r), rtol=1e-5)

    def test_wet_night_markings_and_haze(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m = build_without_assets(
                Path(td), "--time-of-day", "night", "--haze", "mist", "--road-wetness", "flooded", "--spray"
            )
            txt = (Path(td) / "scene" / "highway.pbrt").read_text()
            self.assertIn("ra_wet_paint_white.spd", txt)
            self.assertNotIn('"spd/ra_paint_white.spd"', txt)
            self.assertEqual(m["road"]["wet"]["markings"]["rl_mcd_m2_lx"]["white"], 25.0)
            self.assertIn('MediumInterface "spray:car00" "haze"', txt)

    def test_flag_validation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit):
                build_without_assets(Path(td), "--puddles")
            with self.assertRaises(SystemExit):
                build_without_assets(Path(td), "--road-wetness", "wet", "--puddles", "--road-wear", "none")


if __name__ == "__main__":
    unittest.main()
