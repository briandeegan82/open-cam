"""Highway AEB vulnerable-road-user targets (tools/highway_vru.py): geometry, spectra, scene hooks."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import highway_vru as vru
import numpy as np
from test_highway_scene import build_without_assets


class TestTargets(unittest.TestCase):
    def test_pedestrian_height_and_feet_on_ground(self) -> None:
        for kind, child in (("pedestrian", False), ("child", True)):
            lo, hi = vru.pedestrian(vru.HEIGHT_M[kind], 0.25, child=child).bounds()
            self.assertAlmostEqual(hi[1], vru.HEIGHT_M[kind], delta=0.03 * vru.HEIGHT_M[kind], msg=kind)
            self.assertAlmostEqual(lo[1], 0.0, delta=0.02, msg=kind)
            self.assertLess(hi[0] - lo[0], 0.6 * vru.HEIGHT_M[kind])

    def test_cyclist_size(self) -> None:
        lo, hi = vru.cyclist(1.80, 0.0).bounds()
        self.assertAlmostEqual(lo[1], 0.0, delta=0.01)  # tyres touch the road
        self.assertGreater(hi[2] - lo[2], 1.70)
        self.assertLess(hi[2] - lo[2], 1.95)
        self.assertGreater(hi[1], 1.45)
        self.assertLess(hi[1], 1.85)

    def test_normals_are_unit(self) -> None:
        for mat, meshes in vru.pedestrian(1.8, 0.1).parts.items():
            for p, t, n in meshes:
                self.assertEqual(p.shape, n.shape, mat)
                np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-6)
                self.assertLess(int(t.max()), len(p))

    def test_spectra_physical_and_measured(self) -> None:
        wl = np.arange(360.0, 831.0, 5.0)
        r = vru.usgs_reflectance("09_nylon_blue", wl)
        self.assertTrue(np.all((r >= 0) & (r <= 1)))
        d = np.loadtxt(vru.USGS / "09_nylon_blue.csv", delimiter=",")
        self.assertAlmostEqual(float(r[wl == 550.0][0]), float(np.interp(550.0, d[:, 0], d[:, 1])), places=6)
        self.assertGreater(r[wl == 450.0][0], r[wl == 600.0][0])  # it is blue


class TestScene(unittest.TestCase):
    def test_off_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m = build_without_assets(Path(td))
            self.assertNotIn("vru", m)
            self.assertNotIn("vru:", (Path(td) / "scene" / "highway.pbrt").read_text())

    def test_obstructed_pedestrian(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m = build_without_assets(
                Path(td), "--vru", "child", "--vru-obstruction", "parked-cars", "--vru-distance-m", "25"
            )
            v = m["vru"]
            self.assertEqual(v["type"], "child")
            (x0, y0, z0), (x1, y1, z1) = v["bbox_world_m"]
            self.assertAlmostEqual(y1, 1.154, delta=0.04)
            self.assertLess(z0, 25.0)
            self.assertGreater(z1, 25.0)
            self.assertEqual(len(v["obstruction"]), 2)
            zs = sorted(c["z_m"] for c in v["obstruction"])
            self.assertLess(zs[0] + 2.3, z0)  # cars either side of the target, clear of it
            self.assertGreater(zs[1] - 2.3, z1)
            scene = (Path(td) / "scene" / "highway.pbrt").read_text()
            for mat in ("vru:skin", "vru:top", "vru:trousers"):
                self.assertIn(f'NamedMaterial "{mat}"', scene)
            for inst in ("vru_park0", "vru_park1"):
                self.assertTrue((Path(td) / "scene" / "cars" / f"{inst}.pbrt").is_file())
            self.assertTrue((Path(td) / "scene" / "spd" / "vru_skin.spd").is_file())

    def test_cyclist_along(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            v = build_without_assets(Path(td), "--vru", "cyclist", "--vru-motion", "along", "--vru-offset-m", "-1")[
                "vru"
            ]
            (x0, _, z0), (x1, _, z1) = v["bbox_world_m"]
            self.assertGreater(z1 - z0, 1.7)  # long axis along the road
            self.assertLess(x1 - x0, 0.8)

    def test_bad_args(self) -> None:
        a = argparse.Namespace(vru="pedestrian", vru_distance_m=1.0, vru_gait_phase=0.0)
        with self.assertRaises(SystemExit):
            vru.check_args(a)


if __name__ == "__main__":
    unittest.main()
