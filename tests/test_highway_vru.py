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
            self.assertTrue(v["removed_traffic"])
            self.assertTrue(all(abs(c["distance_m"] - 25.0) < vru.CLEAR_M for c in v["removed_traffic"]))
            self.assertFalse([c for c in m["cars"] if c["lane"] >= 0 and abs(c["distance_m"] - 25.0) < vru.CLEAR_M])
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


class TestImportedRider(unittest.TestCase):
    def test_contact_points_cover_every_rideable_model(self) -> None:
        import two_wheelers as tw

        self.assertEqual(
            set(tw.RIDER), set(tw.MODELS) - {"kickscooter_electric", "moto_yamaha_fz150", "moped_solifer_export"}
        )
        for aid, ((sz, sy), (gz, gy, gw), (_fz, fy, fw)) in tw.RIDER.items():
            half = tw.MODELS[aid][2] / 2
            self.assertTrue(-half < sz < gz < half, aid)  # saddle behind the grips, both on the model
            self.assertTrue(0.15 < fy < sy < gy + 0.25 < 1.5, aid)
            self.assertTrue(0.1 < gw < 0.5 and 0.05 < fw < 0.25, aid)

    def test_rider_sits_on_the_model(self) -> None:
        import two_wheelers as tw

        for aid in ("bike_road", "moto_80s", "scooter_vespa_yellow"):
            (sz, sy), _, _ = tw.RIDER[aid]
            lo, hi = vru.imported_rider(aid, 0.25).bounds()
            self.assertGreater(lo[1], 0.0, aid)  # feet above the road
            self.assertTrue(1.5 < hi[1] < 2.0, aid)  # seated 1.80 m rider
            self.assertTrue(lo[2] < sz < hi[2], aid)
            names = set(vru.imported_rider(aid, 0.25).parts)
            self.assertNotIn("vru:tyre", names)  # no procedural bicycle under an imported one
            self.assertEqual(
                "vru:hair" in names, tw.MODELS[aid][0] == "bicycle", aid
            )  # cycle helmet vs full-face helmet

    def test_bike_args(self) -> None:
        def ns(**kw):
            return argparse.Namespace(**{"vru_distance_m": 30.0, "vru_gait_phase": 0.25, "vru_bike": None, **kw})

        a = ns(vru="motorcyclist")
        vru.check_args(a)
        self.assertEqual(a.vru_bike, vru.DEFAULT_MOTO)
        vru.check_args(ns(vru="cyclist", vru_bike="bike_bmx"))
        for bad in (
            ns(vru="cyclist", vru_bike="moto_80s"),
            ns(vru="motorcyclist", vru_bike="bike_bmx"),
            ns(vru="pedestrian", vru_bike="bike_bmx"),
            ns(vru="none", vru_bike="bike_bmx"),
        ):
            with self.assertRaises(SystemExit):
                vru.check_args(bad)

    def test_procedural_cyclist_unchanged_without_vru_bike(self) -> None:
        names = set(vru.cyclist(1.80, 0.25).parts)
        self.assertIn("vru:tyre", names)
        self.assertIn("vru:hair", names)


class TestBicycleReflectors(unittest.TestCase):
    def test_reflectors_only_when_asked(self) -> None:
        names = {f"vru:reflex_{k}" for k in vru.REFLECTORS}
        self.assertFalse(names & set(vru.cyclist(1.8, 0.0).parts))
        parts = vru.cyclist(1.8, 0.0, reflectors=True).parts
        self.assertTrue(names <= set(parts))
        self.assertEqual(len(parts["vru:reflex_pedal"]), 2)
        self.assertEqual(len(parts["vru:reflex_spoke"]), 2)
        for k in names:
            for _, _, n in parts[k]:
                self.assertTrue(np.allclose(np.linalg.norm(n, axis=1), 1.0))

    def test_reflector_materials(self) -> None:
        import highway_night as night

        with tempfile.TemporaryDirectory() as td:
            spd = Path(td) / "spd"
            spd.mkdir()
            lines = vru.reflector_material_lines(spd, Path(td), night.LIGHT_WL)
            self.assertTrue((spd / "vru_reflex_red.spd").exists())
        made = [ln.split('"')[1] for ln in lines if ln.startswith("MakeNamedMaterial")]
        self.assertEqual(made, [f"vru:reflex_{k}" for k in vru.REFLECTORS])
        self.assertTrue(all('"retroreflective"' in ln for ln in lines if ln.startswith("MakeNamedMaterial")))


if __name__ == "__main__":
    unittest.main()


class TestMotion(unittest.TestCase):
    def test_static_without_exposure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            v = build_without_assets(Path(td), "--vru", "cyclist")["vru"]
            self.assertTrue(v["static"])
            self.assertEqual(v["displacement_m"], 0.0)
            self.assertIsNone(v["wheel_rotation_deg"])

    def test_crossing_cyclist_moves_and_wheels_roll(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            v = build_without_assets(Path(td), "--vru", "cyclist", "--exposure-s", "0.01")["vru"]
            txt = (Path(td) / "scene" / "highway.pbrt").read_text()
        ms = 15.0 / 3.6
        self.assertFalse(v["static"])
        self.assertAlmostEqual(v["displacement_m"], ms * 0.01, places=6)
        np.testing.assert_allclose(v["velocity_world_mps"], [-ms, 0.0, 0.0], atol=1e-6)  # right-to-left
        deg = np.degrees(ms * 0.01 / (vru.WHEEL_D / 2))
        self.assertAlmostEqual(v["wheel_rotation_deg"], deg, places=3)
        self.assertIn(f"Translate 0 0 {ms * 0.01:.6g}", txt)
        self.assertEqual(txt.count(f"Rotate {deg:.6g} 1 0 0"), 2)

    def test_speed_zero_and_wheel_limit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            v = build_without_assets(Path(td), "--vru", "pedestrian", "--exposure-s", "0.01", "--vru-speed-kmh", "0")[
                "vru"
            ]
            self.assertTrue(v["static"])
        with tempfile.TemporaryDirectory() as td, self.assertRaises(SystemExit):
            build_without_assets(Path(td), "--vru", "cyclist", "--exposure-s", "0.25")

    def test_separate_wheels_keep_geometry(self) -> None:
        wheels: list = []
        joint = vru.cyclist(1.80, 0.25)
        body = vru.cyclist(1.80, 0.25, wheels)
        self.assertEqual(len(wheels), 2)
        lo, hi = body.bounds()
        for c, w in wheels:
            wlo, whi = w.bounds()
            lo, hi = np.minimum(lo, wlo + c), np.maximum(hi, whi + c)
        jlo, jhi = joint.bounds()
        np.testing.assert_allclose(lo, jlo, atol=1e-9)
        np.testing.assert_allclose(hi, jhi, atol=1e-9)
