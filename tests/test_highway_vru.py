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

        self.assertEqual(set(tw.RIDER), set(tw.MODELS))
        for aid, (seat, (gz, gy, gw), (fz, fy, fw)) in tw.RIDER.items():
            half = tw.MODELS[aid][2] / 2
            if seat is None:  # kick e-scooter: standing on the deck behind the bar
                self.assertEqual(tw.MODELS[aid][0], "kickscooter")
                self.assertTrue(-half < fz < gz < half and 0.05 < fy < 0.3 < 0.9 < gy, aid)
                continue
            sz, sy = seat
            self.assertTrue(-half < sz < gz < half, aid)  # saddle behind the grips, both on the model
            self.assertTrue(0.15 < fy < sy < gy + 0.25 < 1.5, aid)
            self.assertTrue(0.1 < gw < 0.5 and 0.05 < fw < 0.25, aid)

    def test_rider_sits_on_the_model(self) -> None:
        import two_wheelers as tw

        for aid in ("bike_road", "moto_80s", "scooter_vespa_yellow", "moto_yamaha_fz150", "moped_solifer_export"):
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
        a = ns(vru="escooter")
        vru.check_args(a)
        self.assertEqual(a.vru_bike, vru.DEFAULT_KICK)
        for bad in (
            ns(vru="escooter", vru_bike="moto_80s"),
            ns(vru="motorcyclist", vru_bike="kickscooter_electric"),
            ns(vru="pedestrian", vru_turn_radius_m=10.0),
            ns(vru="cyclist", vru_turn_radius_m=1.0),
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
        for c, w, _ in wheels:
            wlo, whi = w.bounds()
            lo, hi = np.minimum(lo, wlo + c), np.maximum(hi, whi + c)
        jlo, jhi = joint.bounds()
        np.testing.assert_allclose(lo, jlo, atol=1e-9)
        np.testing.assert_allclose(hi, jhi, atol=1e-9)


class TestTurn(unittest.TestCase):
    def _ns(self, **kw):
        return argparse.Namespace(**{"vru": "cyclist", "vru_speed_kmh": None, **kw})

    def test_steady_turn_lean_and_steer(self) -> None:
        t = vru.turn(self._ns(vru_turn_radius_m=10.0), vru.WHEELBASE)
        v = 15.0 / 3.6
        self.assertAlmostEqual(t["lean_deg"], np.degrees(np.arctan(v * v / (9.81 * 10.0))), places=9)  # ~10 deg
        self.assertAlmostEqual(t["steer_deg"], np.degrees(np.arctan(vru.WHEELBASE / 10.0)), places=9)
        right = vru.turn(self._ns(vru_turn_radius_m=-10.0), vru.WHEELBASE)
        self.assertAlmostEqual(right["lean_deg"], -t["lean_deg"])
        o = vru.turn(self._ns(vru_turn_radius_m=10.0, vru_lean_deg=3.0, vru_steer_deg=-2.0), vru.WHEELBASE)
        self.assertEqual((o["lean_deg"], o["steer_deg"]), (3.0, -2.0))
        self.assertEqual(vru.turn(self._ns(), vru.WHEELBASE)["lean_deg"], 0.0)

    def test_procedural_steer_turns_front_end_and_hands(self) -> None:
        straight, steered = vru.cyclist(1.80, 0.25), vru.cyclist(1.80, 0.25, steer_deg=25.0)

        def tyre_width(parts):
            x = np.concatenate([p[:, 0] for p, _, _ in parts.parts["vru:tyre"]])
            return float(np.ptp(x))

        self.assertGreater(tyre_width(steered), tyre_width(straight) + 0.1)  # front wheel swings out sideways
        self.assertAlmostEqual(steered.bounds()[0][1], 0.0, delta=0.02)  # still on the road
        wheels = []
        vru.cyclist(1.80, 0.25, wheels, steer_deg=25.0)
        self.assertIsNone(wheels[0][2])  # rear wheel
        self.assertEqual(wheels[1][2][0], 25.0)

    def test_scene_lean_and_turn_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            v = build_without_assets(Path(td), "--vru", "cyclist", "--vru-turn-radius-m", "10")["vru"]
            txt = (Path(td) / "scene" / "highway.pbrt").read_text()
        self.assertAlmostEqual(v["turn"]["lean_deg"], 10.02, delta=0.02)
        self.assertTrue(v["turn"]["steer_applied"])
        self.assertIn(f"Rotate {-v['turn']['lean_deg']:.6g} 0 0 1", txt)
        with tempfile.TemporaryDirectory() as td:
            s = build_without_assets(Path(td), "--vru", "cyclist")["vru"]
        self.assertEqual(s["turn"]["lean_deg"], 0.0)
        # crossing target: local +x is world +z, so a lean widens the box along the road
        w = lambda m: m["bbox_world_m"][1][2] - m["bbox_world_m"][0][2]  # noqa: E731
        self.assertGreater(w(v), w(s) + 0.1)


class TestEScooter(unittest.TestCase):
    def test_standing_rider_on_the_deck(self) -> None:
        parts = vru.imported_rider("kickscooter_electric", 0.25)
        lo, hi = parts.bounds()
        self.assertTrue(0.05 < lo[1] < 0.2)  # shoes on the deck
        self.assertTrue(1.7 < hi[1] < 1.95)  # standing 1.80 m rider
        self.assertIn("vru:hair", set(parts.parts))  # no full-face helmet

    def test_scene_without_assets_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            v = build_without_assets(Path(td), "--vru", "escooter")["vru"]
        self.assertEqual(v["type"], "escooter")
        self.assertEqual(v["bike"]["asset"], "kickscooter_electric")
        self.assertTrue(v["bike"]["missing"])
