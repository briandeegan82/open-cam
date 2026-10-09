"""In-car camera effects (tools/highway_incar.py, tools/render_time_slices.py); no pbrt needed."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tests"))

import build_highway_scene as hw  # noqa: E402
import highway_incar as hi  # noqa: E402
import render_time_slices as rts  # noqa: E402
from test_highway_scene import build_without_assets  # noqa: E402

WL = np.arange(360.0, 1101.0, 5.0)
FX = [
    "--exposure-s",
    "0.004",
    "--rolling-shutter-line-time-us",
    "25",
    "--ego-speed-kmh",
    "100",
    "--traffic-speed-kmh",
    "lanes",
    "--windscreen",
    "--windscreen-dirt",
    "0.05",
    "--windscreen-rain",
    "0.1",
    "--vms",
]


class TestGlass(unittest.TestCase):
    def test_fresnel_normal_incidence(self) -> None:
        rs, rp = hi.fresnel_unpolarised(1.0)
        self.assertAlmostEqual(rs, ((1.52 - 1) / 2.52) ** 2, places=9)
        self.assertAlmostEqual(rs, rp, places=12)

    def test_green_laminate_spectrum(self) -> None:
        t = hi.windscreen_transmittance(WL)
        vis = WL <= 830
        tv = hi.luminous_transmittance(WL[vis], t[vis])
        self.assertGreater(tv, 0.70)  # ECE R43 / FMVSS 205 minimum
        self.assertLess(tv, 0.85)
        at = dict(zip(WL, t))
        self.assertLess(at[370.0], 0.02)  # PVB UV cut
        self.assertLess(at[1000.0], 0.3 * at[550.0])  # Fe2+ NIR band
        self.assertGreater(at[550.0], at[450.0])  # green tint
        self.assertGreater(at[550.0], at[700.0])

    def test_oblique_incidence_transmits_less(self) -> None:
        t0 = hi.windscreen_transmittance(WL, 0.0)
        t63 = hi.windscreen_transmittance(WL, 63.0)
        self.assertTrue(np.all(t63 < t0))

    def test_measured_csv_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.csv"
            np.savetxt(p, np.c_[WL, hi.windscreen_transmittance(WL)], delimiter=",")
            alpha = hi.alpha_from_transmittance_csv(p, WL, sum(hi.LAMINATE_MM) * 1e-3)
            t2 = hi.windscreen_transmittance(WL, 0.0, alpha_per_m=alpha)
            np.testing.assert_allclose(t2, hi.windscreen_transmittance(WL), atol=0.01)


class TestWindscreenGeometry(unittest.TestCase):
    def test_flat_and_curved_normals(self) -> None:
        for ws in (hi.Windscreen(), hi.Windscreen(radius_h_m=2.5, radius_v_m=8.0)):
            _, _, _, n = ws.basis
            self.assertAlmostEqual(float(np.degrees(np.arccos(n[1]))), ws.rake_deg, places=6)
            p, nn = ws.surface(np.array([0.0]), np.array([0.0]))
            np.testing.assert_allclose(p[0], [0, 0, ws.axis_distance_m])
            np.testing.assert_allclose(nn[0], n, atol=1e-12)
            inner, outer = ws.shell()[:2]
            self.assertTrue(np.all(inner[2] @ n < 0))  # inner face looks back at the lens
            self.assertTrue(np.all(outer[2] @ n > 0))

    def test_glass_covers_the_view(self) -> None:
        for argv in ([], ["--camera", "realistic"], ["--camera", "thinlens", "--thinlens-lens-radius", "0.01"]):
            a = hw.parse_args([*argv, "--windscreen"])
            fx = hi.InCarEffects(a, REPO)
            ws = fx.windscreen
            u0, u1, s0, s1 = ws.footprint(fx.camera_dirs(), 0.0)
            self.assertLess(-ws.width_m / 2, u0)
            self.assertLess(u1, ws.width_m / 2)
            self.assertLess(-ws.below_m, s0)
            self.assertLess(s1, ws.above_m)

    def test_raindrops(self) -> None:
        ws = hi.Windscreen()
        region = (-0.05, 0.05, -0.05, 0.05)
        drops = hi.raindrops(ws, region, 0.15, np.random.default_rng(0))
        area = sum(np.pi * a * a for _, a in drops)
        self.assertAlmostEqual(area / 0.01, 0.15, delta=0.01)
        for i, (c, a) in enumerate(drops[:200]):
            for c2, a2 in drops[i + 1 : 200]:
                self.assertGreaterEqual(np.hypot(*(c - c2)), a + a2)
        p, tri, n = hi.drop_cap_mesh(ws, drops[:3], 45.0)
        self.assertEqual(len(p), len(n))
        self.assertTrue(np.all(n @ ws.basis[3] > -1e-9))  # caps bulge outwards


class TestPWM(unittest.TestCase):
    def test_on_fraction(self) -> None:
        p = hi.PWM(100.0, 0.25, 0.0)
        self.assertAlmostEqual(p.on_fraction(0.0, 0.1), 0.25)
        self.assertAlmostEqual(p.on_fraction(0.003, 0.009), 0.0)  # exposure inside the off time
        self.assertAlmostEqual(p.on_fraction(0.0, 0.0025), 1.0)
        self.assertAlmostEqual(p.on_fraction(0.001, 0.0110), (0.0015 + 0.001) / 0.010)
        self.assertEqual(p.edges(0.0, 0.021), [0.0025, 0.01, 0.0125, 0.02])
        self.assertAlmostEqual(hi.PWM(100.0, 0.25, 0.004).on_fraction(0.0, 0.002), 0.0)
        self.assertEqual(hi.PWM(0.0, 0.3).on_intervals(0.0, 1.0), [(0.0, 1.0)])

    def test_write_emitter_levels(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            inc, e = hi.write_emitter(
                out, "e0", lambda lv: [f"# {lv}"], pwm=hi.PWM(90.0, 0.2), exposure_window_s=(0.003, 0.004)
            )
            self.assertEqual(inc, 'Include "emitters/e0.pbrt"')
            self.assertEqual(e["default_level"], 0.0)
            self.assertEqual((out / e["states"]["on"]).read_text(), "# 1.0\n")
            _, e = hi.write_emitter(out, "e1", lambda lv: [""], pwm=hi.PWM(90.0, 0.2), exposure_window_s=None)
            self.assertAlmostEqual(e["default_level"], 0.2)

    def test_dot_matrix(self) -> None:
        m = hi.dot_matrix(["AB", "C"])
        self.assertEqual(m.shape, (16, 11))
        self.assertEqual(int(m[:7, :5].sum()), 18)  # "A"


class TestSlicePlan(unittest.TestCase):
    EXP = {"integration_time_s": 0.004, "shutter_open_s": 0.0, "rolling_shutter_line_time_s": 25e-6}
    EM = [
        {"id": "s", "include": "emitters/s.pbrt", "states": {"on": "a", "off": "b"}, "pwm": hi.PWM(100, 0.3).as_dict()}
    ]

    def test_weights_sum_to_one_per_band(self) -> None:
        sl = rts.plan_slices(self.EXP, self.EM, 720, 24, 64)
        self.assertEqual(len({(s.y0, s.y1) for s in sl}), 24)
        for y0 in {s.y0 for s in sl}:
            band = [s for s in sl if s.y0 == y0]
            self.assertAlmostEqual(sum(s.weight for s in band), 1.0, places=12)
            self.assertAlmostEqual(band[0].t0, (y0 + (band[0].y1 - y0 - 1) / 2) * 25e-6, places=12)
            on = sum(s.t1 - s.t0 for s in band if s.states["s"] == "on")
            self.assertAlmostEqual(on, hi.PWM(100, 0.3).on_fraction(band[0].t0, band[-1].t1) * 0.004, places=12)

    def test_global_shutter_is_one_band(self) -> None:
        sl = rts.plan_slices({**self.EXP, "rolling_shutter_line_time_s": 0.0}, [], 720, 24, 64)
        self.assertEqual(len(sl), 1)
        self.assertEqual((sl[0].y0, sl[0].y1, sl[0].spp), (0, 720, 64))

    def test_scene_text(self) -> None:
        scene = '    "float iso" [25000]\nCamera "perspective" "float shutteropen" [0] "float shutterclose" [0.004]\nInclude "emitters/s.pbrt"\n'
        sl = rts.Slice(0, 1, 0.001, 0.002, 0.25, {"s": "off"})
        txt = rts.slice_scene_text(scene, sl, self.EM)
        self.assertIn('"float shutteropen" [0.001] "float shutterclose" [0.002]', txt)
        self.assertIn('Include "b"', txt)
        self.assertIn('"float iso" [100000]', txt)


class TestBuilderHooks(unittest.TestCase):
    def test_default_scene_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m = build_without_assets(Path(td))
            scene = (Path(td) / "scene" / "highway.pbrt").read_text()
            for k in ("exposure", "emitters", "windscreen", "ego_motion"):
                self.assertNotIn(k, m)
            self.assertNotIn("speed_kmh", m["cars"][0])
            self.assertNotIn("ActiveTransform", scene)
            self.assertNotIn("shutter", scene)
            self.assertIn('Integrator "path"', scene)

    def test_all_effects(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            for sub in ("a", "b"):
                (Path(td) / sub).mkdir()
            base = build_without_assets(Path(td) / "a")
            m = build_without_assets(Path(td) / "b", *FX)
            sd = Path(td) / "b" / "scene"
            scene = (sd / "highway.pbrt").read_text()
            self.assertEqual(m["lighting"], base["lighting"])  # absolute illuminance untouched
            self.assertEqual(m["exposure"]["integration_time_s"], 0.004)
            self.assertAlmostEqual(m["exposure"]["frame_time_span_s"], 0.004 + 31 * 25e-6)
            self.assertIn('"float shutteropen" [0] "float shutterclose" [0.004]', scene)
            self.assertIn("TransformTimes 0 0.004775", scene)
            self.assertIn('"float iso" [25000]', scene)
            self.assertIn('Integrator "volpath"', scene)
            self.assertIn('MakeNamedMedium "windscreen_laminate"', scene)
            self.assertGreater(m["windscreen"]["rain"]["drops"], 10)
            self.assertTrue((sd / m["windscreen"]["dirt"]["mask"]).is_file())
            self.assertEqual([c["speed_kmh"] for c in m["cars"][:3]], [110.0, 95.0, 125.0])
            self.assertAlmostEqual(m["cars"][-1]["velocity_world_mps"][2], -105 / 3.6)  # oncoming
            (e,) = m["emitters"]
            for f in (e["include"], *e["states"].values()):
                self.assertTrue((sd / f).is_file())
            self.assertIn("AreaLightSource", (sd / e["states"]["on"]).read_text())
            self.assertNotIn("AreaLightSource", (sd / e["states"]["off"]).read_text())
            self.assertEqual(scene.count(f'Include "{e["include"]}"'), 1)
            json.dumps(m)

    def test_option_errors(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                build_without_assets(Path(td), "--rolling-shutter-line-time-us", "10")
            with self.assertRaises(ValueError):
                build_without_assets(Path(td), "--windscreen-rain", "0.1")
            with self.assertRaises(ValueError):
                build_without_assets(Path(td), "--exposure-s", "0.01", "--traffic-speed-kmh", "1,2")


if __name__ == "__main__":
    unittest.main()
