"""Rear lamp clusters (tools/highway_taillamps.py): ray-cast conformity, photometry, spectra, builder hook."""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

import highway_night as night
import highway_taillamps as tl
import numpy as np
from colour_science import tristimulus
from fetch_highway_assets import write_ply
from test_highway_scene import build_without_assets


def _patch_area(shape_line: str) -> float:
    """Projected (x-y) area of a bilinearmesh line written by ``_patches``."""
    p = np.array([float(v) for v in re.search(r'"point3 P" \[([^\]]*)\]', shape_line).group(1).split()])
    q = p.reshape(-1, 4, 3)
    return float(np.sum(np.abs(q[:, 0, 0] - q[:, 1, 0]) * np.abs(q[:, 2, 1] - q[:, 0, 1])))


def _emitted_cd(lines: list[str]) -> float:
    total, lum = 0.0, None
    for ln in lines:
        m = re.search(r'AreaLightSource "diffuse" .*"float scale" \[([^\]]*)\]', ln)
        if m:
            lum = float(m.group(1))
        elif "bilinearmesh" in ln and lum is not None:
            total += lum * _patch_area(ln)
            lum = None
    return total


CAR = {"id": "c", "model": "proxy", "length_m": 4.6, "width_m": 1.8, "height_m": 1.5}


class TestRearLamps(unittest.TestCase):
    def test_ply_mesh_and_cast(self) -> None:
        tri = tl._box_tris(-0.9, 0.9, 0.3, 1.05, -2.3, 2.3)
        pos = tri.reshape(-1, 3)
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "box.ply"
            write_ply(f, pos.astype(np.float32), None, None, np.arange(len(pos)).reshape(-1, 3))
            p, t = tl.read_ply_mesh(f)
        np.testing.assert_allclose(p[t], tri, atol=1e-6)
        z, m = tl.RearSurface(p[t], np.zeros(len(t), int), ["paint"]).cast([0.0, 0.5, 2.0], [0.6, 0.9, 0.6])
        np.testing.assert_allclose(z[:2], -2.3, atol=1e-6)
        self.assertTrue(np.isnan(z[2]))
        self.assertEqual(m[2], -1)

    def test_proxy_lamps_sit_on_the_body_and_meet_the_intensities(self) -> None:
        for braking in (False, True):
            lines, meta = tl.rear_lamp_lines(CAR, braking)
            want = tl.TAIL_CD + (tl.STOP_CD if braking else 0.0)
            for side in meta["sides"]:
                self.assertEqual(side["surface"], "mesh")
                self.assertAlmostEqual(side["intensity_cd"], want, places=2)
            chmsl = tl.CHMSL_CD if braking else 0.0
            self.assertAlmostEqual(_emitted_cd(lines), 2 * want + chmsl, delta=1e-3 * want)
            zs = [
                float(v)
                for ln in lines
                if "bilinearmesh" in ln
                for v in re.search(r"\[([^\]]*)\]", ln).group(1).split()[2::3]
            ]
            # lens 2 mm and emitters 4 mm behind the proxy's rear faces (body z = -2.3, glass z = -1.4)
            self.assertTrue(all(abs(z + 2.302) < 1e-6 or abs(z + 2.304) < 1e-6 or abs(z + 1.402) < 1e-6
                                or abs(z + 1.404) < 1e-6 for z in zs))  # fmt: skip

    def test_led_tail_is_a_contour_and_stop_fills_it(self) -> None:
        _, off = tl.rear_lamp_lines(CAR, False)
        _, on = tl.rear_lamp_lines(CAR, True)
        s_off, s_on = off["sides"][0], on["sides"][0]
        self.assertLess(s_off["emitting_area_m2"], 0.7 * s_off["lens_area_m2"])
        self.assertAlmostEqual(s_on["emitting_area_m2"], s_on["lens_area_m2"], places=4)
        self.assertGreater(s_on["peak_luminance_cd_m2"], 5 * s_off["peak_luminance_cd_m2"])

    def test_heavy_vehicle_chambers_flat_rear(self) -> None:
        truck = {"id": "t", "model": "truck_box", "length_m": 12.5, "width_m": 2.5, "height_m": 3.6}
        lines, meta = tl.rear_lamp_lines(truck, True)
        self.assertEqual({s["surface"] for s in meta["sides"]}, {"flat"})
        self.assertIsNone(meta["chmsl"])
        text = "\n".join(lines)
        for kind in ("red", "amber", "clear"):
            self.assertIn(tl.LENS_SPD[kind], text)
        self.assertAlmostEqual(_emitted_cd(lines), 2 * (tl.TAIL_CD + tl.STOP_CD), places=2)
        # red chamber is the middle third, so it emits less area than the whole lens
        self.assertLess(meta["sides"][0]["emitting_area_m2"], 0.5 * meta["sides"][0]["lens_area_m2"])

    def test_rear_fog_lamp_left_only(self) -> None:
        lines, meta = tl.rear_lamp_lines(CAR, False, rear_fog=True)
        fog = meta["rear_fog"]
        self.assertEqual(fog["side"], "left")
        self.assertAlmostEqual(fog["intensity_cd"], tl.FOG_CD, places=2)
        self.assertAlmostEqual(_emitted_cd(lines), 2 * tl.TAIL_CD + tl.FOG_CD, delta=1e-3 * tl.FOG_CD)
        fog_lines = lines[len(tl.rear_lamp_lines(CAR, False)[0]) :]
        xs = [
            float(v)
            for ln in fog_lines
            if "bilinearmesh" in ln
            for v in re.search(r"\[([^\]]*)\]", ln).group(1).split()[0::3]
        ]
        self.assertTrue(xs and max(xs) < 0)  # car's left (-x) side only
        self.assertIsNone(tl.rear_lamp_lines(CAR, False)[1]["rear_fog"])
        old = {"id": "p", "model": "car_pontiac_gto", "length_m": 5.17, "width_m": 1.87, "height_m": 1.44}
        self.assertIsNone(tl.rear_lamp_lines(old, False, rear_fog=True)[1]["rear_fog"])
        truck = {"id": "t", "model": "truck_box", "length_m": 12.5, "width_m": 2.5, "height_m": 3.6}
        self.assertAlmostEqual(
            tl.rear_lamp_lines(truck, False, rear_fog=True)[1]["rear_fog"]["intensity_cd"], tl.FOG_CD, places=2
        )

    def test_red_spectra_inside_ece_red_box(self) -> None:
        wl = night.LIGHT_WL
        for spd in (tl.red_incandescent_spectrum(wl), night.red_led_spectrum(wl)):
            x, y, z = tristimulus(wl, spd)
            s = x + y + z
            # ECE R48 red: yellow boundary y <= 0.335. (The purple boundary z <= 0.008 is not tested: the
            # colour_science z-bar table has a 0.001 floor in the red, which adds ~0.002 to z here.)
            self.assertLessEqual(y / s, 0.335)
            self.assertGreater(x / s, 0.66)


class TestBuilderHook(unittest.TestCase):
    def test_detailed_by_default_simple_on_request(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m = build_without_assets(Path(td), "--time-of-day", "night")
            scene = (Path(td) / "scene" / "highway.pbrt").read_text()
            lamps = m["lighting"]["artificial"]["tail_lamps"]
            self.assertTrue(all("sides" in t for t in lamps))
            self.assertIn("highway_taillamps.py", scene)
            self.assertTrue((Path(td) / "scene" / "spd" / "taillamp_lens_red.spd").is_file())
            braking = [t for t in lamps if t["braking"]]
            self.assertTrue(braking and braking[0]["chmsl"]["intensity_cd"] == tl.CHMSL_CD)
        with tempfile.TemporaryDirectory() as td:
            m = build_without_assets(Path(td), "--time-of-day", "night", "--tail-lamps", "simple")
            scene = (Path(td) / "scene" / "highway.pbrt").read_text()
            self.assertNotIn("highway_taillamps.py", scene)
            self.assertNotIn("sides", m["lighting"]["artificial"]["tail_lamps"][0])

    def test_rear_fog_auto_with_fog(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m = build_without_assets(Path(td), "--time-of-day", "night")
            self.assertTrue(all(t["rear_fog"] is None for t in m["lighting"]["artificial"]["tail_lamps"]))
        with tempfile.TemporaryDirectory() as td:
            m = build_without_assets(Path(td), "--time-of-day", "night", "--haze", "fog")
            lamps = m["lighting"]["artificial"]["tail_lamps"]
            self.assertTrue(all(t["rear_fog"]["intensity_cd"] == tl.FOG_CD for t in lamps))


class TestReflexReflectors(unittest.TestCase):
    def test_every_layout_but_the_vintage_car_has_reflex_reflectors(self) -> None:
        self.assertEqual({m for m, lay in tl.LAYOUTS.items() if lay.reflex is None}, {"car_vintage"})

    def test_reflex_patches_only_with_a_material(self) -> None:
        truck = {"id": "t", "model": "truck_box", "length_m": 12.5, "width_m": 2.5, "height_m": 3.6}
        for car in (CAR, truck):
            plain, meta0 = tl.rear_lamp_lines(car, False)
            self.assertNotIn("reflex:", "\n".join(plain))
            self.assertNotIn("reflex_reflectors", meta0)
            made: list[tuple[str, float]] = []

            def record(name: str, area: float, made=made) -> list[str]:
                made.append((name, area))
                return []

            lines, meta = tl.rear_lamp_lines(car, False, reflex_material=record)
            text = "\n".join(lines)
            self.assertEqual([n for n, _ in made], [f"reflex:{car['id']}:L", f"reflex:{car['id']}:R"])
            self.assertEqual(text.count('NamedMaterial "reflex:'), 2)
            for (_, a), side in zip(made, meta["reflex_reflectors"], strict=True):
                self.assertAlmostEqual(a, side["area_m2"], places=4)
                self.assertGreater(a, 0.003)
                self.assertLess(a, 0.08)


if __name__ == "__main__":
    unittest.main()
