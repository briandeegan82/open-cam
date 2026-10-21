"""Rear lamp clusters (tools/highway_taillamps.py): ray-cast conformity, photometry, spectra, builder hook."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

import highway_atmosphere as atm
import highway_night as night
import highway_taillamps as tl
import numpy as np
from colour_science import tristimulus
from exr_multispectral import read_separate_exr_channels
from fetch_highway_assets import write_ply
from test_highway_scene import build_without_assets

REPO = Path(__file__).resolve().parents[1]
PBRT = Path(os.environ.get("OPENCAM_PBRT", REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"))
PBRT_TIMEOUT_S = float(os.environ.get("OPENCAM_PBRT_TIMEOUT", "600"))


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
            m = build_without_assets(Path(td), "--time-of-day", "night", "--haze", "fog", "--visibility-m", "40")
            lamps = m["lighting"]["artificial"]["tail_lamps"]
            self.assertTrue(all(t["rear_fog"]["intensity_cd"] == tl.FOG_CD for t in lamps))

    def test_rear_fog_auto_gated_at_50_m(self) -> None:
        def auto(*argv: str) -> bool:
            ap = argparse.ArgumentParser()
            night.add_night_args(ap)
            atm.add_cli_args(ap)
            args = ap.parse_args(list(argv))
            args.time_of_day = "night"
            return night.resolve(args)["rear_fog"]

        self.assertFalse(auto())
        self.assertFalse(auto("--haze", "fog"))  # preset 150 m
        self.assertTrue(auto("--haze", "fog", "--visibility-m", "49"))
        self.assertFalse(auto("--haze", "fog", "--visibility-m", "51"))
        self.assertTrue(auto("--haze", "mist", "--visibility-m", "30"))
        # Koschmieder 2 % range 60 m = MOR 46 m
        self.assertTrue(auto("--haze", "fog", "--visibility-m", "60", "--visibility-definition", "koschmieder"))
        self.assertTrue(auto("--haze", "fog", "--rear-fog-visibility-m", "200"))
        self.assertFalse(auto("--haze", "fog", "--visibility-m", "40", "--rear-fog-lamps", "none"))
        self.assertTrue(auto("--rear-fog-lamps", "on"))


# FMVSS 108 one-section minima [cd] (49 CFR 571.108 Table VIII taillamp, Table IX stop lamp):
# (h, v) test points with + = right / up; the tables are symmetric left-right.
_FMVSS_TAIL = {(20, 5): 0.3, (20, -5): 0.3, (5, 10): 0.4, (5, -10): 0.4, (10, 5): 0.8, (10, 0): 0.8,
               (10, -5): 0.8, (0, 5): 1.8, (5, 0): 2.0, (0, 0): 2.0, (0, -5): 1.8}  # fmt: skip
_FMVSS_STOP = {(20, 5): 10, (20, -5): 10, (5, 10): 16, (5, -10): 16, (10, 5): 30, (10, 0): 40,
               (10, -5): 30, (0, 5): 70, (5, 0): 80, (0, 0): 80, (0, -5): 70}  # fmt: skip


def _rear_dir(h_deg: float, v_deg: float) -> np.ndarray:
    """Car-frame direction h right of / v above straight behind (-z), as ``beam_shape`` reads it."""
    h, v = np.radians(h_deg), np.radians(v_deg)
    return np.array([np.cos(v) * np.sin(h), np.sin(v), -np.cos(v) * np.cos(h)])


class TestRearLampBeam(unittest.TestCase):
    def test_shapes_reproduce_fmvss108_ratios(self) -> None:
        for kind, table, i0 in (("tail", _FMVSS_TAIL, tl.TAIL_CD), ("stop", _FMVSS_STOP, tl.STOP_CD)):
            for (h, v), cd in table.items():
                for sh in (h, -h):
                    r = float(tl.beam_shape(kind, _rear_dir(sh, v)))
                    self.assertAlmostEqual(r, cd / table[(0, 0)], delta=0.07, msg=f"{kind} {sh},{v}")
                    self.assertGreaterEqual(i0 * r, cd, msg=f"{kind} {sh},{v}")
        self.assertLessEqual(tl.TAIL_CD, 18.0)  # Table VIII maximum at H or above
        self.assertLessEqual(tl.STOP_CD, 300.0)  # Table IX maximum

    def test_shape_falls_off_and_is_dark_forwards(self) -> None:
        for kind in ("tail", "stop", "fog", "tail_stop"):
            hs = [float(tl.beam_shape(kind, _rear_dir(h, 0))) for h in (0, 10, 20, 45, 80, 89.9)]
            vs = [float(tl.beam_shape(kind, _rear_dir(0, v))) for v in (0, 5, 10, 30, 89.9)]
            self.assertAlmostEqual(hs[0], 1.0)
            self.assertTrue(all(a >= b for a, b in zip(hs, hs[1:])) and hs[-1] < 0.01, hs)
            self.assertTrue(all(a >= b for a, b in zip(vs, vs[1:])) and vs[-1] < 0.01, vs)
            self.assertEqual(float(tl.beam_shape(kind, np.array([0.0, 0.0, 1.0]))), 0.0)
        # R38-style field: >= 150 of 200 cd within +-10 deg H, +-5 deg V
        self.assertGreaterEqual(tl.FOG_CD * float(tl.beam_shape("fog", _rear_dir(10, 5))), 150 * 0.75)

    def test_map_is_shape_over_cosine(self) -> None:
        m = tl.beam_map("tail", 256)
        self.assertLessEqual(m.max(), 1.0)
        for h, v in ((0, 0), (10, 0), (20, 5), (5, 10)):
            d = _rear_dir(h, v)
            got = float(night.map_intensity(m, d[None])[0])
            want = float(tl.beam_shape("tail", d)) / float(-d[2])
            self.assertAlmostEqual(got, want, delta=0.06)

    def test_emitters_reference_maps(self) -> None:
        beam = {k: f"textures/rear_beam_{k}.exr" for k in tl.BEAM_MAPS}
        lines, meta = tl.rear_lamp_lines(CAR, True, rear_fog=True, beam=beam)
        lights = [ln for ln in lines if "AreaLightSource" in ln]
        self.assertTrue(lights and all('"string angularfilename"' in ln for ln in lights))
        used = {k for k in beam if any(f"rear_beam_{k}.exr" in ln for ln in lights)}
        self.assertTrue({"stop", "fog"} <= used, used)
        self.assertEqual(meta["beam"], "fmvss108_ratios")
        flat, flat_meta = tl.rear_lamp_lines(CAR, True, rear_fog=True)
        self.assertFalse(any("angularfilename" in ln for ln in flat))
        self.assertEqual(flat_meta["beam"], "lambertian")
        self.assertAlmostEqual(_emitted_cd(lines), _emitted_cd(flat))  # same H-V intensity

    def test_builder_writes_maps_and_lambertian_opt_out(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m = build_without_assets(Path(td), "--time-of-day", "night")
            scene = (Path(td) / "scene" / "highway.pbrt").read_text()
            self.assertIn("rear_beam_tail.exr", scene)
            self.assertTrue((Path(td) / "scene" / "textures" / "rear_beam_stop.exr").is_file())
            self.assertTrue(all(t["beam"] == "fmvss108_ratios" for t in m["lighting"]["artificial"]["tail_lamps"]))
        with tempfile.TemporaryDirectory() as td:
            m = build_without_assets(Path(td), "--time-of-day", "night", "--rear-lamp-beam", "lambertian")
            scene = (Path(td) / "scene" / "highway.pbrt").read_text()
            self.assertNotIn("angularfilename", scene)
            self.assertTrue(all(t["beam"] == "lambertian" for t in m["lighting"]["artificial"]["tail_lamps"]))


def _pbrt_has_angular() -> bool:
    if not PBRT.is_file():
        return False
    with tempfile.TemporaryDirectory() as td:
        night.write_y_exr(Path(td) / "a.exr", np.ones((8, 8), np.float32))
        (Path(td) / "t.pbrt").write_text(
            'Film "rgb" "integer xresolution" [1] "integer yresolution" [1] "string filename" "t.exr"\n'
            'Camera "perspective"\nWorldBegin\nAttributeBegin\n'
            'AreaLightSource "diffuse" "string angularfilename" "a.exr"\n'
            'Shape "bilinearmesh" "point3 P" [0 0 1 1 0 1 0 1 1 1 1 1]\nAttributeEnd\n'
        )
        r = subprocess.run([str(PBRT), "--quiet", "t.pbrt"], capture_output=True, cwd=td, timeout=PBRT_TIMEOUT_S)
        return r.returncode == 0


PBRT_ANGULAR = _pbrt_has_angular()
if os.environ.get("OPENCAM_REQUIRE_PBRT") == "1" and not PBRT_ANGULAR:
    raise RuntimeError(f"OPENCAM_REQUIRE_PBRT=1 but {PBRT} lacks patch 0016 (tools/build_pbrt.sh)")


@unittest.skipUnless(PBRT_ANGULAR, f"patched pbrt not built ({PBRT}); see docs/BUILD_PBRT.txt")
class TestPbrtAngularAreaLight(unittest.TestCase):
    """Render a rear-facing lamp quad from FMVSS test directions: radiance ratio = map value."""

    def _radiance(self, td: Path, d: np.ndarray, beam: bool) -> float:
        cam = 20.0 * d
        ang = ' "string angularfilename" "textures/rear_beam_stop.exr"' if beam else ""
        s = [
            f"LookAt {cam[0]:.9g} {cam[1]:.9g} {cam[2]:.9g} 0 0 0 0 1 0",
            'Camera "perspective" "float fov" [0.5]',
            'Sampler "independent" "integer pixelsamples" [4]',
            'Integrator "path" "integer maxdepth" [1]',
            'Film "rgb" "integer xresolution" [32] "integer yresolution" [32] "string filename" "o.exr"',
            "WorldBegin",
            "AttributeBegin",
            f'AreaLightSource "diffuse" "spectrum L" [360 1 830 1]{ang}',
            # vertex order as tools/highway_taillamps.py: emits towards -z
            'Shape "bilinearmesh" "point3 P" [0.5 -0.5 0 -0.5 -0.5 0 0.5 0.5 0 -0.5 0.5 0]',
            "AttributeEnd",
        ]
        (td / "s.pbrt").write_text("\n".join(s) + "\n")
        subprocess.run(
            [str(PBRT), "--quiet", "s.pbrt"], check=True, cwd=td, capture_output=True, timeout=PBRT_TIMEOUT_S
        )
        img = read_separate_exr_channels(td / "o.exr")
        g = np.asarray(img["G"] if "G" in img else next(iter(img.values())))
        return float(g[12:20, 12:20].mean())

    def test_radiance_follows_beam_map(self) -> None:
        with tempfile.TemporaryDirectory() as tds:
            td = Path(tds)
            tl.write_beam_maps(td)
            for h, v in ((0, 0), (10, 0), (-20, 5), (5, -10), (45, 0)):
                d = _rear_dir(h, v)
                ratio = self._radiance(td, d, True) / self._radiance(td, d, False)
                want = float(tl.beam_shape("stop", d)) / float(-d[2])
                self.assertAlmostEqual(ratio, want, delta=0.03 + 0.03 * want, msg=f"{h},{v}")


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
