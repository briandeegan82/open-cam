"""Night/dusk highway lighting (tools/highway_night.py) and the pbrt retroreflective material.

The pbrt-gated tests render minimal scenes with the patched ``retroreflective`` material
(third_party/patches/0001-retroreflective-material.patch) and measure the coefficient of
retroreflection from the image against a white Lambertian reference patch:
``R_A = (L_retro / L_white) cos(b_i) cos(b_o) / pi`` (ASTM E808/E810) and
``R_L = (L_retro / L_white) cos(b_i) / pi`` with b measured from the normal (EN 1436 uses
illumination on a plane perpendicular to the incident light).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import highway_night as night
import numpy as np
from colour_science import tristimulus, xy_chromaticity
from exr_multispectral import read_separate_exr_channels
from highway_sky import equal_area_directions
from synthetic_data import REPO
from test_highway_scene import build_without_assets

PBRT = Path(os.environ.get("OPENCAM_PBRT", REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"))
WL = night.LIGHT_WL


def xy(spd: np.ndarray) -> np.ndarray:
    return xy_chromaticity(np.array(tristimulus(WL, spd)))[0]


class TestNaturalLight(unittest.TestCase):
    def test_twilight_anchors_and_monotonic(self) -> None:
        for elev, lux in night.TWILIGHT_ANCHORS:
            self.assertAlmostEqual(night.twilight_illuminance_lux(elev), lux, delta=lux * 1e-6)
        e = [night.twilight_illuminance_lux(h) for h in np.linspace(0.0, -18.0, 50)]
        self.assertTrue(all(a > b for a, b in zip(e, e[1:], strict=False)))
        self.assertEqual(night.twilight_illuminance_lux(-40.0), night.NIGHT_SKY_LUX)

    def test_moon(self) -> None:
        # Full moon high in a clear sky: ~0.25-0.3 lx (Krisciunas & Schaefer 1991).
        self.assertTrue(0.2 < night.moon_illuminance_lux(0.0, 90.0) < 0.35)
        self.assertLess(night.moon_illuminance_lux(90.0, 35.0), 0.1 * night.moon_illuminance_lux(0.0, 35.0))
        self.assertEqual(night.moon_illuminance_lux(0.0, -5.0), 0.0)


class TestLampSpectra(unittest.TestCase):
    def test_white_sources_cct(self) -> None:
        for k, t in (("halogen", 3200.0), ("led_headlamp", 5700.0), ("led_4000k", 4000.0)):
            self.assertAlmostEqual(night.cct(WL, night.LAMP_SPECTRA[k](WL)), t, delta=150.0, msg=k)

    def test_hps_chromaticity(self) -> None:
        # Typical 250 W HPS: x ~ 0.52, y ~ 0.41, CCT ~ 2000 K (de Groot & van Vliet 1986).
        s = night.hps_spectrum(WL)
        np.testing.assert_allclose(xy(s), (0.52, 0.41), atol=0.02)
        self.assertLess(s[np.searchsorted(WL, 589.0)], 0.5 * s.max())  # self-reversed Na D line

    def test_red_led_inside_ece_red(self) -> None:
        x, y = xy(night.red_led_spectrum(WL))
        self.assertLessEqual(y, 0.335)  # ECE R48 / R7 red boundaries
        self.assertGreaterEqual(y, 0.980 - x)


class TestLowBeam(unittest.TestCase):
    def test_r112_test_points(self) -> None:
        for kind, peak in night.HEADLAMP_PEAK_CD.items():
            for name, h, v, e_min, e_max in night.R112_POINTS:
                e = float(night.low_beam_cd(h, v, peak)) / 25.0**2  # lux on the 25 m screen
                if e_min is not None:
                    self.assertGreaterEqual(e, e_min, f"{kind} {name}")
                if e_max is not None:
                    self.assertLessEqual(e, e_max, f"{kind} {name}")
            for h, v in night.R112_ZONE_III:
                self.assertLessEqual(float(night.low_beam_cd(h, v, peak)) / 625.0, night.R112_ZONE_III_MAX_LUX)

    def test_map_flux_and_lookup(self) -> None:
        d = equal_area_directions(64)
        uv = night.equal_area_sphere_to_square(d)
        c = (np.arange(64) + 0.5) / 64
        u, v = np.meshgrid(c, c)
        np.testing.assert_allclose(uv[..., 0], u, atol=1e-9)
        np.testing.assert_allclose(uv[..., 1], v, atol=1e-9)
        m = night.headlamp_map(256, "halogen")
        self.assertTrue(250.0 < night.flux_lm(m) < 1000.0)  # 55 W H7 low beam out of the lens
        s = night.streetlight_map(256, 16_000.0)
        self.assertAlmostEqual(night.flux_lm(s), 16_000.0, delta=160.0)


class TestStreetLighting(unittest.TestCase):
    def test_road_illuminance_class(self) -> None:
        # EN 13201 / RP-8 freeway lighting: E_avg ~ 10-30 lx, overall uniformity U0 >= 0.4.
        for kind, p in night.STREETLIGHT.items():
            r = night.road_illuminance(p["flux_lm"], -2.2, (0.0, 10.98), (100.0, 148.0))
            self.assertTrue(10.0 < r["e_avg_lux"] < 30.0, kind)
            self.assertGreaterEqual(r["uniformity_u0"], 0.4, kind)


class TestRetroreflectionModel(unittest.TestCase):
    def test_sheeting_meets_astm_d4956_type_iii(self) -> None:
        for colour, minima in night.D4956_TYPE_III.items():
            ra = night.sheeting_ra_param(colour)
            for (alpha, beta), r_min in zip(night.D4956_GEOMETRIES, minima, strict=True):
                r = float(night.retro_ra(*night.sheeting_geometry(alpha, beta), ra, **night._kw(night.SHEETING)))
                self.assertGreaterEqual(r, r_min, f"{colour} {alpha}/{beta}")
            r0 = night.retro_ra(*night.sheeting_geometry(0.2, -4.0), ra, **night._kw(night.SHEETING))
            self.assertAlmostEqual(float(r0), night.SHEETING_MARGIN * minima[0], places=6)

    def test_marking_rl_en1436(self) -> None:
        a, ci, co = night.marking_geometry()
        self.assertAlmostEqual(math.degrees(a), 1.05, places=6)
        for colour, rl in night.MARKING_RL.items():
            ra = night.marking_ra_param(rl)
            r_l = float(night.retro_ra(a, ci, co, ra, **night._kw(night.MARKING))) / co
            self.assertAlmostEqual(1e3 * r_l, rl, places=6, msg=colour)


class TestBuilderNight(unittest.TestCase):
    def test_day_default_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m = build_without_assets(Path(td))
            scene = (Path(td) / "scene" / "highway.pbrt").read_text()
            self.assertNotIn("retroreflective", scene)
            self.assertNotIn("goniometric", scene)
            self.assertNotIn("artificial", m["lighting"])
            self.assertGreater(m["lighting"]["reference_illuminance_lux"], 1e4)

    def test_night_reference_illuminance_and_lights(self) -> None:
        from pbrt_spectral_exr_to_electrons import PBRT_CIE_Y_INTEGRAL

        with tempfile.TemporaryDirectory() as td:
            m = build_without_assets(Path(td), "--time-of-day", "night")
            out = Path(td) / "scene"
            scene = (out / "highway.pbrt").read_text()
            lit = m["lighting"]
            self.assertEqual(lit["time_of_day"], "night")
            self.assertAlmostEqual(lit["reference_illuminance_lux"], night.NIGHT_SKY_LUX)
            self.assertAlmostEqual(
                lit["reference_illuminance_exr_lux"], 683.0 * PBRT_CIE_Y_INTEGRAL * night.NIGHT_SKY_LUX
            )
            self.assertNotIn('"distant"', scene)
            self.assertIn('"retroreflective"', scene)
            self.assertIn('LightSource "goniometric"', scene)
            self.assertIn("AreaLightSource", scene)
            self.assertTrue((out / "spd" / "lamp_led_4000k.spd").is_file())
            self.assertTrue(lit["artificial"]["headlamps"])
            self.assertTrue(lit["artificial"]["tail_lamps"])
            self.assertGreater(lit["artificial"]["streetlights"]["poles"], 0)
            self.assertIn("retroreflection", m)

    def test_haze_at_night_reference_illuminance(self) -> None:
        """Night + fog: reference = natural road illuminance under the medium (MC model), lights unscaled."""
        with tempfile.TemporaryDirectory() as td:
            m = build_without_assets(Path(td), "--time-of-day", "night", "--haze", "fog")
            road = m["atmosphere"]["road_illuminance"]
            t = road["sources"]["night_sky"]["total_transmittance"]
            self.assertLess(t, 0.97)
            self.assertGreater(t, 0.2)
            self.assertAlmostEqual(m["lighting"]["reference_illuminance_lux"], night.NIGHT_SKY_LUX * t, places=9)
            scene = (Path(td) / "scene" / "highway.pbrt").read_text()
            self.assertIn('MakeNamedMedium "haze"', scene)
            self.assertIn('"volpath"', scene)

    def test_haze_at_dusk_light_reference_road(self) -> None:
        """--haze-light-reference road: natural sources scaled so the road gets the no-medium value."""
        with tempfile.TemporaryDirectory() as td:
            args = ("--time-of-day", "dusk", "--moon-phase-deg", "30", "--haze", "mist")
            m = build_without_assets(Path(td), *args, "--haze-light-reference", "road")
            a = m["atmosphere"]
            self.assertGreater(a["light_scale_applied"], 1.0)
            self.assertAlmostEqual(
                m["lighting"]["reference_illuminance_lux"], a["no_medium_horizontal_illuminance_lux"], places=9
            )
            self.assertAlmostEqual(
                m["lighting"]["sky"]["illuminance_horizontal_lux"],
                a["road_illuminance"]["sources"]["twilight_sky"]["horizontal_top_lux"],
                places=6,
            )

    def test_variety_lamp_posts_lit(self) -> None:
        """--lamp-posts on (tools/highway_variety.py): its posts carry the luminaires, no duplicate poles."""
        with tempfile.TemporaryDirectory() as td:
            build_without_assets(Path(td), "--time-of-day", "night", "--seed", "3", "--lamp-posts", "on")
            m = json.loads((Path(td) / "scene" / "highway_manifest.json").read_text())
            sl = m["lighting"]["artificial"]["streetlights"]
            heads = m["variety"]["lamp_posts"]["heads"]
            self.assertEqual(sl["pole_source"], "tools/highway_variety.py lamp posts")
            self.assertEqual(sl["poles"] * 2, len(heads))
            self.assertAlmostEqual(sl["spacing_m"], m["variety"]["lamp_posts"]["spacing_m"])
            txt = (Path(td) / "scene" / "highway.pbrt").read_text()
            self.assertNotIn('NamedMaterial "pole"', txt)
            self.assertGreater(sl["carriageway_illuminance"]["e_avg_lux"], 5.0)

    def test_dusk_moon_and_options(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m = build_without_assets(Path(td), "--time-of-day", "dusk")
            self.assertAlmostEqual(
                m["lighting"]["reference_illuminance_lux"], night.twilight_illuminance_lux(-4.0), places=6
            )
        with tempfile.TemporaryDirectory() as td:
            m = build_without_assets(
                Path(td), "--time-of-day", "night", "--moon-phase-deg", "0", "--streetlights", "hps"
            )
            moon = m["lighting"]["moon"]
            e_n = night.moon_illuminance_lux(0.0, 35.0)
            self.assertAlmostEqual(moon["illuminance_horizontal_lux"], e_n * math.sin(math.radians(35.0)), places=9)
            self.assertTrue(0.0 < moon["moonlit_sky_horizontal_lux"] < moon["illuminance_horizontal_lux"])
            self.assertAlmostEqual(
                m["lighting"]["reference_illuminance_lux"],
                night.NIGHT_SKY_LUX + moon["illuminance_horizontal_lux"] + moon["moonlit_sky_horizontal_lux"],
                places=9,
            )
            self.assertTrue((Path(td) / "scene" / "spd" / "lamp_hps.spd").is_file())
        with tempfile.TemporaryDirectory() as td:
            m = build_without_assets(Path(td), "--headlamps", "led", "--retroreflective", "on")
            scene = (Path(td) / "scene" / "highway.pbrt").read_text()
            self.assertIn('"retroreflective"', scene)
            self.assertIn('LightSource "goniometric"', scene)
            self.assertGreater(m["lighting"]["reference_illuminance_lux"], 1e4)  # still daylight


def _pbrt_has_retro() -> bool:
    if not PBRT.is_file():
        return False
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.pbrt"
        p.write_text(
            'Film "rgb" "integer xresolution" [1] "integer yresolution" [1] "string filename" "t.exr"\n'
            'Camera "perspective"\nWorldBegin\nMaterial "retroreflective"\n'
        )
        r = subprocess.run([str(PBRT), "--quiet", str(p)], capture_output=True, cwd=td)
        return r.returncode == 0


PBRT_RETRO = _pbrt_has_retro()
if os.environ.get("OPENCAM_REQUIRE_PBRT") == "1" and not PBRT_RETRO:
    raise RuntimeError(f"OPENCAM_REQUIRE_PBRT=1 but {PBRT} lacks the retroreflective patch (tools/build_pbrt.sh)")


def _quad(c: np.ndarray, u: np.ndarray, v: np.ndarray) -> str:
    p = [c - u - v, c + u - v, c - u + v, c + u + v]
    return '    Shape "bilinearmesh" "point3 P" [' + " ".join(f"{x:.9g}" for q in p for x in q) + "]"


def _render(td: Path, cam: np.ndarray, look: np.ndarray, up: np.ndarray, light: np.ndarray, body: list[str]) -> dict:
    res, fov = 160, 1.2
    s = [
        f"LookAt {' '.join(map(str, cam))} {' '.join(map(str, look))} {' '.join(map(str, up))}",
        f'Camera "perspective" "float fov" [{fov}]',
        'Sampler "independent" "integer pixelsamples" [16]',
        'Integrator "path" "integer maxdepth" [1]',
        f'Film "rgb" "integer xresolution" [{res}] "integer yresolution" [{res}] "string filename" "out.exr"',
        "WorldBegin",
        f'LightSource "point" "point3 from" [{" ".join(map(str, light))}] "spectrum I" [360 1 830 1]',
        *body,
    ]
    (td / "s.pbrt").write_text("\n".join(s) + "\n")
    subprocess.run([str(PBRT), "--quiet", "--seed", "3", "s.pbrt"], check=True, cwd=td, capture_output=True)
    return read_separate_exr_channels(td / "out.exr")


def _patch_level(img: np.ndarray) -> float:
    v = img[img > 0]
    return float(np.median(v[v > 0.8 * np.percentile(v, 95)]))


def _cosines(centre: np.ndarray, n: np.ndarray, cam: np.ndarray, light: np.ndarray) -> tuple[float, float, float]:
    wi, wo = light - centre, cam - centre
    wi, wo = wi / np.linalg.norm(wi), wo / np.linalg.norm(wo)
    return 2.0 * math.asin(np.linalg.norm(wi - wo) / 2.0), float(n @ wi), float(n @ wo)


RETRO_MAT = (
    'Material "retroreflective" "spectrum reflectance" [360 0 830 0] "spectrum ra" [360 {ra} 830 {ra}]'
    ' "float observationangle" [{a0}] "float lobewidth" [{w}] "float entranceexponent" [{q}]'
)
WHITE_MAT = 'Material "diffuse" "spectrum reflectance" [360 1 830 1]'


@unittest.skipUnless(PBRT_RETRO, f"patched pbrt not built ({PBRT}); see docs/BUILD_PBRT.txt")
class TestPbrtRetroreflection(unittest.TestCase):
    """Measure R_A / R_L from renders of the patched material against the analytic targets."""

    def test_sheeting_ra_astm_geometries(self) -> None:
        p = night.SHEETING
        ra = night.sheeting_ra_param("white")
        mat = RETRO_MAT.format(ra=ra, a0=p["alpha0_deg"], w=p["width_deg"], q=p["q"])
        d = 50.0
        cam, look, up = np.zeros(3), np.array([0.0, 0.0, d]), np.array([0.0, 1.0, 0.0])
        for (alpha, beta), r_min in zip(night.D4956_GEOMETRIES, night.D4956_TYPE_III["white"], strict=True):
            b = math.radians(beta)
            n = np.array([-math.sin(b), 0.0, -math.cos(b)])  # rotated about the vertical axis
            t = np.array([math.cos(b), 0.0, -math.sin(b)])
            light = np.array([d * math.tan(math.radians(alpha)), 0.0, 0.0])
            c_r, c_w = np.array([0.0, 0.14, d]), np.array([0.0, -0.14, d])
            body = [
                "AttributeBegin",
                mat,
                _quad(c_r, 0.25 * t, np.array([0.0, 0.12, 0.0])),
                "AttributeEnd",
                "AttributeBegin",
                WHITE_MAT,
                _quad(c_w, 0.25 * t, np.array([0.0, 0.12, 0.0])),
                "AttributeEnd",
            ]
            with tempfile.TemporaryDirectory() as td:
                g = _render(Path(td), cam, look, up, light, body)["G"]
            h = g.shape[0] // 2
            lr, lw = _patch_level(g[:h]), _patch_level(g[h:])
            a, ci, co = _cosines(c_r, n, cam, light)
            measured = lr / lw * ci * co / math.pi
            expected = float(night.retro_ra(a, ci, co, ra, **night._kw(p)))
            self.assertAlmostEqual(measured / expected, 1.0, delta=0.04, msg=f"{alpha}/{beta}")
            self.assertGreaterEqual(measured, r_min, f"{alpha}/{beta}")

    def test_marking_rl_en1436_geometry(self) -> None:
        p = night.MARKING
        ra = night.marking_ra_param(night.MARKING_RL["white"])
        mat = RETRO_MAT.format(ra=ra, a0=p["alpha0_deg"], w=p["width_deg"], q=p["q"])
        d = 30.0
        cam = np.array([0.0, d * math.tan(math.radians(night.EN1436["view_elev_deg"])), 0.0])
        light = np.array([0.0, d * math.tan(math.radians(night.EN1436["illum_elev_deg"])), 0.0])
        n = np.array([0.0, 1.0, 0.0])
        c_r, c_w = np.array([0.16, 0.0, d]), np.array([-0.16, 0.0, d])
        half = (np.array([0.13, 0.0, 0.0]), np.array([0.0, 0.0, 0.6]))
        body = ["AttributeBegin", mat, _quad(c_r, *half), "AttributeEnd"]
        body += ["AttributeBegin", WHITE_MAT, _quad(c_w, *half), "AttributeEnd"]
        with tempfile.TemporaryDirectory() as td:
            g = _render(Path(td), cam, np.array([0.0, 0.0, d]), np.array([0.0, 1.0, 0.0]), light, body)["G"]
        w = g.shape[1] // 2
        levels = sorted((_patch_level(g[:, :w]), _patch_level(g[:, w:])))
        lw, lr = levels
        _, ci, _ = _cosines(np.array([0.0, 0.0, d]), n, cam, light)
        r_l_mcd = 1e3 * lr / lw * ci / math.pi
        self.assertAlmostEqual(r_l_mcd / night.MARKING_RL["white"], 1.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
