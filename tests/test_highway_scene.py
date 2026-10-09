"""Highway scene: asset manifest, fetch/prepare helpers, sky maths and scene generation (no pbrt)."""

from __future__ import annotations

import base64
import json
import re
import tempfile
import unittest
from pathlib import Path

import build_highway_scene as hw
import fetch_highway_assets as fetch
import highway_atmosphere as atm
import highway_backdrop as backdrop
import highway_sky as sky
import numpy as np
import pbrt_spectral_exr_to_electrons as pbrt_tool
import yaml
from highway_spectra import SURFACES, reflectance
from synthetic_data import REPO


def build_without_assets(out: Path, *extra: str) -> dict:
    """Run the builder against an empty asset cache (proxy cars, analytic sky)."""
    m = yaml.safe_load((REPO / "config" / "highway_assets.yaml").read_text())
    m["cache_dir"] = str(out / "empty_cache")
    am = out / "assets.yaml"
    am.write_text(yaml.safe_dump(m))
    hw.main(
        [
            "--out-dir",
            str(out / "scene"),
            "--asset-manifest",
            str(am),
            "--allow-missing-assets",
            "--xres",
            "48",
            "--yres",
            "32",
            "--pixelsamples",
            "4",
            *extra,
        ]
    )
    return json.loads((out / "scene" / "highway_manifest.json").read_text())


class TestAssetManifest(unittest.TestCase):
    def test_every_asset_is_pinned_and_licensed(self) -> None:
        m = fetch.load_asset_manifest()
        self.assertTrue(m["cache_dir"].startswith("scenes/assets"))
        for aid, a in m["assets"].items():
            self.assertIn(a["license"], ("CC0-1.0", "CC-BY-4.0"), aid)
            self.assertTrue(a["source"].startswith("https://"), aid)
            for f in a.get("files", []):
                self.assertTrue(f["url"].startswith("https://"), aid)
        kinds = {a["kind"] for a in m["assets"].values()}
        self.assertEqual(kinds, {"hdri", "texture", "car_pbrt", "gltf", "glb"})
        for model in hw.CAR_MODELS:
            self.assertGreater(m["assets"][model]["length_m"], 3.5)

    def test_cache_dir_is_gitignored(self) -> None:
        self.assertIn("scenes/assets/", (REPO / ".gitignore").read_text().split())

    def test_unpinned_asset_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.yaml"
            p.write_text(
                yaml.safe_dump(
                    {
                        "assets": {
                            "x": {"kind": "hdri", "title": "x", "source": "s", "license": "CC0-1.0", "files": [{}]}
                        }
                    }
                )
            )
            with self.assertRaises(ValueError):
                fetch.load_asset_manifest(p)


class TestPrepareHelpers(unittest.TestCase):
    def test_parse_pbrt_car(self) -> None:
        text = """Camera "perspective"
WorldBegin
MakeNamedMaterial "Paint"
    "string type" [ "coateddiffuse" ]
    "rgb reflectance" [ 0.2 0.2 0.8 ]
MakeNamedMaterial "Mix"
    "string type" "mix" "string materials" [ "Paint" "Paint" ]
NamedMaterial "Paint"
Shape "plymesh"
    "string filename" [ "models/a.ply" ]
AttributeBegin
    NamedMaterial "Mix"
    Shape "plymesh"
        "string filename" [ "models/b.ply" ]
AttributeEnd
"""
        mats, shapes = fetch.parse_pbrt_car(text)
        self.assertEqual(set(mats), {"Paint", "Mix"})
        self.assertIn("coateddiffuse", mats["Paint"])
        self.assertEqual(
            shapes, [{"material": "Paint", "ply": "models/a.ply"}, {"material": "Mix", "ply": "models/b.ply"}]
        )

    def test_gltf_to_ply_roundtrip(self) -> None:
        pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        idx = np.array([0, 1, 2], dtype=np.uint16)
        blob = pos.tobytes() + idx.tobytes()
        g = {
            "asset": {"version": "2.0"},
            "buffers": [{"byteLength": len(blob), "uri": "data:;base64," + base64.b64encode(blob).decode()}],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36}, {"buffer": 0, "byteOffset": 36}],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
            ],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
            "nodes": [{"mesh": 0, "translation": [0, 2, 0], "scale": [2, 2, 2]}],
            "scenes": [{"nodes": [0]}],
        }
        with tempfile.TemporaryDirectory() as td:
            raw, out = Path(td) / "raw", Path(td) / "out"
            raw.mkdir()
            (raw / "m.gltf").write_text(json.dumps(g))
            info = fetch.prepare_gltf({"gltf": "m.gltf"}, raw, out)
            p = fetch.read_ply_positions(out / info["shapes"][0]["ply"])
        np.testing.assert_allclose(p, [[0, 2, 0], [2, 2, 0], [0, 4, 0]], atol=1e-6)
        self.assertEqual(info["bbox_max"], [2.0, 4.0, 0.0])


class TestSky(unittest.TestCase):
    def test_equal_area_map_is_unit_and_uniform(self) -> None:
        d = sky.equal_area_directions(64)
        np.testing.assert_allclose(np.linalg.norm(d, axis=-1), 1.0, atol=1e-9)
        self.assertAlmostEqual(float(d[..., 2].mean()), 0.0, places=6)
        # Pixels on the square's inner diamond lie exactly on the horizon (z = 0).
        upper = float((d[..., 2] > 1e-12).mean()) + 0.5 * float((np.abs(d[..., 2]) <= 1e-12).mean())
        self.assertAlmostEqual(upper, 0.5, places=6)

    def test_excise_sun_recovers_direction_and_energy(self) -> None:
        h, w = 256, 512
        img = np.full((h, w, 3), 0.5)
        theta, dom = sky.equirect_solid_angles(h, w)
        r, c = 80, 100
        img[r - 1 : r + 2, c - 1 : c + 2] = 5000.0
        _, info = sky.excise_sun(img)
        d_true = sky.equirect_dirs(h, w)[r, c]
        self.assertGreater(float(np.dot(info["sun_dir_light"], d_true)), 0.9999)
        self.assertAlmostEqual(info["e_sun_perp_rel"], 9 * 4999.5 * dom[r], delta=0.03 * 9 * 4999.5 * dom[r])
        self.assertAlmostEqual(info["e_sky_horizontal_rel"], 0.5 * np.pi, delta=0.02)

    def test_clear_sky_model(self) -> None:
        e_dn, e_d = sky.clear_sky_illuminance_lux(43.0)
        self.assertTrue(80_000 < e_dn < 110_000 and 10_000 < e_d < 16_000)
        low, _ = sky.clear_sky_illuminance_lux(6.0)
        self.assertLess(low, 0.5 * e_dn)
        wl = np.arange(400.0, 701.0, 10.0)
        noon, dusk = sky.solar_direct_spectrum(wl, 60.0), sky.solar_direct_spectrum(wl, 5.0)
        self.assertLess(dusk[0] / dusk[-1], noon[0] / noon[-1])  # low sun is redder

    def test_reflectances_physical(self) -> None:
        wl = np.arange(360.0, 831.0, 5.0)
        for name in SURFACES:
            r = reflectance(name, wl)
            self.assertTrue(np.all((r >= 0) & (r <= 1)), name)
        self.assertLess(reflectance("asphalt_aged", wl).max(), 0.2)


class TestBuildHighwayScene(unittest.TestCase):
    def test_scene_without_assets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            m = build_without_assets(out)
            text = (out / "scene" / "highway.pbrt").read_text()
            self.assertIn('LightSource "distant" "spectrum L" "spd/sun.spd"', text)
            self.assertIn('LightSource "infinite"', text)
            self.assertEqual(text.count("float illuminance"), 2)
            self.assertIn('Film "spectral"', text)
            self.assertEqual(len(re.findall(r'^Include "cars/', text, re.M)), len(hw.DEFAULT_TRAFFIC))
            for inc in (out / "scene" / "cars").glob("*.pbrt"):
                self.assertIn("coateddiffuse", inc.read_text())
            self.assertTrue((out / "scene" / "spd" / "sun.spd").is_file())
            self.assertTrue(any((out / "scene" / "textures").glob("sky_cie12_*.exr")))

        self.assertEqual(m["camera"]["type"], "pinhole")
        self.assertAlmostEqual(m["camera"]["lookat"]["eye"][1], 1.35)
        self.assertIn("sky_kloofendal_43d_clear", m["missing_assets_fallback"])
        self.assertEqual({c["model"] for c in m["cars"]}, {"proxy"})
        lt = m["lighting"]
        self.assertAlmostEqual(
            lt["reference_illuminance_lux"],
            lt["sun"]["illuminance_horizontal_lux"] + lt["sky"]["illuminance_horizontal_lux"],
            places=6,
        )
        self.assertGreater(lt["reference_illuminance_lux"], 50_000)
        sun = np.array(lt["sun"]["direction_world"])
        self.assertAlmostEqual(float(np.degrees(np.arcsin(sun[1]))), 45.0, places=6)
        self.assertAlmostEqual(float(np.degrees(np.arctan2(sun[0], sun[2]))), 140.0, places=6)

        scene = pbrt_tool.scene_radiometry_from_manifest(m)
        self.assertEqual(scene["exr_quantity"], "radiance")
        self.assertAlmostEqual(scene["scene_illuminance_lux"], lt["reference_illuminance_lux"])
        self.assertAlmostEqual(
            scene["chart_illuminance_exr_lux"], 683.0 * pbrt_tool.PBRT_CIE_Y_INTEGRAL * lt["reference_illuminance_lux"]
        )

    def test_realistic_camera_low_sun(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m = build_without_assets(
                Path(td), "--camera", "realistic", "--sun-elevation", "6", "--global-illuminance-lux", "5000"
            )
            text = (Path(td) / "scene" / "highway.pbrt").read_text()
        self.assertIn('Camera "realistic"', text)
        self.assertIn("wide_22mm.dat", text)
        self.assertAlmostEqual(m["lighting"]["reference_illuminance_lux"], 5000.0)
        self.assertEqual(pbrt_tool.scene_radiometry_from_manifest(m)["exr_quantity"], "irradiance")
        self.assertEqual(m["camera"]["focus_distance"], 25.0)

    def test_missing_assets_is_an_error_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m = yaml.safe_load((REPO / "config" / "highway_assets.yaml").read_text())
            m["cache_dir"] = str(Path(td) / "none")
            am = Path(td) / "a.yaml"
            am.write_text(yaml.safe_dump(m))
            with self.assertRaises(SystemExit):
                hw.main(["--out-dir", str(Path(td) / "s"), "--asset-manifest", str(am)])


if __name__ == "__main__":
    unittest.main()


class TestHighwayAtmosphere(unittest.TestCase):
    def test_koschmieder_and_angstrom(self) -> None:
        wl = np.array([450.0, 550.0, 650.0])
        for p in atm.PRESETS.values():
            sa, ss, _ = atm.coefficients(p, wl)
            ext = sa + ss
            self.assertAlmostEqual(ext[1], 3.912 / p.visibility_m, delta=1e-4 * ext[1])
            aer = ext - atm.RAYLEIGH_550_PER_M * (wl / 550.0) ** -atm.RAYLEIGH_EXPONENT
            self.assertAlmostEqual(aer[0] / aer[2], (450.0 / 650.0) ** -p.angstrom, places=6)
            self.assertTrue(np.all(sa >= 0) and np.all(ss > 0))
            self.assertAlmostEqual(float(atm.profile_density(p, 1.35)), 1.0, delta=0.02)  # V holds at the road

    def test_presets_order_and_depth(self) -> None:
        names = ["clear", "hazy", "mist", "fog"]
        vis = [atm.PRESETS[n].visibility_m for n in names]
        tau = [atm.optical_depth_vertical(atm.PRESETS[n]) for n in names]
        depth = [atm.integrator_maxdepth(atm.PRESETS[n], 5) for n in names]
        self.assertEqual(vis, sorted(vis, reverse=True))
        self.assertEqual(tau, sorted(tau))
        self.assertEqual(depth, sorted(depth))
        self.assertEqual(atm.integrator_maxdepth(atm.PRESETS["fog"], 5, override=7), 7)
        for n in names:
            self.assertGreater(atm.effective_g(atm.PRESETS[n]), 0.6)

    def test_cli_overrides(self) -> None:
        p = atm.params_from_args(hw.parse_args(["--haze", "fog", "--visibility-m", "60", "--haze-g", "0.8"]))
        self.assertEqual((p.name, p.visibility_m, p.g, p.albedo), ("fog", 60.0, 0.8, atm.PRESETS["fog"].albedo))
        self.assertIsNone(atm.params_from_args(hw.parse_args([])))
        with self.assertRaises(ValueError):
            atm.params_from_args(hw.parse_args(["--haze", "mist", "--visibility-m", "-1"]))

    def test_slab_mc_direct_beam_matches_beer_lambert(self) -> None:
        rng = np.random.default_rng(3)
        for name in ("hazy", "fog"):
            p = atm.PRESETS[name]
            mu = 0.6
            tot, direct = atm.slab_flux_mc(p, 550.0, np.full(40_000, mu), 0.0, rng)
            expect = np.exp(-atm.optical_depth_vertical(p) / mu)
            self.assertAlmostEqual(direct, expect, delta=4 * np.sqrt(expect / 40_000) + 1e-4)
            self.assertGreaterEqual(tot, direct)
            self.assertLessEqual(tot, 1.0)

    def test_scene_with_fog(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            for k in "abc":
                (out / k).mkdir()
            build_without_assets(out / "a", "--haze", "fog")
            build_without_assets(out / "b", "--haze", "fog", "--haze-light-reference", "road")
            build_without_assets(out / "c")
            ms = [json.loads((out / k / "scene" / "highway_manifest.json").read_text()) for k in "abc"]
            txt = [(out / k / "scene" / "highway.pbrt").read_text() for k in "abc"]
        fog, fog_road, plain = ms
        a = fog["atmosphere"]
        self.assertEqual(a["name"], "fog")
        self.assertEqual(fog["lighting"]["reference_illuminance_lux"], a["road_illuminance"]["road_lux"])
        self.assertLess(fog["lighting"]["reference_illuminance_lux"], 0.9 * a["no_medium_horizontal_illuminance_lux"])
        self.assertAlmostEqual(
            fog["lighting"]["reference_illuminance_exr_lux"] / fog["lighting"]["reference_illuminance_lux"],
            plain["lighting"]["reference_illuminance_exr_lux"] / plain["lighting"]["reference_illuminance_lux"],
        )
        self.assertAlmostEqual(
            fog_road["lighting"]["reference_illuminance_lux"], plain["lighting"]["reference_illuminance_lux"], places=3
        )
        self.assertGreater(fog_road["atmosphere"]["light_scale_applied"], 1.0)
        self.assertIsNone(plain["atmosphere"])
        self.assertIsNone(plain["distant_terrain"])
        self.assertIsNotNone(fog["distant_terrain"])
        t = txt[0]
        self.assertIn('Integrator "volpath"', t)
        self.assertLess(t.index('MediumInterface "haze" "haze"'), t.index("Camera "))
        self.assertLess(t.index("Camera "), t.index("MakeNamedMedium"))
        self.assertLess(t.index("MakeNamedMedium"), t.index("WorldBegin"))
        self.assertIn("Distant terrain", t)
        self.assertIn('Integrator "path"', txt[2])
        self.assertNotIn("Medium", txt[2])
        self.assertNotIn("Distant terrain", txt[2])


class TestHighwayBackdrop(unittest.TestCase):
    def test_hills_mesh(self) -> None:
        P, tri, N = backdrop.hills_mesh(5.0, 0.0, seed=0)
        self.assertEqual(tri.max(), len(P) - 1)
        self.assertTrue(np.all(N[:, 1] > 0.5))  # gentle slopes, facing up
        r = np.hypot(P[:, 0] - 5.0, P[:, 2])
        self.assertAlmostEqual(r.min(), backdrop.R0_M, delta=1.0)
        self.assertAlmostEqual(r.max(), backdrop.R1_M, delta=1.0)
        elev = np.degrees(np.arctan2(P[:, 1] - 1.35, r))
        self.assertLess(elev.max(), 2.5)
        self.assertGreater(elev.max(), 1.0)  # visible above the horizon
        P1, _, _ = backdrop.hills_mesh(5.0, 0.0, seed=0)
        P2, _, _ = backdrop.hills_mesh(5.0, 0.0, seed=1)
        np.testing.assert_array_equal(P, P1)
        self.assertGreater(np.abs(P[:, 1] - P2[:, 1]).max(), 10.0)

    def test_canopy_reflectance(self) -> None:
        wl = np.arange(400.0, 1001.0, 10.0)
        rho = backdrop.forest_canopy_reflectance(wl)
        self.assertTrue(np.all((rho > 0.0) & (rho < 0.4)))
        vis = rho[(wl >= 400) & (wl <= 680)]  # red edge starts at ~700 nm
        self.assertLess(vis.max(), 0.07)
        self.assertGreater(rho[wl == 850][0], 4 * vis.mean())  # red edge
