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
        self.assertEqual(kinds, {"hdri", "texture", "car_pbrt", "gltf"})
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
