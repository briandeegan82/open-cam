"""--road / --verge ambientCG surfaces: manifest provenance, spectral shapes, asphalt-only wear."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import build_highway_scene as hw
import yaml
from synthetic_data import REPO
from test_highway_scene import build_without_assets

MANIFEST = yaml.safe_load((REPO / "config" / "highway_assets.yaml").read_text())["assets"]


class TestRoadSurfaces(unittest.TestCase):
    def test_manifest_entries(self):
        for name, spd in {**hw.ROAD_SURFACES, **hw.VERGE_SURFACES}.items():
            e = MANIFEST[f"tex_{name}"]
            self.assertEqual(e["kind"], "texture")
            self.assertEqual(e["license"], "CC0-1.0")
            self.assertTrue(e["source"].startswith("https://ambientcg.com/view?id="))
            self.assertGreater(float(e["tile_size_m"]), 0.5)
            self.assertLessEqual({"color", "normal"}, set(e["maps"]))
            for f in e["files"]:
                self.assertEqual(len(f["sha256"]), 64)
            self.assertEqual(e.get("spd", spd), spd)

    def test_spd_names_known(self):
        from highway_spectra import SURFACES

        for spd in {*hw.ROAD_SURFACES.values(), *hw.VERGE_SURFACES.values()}:
            self.assertIn(spd, SURFACES)

    def test_non_asphalt_rejects_wear(self):
        with tempfile.TemporaryDirectory() as td, self.assertRaises(SystemExit):
            build_without_assets(Path(td), "--road", "concrete046", "--road-wear", "light")

    def test_surface_and_verge_in_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            m = build_without_assets(Path(td), "--road", "pavingstones070", "--verge", "gravel023")
            road = m["road"]
            self.assertEqual((road["surface"], road["surface_spd"], road["verge"]),
                             ("pavingstones070", "concrete", "gravel023"))  # fmt: skip
            self.assertIsNone(road["wear"])
            text = (Path(td) / "scene" / "highway.pbrt").read_text()
            self.assertIn('"spd/concrete.spd"', text)

    def test_default_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            road = build_without_assets(Path(td))["road"]
            self.assertEqual(
                (road["surface"], road["surface_spd"], road["verge"]), ("asphalt031", "asphalt_aged", "grass004")
            )
            self.assertIsNotNone(road["wear"])


if __name__ == "__main__":
    unittest.main()
