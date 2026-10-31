"""Rendered coateddiffuse reflectance matches the measured spectrum (tools/coat_calibration.py)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import coat_calibration as coat  # noqa: E402
import highway_wet as wet  # noqa: E402

PBRT = Path(os.environ.get("OPENCAM_PBRT", REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"))


class TestTable(unittest.TestCase):
    def test_round_trip(self) -> None:
        for eta, r, target in ((1.5, 0.35, 0.12), (1.5, 0.08, 0.04), (1.333, 0.038, 0.3)):
            s = coat.base_reflectance(target, eta, r)
            self.assertAlmostEqual(float(coat.rendered_reflectance(s, eta, r)), target, places=6)

    def test_clips_where_the_coat_cannot_reach(self) -> None:
        self.assertLess(coat.max_reflectance(1.5, 0.4), 0.5)
        self.assertEqual(float(coat.base_reflectance(0.7, 1.5, 0.4)), 1.0)

    def test_wet_target_is_water_surface_plus_lekner_dorf(self) -> None:
        r = wet.ROUGHNESS["wet"]
        rendered = coat.rendered_reflectance(wet.wet_base_reflectance(0.12, r), wet.N_WATER, r)
        expect = coat.rendered_reflectance(0.0, wet.N_WATER, r) + wet.body_albedo(0.12, wet.N_WATER)
        self.assertAlmostEqual(float(rendered), float(expect), places=6)

    def test_calibrate_scene_leaves_lambertian_uses(self) -> None:
        lines = [
            'Texture "t" "spectrum" "scale" "spectrum tex" "spd/asphalt_aged.spd"',
            'MakeNamedMaterial "a" "string type" "coateddiffuse"',
            '    "spectrum reflectance" "spd/rubber.spd"',
            'MakeNamedMaterial "b" "string type" "diffuse" "spectrum reflectance" "spd/rubber.spd"',
            'MakeNamedMaterial "c" "string type" "coateddiffuse" "spectrum reflectance" "spd/carpaint_red.spd"',
        ]
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            for n in ("asphalt_aged", "rubber", "carpaint_red"):
                (d / f"{n}.spd").write_text("400 0.1\n700 0.1\n")

            def write(p, wl, v):
                np.savetxt(p, np.column_stack([wl, v]))

            out, meta = coat.calibrate_scene(lines, d, write)
            self.assertTrue((d / "coat_rubber.spd").is_file())
        self.assertIn("spd/coat_asphalt_aged.spd", out[0])
        self.assertIn("spd/coat_rubber.spd", out[2])
        self.assertIn('"spd/rubber.spd"', out[3])
        self.assertIn('"spd/carpaint_red.spd"', out[4])
        self.assertEqual(meta["rubber"]["roughness"], 0.45)


@unittest.skipUnless(PBRT.is_file(), "needs pbrt")
class TestRendered(unittest.TestCase):
    def test_table_matches_pbrt_off_grid(self) -> None:
        """A substrate and roughness between grid points render what the table says (within 2 %)."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            white = coat._furnace(str(PBRT), tmp, 'Material "diffuse" "spectrum reflectance" [300 1 800 1]')
            for eta, r, target in ((1.5, 0.35, 0.12), (1.5, 0.35, 0.3), (1.333, 0.038, 0.0824)):
                s = float(coat.base_reflectance(target, eta, r))
                got = coat._furnace(
                    str(PBRT),
                    tmp,
                    f'Material "coateddiffuse" "spectrum reflectance" [300 {s} 800 {s}]'
                    f' "float roughness" [{r}] "float eta" [{eta}] "float thickness" [0.0001]',
                )
                self.assertAlmostEqual(got / white / target, 1.0, delta=0.02)


if __name__ == "__main__":
    unittest.main()
