"""Tests for the opt-in measured / spectral highway materials (tools/highway_materials.py)."""

from __future__ import annotations

import filecmp
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import highway_materials as hm
import numpy as np
from synthetic_data import REPO
from test_highway_scene import build_without_assets

PBRT = Path(os.environ.get("OPENCAM_PBRT", REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"))


def _pbrt_has_fluorescent() -> bool:
    if not PBRT.is_file():
        return False
    with tempfile.TemporaryDirectory() as t:
        (Path(t) / "s.pbrt").write_text(
            'Camera "orthographic" Film "rgb" "integer xresolution" [1] "integer yresolution" [1]\nWorldBegin\n'
            'Material "fluorescent" "spectrum excitation" [360 0.5 830 0] "spectrum emission" [500 0 600 1 700 0]\n'
            'Shape "sphere"\n'
        )
        r = subprocess.run([str(PBRT), "--quiet", "s.pbrt"], cwd=t, capture_output=True)
        return r.returncode == 0


PBRT_FLUOR = _pbrt_has_fluorescent()
if os.environ.get("OPENCAM_REQUIRE_PBRT") == "1" and not PBRT_FLUOR:
    raise RuntimeError(f"OPENCAM_REQUIRE_PBRT=1 but {PBRT} lacks the fluorescent patch (tools/build_pbrt.sh)")


class DefaultOffTest(unittest.TestCase):
    def test_explicit_off_flags_are_byte_identical_to_default(self):
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t) / "a", Path(t) / "b"
            a.mkdir()
            b.mkdir()
            build_without_assets(a)
            build_without_assets(
                b, "--car-paint", "analytic", "--spectral-library", "analytic", "--fluorescent-sign", "none"
            )
            sa, sb = a / "scene", b / "scene"
            cmp = filecmp.dircmp(sa, sb, ignore=["highway_manifest.json"])
            self.assertEqual((cmp.left_only, cmp.right_only, cmp.diff_files), ([], [], []))
            for sub in cmp.common_dirs:
                c = filecmp.dircmp(sa / sub, sb / sub)
                self.assertEqual(c.diff_files, [], sub)
            text = (sa / "highway.pbrt").read_text()
            self.assertNotIn("fluorescent", text)
            self.assertNotIn('"measured"', text)
            self.assertFalse((sa / "measured").exists())

    def test_materials_object_is_noop_when_off(self):
        class A:
            pass

        m = hm.Materials(A(), Path("/nonexistent"), np.arange(360.0, 831.0, 5.0), None)
        lines = [
            'MakeNamedMaterial "c:CarPaint" "string type" "coateddiffuse"',
            '    "spectrum reflectance" "spd/carpaint_red.spd"',
        ]
        self.assertIs(m.car_body(lines, "red"), lines)
        self.assertEqual(m.sign_lines(0, 0, None, None), [])
        self.assertEqual(m.manifest(), {})


class CarBodyRewriteTest(unittest.TestCase):
    def test_rgb_chrome_and_black_trim_become_spectral(self):
        class A:
            spectral_library = "usgs"

        with tempfile.TemporaryDirectory() as t:
            written = {}
            m = hm.Materials(
                A(), Path(t), np.arange(360.0, 831.0, 5.0), lambda p, wl, v: written.__setitem__(p.name, v)
            )
            m._black_spd = lambda rgb: "spd/black_plastic_test.spd"
            out = m.car_body(
                [
                    'MakeNamedMaterial "c:Chrome" "string type" "conductor" "rgb eta" [0.2 0.2 0.2] "rgb k" [3 3 3]',
                    'MakeNamedMaterial "c:Seal" "string type" "diffuse" "rgb reflectance" [0.02 0.02 0.021]',
                    'MakeNamedMaterial "c:Logo" "string type" "diffuse" "rgb reflectance" [0.8 0.1 0.1]',
                ],
                "red",
            )
        self.assertIn('"spectrum eta" "spd/cr_eta.spd" "spectrum k" "spd/cr_k.spd"', out[0])
        self.assertNotIn("rgb", out[0])
        self.assertIn('"spectrum reflectance" "spd/black_plastic_test.spd"', out[1])
        self.assertIn('"rgb reflectance" [0.8 0.1 0.1]', out[2])  # chromatic RGB left alone


class FluorescentDyeTest(unittest.TestCase):
    def test_fitted_sheetings_meet_23cfr655_type_xi(self):
        for c, lim in hm.TYPE_XI_LIMITS.items():
            r = hm.sheeting_colour(c)
            self.assertTrue(hm.in_polygon(r["xy"], lim["xy"]), (c, r["xy"]))
            self.assertGreaterEqual(r["Y"], lim["y_min"], c)
            self.assertAlmostEqual(r["YF"], lim["yf_typical"], delta=0.5)

    def test_donaldson_matrix_conserves_energy(self):
        wl = np.arange(360.0, 831.0, 1.0)
        for c, dye in hm.FLUORESCENT_DYES.items():
            mtx = hm.donaldson_matrix(dye, wl)
            self.assertTrue(np.all(np.triu(mtx, 0) == 0), c)  # Stokes: only lex < lem
            reradiated = mtx.sum(axis=0)  # per unit excitation energy, 1 nm bins
            self.assertTrue(np.all(reradiated <= dye.absorptance(wl) + 1e-9), c)
            self.assertTrue(np.all(dye.reflectance(wl) + dye.absorptance(wl) <= 1 + 1e-12), c)


@unittest.skipUnless(PBRT_FLUOR, "pbrt with third_party/patches/0002-fluorescent-material.patch not built")
class PbrtFluorescenceTest(unittest.TestCase):
    def test_rendered_radiance_factor_matches_bispectral_analytic(self):
        import validate_measured_materials as v

        with tempfile.TemporaryDirectory() as t:
            for integ in ("path", "volpath"):
                r = v.fluorescence(Path(t), "orange", integ, nbuckets=12, spp=64)
                self.assertLess(r["max_abs_error"], 0.05 * r["peak_total_factor"], integ)


if __name__ == "__main__":
    unittest.main()
