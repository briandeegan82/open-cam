"""SWIR (> 830 nm) highway spectra: tools/highway_swir.py and its build_highway_scene hooks."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import build_highway_scene as hw  # noqa: E402
import highway_spectral_sky as hss  # noqa: E402
import highway_swir as swir  # noqa: E402
from highway_sky import solar_direct_spectrum  # noqa: E402
from highway_spectra import reflectance  # noqa: E402


def build_offline(out: Path, *extra: str) -> tuple[dict, str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    m = yaml.safe_load((REPO / "config" / "highway_assets.yaml").read_text())
    m["cache_dir"] = str(out / "empty_cache")
    (out / "assets.yaml").write_text(yaml.safe_dump(m))
    hw.main(
        ["--out-dir", str(out / "scene"), "--asset-manifest", str(out / "assets.yaml"), "--allow-missing-assets"]
        + ["--xres", "48", "--yres", "32", "--pixelsamples", "4", *extra]
    )
    scene = out / "scene"
    return json.loads((scene / "highway_manifest.json").read_text()), (scene / "highway.pbrt").read_text(), scene


def read_spd(p: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.loadtxt(p)
    return d[:, 0], d[:, 1]


WL = np.arange(360.0, 1600.0 + 1e-9, 5.0)


class TestSpectra(unittest.TestCase):
    def test_enabled(self) -> None:
        ns = argparse.Namespace
        self.assertFalse(swir.enabled(ns(swir_spectra="auto", spectral_lambda_max=830.0)))
        self.assertTrue(swir.enabled(ns(swir_spectra="auto", spectral_lambda_max=1600.0)))
        self.assertFalse(swir.enabled(ns(swir_spectra="off", spectral_lambda_max=1600.0)))
        self.assertTrue(swir.enabled(ns(swir_spectra="on", spectral_lambda_max=830.0)))

    def test_sun_keeps_visible_and_has_water_bands(self) -> None:
        vis = solar_direct_spectrum(WL, 45.0)
        s = swir.sun_spectrum(WL, vis)
        lo = WL <= swir.JOIN_NM
        np.testing.assert_array_equal(s[lo], vis[lo])
        at = lambda nm: float(np.interp(nm, WL, s))  # noqa: E731
        self.assertAlmostEqual(at(835.0) / at(830.0), 1.0, delta=0.05)  # continuous join
        self.assertLess(at(1380.0), 0.05 * at(1250.0))  # 1.38 um water-vapour band
        self.assertLess(at(1130.0), 0.6 * at(1050.0))  # 1.13 um band

    def test_vegetation_water_absorption(self) -> None:
        for name in swir.VEGETATION:
            vis = reflectance(name, WL)
            r = swir.vegetation(name, WL, vis)
            lo = WL <= swir.LEAF_JOIN_NM
            np.testing.assert_array_equal(r[lo], vis[lo])
            self.assertLess(np.interp(1450.0, WL, r), 0.8 * np.interp(1250.0, WL, r), name)
            self.assertTrue(np.all((r >= 0) & (r <= 1)))

    def test_sky_basis_tail(self) -> None:
        wl, b = hss.basis_spectra()
        w2, b2 = swir.extend_sky_basis(wl, b, 1600.0)
        self.assertEqual(w2[-1], 1600.0)
        np.testing.assert_array_equal(b2[:, : wl.size], b)
        self.assertTrue(np.all(b2 >= 0))
        ratio = np.interp([900.0, 1550.0], w2, b2[0]) / b2[0, wl.size - 1]
        self.assertTrue(np.all(ratio < 1.0) and np.all(ratio > 0.0))  # dark but nonzero SWIR sky


class TestBuilder(unittest.TestCase):
    def test_swir_band_scene(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m, scene, d = build_offline(
                Path(td), "--sky", "analytic", "--spectral-lambda-max", "1600", "--film-lambda-min", "900"
            )
            self.assertIn('"float lambdamin" [900]', scene)
            self.assertIn('"float lambdamax" [1600]', scene)
            self.assertEqual(m["film"]["lambdamin"], 900.0)
            self.assertEqual(m["film"]["spectra_lambda_range_nm"], [360.0, 1600.0])
            self.assertIn("swir", m)
            self.assertIn("sky_basis_0.spd", scene)  # RGB sky switched to the daylight basis
            w, v = read_spd(d / "spd" / "sky_basis_0.spd")
            self.assertEqual(w[-1], 1600.0)
            self.assertGreater(v[-1], 0.0)
            w, sun = read_spd(d / "spd" / "sun.spd")
            self.assertEqual(w[0], 360.0)  # visible kept for the photometric normalisation
            _, _, d0 = build_offline(Path(td) / "vis", "--sky", "analytic")
            w0, sun0 = read_spd(d0 / "spd" / "sun.spd")
            np.testing.assert_allclose(sun[: w0.size], sun0, rtol=1e-5)

    def test_swir_surfaces_lamps_and_gas(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m, _, d = build_offline(
                Path(td),
                *("--sky", "analytic", "--spectral-lambda-max", "1600", "--film-lambda-min", "900"),
                *("--haze", "hazy", "--time-of-day", "night", "--streetlights", "hps"),
            )
            w, white = read_spd(d / "spd" / "paint_road_white.spd")
            vis = w <= swir.JOIN_NM
            np.testing.assert_allclose(white[vis], reflectance("paint_road_white", w[vis]), rtol=1e-5)
            # C-H overtone dip of the TiO2-pigmented polymer proxy (white vinyl) below its 1.25 um value
            self.assertLess(np.interp(1730.0 if w[-1] > 1730 else 1400.0, w, white), np.interp(1250.0, w, white))
            self.assertIn("surfaces", m["swir"])
            w, hps = read_spd(d / "spd" / "lamp_hps.spd")
            self.assertEqual(w[-1], 1600.0)
            self.assertGreater(np.interp(1138.0, w, hps), 10 * np.interp(1250.0, w, hps))
            w, sa = read_spd(d / "spd" / "haze_sigma_a.spd")
            self.assertGreater(np.interp(1380.0, w, sa), 20 * np.interp(1250.0, w, sa))
            self.assertEqual(m["swir"]["path_gas_absorption"]["h2o_gm3"], swir.H2O_DEFAULT_GM3)

    def test_gas_compensation_on_sun(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m, _, d = build_offline(
                Path(td),
                "--sky",
                "analytic",
                "--spectral-lambda-max",
                "1600",
                "--film-lambda-min",
                "900",
                "--spectral-nbuckets",
                "36",
                "--haze",
                "hazy",
            )
            comp = m["atmosphere"]["swir_gas_compensation"]
            w, sun = read_spd(d / "spd" / "sun.spd")
            sun0 = swir.sun_spectrum(w, hw.solar_direct_spectrum(w, m["lighting"]["sun"]["elevation_deg"]))
            np.testing.assert_allclose(sun[w <= 830.0], sun0[w <= 830.0], rtol=1e-4)
            k = np.exp(swir.gas_sigma_a(1380.0, swir.H2O_DEFAULT_GM3) * comp["layer_column_m"] * comp["sun_airmass"])
            self.assertAlmostEqual(np.interp(1380.0, w, sun) / np.interp(1380.0, w, sun0), k, delta=0.02 * k)

    def test_gas_sigma_a(self) -> None:
        wl = np.array([550.0, 820.0, 1250.0, 1380.0, 1870.0])
        sa = swir.gas_sigma_a(wl, 10.0)
        self.assertEqual(sa[0], 0.0)
        self.assertEqual(sa[1], 0.0)  # visible / NIR below the join untouched
        t100 = np.exp(-100.0 * sa)
        self.assertGreater(t100[2], 0.99)  # 1.25 um window
        self.assertLess(t100[3], 0.9)  # 1.38 um band over 100 m of humid air
        self.assertLess(t100[4], t100[3] + 0.1)
        np.testing.assert_allclose(swir.gas_sigma_a(wl, 20.0), 2 * sa)

    def test_skin_water_absorption_continues_past_1100_nm(self) -> None:
        import nir_skin

        mua = nir_skin.water_mua(np.array([1090.0, 1100.0, 1110.0, 1450.0]))
        self.assertAlmostEqual(mua[2] / mua[1], 1.0, delta=0.5)  # Segelstein meets Hale & Querry
        self.assertGreater(mua[3], 20.0)  # 1.45 um band, ~30 /cm
        r = nir_skin.skin_reflectance(np.array([1100.0, 1450.0]))
        self.assertLess(r[1], 0.3 * r[0])

    def test_default_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m, scene, _ = build_offline(Path(td), "--sky", "analytic")
            self.assertNotIn("swir", m)
            self.assertIn('"float lambdamin" [360]', scene)
            self.assertNotIn("sky_basis", scene)

    def test_film_band_outside_grid(self) -> None:
        with tempfile.TemporaryDirectory() as td, self.assertRaises(SystemExit):
            build_offline(Path(td), "--sky", "analytic", "--film-lambda-max", "1000")


if __name__ == "__main__":
    unittest.main()
