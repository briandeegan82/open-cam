from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import build_highway_scene as hw
import highway_road_wear as rw
import numpy as np
from highway_spectra import reflectance
from PIL import Image
from test_highway_scene import build_without_assets


def _geom() -> dict:
    L = hw.Layout
    lanes = [[(L.lane_center(s * i - (s < 0)), 0.6 + 0.2 * i) for i in range(hw.N_LANES)] for s in (1, -1)]
    return {
        "x0": L.opp_paved,
        "x1": L.right_paved,
        "z0": hw.ROAD_Z0,
        "z1": hw.ROAD_Z1,
        "median_x": L.median_l + hw.MEDIAN_W / 2,
        "period": 8 * (hw.DASH + hw.GAP),
        "carriageways": [(L.median_l + hw.MEDIAN_W, L.right_paved, lanes[0]), (L.opp_paved, L.median_l, lanes[1])],
        "lane_lines": [hw.LANE_W, 2 * hw.LANE_W, L.opp_inner - hw.LANE_W, L.opp_inner - 2 * hw.LANE_W],
    }


def _maps(level: str = "moderate", seed: int = 7, texel: float = 0.05) -> tuple[rw.RoadWear, dict]:
    w = rw.RoadWear(level, seed, Path(tempfile.gettempdir()), _geom(), texel)
    return w, w.wear_maps()


class TestNoiseAndDetile(unittest.TestCase):
    def test_periodic_noise(self) -> None:
        n = rw.periodic_noise(np.random.default_rng(0), (400, 120), (0.05, 0.05), (2.0, 0.5))
        self.assertEqual(n.shape, (400, 120))
        self.assertAlmostEqual(float(n.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(n.std()), 1.0, places=4)
        # periodic: first and last rows/cols are neighbours
        self.assertGreater(np.corrcoef(n[0], n[-1])[0, 1], 0.9)
        self.assertGreater(np.corrcoef(n[:, 0], n[:, -1])[0, 1], 0.5)

    def test_detile_preserves_statistics(self) -> None:
        rng = np.random.default_rng(1)
        lum = 1.0 + 0.3 * rw.periodic_noise(rng, (64, 64), (1, 1), (2, 2))
        rough = 0.5 + 0.1 * rw.periodic_noise(rng, (64, 64), (1, 1), (2, 2))
        nx, ny = 0.2 * rw.periodic_noise(rng, (64, 64), (1, 1), (2, 2)), np.zeros((64, 64))
        out = rw.detile([lum, rough, nx, ny], 192, 4, np.random.default_rng(2))
        self.assertEqual(out[0].shape, (192, 192))
        self.assertAlmostEqual(float(out[0].mean()), float(lum.mean()), delta=0.05)
        self.assertAlmostEqual(float(out[0].std()), float(lum.std()), delta=0.3 * float(lum.std()))
        # rotated copies move x-tilt into y, and the result is not a plain repeat of the source
        self.assertGreater(float(np.abs(out[3]).mean()), 0.01)
        self.assertFalse(np.allclose(out[0][:64, :64], lum))

    def test_linear_normal_map_png(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "n.jpg"
            Image.fromarray(np.full((8, 8, 3), [128, 128, 255], np.uint8)).save(src, quality=100)
            dst = rw.linear_normal_map(src, Path(td) / "out")
            self.assertEqual(dst.suffix, ".png")
            self.assertEqual(np.asarray(Image.open(dst)).shape, (8, 8, 3))
            self.assertEqual(rw.linear_normal_map(dst, Path(td)), dst)


class TestWearMaps(unittest.TestCase):
    def test_seeded_and_bounded(self) -> None:
        _, a = _maps(seed=3)
        _, b = _maps(seed=3)
        _, c = _maps(seed=4)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
            self.assertTrue(np.isfinite(a[k]).all())
            self.assertGreaterEqual(float(a[k].min()), 0.0)
        self.assertFalse(np.array_equal(a["mul"], c["mul"]))
        for k in ("fresh", "gloss", "seal", "blend"):
            self.assertLessEqual(float(a[k].max()), 1.0)

    def test_wheel_paths_are_polished(self) -> None:
        w, m = _maps(level="heavy")
        g = w.geom
        xs = g["x0"] + (np.arange(m["gloss"].shape[1]) + 0.5) * (g["x1"] - g["x0"]) / m["gloss"].shape[1]
        col = lambda x: int(np.argmin(np.abs(xs - x)))  # noqa: E731
        xc = hw.Layout.lane_center(2)
        path = m["gloss"][:, col(xc + rw.WHEEL_OFFSET_M)].mean()
        between = m["gloss"][:, col(xc + 1.6)].mean()
        self.assertGreater(path, 2.0 * between)
        self.assertLess(m["mul"][:, col(xc + rw.WHEEL_OFFSET_M)].mean(), m["mul"][:, col(xc + 1.6)].mean())

    def test_luminous_reflectance_in_measured_range(self) -> None:
        wl = np.arange(360.0, 831.0, 5.0)
        aged = reflectance("asphalt_aged", wl)
        prev = None
        for level in ("light", "moderate", "heavy"):
            w, m = _maps(level=level)
            st = w.reflectance_stats(m, wl, aged)
            self.assertGreater(st["p05"], 0.05)  # never darker than new asphalt on average
            self.assertGreater(st["mean"], 0.10)
            self.assertLess(st["p95"], 0.18)
            if prev is not None:
                self.assertLessEqual(st["mean"], prev + 1e-3)
            prev = st["mean"]
        self.assertAlmostEqual(rw.luminous_reflectance(wl, np.full_like(wl, 0.3)), 0.3)

    def test_marking_wear_level(self) -> None:
        w = rw.RoadWear("moderate", 1, Path(tempfile.gettempdir()), _geom())
        m = w._marking_wear(0.25, 16 * (hw.DASH + hw.GAP))
        self.assertAlmostEqual(float(m.mean()), 0.25, delta=0.08)
        self.assertLessEqual(float(m.max()), 0.97 + 1e-6)
        self.assertGreater(float(m[:, [0, -1]].mean()), float(m[:, 6:10].mean()))  # edges chip first


class TestBuilderHook(unittest.TestCase):
    def test_default_scene_has_wear(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m = build_without_assets(Path(td), "--road-wear-texel-m", "0.1")
            text = (Path(td) / "scene" / "highway.pbrt").read_text()
            tex = sorted(p.name for p in (Path(td) / "scene" / "textures" / "road_wear").iterdir())
        self.assertIn('MakeNamedMaterial "asphalt" "string type" "mix"', text)
        self.assertIn('"string mapping" "uv"', text)
        self.assertIn('NamedMaterial "rw:paint_dash"', text)
        self.assertNotIn('NamedMaterial "paint_white"\n', text)
        wear = m["road"]["wear"]
        self.assertEqual((wear["level"], wear["seed"]), ("moderate", 7))
        n_rpm = text.count('ObjectInstance "rw:rpm_')
        self.assertEqual(n_rpm, wear["raised_pavement_markers"]["placed"])
        self.assertGreater(n_rpm, 200)
        self.assertIn("wear_moderate_s7_mul.exr", tex)
        self.assertEqual(wear["anti_tiling"]["assets"], [])  # offline: no texture assets
        self.assertGreater(m["lighting"]["reference_illuminance_lux"], 50_000)  # lighting untouched

    def test_wear_none_and_seed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m = build_without_assets(Path(td), "--road-wear", "none")
            text = (Path(td) / "scene" / "highway.pbrt").read_text()
        self.assertIsNone(m["road"]["wear"])
        self.assertNotIn("rw:", text)
        self.assertIn('NamedMaterial "paint_white"', text)
        with tempfile.TemporaryDirectory() as td:
            m = build_without_assets(
                Path(td), "--road-wear", "heavy", "--road-wear-seed", "11", "--road-wear-texel-m", "0.1"
            )
        self.assertEqual((m["road"]["wear"]["level"], m["road"]["wear"]["seed"]), ("heavy", 11))

    def test_wear_follows_curved_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m = build_without_assets(Path(td), "--curve-radius", "800", "--road-wear-texel-m", "0.1")
            text = (Path(td) / "scene" / "highway.pbrt").read_text()
        self.assertTrue(m["road"]["wear"]["raised_pavement_markers"]["placed"] > 200)
        rpm = [ln for ln in text.splitlines() if 'ObjectInstance "rw:rpm_' in ln]
        self.assertTrue(all("Rotate" in ln for ln in rpm))  # placed via the alignment hook
        far = [ln for ln in rpm if float(ln.split()[4]) > 800.0]
        self.assertTrue(far and all(abs(float(ln.split()[2])) > 50.0 for ln in far))  # bent away from x ~ 0


if __name__ == "__main__":
    unittest.main()
