"""Objaverse passenger cars (highway_variety.CARS_GLB): manifest provenance, car pool, glb rear lamps."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import highway_taillamps as tl
import highway_variety as hv
import numpy as np
import yaml
from fetch_highway_assets import write_ply

REPO = Path(__file__).resolve().parents[1]
CLASSIC = ("car_bmw_m6", "car_pontiac_gto", "car_vintage")


def resolve(*argv: str) -> hv.Variety:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    hv.add_arguments(ap)
    return hv.resolve(ap.parse_args(list(argv)), CLASSIC, 1)


def _box(x0, x1, y0, y1, z0, z1) -> np.ndarray:
    return tl._box_tris(x0, x1, y0, y1, z0, z1)


class TestManifest(unittest.TestCase):
    def test_every_glb_car_is_pinned_cc_by_with_attribution(self) -> None:
        assets = yaml.safe_load((REPO / "config" / "highway_assets.yaml").read_text())["assets"]
        for m in hv.CARS_GLB:
            a = assets[m]
            self.assertEqual(a["kind"], "glb", m)
            self.assertEqual(a["license"], "CC-BY-4.0", m)
            uid = a["glb"].removesuffix(".glb")
            self.assertEqual(a["source"], f"https://sketchfab.com/3d-models/{uid}")
            self.assertIn(a["author"].removesuffix(" (Sketchfab)"), a["license_note"])
            self.assertIn("CC BY 4.0", a["license_note"])
            (f,) = a["files"]
            self.assertRegex(
                f["url"], r"^https://huggingface\.co/datasets/allenai/objaverse/resolve/[0-9a-f]{40}/glbs/"
            )
            self.assertTrue(f["url"].endswith(a["glb"]))
            self.assertRegex(f["sha256"], r"^[0-9a-f]{64}$")

    def test_specs(self) -> None:
        for m, s in hv.CARS_GLB.items():
            self.assertIn(s["forward"], hv._YAW, m)
            self.assertTrue(3.5 < s["length_m"] < 6.0, m)
            self.assertIn(m, tl.LAYOUTS)
            self.assertNotIn(m, hv.HEAVY)
            self.assertIs(hv.glb_spec(m), s)


class TestCarPool(unittest.TestCase):
    def test_classic_by_default_extended_adds_glb_cars(self) -> None:
        seen_c, seen_e = set(), set()
        for seed in range(30):
            seen_c |= {t[2] for t in resolve("--seed", str(seed)).traffic}
            seen_e |= {t[2] for t in resolve("--seed", str(seed), "--car-pool", "extended").traffic}
        self.assertFalse(seen_c & set(hv.CARS_GLB))
        self.assertGreaterEqual(len(seen_e & set(hv.CARS_GLB)), len(hv.CARS_GLB) - 3)
        self.assertTrue(seen_e & set(CLASSIC))

    def test_extended_pool_uses_per_model_length_for_spacing(self) -> None:
        for seed in range(20):
            by_lane: dict[int, list] = {}
            for lane, s, model, _, _ in resolve("--seed", str(seed), "--car-pool", "extended").traffic:
                ln = (hv.glb_spec(model) or {"length_m": 4.7})["length_m"]
                by_lane.setdefault(lane, []).append((s - ln / 2, s + ln / 2, model))
            for iv in by_lane.values():
                iv.sort()
                for (_, e0, m0), (s1, _, m1) in zip(iv, iv[1:]):
                    if {m0, m1} != {"truck_daf_cf_tractor", "trailer_box"}:
                        self.assertGreater(s1 - e0, 9.99)


class TestGlass(unittest.TestCase):
    def test_names_and_alpha(self) -> None:
        for n in ("Glass_MAT", "CAR_Window", "WindShield", "vitre", "glass_window"):
            self.assertTrue(hv.is_glass_material({"name": n}), n)
        self.assertTrue(hv.is_glass_material({"name": "lens", "alpha_mode": "BLEND"}))
        self.assertFalse(hv.is_glass_material({"name": "Car_Paint"}))


class TestGlbRearLamps(unittest.TestCase):
    """Synthetic glb car in centimetres, front along -x: body, a red lamp behind a clear outer lens, rear window."""

    def _prepared(self, root: Path) -> None:
        prep = root / "prepared" / "car_test_glb"
        parts = [
            ("body", _box(-225, 225, 0, 140, -90, 90)),
            ("tail_red", _box(225, 226, 80, 100, 55, 85)),
            ("tail_red", _box(225, 226, 80, 100, -85, -55)),
            ("lamp_glass", _box(227, 228, 78, 102, 53, 87)),
            ("lamp_glass", _box(227, 228, 78, 102, -87, -53)),
            ("rear_window", _box(225, 228, 105, 135, -70, 70)),
        ]
        names = ["body", "tail_red", "lamp_glass", "rear_window"]
        shapes = []
        for k, (mat, tri) in enumerate(parts):
            pos = tri.reshape(-1, 3)
            write_ply(prep / f"m{k}.ply", pos.astype(np.float32), None, None, np.arange(len(pos)).reshape(-1, 3))
            shapes.append({"ply": f"m{k}.ply", "material": names.index(mat)})
        info = {"asset": {"kind": "glb"}, "raw_dir": ".", "shapes": shapes,
                "materials": [{"name": n, "base_color_factor": [0.5, 0.5, 0.5, 1]} for n in names]}  # fmt: skip
        (prep / "prepared.json").write_text(json.dumps(info))

    def test_lamp_on_red_lens_behind_clear_cover(self) -> None:
        spec = {"cls": "car", "forward": "-x", "length_m": 4.5, "paint": ["body"], "lens": ["tail_red"]}
        layout = tl.LampLayout("led", (0.60, 0.0, 0.45, 1.40), ("tail_red",), from_edge=True)
        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch.dict(hv.CARS_GLB, {"car_test_glb": spec}),
            mock.patch.dict(tl.LAYOUTS, {"car_test_glb": layout}),
        ):
            self._prepared(Path(td))
            surf = tl.rear_surface("car_test_glb", td)
            self.assertNotIn(surf.materials.index("lamp_glass"), set(surf.mat.tolist()))
            self.assertNotIn(surf.materials.index("rear_window"), set(surf.mat.tolist()))
            s = 4.5 / 453.0
            self.assertAlmostEqual(float(surf.tri[..., 2].min()), -2.25, delta=0.03)
            car = {"id": "c", "model": "car_test_glb", "length_m": 4.5, "width_m": 1.8, "height_m": 1.4}
            lines, meta = tl.rear_lamp_lines(car, False, td)
            self.assertEqual([sd["surface"] for sd in meta["sides"]], ["mesh", "mesh"])
            for sd in meta["sides"]:
                self.assertAlmostEqual(sd["lens_area_m2"], 0.30 * 0.20 * (100 * s) ** 2, delta=0.02)
            w, h = hv.glb_size("car_test_glb", json.loads((Path(td) / "prepared/car_test_glb/prepared.json").read_text()),
                                Path(td) / "prepared/car_test_glb")[1:]  # fmt: skip
            self.assertAlmostEqual(w, 1.80 * 100 * s, delta=0.05)
            self.assertAlmostEqual(h, 1.40 * 100 * s, delta=0.05)


if __name__ == "__main__":
    unittest.main()
