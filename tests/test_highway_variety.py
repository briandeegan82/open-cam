"""Seeded highway scene variety (tools/highway_variety.py): reproducibility, traffic, alignment."""

from __future__ import annotations

import argparse
import json
import re
import struct
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import build_highway_scene as hw
import fetch_highway_assets as fetch
import highway_variety as hv
import numpy as np
from test_highway_scene import build_without_assets

CARS = ("car_bmw_m6", "car_pontiac_gto", "car_vintage")


def resolve(*argv: str) -> hv.Variety:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    hv.add_arguments(ap)
    return hv.resolve(ap.parse_args(list(argv)), CARS, 1)


def read_scene(out: Path) -> str:
    parts = [(out / "scene" / "highway.pbrt").read_text()]
    parts += [p.read_text() for p in sorted((out / "scene" / "cars").glob("*.pbrt"))]
    return "\n".join(parts)


class TestSeededBuild(unittest.TestCase):
    def tearDown(self) -> None:
        hv.WARP = None

    def test_default_scene_unchanged_and_seed_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            for k in "dabc":
                (Path(td) / k).mkdir()
            d0 = build_without_assets(Path(td) / "d")
            a = build_without_assets(Path(td) / "a", "--seed", "11", "--lamp-posts", "on", "--gantry", "on",
                                     "--overpass", "on", "--curve-radius", "900", "--grade", "3")  # fmt: skip
            b = build_without_assets(Path(td) / "b", "--seed", "11", "--lamp-posts", "on", "--gantry", "on",
                                     "--overpass", "on", "--curve-radius", "900", "--grade", "3")  # fmt: skip
            c = build_without_assets(Path(td) / "c", "--seed", "12")
            d_text, a_text, b_text = (read_scene(Path(td) / k) for k in "dab")
        self.assertEqual(a_text.replace(str(Path(td) / "a"), ""), b_text.replace(str(Path(td) / "b"), ""))
        self.assertEqual(a["cars"], b["cars"])
        self.assertNotEqual(a["cars"], c["cars"])
        # no seed: the fixed traffic of the reference scene, straight level road, no new structures
        self.assertEqual(len(d0["cars"]), len(hw.DEFAULT_TRAFFIC))
        self.assertEqual(d0["variety"]["seed"], None)
        self.assertIsNone(d0["variety"]["alignment"]["curve_radius_m"])
        self.assertEqual(d0["variety"]["alignment"]["grade_percent"], 0.0)
        self.assertNotIn("lamp_housing", d_text)
        self.assertNotIn("gantry_sign", d_text)
        # seeded: structures present, every vehicle paint is a written spectral basecoat
        for name in ("lamp_housing", "luminaire_lens", "gantry_sign_0"):
            self.assertIn(name, a_text)
        self.assertGreater(len(a["variety"]["lamp_posts"]["heads"]), 20)
        self.assertIsNotNone(a["variety"]["overpass_distance_m"])
        paints = set(re.findall(r'"spd/carpaint_([a-z]+_\d\d)\.spd"', a_text))
        self.assertEqual(paints, {car["paint"] for car in a["cars"]})
        # absolute radiometry is unaffected by the scene content
        for m in (a, c):
            self.assertEqual(m["lighting"], d0["lighting"])
        self.assertEqual(a_text.count("float illuminance"), 2)

    def test_seeded_paint_spds_written(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m = build_without_assets(Path(td), "--seed", "3")
            spd = Path(td) / "scene" / "spd"
            for car in m["cars"]:
                vals = np.loadtxt(spd / f"carpaint_{car['paint']}.spd")
                self.assertTrue(np.all((vals[:, 1] > 0) & (vals[:, 1] < 1)))


class TestTraffic(unittest.TestCase):
    def test_both_carriageways_heavy_vehicles_no_overlap(self) -> None:
        lanes, models = Counter(), Counter()
        for seed in range(40):
            v = resolve("--seed", str(seed))
            self.assertEqual(v.traffic, resolve("--seed", str(seed)).traffic)
            by_lane: dict[int, list] = {}
            for lane, s, model, paint, _ in v.traffic:
                lanes[lane] += 1
                models[model] += 1
                self.assertIn(paint, v.paints)
                length = hv.HEAVY[model]["length_m"] if model in hv.HEAVY else 4.7
                by_lane.setdefault(lane, []).append((s - length / 2, s + length / 2, model))
            for iv in by_lane.values():
                iv.sort()
                for (_, e0, m0), (s1, _, m1) in zip(iv, iv[1:]):
                    coupled = {m0, m1} == {"truck_daf_cf_tractor", "trailer_box"}
                    self.assertGreater(s1 - e0, -1.81 if coupled else 9.99)
        self.assertEqual(set(lanes), {0, 1, 2, -1, -2, -3})
        for m in (*CARS, *hv.HEAVY):
            self.assertGreater(models[m], 0, m)
        self.assertEqual(models["truck_daf_cf_tractor"], models["trailer_box"])

    def test_paint_distribution(self) -> None:
        self.assertAlmostEqual(sum(hv.PAINT_SHARES.values()), 1.0)
        rng = np.random.default_rng(0)
        n = 20000
        got = Counter(hv._pick(rng, hv.PAINT_SHARES) for _ in range(n))
        for colour, share in hv.PAINT_SHARES.items():
            self.assertAlmostEqual(got[colour] / n, share, delta=0.012)
        neutral = sum(got[c] for c in ("white", "black", "silver", "gray")) / n
        self.assertGreater(neutral, 0.7)
        wl = np.arange(360.0, 831.0, 5.0)
        for colour in hv.EXTRA_PAINTS:
            r = hv.paint_reflectance(colour, wl, None)
            self.assertTrue(np.all((r > 0) & (r < 1)))


class TestAlignment(unittest.TestCase):
    def test_straight_is_identity(self) -> None:
        a = hv.Alignment()
        self.assertTrue(a.straight)
        p = np.array([[3.0, 1.0, 50.0], [-2000.0, 0.0, 9000.0]])
        np.testing.assert_allclose(a.warp(p), p, atol=1e-9)

    def test_curve_and_grade(self) -> None:
        a = hv.Alignment(radius_m=800.0, clothoid_m=120.0, curve_start_m=100.0, grade=0.03, grade_start_m=50.0)
        np.testing.assert_allclose(a.warp(np.array([[1.5, 1.35, 0.0]])), [[1.5, 1.35, 0.0]], atol=1e-9)
        th, _, _, e, g = a.frame(np.array([0.0, 160.0, 220.0, 400.0, 1000.0, 3000.0]))
        k = np.interp([160.0, 220.0, 400.0], a._s, a._k)
        np.testing.assert_allclose(k, [0.5 / 800.0, 1 / 800.0, 1 / 800.0], rtol=1e-3)  # clothoid ramp, arc
        self.assertAlmostEqual(float(np.degrees(th[-1])), 30.0, places=2)  # total turn, back to tangent
        self.assertAlmostEqual(float(g[4]), 0.03)
        self.assertAlmostEqual(float(g[-1]), 0.0)
        self.assertGreater(float(e[-1]), 30.0)
        pos, heading, pitch = a.place(0.0, 0.0, 1000.0)
        self.assertAlmostEqual(heading, 30.0, places=2)
        self.assertAlmostEqual(math_tan(pitch), 0.03, places=6)
        # no folding anywhere on the terrain (Jacobian of the plan-view map stays positive)
        xs = np.concatenate([-np.geomspace(1, 3000, 60), np.geomspace(1, 3000, 60)])
        zs = np.linspace(-500, 12000, 400)
        X, Z = np.meshgrid(xs, zs)
        P = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], -1)
        h = 0.01
        W = a.warp(P)
        dx = (a.warp(P + [h, 0, 0]) - W)[:, [0, 2]] / h
        dz = (a.warp(P + [0, 0, h]) - W)[:, [0, 2]] / h
        det = dx[:, 0] * dz[:, 1] - dx[:, 1] * dz[:, 0]
        self.assertTrue(np.all(det < 0) or np.all(det > 0))

    def test_refine_along_s_conforming(self) -> None:
        p = np.array([[0, 0, 0], [3, 0, 0], [3, 0, 400], [0, 0, 400]], float)
        tri = np.array([[0, 2, 1], [0, 3, 2]])
        p2, t2, uv2 = hv.refine_along_s(p, tri, p[:, [0, 2]])

        def area(pp, tt):
            return np.cross(pp[tt[:, 1]] - pp[tt[:, 0]], pp[tt[:, 2]] - pp[tt[:, 0]])

        np.testing.assert_allclose(area(p2, t2).sum(0), area(p, tri).sum(0))
        dz = np.abs(np.diff(p2[t2][:, [0, 1, 2, 0], 2], axis=1))
        self.assertLessEqual(dz.max(), 6.0 + 1e-9)
        edges = Counter(tuple(sorted(e)) for t in t2 for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])))
        rim = [e for e, c in edges.items() if c == 1]  # a T-junction would leave an open edge inside
        on_rim = np.isin(p2[:, 0], (0.0, 3.0)) | np.isin(p2[:, 2], (0.0, 400.0))
        self.assertTrue(all(on_rim[i] and on_rim[j] for i, j in rim))
        self.assertAlmostEqual(sum(float(np.linalg.norm(p2[i] - p2[j])) for i, j in rim), 806.0)
        self.assertEqual(uv2.shape, (len(p2), 2))

    def test_flags_override_seed(self) -> None:
        v = resolve("--seed", "4", "--curve-radius", "0", "--grade", "0", "--gantry", "off", "--overpass", "on")
        self.assertTrue(v.alignment.straight)
        self.assertFalse(v.gantry)
        self.assertTrue(v.overpass)
        self.assertFalse(resolve().lamp_posts)
        kinds = Counter(k for _, k in resolve("--seed", "9").species)
        self.assertGreaterEqual(kinds["tree"], 1)
        self.assertGreaterEqual(kinds["shrub"], 1)


def math_tan(deg: float) -> float:
    return float(np.tan(np.radians(deg)))


class TestGlb(unittest.TestCase):
    def test_unpack_and_prepare_rejects_corrupt_vertices(self) -> None:
        pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [3e38, 1, 1]], np.float32)
        idx = np.array([0, 1, 2, 0, 2, 3, 2, 3, 4], np.uint32)
        png = bytes.fromhex(
            "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
            "1f15c4890000000d49444154789c6360f8cf000000030101005d0c4b6b0000000049454e44ae426082"
        )
        binary = pos.tobytes() + idx.tobytes() + png
        g = {
            "asset": {"version": "2.0"},
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
            "materials": [{"name": "Body", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
            "textures": [{"source": 0}],
            "images": [{"bufferView": 2, "mimeType": "image/png"}],
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5126,
                    "count": 5,
                    "type": "VEC3",
                    "min": [0, 0, 0],
                    "max": [1, 1, 1],
                },
                {"bufferView": 1, "componentType": 5125, "count": 9, "type": "SCALAR"},
            ],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": 60},
                {"buffer": 0, "byteOffset": 60, "byteLength": 36},
                {"buffer": 0, "byteOffset": 96, "byteLength": len(png)},
            ],
            "buffers": [{"byteLength": len(binary)}],
        }
        js = json.dumps(g).encode()
        js += b" " * (-len(js) % 4)
        binary += b"\0" * (-len(binary) % 4)
        glb = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(binary))
        glb += struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(binary), 0x004E4942) + binary
        with tempfile.TemporaryDirectory() as td:
            raw, out = Path(td) / "raw", Path(td) / "out"
            raw.mkdir()
            (raw / "m.glb").write_bytes(glb)
            gltf = fetch.unpack_glb(raw / "m.glb", raw / "unpacked")
            self.assertTrue((raw / "unpacked" / "image_000.png").read_bytes() == png)
            out.mkdir()
            info = fetch.prepare_gltf({"gltf": str(gltf.relative_to(raw))}, raw, out)
            self.assertEqual(info["shapes"][0]["triangles"], 2)
            np.testing.assert_allclose(info["bbox_max"], [1, 1, 1])
            self.assertTrue(np.isfinite(fetch.read_ply_positions(out / info["shapes"][0]["ply"])).all())

    def test_read_ply_binary_starting_with_newline_byte(self) -> None:
        x0 = struct.unpack("<f", b"\x0a\x0d\x80\x3f")[0]
        pos = np.array([[x0, 2, 3], [4, 5, 6], [7, 8, 9]], np.float64)
        with tempfile.TemporaryDirectory() as td:
            fetch.write_ply(Path(td) / "a.ply", pos, None, None, np.array([[0, 1, 2]]))
            np.testing.assert_allclose(fetch.read_ply_positions(Path(td) / "a.ply"), pos, rtol=1e-6)

    def test_proxy_heavy_vehicle(self) -> None:
        lines = hv.proxy_vehicle("bus_town", "spd/carpaint_white.spd", "car00", hw.mesh, hw.box)
        self.assertIn("coateddiffuse", lines[0])
        self.assertIsNone(hv.proxy_vehicle("car_bmw_m6", "x", "car00", hw.mesh, hw.box))


if __name__ == "__main__":
    unittest.main()
