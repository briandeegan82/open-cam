"""Traffic-sign scene: sign asset manifest, SVG preparation, materials and scene integration (no pbrt)."""

from __future__ import annotations

import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path

import build_highway_scene as hw
import fetch_highway_assets as fetch
import highway_signs as signs
import numpy as np
import yaml
from PIL import Image
from synthetic_data import REPO

HAVE_CAIRO = importlib.util.find_spec("cairosvg") is not None
SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
    '<circle cx="50" cy="50" r="49" fill="#c1121c"/><circle cx="50" cy="50" r="38" fill="#ffffff"/>'
    '<rect x="30" y="40" width="40" height="20" fill="#000000"/></svg>'
)


def fake_prepared(root: Path, aid: str, a: dict) -> None:
    """A prepared sign (disc: red rim, white face, black bar) without cairosvg."""
    out = root / "prepared" / aid
    out.mkdir(parents=True)
    yy, xx = np.mgrid[:64, :64] - 31.5
    r = np.hypot(xx, yy)
    disc, face = r < 31, r < 24
    bar = face & (abs(yy) < 5) & (abs(xx) < 15)
    m = {"red": disc & ~face, "white": face & ~bar, "black": bar}
    Image.fromarray((disc * 255).astype(np.uint8)).save(out / "alpha.png")
    for c, v in m.items():
        Image.fromarray((v * 255).astype(np.uint8)).save(out / f"mask_{c}.png")
    info = {
        "sign_set": a["sign_set"],
        "code": a["code"],
        "width_m": a["width_m"],
        "height_m": a["width_m"],
        "png_size": [64, 64],
        "alpha": "alpha.png",
        "masks": {c: f"mask_{c}.png" for c in m},
        "colour_fraction": {c: round(float(v.sum() / disc.sum()), 4) for c, v in m.items()},
        "version": fetch.PREPARED_VERSION,
        "asset": a,
    }
    (out / "prepared.json").write_text(json.dumps(info))


def build(td: Path, *extra: str, sign_set: str = "stvo") -> tuple[str, dict]:
    m = yaml.safe_load((REPO / "config" / "highway_assets.yaml").read_text())
    m["cache_dir"] = str(td / "cache")
    for aid in signs.set_assets(m, sign_set):
        fake_prepared(td / "cache", aid, m["assets"][aid])
    am = td / "assets.yaml"
    am.write_text(yaml.safe_dump(m))
    out = td / "scene"
    hw.main(
        ["--out-dir", str(out), "--asset-manifest", str(am), "--allow-missing-assets"]
        + ["--xres", "48", "--yres", "32", "--pixelsamples", "4", "--sign-set", sign_set, *extra]
    )
    return (out / "highway.pbrt").read_text(), json.loads((out / "highway_manifest.json").read_text())


class TestSignManifest(unittest.TestCase):
    def test_sign_assets_pinned_and_public_domain(self):
        m = fetch.load_asset_manifest(REPO / "config" / "highway_assets.yaml")
        ids = [k for k, a in m["assets"].items() if a["kind"] == "sign_svg"]
        self.assertGreaterEqual(len(signs.set_assets(m, "stvo")), 6)
        self.assertGreaterEqual(len(signs.set_assets(m, "mutcd")), 4)
        for aid in ids:
            a = m["assets"][aid]
            self.assertEqual(a["license"], "public-domain")
            self.assertTrue(a["source"].startswith("https://commons.wikimedia.org/wiki/File:"))
            self.assertTrue(0.3 < a["width_m"] < 1.5)
            self.assertEqual([f["path"] for f in a["files"]], [a["svg"]])
            self.assertRegex(a["files"][0]["sha256"], r"^[0-9a-f]{64}$")


@unittest.skipUnless(HAVE_CAIRO, "cairosvg not installed")
class TestPrepareSign(unittest.TestCase):
    def test_masks_partition_the_cutout(self):
        with tempfile.TemporaryDirectory() as td:
            raw, out = Path(td) / "raw", Path(td) / "out"
            raw.mkdir(), out.mkdir()
            (raw / "s.svg").write_text(SVG)
            a = {"svg": "s.svg", "sign_set": "stvo", "code": "x", "width_m": 0.6}
            info = fetch.prepare_sign(a, raw, out, width_px=200)
            self.assertEqual(set(info["masks"]), {"red", "white", "black"})
            alpha = np.asarray(Image.open(out / "alpha.png")) > 127
            total = sum((np.asarray(Image.open(out / f)) > 127).astype(int) for f in info["masks"].values())
            np.testing.assert_array_equal(total, alpha.astype(int))
            self.assertFalse(alpha[0, 0])  # corner outside the disc is cut out
            self.assertAlmostEqual(info["height_m"], 0.6)
            self.assertAlmostEqual(sum(info["colour_fraction"].values()), 1.0, places=3)


class TestSignScene(unittest.TestCase):
    def test_default_has_no_sign_row(self):
        with tempfile.TemporaryDirectory() as td:
            m = yaml.safe_load((REPO / "config" / "highway_assets.yaml").read_text())
            m["cache_dir"] = str(Path(td) / "cache")
            am = Path(td) / "a.yaml"
            am.write_text(yaml.safe_dump(m))
            hw.main(["--out-dir", str(Path(td) / "s"), "--asset-manifest", str(am), "--allow-missing-assets"])
            self.assertNotIn("sign:", (Path(td) / "s" / "highway.pbrt").read_text())
            self.assertNotIn("signs", json.loads((Path(td) / "s" / "highway_manifest.json").read_text()))

    def test_day_signs_diffuse_with_alpha_and_ground_truth(self):
        with tempfile.TemporaryDirectory() as td:
            txt, man = build(Path(td))
            self.assertIn('MakeNamedMaterial "sign:red" "string type" "diffuse"', txt)
            n = len(man["signs"]["signs"])
            self.assertGreaterEqual(n, 6)
            self.assertEqual(txt.count('"texture alpha"'), 2 * n)
            d = [s["distance_m"] for s in man["signs"]["signs"]]
            self.assertEqual(d, [25.0 + 15.0 * i for i in range(n)])
            for s in man["signs"]["signs"]:
                lo, hi = np.array(s["bbox_world_m"])
                self.assertAlmostEqual(hi[1] - lo[1], s["height_m"], places=3)
                self.assertAlmostEqual(lo[1], 2.0, places=3)
                self.assertEqual(s["license"], "public-domain")
            for f in re.findall(r'"string filename" "([^"]+)"', txt):
                self.assertTrue((Path(td) / "scene" / f).is_file(), f)

    def test_night_signs_retroreflective_and_sheeting_scale(self):
        with tempfile.TemporaryDirectory() as td:
            txt, man = build(Path(td), "--time-of-day", "night")
            self.assertIn('MakeNamedMaterial "sign:white" "string type" "retroreflective"', txt)
            ra3 = np.loadtxt(Path(td) / "scene" / "spd" / "ra_sign_white.spd")[:, 1]
        with tempfile.TemporaryDirectory() as td:
            _, man = build(Path(td), "--time-of-day", "night", "--sign-sheeting", "type11")
            ra11 = np.loadtxt(Path(td) / "scene" / "spd" / "ra_sign_white.spd")[:, 1]
            self.assertEqual(man["signs"]["sheeting"], "type11")
        np.testing.assert_allclose(ra11, ra3 * signs.SHEETING_SCALE["type11"], rtol=1e-4)

    def test_mix_chain_covers_every_colour(self):
        info = {"masks": {"red": "r.png", "white": "w.png", "black": "b.png"}}
        lines, top = signs.sign_material("s", Path("/p"), info, Path("/p"))
        txt = "\n".join(lines)
        self.assertEqual(top, "s:mix0")
        for c in ("red", "white", "black"):
            self.assertIn(f'"sign:{c}"', txt)
        self.assertEqual(txt.count('"string type" "mix"'), 2)

    def test_bad_args(self):
        with self.assertRaises(SystemExit):
            hw.main(["--sign-set", "stvo", "--sign-spacing-m", "0", "--out-dir", "/tmp/x_sign_bad"])


if __name__ == "__main__":
    unittest.main()
