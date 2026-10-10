#!/usr/bin/env python3
"""Procedural IQ-lab diorama: a tabletop scene that stresses every IQ-lab axis at once (no external assets).

Contents (all procedural / repo spectra):
* spectral 24-patch ColorChecker (``spectra/xrite``) and the 12-patch skin-tone chart (``iqlab/skin.py``),
* dead-leaves texture card (``iqlab/dead_leaves.py``), 36-spoke Siemens star card, 5 deg slanted-edge card,
* matte, glossy (coated diffuse), metal (conductor) and glass spheres in front of the cards (focus depth),
* a bright window in the back wall (HDR + flare/veiling glare) and a small practical lamp bulb,
* a dim overhead panel as fill light.

``diorama.json`` gives raster ROIs (pinhole projection) for every card and chart patch, the window and
the lamp, so renders can be scored with ``iqlab`` directly (texture MTF on the dead-leaves ROI, SFR on the
edge, colour on the charts, window/shadow ratio for scene DR). For a realistic lens the ROIs are
approximate (distortion); locate features in the image instead.

    venv/bin/python tools/build_iq_diorama.py --out-dir scenes/iq_lab/diorama
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import build_image_quality_targets as iq
import build_skin_tone_chart as skin_chart
import imageio.v3 as iio
import numpy as np
from colour_science import load_colorchecker
from iqlab.dead_leaves import dead_leaves

REPO = Path(__file__).resolve().parent.parent
CARD_Z = -0.3
CARD_Y0 = 0.12


def siemens_star(size: int = 1024, spokes: int = 36, low: float = 0.05, high: float = 0.8, ss: int = 4) -> np.ndarray:
    n = size * ss
    c = (np.arange(n) + 0.5) / n * 2 - 1
    x, y = np.meshgrid(c, -c)
    star = np.where(np.sin(spokes / 2 * np.arctan2(y, x) * 2) > 0, high, low)
    star = np.where(np.hypot(x, y) > 0.98, 0.5 * (low + high), star)
    return star.reshape(size, ss, size, ss).mean((1, 3))


def slanted_edge(
    size: int = 512, angle_deg: float = 5.0, low: float = 0.1, high: float = 0.8, ss: int = 4
) -> np.ndarray:
    n = size * ss
    c = (np.arange(n) + 0.5) / n - 0.5
    x, y = np.meshgrid(c, c)
    t = math.radians(angle_deg)
    img = np.where(x * math.cos(t) + y * math.sin(t) > 0, high, low)
    return img.reshape(size, ss, size, ss).mean((1, 3))


def _png16(path: Path, img: np.ndarray) -> None:
    iio.imwrite(path, np.round(np.clip(img, 0, 1) * 65535).astype(np.uint16))


def _quad(x0, y0, x1, y1, z) -> str:
    return (
        f'Shape "bilinearmesh" "point3 P" [ {x0:.6g} {y0:.6g} {z:.6g}  {x1:.6g} {y0:.6g} {z:.6g}  '
        f'{x0:.6g} {y1:.6g} {z:.6g}  {x1:.6g} {y1:.6g} {z:.6g} ] "point2 uv" [0 0 1 0 0 1 1 1]'
    )


def _spd(path: Path, wl: np.ndarray, val: np.ndarray) -> None:
    path.write_text("".join(f"{a:.1f} {b:.6g}\n" for a, b in zip(wl, val)))


class Projector:
    """Pinhole projection matching the scene camera (looking along -z; world +x -> raster right)."""

    def __init__(self, args: argparse.Namespace):
        self.cx, self.cy = args.xres / 2.0, args.yres / 2.0
        self.f = (min(args.xres, args.yres) / 2.0) / math.tan(math.radians(args.fov) / 2.0)
        self.h, self.d = float(args.cam_height), float(args.cam_dist)

    def __call__(self, x, y, z):
        s = self.f / (self.d - z)
        return self.cx + x * s, self.cy - (y - self.h) * s

    def roi(self, x0, y0, x1, y1, z, margin=0.15) -> list[int]:
        (ax, ay), (bx, by) = self(x0, y0, z), self(x1, y1, z)
        lx, hx, ly, hy = min(ax, bx), max(ax, bx), min(ay, by), max(ay, by)
        mx, my = (hx - lx) * margin, (hy - ly) * margin
        return [int(math.ceil(lx + mx)), int(math.ceil(ly + my)), int(hx - mx), int(hy - my)]


def _patch_grid(lines, meta, proj, out, prefix, refl, wl, cols, x0, size, gap):
    rows = math.ceil(len(refl) / cols)
    y_top = CARD_Y0 + rows * size + (rows - 1) * gap
    lines.append(
        f'AttributeBegin\n    Material "diffuse" "rgb reflectance" [0.03 0.03 0.03]\n    '
        f"{_quad(x0 - gap, CARD_Y0 - gap, x0 + cols * (size + gap), y_top + gap, CARD_Z - 0.002)}\nAttributeEnd"
    )
    for k, r in enumerate(refl):
        rr, cc = divmod(k, cols)
        px0 = x0 + cc * (size + gap)
        py1 = y_top - rr * (size + gap)
        name = f"{prefix}_{k:02d}"
        _spd(out / "spd" / f"{name}.spd", wl, r)
        lines.append(
            f'AttributeBegin\n    Material "diffuse" "spectrum reflectance" "spd/{name}.spd"\n    '
            f"{_quad(px0, py1 - size, px0 + size, py1, CARD_Z)}\nAttributeEnd"
        )
        meta.append({"name": name, "roi_xyxy": proj.roi(px0, py1 - size, px0 + size, py1, CARD_Z, 0.2)})
    return x0 + cols * (size + gap) + gap


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=REPO / "scenes" / "iq_lab" / "diorama")
    ap.add_argument("--window-radiance", type=float, default=3000.0, help="window luminance-ish scale (HDR stress)")
    ap.add_argument("--lamp-radiance", type=float, default=20000.0)
    ap.add_argument("--fill-radiance", type=float, default=3.0)
    ap.add_argument("--camera", choices=("perspective", "thinlens", "realistic"), default="perspective")
    ap.add_argument("--lensfile", default=iq.DEFAULT_REALISTIC_LENSFILE)
    ap.add_argument("--aperture-diameter-mm", type=float, default=4.0)
    ap.add_argument("--focus-distance", type=float, default=None, help="default: card plane")
    ap.add_argument("--thinlens-lens-radius", type=float, default=0.0)
    ap.add_argument("--thinlens-focal-distance", type=float, default=None)
    ap.add_argument("--fov", type=float, default=40.0)
    ap.add_argument("--cam-dist", type=float, default=1.6)
    ap.add_argument("--cam-height", type=float, default=0.2)
    ap.add_argument("--film", choices=("spectral", "rgb"), default="spectral")
    ap.add_argument("--xres", type=int, default=1440)
    ap.add_argument("--yres", type=int, default=960)
    ap.add_argument("--pixelsamples", type=int, default=256)
    ap.add_argument("--maxdepth", type=int, default=6)
    ap.add_argument("--spectral-nbuckets", type=int, default=32)
    ap.add_argument("--spectral-lambda-min", type=float, default=360.0)
    ap.add_argument("--spectral-lambda-max", type=float, default=830.0)
    ap.add_argument("--seed", type=int, default=0, help="dead-leaves seed")
    return ap


def write_diorama(args: argparse.Namespace) -> dict:
    out = Path(args.out_dir)
    (out / "spd").mkdir(parents=True, exist_ok=True)
    (out / "textures").mkdir(exist_ok=True)
    if args.focus_distance is None and args.thinlens_focal_distance is None:
        args.focus_distance = args.thinlens_focal_distance = args.cam_dist - CARD_Z
    proj = Projector(args)
    wl = np.arange(360.0, 831.0, 5.0)
    film = f"iq_diorama_{args.film}.exr"
    h, d = args.cam_height, args.cam_dist
    lines = [
        "# Generated by tools/build_iq_diorama.py",
        'Option "seed" 0',
        'ColorSpace "srgb"',
        f'Sampler "zsobol" "integer pixelsamples" [{int(args.pixelsamples)}]',
        f'Integrator "volpath" "integer maxdepth" [{int(args.maxdepth)}] "bool regularize" true',
        'PixelFilter "gaussian"',
        *iq._film_block(args, film),
        "Scale -1 1 1",  # un-mirror pbrt's LookAt so world +x is raster right (charts read normally)
        f"LookAt 0 {h:.6g} {d:.6g}  0 {h:.6g} 0  0 1 0",
        *iq._camera_block(args, out, REPO),
        "WorldBegin",
        # room: table, back wall, left wall
        'AttributeBegin\n    Material "diffuse" "rgb reflectance" [0.35 0.25 0.16]\n    '
        'Shape "bilinearmesh" "point3 P" [ -2 0 -1  2 0 -1  -2 0 2  2 0 2 ]\nAttributeEnd',
        f'AttributeBegin\n    Material "diffuse" "rgb reflectance" [0.55 0.55 0.52]\n    {_quad(-2, 0, 2, 2.5, -1.0)}\nAttributeEnd',
        'AttributeBegin\n    Material "diffuse" "rgb reflectance" [0.45 0.47 0.5]\n    '
        'Shape "bilinearmesh" "point3 P" [ -1.2 0 -1  -1.2 0 2  -1.2 2.5 -1  -1.2 2.5 2 ]\nAttributeEnd',
        # window (two-sided so orientation cannot hide it), practical lamp, fill panel
        f'AttributeBegin\n    AreaLightSource "diffuse" "blackbody L" [6500] "float scale" [{args.window_radiance:.6g}] '
        f'"bool twosided" true\n    Material "diffuse" "rgb reflectance" [0 0 0]\n    {_quad(0.35, 0.3, 0.95, 0.9, -0.99)}\nAttributeEnd',
        f'AttributeBegin\n    Translate 0.75 0.35 -0.1\n    AreaLightSource "diffuse" "blackbody L" [2700] '
        f'"float scale" [{args.lamp_radiance:.6g}]\n    Shape "sphere" "float radius" [0.02]\nAttributeEnd',
        'AttributeBegin\n    Material "diffuse" "rgb reflectance" [0.2 0.2 0.2]\n    Translate 0.75 0.0 -0.1\n'
        '    Shape "cylinder" "float radius" [0.006] "float zmin" [0] "float zmax" [0.33]\nAttributeEnd'.replace(
            "Translate 0.75 0.0 -0.1\n", "Translate 0.75 0.0 -0.1\n    Rotate -90 1 0 0\n"
        ),
        f'AttributeBegin\n    AreaLightSource "diffuse" "blackbody L" [5000] "float scale" [{args.fill_radiance:.6g}] '
        f'"bool twosided" true\n    Material "diffuse" "rgb reflectance" [0 0 0]\n'
        '    Shape "bilinearmesh" "point3 P" [ -1 2.4 -0.8  1 2.4 -0.8  -1 2.4 1.5  1 2.4 1.5 ]\nAttributeEnd',
    ]
    meta: dict = {"film_output": film, "cards": {}, "colorchecker": [], "skin": [], "spheres": []}
    cc = load_colorchecker(REPO, wl)
    x = _patch_grid(lines, meta["colorchecker"], proj, out, "cc", cc.reflectance, wl, 6, -0.82, 0.042, 0.006)
    skin_refl = [p["reflectance"] for p in skin_chart.patches(wl)]
    meta["skin_names"] = [p["name"] for p in skin_chart.patches(wl)]
    x = _patch_grid(lines, meta["skin"], proj, out, "skin", skin_refl, wl, 4, x + 0.03, 0.042, 0.006)
    cards = {
        "dead_leaves": dead_leaves(1024, seed=int(args.seed)),
        "siemens_star": siemens_star(),
        "slanted_edge": slanted_edge(),
    }
    size = 0.2
    x += 0.03
    for name, img in cards.items():
        _png16(out / "textures" / f"{name}.png", img)
        lines.append(
            f'Texture "{name}" "spectrum" "imagemap" "string filename" "textures/{name}.png" "string encoding" "linear"\n'
            f'AttributeBegin\n    Material "diffuse" "texture reflectance" "{name}"\n    '
            f"{_quad(x, CARD_Y0, x + size, CARD_Y0 + size, CARD_Z)}\nAttributeEnd"
        )
        meta["cards"][name] = {
            "world": [x, CARD_Y0, x + size, CARD_Y0 + size, CARD_Z],
            "roi_xyxy": proj.roi(x, CARD_Y0, x + size, CARD_Y0 + size, CARD_Z, 0.05),
        }
        x += size + 0.03
    spheres = [
        ("matte", '"diffuse" "rgb reflectance" [0.5 0.5 0.5]'),
        ("glossy", '"coateddiffuse" "rgb reflectance" [0.6 0.08 0.05] "float roughness" [0.02]'),
        ("metal", '"conductor" "float roughness" [0.05]'),
        ("glass", '"dielectric" "float eta" [1.5]'),
    ]
    for k, (name, mat) in enumerate(spheres):
        sx, r = -0.45 + 0.3 * k, 0.06
        lines.append(
            f"AttributeBegin\n    Material {mat}\n    Translate {sx:.6g} {r:.6g} 0.25\n"
            f'    Shape "sphere" "float radius" [{r}]\nAttributeEnd'
        )
        meta["spheres"].append(
            {"name": name, "center": [sx, r, 0.25], "roi_xyxy": proj.roi(sx - r, 0, sx + r, 2 * r, 0.25, 0.25)}
        )
    meta["window"] = {"roi_xyxy": proj.roi(0.35, 0.3, 0.95, 0.9, -0.99, 0.1)}
    meta["lamp"] = {"xy": list(proj(0.75, 0.35, -0.1))}
    meta["shadow_reference"] = {
        "roi_xyxy": proj.roi(-1.1, 0.4, -0.85, 0.9, -1.0, 0.1),
        "note": "back wall, left of the cards",
    }
    (out / "iq_diorama.pbrt").write_text("\n".join(lines) + "\n")
    (out / "diorama.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    write_diorama(args)
    print(f"wrote {args.out_dir}/iq_diorama.pbrt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
