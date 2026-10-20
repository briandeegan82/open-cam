"""Opt-in traffic-sign scene: real sign artwork on retroreflective sheeting (``--sign-set``).

``--sign-set stvo`` (German StVO) or ``--sign-set mutcd`` (US MUTCD) puts a row of signs on posts
beside the right shoulder, one every ``--sign-spacing-m`` from ``--sign-start-m``. The artwork comes
from public-domain Wikimedia Commons SVGs listed in ``config/highway_assets.yaml`` (kind
``sign_svg``; StVO signs are German official works, MUTCD signs US federal works). Each SVG is
rasterised by ``tools/fetch_highway_assets.py`` into an alpha cutout (octagons, triangles and
discs keep their shape) and one mask per sheeting colour, so every colour region is its own
spectral material. Nothing is committed; the default scene is unchanged.

* Daylight: diffuse sheeting reflectances from ``highway_spectra`` (analytic shapes).
* Night/dusk with retroreflection on: each colour is the patched pbrt ``retroreflective`` material
  scaled to ASTM D4956 Type III minima x ``SHEETING_MARGIN`` (``highway_night``). Blue has no
  entry there; it uses ``BLUE_RA_FRACTION_OF_WHITE`` of white, a nominal value. ``--sign-sheeting``
  scales all colours by ``SHEETING_SCALE`` (Type I / III / XI white minima ratios, nominal values
  not checked against the paywalled standard).
* Sizes are nominal (StVO size class 2, MUTCD conventional road) and recorded per asset.
* Ground truth: ``highway_manifest.json`` -> ``signs`` has each sign's code, world bounding box,
  distance, size, colour fractions, source and licence.
* Not modelled: sheeting ageing/dirt, legend embossing, fluorescent sheeting (see
  ``--fluorescent-sign`` for a plain fluorescent face), sign-face specular sheen.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import highway_night as night
import numpy as np

SETS = ("none", "stvo", "mutcd")
COLOURS = ("white", "red", "yellow", "blue", "green", "black")
BLUE_RA_FRACTION_OF_WHITE = 0.08
SHEETING_SCALE = {"type1": 70.0 / 250.0, "type3": 1.0, "type11": 580.0 / 250.0}


def add_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("traffic signs (tools/highway_signs.py, opt-in)")
    g.add_argument("--sign-set", choices=SETS, default="none", help="Row of real sign artwork beside the road.")
    g.add_argument("--sign-start-m", type=float, default=25.0, help="Distance of the first sign [m].")
    g.add_argument("--sign-spacing-m", type=float, default=15.0, help="Spacing between signs [m].")
    g.add_argument("--sign-offset-m", type=float, default=1.0, help="Gap from the paved edge to the sign [m].")
    g.add_argument("--sign-bottom-m", type=float, default=2.0, help="Height of the sign's lower edge [m].")
    g.add_argument("--sign-sheeting", choices=tuple(SHEETING_SCALE), default="type3")


def enabled(args: argparse.Namespace) -> bool:
    return getattr(args, "sign_set", "none") != "none"


def check_args(args: argparse.Namespace) -> None:
    if not enabled(args):
        return
    if args.sign_start_m <= 0 or args.sign_spacing_m <= 0:
        raise SystemExit("error: --sign-start-m and --sign-spacing-m must be > 0")
    if args.sign_bottom_m < 0:
        raise SystemExit("error: --sign-bottom-m must be >= 0")


def set_assets(amanifest: dict, sign_set: str) -> list[str]:
    return sorted(k for k, a in amanifest["assets"].items() if a["kind"] == "sign_svg" and a["sign_set"] == sign_set)


def material_lines(args, spd: Path, out_dir: Path, wl: np.ndarray, retro: bool) -> list[str]:
    """``sign:<colour>`` materials: diffuse by day, scaled retroreflective sheeting when ``retro``."""
    L = []
    scale = SHEETING_SCALE[args.sign_sheeting]
    for c in COLOURS:
        r = os.path.relpath(spd / f"sheeting_{c}.spd", out_dir)
        if not retro:
            L.append(f'MakeNamedMaterial "sign:{c}" "string type" "diffuse" "spectrum reflectance" "{r}"')
            continue
        p = night.SHEETING
        L += [
            f'MakeNamedMaterial "sign:{c}" "string type" "retroreflective" "spectrum reflectance" "{r}"',
            f'    "float observationangle" [{p["alpha0_deg"]}] "float lobewidth" [{p["width_deg"]}]'
            f' "float entranceexponent" [{p["q"]}]',
        ]
        if c == "black":
            continue
        val = (
            night.sheeting_ra_param("white") * BLUE_RA_FRACTION_OF_WHITE if c == "blue" else night.sheeting_ra_param(c)
        )
        f = spd / f"ra_sign_{c}.spd"
        night._write_spd(f, wl, scale * night.retro_ra_spectrum(wl, f"sheeting_{c}", val))
        L.append(f'    "spectrum ra" "{os.path.relpath(f, out_dir)}"')
    return L + [""]


def sign_material(name: str, pdir: Path, info: dict, out_dir: Path) -> tuple[list[str], str]:
    """Nested ``mix`` over the sign's colour masks: level k picks colour k where its mask is 1."""
    cols = [c for c in COLOURS if c in info["masks"]]
    L, inner = [], f"sign:{cols[-1]}"
    for i in range(len(cols) - 2, -1, -1):
        c = cols[i]
        f = os.path.relpath(pdir / info["masks"][c], out_dir)
        lvl = f"{name}:mix{i}"
        L += [
            f'Texture "{name}:{c}" "float" "imagemap" "string filename" "{f}" "string encoding" "linear"',
            # mix amount = weight of the second material, so 1 - mask
            f'Texture "{name}:amt{i}" "float" "mix" "float tex1" [1] "float tex2" [0] "texture amount" "{name}:{c}"',
            f'MakeNamedMaterial "{lvl}" "string type" "mix" "string materials" ["sign:{c}" "{inner}"]',
            f'    "texture amount" "{name}:amt{i}"',
        ]
        inner = lvl
    return L, inner


def scene_lines(
    args, amanifest: dict, asset, aroot: Path, out_dir: Path, x_edge: float, mesh, box, warp
) -> tuple[list[str], dict]:
    if not enabled(args):
        return [], {}
    L, signs = [], []
    z = args.sign_start_m
    for aid in set_assets(amanifest, args.sign_set):
        got = asset(aid)
        if got is None:
            continue
        info, _raw = got
        pdir = aroot / "prepared" / aid
        name = f"sign_{aid}"
        mlines, mat = sign_material(name, pdir, info, out_dir)
        w, h = info["width_m"], info["height_m"]
        x0 = x_edge + args.sign_offset_m
        y0 = args.sign_bottom_m
        p = np.array([(x0, y0, z), (x0 + w, y0, z), (x0 + w, y0 + h, z), (x0, y0 + h, z)])
        back = p + np.array([0.0, 0.0, 0.004])
        uv = np.array([(0, 0), (1, 0), (1, 1), (0, 1)])
        tri = np.array([(0, 2, 1), (0, 3, 2)])
        alpha = os.path.relpath(pdir / info["alpha"], out_dir)
        L += [
            *mlines,
            f'Texture "{name}:alpha" "float" "imagemap" "string filename" "{alpha}" "string encoding" "linear"'
            ' "string filter" "point"',
            "AttributeBegin",
            f'NamedMaterial "{mat}"',
            *mesh(p, tri, uv),
            f'    "texture alpha" "{name}:alpha"',
            'NamedMaterial "galvanized"',
            *mesh(back, tri[:, ::-1], uv),
            f'    "texture alpha" "{name}:alpha"',
            *mesh(*box(x0 + w / 2, -0.1, z + 0.05, 0.076, y0 + 0.5 * h + 0.1, 0.076)),
            "AttributeEnd",
            "",
        ]
        wp = warp(p)
        a = amanifest["assets"][aid]
        signs.append(
            {
                "asset": aid,
                "sign_set": info["sign_set"],
                "code": info["code"],
                "title": a["title"],
                "distance_m": z,
                "width_m": w,
                "height_m": round(h, 4),
                "bbox_world_m": [np.round(wp.min(0), 4).tolist(), np.round(wp.max(0), 4).tolist()],
                "colour_fraction": info["colour_fraction"],
                "source": a["source"],
                "license": a["license"],
            }
        )
        z += args.sign_spacing_m
    meta = {
        "sign_set": args.sign_set,
        "sheeting": args.sign_sheeting,
        "sheeting_scale": SHEETING_SCALE[args.sign_sheeting],
        "signs": signs,
        "notes": [
            "Sizes nominal (StVO size class 2 / MUTCD conventional), not checked against the regulations.",
            "Sheeting reflectances are analytic; blue R_A and Type I/XI scales are nominal.",
        ],
    }
    return L, meta
