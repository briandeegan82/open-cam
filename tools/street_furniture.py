"""Opt-in street furniture and imported traffic-signal heads for the urban scene.

Props are Poly Haven (CC0) and Objaverse/Sketchfab (CC-BY-4.0) meshes listed in
config/highway_assets.yaml. They are rendered with their sRGB base-colour textures (no measured
spectra). Imported signal heads only replace the procedural housing: the aspects stay the
procedural LED emitters (same luminance, spectrum and PWM), placed on the model's lenses.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
from fetch_highway_assets import read_ply_positions

# aid -> (placement role, nominal height [m] or None when the asset is metric, file is a row of variants)
PROPS = {
    "prop_fire_hydrant": ("kerb", None, True),
    "prop_metal_trash_can": ("kerb", None, True),
    "prop_painted_wooden_bench": ("facade", None, False),
    "prop_utility_box_01": ("facade", None, False),
    "prop_utility_box_02": ("facade", None, False),
    "prop_barrel_03": ("facade", None, False),
    "prop_concrete_road_barrier": ("barrier", None, False),
    "prop_concrete_road_barrier_02": ("barrier", None, False),
    "cone_orange": ("cone", 0.75, False),
    "cone_reflective": ("cone", 0.75, False),
    "cone_weathered": ("cone", 0.75, False),
    "drum_traffic": ("drum", 1.0, False),
    "drum_construction": ("drum", 1.0, False),
}
# --signal-model -> (asset, lens centre x, middle-aspect y, aspect pitch, lens-face z) in prepared
# coordinates; both heads face +z with red on top.
SIGNAL_MODELS = {
    "nyc80s": ("signal_nyc80s", 0.002, 0.005, 0.298, 0.186),
    "backplate12": ("signal_12in_backplate", 0.0, 0.0, 0.288, 0.114),
}
FACE_YAW = {"-z": 180.0, "-x": -90.0, "+x": 90.0, "+z": 0.0}


def _f(v: float) -> str:
    return f"{v:.6g}"


def variants(aid: str, info: dict, prep: Path) -> list[dict]:
    """Shape groups of a prepared asset with robust bounds; row files split by the sign of shape centre x."""
    pts = [read_ply_positions(prep / s["ply"]) for s in info["shapes"]]
    groups = {0: list(range(len(pts)))}
    if PROPS.get(aid, (None, None, False))[2]:
        groups = {}
        for k, p in enumerate(pts):
            groups.setdefault(int(np.median(p[:, 0]) > 0), []).append(k)
    out = []
    for g, idx in sorted(groups.items()):
        p = np.concatenate([pts[k] for k in idx])
        p = p[np.isfinite(p).all(1)]
        out.append(
            {"name": f"{aid}_v{g}", "shapes": idx, "lo": np.percentile(p, 0.05, 0), "hi": np.percentile(p, 99.95, 0)}
        )
    return out


def retro_band_mask(texture: Path) -> np.ndarray:
    """White sheeting bands of a cone/drum base-colour texture: bright, near-neutral texels (0/1)."""
    from PIL import Image

    rgb = np.asarray(Image.open(texture).convert("RGB"), dtype=np.float32) / 255.0
    lo, hi = rgb.min(-1), rgb.max(-1)
    return ((lo > 0.55) & (hi - lo < 0.2)).astype(np.float32)


def object_lines(
    name: str, info: dict, prep: Path, raw: Path, out_dir: Path, shapes: list[int], retro_bands: bool = False
) -> list[str]:
    """ObjectBegin block in prepared coordinates (instances add the placement transform).

    ``retro_bands``: texels of retro_band_mask() use the builder's (retroreflective) ``sheet_white``.
    """
    out = [f'ObjectBegin "{name}"']
    used = sorted({info["shapes"][k]["material"] for k in shapes}, key=str)
    names = [m["name"] for m in info["materials"]]
    for mk in used:
        k = mk if isinstance(mk, int) else names.index(mk)
        m = info["materials"][k]
        mn = f"{name}:{k}"
        if m.get("base_color_texture") and retro_bands:
            from highway_road_wear import write_float_exr

            rel = os.path.relpath(raw / m["base_color_texture"], out_dir)
            f = out_dir / "textures" / "retro_bands" / f"{name.replace(':', '_')}_{k}.exr"
            write_float_exr(f, retro_band_mask(raw / m["base_color_texture"]))
            out += [
                f'Texture "{mn}:c" "spectrum" "imagemap" "string filename" "{rel}" "string encoding" "sRGB"',
                f'Texture "{mn}:band" "float" "imagemap" "string filename" "{os.path.relpath(f, out_dir)}"',
                f'MakeNamedMaterial "{mn}:base" "string type" "diffuse" "texture reflectance" "{mn}:c"',
                f'MakeNamedMaterial "{mn}" "string type" "mix" "string materials" ["{mn}:base" "sheet_white"]'
                f' "texture amount" "{mn}:band"',
            ]
        elif m.get("base_color_texture"):
            rel = os.path.relpath(raw / m["base_color_texture"], out_dir)
            out += [
                f'Texture "{mn}:c" "spectrum" "imagemap" "string filename" "{rel}" "string encoding" "sRGB"',
                f'MakeNamedMaterial "{mn}" "string type" "diffuse" "texture reflectance" "{mn}:c"',
            ]
        else:
            bc = [min(max(v, 0.02), 0.9) for v in m.get("base_color_factor", (0.5, 0.5, 0.5))[:3]]
            out.append(f'MakeNamedMaterial "{mn}" "string type" "diffuse" "rgb reflectance" [{" ".join(map(_f, bc))}]')
    for k in shapes:
        sh = info["shapes"][k]
        mk = sh["material"] if isinstance(sh["material"], int) else names.index(sh["material"])
        out += [
            f'NamedMaterial "{name}:{mk}"',
            f'Shape "plymesh" "string filename" "{os.path.relpath(prep / sh["ply"], out_dir)}"',
        ]
    return out + ["ObjectEnd", ""]


def instance_lines(obj: str, pos, yaw_deg: float, scale: float, origin) -> list[str]:
    """Place prepared point `origin` at world `pos`, rotated about +y and uniformly scaled."""
    return [
        "AttributeBegin",
        f"    Translate {_f(pos[0])} {_f(pos[1])} {_f(pos[2])}",
        f"    Rotate {_f(yaw_deg)} 0 1 0",
        f"    Scale {_f(scale)} {_f(scale)} {_f(scale)}",
        f"    Translate {_f(-origin[0])} {_f(-origin[1])} {_f(-origin[2])}",
        f'    ObjectInstance "{obj}"',
        "AttributeEnd",
    ]


def world_box(v: dict, pos, yaw_deg: float, scale: float, origin) -> list[list[float]]:
    """Axis-aligned world box of a placed variant (ground truth)."""
    lo, hi = v["lo"], v["hi"]
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) - origin
    a = math.radians(yaw_deg)
    rot = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    w = (scale * c) @ rot.T + np.asarray(pos)
    return [w.min(0).round(3).tolist(), w.max(0).round(3).tolist()]


def placement(v: dict, height_m: float | None) -> tuple[float, np.ndarray]:
    """Scale to the nominal height (or keep metric) and the base-centre origin."""
    lo, hi = v["lo"], v["hi"]
    s = height_m / (hi[1] - lo[1]) if height_m else 1.0
    return s, np.array([0.5 * (lo[0] + hi[0]), lo[1], 0.5 * (lo[2] + hi[2])])


def pavement_slots(rng: np.random.Generator, z_ranges, avoid_z, spacing: float = 7.0) -> list[float]:
    """Jittered positions along a pavement, keeping 1.5 m clear of poles and other fixtures."""
    out = []
    for z0, z1 in z_ranges:
        for z in np.arange(z0, z1, spacing):
            zz = float(z + rng.uniform(-1.5, 1.5))
            if z0 <= zz <= z1 and all(abs(zz - a) > 1.5 for a in avoid_z):
                out.append(zz)
    return out


def work_zone_layout(x_kerb: float, x_lane: float, z0: float, z1: float, taper_m: float = 6.0, pitch: float = 1.5):
    """Channelizer line closing the strip between the kerb and x_lane: taper from the kerb, then along x_lane."""
    pts = []
    for z in np.arange(z0, z1 + 1e-9, pitch):
        t = min((z - z0) / taper_m, 1.0)
        pts.append((x_kerb + t * (x_lane - x_kerb), float(z)))
    return pts


def load_json(p: Path) -> dict:
    return json.loads(p.read_text())
