"""Street fronts assembled from the Poly Haven CC0 modular facade kits (urban ``--facade-style kits``).

Both kits are laid out on a 3 m grid: wall panels span local x in [-3, 0] with the wall plane at z = 0
and +z pointing out of the building; window/door inserts share the panel's name suffix
(``wall_window_<X>`` + ``window_<X>``, ``wall_door_<X>`` + ``door_<X>``).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from build_highway_scene import mesh
from fetch_highway_assets import read_ply_positions

KITS = ("kit_urban_apartments", "kit_factory")
BAY = 3.0
STOREY = 3.0
RECESS = 0.4  # lot box sits this far behind the kit wall plane
INTERIOR = 0.35  # room-back quad behind each window / glazed door


def _n(v: float) -> str:
    return f"{float(v):.5g}"


def parts(info: dict, prep: Path) -> dict:
    """Kit parts by glTF node name, with the node's layout offset and its local bounds."""
    out: dict = {}
    for sh in info["shapes"]:
        p = out.setdefault(
            sh["node"],
            {
                "shapes": [],
                "origin": np.asarray(sh["origin"], float),
                "lo": np.full(3, np.inf),
                "hi": np.full(3, -np.inf),
            },
        )
        pos = read_ply_positions(prep / sh["ply"]) - p["origin"]
        p["lo"], p["hi"] = np.minimum(p["lo"], pos.min(0)), np.maximum(p["hi"], pos.max(0))
        p["shapes"].append(sh)
    return out


def pairs(P: dict, wall: str, insert: str) -> list[tuple[str, str]]:
    """Flat 3 m wall panels with a matching insert (angled bays, corners and garages excluded)."""
    res = []
    for n in sorted(P):
        if not n.startswith(wall):
            continue
        ins = insert + n[len(wall) :]
        w = P[n]["hi"] - P[n]["lo"]
        if ins in P and abs(w[0] - BAY) < 0.05 and P[n]["lo"][2] > -0.5 and P[ins]["lo"][2] > -0.5:
            res.append((n, ins))
    return res


def trim(P: dict, prefix: str) -> str | None:
    c = sorted(
        n for n in P if n.startswith(prefix) and "standard" in n and abs(P[n]["hi"][0] - P[n]["lo"][0] - BAY) < 0.05
    )
    return c[0] if c else None


def material_lines(aid: str, info: dict, raw: Path, out_dir: Path) -> list[str]:
    """Texture colours as diffuse reflectance (sRGB maps, not measured spectra); glass is a thin dielectric."""
    L = []
    for k, m in enumerate(info["materials"]):
        mn = f"{aid}:{k}"
        if m.get("alpha_mode") == "BLEND" or "glass" in m["name"].lower():
            L.append(f'MakeNamedMaterial "{mn}" "string type" "thindielectric" "float eta" [1.52]')
        elif m.get("base_color_texture"):
            rel = os.path.relpath(raw / m["base_color_texture"], out_dir)
            L += [
                f'Texture "{mn}:c" "spectrum" "imagemap" "string filename" "{rel}" "string encoding" "sRGB"',
                f'MakeNamedMaterial "{mn}" "string type" "diffuse" "texture reflectance" "{mn}:c"',
            ]
        else:
            L.append(f'MakeNamedMaterial "{mn}" "string type" "diffuse" "rgb reflectance" [0.5 0.5 0.5]')
    return L


def object_lines(aid: str, node: str, part: dict, prep: Path, out_dir: Path) -> list[str]:
    L = [f'ObjectBegin "{aid}:{node}"']
    for sh in part["shapes"]:
        rel = os.path.relpath(prep / sh["ply"], out_dir)
        L += [f'NamedMaterial "{aid}:{sh["material"]}"', f'Shape "plymesh" "string filename" "{rel}"']
    return L + ["ObjectEnd"]


def frame(s: int, x_face: float, zb: float, wb: float):
    """World placement of a bay [zb, zb + wb] on the side-s street face: (yaw, z anchor of local x = 0)."""
    return (-90.0, zb + wb) if s > 0 else (90.0, zb)


def instance(aid: str, node: str, part: dict, x: float, y: float, z: float, yaw: float, sx: float) -> list[str]:
    o = part["origin"]
    return [
        "AttributeBegin",
        f"Translate {_n(x)} {_n(y)} {_n(z)}",
        f"Rotate {_n(yaw)} 0 1 0",
        f"Scale {_n(sx)} 1 1",
        f"Translate {_n(-o[0])} {_n(-o[1])} {_n(-o[2])}",
        f'ObjectInstance "{aid}:{node}"',
        "AttributeEnd",
    ]


def room_quad(part: dict, s: int, x_face: float, y: float, z_anchor: float, sx: float) -> np.ndarray:
    """Room-back quad behind an insert's opening (world points, facing the street)."""
    (lx0, ly0), (lx1, ly1) = part["lo"][:2], part["hi"][:2]
    zs = [z_anchor + sx * lx0, z_anchor + sx * lx1] if s > 0 else [z_anchor - sx * lx0, z_anchor - sx * lx1]
    za, zc = min(zs), max(zs)
    x = x_face + s * INTERIOR
    return np.array([(x, y + ly0, za), (x, y + ly0, zc), (x, y + ly1, zc), (x, y + ly1, za)])


def street_fronts(
    lots: list[dict],
    kits: dict,
    build_x: float,
    seed: int,
    dark: bool,
    lit: float,
    out_dir: Path,
    window_luminance: float,
) -> tuple[list[str], list[dict], dict]:
    """Kit street fronts on every lot; returns pbrt lines, per-lot metadata and window counts."""
    rng = np.random.default_rng([seed, 12])
    kp = {}
    for aid, (info, raw) in kits.items():
        prep = raw.parent.parent / "prepared" / aid
        P = parts(info, prep)
        kp[aid] = {
            "info": info,
            "raw": raw,
            "prep": prep,
            "P": P,
            "win": pairs(P, "wall_window_", "window_"),
            "door": pairs(P, "wall_door_", "door_"),
            "cornice": trim(P, "cornice"),
            "crown": trim(P, "crown"),
            "pier": next((n for n in sorted(P) if n.startswith("wall_pier_standard")), None),
        }
    kp = {a: k for a, k in kp.items() if k["win"] and k["door"]}
    if not kp:
        return [], [], {}
    used: dict[str, set] = {a: set() for a in kp}
    inst: list[str] = []
    rooms: dict[str, list] = {"window_lit": [], "window_dark": []}
    meta = []
    for k, lot in enumerate(lots):
        aid = list(kp)[int(rng.integers(len(kp)))]
        K = kp[aid]
        up = K["win"][int(rng.integers(len(K["win"])))]
        door = K["door"][int(rng.integers(len(K["door"])))]
        gwin = K["win"][int(rng.integers(len(K["win"])))]
        s, x_face = lot["side"], lot["side"] * build_x
        w = lot["z1"] - lot["z0"]
        n = max(1, int(round(w / BAY)))
        wb = w / n
        sx = wb / BAY
        for f in range(lot["storeys"]):
            y = STOREY * f
            for b in range(n):
                wall, ins = (door if b == n // 2 else gwin) if f == 0 else up
                yaw, za = frame(s, x_face, lot["z0"] + b * wb, wb)
                for node in (wall, ins):
                    inst += instance(aid, node, K["P"][node], x_face, y, za, yaw, sx)
                    used[aid].add(node)
                on = dark and rng.random() < lit
                rooms["window_lit" if on else "window_dark"].append(room_quad(K["P"][ins], s, x_face, y, za, sx))
                for tr, ty in ((K["cornice"], y + STOREY) if f < lot["storeys"] - 1 else (K["crown"], y + STOREY),):
                    if tr:
                        inst += instance(aid, tr, K["P"][tr], x_face, ty, za, yaw, sx)
                        used[aid].add(tr)
            if K["pier"]:
                for zp in (lot["z0"], lot["z1"]):
                    inst += instance(aid, K["pier"], K["P"][K["pier"]], x_face, y, zp, frame(s, x_face, 0, 0)[0], 1.0)
                used[aid].add(K["pier"])
        meta.append(
            {"lot": k, "kit": aid, "upper": up[0], "door": door[0], "ground": gwin[0], "bays": n, "bay_m": round(wb, 3)}
        )
    L = ["# ---- facade kits (Poly Haven CC0)"]
    for aid, K in kp.items():
        if not used[aid]:
            continue
        L += material_lines(aid, K["info"], K["raw"], out_dir)
        for node in sorted(used[aid]):
            L += object_lines(aid, node, K["P"][node], K["prep"], out_dir)
    L += inst
    for name, quads in rooms.items():
        if not quads:
            continue
        Pq = np.concatenate(quads)
        T = np.concatenate([np.array([(0, 1, 2), (0, 2, 3)]) + 4 * i for i in range(len(quads))])
        L += ["AttributeBegin", f'NamedMaterial "{name}"']
        if name == "window_lit":
            L.append(
                f'AreaLightSource "diffuse" "spectrum L" "spd/lamp_window.spd" "float scale" [{_n(window_luminance)}]'
            )
        L += [*mesh(Pq, T), "AttributeEnd"]
    return L + [""], meta, {"windows_lit": len(rooms["window_lit"]), "windows_dark": len(rooms["window_dark"])}
