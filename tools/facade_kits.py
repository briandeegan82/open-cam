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
PLINTH = 0.8  # procedural facades: 0.8 m plinth + 3.2 m storeys (build_urban_scene.facade_lines)
MATCH_STOREY = 3.2


def lot_height(storeys: int, storey: float | None) -> float:
    """Kit wall height of a lot: 3 m kit storeys, or the procedural plinth + ``storey`` storeys."""
    return STOREY * storeys if storey is None else PLINTH + storey * storeys


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
    """Straight 3 m trim piece (angled, corner and end pieces skipped)."""
    c = sorted(
        n
        for n in P
        if n.startswith(prefix)
        and not any(w in n for w in ("angled", "corner", "end"))
        and abs(P[n]["hi"][0] - P[n]["lo"][0] - BAY) < 0.05
        and P[n]["lo"][2] > -0.5
    )
    return c[0] if c else None


def metal_maps(raw: Path) -> dict[str, tuple[float, Path]]:
    """glTF material name -> (metallicFactor, metallic-roughness texture) for materials that can be
    metallic. The metallic value per texel is factor x the texture's blue channel (glTF 2.0)."""
    import json
    import struct

    out: dict[str, tuple[float, Path]] = {}
    for f in sorted(raw.rglob("*.gltf")) + sorted(raw.rglob("*.glb")):
        if f.suffix == ".glb":
            b = f.read_bytes()
            j = json.loads(b[20 : 20 + struct.unpack("<I", b[12:16])[0]])
        else:
            j = json.loads(f.read_text())
        for m in j.get("materials", []):
            pbr = m.get("pbrMetallicRoughness", {})
            fac = float(pbr.get("metallicFactor", 1.0))
            t = pbr.get("metallicRoughnessTexture")
            if fac > 0 and t is not None:
                uri = j["images"][j["textures"][t["index"]]["source"]].get("uri")
                if uri:
                    out[m["name"]] = (fac, f.parent / uri)
        break
    return out


def _metal_mask(arm: Path, fac: float, dst: Path) -> float:
    """Write factor x blue channel as an 8-bit grey PNG; returns the mean glTF roughness (green) of
    the metallic texels (0 if none)."""
    from PIL import Image

    a = np.asarray(Image.open(arm).convert("RGB"), dtype=np.float64) / 255.0
    metal = np.clip(fac * a[..., 2], 0.0, 1.0)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(metal * 255).astype(np.uint8), mode="L").save(dst)
    sel = metal >= 0.5
    return float(a[..., 1][sel].mean()) if sel.any() else 0.0


def material_lines(aid: str, info: dict, raw: Path, out_dir: Path, metals: bool = False) -> list[str]:
    """Texture colours as diffuse reflectance (sRGB maps, not measured spectra); glass is a thin dielectric.

    ``metals``: where the glTF metallic map says metal, a pbrt ``mix`` picks a conductor whose
    normal-incidence reflectance is the base-colour map (pbrt's artist-style "reflectance"
    conductor; GGX alpha = roughness^2 as in glTF, the mean over the metallic texels); elsewhere
    the diffuse material."""
    L = []
    mm = metal_maps(raw) if metals else {}
    for k, m in enumerate(info["materials"]):
        mn = f"{aid}:{k}"
        if m.get("alpha_mode") == "BLEND" or "glass" in m["name"].lower():
            L.append(f'MakeNamedMaterial "{mn}" "string type" "thindielectric" "float eta" [1.52]')
        elif m.get("base_color_texture"):
            rel = os.path.relpath(raw / m["base_color_texture"], out_dir)
            L.append(f'Texture "{mn}:c" "spectrum" "imagemap" "string filename" "{rel}" "string encoding" "sRGB"')
            mask = out_dir / "textures" / f"{aid}_{k}_metal.png"
            rough = _metal_mask(mm[m["name"]][1], mm[m["name"]][0], mask) if m["name"] in mm else 0.0
            if m["name"] in mm and rough > 0:
                mrel = os.path.relpath(mask, out_dir)
                L += [
                    f'Texture "{mn}:metal" "float" "imagemap" "string filename" "{mrel}" "string encoding" "linear"',
                    f'MakeNamedMaterial "{mn}:d" "string type" "diffuse" "texture reflectance" "{mn}:c"',
                    f'MakeNamedMaterial "{mn}:m" "string type" "conductor" "texture reflectance" "{mn}:c" '
                    f'"float roughness" [{_n(rough**2)}] "bool remaproughness" false',
                    f'MakeNamedMaterial "{mn}" "string type" "mix" "string materials" ["{mn}:d" "{mn}:m"] '
                    f'"texture amount" "{mn}:metal"',
                ]
            else:
                L.append(f'MakeNamedMaterial "{mn}" "string type" "diffuse" "texture reflectance" "{mn}:c"')
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


def instance(
    aid: str, node: str, part: dict, x: float, y: float, z: float, yaw: float, sx: float, sy: float = 1.0
) -> list[str]:
    o = part["origin"]
    return [
        "AttributeBegin",
        f"Translate {_n(x)} {_n(y)} {_n(z)}",
        f"Rotate {_n(yaw)} 0 1 0",
        f"Scale {_n(sx)} {_n(sy)} 1",
        f"Translate {_n(-o[0])} {_n(-o[1])} {_n(-o[2])}",
        f'ObjectInstance "{aid}:{node}"',
        "AttributeEnd",
    ]


def room_quad(part: dict, s: int, x_face: float, y: float, z_anchor: float, sx: float, sy: float = 1.0) -> np.ndarray:
    """Room-back quad behind an insert's opening (world points, facing the street)."""
    (lx0, ly0), (lx1, ly1) = part["lo"][:2], part["hi"][:2]
    ly0, ly1 = sy * ly0, sy * ly1
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
    storey: float | None = None,
    metals: bool = False,
) -> tuple[list[str], list[dict], dict]:
    """Kit street fronts on every lot; returns pbrt lines, per-lot metadata and window counts.

    ``storey``: None keeps the kit's 3 m storeys. A height (3.2 m) matches the procedural facades:
    upper storeys are the 3 m panels stretched to ``storey``, and the ground floor is
    ``PLINTH + storey`` tall, with the kit's base course under the ground-floor windows and a
    plain wall strip above the door. A lot with ``xform = (yaw_deg, z)`` is built in the
    main-street frame and placed by ``Translate 0 0 z`` + ``Rotate yaw_deg 0 1 0`` (cross street)."""
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
            "base": trim(P, "base"),
            "plain": next((n for n in sorted(P) if n.startswith("wall_standard_standard")), None),
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
        lot_inst: list[str] = []
        lot_rooms: list[tuple[str, np.ndarray]] = []

        def put(
            node: str, y: float, za: float, yaw: float, sy: float = 1.0, _a=aid, _K=K, _i=lot_inst, _x=x_face, _sx=sx
        ) -> None:
            _i.extend(instance(_a, node, _K["P"][node], _x, y, za, yaw, _sx, sy))
            used[_a].add(node)

        for f in range(lot["storeys"]):
            if storey is None:
                y, top, sy = STOREY * f, STOREY * (f + 1), 1.0
            else:
                y = 0.0 if f == 0 else PLINTH + storey * f
                top, sy = PLINTH + storey * (f + 1), storey / STOREY
            for b in range(n):
                wall, ins = (door if b == n // 2 else gwin) if f == 0 else up
                yaw, za = frame(s, x_face, lot["z0"] + b * wb, wb)
                y_ins = y
                if storey is not None and f == 0:
                    if b == n // 2:  # door from the pavement, plain wall strip above it
                        if K["plain"]:
                            put(K["plain"], storey, za, yaw, PLINTH / STOREY)
                    else:  # base course under the ground-floor window
                        if K["base"]:
                            base_h = float(K["P"][K["base"]]["hi"][1] - K["P"][K["base"]]["lo"][1])
                            put(K["base"], 0.0, za, yaw, PLINTH / base_h)
                        y_ins = PLINTH
                for node in (wall, ins):
                    put(node, y_ins, za, yaw, sy)
                on = dark and rng.random() < lit
                lot_rooms.append(
                    ("window_lit" if on else "window_dark", room_quad(K["P"][ins], s, x_face, y_ins, za, sx, sy))
                )
                tr = K["cornice"] if f < lot["storeys"] - 1 else K["crown"]
                if tr:
                    put(tr, top, za, yaw)
            if K["pier"]:
                for zp in (lot["z0"], lot["z1"]):
                    psy = sy if storey is None or f else (PLINTH + storey) / STOREY
                    lot_inst.extend(
                        instance(aid, K["pier"], K["P"][K["pier"]], x_face, y, zp, frame(s, x_face, 0, 0)[0], 1.0, psy)
                    )
                used[aid].add(K["pier"])
        xf = lot.get("xform")
        if xf:
            yaw_l, tz = xf
            c, sn = np.cos(np.radians(yaw_l)), np.sin(np.radians(yaw_l))
            lot_inst = [
                "AttributeBegin",
                f"Translate 0 0 {_n(tz)}",
                f"Rotate {_n(yaw_l)} 0 1 0",
                *lot_inst,
                "AttributeEnd",
            ]
            lot_rooms = [
                (nm, np.stack([q[:, 0] * c + q[:, 2] * sn, q[:, 1], -q[:, 0] * sn + q[:, 2] * c + tz], 1))
                for nm, q in lot_rooms
            ]
        inst += lot_inst
        for nm, q in lot_rooms:
            rooms[nm].append(q)
        meta.append(
            {
                "lot": k,
                "kit": aid,
                "upper": up[0],
                "door": door[0],
                "ground": gwin[0],
                "bays": n,
                "bay_m": round(wb, 3),
                **({"street": "cross"} if xf else {}),
            }
        )
    L = ["# ---- facade kits (Poly Haven CC0)"]
    for aid, K in kp.items():
        if not used[aid]:
            continue
        L += material_lines(aid, K["info"], K["raw"], out_dir, metals)
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
