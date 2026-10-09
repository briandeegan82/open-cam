#!/usr/bin/env python3
"""Fetch and prepare the third-party highway assets listed in config/highway_assets.yaml.

Downloads go to ``scenes/assets/highway/raw/<asset>/`` (gitignored) and are checked against
the pinned sha256 (or a pinned git commit). Each asset is then prepared for
tools/build_highway_scene.py under ``scenes/assets/highway/prepared/<asset>/``:

* ``hdri``     -> sun removed and measured (``sun.json``) + equal-area ``sky_equiarea.exr``
* ``car_pbrt`` -> ``car.json``: materials, PLY shape list and bounding box of the car
* ``gltf``     -> PLY meshes (node transforms baked) + ``model.json`` with materials
* ``glb``      -> binary glTF unpacked to .gltf/.bin/images, then prepared like ``gltf``
* ``texture``  -> zip extracted; maps referenced from the asset manifest

    venv/bin/python tools/fetch_highway_assets.py            # everything
    venv/bin/python tools/fetch_highway_assets.py --list
    venv/bin/python tools/fetch_highway_assets.py sky_kloofendal_43d_clear car_bmw_m6
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shutil
import struct
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO / "config" / "highway_assets.yaml"
PREPARED_VERSION = 2


def load_asset_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    data = yaml.safe_load(Path(path).read_text())
    for aid, a in data["assets"].items():
        for key in ("kind", "source", "license", "title"):
            if not a.get(key):
                raise ValueError(f"asset {aid}: missing {key!r}")
        if not a.get("files") and not a.get("git"):
            raise ValueError(f"asset {aid}: needs 'files' or 'git'")
        for f in a.get("files", []):
            if not re.fullmatch(r"[0-9a-f]{64}", str(f.get("sha256", ""))):
                raise ValueError(f"asset {aid}: file {f.get('path')} needs a sha256")
        if a.get("git") and not re.fullmatch(r"[0-9a-f]{40}", str(a["git"].get("commit", ""))):
            raise ValueError(f"asset {aid}: git source must pin a full commit hash")
    return data


def cache_root(repo: Path, manifest: dict) -> Path:
    return (repo / manifest.get("cache_dir", "scenes/assets/highway")).resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, sha256: str) -> None:
    if dest.is_file() and sha256_file(dest) == sha256:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "open-cam-asset-fetch/1"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as fh:
        shutil.copyfileobj(r, fh, 1 << 20)
    got = sha256_file(tmp)
    if got != sha256:
        tmp.unlink()
        raise RuntimeError(f"checksum mismatch for {url}: expected {sha256}, got {got}")
    tmp.replace(dest)


def _fetch_git(spec: dict, dest: Path) -> None:
    stamp = dest / ".git_commit"
    if stamp.is_file() and stamp.read_text().strip() == spec["commit"]:
        return
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    def git(*a: str) -> None:
        subprocess.run(["git", "-C", str(dest), *a], check=True)

    git("init", "-q")
    git("remote", "add", "origin", spec["repo"])
    git("config", "core.sparseCheckout", "true")
    (dest / ".git" / "info" / "sparse-checkout").write_text("\n".join(f"/{p}" for p in spec["paths"]) + "\n")
    git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", spec["commit"])
    git("checkout", "-q", "FETCH_HEAD")
    head = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != spec["commit"]:
        raise RuntimeError(f"git checkout gave {head}, expected pinned {spec['commit']}")
    stamp.write_text(spec["commit"] + "\n")


def fetch_raw(aid: str, asset: dict, root: Path) -> Path:
    raw = root / "raw" / aid
    if asset.get("git"):
        _fetch_git(asset["git"], raw)
    for f in asset.get("files", []):
        dest = raw / f["path"]
        _download(f["url"], dest, f["sha256"])
        if "extract" in f:
            stamp = raw / f"{f['path']}.extracted"
            if not stamp.is_file() or stamp.read_text().strip() != f["sha256"]:
                with zipfile.ZipFile(dest) as z:
                    target = (raw / f["extract"]).resolve()
                    for m in z.namelist():  # refuse path traversal
                        if not (target / m).resolve().is_relative_to(target):
                            raise RuntimeError(f"unsafe zip member {m}")
                    z.extractall(target)
                stamp.write_text(f["sha256"] + "\n")
    return raw


# ---------------------------------------------------------------------------- PLY
_PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def read_ply_positions(path: Path) -> np.ndarray:
    """Vertex positions of a binary little-endian PLY (only what is needed for bounds)."""
    data = Path(path).read_bytes()
    end = data.index(b"end_header") + len(b"end_header")
    end += 2 if data[end : end + 2] == b"\r\n" else 1  # exactly one EOL; binary data may start with 0x0a/0x0d
    header = data[:end].decode("latin1").split("\n")
    if not any("binary_little_endian" in ln for ln in header):
        raise ValueError(f"{path}: only binary_little_endian PLY supported")
    n, props, in_vertex = 0, [], False
    for ln in header:
        t = ln.split()
        if t[:1] == ["element"]:
            in_vertex = t[1] == "vertex"
            n = int(t[2]) if in_vertex else n
        elif t[:1] == ["property"] and in_vertex:
            props.append((t[2], _PLY_TYPES[t[1]]))
    v = np.frombuffer(data, dtype=np.dtype(props), count=n, offset=end)
    return np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64)


def robust_bounds(points: np.ndarray, limit: float = 100.0) -> tuple[np.ndarray, np.ndarray] | None:
    """Bounds ignoring non-finite and stray (|coord| > limit) vertices some exported meshes contain."""
    p = points[np.isfinite(points).all(1) & (np.abs(points) < limit).all(1)]
    return (p.min(0), p.max(0)) if len(p) else None


def write_ply(path: Path, pos: np.ndarray, nrm: np.ndarray | None, uv: np.ndarray | None, tri: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    cols = [pos]
    if nrm is not None:
        fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
        cols.append(nrm)
    if uv is not None:
        fields += [("u", "<f4"), ("v", "<f4")]
        cols.append(uv)
    vert = np.zeros(len(pos), dtype=fields)
    flat = np.concatenate(cols, axis=1)
    for i, (name, _) in enumerate(fields):
        vert[name] = flat[:, i]
    faces = np.zeros(len(tri), dtype=[("n", "u1"), ("i", "<i4", 3)])
    faces["n"], faces["i"] = 3, tri
    head = ["ply", "format binary_little_endian 1.0", f"element vertex {len(pos)}"]
    head += [f"property float {name}" for name, _ in fields]
    head += [f"element face {len(tri)}", "property list uint8 int vertex_indices", "end_header"]
    with open(path, "wb") as fh:
        fh.write(("\n".join(head) + "\n").encode())
        fh.write(vert.tobytes())
        fh.write(faces.tobytes())


# ---------------------------------------------------------------------------- prepare
def prepare_hdri(asset: dict, raw: Path, out: Path, n: int = 1024) -> dict:
    from highway_sky import equirect_to_equal_area, excise_sun, read_rgb_exr, write_rgb_exr

    img = read_rgb_exr(raw / asset["files"][0]["path"])
    sky, sun = excise_sun(img)
    write_rgb_exr(out / "sky_equiarea.exr", equirect_to_equal_area(sky, n))
    return {"sky_equiarea": "sky_equiarea.exr", **sun}


_TOP = re.compile(r"^[A-Za-z]")


def parse_pbrt_car(text: str) -> tuple[dict[str, str], list[dict]]:
    """Named materials (body text) and plymesh shapes (material, filename) of a pbrt scene."""
    world = text.split("WorldBegin", 1)[1]
    materials: dict[str, str] = {}
    shapes: list[dict] = []
    current = None
    lines = world.splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        m = re.match(r'MakeNamedMaterial\s+"([^"]+)"', s)
        if m:
            body = []
            i += 1
            while i < len(lines) and (lines[i].strip().startswith('"') or not lines[i].strip()):
                body.append(lines[i].strip())
                i += 1
            materials[m.group(1)] = "\n".join(b for b in body if b)
            continue
        m = re.match(r'NamedMaterial\s+"([^"]+)"', s)
        if m:
            current = m.group(1)
        elif s.startswith('Shape "plymesh"'):
            j, fn = i, None
            while j < len(lines) and fn is None and j < i + 6:
                fm = re.search(r'"string filename"\s*\[?\s*"([^"]+)"', lines[j])
                fn = fm.group(1) if fm else None
                j += 1
            if fn is None:
                raise ValueError(f"plymesh without filename near line {i}")
            shapes.append({"material": current, "ply": fn})
        i += 1
    return materials, shapes


def prepare_car(asset: dict, raw: Path, out: Path) -> dict:
    scene = raw / asset["scene_file"]
    base = scene.parent
    materials, shapes = parse_pbrt_car(scene.read_text())
    excl_m = set(asset.get("exclude_materials", []))
    excl_s = set(asset.get("exclude_meshes", []))
    keep = [s for s in shapes if s["material"] not in excl_m and s["ply"] not in excl_s]
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for s in keep:
        b = robust_bounds(read_ply_positions(base / s["ply"]))
        if b is not None:
            lo, hi = np.minimum(lo, b[0]), np.maximum(hi, b[1])
        s["ply"] = str((base / s["ply"]).relative_to(raw))
    roles = asset.get("roles", {})
    missing = [m for ms in roles.values() for m in ms if m not in materials]
    if missing:
        raise ValueError(f"car roles reference unknown materials {missing}")
    return {
        "materials": materials,  # all: mix materials refer to others by name
        "shapes": keep,
        "bbox_min": lo.tolist(),
        "bbox_max": hi.tolist(),
        "up_axis": "y",
        "front_axis": "-x",
        "roles": roles,
    }


_GLTF_COMP = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
_GLTF_N = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _gltf_accessor(g: dict, buffers: list[bytes], idx: int) -> np.ndarray:
    acc = g["accessors"][idx]
    bv = g["bufferViews"][acc["bufferView"]]
    dt = np.dtype(_GLTF_COMP[acc["componentType"]])
    ncomp = _GLTF_N[acc["type"]]
    start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride", dt.itemsize * ncomp)
    buf = buffers[bv["buffer"]]
    if stride == dt.itemsize * ncomp:
        a = np.frombuffer(buf, dtype=dt, count=acc["count"] * ncomp, offset=start).reshape(acc["count"], ncomp)
    else:
        rows = np.lib.stride_tricks.as_strided(
            np.frombuffer(buf, dtype=np.uint8, offset=start), (acc["count"], stride), (stride, 1)
        )
        a = rows[:, : dt.itemsize * ncomp].copy().view(dt).reshape(acc["count"], ncomp)
    return a.astype(np.float64 if dt.kind == "f" else np.int64)


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    t = np.eye(4)
    t[:3, 3] = node.get("translation", [0, 0, 0])
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    r = np.eye(4)
    r[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    s = np.diag([*node.get("scale", [1, 1, 1]), 1.0])
    return t @ r @ s


def prepare_gltf(asset: dict, raw: Path, out: Path) -> dict:
    """Convert a glTF 2.0 (separate .bin, triangle primitives) to PLY + material list."""
    gpath = raw / asset["gltf"]
    g = json.loads(gpath.read_text())
    buffers = []
    for b in g["buffers"]:
        uri = b["uri"]
        if uri.startswith("data:"):
            buffers.append(base64.b64decode(uri.split(",", 1)[1]))
        else:
            buffers.append((gpath.parent / uri).read_bytes())
    mats = []
    for m in g.get("materials", []):
        pbr = m.get("pbrMetallicRoughness", {})
        tex = pbr.get("baseColorTexture")
        img = None
        if tex is not None:
            src = g["textures"][tex["index"]]["source"]
            img = str((gpath.parent / g["images"][src]["uri"]).relative_to(raw))
        mats.append(
            {
                "name": m.get("name", f"mat{len(mats)}"),
                "base_color_texture": img,
                "base_color_factor": pbr.get("baseColorFactor", [1, 1, 1, 1])[:3],
                "roughness": float(pbr.get("roughnessFactor", 1.0)),
                "metallic": float(pbr.get("metallicFactor", 1.0)),
                "alpha": float(pbr.get("baseColorFactor", [1, 1, 1, 1])[3]),
                "alpha_mode": m.get("alphaMode", "OPAQUE"),
                "double_sided": bool(m.get("doubleSided", False)),
            }
        )
    shapes: list[dict] = []
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)

    def visit(ni: int, parent: np.ndarray) -> None:
        nonlocal lo, hi
        node = g["nodes"][ni]
        mtx = parent @ _node_matrix(node)
        if "mesh" in node:
            for prim in g["meshes"][node["mesh"]]["primitives"]:
                if prim.get("mode", 4) != 4:
                    continue
                at = prim["attributes"]
                pos = _gltf_accessor(g, buffers, at["POSITION"])
                acc = g["accessors"][at["POSITION"]]
                bad = ~np.isfinite(pos).all(1)
                if "min" in acc and "max" in acc:  # spec-mandatory bounds; reject corrupt vertices
                    tol = 1e-3 * (np.ptp([acc["min"], acc["max"]], axis=0).max() + 1.0)
                    bad |= ((pos < np.array(acc["min"]) - tol) | (pos > np.array(acc["max"]) + tol)).any(1)
                pos = pos @ mtx[:3, :3].T + mtx[:3, 3]
                nrm = None
                if "NORMAL" in at:
                    nm = np.linalg.inv(mtx[:3, :3]).T
                    nrm = _gltf_accessor(g, buffers, at["NORMAL"]) @ nm.T
                    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
                uv = None
                if "TEXCOORD_0" in at:
                    uv = _gltf_accessor(g, buffers, at["TEXCOORD_0"])
                    uv[:, 1] = 1.0 - uv[:, 1]  # glTF v down -> pbrt v up
                if "indices" in prim:
                    tri = _gltf_accessor(g, buffers, prim["indices"]).reshape(-1, 3)
                else:
                    tri = np.arange(len(pos)).reshape(-1, 3)
                if bad.any():
                    tri = tri[~bad[tri].any(1)]
                    pos[bad] = 0.0
                if not len(tri):
                    continue
                name = f"mesh_{len(shapes):03d}.ply"
                write_ply(out / name, pos, nrm, uv, tri)
                used = pos[np.unique(tri)]
                lo, hi = np.minimum(lo, used.min(0)), np.maximum(hi, used.max(0))
                shapes.append({"ply": name, "material": int(prim.get("material", 0)), "triangles": int(len(tri))})
        for c in node.get("children", []):
            visit(c, mtx)

    for ni in g["scenes"][g.get("scene", 0)]["nodes"]:
        visit(ni, np.eye(4))
    return {"materials": mats, "shapes": shapes, "bbox_min": lo.tolist(), "bbox_max": hi.tolist(), "up_axis": "y"}


def unpack_glb(glb: Path, dest: Path) -> Path:
    """Split a binary glTF (.glb) into .gltf + .bin + image files so prepare_gltf can read it."""
    data = glb.read_bytes()
    magic, version, _ = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67 or version != 2:
        raise ValueError(f"{glb}: not a glTF 2.0 binary")
    off, chunks = 12, {}
    while off < len(data):
        length, ctype = struct.unpack_from("<II", data, off)
        chunks[ctype] = data[off + 8 : off + 8 + length]
        off += 8 + length
    g = json.loads(chunks[0x4E4F534A])
    binary = chunks.get(0x004E4942, b"")
    dest.mkdir(parents=True, exist_ok=True)
    ext = {"image/png": ".png", "image/jpeg": ".jpg"}
    for i, img in enumerate(g.get("images", [])):
        if "bufferView" not in img:
            continue
        bv = g["bufferViews"][img.pop("bufferView")]
        name = f"image_{i:03d}{ext.get(img.pop('mimeType', ''), '.png')}"
        start = bv.get("byteOffset", 0)
        (dest / name).write_bytes(binary[start : start + bv["byteLength"]])
        img["uri"] = name
    (dest / "buffer.bin").write_bytes(binary)
    g["buffers"] = [{"uri": "buffer.bin", "byteLength": len(binary)}]
    out = dest / (glb.stem + ".gltf")
    out.write_text(json.dumps(g))
    return out


def prepare(aid: str, asset: dict, root: Path, *, force: bool = False) -> Path:
    raw = fetch_raw(aid, asset, root)
    out = root / "prepared" / aid
    info_path = out / "prepared.json"
    if info_path.is_file() and not force:
        info = json.loads(info_path.read_text())
        if info.get("version") == PREPARED_VERSION and info.get("asset") == asset:
            return out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    kind = asset["kind"]
    if kind == "hdri":
        data = prepare_hdri(asset, raw, out)
    elif kind == "car_pbrt":
        data = prepare_car(asset, raw, out)
    elif kind == "gltf":
        data = prepare_gltf(asset, raw, out)
    elif kind == "glb":
        gltf = unpack_glb(raw / asset["glb"], raw / "unpacked")
        data = prepare_gltf({**asset, "gltf": str(gltf.relative_to(raw))}, raw, out)
    elif kind == "texture":
        missing = [m for m in asset.get("maps", {}).values() if not (raw / m).is_file()]
        if missing:
            raise FileNotFoundError(f"{aid}: texture maps missing after extraction: {missing}")
        data = {"maps": asset.get("maps", {}), "tile_size_m": asset.get("tile_size_m", 1.0)}
    else:
        raise ValueError(f"{aid}: unknown asset kind {kind!r}")
    info = {"version": PREPARED_VERSION, "asset_id": aid, "asset": asset, "raw_dir": f"../../raw/{aid}", **data}
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("assets", nargs="*", help="Asset ids (default: all).")
    ap.add_argument("--repo-root", type=Path, default=REPO)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--list", action="store_true", help="List assets with source and licence, then exit.")
    ap.add_argument("--force", action="store_true", help="Re-run preparation even if up to date.")
    args = ap.parse_args(argv)
    repo = args.repo_root.resolve()
    manifest = load_asset_manifest(args.manifest or repo / "config" / "highway_assets.yaml")
    assets = manifest["assets"]
    if args.list:
        for aid, a in assets.items():
            print(f"{aid:34s} {a['kind']:9s} {a['license']:9s} {a['source']}")
        return
    unknown = [a for a in args.assets if a not in assets]
    if unknown:
        ap.error(f"unknown asset ids {unknown}; see --list")
    root = cache_root(repo, manifest)
    for aid in args.assets or list(assets):
        print(f"[{aid}] {assets[aid]['title']} ({assets[aid]['license']})")
        out = prepare(aid, assets[aid], root, force=args.force)
        print(f"  ready: {out.relative_to(repo) if out.is_relative_to(repo) else out}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
