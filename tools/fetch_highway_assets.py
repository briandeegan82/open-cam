#!/usr/bin/env python3
"""Fetch CC0 car assets and convert them to per-slot binary PLY meshes.

Reads config/highway_assets.yaml, downloads each referenced pack (sha256
verified; if a pinned URL has rotted, re-scrapes the pack's fallback page for
the current zip), then for every model:

- parses the OBJ (v / f / g / usemtl; polygons triangulated fan-wise),
- assigns faces to semantic slots (paint / glass / tire / trim) by matching
  group and material names against the registry's slot_map globs,
- normalizes: forward axis -> -Z, uniform scale to the registry length,
  min-y -> 0, x/z centered,
- writes scenes/assets/cars/<model>/{paint,tire,...}.ply (binary little
  endian) plus meta.json (license, source, sha256, bbox, triangle counts).

Offline use: --local-zip PATH imports a user-supplied zip for a pack instead
of downloading. The scene builder falls back to procedural cars whenever a
model directory is missing, so this script is never a hard dependency.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import re
import struct
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent

_UA = {"User-Agent": "open-cam-asset-fetcher/1.0"}


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def fetch_pack_zip(pack: dict, local_zip: Path | None) -> tuple[bytes, str]:
    """Return (zip bytes, sha256). Verifies the pinned hash when present."""
    if local_zip is not None:
        data = local_zip.read_bytes()
    else:
        try:
            data = _download(pack["url"])
        except Exception as e:  # URL rot: re-scrape the asset page
            page_url = pack.get("fallback_page")
            if not page_url:
                raise
            print(f"  pinned URL failed ({e}); scraping {page_url}", flush=True)
            html = _download(page_url).decode("utf-8", "replace")
            m = re.search(r"['\"]([^'\"]*?\.zip)['\"]", html)
            if not m:
                raise RuntimeError(f"no .zip link found on {page_url}") from e
            zip_url = m.group(1)
            if zip_url.startswith("media/"):
                zip_url = page_url.split("/assets/")[0] + "/" + zip_url
            print(f"  found {zip_url}", flush=True)
            data = _download(zip_url)

    digest = hashlib.sha256(data).hexdigest()
    pinned = pack.get("sha256")
    if pinned and digest != pinned:
        print(f"  WARNING: sha256 mismatch\n    pinned:  {pinned}\n    got:     {digest}\n"
              "  (upstream pack updated? review and re-pin in config/highway_assets.yaml)",
              flush=True)
    return data, digest


# ---------------------------------------------------------------------------
# OBJ parsing
# ---------------------------------------------------------------------------

def parse_obj(text: str) -> tuple[np.ndarray, list[tuple[str, str, list[list[int]]]]]:
    """Minimal OBJ parser.

    Returns (vertices (N,3), chunks) where each chunk is
    (group_name, material_name, faces) with faces as vertex-index triangles
    (0-based, fan-triangulated).
    """
    verts: list[list[float]] = []
    chunks: list[tuple[str, str, list[list[int]]]] = []
    group, material = "", ""
    faces: list[list[int]] = []

    def flush():
        nonlocal faces
        if faces:
            chunks.append((group, material, faces))
            faces = []

    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith(("g ", "o ")):
            flush()
            group = line.split(maxsplit=1)[1] if " " in line else ""
        elif line.startswith("usemtl"):
            flush()
            material = line.split(maxsplit=1)[1] if " " in line else ""
        elif line.startswith("f "):
            idx = []
            for tok in line.split()[1:]:
                vi = tok.split("/")[0]
                i = int(vi)
                idx.append(i - 1 if i > 0 else len(verts) + i)
            for k in range(1, len(idx) - 1):  # fan triangulation
                faces.append([idx[0], idx[k], idx[k + 1]])
    flush()
    return np.asarray(verts, dtype=np.float64), chunks


def assign_slots(chunks, slot_map: dict[str, list[str]]) -> dict[str, list[list[int]]]:
    """Map chunks to slots by group/material glob match (slot_map order wins)."""
    out: dict[str, list[list[int]]] = {}
    applied: list[tuple[str, str, str]] = []
    for group, material, faces in chunks:
        slot = "paint"
        for cand, patterns in slot_map.items():
            names = (group.lower(), material.lower())
            if any(fnmatch.fnmatch(n, p.lower()) for n in names for p in patterns):
                slot = cand
                break
        out.setdefault(slot, []).extend(faces)
        applied.append((group or "-", material or "-", slot))
    for g, m, s in applied:
        print(f"    group={g!r} material={m!r} -> {s}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Normalization + PLY output
# ---------------------------------------------------------------------------

_FWD_ROT = {  # yaw (deg) about +Y taking the source forward axis to -Z
    "-z": 0.0, "+x": 90.0, "+z": 180.0, "-x": -90.0,
}


def normalize(verts: np.ndarray, forward: str, length_m: float) -> np.ndarray:
    if forward not in _FWD_ROT:
        raise ValueError(f"unsupported forward axis {forward!r} (want one of {list(_FWD_ROT)})")
    a = np.deg2rad(_FWD_ROT[forward])
    ca, sa = np.cos(a), np.sin(a)
    rot = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
    v = verts @ rot.T
    ext = v.max(0) - v.min(0)
    if ext[2] <= 0:
        raise ValueError("degenerate mesh (zero length)")
    v = v * (length_m / ext[2])
    v[:, 1] -= v[:, 1].min()                       # wheels on the ground
    center = (v.max(0) + v.min(0)) / 2.0
    v[:, 0] -= center[0]
    v[:, 2] -= center[2]
    return v


def write_ply(path: Path, verts: np.ndarray, faces: list[list[int]]) -> tuple[int, int]:
    """Binary little-endian PLY with compacted vertices for this face subset."""
    f = np.asarray(faces, dtype=np.int64)
    used = np.unique(f.reshape(-1))
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    v = verts[used].astype("<f4")
    fi = remap[f].astype("<i4")

    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(v)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(fi)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    with path.open("wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(v.tobytes())
        for tri in fi:
            fh.write(struct.pack("<B3i", 3, *tri))
    return len(v), len(fi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=REPO)
    ap.add_argument("--registry", default="config/highway_assets.yaml")
    ap.add_argument("--out-dir", default="scenes/assets/cars")
    ap.add_argument("--models", default=None,
                    help="Comma list of models to fetch (default: all in registry).")
    ap.add_argument("--local-zip", type=Path, default=None,
                    help="Use a local pack zip instead of downloading.")
    ap.add_argument("--force", action="store_true", help="Rebuild existing model dirs.")
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    reg = yaml.safe_load((repo / args.registry).read_text())
    out_root = repo / args.out_dir
    wanted = ([m.strip() for m in args.models.split(",")] if args.models
              else list(reg["models"]))

    pack_cache: dict[str, tuple[zipfile.ZipFile, str]] = {}
    for model in wanted:
        if model not in reg["models"]:
            raise SystemExit(f"model '{model}' not in registry (known: {list(reg['models'])})")
        m = reg["models"][model]
        out_dir = out_root / model
        if (out_dir / "meta.json").exists() and not args.force:
            print(f"{model}: exists, skipping (--force to rebuild)", flush=True)
            continue

        pack_name = m["pack"]
        if pack_name not in pack_cache:
            print(f"fetching pack {pack_name} ...", flush=True)
            data, digest = fetch_pack_zip(reg["packs"][pack_name], args.local_zip)
            pack_cache[pack_name] = (zipfile.ZipFile(io.BytesIO(data)), digest)
        zf, digest = pack_cache[pack_name]
        pack = reg["packs"][pack_name]

        obj_path = f"{pack['obj_dir']}/{m['obj']}"
        print(f"{model}: {obj_path}", flush=True)
        verts, chunks = parse_obj(zf.read(obj_path).decode("utf-8", "replace"))
        slots = assign_slots(chunks, reg["slot_map"])
        verts = normalize(verts, m["forward"], float(m["length_m"]))

        out_dir.mkdir(parents=True, exist_ok=True)
        counts = {}
        for slot, faces in slots.items():
            nv, nf = write_ply(out_dir / f"{slot}.ply", verts, faces)
            counts[slot] = {"vertices": nv, "triangles": nf}
        meta = {
            "model": model, "pack": pack_name, "source_obj": obj_path,
            "license": pack.get("license", "unknown"),
            "source_url": pack.get("url"), "pack_sha256": digest,
            "length_m": m["length_m"], "forward_normalized": "-z",
            "bbox_min": verts.min(0).round(4).tolist(),
            "bbox_max": verts.max(0).round(4).tolist(),
            "slots": counts,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"  -> {out_dir.relative_to(repo)} "
              f"({sum(c['triangles'] for c in counts.values())} tris, slots: {sorted(counts)})",
              flush=True)


if __name__ == "__main__":
    main()
