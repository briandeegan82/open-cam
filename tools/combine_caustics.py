#!/usr/bin/env python3
"""Render and sum a caustic split diorama (``build_iq_diorama.py``, default ``--caustics auto``).

The path pass (volpath ``"bool skipcaustics"``) and the caustic pass (sppm ``"bool causticsonly"``,
pbrt patch 0021) cover disjoint light paths, so the film output is their per-channel sum.

Scenes built without the split (``--caustics path``, RGB film, SPPM) render as one pass.

  python tools/combine_caustics.py OUT_DIR [--render] [--pbrt PATH] [--seed N] [--tag _seed1]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from _repo import ensure_pbrt_output_dir

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exr_multispectral import read_separate_exr_channels, write_separate_channels_exr  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def tagged(name: str, tag: str) -> str:
    """``name`` with ``tag`` inserted before the suffix (``a.exr``, ``_seed1`` -> ``a_seed1.exr``)."""
    p = Path(name)
    return f"{p.stem}{tag}{p.suffix}"


def combine(scene_dir: Path, tag: str = "") -> Path:
    """Sum the path and caustic EXRs of ``scene_dir`` into its ``film_output``; return that path."""
    scene_dir = Path(scene_dir)
    meta = json.loads((scene_dir / "diorama.json").read_text())
    split = meta.get("caustic_split")
    if not split:
        raise ValueError(f"{scene_dir}/diorama.json has no caustic_split (build with --caustics photons)")
    a = read_separate_exr_channels(scene_dir / tagged(split["path_output"], tag))
    b = read_separate_exr_channels(scene_dir / tagged(split["caustic_output"], tag))
    if set(a) != set(b):
        raise ValueError(f"channel mismatch: {sorted(set(a) ^ set(b))}")
    out = scene_dir / tagged(meta["film_output"], tag)
    write_separate_channels_exr(out, {k: (a[k].astype(np.float64) + b[k]).astype(np.float32) for k in a})
    return out


def render(scene_dir: Path, pbrt: Path, extra: list[str] = (), tag: str = "") -> Path:
    """Render the diorama in ``scene_dir`` with ``pbrt`` (``extra`` e.g. ``--seed``); return the film EXR.

    A split scene runs both passes and sums them. ``tag`` renders to tagged file names (``--outfile``),
    e.g. an independent ``--seed 1`` twin next to the main render."""
    scene_dir = Path(scene_dir).resolve()
    meta = json.loads((scene_dir / "diorama.json").read_text())
    split = meta.get("caustic_split")
    passes = (
        [(split["path_scene"], split["path_output"]), (split["caustic_scene"], split["caustic_output"])]
        if split
        else [("iq_diorama.pbrt", meta["film_output"])]
    )
    if split and any(a in ("--gpu", "--wavefront") for a in extra):
        raise ValueError(
            "the wavefront/GPU integrator ignores skipcaustics/causticsonly, so a split scene would "
            "count the caustic twice; rebuild with --caustics path"
        )
    for scene, output in passes:
        outfile = ["--outfile", tagged(output, tag)] if tag else []
        ensure_pbrt_output_dir(scene_dir / scene, scene_dir, outfile[1] if outfile else None)
        subprocess.run([str(pbrt), "--quiet", *extra, *outfile, scene], cwd=scene_dir, check=True)
    return combine(scene_dir, tag) if split else scene_dir / tagged(meta["film_output"], tag)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--render", action="store_true", help="run the pbrt pass(es) first")
    ap.add_argument("--pbrt", type=Path, default=REPO / "third_party/pbrt-v4/build/pbrt")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", default="", help="suffix for the output file names (with --render)")
    args = ap.parse_args(argv)
    extra = [] if args.seed is None else ["--seed", str(args.seed)]
    out = render(args.scene_dir, args.pbrt, extra, args.tag) if args.render else combine(args.scene_dir, args.tag)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
