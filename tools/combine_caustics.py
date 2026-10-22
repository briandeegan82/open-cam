#!/usr/bin/env python3
"""Render and sum a caustic split scene (``build_iq_diorama.py --caustics photons``).

The path pass (volpath ``"bool skipcaustics"``) and the caustic pass (sppm ``"bool causticsonly"``,
pbrt patch 0021) cover disjoint light paths, so the film output is their per-channel sum.

  python tools/combine_caustics.py OUT_DIR [--render] [--pbrt PATH] [--seed N]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exr_multispectral import read_separate_exr_channels, write_separate_channels_exr  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def combine(scene_dir: Path) -> Path:
    """Sum the path and caustic EXRs of ``scene_dir`` into its ``film_output``; return that path."""
    scene_dir = Path(scene_dir)
    meta = json.loads((scene_dir / "diorama.json").read_text())
    split = meta.get("caustic_split")
    if not split:
        raise ValueError(f"{scene_dir}/diorama.json has no caustic_split (build with --caustics photons)")
    a = read_separate_exr_channels(scene_dir / split["path_output"])
    b = read_separate_exr_channels(scene_dir / split["caustic_output"])
    if set(a) != set(b):
        raise ValueError(f"channel mismatch: {sorted(set(a) ^ set(b))}")
    out = scene_dir / meta["film_output"]
    write_separate_channels_exr(out, {k: (a[k].astype(np.float64) + b[k]).astype(np.float32) for k in a})
    return out


def render(scene_dir: Path, pbrt: Path, extra: list[str] = ()) -> Path:
    """Run both passes with ``pbrt`` (same extra arguments, e.g. ``--seed``) and combine them."""
    scene_dir = Path(scene_dir).resolve()
    split = json.loads((scene_dir / "diorama.json").read_text())["caustic_split"]
    for scene in (split["path_scene"], split["caustic_scene"]):
        subprocess.run([str(pbrt), "--quiet", *extra, scene], cwd=scene_dir, check=True)
    return combine(scene_dir)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--render", action="store_true", help="run both pbrt passes first")
    ap.add_argument("--pbrt", type=Path, default=REPO / "third_party/pbrt-v4/build/pbrt")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    extra = [] if args.seed is None else ["--seed", str(args.seed)]
    out = render(args.scene_dir, args.pbrt, extra) if args.render else combine(args.scene_dir)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
