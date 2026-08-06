#!/usr/bin/env python3
"""Batch-render ColorChecker patch pairs under an angled light, one image per pair.

For each (left, right) pair: build the scene (tools/build_patch_pair_scene.py), render
with pbrt (GPU by default), and write a tonemapped sRGB PNG preview plus the raw EXR to
out/patch_pairs/.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
PY = str(REPO / "venv" / "bin" / "python")
PBRT = str(REPO / "third_party" / "pbrt-v4" / "build" / "pbrt")
sys.path.insert(0, str(REPO / "tools"))
from exr_multispectral import read_separate_exr_channels  # noqa: E402

# Pairs requested for analysis.
PAIRS = [(3, 8), (8, 13), (3, 5), (2, 7), (7, 12), (12, 16), (12, 15), (9, 15),
         (5, 10), (11, 14), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24)]


def tonemap(exr_path: Path, png_path: Path, percentile: float) -> None:
    d = read_separate_exr_channels(exr_path)
    rgb = np.stack([d["R"], d["G"], d["B"]], -1).astype(np.float64)
    p = np.percentile(rgb, percentile)
    x = np.clip(rgb / max(p, 1e-9), 0.0, 1.0)
    srgb = np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
    Image.fromarray((srgb * 255).astype(np.uint8)).save(png_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=REPO / "out" / "patch_pairs")
    ap.add_argument("--light-type", default="point", choices=("point", "spot", "distant"))
    ap.add_argument("--light-direction", default="north",
                    choices=("north", "south", "east", "west"))
    ap.add_argument("--light-angle-deg", type=float, default=45.0)
    ap.add_argument("--light-distance", type=float, default=1.8)  # locked: 59% falloff, mirror-symmetric
    ap.add_argument("--light-scale", type=float, default=40.0)
    ap.add_argument("--ambient-scale", type=float, default=4.0)  # locked: light ambient fill
    ap.add_argument("--camera", default="perspective",
                    choices=("perspective", "pinhole", "realistic"))
    ap.add_argument("--fov", type=float, default=42.0)
    ap.add_argument("--pixelsamples", type=int, default=256)
    ap.add_argument("--preview-percentile", type=float, default=99.5)
    ap.add_argument("--gpu", choices=("true", "false"), default="true")
    ap.add_argument("--pairs", type=str, default=None,
                    help='Override pair list, e.g. "3-8,8-13". Default: the analysis set.')
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = PAIRS
    if args.pairs:
        pairs = [tuple(int(x) for x in p.split("-")) for p in args.pairs.split(",")]

    for left, right in pairs:
        tag = f"{left:02d}_{right:02d}"
        exr = out_dir / f"patch_pair_{tag}.exr"
        png = out_dir / f"patch_pair_{tag}.png"
        scene = REPO / "scenes" / "generated" / f"patch_pair_{tag}.pbrt"
        print(f"\n== pair {left}-{right} ==", flush=True)
        subprocess.run([PY, str(REPO / "tools" / "build_patch_pair_scene.py"),
                        "--repo-root", str(REPO),
                        "--left-patch", str(left), "--right-patch", str(right),
                        "--light-type", args.light_type,
                        "--light-direction", args.light_direction,
                        "--light-angle-deg", str(args.light_angle_deg),
                        "--light-distance", str(args.light_distance),
                        "--light-scale", str(args.light_scale),
                        "--ambient-scale", str(args.ambient_scale),
                        "--camera", args.camera, "--fov", str(args.fov),
                        "--film", "rgb", "--pixelsamples", str(args.pixelsamples),
                        "--film-output", str(exr)],
                       cwd=str(REPO), check=True)
        pbrt_cmd = [PBRT] + (["--gpu"] if args.gpu == "true" else []) + [str(scene)]
        subprocess.run(pbrt_cmd, cwd=str(REPO), check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tonemap(exr, png, args.preview_percentile)
        print(f"  wrote {png.relative_to(REPO)}", flush=True)

    print(f"\nDone: {len(pairs)} pairs -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
