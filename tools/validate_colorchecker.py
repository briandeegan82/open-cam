#!/usr/bin/env python3
"""Validate spectral data and optionally run pbrt and summarize the rendered EXR."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# The CIE 1931 observer and the tristimulus integral live in colour_science,
# shared with the ISP demo so the validator and the demo cannot disagree.
from colour_science import tristimulus


def check_neutral_luminance(repo: Path) -> bool:
    npz_path = repo / "scenes" / "generated" / "spectral_reference_1nm.npz"
    if not npz_path.is_file():
        print(f"skip neutral ladder check: missing {npz_path} (run build_colorchecker_scene.py)", file=sys.stderr)
        return True
    data = np.load(npz_path)
    lam = data["wavelength_nm"]
    e = data["illuminant"]
    refl = data["reflectance"]
    yvals = []
    for i in range(18, 24):
        spd = refl[i] * e
        _, y, _ = tristimulus(lam, spd)
        yvals.append(y)
    ok = all(yvals[j] > yvals[j + 1] for j in range(len(yvals) - 1))
    if not ok:
        print("neutral patches19–24: expected strictly decreasing luminance Y under D55*R", file=sys.stderr)
        print("Y:", [round(v, 6) for v in yvals], file=sys.stderr)
    else:
        print("neutral ladder (patches 19–24): Y decreases OK")
        print("  Y:", [round(v, 6) for v in yvals])
    return ok


def summarize_exr(path: Path, imgtool: Optional[Path]) -> None:
    if imgtool and imgtool.is_file():
        r = subprocess.run([str(imgtool), "info", str(path)], capture_output=True, text=True, check=False)
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                if "resolution" in line or "avg" in line or "samples per pixel" in line:
                    print(line.strip())
            return
    try:
        import imageio.v3 as iio
    except ImportError:
        print("install imageio or build pbrt imgtool for EXR stats", file=sys.stderr)
        return
    img = iio.imread(path)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    flat = np.reshape(img, (-1, img.shape[-1]))
    print(f"EXR {path}: shape={img.shape} mean={np.mean(flat, axis=0)} min={np.min(flat, axis=0)} max={np.max(flat, axis=0)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--repo-root", type=Path, default=root)
    ap.add_argument("--render", action="store_true", help="run pbrt on scenes/generated/colorchecker.pbrt")
    ap.add_argument(
        "--pbrt",
        type=Path,
        default=None,
        help="pbrt executable (default: third_party/pbrt-v4/build/pbrt)",
    )
    ap.add_argument("--exr", type=Path, default=None, help="summarize this EXR (default: out/colorchecker.exr)")
    ap.add_argument(
        "--imgtool",
        type=Path,
        default=None,
        help="pbrt imgtool binary (default: third_party/pbrt-v4/build/imgtool)",
    )
    args = ap.parse_args()

    repo = args.repo_root
    ok = check_neutral_luminance(repo)

    pbrt_bin = args.pbrt or (repo / "third_party" / "pbrt-v4" / "build" / "pbrt")
    scene = repo / "scenes" / "generated" / "colorchecker.pbrt"
    exr_out = args.exr or (repo / "out" / "colorchecker.exr")
    imgtool = args.imgtool or (repo / "third_party" / "pbrt-v4" / "build" / "imgtool")

    if args.render:
        if not pbrt_bin.is_file():
            print(f"error: pbrt not found at {pbrt_bin}", file=sys.stderr)
            sys.exit(2)
        if not scene.is_file():
            print(f"error: scene missing {scene}; run tools/build_colorchecker_scene.py", file=sys.stderr)
            sys.exit(2)
        r = subprocess.run([str(pbrt_bin), str(scene)], cwd=str(repo), check=False)
        if r.returncode != 0:
            sys.exit(r.returncode)

    if exr_out.is_file():
        print("EXR summary:")
        summarize_exr(exr_out, imgtool)
    elif args.render:
        print(f"warning: expected output missing {exr_out}", file=sys.stderr)

    manifest = repo / "scenes" / "generated" / "colorchecker_manifest.json"
    if manifest.is_file():
        print("manifest:", json.loads(manifest.read_text())["scene"])

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
