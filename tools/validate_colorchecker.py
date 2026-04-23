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

# NumPy compatibility helper (np.trapezoid in newer versions, np.trapz in older).
if hasattr(np, "trapezoid"):
    _trapz = np.trapezoid
else:
    _trapz = np.trapz

# CIE 1931 2-degree standard color matching functions, 5 nm steps380–780 nm.
# Source: CIE 1931 2° (tabulated values commonly distributed with color science texts).
_CMF_WL = np.arange(380, 781, 5, dtype=np.float64)
_CMF_X = np.array(
    [
        0.001368, 0.002236, 0.004243, 0.007650, 0.014310, 0.023120, 0.043510, 0.077630,
        0.134380, 0.214770, 0.283900, 0.328500, 0.348280, 0.348060, 0.336200, 0.318700,
        0.290800, 0.251100, 0.195360, 0.142100, 0.095640, 0.057950, 0.032010, 0.014700,
        0.004900, 0.002400, 0.009300, 0.029100, 0.063270, 0.109600, 0.165500, 0.225750,
        0.290400, 0.359700, 0.433450, 0.512050, 0.594500, 0.678400, 0.762100, 0.842500,
        0.916300, 0.978600, 1.026300, 1.056700, 1.062200, 1.045600, 1.002600, 0.938400,
        0.854450, 0.751400, 0.642400, 0.541900, 0.447900, 0.360800, 0.283500, 0.218700,
        0.164900, 0.121200, 0.087400, 0.063600, 0.046770, 0.032900, 0.022700, 0.015840,
        0.011359, 0.008111, 0.005790, 0.004109, 0.002899, 0.002049, 0.001440, 0.001000,
        0.000690, 0.000476, 0.000332, 0.000235, 0.000166, 0.000117, 0.000083, 0.000059,
        0.000042,
    ],
    dtype=np.float64,
)
_CMF_Y = np.array(
    [
        0.000039, 0.000064, 0.000120, 0.000217, 0.000396, 0.000640, 0.001210, 0.002180,
        0.004000, 0.007300, 0.011600, 0.016840, 0.023000, 0.029800, 0.038000, 0.048000,
        0.060000, 0.073900, 0.090980, 0.112600, 0.139020, 0.169300, 0.208020, 0.258600,
        0.323000, 0.407300, 0.503000, 0.608200, 0.710000, 0.793200, 0.862000, 0.914850,
        0.954000, 0.980300, 0.995000, 1.000000, 0.995000, 0.978600, 0.952000, 0.915400,
        0.870000, 0.816300, 0.757000, 0.694900, 0.631000, 0.566800, 0.503000, 0.441200,
        0.381000, 0.321000, 0.265000, 0.217000, 0.175000, 0.138200, 0.107000, 0.081600,
        0.061000, 0.044580, 0.032000, 0.023200, 0.017000, 0.011920, 0.008210, 0.005723,
        0.004102, 0.002929, 0.002091, 0.001484, 0.001047, 0.000740, 0.000520, 0.000361,
        0.000249, 0.000172, 0.000120, 0.000083, 0.000057, 0.000039, 0.000027, 0.000018,
        0.000012,
    ],
    dtype=np.float64,
)
_CMF_Z = np.array(
    [
        0.006450, 0.010550, 0.020050, 0.036210, 0.067850, 0.110200, 0.207400, 0.371300,
        0.645600, 1.039050, 1.385600, 1.622960, 1.747060, 1.782600, 1.772110, 1.744100,
        1.669200, 1.528100, 1.287640, 1.041900, 0.812950, 0.616200, 0.465180, 0.353300,
        0.272000, 0.212300, 0.158200, 0.111700, 0.078250, 0.057250, 0.042160, 0.029840,
        0.020300, 0.013400, 0.008750, 0.005750, 0.003900, 0.002750, 0.002100, 0.001800,
        0.001650, 0.001400, 0.001100, 0.001000, 0.001800, 0.002900, 0.004900, 0.007400,
        0.009300, 0.008800, 0.007700, 0.005900, 0.004500, 0.003400, 0.002400, 0.001800,
        0.001400, 0.001100, 0.001000, 0.001000, 0.001000, 0.001000, 0.001000, 0.001000,
        0.001000, 0.001000, 0.001000, 0.001000, 0.001000, 0.001000, 0.001000, 0.001000,
        0.001000, 0.001000, 0.001000, 0.001000, 0.001000, 0.001000, 0.001000, 0.001000,
        0.001000,
    ],
    dtype=np.float64,
)


def tristimulus(
    wavelength_nm: np.ndarray,
    spd: np.ndarray,
) -> tuple[float, float, float]:
    """Integrate L_e(lambda) * cmf(lambda) dlambda (simple trapezoid, 1 nm grid assumed)."""
    xb = np.interp(wavelength_nm, _CMF_WL, _CMF_X, left=0.0, right=0.0)
    yb = np.interp(wavelength_nm, _CMF_WL, _CMF_Y, left=0.0, right=0.0)
    zb = np.interp(wavelength_nm, _CMF_WL, _CMF_Z, left=0.0, right=0.0)
    x = float(_trapz(spd * xb, wavelength_nm))
    y = float(_trapz(spd * yb, wavelength_nm))
    z = float(_trapz(spd * zb, wavelength_nm))
    return x, y, z


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
