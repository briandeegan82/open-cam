#!/usr/bin/env python3
"""Score a rendered / captured skin-tone chart (``build_skin_tone_chart.py``).

Input: a linear sRGB image (pbrt RGB film EXR, a pipeline output EXR/NPY in linear sRGB, or
``--rgb-is-xyz`` for XYZ). Patch means -> XYZ -> L*a*b* with the chart's own white patch as the
reference white (so camera white balance and exposure drop out), compared with the spectral reference
under the same illuminant: CIEDE2000, delta L*/C*/hue, ITA.

    venv/bin/python tools/run_skin_tone_test.py --chart scenes/iq_lab/skin_tones/chart.json \\
        --illuminant D65 --image out/skin_tones_D65_rgb.exr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from colour_science import srgb_linear_to_xyz, xyz_to_lab
from iqlab import skin


def score(chart: dict, img: np.ndarray, illuminant: str, *, rgb_is_xyz: bool = False) -> dict:
    img = np.asarray(img, dtype=np.float64)[..., :3]
    means = np.array(
        [img[y0:y1, x0:x1].reshape(-1, 3).mean(0) for x0, y0, x1, y1 in (p["roi_xyxy"] for p in chart["patches"])]
    )
    xyz = means if rgb_is_xyz else srgb_linear_to_xyz(means)
    wi = chart["white_index"]
    white_ref_y = chart["patches"][wi]["reference_lab"][illuminant][0]
    y_scale = ((white_ref_y + 16) / 116) ** 3  # luminance factor of the 90 % white
    lab = xyz_to_lab(xyz, xyz[wi] / y_scale)
    ref = np.array([p["reference_lab"][illuminant] for p in chart["patches"]])
    skin_idx = [k for k, p in enumerate(chart["patches"]) if "skin" in p["name"]]
    m = skin.skin_metrics(lab[skin_idx], ref[skin_idx])
    rows = [
        {
            "name": chart["patches"][k]["name"],
            "lab": lab[k].tolist(),
            "reference_lab": ref[k].tolist(),
            **{key: float(v[i]) for key, v in m.items()},
        }
        for i, k in enumerate(skin_idx)
    ]
    de = m["delta_e00"]
    return {
        "illuminant": illuminant,
        "patches": rows,
        "mean_delta_e00": float(de.mean()),
        "max_delta_e00": float(de.max()),
    }


def _load(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    from exr_multispectral import linear_rgb_from_exr  # noqa: PLC0415

    return linear_rgb_from_exr(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chart", type=Path, required=True)
    ap.add_argument("--illuminant", required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--rgb-is-xyz", action="store_true")
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args(argv)
    res = score(json.loads(args.chart.read_text()), _load(args.image), args.illuminant, rgb_is_xyz=args.rgb_is_xyz)
    for r in res["patches"]:
        print(
            f"{r['name']:24s} dE00 {r['delta_e00']:5.2f}  dL {r['delta_L']:+5.1f}  dC {r['delta_C']:+5.1f}  "
            f"dh {r['delta_hue_deg']:+5.1f} deg  ITA {r['ita_reference']:+6.1f} -> {r['ita_measured']:+6.1f}"
        )
    print(f"mean dE00 {res['mean_delta_e00']:.2f}, max {res['max_delta_e00']:.2f}")
    if args.out_json:
        args.out_json.write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
