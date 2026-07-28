#!/usr/bin/env python3
"""Measure per-patch overexposure clipping for a ColorChecker render.

Loads the forward-model electron image (out/sensor_forward_electrons.npz,
key ``electrons_rgb``, shape HxWx3, pre-full-well), locates the 24-patch 6x4
grid via gradient projections, and reports for each patch:

  * mean per-channel electrons (central 40% window)
  * max-channel mean electrons vs the sensor full-well capacity
  * fraction of the patch window at/above full well (saturated pixels)

A patch counts as "clipped from overexposure" when its central-window
max-channel MEAN electrons reach >= clip_frac * full_well (default 0.98),
i.e. the patch is blown out.  Because electrons scale linearly with the
render's target illuminance (lux), the script also predicts the target lux
that yields a requested number of clipped patches.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from camera_model import load_camera_model  # noqa: E402

PATCH_NAMES = [
    "dark_skin", "light_skin", "blue_sky", "foliage", "blue_flower", "bluish_green",
    "orange", "purplish_blue", "moderate_red", "purple", "yellow_green", "orange_yellow",
    "blue", "green", "red", "yellow", "magenta", "cyan",
    "white", "neutral_8", "neutral_65", "neutral_5", "neutral_35", "black",
]


def find_grid_extent(lum: np.ndarray) -> tuple[int, int, int, int]:
    """Return (x0, x1, y0, y1): outer pixel bounds of the 6x4 patch array.

    The patches sit on a uniform grey surround; patch<->gap boundaries create
    strong gradients only across the chart.  Summing |gradient| along each axis
    yields a profile that is elevated over the chart extent.
    """
    gx = np.abs(np.diff(lum, axis=1))
    gy = np.abs(np.diff(lum, axis=0))
    col_profile = gx.sum(axis=0)           # length W-1, indexed by x
    row_profile = gy.sum(axis=1)           # length H-1, indexed by y

    def extent(profile: np.ndarray) -> tuple[int, int]:
        thr = 0.15 * profile.max()
        idx = np.where(profile > thr)[0]
        return int(idx.min()), int(idx.max())

    x0, x1 = extent(col_profile)
    y0, y1 = extent(row_profile)
    return x0, x1, y0, y1


def patch_windows(x0, x1, y0, y1, win_frac=0.4):
    """Yield (index, name, (r0,r1,c0,c1)) sample windows for the 24 patches.

    World geometry: board 2.9 x 1.92, patch 0.45, gap 0.04, pitch 0.49.
    Detected [x0,x1]/[y0,y1] map to the outer edges of the patch array
    (full board span).  Patch centre fractions come from the world layout.
    """
    board_w, board_h = 2.9, 1.92
    pitch, pw, ph = 0.49, 0.45, 0.45
    span_x = x1 - x0
    span_y = y1 - y0
    half_wx = 0.5 * win_frac * (pw / board_w) * span_x
    half_wy = 0.5 * win_frac * (ph / board_h) * span_y
    for row in range(4):
        for col in range(6):
            idx = row * 6 + col
            fx = (col * pitch + 0.5 * pw) / board_w   # 0..1 across board
            fy = (row * pitch + 0.5 * ph) / board_h
            cx = x0 + fx * span_x
            cy = y0 + fy * span_y
            c0, c1 = int(round(cx - half_wx)), int(round(cx + half_wx))
            r0, r1 = int(round(cy - half_wy)), int(round(cy + half_wy))
            yield idx, PATCH_NAMES[idx], (r0, r1, c0, c1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", type=Path, default=REPO / "out/sensor_forward_electrons.npz")
    ap.add_argument("--camera-model-config", type=Path,
                    default=REPO / "config/camera_recipes/iphone_8_rggb.yaml")
    ap.add_argument("--current-lux", type=float, required=True,
                    help="target_illuminance_lux used to produce --npz")
    ap.add_argument("--extent", type=str, default=None,
                    help="Fixed grid extent 'x0,x1,y0,y1' (geometry is exposure-"
                         "independent; pin it from an unsaturated baseline so heavy "
                         "clipping cannot perturb auto-detection).")
    ap.add_argument("--clip-frac", type=float, default=0.98,
                    help="patch clipped when max-channel mean >= clip_frac*full_well")
    ap.add_argument("--target-clipped", type=int, default=5,
                    help="how many clipped patches to solve the predicted lux for")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    cm = load_camera_model(args.camera_model_config)
    full_well = float(cm["noise"]["adc"]["full_well_e"])

    data = np.load(args.npz, allow_pickle=True)
    e = np.asarray(data["electrons_rgb"], dtype=np.float64)  # HxWx3
    lum = e.max(axis=2)

    if args.extent:
        x0, x1, y0, y1 = (int(v) for v in args.extent.split(","))
    else:
        x0, x1, y0, y1 = find_grid_extent(lum)
    rows = []
    for idx, name, (r0, r1, c0, c1) in patch_windows(x0, x1, y0, y1):
        win = e[r0:r1, c0:c1, :]
        mean_rgb = win.reshape(-1, 3).mean(axis=0)
        max_mean = float(mean_rgb.max())
        sat_frac = float((win.max(axis=2) >= args.clip_frac * full_well).mean())
        rows.append({
            "index": idx + 1, "name": name,
            "mean_rgb_e": [round(float(v), 1) for v in mean_rgb],
            "max_channel_mean_e": round(max_mean, 1),
            "fullwell_fraction": round(max_mean / full_well, 3),
            "saturated_pixel_frac": round(sat_frac, 3),
            "clipped": bool(max_mean >= args.clip_frac * full_well),
        })

    n_clipped = sum(r["clipped"] for r in rows)
    # Predict lux for N clipped: electrons scale linearly with lux, so the Nth
    # brightest patch reaches full_well when lux = current_lux * full_well / m_N,
    # where m_N is that patch's current max-channel mean.
    maxes = sorted((r["max_channel_mean_e"] for r in rows), reverse=True)
    predictions = {}
    for n in sorted(set([args.target_clipped, 4, 5, 6, 7, 8])):
        if 1 <= n <= 24:
            m_n = maxes[n - 1]
            lux_n = args.current_lux * (args.clip_frac * full_well) / max(1e-9, m_n)
            predictions[str(n)] = round(lux_n, 1)

    report = {
        "npz": str(args.npz),
        "current_lux": args.current_lux,
        "full_well_e": full_well,
        "clip_frac": args.clip_frac,
        "grid_extent_px": {"x0": x0, "x1": x1, "y0": y0, "y1": y1},
        "n_patches_clipped": n_clipped,
        "pct_clipped": round(100.0 * n_clipped / 24, 1),
        "predicted_lux_for_n_clipped": predictions,
        "patches": rows,
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
