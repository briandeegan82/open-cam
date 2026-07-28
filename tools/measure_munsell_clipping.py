#!/usr/bin/env python3
"""Measure per-patch overexposure clipping across Munsell hue-family renders.

Each hue family is a separate scene/electron image; patches are laid out on a
value(row) x chroma(col) grid over a centred grey surround, with empty cells
where no chip exists (ragged grid).  The camera is identical for every hue and
each board is centred at the world origin, so a single (center_x, center_y,
scale_px_per_world_unit) calibration projects every board.  Patch (row,col) is
reconstructed from each chip's value/chroma exactly as build_munsell_scenes.py
lays them out.

A patch is "clipped from overexposure" when its central-window max-channel MEAN
electrons reach >= clip_frac * full_well.  Because electrons scale linearly with
integration time, this module also reports, per candidate integration time, the
global fraction of clipped patches -- used to tune exposure to a target (~20%).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from camera_model import load_camera_model  # noqa: E402


def _value_group_key(value):
    return math.inf if value is None else float(value)


def reconstruct_rowcol(patches: list[dict]) -> list[tuple[int, int]]:
    """Return (row_from_top, col) for each patch, matching layout_value_chroma."""
    values = sorted({p.get("value") for p in patches}, key=_value_group_key)
    rows = len(values)
    row_from_bottom = {v: i for i, v in enumerate(values)}
    # group by value, order each row by chroma (then hue_step, chip index)
    by_value: dict = {}
    for p in patches:
        by_value.setdefault(p.get("value"), []).append(p)
    col_of = {}
    for v, group in by_value.items():
        ordered = sorted(
            group,
            key=lambda p: (
                float(p.get("chroma") if p.get("chroma") is not None else math.inf),
                float(p.get("hue_step") if p.get("hue_step") is not None else math.inf),
                int(p.get("chip_index_1based", 0)),
            ),
        )
        for col, p in enumerate(ordered):
            col_of[p["patch_index"]] = col
    out = []
    for p in patches:
        rft = rows - 1 - row_from_bottom[p.get("value")]
        out.append((rft, col_of[p["patch_index"]]))
    return out


def patch_centers_px(manifest: dict, rowcol, cx, cy, s):
    """World patch centres -> pixel centres for a centred board (no mirror needed:
    col increases left->right and row_from_top increases top->bottom in image)."""
    g = manifest["geometry"]
    pw, ph, gap = g["patch_width"], g["patch_height"], g["gap"]
    board_w, board_h = g["board_size"]
    pitch_x, pitch_y = pw + gap, ph + gap
    centers = []
    for (row, col) in rowcol:
        fx = (col * pitch_x + 0.5 * pw) / board_w
        fy = (row * pitch_y + 0.5 * ph) / board_h
        u = cx + (fx - 0.5) * s * board_w
        v = cy + (fy - 0.5) * s * board_h
        centers.append((u, v))
    return centers, (pw / board_w, ph / board_h)


def sample_windows(e, centers, wfrac, s, board_w, board_h, pw, ph):
    half_wx = 0.5 * wfrac * pw * s
    half_wy = 0.5 * wfrac * ph * s
    H, W = e.shape[:2]
    wins = []
    for (u, v) in centers:
        c0, c1 = int(round(u - half_wx)), int(round(u + half_wx))
        r0, r1 = int(round(v - half_wy)), int(round(v + half_wy))
        c0, c1 = max(0, c0), min(W, c1)
        r0, r1 = max(0, r0), min(H, r1)
        wins.append((r0, r1, c0, c1))
    return wins


def fit_scale(e, manifest, rowcol, cx0, cy0, s0):
    """Refine (cx,cy,s) by minimising mean within-window coefficient of variation
    of luminance across filled patches (aligned windows sit inside uniform patches)."""
    g = manifest["geometry"]
    pw, ph = g["patch_width"], g["patch_height"]
    board_w, board_h = g["board_size"]
    lum = e.mean(axis=2)

    def cost(cx, cy, s):
        centers, _ = patch_centers_px(manifest, rowcol, cx, cy, s)
        wins = sample_windows(lum[..., None], centers, 0.5, s, board_w, board_h, pw, ph)
        cvs = []
        for (r0, r1, c0, c1) in wins:
            if r1 - r0 < 3 or c1 - c0 < 3:
                return 1e9
            w = lum[r0:r1, c0:c1]
            m = w.mean()
            if m > 1e-6:
                cvs.append(w.std() / m)
        return float(np.mean(cvs)) if cvs else 1e9

    best = (cost(cx0, cy0, s0), cx0, cy0, s0)
    # coarse-to-fine search over scale and centre
    for s in np.linspace(s0 * 0.7, s0 * 1.3, 25):
        for dcx in np.linspace(-20, 20, 9):
            for dcy in np.linspace(-20, 20, 9):
                c = cost(cx0 + dcx, cy0 + dcy, s)
                if c < best[0]:
                    best = (c, cx0 + dcx, cy0 + dcy, s)
    _, cx, cy, s = best
    for s2 in np.linspace(s * 0.95, s * 1.05, 21):
        for dcx in np.linspace(-5, 5, 11):
            for dcy in np.linspace(-5, 5, 11):
                c = cost(cx + dcx, cy + dcy, s2)
                if c < best[0]:
                    best = (c, cx + dcx, cy + dcy, s2)
    return best  # (cost, cx, cy, s)


def measure_hue(npz_path, manifest, cx, cy, s, full_well, clip_frac, wfrac=0.5):
    e = np.asarray(np.load(npz_path)["electrons_rgb"], dtype=np.float64)
    g = manifest["geometry"]
    pw, ph = g["patch_width"], g["patch_height"]
    board_w, board_h = g["board_size"]
    patches = manifest["patches"]
    rowcol = reconstruct_rowcol(patches)
    centers, _ = patch_centers_px(manifest, rowcol, cx, cy, s)
    wins = sample_windows(e, centers, wfrac, s, board_w, board_h, pw, ph)
    rows = []
    for p, (r0, r1, c0, c1) in zip(patches, wins):
        win = e[r0:r1, c0:c1, :]
        mean_rgb = win.reshape(-1, 3).mean(axis=0)
        max_mean = float(mean_rgb.max())
        rows.append({
            "label": p["label"], "value": p.get("value"), "chroma": p.get("chroma"),
            "max_channel_mean_e": max_mean,
            "clipped": bool(max_mean >= clip_frac * full_well),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--electrons-glob", default="out/munsell_electrons/*_electrons.npz")
    ap.add_argument("--scenes-root", default="scenes/generated/munsell")
    ap.add_argument("--camera-model-config", default="config/camera_recipes/iphone_8_rggb.yaml")
    ap.add_argument("--current-t", type=float, required=True, help="integration time (s) used for the npz files")
    ap.add_argument("--clip-frac", type=float, default=0.98)
    ap.add_argument("--target-pct", type=float, default=20.0)
    ap.add_argument("--cx", type=float, default=480.0)
    ap.add_argument("--cy", type=float, default=320.0)
    ap.add_argument("--s0", type=float, default=196.0, help="initial px-per-world-unit guess")
    ap.add_argument("--fit-on", default=None, help="hue slug to auto-fit calibration on (else use cx,cy,s0)")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    cm = load_camera_model(Path(args.camera_model_config))
    full_well = float(cm["noise"]["adc"]["full_well_e"])

    npzs = sorted(glob.glob(args.electrons_glob))
    if not npzs:
        raise SystemExit(f"no electron npz found: {args.electrons_glob}")

    def manifest_for(npz):
        slug = Path(npz).name.replace("_electrons.npz", "").replace("munsell_", "")
        mp = Path(args.scenes_root) / slug / f"munsell_{slug}_manifest.json"
        return slug, json.loads(Path(mp).read_text())

    cx, cy, s = args.cx, args.cy, args.s0
    if args.fit_on:
        npz = next(n for n in npzs if f"munsell_{args.fit_on}_" in n or Path(n).name.startswith(f"munsell_{args.fit_on}_"))
        slug, man = manifest_for(npz)
        e = np.asarray(np.load(npz)["electrons_rgb"], dtype=np.float64)
        rowcol = reconstruct_rowcol(man["patches"])
        c, cx, cy, s = fit_scale(e, man, rowcol, args.cx, args.cy, args.s0)
        print(f"# calibrated on {slug}: cx={cx:.1f} cy={cy:.1f} s={s:.2f} (cost={c:.4f})", file=sys.stderr)

    all_rows = []
    per_hue = {}
    for npz in npzs:
        slug, man = manifest_for(npz)
        rows = measure_hue(npz, man, cx, cy, s, full_well, args.clip_frac)
        per_hue[slug] = {"n": len(rows), "clipped": sum(r["clipped"] for r in rows)}
        for r in rows:
            r["hue"] = slug
        all_rows.extend(rows)

    n = len(all_rows)
    n_clip = sum(r["clipped"] for r in all_rows)
    # tune integration time: electrons ~ t, so a patch clips when
    # max_e * (t/current_t) >= clip_frac*full_well.
    maxes = np.array(sorted((r["max_channel_mean_e"] for r in all_rows), reverse=True))
    k = max(1, int(round(args.target_pct / 100.0 * n)))
    t_for_target = args.current_t * (args.clip_frac * full_well) / max(1e-9, maxes[k - 1])
    report = {
        "calibration": {"cx": cx, "cy": cy, "s_px_per_world": s},
        "current_t_s": args.current_t, "full_well_e": full_well, "clip_frac": args.clip_frac,
        "n_patches": n, "n_clipped": n_clip, "pct_clipped": round(100.0 * n_clip / n, 2),
        "target_pct": args.target_pct,
        "integration_time_for_target_s": round(t_for_target, 4),
        "per_hue": per_hue,
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
