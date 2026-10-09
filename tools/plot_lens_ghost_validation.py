#!/usr/bin/env python3
"""Validation figure for traced lens ghosts (tools/lens_ghosts.py); see docs/LENS_GHOSTS.md.

Panels: (a) coating reflectance spectra, (b) ghost centre vs field angle - exact trace against the
paraxial ghost matrices, (c-e) a pbrt render with traced ghosts for uncoated / MgF2 / QHQ coatings.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import lens_coatings as lc
import lens_ghosts as lg
import matplotlib
import numpy as np
from exr_multispectral import parse_s0_wavelength_nm, read_separate_exr_channels

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COATINGS = ("uncoated", "mgf2", "qhq")


def _signal(path: Path) -> np.ndarray:
    ch = read_separate_exr_channels(path)
    names = [n for n in ch if parse_s0_wavelength_nm(n) is not None]
    lams = np.array([parse_s0_wavelength_nm(n) for n in names])
    from exr_multispectral import trapezoid_weights_nm  # noqa: PLC0415

    w = trapezoid_weights_nm(lams)
    return sum(wk * ch[n].astype(np.float64) for wk, n in zip(w, names))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lens-file", type=Path, default=Path("config/lenses/wide_22mm.dat"))
    ap.add_argument("--aperture-diameter-mm", type=float, default=8.0)
    ap.add_argument("--focus-distance-m", type=float, default=3.0)
    ap.add_argument("--base-exr", type=Path, required=True)
    ap.add_argument("--ghost-exr-pattern", default=None, help="e.g. out/ghosts/x_{coating}.exr")
    ap.add_argument("--out-png", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    metrics: dict = {}
    fig = plt.figure(figsize=(15, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.85])

    # (a) coatings
    ax = fig.add_subplot(gs[0, 0])
    lam = np.linspace(380, 780, 201)
    for name in COATINGS:
        c = lc.coating_for_interface(name, 1.0, 1.62)
        r, _ = lc.stack_rt(lam, 0.0, 1.0, 1.62, c.layers)
        r45, _ = lc.stack_rt(lam, math.sin(math.radians(45)), 1.0, 1.62, c.layers)
        (line,) = ax.plot(lam, 100 * r, label=f"{name} 0 deg")
        ax.plot(lam, 100 * r45, "--", color=line.get_color(), label=f"{name} 45 deg")
        metrics[f"R_{name}_550nm_normal_pct"] = float(100 * np.interp(550, lam, r))
        band = (lam >= 420) & (lam <= 680)
        metrics[f"R_{name}_420_680_mean_pct"] = float(100 * r[band].mean())
    ax.set(
        xlabel="wavelength [nm]",
        ylabel="reflectance per air-glass surface [%]",
        title="(a) coating model, n_glass=1.62",
    )
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 20)
    ax.legend(fontsize=7, ncol=2)

    # (b) ghost position vs field angle
    ax = fig.add_subplot(gs[0, 1])
    lens = lg.load_lens(
        args.lens_file,
        aperture_diameter_mm=args.aperture_diameter_mm,
        focus_distance_m=args.focus_distance_m,
        coating="mgf2",
    )
    ref = lg.ghost_energy_fractions(lens, lg.direction_from_angles(math.radians(10.0)), [550.0], n_grid=64)
    top = sorted(ref["ghosts"].items(), key=lambda kv: -kv[1]["energy"][0])[:6]
    thetas = np.arange(0.5, 26.0, 1.5)
    max_dev = 0.0
    for pair, _info in top:
        steps = lg.ghost_path(lens, *pair)
        xs = []
        for th in thetas:
            o, d, _c, _s = lg.collimated_beam(lens, lg.direction_from_angles(math.radians(th)), 1)

            # Chief ray: through the stop centre, found by 1-D search on the entrance height.
            def stop_x(y0, th=th):
                oo = np.array([[y0 - 5 * math.tan(math.radians(th)), 0.0, -5.0]])
                dd = np.array([[math.sin(math.radians(th)), 0.0, math.cos(math.radians(th))]])
                st = lg._State.new(oo, dd)
                for k in range(lens.stop_index + 1):
                    st = st.step(lens, k, "T")
                return float(st.p[0, 0]), oo, dd

            lo, hi = -lens.surfaces[0].ap_r, lens.surfaces[0].ap_r
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if stop_x(mid)[0] > 0:
                    hi = mid
                else:
                    lo = mid
            _x, oo, dd = stop_x(0.5 * (lo + hi))
            res = lg.trace(lens, steps, oo, dd)
            xs.append(float(res.film_xy[0, 0]) if res.alive[0] else np.nan)
        xs = np.array(xs)
        par = np.array([lg.paraxial_chief_height(lens, steps, math.tan(math.radians(t))) for t in thetas])
        (line,) = ax.plot(thetas, xs, "o", ms=4, label=f"ghost {pair}")
        ax.plot(thetas, par, "-", color=line.get_color(), lw=0.8)
        small = thetas <= 3.0
        dev = np.nanmax(np.abs(xs[small] - par[small])) if np.any(np.isfinite(xs[small])) else 0.0
        max_dev = max(max_dev, float(dev))
    metrics["ghost_chief_ray_vs_paraxial_max_dev_mm_theta_le_3deg"] = max_dev
    ax.set(
        xlabel="source field angle [deg]",
        ylabel="ghost chief-ray film height [mm]",
        title="(b) ghost position: exact trace (o) vs paraxial (-)",
    )
    ax.legend(fontsize=7)

    # energy table
    ax = fig.add_subplot(gs[0, 2])
    for name in COATINGS:
        lz = lg.load_lens(
            args.lens_file,
            aperture_diameter_mm=args.aperture_diameter_mm,
            focus_distance_m=args.focus_distance_m,
            coating=name,
        )
        tot, tp = [], []
        for th in thetas:
            r = lg.ghost_energy_fractions(lz, lg.direction_from_angles(math.radians(th)), [550.0], n_grid=48)
            tot.append(sum(v["energy"][0] for v in r["ghosts"].values()))
            tp.append(r["primary_transmittance"][0])
        ax.semilogy(thetas, tot, label=f"{name}: sum of 66 ghosts")
        metrics[f"ghost_total_fraction_{name}_550nm_10deg"] = float(np.interp(10.0, thetas, tot))
        metrics[f"primary_T_{name}_550nm_10deg"] = float(np.interp(10.0, thetas, tp))
    ax.set(
        xlabel="source field angle [deg]",
        ylabel="ghost energy / primary (550 nm)",
        title="(c) total two-reflection ghost energy",
    )
    ax.legend(fontsize=8)

    # (d-f) scene
    base = _signal(args.base_exr)
    vmax = np.log10(base.max())
    vmin = vmax - 7.0
    for k, name in enumerate(COATINGS):
        ax = fig.add_subplot(gs[1, k])
        if args.ghost_exr_pattern:
            img = _signal(Path(args.ghost_exr_pattern.format(coating=name)))
            ghost = img - base
            metrics[f"scene_ghost_to_image_energy_{name}"] = float(ghost.sum() / base.sum())
            dark = base < 1e-6 * base.max()
            metrics[f"scene_dark_field_mean_lift_rel_peak_{name}"] = (
                float(ghost[dark].mean() / base.max()) if dark.any() else 0.0
            )
        else:
            img = base
        ax.imshow(np.log10(np.maximum(img, 10**vmin)), cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(f"({'def'[k]}) {name}: log10 radiance, 7 decades")
        ax.set_axis_off()
    fig.suptitle(
        f"Traced lens ghosts - {args.lens_file.name}, D={args.aperture_diameter_mm} mm stop, pbrt RealisticCamera render"
    )
    fig.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=110)
    if args.out_json:
        args.out_json.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
