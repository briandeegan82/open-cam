#!/usr/bin/env python3
"""Validation figure for the CRA / microlens angular response (tools/pixel_angular_response.py).

Runs the real pipeline entry point (``spectral_radiance_to_electrons``) on a flat D65 spectral
radiance field with the default sensor QE curves and pixel pitch, traced through
``config/lenses/wide_22mm.dat`` on a pbrt film of the given diagonal. It plots, against
normalised image height along the frame diagonal:

  (a) the traced chief-ray angle, the paraxial exit-pupil CRA, the energy-centroid CRA and the
      traced relative illumination (pupil vignetting × cos⁴), with cos⁴ of the CRA for reference;
  (b) the pixel angular response of an unshifted pixel per channel vs incidence angle
      (collimated light), closed form vs an independent Monte-Carlo ray model;
  (c) luminance (G) shading and (d) R/G, B/G colour shading for several microlens-shift designs.

Shading in (c)/(d) is the angular response only: the input is flat radiance, so natural
vignetting, which pbrt's realistic camera applies during the render, is not included.

Usage: venv/bin/python tools/validate_pixel_angular_response.py --out out/cra_validation
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tests"))

import pixel_angular_response as par  # noqa: E402
from pbrt_spectral_exr_to_electrons import qe_stack_on_lambdas, spectral_radiance_to_electrons  # noqa: E402
from qe_curves import read_csv_curve  # noqa: E402

DESIGNS = [
    ("matched (designed for this lens)", {"mode": "matched"}),
    ("linear, max CRA 15° (under-shifted)", {"mode": "linear", "max_cra_deg": 15.0}),
    ("linear, max CRA 40° (over-shifted)", {"mode": "linear", "max_cra_deg": 40.0}),
    ("no microlens shift", {"mode": "none"}),
]


def _diag(maps_like: np.ndarray, n: int) -> np.ndarray:
    idx = np.arange(n)
    return maps_like[idx, idx]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "out/cra_validation")
    ap.add_argument("--lensfile", default="config/lenses/wide_22mm.dat")
    ap.add_argument("--film-diagonal-mm", type=float, default=25.0)
    ap.add_argument("--res", type=int, default=96, help="Square frame size in pixels.")
    ap.add_argument("--grid", type=int, default=15, help="Field grid points per axis (odd).")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import yaml
    from test_pixel_angular_response import _monte_carlo_response

    sensor_yaml = yaml.safe_load((REPO / "config/sensor_models/default.yaml").read_text())
    sensor = sensor_yaml["sensor"]
    model = dict(sensor_yaml["sensor_forward"]["model"])
    pitch = float(sensor["pixel_pitch_um"])
    lens_cfg = {"camera": "realistic", "realistic_lensfile": args.lensfile}
    geometry = {"lensfile": args.lensfile, "film_diagonal_mm": args.film_diagonal_mm}
    lam = np.arange(400.0, 701.0, 10.0)
    d65_wl, d65 = read_csv_curve(REPO / "spectra/illuminant/interpolated/D65.csv")
    L = np.broadcast_to(np.interp(lam, d65_wl, d65)[None, None, :] * 1e-3, (args.res, args.res, lam.size)).astype(
        np.float32
    )
    scene = {"camera_geometry": geometry}
    e_off, meta_off = spectral_radiance_to_electrons(
        L, lam, repo=REPO, sensor=sensor, model=model, lens_cfg=lens_cfg, scene=scene
    )
    f_number = float(meta_off["f_number"])

    n = args.res
    ii = np.arange(n)
    h_px = np.hypot(ii + 0.5 - n / 2, ii + 0.5 - n / 2) / math.hypot(n / 2 - 0.5, n / 2 - 0.5)
    half = slice(n // 2, n)
    results: dict = {
        "lensfile": args.lensfile,
        "film_diagonal_mm": args.film_diagonal_mm,
        "pixel_pitch_um": pitch,
        "f_number": f_number,
    }

    # (a) incidence geometry along +x+y diagonal
    tl = par.TracedLens.from_file(REPO / args.lensfile)
    xp = par.paraxial_exit_pupil_distance_mm(tl.rows)
    r_corner = 0.5 * args.film_diagonal_mm
    hs = np.linspace(0.0, r_corner, 26)
    cra_traced, cra_cent, ri = [], [], []
    psa0 = tl.incidence_directions(0.0, 0.0, 48)[2].sum()
    for h in hs:
        cra_traced.append(tl.chief_ray_angle_deg(h))
        a, b, w = tl.incidence_directions(h, 0.0, 48)
        cra_cent.append(math.degrees(math.asin(abs(float(np.average(a, weights=w))))))
        ri.append(float(w.sum() / psa0))
    cra_parax = np.degrees(np.arctan(hs / xp))
    results["chief_ray"] = {
        "image_height_mm": hs.round(3).tolist(),
        "traced_deg": np.round(cra_traced, 3).tolist(),
        "paraxial_exit_pupil_deg": cra_parax.round(3).tolist(),
        "centroid_deg": np.round(cra_cent, 3).tolist(),
        "relative_illumination": np.round(ri, 4).tolist(),
        "exit_pupil_distance_mm": xp,
    }

    # (b) angular response vs incidence angle, closed form vs Monte Carlo
    st = par.PixelStack.from_config(None, pitch)
    thetas = np.linspace(0.0, 40.0, 41)
    ref = par.normal_incidence_response(lam, st, repo=REPO)
    qe = qe_stack_on_lambdas(REPO, sensor["quantum_efficiency"], lam).T  # [K, 3]
    d65_l = np.interp(lam, d65_wl, d65)
    weights = qe * (d65_l * lam)[:, None]
    ang_cf = []
    for t in thetas:
        r = par.pixel_response(
            np.array([math.sin(math.radians(t))]), np.zeros(1), np.ones(1), (0.0, 0.0), lam, st, repo=REPO
        )
        ang_cf.append((weights * (r[1, 1] / ref)[:, None]).sum(0) / weights.sum(0))
    ang_cf = np.array(ang_cf)
    mc_thetas = [0.0, 10.0, 20.0, 30.0]
    mc_lams = {"B 450 nm": 450.0, "G 540 nm": 540.0, "R 620 nm": 620.0}
    mc = {}
    for name, lmb in mc_lams.items():
        r_ref = par.normal_incidence_response(np.array([lmb]), st, repo=REPO)[0]
        cf = [
            float(
                par.pixel_response(
                    np.array([math.sin(math.radians(t))]),
                    np.zeros(1),
                    np.ones(1),
                    (0.0, 0.0),
                    np.array([lmb]),
                    st,
                    repo=REPO,
                )[1, 1, 0]
            )
            / r_ref
            for t in thetas
        ]
        mc_ref = _monte_carlo_response(0.0, lmb, st)[0]
        mcv = [_monte_carlo_response(t, lmb, st)[0] / mc_ref for t in mc_thetas]
        cf_at = [float(np.interp(t, thetas, cf)) for t in mc_thetas]
        mc[name] = {
            "closed_form": cf,
            "mc": mcv,
            "max_abs_diff": float(np.max(np.abs(np.array(mcv) - np.array(cf_at)))),
        }
    results["angular_response"] = {
        "theta_deg": thetas.tolist(),
        "channel_rgb_d65": ang_cf.round(4).tolist(),
        "half_response_angle_deg": {
            c: float(np.interp(0.5, ang_cf[::-1, i], thetas[::-1])) for i, c in enumerate("RGB")
        },
        "mc_check": {
            k: {
                "theta_deg": mc_thetas,
                "mc": np.round(v["mc"], 4).tolist(),
                "max_abs_diff": round(v["max_abs_diff"], 4),
            }
            for k, v in mc.items()
        },
    }

    # (c, d) end-to-end shading along the diagonal
    shading = {}
    for label, shift in DESIGNS:
        cfg = {"enabled": True, "incidence": "traced", "microlens_shift": shift, "field_grid": [args.grid, args.grid]}
        e_on, meta = spectral_radiance_to_electrons(
            L,
            lam,
            repo=REPO,
            sensor=sensor,
            model={**model, "pixel_angular_response": cfg},
            lens_cfg=lens_cfg,
            scene=scene,
        )
        rel = _diag(e_on / e_off, n)
        shading[label] = {
            "G": rel[:, 1],
            "R/G": rel[:, 0] / rel[:, 1],
            "B/G": rel[:, 2] / rel[:, 1],
            "meta": meta["pixel_angular_response"],
        }
    c0 = n // 2
    summary = {}
    for label, s in shading.items():
        corner = 0
        summary[label] = {
            "G_center": round(float(s["G"][c0]), 4),
            "G_corner": round(float(s["G"][corner]), 4),
            "G_corner_over_center": round(float(s["G"][corner] / s["G"][c0]), 4),
            "R/G_corner_over_center": round(float(s["R/G"][corner] / s["R/G"][c0]), 4),
            "B/G_corner_over_center": round(float(s["B/G"][corner] / s["B/G"][c0]), 4),
        }
    results["shading"] = summary
    results["max_chief_ray_deg"] = float(max(cra_traced))

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    ax = axs[0, 0]
    hn = hs / r_corner
    ax.plot(hn, cra_traced, "k-", label="traced chief ray (stop centre)")
    ax.plot(hn, cra_parax, "k:", label=f"paraxial exit pupil (XP = {xp:.1f} mm)")
    ax.plot(hn, cra_cent, "C1--", label="energy-centroid CRA (traced pupil)")
    ax.set_xlabel("normalised image height")
    ax.set_ylabel("CRA [deg]")
    ax2 = ax.twinx()
    ax2.plot(hn, ri, "C0-", label="traced relative illumination")
    ax2.plot(hn, np.cos(np.radians(cra_traced)) ** 4, "C0:", label="cos⁴(CRA)")
    ax2.set_ylabel("relative illumination", color="C0")
    ax2.set_ylim(0, 1.05)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center left")
    ax.set_title(f"(a) {Path(args.lensfile).name}, film diagonal {args.film_diagonal_mm:g} mm, f/{f_number:.2f}")

    ax = axs[0, 1]
    for i, c in enumerate("RGB"):
        ax.plot(thetas, ang_cf[:, i], color={"R": "r", "G": "g", "B": "b"}[c], label=f"{c} channel (D65, closed form)")
    for name, v in mc.items():
        col = {"B": "b", "G": "g", "R": "r"}[name[0]]
        ax.plot(mc_thetas, v["mc"], "o", mfc="none", color=col, label=f"Monte-Carlo rays, {name}")
    ax.set_xlabel("incidence angle in air [deg] (collimated, unshifted pixel)")
    ax.set_ylabel("relative QE")
    ax.set_title(f"(b) pixel angular response, pitch {pitch:g} µm, default stack")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    hd = h_px[half]
    for k, (label, s) in enumerate(shading.items()):
        axs[1, 0].plot(hd, s["G"][half], color=f"C{k}", label=label)
        axs[1, 1].plot(hd, s["R/G"][half], color=f"C{k}", ls="-", label=f"R/G {label}")
        axs[1, 1].plot(hd, s["B/G"][half], color=f"C{k}", ls="--", label=f"B/G {label}")
    axs[1, 0].set_title("(c) luminance (G) shading from pixel angular response")
    axs[1, 0].set_ylabel("G electrons / G without angular response")
    axs[1, 1].set_title("(d) colour shading (solid R/G, dashed B/G)")
    axs[1, 1].set_ylabel("ratio relative to no angular response")
    for ax in axs[1]:
        ax.set_xlabel("normalised image height (diagonal)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    png = args.out / "cra_shading_validation.png"
    fig.savefig(png, dpi=120)
    (args.out / "cra_shading_validation.json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps({k: results[k] for k in ("f_number", "max_chief_ray_deg", "shading")}, indent=2))
    print(
        json.dumps(results["angular_response"]["half_response_angle_deg"]),
        json.dumps(results["angular_response"]["mc_check"]),
    )
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
