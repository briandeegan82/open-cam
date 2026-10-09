#!/usr/bin/env python3
"""Validate the HDR pixel models: analytic SNR(signal) vs Monte Carlo, SNR-vs-illuminance figure.

For every recipe the merged HDR signal is simulated with :func:`hdr_pixel.simulate_captures`
+ :func:`hdr_pixel.merge_captures` (the code path used by apply_emva_noise.py) on N uniform
pixels per signal level.  Temporal variance is estimated EMVA-style from two frames with the
same fixed-pattern maps, var_t = var(A - B) / 2; total variance (incl. PRNU/DSNU, EMVA 1288
4.0 Linear Sec. 8.5) from a single frame.  Both are compared with :func:`hdr_pixel.theory_snr`
outside +-4 sigma bands around each transition threshold (where the deterministic-switching
theory does not apply); dip levels just outside each band are always included.

Illuminance axis: sensor-plane illuminance (lux) of a D65 spectrum, converted with the
recipe's green-channel QE (x IRCF), pixel pitch, fill factor and integration time.

Example::

    venv/bin/python tools/validate_hdr_model.py --figure out/hdr_validation/hdr_snr_vs_illuminance.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import hdr_pixel as hp
import numpy as np
from camera_model import load_camera_model, noise_config_from_camera_model
from colour_science import CMF_WAVELENGTH_NM, CMF_Y
from emva_theory import temporal_variance_electrons_squared
from qe_curves import read_csv_curve

H_PLANCK = 6.62607015e-34
C_LIGHT = 299792458.0
REPO = Path(__file__).resolve().parent.parent
DEFAULT_RECIPES = ("default_hdr_dcg", "default_hdr_split_pixel", "default_hdr_lofic", "default_hdr_3exp")


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def electrons_per_lux(cfg: dict, repo: Path = REPO) -> float:
    """Reference-photodiode electrons per lux (sensor plane, D65, green channel) per integration."""
    sensor = cfg["sensor"]
    qe = sensor.get("quantum_efficiency") or {}
    wl = np.arange(380.0, 781.0, 1.0)
    d65 = np.interp(wl, *read_csv_curve(repo / "spectra/illuminant/interpolated/D65.csv"))
    q = np.interp(wl, *read_csv_curve(repo / qe.get("green_csv", "spectra/QE/interpolated/QE_green.csv")))
    if qe.get("ircf_csv"):
        q = q * np.interp(wl, *read_csv_curve(repo / qe["ircf_csv"]))
    v = np.interp(wl, np.asarray(CMF_WAVELENGTH_NM, float), np.asarray(CMF_Y, float))
    k = 1.0 / (683.0 * _trapz(d65 * v, wl))  # W m^-2 nm^-1 per unit D65 at 1 lux
    photo_e = _trapz(k * d65 * q * wl * 1e-9 / (H_PLANCK * C_LIGHT), wl)  # e- s^-1 m^-2 lux^-1
    area = (float(sensor.get("pixel_pitch_um", 1.4)) * 1e-6) ** 2 * float(sensor.get("fill_factor", 1.0))
    return photo_e * area * float(sensor.get("integration_time_s", 0.01))


def transition_bands(arch: hp.HdrArchitecture, n_sigma: float = 4.0) -> list[tuple[float, float, float]]:
    """(transition, lo, hi) in reference electrons for each readout's drop-out threshold."""
    out = []
    for r in arch.readouts:
        c = arch.collector(r.collector)
        thr = arch.threshold_fraction * arch.saturation_e(r)
        sig = math.sqrt(thr + r.sigma_e**2 + r.K_e_per_DN**2 / 12.0)
        out.append(
            (
                (thr - arch.readout_dark_e(r)) / c.response,
                *((thr + s * n_sigma * sig - arch.readout_dark_e(r)) / c.response for s in (-1, 1)),
            )
        )
    return out


def monte_carlo(arch: hp.HdrArchitecture, mu: float, n: int, seed: int) -> dict:
    sig = np.full(n, float(mu))
    frames = []
    for f in (1, 2):
        dn = hp.simulate_captures(
            arch, sig, np.random.default_rng([seed, f]), spatial_rng=np.random.default_rng([seed, 99])
        )
        e, _ = hp.merge_captures(arch, dn)
        frames.append(hp.compander_roundtrip_e(arch, e))
    a, b = frames
    var_t = float(np.var(a - b, ddof=1) / 2.0)
    var_tot = float(np.var(a, ddof=1))
    return {
        "mean_e": float(0.5 * (a.mean() + b.mean())),
        "snr_temporal": mu / math.sqrt(var_t) if var_t > 0 else math.inf,
        "snr_total": mu / math.sqrt(var_tot) if var_tot > 0 else math.inf,
        "sem_e": math.sqrt(var_tot / n),
    }


def validate_recipe(recipe: str, *, trials: int, seed: int, levels: int = 24) -> dict:
    cm = load_camera_model(REPO / "config" / "camera_recipes" / f"{recipe}.yaml")
    cfg = noise_config_from_camera_model(cm, "", "")
    base = hp.base_from_noise_config(cfg)
    arch = hp.build_architecture(cfg["hdr"], base)
    bands = transition_bands(arch)
    mu_max = arch.max_reference_e
    grid = list(np.geomspace(3.0, 0.97 * mu_max, levels))
    dips = []
    for t, lo, hi in bands[:-1]:
        below, above = lo * 0.98, hi * 1.02
        grid += [below, above]
        dips.append((t, below, above))
    grid = sorted(g for g in grid if 0 < g < mu_max)
    in_band = [any(lo <= g <= hi for _, lo, hi in bands) for g in grid]
    th_t = hp.theory_snr(arch, np.array(grid))["snr"]
    th_s = hp.theory_snr(arch, np.array(grid), include_spatial=True)["snr"]
    pts = []
    for i, g in enumerate(grid):
        mc = monte_carlo(arch, g, trials, seed + i)
        pts.append(
            {
                "mu_e": g,
                "in_transition_band": in_band[i],
                "theory_snr_temporal": float(th_t[i]),
                "theory_snr_total": float(th_s[i]),
                **mc,
            }
        )
    ok = [p for p in pts if not p["in_transition_band"]]
    rel_t = [abs(p["snr_temporal"] / p["theory_snr_temporal"] - 1) for p in ok]
    rel_s = [abs(p["snr_total"] / p["theory_snr_total"] - 1) for p in ok]
    bias = [abs(p["mean_e"] - p["mu_e"]) / max(p["sem_e"], 1e-12) for p in ok]
    dip_rows = []
    for t, below, above in dips:
        tb, ta = hp.theory_snr(arch, np.array([below, above]))["snr"]
        mb = next(p for p in pts if p["mu_e"] == below)
        ma = next(p for p in pts if p["mu_e"] == above)
        dip_rows.append(
            {
                "transition_e": t,
                "theory_dip_dB": float(20 * math.log10(tb / ta)),
                "mc_dip_dB": float(20 * math.log10(mb["snr_temporal"] / ma["snr_temporal"])),
            }
        )
    return {
        "recipe": recipe,
        "architecture": arch.name,
        "dynamic_range_dB": hp.dynamic_range_db(arch),
        "max_reference_e": mu_max,
        "electrons_per_lux": electrons_per_lux(cfg),
        "transitions_e": [b[0] for b in bands],
        "n_levels_compared": len(ok),
        "max_rel_err_snr_temporal": max(rel_t),
        "max_rel_err_snr_total": max(rel_s),
        "max_mean_bias_sem": max(bias),
        "dips": dip_rows,
        "points": pts,
        "_arch": arch,
        "_base": base,
    }


def plot(results: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    cols = 2 if n > 1 else 1
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 4.2 * rows), squeeze=False)
    for ax, r in zip(axes.ravel(), results):
        arch, base, epl = r["_arch"], r["_base"], r["electrons_per_lux"]
        mu = np.geomspace(1.0, arch.max_reference_e * 0.999, 1500)
        db = lambda s: 20 * np.log10(np.asarray(s, float))  # noqa: E731
        ax.plot(mu / epl, db(hp.theory_snr(arch, mu)["snr"]), "C0-", lw=1.6, label="theory, temporal")
        ax.plot(
            mu / epl,
            db(hp.theory_snr(arch, mu, include_spatial=True)["snr"]),
            "C1--",
            lw=1.2,
            label="theory, total (PRNU/DSNU)",
        )
        dark = base["dark_current_e_per_s"] * base["t_int_s"] * base["dark_temp_scale"]
        m1 = np.geomspace(1.0, 0.9 * base["full_well_e"], 400)
        v1 = [temporal_variance_electrons_squared(m, base["sigma_d_e"], use_poisson=True, mu_dark_e=dark) for m in m1]
        s1 = m1 / np.sqrt(np.asarray(v1) + base["K_e_per_DN"] ** 2 / 12.0)
        ax.plot(m1 / epl, db(s1), color="0.6", lw=1.0, ls=":", label="single capture (base model)")
        p = r["points"]
        x = np.array([q["mu_e"] for q in p]) / epl
        ax.plot(x, db([q["snr_temporal"] for q in p]), "o", mfc="none", color="C0", ms=5, label="Monte Carlo, temporal")
        ax.plot(x, db([q["snr_total"] for q in p]), "x", color="C1", ms=5, label="Monte Carlo, total")
        for t in r["transitions_e"][:-1]:
            ax.axvline(t / epl, color="0.8", lw=0.8, zorder=0)
        ax.set_xscale("log")
        ax.set_xlabel("sensor-plane illuminance [lux, D65]  (t_int = %.0f ms)" % (base["t_int_s"] * 1e3))
        ax.set_ylabel("SNR [dB]")
        ax.set_title(f"{r['recipe']}: {arch.name}, DR {r['dynamic_range_dB']:.1f} dB")
        ax.set_ylim(0, None)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7, loc="upper left")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipes", nargs="+", default=list(DEFAULT_RECIPES))
    ap.add_argument("--trials", type=int, default=20000, help="Pixels per signal level.")
    ap.add_argument("--levels", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snr-rtol", type=float, default=0.03, help="Max relative SNR error outside transition bands.")
    ap.add_argument("--max-bias-sem", type=float, default=5.0, help="Max |mean - mu| in standard errors.")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--figure", type=Path, default=None)
    args = ap.parse_args(argv)
    results = [validate_recipe(r, trials=args.trials, seed=args.seed, levels=args.levels) for r in args.recipes]
    failed = False
    for r in results:
        bad = r["max_rel_err_snr_temporal"] > args.snr_rtol or r["max_rel_err_snr_total"] > args.snr_rtol
        bad |= r["max_mean_bias_sem"] > args.max_bias_sem
        failed |= bad
        dips = ", ".join(f"{d['theory_dip_dB']:.1f}/{d['mc_dip_dB']:.1f} dB" for d in r["dips"])
        print(
            f"{'FAIL' if bad else 'PASS'} {r['recipe']:<26} DR={r['dynamic_range_dB']:.1f} dB  "
            f"max|dSNR/SNR| temporal={r['max_rel_err_snr_temporal']:.4f} total={r['max_rel_err_snr_total']:.4f}  "
            f"bias={r['max_mean_bias_sem']:.2f} SEM  dips theory/MC: {dips}  ({r['n_levels_compared']} levels)"
        )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
        args.json_out.write_text(json.dumps(clean, indent=2) + "\n")
    if args.figure:
        plot(results, args.figure)
        print(f"wrote {args.figure}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
