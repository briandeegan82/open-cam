#!/usr/bin/env python3
"""Ground-truth validation of the ``tools/iqlab`` metrics: bias and spread against analytic answers.

Each suite builds synthetic inputs whose true answer is known in closed form, runs the metric over
``--trials`` noise realisations and reports the truth, the mean measured value, the bias and the
standard deviation. ``TOLERANCES`` sets the pass limits shared with ``tests/test_iqlab_ground_truth.py``.

* ``edge``: slanted edges with a Gaussian PSF (sigma) times the pixel box aperture, area-sampled.
  Truth: ``MTF(f) = exp(-2 pi^2 sigma^2 f^2) |sinc f|``; MTF50 from a root find; CPIQ acutance of the
  true MTF. Measured: ``sfr_analysis.slanted_edge_sfr`` -> MTF50, MTF RMS error (0..Nyquist), acutance.
* ``texture``: dead-leaves target with Gaussian blur and white noise, noise PSD from an independent
  flat patch. Truth: Gaussian MTF. Measured: ``dead_leaves.texture_mtf`` RMS error, texture acutance.
* ``snr``: Poisson + read-noise flats (and PRNU for total noise). Truth: ``S / sqrt(S + r^2 [+ (pS)^2])``.
  Measured: ``snr.patch_stats`` (single frame) and ``snr.temporal_patch_stats`` (frame stack).
* ``dr``: SNR = 1 and SNR = 10 dynamic range from a clipped patch ladder vs the analytic thresholds.
* ``cdp``: Gaussian dark/bright patches. Truth: CDP by 1-D quadrature of the pair-contrast distribution.
* ``colour``: patches with a known L*C*h shift from a reference, rendered as noisy linear sRGB at an
  arbitrary exposure and scored by ``run_skin_tone_test.score`` (white-patch normalised). Truth: the
  injected Delta L*, Delta C*, Delta h and the resulting CIEDE2000.

    venv/bin/python tools/validate_iqlab_metrics.py --out-dir out/iq_lab/validation --figure
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import sfr_analysis as sfr
from colour_science import WHITE_D65, delta_e_2000, xyz_to_srgb_linear
from iqlab import cpiq, dead_leaves, p2020, snr
from iqlab.skin import lab_to_xyz
from run_skin_tone_test import score as score_chart
from scipy import ndimage, optimize
from scipy.stats import norm

#: Pass limits per suite and noise tier ("clean": noise-free or edge SNR >= 100; "noisy": the noisiest
#: case). ``*_pct`` are relative errors in %, ``*_abs`` absolute. ``bias`` limits apply to the mean over
#: trials, ``rms`` / ``*_abs`` limits to sqrt(bias^2 + std^2) (colour: max |bias| + max std over patches).
TOLERANCES = {
    "edge": {
        "clean": {"mtf50_bias_pct": 3.5, "mtf50_rms_pct": 4.0, "mtf_rms_abs": 0.02, "acutance_abs": 0.015},
        "noisy": {"mtf50_bias_pct": 7.0, "mtf50_rms_pct": 8.0, "mtf_rms_abs": 0.05, "acutance_abs": 0.02},
    },
    "texture": {
        "clean": {"mtf_rms_abs": 0.005, "acutance_abs": 0.005},
        "noisy": {"mtf_rms_abs": 0.06, "acutance_abs": 0.03},
    },
    "snr": {"all": {"snr_bias_pct": 1.0, "snr_rms_pct": 3.0}},
    "dr": {"all": {"db_abs": 0.4}},
    "cdp": {"all": {"abs": 0.006}},
    "colour": {
        "clean": {"de00_abs": 1e-6, "dL_abs": 1e-6, "dC_abs": 1e-6, "dh_deg_abs": 1e-6},
        "noisy": {"de00_abs": 0.3, "dL_abs": 0.15, "dC_abs": 0.35, "dh_deg_abs": 1.0},
    },
}


def limit_keys(name: str) -> list[str]:
    return list(next(iter(TOLERANCES[name].values())))


VIEW_HEIGHT_PX = 1080


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def _summary(truth: float, values: list[float]) -> dict:
    v = np.asarray(values, dtype=np.float64)
    bias = float(np.mean(v) - truth)
    return {
        "truth": float(truth),
        "mean": float(np.mean(v)),
        "bias": bias,
        "std": float(np.std(v, ddof=1) if v.size > 1 else 0.0),
    }


# --------------------------------------------------------------------------------------------------
# Edge SFR
# --------------------------------------------------------------------------------------------------
def true_edge_mtf(f: np.ndarray, sigma_px: float) -> np.ndarray:
    return sfr.gaussian_mtf(f, sigma_px) * sfr.pixel_aperture_mtf(f)


def true_edge_mtf50(sigma_px: float) -> float:
    return float(optimize.brentq(lambda f: true_edge_mtf(np.array([f]), sigma_px)[0] - 0.5, 1e-4, 1.5))


def blurred_slanted_edge(sigma_px: float, angle_deg: float, size: int = 100, supersample: int = 8) -> np.ndarray:
    """Edge * Gaussian PSF on a fine grid, then box-averaged: PSF (x) pixel aperture, exactly sampled."""
    s = supersample
    yy, xx = np.mgrid[0 : size * s, 0 : size * s].astype(np.float64)
    x, y = (xx + 0.5) / s, (yy + 0.5) / s
    edge_x = size / 2.0 + (y - size / 2.0) * math.tan(math.radians(angle_deg))
    fine = np.where(x < edge_x, 0.95, 0.05)
    fine = ndimage.gaussian_filter(fine, sigma_px * s, mode="nearest")
    return fine.reshape(size, s, size, s).mean(axis=(1, 3))


def edge_suite(
    sigmas=(0.5, 0.8, 1.2, 2.0), angles=(3.0, 5.0, 8.0), snrs=(math.inf, 100.0, 30.0), trials: int = 10, seed: int = 0
) -> list[dict]:
    rng = np.random.default_rng(seed)
    view = cpiq.viewing_condition("monitor_100pct", VIEW_HEIGHT_PX)
    f_fine = np.linspace(0, 0.5, 501)
    rows = []
    for sigma in sigmas:
        mtf50_true = true_edge_mtf50(sigma)
        acu_true = cpiq.acutance(f_fine, true_edge_mtf(f_fine, sigma), view)
        for angle in angles:
            clean = blurred_slanted_edge(sigma, angle)
            for edge_snr in snrs:
                n = 1 if math.isinf(edge_snr) else trials
                m50, err, acu = [], [], []
                for _ in range(n):
                    img = clean if math.isinf(edge_snr) else clean + rng.normal(0, 0.9 / edge_snr, clean.shape)
                    r = sfr.slanted_edge_sfr(img)
                    band = r.frequency_cy_per_px <= 0.5
                    f, m = r.frequency_cy_per_px[band], r.mtf[band]
                    m50.append(r.mtf50_cy_per_px)
                    err.append(_rms(m - true_edge_mtf(f, sigma)))
                    acu.append(cpiq.acutance(f, m, view))
                s50, sa = _summary(mtf50_true, m50), _summary(acu_true, acu)
                rows.append(
                    {
                        "sigma_px": sigma,
                        "angle_deg": angle,
                        "edge_snr": None if math.isinf(edge_snr) else edge_snr,
                        "tier": "clean" if edge_snr >= 100 else "noisy",
                        "trials": n,
                        "mtf50": s50,
                        "mtf50_bias_pct": 100 * s50["bias"] / mtf50_true,
                        "mtf50_rms_pct": 100 * math.hypot(s50["bias"], s50["std"]) / mtf50_true,
                        "mtf_rms_abs": float(np.mean(err)),
                        "acutance": sa,
                        "acutance_abs": math.hypot(sa["bias"], sa["std"]),
                    }
                )
    return rows


# --------------------------------------------------------------------------------------------------
# Dead-leaves texture MTF
# --------------------------------------------------------------------------------------------------
def texture_suite(
    sigmas=(0.8, 1.5), noise_sigmas=(0.0, 0.01, 0.03), trials: int = 4, size: int = 256, seed: int = 0
) -> list[dict]:
    rng = np.random.default_rng(seed)
    view = cpiq.viewing_condition("monitor_100pct", VIEW_HEIGHT_PX)
    rows = []
    for sigma in sigmas:
        for sn in noise_sigmas:
            err, acu, acu_true = [], [], []
            for t in range(trials):
                ideal = dead_leaves.dead_leaves(size, r_min=1.0, r_max=60.0, seed=seed + t)
                captured = ndimage.gaussian_filter(ideal, sigma, mode="wrap")
                flat = None
                if sn > 0:
                    captured = captured + rng.normal(0, sn, ideal.shape)
                    flat = rng.normal(0, sn, ideal.shape)
                f, m = dead_leaves.texture_mtf(captured, ideal, noise_patch=flat)
                truth = sfr.gaussian_mtf(f, sigma)
                band = (f > 0.02) & (f < 0.3)
                err.append(_rms(m[band] - truth[band]))
                acu.append(cpiq.acutance(f, m, view))
                acu_true.append(cpiq.acutance(f, truth, view))
            sa = _summary(float(np.mean(acu_true)), acu)
            rows.append(
                {
                    "sigma_px": sigma,
                    "noise_sigma": sn,
                    "tier": "clean" if sn == 0 else "noisy",
                    "trials": trials,
                    "mtf_rms_abs": float(np.mean(err)),
                    "acutance": sa,
                    "acutance_abs": math.hypot(sa["bias"], sa["std"]),
                }
            )
    return rows


# --------------------------------------------------------------------------------------------------
# SNR and dynamic range
# --------------------------------------------------------------------------------------------------
def snr_suite(
    levels_e=(5.0, 50.0, 500.0, 5000.0), read_e: float = 3.0, prnu: float = 0.01, trials: int = 10, seed: int = 0
) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    shape = (64, 64)
    for s in levels_e:
        truth_t = float(snr.shot_read_snr(np.array([s]), read_e)[0])
        truth_total = s / math.sqrt(s + read_e**2 + (prnu * s) ** 2)
        single, temporal, total = [], [], []
        for _ in range(trials):
            gain = 1.0 + rng.normal(0, prnu, shape)
            frames = rng.poisson(s * gain, (8, *shape)) + rng.normal(0, read_e, (8, *shape))
            single.append(snr.patch_stats(rng.poisson(s, shape) + rng.normal(0, read_e, shape)).snr)
            temporal.append(snr.temporal_patch_stats(frames).snr)
            total.append(snr.patch_stats(frames[0]).snr)
        for kind, truth, vals in (
            ("single_frame", truth_t, single),
            ("temporal_8frames_prnu", truth_t, temporal),
            ("total_with_prnu", truth_total, total),
        ):
            st = _summary(truth, vals)
            rows.append(
                {
                    "signal_e": s,
                    "kind": kind,
                    "trials": trials,
                    "snr": st,
                    "snr_bias_pct": 100 * st["bias"] / truth,
                    "snr_rms_pct": 100 * math.hypot(st["bias"], st["std"]) / truth,
                }
            )
    return rows


def _snr_threshold_analytic(threshold: float, read_e: float) -> float:
    # S^2 = T^2 (S + r^2) -> S = (T^2 + sqrt(T^4 + 4 T^2 r^2)) / 2
    t2 = threshold**2
    return (t2 + math.sqrt(t2 * t2 + 4 * t2 * read_e**2)) / 2


def dr_suite(
    read_noises=(1.5, 3.0, 10.0), full_well: float = 20_000.0, n_levels: int = 41, trials: int = 5, seed: int = 0
) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    levels = np.geomspace(0.5, 2 * full_well, n_levels)
    top = float(levels[levels < full_well * 0.9].max())
    for read in read_noises:
        for thr in (1.0, 10.0):
            truth_db = 20 * math.log10(top / _snr_threshold_analytic(thr, read))
            vals = []
            for _ in range(trials):
                stats = [
                    snr.patch_stats(
                        np.minimum(rng.poisson(s, (48, 48)) + rng.normal(0, read, (48, 48)), full_well),
                        saturation=full_well,
                    )
                    for s in levels
                ]
                dr = snr.dynamic_range(
                    np.array([p.mean for p in stats]),
                    np.array([p.snr for p in stats]),
                    snr_threshold=thr,
                    saturated_fraction=np.array([p.saturated_fraction for p in stats]),
                )
                vals.append(dr.db)
            st = _summary(truth_db, vals)
            rows.append(
                {
                    "read_noise_e": read,
                    "snr_threshold": thr,
                    "trials": trials,
                    "dr_db": st,
                    "db_abs": math.hypot(st["bias"], st["std"]),
                }
            )
    return rows


# --------------------------------------------------------------------------------------------------
# Contrast detection probability
# --------------------------------------------------------------------------------------------------
def cdp_analytic(mu_d: float, sd_d: float, mu_b: float, sd_b: float, epsilon: float = 0.5) -> float:
    """P(|C - C_nom| <= eps C_nom), C = (b - d)/(b + d), Gaussian d and b, C_nom from the means.

    For d > 0, C is increasing in b, so C in [lo, hi] <=> b in [d (1+lo)/(1-lo), d (1+hi)/(1-hi)]
    (upper bound infinite when hi >= 1). Valid when P(d <= 0) is negligible.
    """
    c_nom = (mu_b - mu_d) / (mu_b + mu_d)
    lo, hi = c_nom * (1 - epsilon), c_nom * (1 + epsilon)
    d = np.linspace(max(mu_d - 8 * sd_d, 1e-9), mu_d + 8 * sd_d, 4001)
    b_lo = d * (1 + lo) / (1 - lo)
    p_hi = norm.cdf((d * (1 + hi) / (1 - hi) - mu_b) / sd_b) if hi < 1 else np.ones_like(d)
    inner = p_hi - norm.cdf((b_lo - mu_b) / sd_b)
    w = norm.pdf((d - mu_d) / sd_d) / sd_d
    return float(np.sum(0.5 * (w[1:] * inner[1:] + w[:-1] * inner[:-1]) * np.diff(d)))


def cdp_suite(
    cases=((100, 5, 120, 5), (100, 10, 120, 10), (100, 20, 150, 25), (50, 10, 60, 10), (1000, 30, 1100, 35)),
    epsilon: float = 0.5,
    n_pixels: int = 20_000,
    trials: int = 5,
    seed: int = 0,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for mu_d, sd_d, mu_b, sd_b in cases:
        truth = cdp_analytic(mu_d, sd_d, mu_b, sd_b, epsilon)
        c_nom = (mu_b - mu_d) / (mu_b + mu_d)
        vals = [
            p2020.contrast_detection_probability(
                rng.normal(mu_d, sd_d, n_pixels),
                rng.normal(mu_b, sd_b, n_pixels),
                nominal_contrast=c_nom,
                epsilon=epsilon,
                seed=int(rng.integers(1 << 31)),
            )
            for _ in range(trials)
        ]
        st = _summary(truth, vals)
        rows.append(
            {
                "dark": [mu_d, sd_d],
                "bright": [mu_b, sd_b],
                "trials": trials,
                "cdp": st,
                "abs": math.hypot(st["bias"], st["std"]),
            }
        )
    return rows


# --------------------------------------------------------------------------------------------------
# Colour: known Delta L*, C*, h through the chart scorer
# --------------------------------------------------------------------------------------------------
REFERENCE_LAB = np.array(
    [
        [37.5, 13.0, 14.5],
        [65.0, 18.0, 17.5],
        [50.0, -5.0, -22.0],
        [43.0, -13.0, 22.0],
        [55.0, 9.0, -25.0],
        [70.0, -32.0, 1.0],
        [62.0, 34.0, 56.0],
        [40.0, 10.0, -42.0],
        [51.0, 46.0, 16.0],
        [72.0, -24.0, 57.0],
        [96.0, 0.0, 0.0],
    ]
)
WHITE_INDEX = len(REFERENCE_LAB) - 1


def shift_lch(lab: np.ndarray, d_l: float, d_c: float, d_h_deg: float) -> np.ndarray:
    c = np.hypot(lab[:, 1], lab[:, 2]) + d_c
    h = np.arctan2(lab[:, 2], lab[:, 1]) + np.radians(d_h_deg)
    return np.stack([lab[:, 0] + d_l, c * np.cos(h), c * np.sin(h)], -1)


def colour_chart(patch: int = 24, gap: int = 8) -> dict:
    patches = []
    for k, lab in enumerate(REFERENCE_LAB):
        x0, y0 = gap + (k % 4) * (patch + gap), gap + (k // 4) * (patch + gap)
        name = "white" if k == WHITE_INDEX else f"skin_like_{k:02d}"
        patches.append(
            {"name": name, "roi_xyxy": [x0, y0, x0 + patch, y0 + patch], "reference_lab": {"D65": lab.tolist()}}
        )
    return {"patches": patches, "white_index": WHITE_INDEX}


def colour_suite(
    shifts=((0, 0, 0), (1.0, 0, 0), (0, 2.0, 0), (0, 0, 3.0), (-2.0, -3.0, 5.0)),
    noise_fracs=(0.0, 0.02),
    exposure: float = 0.37,
    trials: int = 5,
    seed: int = 0,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    chart = colour_chart()
    test_idx = [k for k in range(len(REFERENCE_LAB)) if k != WHITE_INDEX]
    rows = []
    for shift in shifts:
        lab_true = REFERENCE_LAB.copy()
        lab_true[test_idx] = shift_lch(REFERENCE_LAB[test_idx], *shift)
        de_true = delta_e_2000(lab_true[test_idx], REFERENCE_LAB[test_idx])
        rgb = xyz_to_srgb_linear(lab_to_xyz(lab_true, WHITE_D65)) * exposure
        h = max(p["roi_xyxy"][3] for p in chart["patches"]) + 8
        w = max(p["roi_xyxy"][2] for p in chart["patches"]) + 8
        clean = np.zeros((h, w, 3))
        for p, c in zip(chart["patches"], rgb, strict=True):
            x0, y0, x1, y1 = p["roi_xyxy"]
            clean[y0:y1, x0:x1] = c
        for nf in noise_fracs:
            n = 1 if nf == 0 else trials
            errs = {"de00": [], "dL": [], "dC": [], "dh_deg": []}
            for _ in range(n):
                # shot-noise-like: sigma proportional to sqrt(signal), nf at the white patch
                img = clean + rng.normal(0, 1, clean.shape) * nf * np.sqrt(np.maximum(clean, 0) * exposure)
                res = score_chart(chart, img, "D65")
                got = res["patches"]
                errs["de00"].append(np.array([g["delta_e00"] for g in got]) - de_true)
                errs["dL"].append(np.array([g["delta_L"] for g in got]) - shift[0])
                errs["dC"].append(np.array([g["delta_C"] for g in got]) - shift[1])
                errs["dh_deg"].append(np.array([g["delta_hue_deg"] for g in got]) - shift[2])
            row = {
                "shift_LCh": list(shift),
                "noise_frac": nf,
                "tier": "clean" if nf == 0 else "noisy",
                "trials": n,
                "de00_true_mean": float(de_true.mean()),
            }
            for key, v in errs.items():
                row[f"{key}_abs"] = float(np.max(np.abs(np.mean(v, axis=0))) + np.max(np.std(v, axis=0)))
            rows.append(row)
    return rows


# --------------------------------------------------------------------------------------------------
SUITES = {
    "edge": edge_suite,
    "texture": texture_suite,
    "snr": snr_suite,
    "dr": dr_suite,
    "cdp": cdp_suite,
    "colour": colour_suite,
}


def check(name: str, rows: list[dict]) -> list[str]:
    """Return human-readable failures of ``rows`` against ``TOLERANCES[name]``."""
    fails = []
    for r in rows:
        limits = TOLERANCES[name][r.get("tier", "all")]
        for key, lim in limits.items():
            if not abs(r[key]) <= lim:
                ident = {k: v for k, v in r.items() if not isinstance(v, dict) and k not in limits}
                fails.append(f"{name} {ident}: {key}={r[key]:.4g} > {lim}")
    return fails


def _markdown(results: dict) -> str:
    out = ["# IQ lab ground-truth validation", ""]
    for name, rows in results.items():
        keys = limit_keys(name)
        out += [f"## {name}", ""]
        out += [
            f"limits ({tier}): " + ", ".join(f"`{k}` <= {v:g}" for k, v in lim.items())
            for tier, lim in TOLERANCES[name].items()
        ]
        out.append("")
        ident = [k for k, v in rows[0].items() if not isinstance(v, dict) and k not in keys and k != "trials"]
        out.append("| " + " | ".join(ident + keys) + " |")
        out.append("|" + "---|" * (len(ident) + len(keys)))
        for r in rows:
            cells = [f"{r[k]:.3g}" if isinstance(r[k], float) else str(r[k]) for k in ident] + [
                f"{r[k]:.3g}" for k in keys
            ]
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def _figure(results: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for snr_val, mk in ((None, "o"), (100.0, "s"), (30.0, "^")):
        sel = [r for r in results["edge"] if r["edge_snr"] == snr_val and r["angle_deg"] == 5.0]
        ax[0].errorbar(
            [r["mtf50"]["truth"] for r in sel],
            [r["mtf50"]["mean"] for r in sel],
            yerr=[r["mtf50"]["std"] for r in sel],
            fmt=mk,
            label="noise-free" if snr_val is None else f"edge SNR {snr_val:g}",
        )
    lim = [0, 0.4]
    ax[0].plot(lim, lim, "k--", lw=0.8)
    ax[0].set(xlabel="true MTF50 (cy/px)", ylabel="measured MTF50", title="slanted edge, 5 deg")
    ax[0].legend(fontsize=8)
    for kind, mk in (("single_frame", "o"), ("temporal_8frames_prnu", "s"), ("total_with_prnu", "^")):
        sel = [r for r in results["snr"] if r["kind"] == kind]
        ax[1].errorbar(
            [r["signal_e"] for r in sel],
            [r["snr_bias_pct"] for r in sel],
            yerr=[100 * r["snr"]["std"] / r["snr"]["truth"] for r in sel],
            fmt=mk,
            label=kind,
        )
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set(xscale="log", xlabel="signal (e-)", ylabel="SNR error (%)", title="flat-field SNR")
    ax[1].legend(fontsize=8)
    c = results["cdp"]
    ax[2].errorbar(range(len(c)), [r["cdp"]["truth"] for r in c], fmt="k_", ms=20, label="quadrature")
    ax[2].errorbar(
        range(len(c)), [r["cdp"]["mean"] for r in c], yerr=[r["cdp"]["std"] for r in c], fmt="o", label="iqlab"
    )
    ax[2].set(xlabel="case", ylabel="CDP", title="contrast detection probability", ylim=(0, 1.05))
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suites", nargs="+", choices=sorted(SUITES), default=list(SUITES))
    ap.add_argument("--trials", type=int, default=None, help="override each suite's default trial count")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("out/iq_lab/validation"))
    ap.add_argument("--figure", action="store_true")
    args = ap.parse_args(argv)
    kw = {"seed": args.seed} | ({"trials": args.trials} if args.trials else {})
    results = {name: SUITES[name](**kw) for name in args.suites}
    fails = [f for name, rows in results.items() for f in check(name, rows)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "iqlab_validation.json").write_text(
        json.dumps({"tolerances": TOLERANCES, "results": results, "failures": fails}, indent=2)
    )
    (args.out_dir / "iqlab_validation.md").write_text(_markdown(results))
    if args.figure and {"edge", "snr", "cdp"} <= set(results):
        _figure(results, args.out_dir / "iqlab_validation.png")
    for f in fails:
        print("FAIL", f)
    print(f"{sum(len(r) for r in results.values())} cases, {len(fails)} failures -> {args.out_dir}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
