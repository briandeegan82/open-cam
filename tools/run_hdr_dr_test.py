#!/usr/bin/env python3
"""HDR dynamic-range test: SNR, SNR=1 / SNR=10 dynamic range and P2020 CDP per chart level.

Two modes, both scored with :mod:`iqlab`:

* **model** (default): for each camera recipe, every level of the ``build_hdr_dr_chart`` chart is
  simulated as a uniform patch through the recipe's pixel (``hdr_pixel`` DCG / split-pixel / LOFIC /
  multi-exposure merge, or a single linear readout for non-HDR recipes), two frames per patch.
  Exposure: the brightest patch half sits at ``--top-fraction`` x the recipe's saturation.
* **measured**: ``--hdr-npz frame1.npz [frame2.npz]`` scores linear merged electrons (``hdr_e`` from
  ``apply_emva_noise``'s ``noisy_hdr/hdr_linear.npz``, or any HxW linear array) using the chart ROIs.
  Two frames give temporal SNR (var(A-B)/2); one frame gives total (temporal + FPN) SNR.

    venv/bin/python tools/run_hdr_dr_test.py --out-dir out/iq_lab/hdr_dr
    venv/bin/python tools/run_hdr_dr_test.py --chart scenes/iq_lab/hdr_dr_chart/chart.json \\
        --hdr-npz run1/noisy_hdr/hdr_linear.npz run2/noisy_hdr/hdr_linear.npz --out-dir out/iq_lab/hdr_dr_pbrt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import types
from pathlib import Path

import hdr_pixel as hp
import numpy as np
from build_hdr_dr_chart import build_parser as chart_parser
from build_hdr_dr_chart import layout
from camera_model import load_camera_model, noise_config_from_camera_model
from iqlab import p2020, snr
from validate_hdr_model import electrons_per_lux

REPO = Path(__file__).resolve().parent.parent
DEFAULT_RECIPES = ("default", "default_hdr_dcg", "default_hdr_split_pixel", "default_hdr_lofic", "default_hdr_3exp")


def linear_architecture(base: dict) -> hp.HdrArchitecture:
    """Single-readout pixel equivalent to the recipe's non-HDR EMVA model."""
    fw = float(base["full_well_e"])
    t = float(base["t_int_s"])
    col = hp.Collector(
        "pd",
        1.0,
        fw,
        dark_e=max(0.0, float(base["dark_current_e_per_s"]) * t * float(base.get("dark_temp_scale", 1.0))),
        prnu_std=float(base["prnu_std"]),
        dsnu_std_e=float(base["dsnu_std_e"]),
        t_int_s=t,
    )
    ro = hp.Readout(
        "main",
        "pd",
        K_e_per_DN=float(base["K_e_per_DN"]),
        sigma_e=float(base["sigma_d_e"]),
        full_well_e=fw,
        bit_depth=int(base["bit_depth"]),
        black_DN=float(base["black_DN"]),
    )
    return hp.HdrArchitecture("linear", (col,), (ro,), threshold_fraction=1.0)


def recipe_architecture(recipe: str) -> tuple[hp.HdrArchitecture, dict]:
    path = Path(recipe)
    if not path.is_file():
        path = REPO / "config" / "camera_recipes" / f"{recipe}.yaml"
    cfg = noise_config_from_camera_model(load_camera_model(path), "", "")
    base = hp.base_from_noise_config(cfg)
    if hp.hdr_enabled(cfg):
        return hp.build_architecture(cfg["hdr"], base), cfg
    return linear_architecture(base), cfg


def simulate_frames(arch: hp.HdrArchitecture, mu_e: float, n_px: int, seed: int, frames: int = 2) -> list[np.ndarray]:
    side = int(math.isqrt(n_px))
    sig = np.full((side, side), float(mu_e))
    out = []
    for f in range(frames):
        dn = hp.simulate_captures(
            arch, sig, np.random.default_rng([seed, f]), spatial_rng=np.random.default_rng([seed])
        )
        e, _ = hp.merge_captures(arch, dn)
        out.append(hp.compander_roundtrip_e(arch, e))
    return out


def analyse(
    chart: dict,
    low: list[list[np.ndarray]],
    high: list[list[np.ndarray]],
    *,
    saturation_e: float | None,
    epsilon: float = 0.5,
    e_per_lux: float | None = None,
) -> dict:
    """Score per-level patch pixels; ``low[i]`` / ``high[i]`` are lists of frames (1 or 2) for level i."""
    rows = []
    for p, lo, hi in zip(chart["patches"], low, high, strict=True):
        st = snr.patch_stats(lo[0], saturation=saturation_e)
        noise = snr.temporal_patch_stats(np.stack(lo[:2])).std if len(lo) >= 2 else st.std
        s = st.mean / noise if noise > 0 else math.inf
        cdp = p2020.contrast_detection_probability(
            lo[0], hi[0], nominal_contrast=chart["michelson_contrast"], epsilon=epsilon, n_pairs=50_000
        )
        rows.append(
            {
                "index": p["index"],
                "level_db": p["level_db"],
                "mean_e": st.mean,
                "noise_e": noise,
                "snr": s,
                "snr_db": 20 * math.log10(s) if s > 0 else -math.inf,
                "saturated_fraction": st.saturated_fraction,
                "cdp": cdp,
                **({"sensor_lux": st.mean / e_per_lux} if e_per_lux else {}),
            }
        )
    sig = np.array([r["mean_e"] for r in rows])
    s = np.array([r["snr"] for r in rows])
    sat = np.array([r["saturated_fraction"] for r in rows])
    out = {"patches": rows}
    for thr in (1.0, 10.0):
        try:
            dr = snr.dynamic_range(sig, s, snr_threshold=thr, saturated_fraction=sat)
            out[f"dr_snr{thr:g}_db"] = dr.db
            out[f"min_signal_snr{thr:g}_e"] = dr.min_signal
            out["max_unsaturated_e"] = dr.max_signal
        except ValueError:
            out[f"dr_snr{thr:g}_db"] = float("nan")
    ok = [r for r in rows if r["saturated_fraction"] <= 0.001]
    out["cdp_min_level_db_at_0p9"] = min((r["level_db"] for r in ok if r["cdp"] >= 0.9), default=float("nan"))
    return out


def run_model(
    recipe: str, chart: dict, *, top_fraction: float = 1.0, n_px: int = 4096, seed: int = 0, epsilon: float = 0.5
) -> dict:
    arch, cfg = recipe_architecture(recipe)
    sat = arch.max_reference_e
    scale = top_fraction * sat / max(p["high"]["relative_luminance"] for p in chart["patches"])
    low, high = [], []
    for i, p in enumerate(chart["patches"]):
        low.append(simulate_frames(arch, p["low"]["relative_luminance"] * scale, n_px, seed * 1000 + 2 * i))
        high.append(simulate_frames(arch, p["high"]["relative_luminance"] * scale, n_px, seed * 1000 + 2 * i + 1, 1))
    res = analyse(chart, low, high, saturation_e=0.98 * sat, epsilon=epsilon, e_per_lux=electrons_per_lux(cfg))
    res.update(
        recipe=recipe,
        architecture=arch.name,
        saturation_e=sat,
        theory_dr_db=hp.dynamic_range_db(arch),
        electrons_per_lux=electrons_per_lux(cfg),
    )
    mu = np.geomspace(1e-2, sat, 2000)
    th = hp.theory_snr(arch, mu)["snr"]
    for thr in (1, 10):
        m = res.get(f"min_signal_snr{thr}_e")
        res[f"dr_snr{thr}_to_saturation_db"] = 20 * math.log10(sat / m) if m and m > 0 else float("nan")
        res[f"theory_dr_snr{thr}_db"] = 20 * math.log10(sat / snr.snr_threshold_signal(mu, th, float(thr)))
    return res


def _roi(img: np.ndarray, roi: list[int]) -> np.ndarray:
    x0, y0, x1, y1 = roi
    return np.asarray(img, dtype=np.float64)[y0:y1, x0:x1]


def run_measured(
    chart: dict, frames: list[np.ndarray], *, saturation_e: float | None = None, epsilon: float = 0.5
) -> dict:
    if not 1 <= len(frames) <= 2:
        raise ValueError("pass one or two frames")
    for f in frames:
        if f.shape[:2] != (chart["yres"], chart["xres"]):
            raise ValueError(f"frame shape {f.shape} does not match chart {chart['yres']}x{chart['xres']}")
    low = [[_roi(f, p["low"]["roi_xyxy"]) for f in frames] for p in chart["patches"]]
    high = [[_roi(frames[0], p["high"]["roi_xyxy"])] for p in chart["patches"]]
    return {"mode": "measured", **analyse(chart, low, high, saturation_e=saturation_e, epsilon=epsilon)}


def default_chart(**overrides) -> dict:
    ns = chart_parser().parse_args([])
    vars(ns).update(overrides)
    return layout(types.SimpleNamespace(**vars(ns)))


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot(results: list[dict], path: Path) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for r in results:
        x = [p["mean_e"] for p in r["patches"]]
        label = f"{r.get('recipe', 'measured')} ({r['dr_snr1_db']:.0f} dB)"
        a1.semilogx(x, [p["snr_db"] for p in r["patches"]], "o-", ms=3, label=label)
        a2.semilogx(x, [p["cdp"] for p in r["patches"]], "o-", ms=3, label=r.get("recipe", "measured"))
    a1.axhline(0, color="k", lw=0.5)
    a1.axhline(20, color="k", lw=0.5, ls="--")
    a1.set(xlabel="signal (e-)", ylabel="SNR (dB)", title="SNR (SNR=1 and SNR=10 lines)")
    a2.set(xlabel="signal (e-)", ylabel="CDP", title="Contrast detection probability", ylim=(-0.02, 1.02))
    a1.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--chart", type=Path, default=None, help="chart.json from build_hdr_dr_chart (default layout if omitted)"
    )
    ap.add_argument("--recipes", nargs="+", default=list(DEFAULT_RECIPES))
    ap.add_argument("--hdr-npz", type=Path, nargs="+", default=None, help="measured mode: 1-2 linear-electron frames")
    ap.add_argument("--npz-key", default="hdr_e")
    ap.add_argument("--saturation-e", type=float, default=None, help="measured mode: saturation level")
    ap.add_argument("--top-fraction", type=float, default=1.0)
    ap.add_argument("--pixels-per-patch", type=int, default=4096)
    ap.add_argument("--epsilon", type=float, default=0.5, help="CDP tolerance")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=REPO / "out" / "iq_lab" / "hdr_dr")
    ap.add_argument("--figure", action="store_true", help="also write snr_cdp.png")
    args = ap.parse_args(argv)

    chart = json.loads(args.chart.read_text()) if args.chart else default_chart()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.hdr_npz:
        frames = []
        for p in args.hdr_npz:
            data = np.load(p)
            frames.append(np.asarray(data[args.npz_key] if hasattr(data, "files") else data, dtype=np.float64))
        frames = [f[..., 1] if f.ndim == 3 else f for f in frames]
        results = [run_measured(chart, frames, saturation_e=args.saturation_e, epsilon=args.epsilon)]
    else:
        results = [
            run_model(
                r,
                chart,
                top_fraction=args.top_fraction,
                n_px=args.pixels_per_patch,
                seed=args.seed,
                epsilon=args.epsilon,
            )
            for r in args.recipes
        ]
    for r in results:
        name = r.get("recipe", "measured")
        _write_csv(args.out_dir / f"{Path(name).stem}_patches.csv", r["patches"])
        print(
            f"{name:28s} DR(SNR=1) {r['dr_snr1_db']:6.1f} dB  DR(SNR=10) {r['dr_snr10_db']:6.1f} dB"
            + (
                f"  (to saturation {r['dr_snr1_to_saturation_db']:6.1f} dB, theory {r['theory_dr_snr1_db']:6.1f} dB)"
                if "theory_dr_db" in r
                else ""
            )
        )
    (args.out_dir / "hdr_dr_results.json").write_text(json.dumps({"chart": chart, "results": results}, indent=2))
    if args.figure:
        plot(results, args.out_dir / "snr_cdp.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
