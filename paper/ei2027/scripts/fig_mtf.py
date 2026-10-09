"""Slanted-edge MTF through the traced double-Gauss 50 mm lens, per wavelength.

Reads the spectral pbrt-v4 renders produced by render_scenes.sh
(build/edge_*.exr), integrates the buckets with the default green QE x IRCF,
and runs tools/sfr_analysis.slanted_edge_sfr. Because the film is spectral,
the same render also gives a per-wavelength MTF50 (longitudinal chromatic
aberration of the traced prescription -- but see docs/SFR_SPECTRAL_RIPPLE.md:
dgauss.50mm.dat has one refractive index per element, so the traced lens has
no dispersion and the per-bucket spread is pbrt Monte Carlo noise).

Run from the repository root:
    venv/bin/python paper/ei2027/scripts/fig_mtf.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))

from exr_multispectral import spectral_buckets_from_exr, trapezoid_weights_nm  # noqa: E402
from qe_curves import read_csv_curve  # noqa: E402
from sensor_radiometry import integrate_spectral_planes, spectral_electron_weights  # noqa: E402
from sfr_analysis import DEFAULT_OVERSAMPLING, slanted_edge_sfr  # noqa: E402

BUILD = REPO / "paper" / "ei2027" / "build"
OUT = REPO / "paper" / "ei2027" / "figures"
QE = REPO / "spectra" / "QE" / "interpolated"
CASES = [
    ("f2_focused", "f/2, in focus (3.2 m)"),
    ("f2_defocus", "f/2, focus 2.6 m"),
    ("f56_defocus", "f/5.6, focus 2.6 m"),
]
ROI_HALF_H = 120
ROI_HALF_W = 40


def edge_roi(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    band = img[h // 2 - ROI_HALF_H : h // 2 + ROI_HALF_H]
    grad = np.abs(np.diff(band.mean(axis=0)))
    c = int(np.argmax(grad[ROI_HALF_W : w - ROI_HALF_W])) + ROI_HALF_W
    return band[:, c - ROI_HALF_W : c + ROI_HALF_W]


def edge_sfr(roi: np.ndarray, oversampling: int = DEFAULT_OVERSAMPLING) -> dict:
    """ISO 12233-style SFR via tools/sfr_analysis (signed-derivative, robust fit).

    ``flatfield=True`` divides each row by a line fitted to its bright plateau:
    the 80-px ROI sits well off-axis and carries a ~3 % vignetting ramp.
    """
    r = slanted_edge_sfr(roi, oversampling, flatfield=True)
    return {
        "angle_deg": r.angle_deg,
        "frequency": r.frequency_cy_per_px,
        "mtf": r.mtf,
        "mtf50": r.mtf50_cy_per_px,
        "mtf10": r.mtf10_cy_per_px,
        "nyquist": r.mtf_at_nyquist,
    }


def curve_on(lam: np.ndarray, path: Path) -> np.ndarray:
    x, y = read_csv_curve(path)
    return np.interp(lam, x, y, left=0.0, right=0.0)


def main() -> None:
    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for name, label in CASES:
        planes, lam = spectral_buckets_from_exr(BUILD / f"edge_{name}.exr")
        qe_g = curve_on(lam, QE / "QE_green.csv") * curve_on(lam, QE / "QE_IRCF.csv")
        wts = spectral_electron_weights(lam, qe_g[:, None], trapezoid_weights_nm(lam), 1.0, 1.0)
        green = integrate_spectral_planes(planes, wts)[..., 0]
        res = edge_sfr(edge_roi(green))
        per_lam = []
        for k in range(lam.size):
            try:
                per_lam.append(edge_sfr(edge_roi(planes[..., k]))["mtf50"])
            except Exception:  # noqa: BLE001 - low-signal edge buckets
                per_lam.append(float("nan"))
        results[name] = {
            "label": label,
            "edge_angle_deg": res["angle_deg"],
            "mtf50_cy_per_px": res["mtf50"],
            "mtf10_cy_per_px": res["mtf10"],
            "mtf_at_nyquist": res["nyquist"],
            "lambda_nm": lam.tolist(),
            "mtf50_per_lambda_cy_per_px": per_lam,
        }
        sel = res["frequency"] <= 1.0
        axes[0].plot(res["frequency"][sel], res["mtf"][sel], lw=1, label=label)
        axes[1].plot(lam, per_lam, "o-", ms=2, lw=0.8, label=label)
    axes[0].axvline(0.5, color="0.5", ls=":", lw=1)
    axes[0].set_xlabel("frequency [cy/px]")
    axes[0].set_ylabel("MTF (green channel)")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Slanted-edge SFR, dgauss 50 mm", fontsize=9)
    axes[1].set_xlabel("wavelength bucket [nm]")
    axes[1].set_ylabel("MTF50 [cy/px]")
    axes[1].set_title("Per-bucket MTF50", fontsize=9)
    for ax in axes:
        ax.tick_params(labelsize=7)
        ax.grid(True, lw=0.3, alpha=0.5)
        ax.legend(fontsize=6)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_mtf.pdf")
    (OUT / "mtf_summary.json").write_text(json.dumps(results, indent=2))
    for name, r in results.items():
        pl = np.array(r["mtf50_per_lambda_cy_per_px"])
        print(
            name,
            {k: round(v, 4) for k, v in r.items() if isinstance(v, float)},
            "per-lambda MTF50 min/max",
            np.nanmin(pl).round(4),
            np.nanmax(pl).round(4),
            "argmax nm",
            r["lambda_nm"][int(np.nanargmax(pl))],
        )


if __name__ == "__main__":
    main()
