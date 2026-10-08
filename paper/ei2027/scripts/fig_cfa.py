"""Non-Bayer CFA study on the same pbrt-v4 spectral ColorChecker render.

open-cam's sampling/demosaic path (tools/apply_emva_noise.py) implements only
2x2 Bayer phases; RCCB/RYYCy sensor models are RGB proxies. This script shows
what the hyperspectral render nonetheless enables *outside* that path: any
channel set can be integrated from the same spectral buckets with
tools/sensor_radiometry.spectral_electron_weights, sampled on an arbitrary
mosaic tile and compared with the analytic EMVA temporal-noise model.
Absolute electrons are anchored to the pipeline output of
tools/pbrt_spectral_exr_to_electrons.py (default recipe, 500 lux).

Run from the repository root (after render_scenes.sh and make_figures.sh steps):
    venv/bin/python paper/ei2027/scripts/fig_cfa.py
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

from camera_model import load_camera_model  # noqa: E402
from exr_multispectral import spectral_buckets_from_exr, trapezoid_weights_nm  # noqa: E402
from qe_curves import read_csv_curve  # noqa: E402
from sensor_radiometry import integrate_spectral_planes, spectral_electron_weights  # noqa: E402

BUILD = REPO / "paper" / "ei2027" / "build"
OUT = REPO / "paper" / "ei2027" / "figures"
QE = REPO / "spectra" / "QE" / "interpolated"

# Channel name -> QE CSV (all multiplied by the default IRCF). "W" is the
# unfiltered (mono) silicon QE, i.e. a physically clear pixel.
CHANNELS = {"R": "QE_red", "G": "QE_green", "B": "QE_blue", "W": "QE_mono", "Ye": "QE_yellow", "Cy": "QE_cyan"}
CFAS = {
    "Bayer RGGB": [["R", "G"], ["G", "B"]],
    "Quad Bayer": [["R", "R", "G", "G"], ["R", "R", "G", "G"], ["G", "G", "B", "B"], ["G", "G", "B", "B"]],
    "RGBW": [["R", "G"], ["W", "B"]],
    "RCCB (clear=W)": [["R", "W"], ["W", "B"]],
    "RCCB (repo proxy C=Cy)": [["R", "Cy"], ["Cy", "B"]],
    "RYYCy": [["R", "Ye"], ["Ye", "Cy"]],
}
LUMA = {
    "Bayer RGGB": "G",
    "Quad Bayer": "G",
    "RGBW": "W",
    "RCCB (clear=W)": "W",
    "RCCB (repo proxy C=Cy)": "Cy",
    "RYYCy": "Ye",
}
# Patch centres (row, col) in the 960x640 render, measured from patch edges
# (left edges x=214+89.8*i, top edges y=143+90*j, 82 px patches).
PATCH_CX = [255 + round(89.8 * i) for i in range(6)]
PATCH_CY = [184 + 90 * j for j in range(4)]
HALF = 15
NEUTRAL = list(range(18, 24))  # patches 19..24 (white..black), 0-based
LOW_LIGHT_SCALE = 10.0 / 500.0  # 10 lux vs the 500 lux calibration


def curve_on(lam: np.ndarray, name: str) -> np.ndarray:
    x, y = read_csv_curve(QE / f"{name}.csv")
    return np.interp(lam, x, y, left=0.0, right=0.0)


def channel_electrons(ill: str, anchor_scale: float | None = None) -> tuple[dict, np.ndarray, np.ndarray, float, float]:
    planes, lam = spectral_buckets_from_exr(BUILD / f"cc_{ill}.exr")
    ircf = curve_on(lam, "QE_IRCF")
    names = list(CHANNELS)
    qe = np.stack([curve_on(lam, CHANNELS[n]) * ircf for n in names], axis=1)
    wts = spectral_electron_weights(lam, qe, trapezoid_weights_nm(lam), 1.0, 1.0)
    rel = integrate_spectral_planes(planes, wts)
    cv = float("nan")
    if anchor_scale is None:
        ref = np.load(BUILD / f"cc_{ill}_default_e.npz")["electrons_rgb"]
        ratios = []
        for cy in PATCH_CY:
            for cx in PATCH_CX:
                sl = (slice(cy - HALF, cy + HALF), slice(cx - HALF, cx + HALF))
                ratios.append(ref[sl][..., 1].mean() / rel[sl][..., names.index("G")].mean())
        ratios = np.array(ratios)
        anchor_scale = float(np.median(ratios))
        cv = float(ratios.std() / ratios.mean())
    elec = {n: rel[..., i] * anchor_scale for i, n in enumerate(names)}
    return elec, lam, qe, anchor_scale, cv


def patch_means(elec: dict) -> dict:
    out = {}
    for n, img in elec.items():
        out[n] = [float(img[cy - HALF : cy + HALF, cx - HALF : cx + HALF].mean()) for cy in PATCH_CY for cx in PATCH_CX]
    return out


def mosaic(elec: dict, tile: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
    h, w = next(iter(elec.values())).shape
    th, tw = len(tile), len(tile[0])
    raw = np.zeros((h, w))
    rgb = np.zeros((h, w, 3))
    tint = {
        "R": (1, 0.25, 0.25),
        "G": (0.3, 1, 0.3),
        "B": (0.3, 0.45, 1),
        "W": (1, 1, 1),
        "Ye": (1, 1, 0.3),
        "Cy": (0.3, 1, 1),
    }
    for i in range(th):
        for j in range(tw):
            n = tile[i][j]
            raw[i::th, j::tw] = elec[n][i::th, j::tw]
            rgb[i::th, j::tw] = elec[n][i::th, j::tw, None] * np.array(tint[n])
    return raw, rgb


def snr(mu: float, sigma_d: float) -> float:
    return mu / np.sqrt(mu + sigma_d**2)


def main() -> None:
    cam = load_camera_model(REPO / "config" / "camera_recipes" / "default.yaml")
    sigma_d = float(cam["noise"]["emva"]["sigma_d_e"])
    full_well = float(cam["noise"]["adc"]["full_well_e"])

    elec, lam, qe, scale, cv = channel_electrons("D65")
    elec_a, *_ = channel_electrons("A", anchor_scale=None)
    pm, pm_a = patch_means(elec), patch_means(elec_a)

    white = 18
    proxy_ratio = np.array(pm["Cy"]) / np.array(pm["W"])
    proxy_ratio /= proxy_ratio[white]
    rows = {}
    for cfa, luma in LUMA.items():
        mu_500 = pm[luma][21]  # patch 22, neutral 5 (~19% reflectance)
        mu_10 = mu_500 * LOW_LIGHT_SCALE
        rows[cfa] = {
            "luma_channel": luma,
            "mu_e_patch22_500lux": mu_500,
            "snr_db_patch22_10lux": 20 * np.log10(snr(mu_10, sigma_d)),
            "white_patch_fraction_of_full_well": pm[luma][white] / full_well,
            "luma_over_G_D65_patch22": pm[luma][21] / pm["G"][21],
            "luma_over_G_A_patch22": pm_a[luma][21] / pm_a["G"][21],
        }
    ref_db = rows["Bayer RGGB"]["snr_db_patch22_10lux"]
    for r in rows.values():
        r["snr_gain_db_vs_bayer_10lux"] = r["snr_db_patch22_10lux"] - ref_db
    summary = {
        "anchor_scale_e_per_unit": scale,
        "anchor_ratio_cv_over_24_patches": cv,
        "sigma_d_e": sigma_d,
        "full_well_e": full_well,
        "cyan_proxy_over_clear_normalised_to_white": {
            "min": float(proxy_ratio.min()),
            "max": float(proxy_ratio.max()),
            "argmin_patch": int(np.argmin(proxy_ratio)) + 1,
            "argmax_patch": int(np.argmax(proxy_ratio)) + 1,
        },
        "cfas": rows,
        "neutral_patch_means_D65": {n: [pm[n][k] for k in NEUTRAL] for n in pm},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cfa_summary.json").write_text(json.dumps(summary, indent=2))

    # Figure 1: channel QE x IRCF curves used here.
    colours = {"R": "tab:red", "G": "tab:green", "B": "tab:blue", "W": "0.2", "Ye": "goldenrod", "Cy": "tab:cyan"}
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for i, n in enumerate(CHANNELS):
        ax.plot(lam, qe[:, i], color=colours[n], lw=1.1, label=n)
    ax.set_xlabel("wavelength [nm]", fontsize=8)
    ax.set_ylabel(r"QE $\times$ IRCF", fontsize=8)
    ax.set_xlim(380, 780)
    ax.tick_params(labelsize=7)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.legend(fontsize=6, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_qe.pdf")
    plt.close(fig)

    # Figure 2: raw mosaics of the same spectral render, crop at the corner of patches 14/15/20/21.
    names = list(CFAS)
    fig, axes = plt.subplots(2, 3, figsize=(3.4, 2.6))
    crop = (slice(374, 414), slice(366, 406))
    vmax = np.percentile(elec["W"][crop], 99)
    for ax, name in zip(axes.ravel(), names, strict=True):
        _, rgb = mosaic(elec, CFAS[name])
        ax.imshow(np.clip(rgb[crop] / vmax, 0, 1) ** (1 / 2.2), interpolation="nearest")
        ax.set_title(name.replace(" (", "\n("), fontsize=5.5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(pad=0.3, h_pad=0.6)
    fig.savefig(OUT / "fig_cfa_mosaics.pdf", dpi=300)
    plt.close(fig)

    print(json.dumps({k: v for k, v in summary.items() if k != "neutral_patch_means_D65"}, indent=2))


if __name__ == "__main__":
    main()
