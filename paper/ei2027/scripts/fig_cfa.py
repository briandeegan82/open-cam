"""Generic-CFA study through the pipeline, on the pbrt-v4 spectral ColorChecker renders.

Each recipe with a true periodic ``cfa.layout`` (docs/CFA_NIR.md) goes through
tools/apply_emva_noise.py unchanged: electrons are integrated per channel from the spectral EXR
(``processing.linear_exr_mode: integrate_qe``), each site keeps its own channel, and shot noise,
PRNU/DSNU, crosstalk, charge binning, full well and ADC act per site. The Bayer reference is the
default recipe on the same path. Reported per layout, on the raw mosaic:

* the luminance channel's electrons on the neutral 5 patch at 50 lux, D65 and CIE A;
* its temporal SNR at 10 lux, measured from two noise seeds (fixed-pattern terms and render noise
  cancel in the difference), next to the EMVA prediction mu / sqrt(mu + sigma_d^2);
* CIEDE2000 on the 24 patches after the recipe's spectral ColorChecker CCM, for the noise-free
  spectral fit and for the demosaiced pipeline output at 50 lux.

Run from the repository root after render_scenes.sh (make_figures.sh calls it):
    venv/bin/python paper/ei2027/scripts/fig_cfa.py
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))

import apply_emva_noise  # noqa: E402
import cfa_mosaic as cm  # noqa: E402
import colour_science as cs  # noqa: E402
import run_recipe_scorecard as rs  # noqa: E402
from camera_model import load_camera_model, noise_config_from_camera_model  # noqa: E402

BUILD = REPO / "paper" / "ei2027" / "build"
OUT = REPO / "paper" / "ei2027" / "figures"
T_INT = 0.12  # exposure_time_override_s from config/pipeline.yaml
# 50 lux keeps every layout below ADC full scale on the white patch: the 10-bit ADC (K x 1023 ~ 4.9 ke-)
# clips the 2x2 charge-binned quad-Bayer sum above ~100 lux at 0.12 s, although its full well is 4x.
LUX, LOW_LUX = 50.0, 10.0

# recipe -> (label, luminance channel)
RECIPES = {
    "default": ("Bayer (GBRG)", "G"),
    "default_quad_bayer": ("Quad Bayer (2x2 charge bin)", "G"),
    "default_rgbw": ("RGBW 2x2", "W"),
    "default_rgbw_4x4": ("RGBW 4x4", "W"),
    "default_rccb": ("RCCB", "C"),
    "default_ryycy": ("RYYCy", "Ye"),
}
# Patch centres in the 960x640 render (left edges x=214+89.8*i, top edges y=143+90*j, 82 px patches).
PATCH_CX = [255 + round(89.8 * i) for i in range(6)]
PATCH_CY = [184 + 90 * j for j in range(4)]
HALF = 15
WHITE, NEUTRAL5 = 18, 21  # 0-based: patch 19 (white 9.5), patch 22 (neutral 5, ~19 % reflectance)


def model_for(recipe: str, lux: float) -> dict:
    model = copy.deepcopy(load_camera_model(rs.RECIPE_DIR / f"{recipe}.yaml"))
    model.setdefault("processing", {})["linear_exr_mode"] = "integrate_qe"
    model["sensor_forward"]["model"]["calibration"]["target_illuminance_lux"] = float(lux)
    return model


def run(recipe: str, ill: str, lux: float, seed: int) -> tuple[np.ndarray, cm.CfaLayout, dict]:
    """Raw electron mosaic of one pipeline run (fixed 0.12 s exposure, no auto-exposure)."""
    model = model_for(recipe, lux)
    exr = BUILD / f"cc_{ill}.exr"
    tmp = BUILD / "cfa_runs" / f"{recipe}_{ill}_{lux:g}_{seed}"
    tmp.mkdir(parents=True, exist_ok=True)
    raw_out = tmp / "raw.raw16"
    with rs._patched(model, exr, raw_out), contextlib.redirect_stdout(sys.stderr):
        sys.argv = [
            "apply_emva_noise.py", "--repo-root", str(REPO),
            "--camera-model-config", str(rs.RECIPE_DIR / f"{recipe}.yaml"),
            "--linear-exr", str(exr),
            "--scene-manifest-json", str(BUILD / "scenes" / f"cc_{ill}" / "colorchecker_manifest.json"),
            "--integration-time-s", str(T_INT), "--raw-out", str(raw_out),
            "--preview-color-correction-enabled", "false", "--seed", str(seed),
        ]  # fmt: skip
        apply_emva_noise.main()
    stats = json.loads((tmp / "raw_png" / "run_stats.json").read_text())
    raw = np.fromfile(raw_out, dtype=np.uint16).astype(np.float64)
    h = int(round(math.sqrt(raw.size * 640 / 960)))
    e = (raw.reshape(h, -1) - float(stats["black_level_DN"])) * float(stats["K_effective_e_per_DN"])
    layout = rs.recipe_layout(noise_config_from_camera_model(model, "", ""))
    if h != 640:  # charge-binned output
        layout = cm.binned_layout(layout, 640 // h)
    return e, layout, stats


def patch_slices(scale: int) -> list[tuple[slice, slice]]:
    hw = HALF // scale
    return [
        (slice(cy // scale - hw, cy // scale + hw), slice(cx // scale - hw, cx // scale + hw))
        for cy in PATCH_CY
        for cx in PATCH_CX
    ]


def site_values(e: np.ndarray, layout: cm.CfaLayout, ch: str, sl: tuple[slice, slice]) -> np.ndarray:
    idx = layout.site_index(*e.shape)
    return e[sl][idx[sl] == layout.channels.index(ch)]


def ideal_cam(layout: cm.CfaLayout, ircf: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Noise-free spectral camera responses of the 24 patches (perfect diffuser = 1) and XYZ (D65)."""
    lam = np.arange(380.0, 731.0, 1.0)
    chart = cs.load_colorchecker(cm.DATA_ROOT, lam)
    _, spd = cs.load_illuminant(cm.DATA_ROOT, "D65", lam)
    qe = cm.qe_stack_for_layout(REPO, layout, lam, ircf_csv=ircf).astype(np.float64)
    cam = cs.integrate_spectra(lam, cs.radiance_spectra(chart.reflectance, spd), qe)
    cam /= np.maximum(cs.integrate_spectra(lam, np.atleast_2d(spd), qe)[0], 1e-30)[None, :]
    xyz = cs.xyz_from_spectra(lam, chart.reflectance, spd)
    return cam, xyz, lam


def de00(cam: np.ndarray, xyz_ref: np.ndarray, ccm: np.ndarray) -> np.ndarray:
    xyz = cs.srgb_linear_to_xyz(cm.apply_channel_ccm(cam, ccm))
    xyz = xyz / xyz[WHITE, 1]
    ref = xyz_ref / xyz_ref[WHITE, 1]
    return cs.delta_e_2000(cs.xyz_to_lab(xyz), cs.xyz_to_lab(ref))


def main() -> None:
    rows, crops = {}, {}
    for recipe, (label, luma) in RECIPES.items():
        model = model_for(recipe, LUX)
        sigma_d = float(model["noise"]["emva"]["sigma_d_e"])
        ircf = model["sensor"]["quantum_efficiency"].get("ircf_csv")
        if not (model.get("cfa") or {}).get("ircf", True):
            ircf = None
        e, layout, stats = run(recipe, "D65", LUX, 0)
        scale = 640 // e.shape[0]
        sls = patch_slices(scale)
        mu = float(site_values(e, layout, luma, sls[NEUTRAL5]).mean())
        e_a, _, _ = run(recipe, "A", LUX, 0)
        mu_a = float(site_values(e_a, layout, luma, sls[NEUTRAL5]).mean())
        lo = [run(recipe, "D65", LOW_LUX, s)[0] for s in (1, 2)]
        v1, v2 = (site_values(x, layout, luma, sls[NEUTRAL5]) for x in lo)
        mu_lo = float(0.5 * (v1.mean() + v2.mean()))
        sigma_lo = float(np.std(v1 - v2) / math.sqrt(2.0))

        ccm, _ = cm.colorchecker_spectral_ccm(REPO, layout, ircf_csv=ircf)
        cam_ideal, xyz_ref, _ = ideal_cam(layout, ircf)
        chans = cm.demosaic(np.clip(e, 0.0, None), layout, str((model.get("cfa") or {}).get("demosaic", "gradient")))
        pm = np.array([chans[sl].reshape(-1, chans.shape[2]).mean(axis=0) for sl in sls])
        cam_pipe = pm / pm[WHITE] * cam_ideal[WHITE]
        de_ideal, de_pipe = de00(cam_ideal, xyz_ref, ccm), de00(cam_pipe, xyz_ref, ccm)

        rows[recipe] = {
            "label": label,
            "layout": [list(r) for r in layout.tile],
            "luma_channel": luma,
            "demosaic": str((model.get("cfa") or {}).get("demosaic", "gradient")),
            "binning": str((model.get("cfa") or {}).get("binning", "none")),
            "raw_shape": list(e.shape),
            "sigma_d_e": sigma_d,
            "mu_e_neutral5_lux_D65": mu,
            "mu_e_neutral5_lux_A": mu_a,
            "mu_e_neutral5_10lux_D65": mu_lo,
            "sigma_temporal_e_10lux_measured": sigma_lo,
            "snr_db_10lux_measured": 20 * math.log10(mu_lo / sigma_lo),
            "snr_db_10lux_emva": 20 * math.log10(mu_lo / math.sqrt(mu_lo + sigma_d**2)),
            "adc_clip_e": float(stats["K_effective_e_per_DN"])
            * ((1 << int(stats["bit_depth"])) - 1 - float(stats["black_level_DN"])),
            "white_patch_luma_fraction_of_full_well": float(
                site_values(e, layout, luma, sls[WHITE]).mean() / float(stats["full_well_effective_e"])
            ),
            "de00_mean_ideal_ccm": float(de_ideal.mean()),
            "de00_max_ideal_ccm": float(de_ideal.max()),
            "de00_mean_pipeline_lux": float(de_pipe.mean()),
            "de00_max_pipeline_lux": float(de_pipe.max()),
        }
        # Display crop: corner of patches 14/15/20/21 after demosaic + white balance + CCM.
        y0, y1, x0, x1 = 320 // scale, 500 // scale, 300 // scale, 480 // scale
        wroi = [sls[WHITE][1].start, sls[WHITE][0].start, sls[WHITE][1].stop, sls[WHITE][0].stop]
        rgb = rs.to_srgb_linear(e, layout, ccm, wroi)
        crops[recipe] = rgb[y0:y1, x0:x1] / rgb[sls[WHITE]].mean()

    ref = rows["default"]
    for r in rows.values():
        r["luma_over_bayer_G_D65"] = r["mu_e_neutral5_lux_D65"] / ref["mu_e_neutral5_lux_D65"]
        r["luma_over_bayer_G_A"] = r["mu_e_neutral5_lux_A"] / ref["mu_e_neutral5_lux_A"]
        r["snr_gain_db_vs_bayer_10lux_measured"] = r["snr_db_10lux_measured"] - ref["snr_db_10lux_measured"]
    summary = {"integration_time_s": T_INT, "lux": LUX, "low_lux": LOW_LUX, "cfas": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cfa_summary.json").write_text(json.dumps(summary, indent=2))

    # Figure 1: per-channel QE x IRCF of the default research camera (cfa_mosaic.DEFAULT_CHANNEL_QE).
    dflt = model_for("default", LUX)
    ircf = dflt["sensor"]["quantum_efficiency"].get("ircf_csv")
    lam = np.arange(380.0, 781.0, 1.0)
    names = ["R", "G", "B", "W", "Ye", "Cy"]
    lay = cm.resolve_layout({"layout": [["R", "G", "B"], ["W", "Ye", "Cy"]]})
    qe = cm.qe_stack_for_layout(REPO, lay, lam, ircf_csv=ircf)
    colours = {"R": "tab:red", "G": "tab:green", "B": "tab:blue", "W": "0.2", "Ye": "goldenrod", "Cy": "tab:cyan"}
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for n in names:
        ax.plot(lam, qe[lay.channels.index(n)], color=colours[n], lw=1.1, label="W, C" if n == "W" else n)
    ax.set_xlabel("wavelength [nm]", fontsize=8)
    ax.set_ylabel(r"QE $\times$ IRCF", fontsize=8)
    ax.set_xlim(380, 780)
    ax.tick_params(labelsize=7)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.legend(fontsize=6, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_qe.pdf")
    plt.close(fig)

    # Figure 2: pipeline output (demosaiced, white-balanced, CCM) for each layout, same crop.
    fig, axes = plt.subplots(2, 3, figsize=(3.4, 2.55))
    for ax, (recipe, img) in zip(axes.ravel(), crops.items(), strict=True):
        srgb = np.clip(img, 0, 1)
        srgb = np.where(srgb <= 0.0031308, 12.92 * srgb, 1.055 * srgb ** (1 / 2.4) - 0.055)
        ax.imshow(srgb, interpolation="nearest")
        ax.set_title(RECIPES[recipe][0], fontsize=5.5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(pad=0.3, h_pad=0.6)
    fig.savefig(OUT / "fig_cfa_mosaics.pdf", dpi=300)
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
