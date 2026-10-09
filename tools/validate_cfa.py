"""Full-pipeline validation of the generic CFA model on a real pbrt ColorChecker render.

Runs ``apply_emva_noise`` (integrate_qe, ``cfa.layout``) on a spectral ColorChecker EXR for
Bayer RGGB, RGBW, RCCB (clear = unfiltered Si), RYYCy and Quad Bayer (unbinned and 2x2
charge-binned) and compares with the analytic numbers of ``paper/ei2027/scripts/fig_cfa.py``
(``paper/ei2027/figures/cfa_summary.json``; same render settings: default recipe, 500 lux,
t = 0.12 s, patch 22 = neutral 5):

* mean luma-site electrons at 500 lux (pipeline vs analytic integral);
* luma-site SNR at 10 lux, pipeline with shot + read noise only (comparable to the analytic
  ``mu / sqrt(mu + sigma_d^2)``) and with the recipe's full EMVA noise (PRNU, DSNU, dark, ADC);
* CIEDE2000 of the 24 patches after a white-preserving channel->linear-sRGB CCM fitted on the
  demosaiced render (``cfa_mosaic.fit_channel_ccm``), 100 lux (white patch unclipped), full noise.

Means are measured at 100 lux (25 lux when 2x2-binned) and scaled to 500 lux (signal is linear).

Usage (stock pbrt render, see paper/ei2027/scripts/render_scenes.sh)::

    venv/bin/python tools/validate_cfa.py --exr paper/ei2027/build/cc_D65.exr \
        --manifest paper/ei2027/build/scenes/cc_D65/colorchecker_manifest.json --out-dir out/cfa_validation
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import apply_emva_noise  # noqa: E402
import cfa_mosaic as cm  # noqa: E402
from camera_model import load_camera_model, noise_config_from_camera_model  # noqa: E402

# Patch centres (row, col) in the 960x640 render (as in paper/ei2027/scripts/fig_cfa.py).
PATCH_CX = [255 + round(89.8 * i) for i in range(6)]
PATCH_CY = [184 + 90 * j for j in range(4)]
HALF = 15
PATCH22 = 21
WHITE = 18
CASES = {
    "Bayer RGGB": ("RGGB", "none", "G", "Bayer RGGB"),
    "RGBW": ("RGBW", "none", "W", "RGBW"),
    "RCCB": ("RCCB", "none", "C", "RCCB (clear=W)"),
    "RYYCy": ("RYYCY", "none", "Ye", "RYYCy"),
    "Quad Bayer": ("QUAD_BAYER", "none", "G", "Quad Bayer"),
    "Quad Bayer 2x2 charge bin": ("QUAD_BAYER", "charge", "G", None),
}


def run_pipeline(exr: Path, manifest: Path, layout: str, binning: str, lux: float, noise: str, tmp: Path):
    model = copy.deepcopy(load_camera_model(REPO / "config/camera_recipes/default.yaml"))
    model["cfa"] = {**model.get("cfa", {}), "layout": layout, "binning": binning, "demosaic": "gradient"}
    model["cfa"].pop("channels", None)
    cal = model.setdefault("sensor_forward", {}).setdefault("model", {}).setdefault("calibration", {})
    cal["target_illuminance_lux"] = float(lux)
    emva = model["noise"]["emva"]
    if noise == "shot_read":
        for k in ("prnu_std_fraction", "dsnu_std_e", "dark_current_e_per_s", "row_fpn_std_e", "column_fpn_std_e"):
            emva[k] = 0.0
        emva.pop("dsnu_rate_std_fraction", None)
        emva.pop("flicker_noise_std_e", None)
    raw_out = tmp / f"{layout}_{binning}_{lux:g}_{noise}.raw16"

    def _cfg(cam, **_kw):
        return noise_config_from_camera_model(cam, str(exr), str(raw_out))

    orig = apply_emva_noise.load_camera_model, apply_emva_noise.noise_config_from_camera_model
    apply_emva_noise.load_camera_model = lambda _p: model
    apply_emva_noise.noise_config_from_camera_model = _cfg
    argv = sys.argv
    try:
        sys.argv = [
            "apply_emva_noise.py",
            "--repo-root",
            str(REPO),
            "--camera-model-config",
            str(REPO / "config/camera_recipes/default.yaml"),
            "--integration-time-s",
            "0.12",
            "--scene-manifest-json",
            str(manifest),
            "--preview-color-correction-enabled",
            "false",
            "--seed",
            "7",
        ]
        apply_emva_noise.main()
    finally:
        sys.argv = argv
        apply_emva_noise.load_camera_model, apply_emva_noise.noise_config_from_camera_model = orig
    stats = json.loads((raw_out.parent / f"{raw_out.stem}_png" / "run_stats.json").read_text())
    h, w = 640 // (2 if binning != "none" else 1), 960 // (2 if binning != "none" else 1)
    raw = np.fromfile(raw_out, dtype=np.uint16).reshape(h, w).astype(np.float64)
    return raw, stats, model


def patch_sites(raw: np.ndarray, layout: cm.CfaLayout, ch: str, patch: int, scale: int) -> np.ndarray:
    cy, cx = PATCH_CY[patch // 6] // scale, PATCH_CX[patch % 6] // scale
    hw = HALF // scale
    th, tw = len(layout.tile), len(layout.tile[0])
    yy, xx = np.mgrid[cy - hw : cy + hw, cx - hw : cx + hw]
    names = np.array([[layout.tile[y % th][x % tw] for x in range(tw)] for y in range(th)])
    sel = names[yy % th, xx % tw] == ch
    return raw[yy[sel], xx[sel]]


def analytic_patch22(exr: Path, manifest: Path, lux: float) -> dict:
    """fig_cfa.py method: legacy RGB electrons anchor G; other channels by relative spectral integrals."""
    from exr_multispectral import spectral_buckets_from_exr, trapezoid_weights_nm  # noqa: PLC0415
    from pbrt_spectral_exr_to_electrons import (  # noqa: PLC0415
        scene_radiometry_from_manifest,
        spectral_radiance_to_electrons,
    )
    from sensor_radiometry import spectral_electron_weights  # noqa: PLC0415

    cam = load_camera_model(REPO / "config/camera_recipes/default.yaml")
    spec, lam = spectral_buckets_from_exr(exr)
    model = copy.deepcopy(cam["sensor_forward"]["model"])
    model["calibration"] = {**model.get("calibration", {}), "target_illuminance_lux": float(lux)}
    sensor = {**cam["sensor"], "integration_time_s": 0.12}
    e_rgb, _ = spectral_radiance_to_electrons(
        spec,
        lam,
        repo=REPO,
        sensor=sensor,
        model=model,
        lens_cfg=cam.get("lens"),
        tag="validate_cfa",
        scene=scene_radiometry_from_manifest(json.loads(manifest.read_text())),
    )
    cy, cx = PATCH_CY[PATCH22 // 6], PATCH_CX[PATCH22 % 6]
    sl = (slice(cy - HALF, cy + HALF), slice(cx - HALF, cx + HALF))
    names = ["G", "W", "C", "Ye"]
    lay = cm.resolve_layout({"layout": [names]})
    qe = cm.qe_stack_for_layout(REPO, lay, lam, ircf_csv=sensor["quantum_efficiency"].get("ircf_csv")).T
    w = spectral_electron_weights(lam, qe, trapezoid_weights_nm(lam), 1.0, 1.0)
    rel = spec[sl].astype(np.float64).mean(axis=(0, 1)) @ w
    anchor = float(e_rgb[sl][..., 1].mean()) / rel[0]
    return {n: float(rel[i] * anchor) for i, n in enumerate(lay.channels)}


def colour_error(raw: np.ndarray, layout: cm.CfaLayout, black: float, scale: int) -> dict:
    chans = cm.demosaic(raw - black, layout, "gradient")
    cam = np.array(
        [
            chans[
                PATCH_CY[p // 6] // scale - HALF // scale : PATCH_CY[p // 6] // scale + HALF // scale,
                PATCH_CX[p % 6] // scale - HALF // scale : PATCH_CX[p % 6] // scale + HALF // scale,
            ].mean(axis=(0, 1))
            for p in range(24)
        ]
    )
    cam = cam / cam[WHITE].mean()
    tgt, xyz, _ = cm.colorchecker_targets(REPO, "D65")
    ccm = cm.fit_channel_ccm(cam, tgt, white_index=WHITE)
    de = cm.delta_e2000_after_ccm(cam, xyz, ccm)
    return {"de2000_mean": float(de.mean()), "de2000_max": float(de.max()), "de2000_per_patch": de.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exr", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--summary", type=Path, default=REPO / "paper/ei2027/figures/cfa_summary.json")
    ap.add_argument("--out-dir", type=Path, default=REPO / "out/cfa_validation")
    args = ap.parse_args()
    ref = json.loads(args.summary.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {"exr": str(args.exr), "cases": {}}
    ana500 = analytic_patch22(args.exr, args.manifest, 500.0)
    results["analytic_patch22_500lux_e"] = ana500
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name, (lay_name, binning, luma, ref_key) in CASES.items():
            lay = cm.resolve_layout({"layout": lay_name})
            out_lay = cm.binned_layout(lay) if binning != "none" else lay
            scale = 2 if binning != "none" else 1
            # 100 lux keeps the white patch below full well / ADC range (25 lux for 2x2-binned sums).
            bright = 100.0 / scale**2
            row: dict = {"layout": lay_name, "binning": binning, "luma_channel": luma, "bright_lux": bright}
            for lux, noise in ((bright, "shot_read"), (10.0, "shot_read"), (10.0, "full"), (bright, "full")):
                raw, st, model = run_pipeline(args.exr, args.manifest, lay_name, binning, lux, noise, tmp)
                k, black = st["K_effective_e_per_DN"], st["black_level_DN"]
                s = (patch_sites(raw, out_lay, luma, PATCH22, scale) - black) * k
                mu, sd = float(s.mean()), float(s.std(ddof=1))
                key = f"{'bright' if lux == bright else f'{lux:g}lux'}_{noise}"
                row[f"mu_e_{key}"] = mu
                row[f"snr_db_{key}"] = 20 * np.log10(mu / sd) if sd > 0 else float("nan")
                row[f"n_sites_{key}"] = int(s.size)
                if lux == bright and noise == "full":
                    row.update(colour_error(raw, out_lay, black, scale))
            sig_d = float(model["noise"]["emva"]["sigma_d_e"])
            nbin = 4.0 if binning != "none" else 1.0
            mu_a = ana500[luma] * nbin
            mu_a10 = mu_a * 10.0 / 500.0
            row["analytic_mu_e_500lux"] = mu_a
            row["mu_e_500lux_shot_read"] = row["mu_e_bright_shot_read"] * 500.0 / bright
            row["pipeline_over_analytic_mu_500lux"] = row["mu_e_500lux_shot_read"] / mu_a
            row["analytic_snr_db_10lux"] = 20 * np.log10(mu_a10 / np.sqrt(mu_a10 + sig_d**2))
            row["luma_over_G_pipeline"] = row["mu_e_500lux_shot_read"] / (
                results["cases"]["Bayer RGGB"]["mu_e_500lux_shot_read"] if results["cases"] else mu_a
            )
            if ref_key:
                r = ref["cfas"][ref_key]
                row["ref_mu_e_500lux"] = r["mu_e_patch22_500lux"]
                row["ref_snr_db_10lux"] = r["snr_db_patch22_10lux"]
                row["ref_luma_over_G"] = r["luma_over_G_D65_patch22"]
            results["cases"][name] = row
            print(
                f"{name:26s} mu500={row['mu_e_500lux_shot_read']:7.1f} analytic={mu_a:7.1f}"
                f" (x{row['pipeline_over_analytic_mu_500lux']:.4f}) json={row.get('ref_mu_e_500lux', float('nan')):6.1f}"
                f" luma/G={row['luma_over_G_pipeline']:.3f} (json {row.get('ref_luma_over_G', float('nan')):.3f})"
                f"  SNR10 s+r={row['snr_db_10lux_shot_read']:5.2f} full={row['snr_db_10lux_full']:5.2f}"
                f" analytic={row['analytic_snr_db_10lux']:5.2f} json={row.get('ref_snr_db_10lux', float('nan')):5.2f}"
                f"  dE00 mean={row['de2000_mean']:.2f}"
                f" max={row['de2000_max']:.2f}"
            )
    (args.out_dir / "cfa_validation.json").write_text(json.dumps(results, indent=2) + "\n")
    plot(results, args.out_dir / "cfa_validation.png")


def plot(results: dict, path: Path) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    names = list(results["cases"])
    rows = [results["cases"][n] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].bar(x - 0.27, [r["mu_e_500lux_shot_read"] for r in rows], 0.27, label="pipeline (cfa.layout)")
    ax[0].bar(x, [r["analytic_mu_e_500lux"] for r in rows], 0.27, label="analytic, this render")
    ax[0].bar(
        x + 0.27,
        [r.get("ref_luma_over_G", np.nan) * rows[0]["analytic_mu_e_500lux"] for r in rows],
        0.27,
        label="cfa_summary.json luma/G x Bayer",
    )
    ax[0].set_ylabel("luma-site mean electrons, patch 22 (scaled to 500 lux)")
    ax[1].bar(x - 0.27, [r["snr_db_10lux_shot_read"] for r in rows], 0.27, label="pipeline shot+read")
    ax[1].bar(x, [r["snr_db_10lux_full"] for r in rows], 0.27, label="pipeline full EMVA noise")
    ax[1].bar(x + 0.27, [r["analytic_snr_db_10lux"] for r in rows], 0.27, label="analytic mu/sqrt(mu+sd^2)")
    ax[1].set_ylabel("luma-site SNR [dB], patch 22, 10 lux")
    ax[2].bar(x - 0.2, [r["de2000_mean"] for r in rows], 0.4, label="mean")
    ax[2].bar(x + 0.2, [r["de2000_max"] for r in rows], 0.4, label="max")
    ax[2].set_ylabel("CIEDE2000 after CCM (24 patches, 100 lux)")
    for a in ax:
        a.set_xticks(x, names, rotation=30, ha="right", fontsize=8)
        a.legend(fontsize=7, loc="lower right")
        a.grid(axis="y", alpha=0.3)
    fig.suptitle("open-cam generic CFA: full pipeline on pbrt ColorChecker (D65, default recipe, t=0.12 s)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
