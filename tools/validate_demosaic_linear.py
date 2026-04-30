#!/usr/bin/env python3
"""Validate Bayer+bilinear demosaic against linear RGB reference in DN space."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from apply_emva_noise import bayer_sample_rgb, bilinear_demosaic, load_electrons_npz
from camera_model import load_camera_model, noise_config_from_camera_model


def chart_interior_mask(
    manifest: dict,
    yres: int,
    xres: int,
    inset_frac: float,
) -> np.ndarray:
    cam_dist = float(manifest["camera"]["cam_dist"])
    _fov_raw = manifest["camera"].get("fov_deg")
    if _fov_raw is None:
        raise RuntimeError(
            "chart_interior_mask requires fov_deg in the scene manifest. "
            "Realistic camera scenes do not record an authoritative fov_deg; "
            "patch-interior masking is unavailable for this scene."
        )
    fov_deg = float(_fov_raw)
    pw = float(manifest["geometry"]["patch_width"])
    ph = float(manifest["geometry"]["patch_height"])
    gap = float(manifest["geometry"]["gap"])
    board_w = float(manifest["geometry"]["board_size"][0])
    board_h = float(manifest["geometry"]["board_size"][1])

    jj, ii = np.meshgrid(np.arange(yres, dtype=np.float64), np.arange(xres, dtype=np.float64), indexing="ij")
    x_ndc = 2.0 * ((ii + 0.5) / xres) - 1.0
    y_ndc = 1.0 - 2.0 * ((jj + 0.5) / yres)
    aspect = xres / yres
    tan_half = np.tan(np.deg2rad(fov_deg) * 0.5)
    xw = -x_ndc * cam_dist * tan_half * aspect
    yw = y_ndc * cam_dist * tan_half

    mask = np.zeros((yres, xres), dtype=bool)
    x0 = -board_w / 2.0
    y_top = board_h / 2.0
    dx = max(0.0, pw * inset_frac)
    dy = max(0.0, ph * inset_frac)
    for row in range(4):
        for col in range(6):
            xL = x0 + col * (pw + gap)
            xR = xL + pw
            px0 = -xR + dx
            px1 = -xL - dx
            py1 = y_top - row * (ph + gap) - dy
            py0 = y_top - row * (ph + gap) - ph + dy
            if px1 <= px0 or py1 <= py0:
                continue
            mask |= (xw >= px0) & (xw <= px1) & (yw >= py0) & (yw <= py1)
    return mask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    repo_default = Path(__file__).resolve().parent.parent
    ap.add_argument("--repo-root", type=Path, default=repo_default)
    ap.add_argument("--config", type=Path, default=None, help="noise YAML config path")
    ap.add_argument("--camera-model-config", type=Path, default=None, help="Camera model YAML path (preferred).")
    ap.add_argument(
        "--electrons-npz",
        type=Path,
        default=None,
        help="Expected HxWx3 electrons input. Defaults to out/sensor_forward_electrons.npz",
    )
    ap.add_argument("--crop", type=int, default=2, help="Ignore this border width in metrics.")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional scene manifest for patch-interior masking (default: scenes/generated/colorchecker_manifest.json)",
    )
    ap.add_argument(
        "--patch-inset-frac",
        type=float,
        default=0.1,
        help="Inset each patch by this fraction for masked metrics (0 disables masked metrics).",
    )
    ap.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path.")
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    if args.camera_model_config is not None:
        cfg_path = args.camera_model_config.resolve()
        camera_model = load_camera_model(cfg_path)
        cfg = noise_config_from_camera_model(
            camera_model,
            linear_rgb_in="out/colorchecker_spectral.exr",
            raw_out="out/colorchecker_noisy.raw16",
        )
    else:
        cfg_path = (args.config or (repo / "config" / "noise_emva.yaml")).resolve()
        cfg = yaml.safe_load(cfg_path.read_text())
    bayer_cfg = cfg.get("bayer") or {}
    adc = cfg.get("adc", {})
    emva = cfg.get("emva", {})

    if not bool(bayer_cfg.get("enabled", False)):
        raise RuntimeError("bayer.enabled is false in config; validation is for Bayer+demosaic path.")

    pattern = str(bayer_cfg.get("pattern", "RGGB"))
    npz_path = (args.electrons_npz or (repo / "out" / "sensor_forward_electrons.npz")).resolve()
    signal_e = load_electrons_npz(npz_path).astype(np.float64)

    bit_depth = int(adc.get("bit_depth", 12))
    if "full_well_e" not in adc:
        raise KeyError(
            "adc.full_well_e is required but not set in the sensor config. "
            "This value determines the ADC clipping point; there is no safe generic default."
        )
    full_well_e = float(adc["full_well_e"])
    K_e_per_DN = float(emva.get("overall_system_gain_K_e_per_DN", 0.08))
    black_dn = float(emva.get("black_level_DN", 64.0))
    max_dn = float((1 << bit_depth) - 1)

    dn_ref = np.clip(np.clip(signal_e, 0.0, full_well_e) / K_e_per_DN + black_dn, 0.0, max_dn)
    bayer_dn = bayer_sample_rgb(dn_ref.astype(np.float32), pattern).astype(np.float64)
    dn_dem = bilinear_demosaic(bayer_dn, pattern).astype(np.float64)

    h, w, _ = dn_ref.shape
    c = max(0, int(args.crop))
    if c * 2 < h and c * 2 < w:
        ref_eval = dn_ref[c : h - c, c : w - c, :]
        dem_eval = dn_dem[c : h - c, c : w - c, :]
    else:
        ref_eval = dn_ref
        dem_eval = dn_dem

    err = dem_eval - ref_eval
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    max_abs = float(np.max(np.abs(err)))
    per_channel_mae = np.mean(np.abs(err), axis=(0, 1)).tolist()
    per_channel_rmse = np.sqrt(np.mean(err**2, axis=(0, 1))).tolist()

    metrics = {
        "config": str(cfg_path),
        "electrons_npz": str(npz_path),
        "pattern": pattern,
        "crop": c,
        "mae_dn": mae,
        "rmse_dn": rmse,
        "max_abs_dn": max_abs,
        "mae_dn_rgb": per_channel_mae,
        "rmse_dn_rgb": per_channel_rmse,
        "eval_shape": list(ref_eval.shape),
    }

    manifest_path = (args.manifest or (repo / "scenes" / "generated" / "colorchecker_manifest.json")).resolve()
    inset = float(args.patch_inset_frac)
    if inset > 0 and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        try:
            mask_full = chart_interior_mask(manifest, yres=h, xres=w, inset_frac=inset)
        except RuntimeError as exc:
            print(f"[validate_demosaic] patch-interior masking skipped: {exc}", file=sys.stderr)
            mask_full = None
        if mask_full is not None:
            if c * 2 < h and c * 2 < w:
                mask_eval = mask_full[c : h - c, c : w - c]
            else:
                mask_eval = mask_full
            if np.any(mask_eval):
                e2 = err[mask_eval]
                metrics["masked"] = {
                    "manifest": str(manifest_path),
                    "patch_inset_frac": inset,
                    "pixel_count": int(mask_eval.sum()),
                    "mae_dn": float(np.mean(np.abs(e2))),
                    "rmse_dn": float(np.sqrt(np.mean(e2**2))),
                    "max_abs_dn": float(np.max(np.abs(e2))),
                    "mae_dn_rgb": np.mean(np.abs(e2), axis=0).tolist(),
                    "rmse_dn_rgb": np.sqrt(np.mean(e2**2, axis=0)).tolist(),
                }

    print(json.dumps(metrics, indent=2))
    if args.json_out is not None:
        out = args.json_out.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
