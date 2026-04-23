#!/usr/bin/env python3
"""Apply a simple spatial PSF to a pbrt SpectralFilm EXR (all float channels).

Intended as a post-render optics step (measured MTF / defocus approximation) after
PBRT. Uses separable Gaussian blur by default (numpy only). Does not model chromatic
aberration per wavelength unless you extend this tool.

See config/optics.yaml and README (optics section).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from camera_model import load_camera_model
from exr_multispectral import read_separate_exr_channels, write_separate_channels_exr


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.ones(1, dtype=np.float64)
    r = int(max(1, np.ceil(3.0 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= np.sum(k)
    return k


def separable_gaussian_blur_2d(img: np.ndarray, sigma: float) -> np.ndarray:
    """HxW float — reflect-pad separable Gaussian."""
    if sigma <= 0:
        return np.asarray(img, dtype=np.float32, copy=False)
    k = _gaussian_kernel_1d(sigma)
    if k.size == 1:
        return np.asarray(img, dtype=np.float32, copy=False)
    pad = k.size // 2
    acc = np.asarray(img, dtype=np.float64)
    # horizontal
    x = np.pad(acc, ((0, 0), (pad, pad)), mode="reflect")
    tmp = np.empty_like(acc)
    for i in range(acc.shape[0]):
        tmp[i, :] = np.convolve(x[i, :], k, mode="valid")
    # vertical
    y = np.pad(tmp, ((pad, pad), (0, 0)), mode="reflect")
    out = np.empty_like(acc)
    for j in range(acc.shape[1]):
        out[:, j] = np.convolve(y[:, j], k, mode="valid")
    return out.astype(np.float32, copy=False)


def apply_stray_light(arr: np.ndarray, cfg: dict) -> np.ndarray:
    """Apply simple veiling glare + broad halo stray-light model."""
    if not bool(cfg.get("enabled", False)):
        return np.asarray(arr, dtype=np.float32, copy=False)

    out = np.asarray(arr, dtype=np.float32, copy=False)
    veiling = float(cfg.get("veiling_glare_fraction", 0.0))
    veiling = float(np.clip(veiling, 0.0, 1.0))
    if veiling > 0.0:
        mean_l = float(np.mean(out))
        out = (1.0 - veiling) * out + veiling * mean_l

    halo_sigma = float(cfg.get("halo_sigma_pixels", 0.0))
    halo_strength = float(cfg.get("halo_strength", 0.0))
    halo_strength = float(np.clip(halo_strength, 0.0, 1.0))
    if halo_sigma > 0.0 and halo_strength > 0.0:
        halo = separable_gaussian_blur_2d(out, halo_sigma)
        # Renormalize to avoid creating net scene energy.
        out = (out + halo_strength * halo) / (1.0 + halo_strength)

    return out.astype(np.float32, copy=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    repo_default = Path(__file__).resolve().parent.parent
    ap.add_argument("--repo-root", type=Path, default=repo_default)
    ap.add_argument("--exr-in", type=Path, required=True, help="Input multispectral EXR.")
    ap.add_argument("--exr-out", type=Path, default=None, help="Output EXR (default: overwrite --exr-in).")
    ap.add_argument("--camera-model-config", type=Path, default=None, help="Camera model YAML path (preferred).")
    ap.add_argument("--config", type=Path, default=None, help="config/optics.yaml")
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    if args.camera_model_config is not None:
        cfg_path = args.camera_model_config.resolve()
        if not cfg_path.is_file():
            print(f"error: missing {cfg_path}", file=sys.stderr)
            sys.exit(1)
        camera_model = load_camera_model(cfg_path)
        psf = (camera_model.get("lens", {}) or {}).get("post_psf", {}) or {}
    else:
        cfg_path = (args.config or (repo / "config" / "optics.yaml")).resolve()
        if not cfg_path.is_file():
            print(f"error: missing {cfg_path}", file=sys.stderr)
            sys.exit(1)
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        psf = cfg.get("post_psf", {}) or {}
    if not bool(psf.get("enabled", False)):
        print("post_psf.enabled is false; nothing to do.", file=sys.stderr)
        sys.exit(0)

    mode = str(psf.get("mode", "gaussian")).lower()
    if mode == "none":
        print('post_psf.mode is "none"; nothing to do.', file=sys.stderr)
        sys.exit(0)
    if mode != "gaussian":
        raise ValueError(f'post_psf.mode must be "gaussian" or "none", got {mode!r}')

    sigma = float(psf.get("sigma_pixels", 0.0))
    if sigma < 0:
        raise ValueError("sigma_pixels must be >= 0")
    stray = psf.get("stray_light", {}) or {}

    exr_in = args.exr_in if args.exr_in.is_absolute() else (repo / args.exr_in).resolve()
    exr_out = (args.exr_out or exr_in).resolve()
    if not exr_in.is_file():
        raise FileNotFoundError(exr_in)

    chans = read_separate_exr_channels(exr_in)
    out: dict[str, np.ndarray] = {}
    for name, arr in chans.items():
        blurred = separable_gaussian_blur_2d(arr, sigma)
        out[name] = apply_stray_light(blurred, stray)

    exr_out.parent.mkdir(parents=True, exist_ok=True)
    write_separate_channels_exr(exr_out, out)
    print(
        f"wrote {exr_out} (gaussian sigma_px={sigma}, "
        f"stray_light={'on' if bool(stray.get('enabled', False)) else 'off'})"
    )


if __name__ == "__main__":
    main()
