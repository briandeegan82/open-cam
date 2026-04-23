#!/usr/bin/env python3
"""Apply a YAML-driven EMVA-style sensor/noise model to a linear EXR render."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

from camera_model import load_camera_model, noise_config_from_camera_model
from exr_multispectral import (
    linear_rgb_from_exr,
    spectral_buckets_from_exr,
    trapezoid_weights_nm,
)


def read_csv_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    wl: list[float] = []
    val: list[float] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r",\s*", line, maxsplit=1)
        if len(parts) != 2:
            continue
        wl.append(float(parts[0]))
        val.append(float(parts[1]))
    if not wl:
        raise ValueError(f"no data in CSV curve: {path}")
    w = np.asarray(wl, dtype=np.float64)
    v = np.asarray(val, dtype=np.float64)

    ok = np.isfinite(w) & np.isfinite(v)
    w = w[ok]
    v = v[ok]
    if w.size == 0:
        raise ValueError(f"no finite samples in CSV curve: {path}")

    # Some imported camera QE CSVs are normalized-domain traces (0..1-ish) rather than nm.
    # Map them into a visible wavelength domain so interpolation onto spectral buckets is valid.
    if float(np.max(w)) <= 10.0:
        wmin = float(np.min(w))
        wmax = float(np.max(w))
        if wmax - wmin <= 1e-12:
            raise ValueError(f"invalid wavelength axis in CSV curve: {path}")
        w = 380.0 + (w - wmin) * (400.0 / (wmax - wmin))
        print(f"warning: mapped normalized wavelength axis to 380..780 nm for {path}", file=sys.stderr)

    idx = np.argsort(w)
    return w[idx], v[idx]


def load_qe_curves_rgb(repo: Path, qe_cfg: dict) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Load QE curves and auto-correct likely imported red/blue inversion."""
    r = read_csv_curve((repo / qe_cfg["red_csv"]).resolve())
    g = read_csv_curve((repo / qe_cfg["green_csv"]).resolve())
    b = read_csv_curve((repo / qe_cfg["blue_csv"]).resolve())
    r_peak = float(r[0][int(np.argmax(r[1]))])
    g_peak = float(g[0][int(np.argmax(g[1]))])
    b_peak = float(b[0][int(np.argmax(b[1]))])
    if r_peak < g_peak and b_peak > g_peak:
        print(
            "warning: detected likely QE red/blue inversion; swapping channels "
            f"(red_peak={r_peak:.1f}nm, green_peak={g_peak:.1f}nm, blue_peak={b_peak:.1f}nm)",
            file=sys.stderr,
        )
        r, b = b, r
    return r, g, b


def mean_effective_qe(curve_csv: Path, ircf_csv: Path | None) -> float:
    wl, q = read_csv_curve(curve_csv)
    if ircf_csv is not None:
        ir_wl, ir = read_csv_curve(ircf_csv)
        ir_i = np.interp(wl, ir_wl, ir, left=0.0, right=0.0)
        q = q * ir_i
    q = np.clip(q, 0.0, 1.0)
    return float(np.mean(q))


def load_electrons_npz(path: Path) -> np.ndarray:
    data = np.load(path)
    if "electrons_rgb" not in data:
        raise ValueError(f"{path} missing 'electrons_rgb' array")
    arr = np.asarray(data["electrons_rgb"], dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"expected HxWx3 electrons_rgb, got shape {arr.shape}")
    return np.clip(arr[:, :, :3], 0.0, None)


def to_png16(rgb_dn: np.ndarray, bit_depth: int) -> np.ndarray:
    max_dn = float((1 << bit_depth) - 1)
    if max_dn <= 0:
        raise ValueError("invalid bit depth")
    scaled = np.clip(rgb_dn / max_dn, 0.0, 1.0)
    return np.rint(scaled * 65535.0).astype(np.uint16)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1 sRGB OETF: linear light [0,1] -> nonlinear sRGB [0,1]."""
    x = np.clip(np.asarray(linear, dtype=np.float64), 0.0, 1.0)
    lo = 12.92 * x
    hi = 1.055 * np.power(x, 1.0 / 2.4) - 0.055
    return np.where(x <= 0.0031308, lo, hi).astype(np.float32)


def gray_world_gains(rgb: np.ndarray) -> np.ndarray:
    """Compute per-channel gray-world gains from an HxWx3 image."""
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"gray_world_gains expects HxWx3, got {rgb.shape}")
    m = np.mean(np.asarray(rgb[:, :, :3], dtype=np.float64), axis=(0, 1))
    target = float(np.mean(m))
    eps = 1e-9
    gains = target / np.maximum(m, eps)
    return gains.astype(np.float32)


def apply_rgb_gains(rgb: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """Apply RGB channel gains to HxWx3 array."""
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"apply_rgb_gains expects HxWx3, got {rgb.shape}")
    g = np.asarray(gains, dtype=np.float32).reshape(1, 1, 3)
    return (np.asarray(rgb[:, :, :3], dtype=np.float32) * g).astype(np.float32)


def apply_preview_wb_dn(rgb_dn: np.ndarray, black_dn: float, gains: np.ndarray) -> np.ndarray:
    """Apply WB gains in signal domain (DN above black), preserving black pedestal."""
    sig = np.clip(np.asarray(rgb_dn, dtype=np.float32) - float(black_dn), 0.0, None)
    sig_wb = apply_rgb_gains(sig, gains)
    return np.clip(sig_wb + float(black_dn), 0.0, None).astype(np.float32)


def fit_ccm_lstsq(
    src_rgb: np.ndarray,
    tgt_rgb: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Fit a 3x3 CCM (src @ M ~= tgt) by least squares."""
    s = np.asarray(src_rgb, dtype=np.float64).reshape(-1, 3)
    t = np.asarray(tgt_rgb, dtype=np.float64).reshape(-1, 3)
    if s.shape != t.shape:
        raise ValueError(f"CCM fit expects matching shapes, got {s.shape} vs {t.shape}")
    if mask is not None:
        m = np.asarray(mask, dtype=bool).reshape(-1)
        if m.size != s.shape[0]:
            raise ValueError("CCM mask size mismatch")
        s = s[m]
        t = t[m]
    if s.shape[0] < 16:
        return np.eye(3, dtype=np.float32)
    # Match target intensity scale to source to avoid near-zero CCM coefficients.
    s_l = np.mean(s, axis=1)
    t_l = np.mean(t, axis=1)
    s_ref = float(np.percentile(s_l, 50.0))
    t_ref = float(np.percentile(t_l, 50.0))
    if t_ref > 1e-9:
        t = t * (s_ref / t_ref)
    M, *_ = np.linalg.lstsq(s, t, rcond=None)
    return M.astype(np.float32)


def apply_ccm(rgb: np.ndarray, ccm: np.ndarray) -> np.ndarray:
    """Apply a 3x3 color correction matrix to HxWx3 image."""
    x = np.asarray(rgb, dtype=np.float32)
    y = np.tensordot(x, np.asarray(ccm, dtype=np.float32), axes=([2], [0]))
    return np.clip(y, 0.0, None).astype(np.float32)


def sanitize_ccm(ccm: np.ndarray) -> np.ndarray:
    """Reject pathological CCMs that collapse preview intensity."""
    m = np.asarray(ccm, dtype=np.float32)
    if not np.all(np.isfinite(m)):
        return np.eye(3, dtype=np.float32)
    # Keep average channel gain in a sane range for preview mapping.
    tr = float(np.trace(m) / 3.0)
    if tr < 0.2 or tr > 5.0:
        return np.eye(3, dtype=np.float32)
    return m


def fit_diag_ccm(src_rgb: np.ndarray, tgt_rgb: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Fit a diagonal CCM from channel chromaticity (brightness preserving)."""
    s = np.asarray(src_rgb, dtype=np.float64).reshape(-1, 3)
    t = np.asarray(tgt_rgb, dtype=np.float64).reshape(-1, 3)
    if mask is not None:
        m = np.asarray(mask, dtype=bool).reshape(-1)
        s = s[m]
        t = t[m]
    if s.shape[0] < 16:
        return np.eye(3, dtype=np.float32)
    ms = np.mean(s, axis=0)
    mt = np.mean(t, axis=0)
    # Match color balance, not absolute brightness; preview stretch handles exposure.
    ms_n = ms / max(1e-9, float(np.mean(ms)))
    mt_n = mt / max(1e-9, float(np.mean(mt)))
    g = mt_n / np.maximum(ms_n, 1e-9)
    # Clamp to keep preview corrections stable.
    g = np.clip(g, 0.25, 4.0)
    return np.diag(g.astype(np.float32))


def to_png8_preview(
    rgb_dn: np.ndarray,
    bit_depth: int,
    black_dn: float,
    white_dn: float | None = None,
    *,
    srgb: bool = False,
) -> np.ndarray:
    """Make a viewable preview PNG from DN values.

    DN around black level can look nearly black in 8-bit if mapped against full
    ADC range. This maps [black_dn, white_dn] to linear [0, 1], then to 8-bit.
    If ``srgb`` is True, apply the sRGB transfer (gamma) after linear stretch.
    """
    max_dn = float((1 << bit_depth) - 1)
    hi = float(white_dn) if white_dn is not None else max_dn
    hi = min(max_dn, max(black_dn + 1.0, hi))
    scaled = (rgb_dn - black_dn) / (hi - black_dn)
    scaled = np.clip(scaled, 0.0, 1.0)
    if srgb:
        scaled = linear_to_srgb(scaled)
    return np.rint(scaled * 255.0).astype(np.uint8)


def write_png(path: Path, arr: np.ndarray) -> None:
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise RuntimeError("imageio is required to write PNG outputs.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, arr)


def write_raw16(path: Path, raw_u16: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_u16.astype(np.uint16).tofile(path)


_BAYER_PHASE: dict[str, tuple[int, int, int, int]] = {
    # Phase (row%2, col%2): (0,0), (0,1), (1,0), (1,1) -> R=0, G=1, B=2
    "RGGB": (0, 1, 1, 2),
    "BGGR": (2, 1, 1, 0),
    "GRBG": (1, 0, 2, 1),
    "GBRG": (1, 2, 0, 1),
}

# Roll raw by (-nr,-nc) so pixel (0,0) sits on an R site in RGGB order.
_BAYER_NEG_ROLL_TO_RGGB: dict[str, tuple[int, int]] = {
    "RGGB": (0, 0),
    "BGGR": (1, 1),
    "GRBG": (0, 1),
    "GBRG": (1, 0),
}


def _demosaic_rggb_bilinear(raw: np.ndarray) -> np.ndarray:
    """Bilinear demosaic assuming RGGB layout. raw is HxW; returns HxWx3 (R,G,B)."""
    raw = np.asarray(raw, dtype=np.float64)
    h0, w0 = int(raw.shape[0]), int(raw.shape[1])
    ph, pw = h0 % 2, w0 % 2
    if ph or pw:
        raw = np.pad(raw, ((0, ph), (0, pw)), mode="edge")
    h, w = int(raw.shape[0]), int(raw.shape[1])
    # Reflect avoids constant-edge pads that place R/G/B at wrong phase on borders.
    xp = np.pad(raw, ((1, 1), (1, 1)), mode="reflect")
    R = np.zeros((h, w))
    G = np.zeros((h, w))
    B = np.zeros((h, w))
    R[0::2, 0::2] = raw[0::2, 0::2]
    G[0::2, 1::2] = raw[0::2, 1::2]
    G[1::2, 0::2] = raw[1::2, 0::2]
    B[1::2, 1::2] = raw[1::2, 1::2]

    # R: greens on red rows (horizontal R at j±1), greens on blue rows (vertical R at i±1).
    # raw(i,j)=xp[i+1,j+1]; e.g. G at (0,1) uses R at xp[1,1] and xp[1,3].
    R[0::2, 1::2] = 0.5 * (xp[1 : h + 1 : 2, 1:w:2] + xp[1 : h + 1 : 2, 3 : w + 2 : 2])
    R[1::2, 0::2] = 0.5 * (xp[1:h:2, 1 : w + 1 : 2] + xp[3 : h + 2 : 2, 1 : w + 1 : 2])
    # R at B pixels (odd, odd): average four diagonal R neighbors.
    R[1::2, 1::2] = 0.25 * (
        xp[1:h:2, 1:w:2] + xp[1:h:2, 3 : w + 2 : 2] + xp[3 : h + 2 : 2, 1:w:2] + xp[3 : h + 2 : 2, 3 : w + 2 : 2]
    )

    # G: at R and B sites (cross neighbors).
    G[0::2, 0::2] = 0.25 * (
        xp[0:h - 1 : 2, 1 : w + 1 : 2]
        + xp[2 : h + 1 : 2, 1 : w + 1 : 2]
        + xp[1 : h + 1 : 2, 0:w:2]
        + xp[1 : h + 1 : 2, 2 : w + 1 : 2]
    )
    G[1::2, 1::2] = 0.25 * (
        xp[1:h:2, 2 : w + 1 : 2]
        + xp[3 : h + 2 : 2, 2 : w + 1 : 2]
        + xp[2 : h + 1 : 2, 1:w:2]
        + xp[2 : h + 1 : 2, 3 : w + 2 : 2]
    )

    # B: G on blue row (horizontal B); G on red row (vertical B). Mirrors the R green-site logic.
    B[1::2, 0::2] = 0.5 * (xp[2 : h + 1 : 2, 0:w - 1 : 2] + xp[2 : h + 1 : 2, 2 : w + 1 : 2])
    B[0::2, 1::2] = 0.5 * (xp[0:h - 1 : 2, 2 : w + 1 : 2] + xp[2 : h + 1 : 2, 2 : w + 1 : 2])
    # B at R pixels (even, even): average four diagonal B neighbors.
    B[0::2, 0::2] = 0.25 * (
        xp[0:h - 1 : 2, 0:w - 1 : 2]
        + xp[0:h - 1 : 2, 2 : w + 1 : 2]
        + xp[2 : h + 1 : 2, 0:w - 1 : 2]
        + xp[2 : h + 1 : 2, 2 : w + 1 : 2]
    )

    out = np.stack([R, G, B], axis=2)
    if ph or pw:
        out = out[:h0, :w0, :]
    return out


def demosaic_requested(bayer_cfg: dict) -> bool:
    d = bayer_cfg.get("demosaic", "bilinear")
    if d is None or d is False:
        return False
    if d is True:
        return True
    s = str(d).lower().strip()
    if s in ("0", "false", "none", "off", "no"):
        return False
    if s in ("1", "true", "yes", "on", "bilinear"):
        return True
    raise ValueError(f'unsupported cfa.demosaic value {d!r}; use true/false or "bilinear"')


def bilinear_demosaic(mono: np.ndarray, pattern: str) -> np.ndarray:
    """CFA HxW -> full RGB HxWx3 using edge-padded bilinear interpolation."""
    p = pattern.upper()
    if p not in _BAYER_NEG_ROLL_TO_RGGB:
        raise ValueError(f"unknown Bayer pattern {pattern!r}")
    nr, nc = _BAYER_NEG_ROLL_TO_RGGB[p]
    r = mono
    if nr != 0 or nc != 0:
        r = np.roll(np.roll(mono, -nr, axis=0), -nc, axis=1)
    rgb = _demosaic_rggb_bilinear(r)
    if nr != 0 or nc != 0:
        rgb = np.roll(np.roll(rgb, nr, axis=0), nc, axis=1)
    return rgb.astype(np.float32)


def bayer_sample_rgb(rgb: np.ndarray, pattern: str) -> np.ndarray:
    """One photosite per pixel: HxWx3 -> HxW mono electrons (or DN later)."""
    p = pattern.upper()
    if p not in _BAYER_PHASE:
        raise ValueError(f"unknown Bayer pattern {pattern!r}; use one of {sorted(_BAYER_PHASE)}")
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"expected HxWx3 for Bayer sampling, got {rgb.shape}")
    lut = np.array(_BAYER_PHASE[p], dtype=np.intp)
    h, w = rgb.shape[0], rgb.shape[1]
    jj, ii = np.meshgrid(np.arange(h, dtype=np.intp), np.arange(w, dtype=np.intp), indexing="ij")
    tid = (jj & 1) * 2 + (ii & 1)
    ch = lut[tid]
    return rgb[jj, ii, ch]


def to_png8_preview_gray(
    mono_dn: np.ndarray,
    bit_depth: int,
    black_dn: float,
    white_dn: float | None = None,
    *,
    srgb: bool = False,
) -> np.ndarray:
    """Stack mono DN to RGB for an 8-bit preview."""
    rgb = np.stack([mono_dn, mono_dn, mono_dn], axis=2)
    return to_png8_preview(rgb, bit_depth, black_dn=black_dn, white_dn=white_dn, srgb=srgb)


def mono_dn_to_u16(mono_dn: np.ndarray, bit_depth: int) -> np.ndarray:
    max_dn = float((1 << bit_depth) - 1)
    scaled = np.clip(mono_dn / max_dn, 0.0, 1.0)
    return np.rint(scaled * 65535.0).astype(np.uint16)


def _require_positive(name: str, value: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return value


def qe_curve_on_lambdas(
    lambdas_nm: np.ndarray,
    qe_csv: Path,
    ircf_csv: Path | None,
) -> np.ndarray:
    wl, q = read_csv_curve(qe_csv)
    if ircf_csv is not None:
        ir_wl, ir = read_csv_curve(ircf_csv)
        ir_i = np.interp(wl, ir_wl, ir, left=0.0, right=0.0)
        q = q * ir_i
    q = np.clip(q, 0.0, 1.0)
    return np.interp(lambdas_nm, wl, q, left=0.0, right=0.0).astype(np.float32)


def integrate_exr_spectral_qe(
    exr_path: Path,
    repo: Path,
    qe_cfg: dict,
) -> np.ndarray:
    """HxWx3 proxy signal: Σ_k L(λ_k) QE_c(λ_k) w_k (trapezoid weights in nm)."""
    spec, lam = spectral_buckets_from_exr(exr_path.resolve())
    w = trapezoid_weights_nm(lam).astype(np.float32)
    ircf = qe_cfg.get("ircf_csv")
    ircf_path = (repo / ircf).resolve() if ircf else None
    q_r_src, q_g_src, q_b_src = load_qe_curves_rgb(repo, qe_cfg)
    q_r = np.interp(lam, q_r_src[0], np.clip(q_r_src[1], 0.0, 1.0), left=0.0, right=0.0).astype(np.float32)
    q_g = np.interp(lam, q_g_src[0], np.clip(q_g_src[1], 0.0, 1.0), left=0.0, right=0.0).astype(np.float32)
    q_b = np.interp(lam, q_b_src[0], np.clip(q_b_src[1], 0.0, 1.0), left=0.0, right=0.0).astype(np.float32)
    if ircf_path is not None:
        ir_wl, ir = read_csv_curve(ircf_path)
        ir_i = np.interp(lam, ir_wl, np.clip(ir, 0.0, 1.0), left=0.0, right=0.0).astype(np.float32)
        q_r *= ir_i
        q_g *= ir_i
        q_b *= ir_i
    acc_r = np.sum(spec * (q_r * w), axis=2)
    acc_g = np.sum(spec * (q_g * w), axis=2)
    acc_b = np.sum(spec * (q_b * w), axis=2)
    return np.stack([acc_r, acc_g, acc_b], axis=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    repo_default = Path(__file__).resolve().parent.parent
    ap.add_argument("--repo-root", type=Path, default=repo_default)
    ap.add_argument("--config", type=Path, default=None, help="YAML config path")
    ap.add_argument(
        "--camera-model-config",
        type=Path,
        default=None,
        help="Camera model YAML path (preferred).",
    )
    ap.add_argument("--electrons-npz", type=Path, default=None, help="Optional precomputed electrons image (HxWx3).")
    ap.add_argument(
        "--linear-exr",
        type=Path,
        default=None,
        help="Override output.linear_rgb_in (use with pipeline paths.exr_out).",
    )
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument(
        "--auto-exposure",
        action="store_true",
        help="Enable percentile-based exposure normalization to full-well.",
    )
    ap.add_argument(
        "--exposure-scale",
        type=float,
        default=None,
        help="Manual electrons-per-unit scale (overrides YAML when provided).",
    )
    ap.add_argument(
        "--target-fullwell-percentile",
        type=float,
        default=99.5,
        help="Percentile used for auto exposure scaling to full well",
    )
    ap.add_argument(
        "--target-fullwell-fraction",
        type=float,
        default=0.8,
        help="Auto exposure target fraction of full well at selected percentile",
    )
    ap.add_argument(
        "--preview-percentile",
        type=float,
        default=99.5,
        help="Percentile used to set white point for preview PNGs.",
    )
    ap.add_argument(
        "--preview-no-normalize",
        action="store_true",
        help="Disable preview percentile normalization; map PNGs against full ADC range.",
    )
    ap.add_argument(
        "--integration-time-s",
        type=float,
        default=None,
        help="Optional integration time override in seconds (replaces sensor.integration_time_s).",
    )
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    cfg_path = None
    if args.camera_model_config is not None:
        cfg_path = args.camera_model_config.resolve()
        if not cfg_path.is_file():
            raise FileNotFoundError(f"missing camera model config: {cfg_path}")
        camera_model = load_camera_model(cfg_path)
        cfg = noise_config_from_camera_model(
            camera_model,
            linear_rgb_in="out/colorchecker_spectral.exr",
            raw_out="out/colorchecker_noisy.raw16",
        )
    else:
        cfg_path = (args.config or (repo / "config" / "noise_emva.yaml")).resolve()
        if not cfg_path.is_file():
            raise FileNotFoundError(f"missing config: {cfg_path}")
        cfg = yaml.safe_load(cfg_path.read_text())
    sensor = cfg.get("sensor", {})
    qe_cfg = sensor.get("quantum_efficiency", {})
    emva = cfg.get("emva", {})
    adc = cfg.get("adc", {})
    out_cfg = cfg.get("output", {})
    proc_cfg = cfg.get("processing", {})
    bayer_cfg = cfg.get("bayer") or {}

    exr_in = (repo / out_cfg.get("linear_rgb_in", "out/colorchecker.exr")).resolve()
    if args.linear_exr is not None:
        exr_in = args.linear_exr.resolve()
    raw_out = (repo / out_cfg.get("raw_out", "out/colorchecker_noisy.raw16")).resolve()
    png_dir = raw_out.parent / f"{raw_out.stem}_png"

    electrons_npz = args.electrons_npz.resolve() if args.electrons_npz is not None else None

    ircf = qe_cfg.get("ircf_csv")
    ircf_path = (repo / ircf).resolve() if ircf else None
    q_r_src, q_g_src, q_b_src = load_qe_curves_rgb(repo, qe_cfg)
    qe_r = 0.0
    qe_g = 0.0
    qe_b = 0.0
    for i, (q_wl, q_v) in enumerate((q_r_src, q_g_src, q_b_src)):
        q = np.clip(np.asarray(q_v, dtype=np.float64), 0.0, 1.0)
        if ircf_path is not None:
            ir_wl, ir = read_csv_curve(ircf_path)
            ir_i = np.interp(q_wl, ir_wl, ir, left=0.0, right=0.0)
            q = np.clip(q * ir_i, 0.0, 1.0)
        if i == 0:
            qe_r = float(np.mean(q))
        elif i == 1:
            qe_g = float(np.mean(q))
        else:
            qe_b = float(np.mean(q))
    qe_vec = np.array([qe_r, qe_g, qe_b], dtype=np.float32)
    qe_max = float(np.max(qe_vec))
    if not np.isfinite(qe_max) or qe_max <= 0.0:
        raise ValueError(
            "computed QE response is non-positive; check QE/IRCF curves in camera model "
            f"{cfg_path}"
        )
    qe_vec /= qe_max

    full_well_e = float(adc.get("full_well_e", 13500.0))
    K_e_per_DN = float(emva.get("overall_system_gain_K_e_per_DN", 0.08))
    sigma_d_e = float(emva.get("sigma_d_e", 2.0))
    dsnu_std_e = float(emva.get("dsnu_std_e", 0.3))
    prnu_std = float(emva.get("prnu_std_fraction", 0.005))
    black_dn = float(emva.get("black_level_DN", 64.0))
    bit_depth = int(adc.get("bit_depth", 12))
    if bit_depth < 1:
        raise ValueError(f"adc.bit_depth must be >= 1, got {bit_depth!r}")
    full_well_e = _require_positive("adc.full_well_e", full_well_e)
    K_e_per_DN = _require_positive("emva.overall_system_gain_K_e_per_DN", K_e_per_DN)
    use_poisson = bool(emva.get("use_poisson_shot_noise", True))
    t_int_s = float(sensor.get("integration_time_s", 0.01))
    if args.integration_time_s is not None:
        t_int_s = float(args.integration_time_s)
    _require_positive("sensor.integration_time_s", t_int_s)
    dark_current_e_per_s = float(emva.get("dark_current_e_per_s", 0.0))
    dark_ref_temp_c = float(emva.get("dark_current_reference_temp_c", 20.0))
    dark_temp_c = float(emva.get("temperature_c", dark_ref_temp_c))
    dark_doubling_c = float(emva.get("dark_current_doubling_per_c", 6.0))
    row_fpn_std_e = float(emva.get("row_fpn_std_e", 0.0))
    col_fpn_std_e = float(emva.get("column_fpn_std_e", 0.0))
    adc_inl_quad_fraction = float(adc.get("inl_quadratic_fraction", 0.0))
    adc_dnl_std_lsb = float(adc.get("dnl_std_lsb", 0.0))
    adc_clipping = bool(adc.get("clipping", True))

    exr_mode = str(proc_cfg.get("linear_exr_mode", "rgb")).lower()
    if exr_mode not in ("rgb", "integrate_qe"):
        raise ValueError('processing.linear_exr_mode must be "rgb" or "integrate_qe"')

    yaml_auto = bool(proc_cfg.get("auto_exposure", False))
    auto_exposure = bool(args.auto_exposure or yaml_auto)
    exposure_scale = float(proc_cfg.get("exposure_scale_e_per_unit", 1.0))
    if args.exposure_scale is not None:
        exposure_scale = float(args.exposure_scale)
    wb_cfg = proc_cfg.get("preview_white_balance", {}) or {}
    wb_enabled = bool(wb_cfg.get("enabled", True))
    wb_method = str(wb_cfg.get("method", "gray_world")).lower().strip()
    if wb_enabled and wb_method != "gray_world":
        raise ValueError('processing.preview_white_balance.method must be "gray_world"')
    ccm_cfg = proc_cfg.get("preview_color_correction", {}) or {}
    ccm_enabled = bool(ccm_cfg.get("enabled", True))
    ccm_method = str(ccm_cfg.get("method", "diag_exr_reference")).lower().strip()
    if ccm_enabled and ccm_method not in ("diag_exr_reference", "lstsq_exr_reference"):
        raise ValueError(
            'processing.preview_color_correction.method must be "diag_exr_reference" or "lstsq_exr_reference"'
        )

    signal_source = "linear_exr"
    if electrons_npz is not None:
        signal_e = load_electrons_npz(electrons_npz)
        signal_source = "electrons_npz"
        if auto_exposure:
            pctl = float(np.percentile(signal_e, args.target_fullwell_percentile))
            if pctl <= 0:
                raise RuntimeError("electrons_npz has non-positive signal for auto exposure")
            exposure_scale = float(args.target_fullwell_fraction) * full_well_e / pctl
        signal_e = signal_e * exposure_scale
    else:
        if exr_mode == "integrate_qe":
            signal_source = "linear_exr_integrate_qe"
            try:
                rgb = integrate_exr_spectral_qe(exr_in, repo, qe_cfg)
                rgb = np.clip(rgb, 0.0, None)
                e_nominal = rgb
            except ValueError as exc:
                # Some renders are RGB EXRs even when integrate_qe is configured.
                # Fall back to per-channel QE scaling so mixed pipelines do not fail.
                if "no S0.*nm spectral channels" not in str(exc):
                    raise
                print(
                    f"Warning: integrate_qe requested but EXR has no spectral planes: {exr_in}. "
                    "Falling back to RGB QE scaling."
                )
                signal_source = "linear_exr_rgb_fallback"
                rgb = linear_rgb_from_exr(exr_in)
                rgb = np.clip(rgb, 0.0, None)
                e_nominal = rgb * qe_vec[None, None, :]
        else:
            rgb = linear_rgb_from_exr(exr_in)
            rgb = np.clip(rgb, 0.0, None)
            e_nominal = rgb * qe_vec[None, None, :]

        if auto_exposure:
            pctl = float(np.percentile(e_nominal, args.target_fullwell_percentile))
            if pctl <= 0:
                raise RuntimeError("input image has zero signal after QE weighting")
            exposure_scale = float(args.target_fullwell_fraction) * full_well_e / pctl
        signal_e = e_nominal * exposure_scale

    rng = np.random.default_rng(args.seed)

    bayer_on = bool(bayer_cfg.get("enabled", False))
    bayer_pat = str(bayer_cfg.get("pattern", "RGGB"))
    demosaic_on = bayer_on and demosaic_requested(bayer_cfg)
    demosaic_srgb = bool(bayer_cfg.get("demosaic_srgb", False))

    if bayer_on:
        signal_e = bayer_sample_rgb(signal_e, bayer_pat)

    # Fixed-pattern components.
    prnu_map = 1.0 + rng.normal(0.0, prnu_std, size=signal_e.shape).astype(np.float32)
    dark_temp_scale = 2.0 ** ((dark_temp_c - dark_ref_temp_c) / max(1e-6, dark_doubling_c))
    dark_mean_e = max(0.0, dark_current_e_per_s * t_int_s * dark_temp_scale)
    dark_e = rng.poisson(dark_mean_e, size=signal_e.shape).astype(np.float32)
    if signal_e.ndim == 2:
        row_fpn = rng.normal(0.0, row_fpn_std_e, size=(signal_e.shape[0], 1)).astype(np.float32)
        col_fpn = rng.normal(0.0, col_fpn_std_e, size=(1, signal_e.shape[1])).astype(np.float32)
        rc_fpn = row_fpn + col_fpn
    else:
        row_fpn = rng.normal(0.0, row_fpn_std_e, size=(signal_e.shape[0], 1, 1)).astype(np.float32)
        col_fpn = rng.normal(0.0, col_fpn_std_e, size=(1, signal_e.shape[1], 1)).astype(np.float32)
        rc_fpn = row_fpn + col_fpn
    if signal_e.ndim == 2:
        dsnu_map = rng.normal(0.0, dsnu_std_e, size=signal_e.shape).astype(np.float32)
        signal_fpn_e = np.clip(signal_e * prnu_map + dsnu_map + rc_fpn + dark_e, 0.0, None)
    else:
        dsnu_map = rng.normal(0.0, dsnu_std_e, size=signal_e.shape[:2]).astype(np.float32)
        dsnu_map = dsnu_map[:, :, None]
        signal_fpn_e = np.clip(signal_e * prnu_map + dsnu_map + rc_fpn + dark_e, 0.0, None)

    if use_poisson:
        shot_e = rng.poisson(np.maximum(signal_fpn_e, 0.0)).astype(np.float32)
    else:
        shot_e = signal_fpn_e.astype(np.float32)

    read_e = rng.normal(0.0, sigma_d_e, size=signal_e.shape).astype(np.float32)
    if adc_clipping:
        total_e = np.clip(shot_e + read_e, 0.0, full_well_e)
    else:
        total_e = np.clip(shot_e + read_e, 0.0, None)

    max_dn = float((1 << bit_depth) - 1)
    if adc_clipping:
        dn_clean = np.clip(signal_e / K_e_per_DN + black_dn, 0.0, max_dn)
        dn_noisy = np.clip(total_e / K_e_per_DN + black_dn, 0.0, max_dn)
    else:
        dn_clean = np.clip(signal_e / K_e_per_DN + black_dn, 0.0, None)
        dn_noisy = np.clip(total_e / K_e_per_DN + black_dn, 0.0, None)
    if adc_inl_quad_fraction != 0.0:
        x = np.clip((dn_noisy - black_dn) / max(1.0, max_dn - black_dn), 0.0, 1.0)
        bow = x * (1.0 - x)
        dn_noisy = np.clip(dn_noisy + (adc_inl_quad_fraction * max_dn) * bow, 0.0, max_dn)
    if adc_dnl_std_lsb > 0.0:
        dn_noisy = np.clip(dn_noisy + rng.normal(0.0, adc_dnl_std_lsb, size=dn_noisy.shape), 0.0, max_dn)

    if bayer_on:
        raw_u16 = np.rint(dn_noisy).astype(np.uint16)
    else:
        raw_u16 = np.rint(np.mean(dn_noisy, axis=2)).astype(np.uint16)
    write_raw16(raw_out, raw_u16)

    wb_gains = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    ccm = np.eye(3, dtype=np.float32)
    ccm_source = "disabled"
    ref_linear = None
    if ccm_enabled:
        try:
            ref_linear = np.clip(linear_rgb_from_exr(exr_in), 0.0, None).astype(np.float32)
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
            ref_linear = None
            ccm_source = "reference_unavailable"
            print(f"warning: preview CCM reference unavailable from {exr_in}: {exc}", file=sys.stderr)
    if wb_enabled and not bayer_on:
        wb_gains = gray_world_gains(np.clip(dn_clean - black_dn, 0.0, None))
        dn_clean = apply_preview_wb_dn(dn_clean, black_dn, wb_gains)
        dn_noisy = apply_preview_wb_dn(dn_noisy, black_dn, wb_gains)
    if ccm_enabled and not bayer_on and ref_linear is not None and ref_linear.shape == dn_clean.shape:
        src = np.clip(dn_clean - black_dn, 0.0, None).astype(np.float32)
        # Ignore very dark pixels so background does not dominate fit.
        yl = np.mean(ref_linear, axis=2)
        lo = np.percentile(yl, 5.0)
        hi = np.percentile(yl, 99.5)
        fit_mask = (yl >= lo) & (yl <= hi)
        if ccm_method == "lstsq_exr_reference":
            ccm = fit_ccm_lstsq(src, ref_linear, fit_mask)
        else:
            ccm = fit_diag_ccm(src, ref_linear, fit_mask)
        ccm = sanitize_ccm(ccm)
        dn_clean = np.clip(apply_ccm(src, ccm) + black_dn, 0.0, None)
        dn_noisy = np.clip(apply_ccm(np.clip(dn_noisy - black_dn, 0.0, None), ccm) + black_dn, 0.0, None)
        ccm_source = ccm_method

    preview_no_normalize = bool(args.preview_no_normalize)
    preview_white_clean = None if preview_no_normalize else float(np.percentile(dn_clean, args.preview_percentile))
    preview_white_noisy = None if preview_no_normalize else float(np.percentile(dn_noisy, args.preview_percentile))
    if bayer_on:
        clean_png8 = to_png8_preview_gray(dn_clean, bit_depth, black_dn=black_dn, white_dn=preview_white_clean)
        noisy_png8 = to_png8_preview_gray(dn_noisy, bit_depth, black_dn=black_dn, white_dn=preview_white_noisy)
    else:
        clean_png8 = to_png8_preview(dn_clean, bit_depth, black_dn=black_dn, white_dn=preview_white_clean)
        noisy_png8 = to_png8_preview(dn_noisy, bit_depth, black_dn=black_dn, white_dn=preview_white_noisy)
    write_png(png_dir / "clean_rgb8.png", clean_png8)
    write_png(png_dir / "noisy_rgb8.png", noisy_png8)

    if bayer_on:
        write_png(png_dir / "noisy_mono_16.png", mono_dn_to_u16(dn_noisy, bit_depth))
    else:
        for i, name in enumerate(["R", "G", "B"]):
            ch = to_png16(dn_noisy[:, :, i], bit_depth)
            write_png(png_dir / f"noisy_{name}_16.png", ch)

    if demosaic_on:
        dn_clean_rgb = bilinear_demosaic(dn_clean, bayer_pat)
        dn_noisy_rgb = bilinear_demosaic(dn_noisy, bayer_pat)
        if wb_enabled:
            wb_gains = gray_world_gains(np.clip(dn_clean_rgb - black_dn, 0.0, None))
            dn_clean_rgb = apply_preview_wb_dn(dn_clean_rgb, black_dn, wb_gains)
            dn_noisy_rgb = apply_preview_wb_dn(dn_noisy_rgb, black_dn, wb_gains)
        if ccm_enabled and ref_linear is not None and ref_linear.shape == dn_clean_rgb.shape:
            src = np.clip(dn_clean_rgb - black_dn, 0.0, None).astype(np.float32)
            yl = np.mean(ref_linear, axis=2)
            lo = np.percentile(yl, 5.0)
            hi = np.percentile(yl, 99.5)
            fit_mask = (yl >= lo) & (yl <= hi)
            if ccm_method == "lstsq_exr_reference":
                ccm = fit_ccm_lstsq(src, ref_linear, fit_mask)
            else:
                ccm = fit_diag_ccm(src, ref_linear, fit_mask)
            ccm = sanitize_ccm(ccm)
            dn_clean_rgb = np.clip(apply_ccm(src, ccm) + black_dn, 0.0, None)
            dn_noisy_rgb = np.clip(
                apply_ccm(np.clip(dn_noisy_rgb - black_dn, 0.0, None), ccm) + black_dn,
                0.0,
                None,
            )
            ccm_source = ccm_method
        pw_d_clean = None if preview_no_normalize else float(np.percentile(dn_clean_rgb, args.preview_percentile))
        pw_d_noisy = None if preview_no_normalize else float(np.percentile(dn_noisy_rgb, args.preview_percentile))
        write_png(
            png_dir / "clean_demosaic_rgb8.png",
            to_png8_preview(
                dn_clean_rgb,
                bit_depth,
                black_dn=black_dn,
                white_dn=pw_d_clean,
                srgb=demosaic_srgb,
            ),
        )
        write_png(
            png_dir / "noisy_demosaic_rgb8.png",
            to_png8_preview(
                dn_noisy_rgb,
                bit_depth,
                black_dn=black_dn,
                white_dn=pw_d_noisy,
                srgb=demosaic_srgb,
            ),
        )

    stats = {
        "config": str(cfg_path),
        "input_exr": str(exr_in),
        "linear_exr_mode": exr_mode if electrons_npz is None else None,
        "raw_out": str(raw_out),
        "png_dir": str(png_dir),
        "seed": args.seed,
        "signal_source": signal_source,
        "electrons_npz": str(electrons_npz) if electrons_npz is not None else None,
        "qe_relative_rgb": qe_vec.tolist(),
        "auto_exposure_enabled": auto_exposure,
        "exposure_scale_e_per_unit": exposure_scale,
        "bit_depth": bit_depth,
        "full_well_e": full_well_e,
        "K_e_per_DN": K_e_per_DN,
        "black_level_DN": black_dn,
        "preview_white_balance_enabled": wb_enabled,
        "preview_white_balance_method": wb_method if wb_enabled else None,
        "preview_white_balance_gains_rgb": wb_gains.tolist(),
        "preview_color_correction_enabled": ccm_enabled,
        "preview_color_correction_method": ccm_method if ccm_enabled else None,
        "preview_color_correction_source": ccm_source,
        "preview_color_correction_matrix_3x3": ccm.tolist(),
        "dark_current_e_per_s": dark_current_e_per_s,
        "dark_current_reference_temp_c": dark_ref_temp_c,
        "temperature_c": dark_temp_c,
        "dark_current_doubling_per_c": dark_doubling_c,
        "dark_mean_e_per_pixel": dark_mean_e,
        "row_fpn_std_e": row_fpn_std_e,
        "column_fpn_std_e": col_fpn_std_e,
        "adc_inl_quadratic_fraction": adc_inl_quad_fraction,
        "adc_dnl_std_lsb": adc_dnl_std_lsb,
        "adc_clipping": adc_clipping,
        "preview_no_normalize": preview_no_normalize,
        "preview_white_dn_percentile": args.preview_percentile,
        "preview_white_dn_value_clean": preview_white_clean,
        "preview_white_dn_value_noisy": preview_white_noisy,
        "bayer_enabled": bayer_on,
        "bayer_pattern": bayer_pat if bayer_on else None,
        "demosaic": ("bilinear" if demosaic_on else None),
        "demosaic_srgb_preview": bool(demosaic_srgb) if demosaic_on else None,
    }
    if bayer_on:
        stats["signal_e_mean_mono"] = float(np.mean(signal_e))
        stats["total_e_mean_mono"] = float(np.mean(total_e))
        stats["dn_noisy_min_mono"] = float(np.min(dn_noisy))
        stats["dn_noisy_max_mono"] = float(np.max(dn_noisy))
    else:
        stats["signal_e_mean_rgb"] = np.mean(signal_e, axis=(0, 1)).tolist()
        stats["total_e_mean_rgb"] = np.mean(total_e, axis=(0, 1)).tolist()
        stats["dn_noisy_min_rgb"] = np.min(dn_noisy, axis=(0, 1)).tolist()
        stats["dn_noisy_max_rgb"] = np.max(dn_noisy, axis=(0, 1)).tolist()
    (png_dir / "run_stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    print(f"Wrote RAW16: {raw_out}")
    print(f"Wrote PNG stack: {png_dir}")
    print(f"Wrote stats: {png_dir / 'run_stats.json'}")


if __name__ == "__main__":
    main()
