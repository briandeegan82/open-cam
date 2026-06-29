#!/usr/bin/env python3
"""Apply a YAML-driven EMVA-style sensor/noise model to a linear EXR render."""

from __future__ import annotations

import argparse
import hashlib
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
from sensor_radiometry import photon_flux_density_from_irradiance


def read_csv_curve(path: Path, *, strict_wavelength_axis: bool = False) -> tuple[np.ndarray, np.ndarray]:
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
        if strict_wavelength_axis:
            raise ValueError(
                "strict QE validation: normalized wavelength axis detected "
                f"in {path}; provide explicit wavelength-in-nm CSV"
            )
        w = 380.0 + (w - wmin) * (450.0 / (wmax - wmin))
        print(f"warning: mapped normalized wavelength axis to 380..830 nm for {path}", file=sys.stderr)

    idx = np.argsort(w)
    return w[idx], v[idx]


def load_qe_curves_rgb(
    repo: Path,
    qe_cfg: dict,
    *,
    strict_qe_validation: bool = False,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Load QE curves and auto-correct likely imported red/blue inversion."""
    r = read_csv_curve((repo / qe_cfg["red_csv"]).resolve(), strict_wavelength_axis=strict_qe_validation)
    g = read_csv_curve((repo / qe_cfg["green_csv"]).resolve(), strict_wavelength_axis=strict_qe_validation)
    b = read_csv_curve((repo / qe_cfg["blue_csv"]).resolve(), strict_wavelength_axis=strict_qe_validation)
    r_peak = float(r[0][int(np.argmax(r[1]))])
    g_peak = float(g[0][int(np.argmax(g[1]))])
    b_peak = float(b[0][int(np.argmax(b[1]))])
    if r_peak < g_peak and b_peak > g_peak:
        if strict_qe_validation:
            raise ValueError(
                "strict QE validation: detected likely QE red/blue inversion; "
                "fix channel assignments in QE CSVs"
            )
        print(
            "warning: detected likely QE red/blue inversion; swapping channels "
            f"(red_peak={r_peak:.1f}nm, green_peak={g_peak:.1f}nm, blue_peak={b_peak:.1f}nm)",
            file=sys.stderr,
        )
        r, b = b, r
    # Warn if any QE curve's measured range ends below 780 nm with non-negligible signal.
    # Above the curve's max wavelength, np.interp returns right=0.0, silently dropping
    # any spectral energy in that band.  If the last non-zero sample is before 780 nm and
    # the IRCF is not responsible for cutting it (IRCF check is left to the caller), the
    # channel response is likely truncated.
    _COVERAGE_WARN_NM = 780.0
    for _ch, (_wl, _v) in (("red", r), ("green", g), ("blue", b)):
        _last_nonzero = float(_wl[_v > 1e-4][-1]) if np.any(_v > 1e-4) else 0.0
        if _last_nonzero > 0 and _last_nonzero < _COVERAGE_WARN_NM:
            print(
                f"warning: {_ch} QE curve last non-zero value at {_last_nonzero:.0f} nm "
                f"(< {_COVERAGE_WARN_NM:.0f} nm); spectral energy above this wavelength "
                "is set to zero by extrapolation — verify IRCF covers the gap or extend the QE CSV.",
                file=sys.stderr,
            )
    return r, g, b


def mean_effective_qe(curve_csv: Path, ircf_csv: Path | None) -> float:
    wl, q = read_csv_curve(curve_csv)
    if ircf_csv is not None:
        ir_wl, ir = read_csv_curve(ircf_csv)
        ir_i = np.clip(np.interp(wl, ir_wl, ir, left=0.0, right=0.0), 0.0, 1.0)
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
    """Apply a 3x3 color correction matrix to HxWx3 image.

    Convention: ``ccm[in_ch, out_ch]`` — equivalently ``y = x @ ccm`` for row-vector
    pixels.  For a neutral gray input ``x = [g, g, g]`` the output is
    ``y[j] = g * sum_i(ccm[i, j])`` (column j sum), so **column sums** control neutral
    gray preservation, not row sums.

    See also :func:`apply_channel_crosstalk`, which uses the transpose convention
    ``m[out_ch, in_ch]`` and applies ``y = x @ m.T`` — pass ``m.T`` to this function
    to achieve the same result.
    """
    x = np.asarray(rgb, dtype=np.float32)
    y = np.tensordot(x, np.asarray(ccm, dtype=np.float32), axes=([2], [0]))
    return np.clip(y, 0.0, None).astype(np.float32)


def sanitize_ccm(ccm: np.ndarray) -> np.ndarray:
    """Reject pathological CCMs; normalize so neutral gray is preserved.

    Because ``apply_ccm`` uses the convention ``y = x @ M`` (column-index = output
    channel), a neutral input ``[g, g, g]`` maps to ``y[j] = g * col_sum(j)``.
    Neutral preservation therefore requires **equal column sums**.  This function
    rescales each column to sum to 1, then guards against a degenerate overall gain.
    """
    m = np.asarray(ccm, dtype=np.float32)
    if not np.all(np.isfinite(m)):
        return np.eye(3, dtype=np.float32)
    # Normalize columns so neutral gray [g,g,g] maps to [g,g,g].
    col_sums = np.sum(m, axis=0, keepdims=True)  # shape (1, 3)
    if np.any(np.abs(col_sums) < 1e-6):
        return np.eye(3, dtype=np.float32)
    m = m / col_sums
    # Keep average diagonal gain in a sane range for preview mapping.
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


def apply_cfa_spatial_crosstalk(cfa_e: np.ndarray, cfg: dict, pattern: str = "RGGB") -> np.ndarray:
    """Apply spatial diffusion crosstalk to a CFA RAW electron image (HxW).

    Models photon diffusion in the epitaxial silicon: photons absorbed near a
    pixel boundary can drift to an adjacent photosite.  Because this acts on the
    CFA mosaic **before** noise is drawn, it correctly creates inter-colour leakage
    (R signal contaminating a G well, etc.) that is physically distinct from the
    global uniform ``sensor.crosstalk`` 3×3 matrix.

    Implementation: convolve the CFA image with a 2-D Gaussian of width
    ``sigma_pixels``.  The kernel is normalised (sum = 1) so total electron count
    is conserved; the blur merely redistributes signal between neighbouring wells.
    Boundary pixels are handled with reflect-padding to avoid edge artefacts.

    Longer wavelengths (red) penetrate deeper into silicon and undergo more lateral
    diffusion before collection, so per-channel sigmas can be set with
    ``sigma_pixels_r``, ``sigma_pixels_g``, ``sigma_pixels_b``.  When all three
    equal ``sigma_pixels`` a single fast full-mosaic Gaussian is used.

    Typical σ values
    ----------------
    σ ≈ 0.20 px  →  ~0.3 % leakage to each nearest-neighbour  (very low-crosstalk BSI)
    σ ≈ 0.30 px  →  ~1.5 % leakage to each nearest-neighbour  (typical BSI CMOS, e.g. Sony)
    σ ≈ 0.50 px  →  ~8 % leakage per nearest-neighbour        (older FSI / large pitch)

    Parameters
    ----------
    cfa_e:
        HxW float32 array of pre-noise electron counts from ``bayer_sample_rgb``.
    cfg:
        ``cfa.spatial_crosstalk`` sub-dict from the camera model YAML.
    pattern:
        Bayer pattern string (e.g. ``"RGGB"``), used for per-channel sigma routing.
    """
    if not bool(cfg.get("enabled", False)):
        return cfa_e
    sigma = float(cfg.get("sigma_pixels", 0.3))
    if sigma <= 0.0:
        return cfa_e
    from scipy.ndimage import gaussian_filter  # noqa: PLC0415

    sigma_r = float(cfg.get("sigma_pixels_r", sigma))
    sigma_g = float(cfg.get("sigma_pixels_g", sigma))
    sigma_b = float(cfg.get("sigma_pixels_b", sigma))

    if sigma_r == sigma_g == sigma_b:
        # All channels identical: single fast full-mosaic blur.
        out = gaussian_filter(cfa_e.astype(np.float64), sigma=sigma, mode="reflect")
        return np.clip(out, 0.0, None).astype(np.float32)

    # Per-channel wavelength-dependent diffusion.
    # Each Bayer phase is blurred independently with its channel-specific sigma,
    # then interleaved back.  sigma is expressed in full-resolution pixels; on the
    # subsampled (2×-decimated) grid the equivalent sigma is sigma_ch / 2.
    pat_upper = pattern.upper()
    if pat_upper not in _BAYER_PHASE:
        # Unknown pattern: fall back to uniform blur.
        out = gaussian_filter(cfa_e.astype(np.float64), sigma=sigma, mode="reflect")
        return np.clip(out, 0.0, None).astype(np.float32)

    phase_lut = _BAYER_PHASE[pat_upper]  # (ch00, ch01, ch10, ch11)  R=0, G=1, B=2
    sigma_ch = (sigma_r, sigma_g, sigma_b)
    out = cfa_e.astype(np.float64).copy()
    for _pi in range(4):
        _row_off, _col_off = _pi >> 1, _pi & 1
        _sig = sigma_ch[phase_lut[_pi]]
        if _sig <= 0.0:
            continue
        sub = cfa_e[_row_off::2, _col_off::2].astype(np.float64)
        # Divide by 2 because the subsampled grid has pixels spaced 2 full-res pixels apart.
        sub_blurred = gaussian_filter(sub, sigma=_sig / 2.0, mode="reflect")
        out[_row_off::2, _col_off::2] = sub_blurred
    return np.clip(out, 0.0, None).astype(np.float32)


def apply_channel_crosstalk(rgb: np.ndarray, m33: np.ndarray) -> np.ndarray:
    """Apply a 3x3 channel mixing matrix to HxWx3 signal.

    Convention: ``m33[out_ch, in_ch]`` — row *j* of the matrix holds the input weights
    that sum into output channel *j*.  Internally this computes ``y = x @ m33.T``.

    **Important:** this is the **transpose** of the ``apply_ccm`` convention
    (``ccm[in_ch, out_ch]``).  If you have a matrix written in the CCM convention,
    pass its transpose here, or use :func:`apply_ccm` directly.

    When ``normalize_rows: true`` (default in config), each row of ``m33`` sums to 1,
    meaning no net gain or loss of photons per output channel — the correct semantic
    for energy-preserving optical/electrical crosstalk.
    """
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"apply_channel_crosstalk expects HxWx3, got {rgb.shape}")
    m = np.asarray(m33, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"crosstalk matrix must be 3x3, got {m.shape}")
    if not np.all(np.isfinite(m)):
        raise ValueError("crosstalk matrix contains non-finite values")
    x = np.asarray(rgb[:, :, :3], dtype=np.float64).reshape(-1, 3)
    y = x @ m.T
    return np.clip(y.reshape(rgb.shape[0], rgb.shape[1], 3), 0.0, None).astype(np.float32)


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


def _spatial_shape(arr: np.ndarray) -> tuple[int, int]:
    if arr.ndim == 2:
        return int(arr.shape[0]), int(arr.shape[1])
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return int(arr.shape[0]), int(arr.shape[1])
    raise ValueError(f"unsupported signal shape for defect model: {arr.shape}")


def apply_hot_stuck_pixel_model(
    signal_e: np.ndarray,
    rng: np.random.Generator,
    cfg: dict,
    full_well_e: float,
    persistent_map_npz: Path | None = None,
    regenerate_persistent_map: bool = False,
) -> tuple[np.ndarray, dict]:
    """Apply optional hot/stuck defect pixels in electron domain."""
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return signal_e.astype(np.float32, copy=False), {
            "enabled": False,
            "hot_pixel_count": 0,
            "stuck_high_count": 0,
            "stuck_low_count": 0,
        }

    out = np.asarray(signal_e, dtype=np.float32).copy()
    h, w = _spatial_shape(out)
    hot_rate = float(cfg.get("hot_pixel_rate", 0.0))
    stuck_high_rate = float(cfg.get("stuck_high_rate", 0.0))
    stuck_low_rate = float(cfg.get("stuck_low_rate", 0.0))
    hot_min_e = float(cfg.get("hot_dark_e_min", 50.0))
    hot_max_e = float(cfg.get("hot_dark_e_max", 5000.0))
    stuck_high_value_e = float(cfg.get("stuck_high_value_e", full_well_e))
    stuck_low_value_e = float(cfg.get("stuck_low_value_e", 0.0))

    hot_rate = float(np.clip(hot_rate, 0.0, 1.0))
    stuck_high_rate = float(np.clip(stuck_high_rate, 0.0, 1.0))
    stuck_low_rate = float(np.clip(stuck_low_rate, 0.0, 1.0))
    hot_min_e = max(0.0, hot_min_e)
    hot_max_e = max(hot_min_e, hot_max_e)
    stuck_high_value_e = float(np.clip(stuck_high_value_e, 0.0, full_well_e))
    stuck_low_value_e = max(0.0, stuck_low_value_e)

    map_source = "random_per_run"
    hot_add = np.zeros((h, w), dtype=np.float32)
    if persistent_map_npz is not None and persistent_map_npz.is_file() and not regenerate_persistent_map:
        data = np.load(persistent_map_npz)
        hot_mask = np.asarray(data["hot_mask"], dtype=bool)
        stuck_high_mask = np.asarray(data["stuck_high_mask"], dtype=bool)
        stuck_low_mask = np.asarray(data["stuck_low_mask"], dtype=bool)
        hot_add = np.asarray(data["hot_add_e"], dtype=np.float32)
        if hot_mask.shape != (h, w) or stuck_high_mask.shape != (h, w) or stuck_low_mask.shape != (h, w) or hot_add.shape != (h, w):
            raise ValueError(
                f"persistent defect map shape mismatch: expected {(h, w)}, "
                f"got hot={hot_mask.shape}, high={stuck_high_mask.shape}, low={stuck_low_mask.shape}, add={hot_add.shape}"
            )
        map_source = "loaded_persistent_map"
    else:
        base = rng.random((h, w))
        hot_mask = base < hot_rate
        stuck_high_mask = (~hot_mask) & (rng.random((h, w)) < stuck_high_rate)
        stuck_low_mask = (~hot_mask) & (~stuck_high_mask) & (rng.random((h, w)) < stuck_low_rate)
        if np.any(hot_mask):
            # Hot-pixel dark current follows an exponential distribution: most hot
            # pixels are only slightly elevated, with a long tail toward very high
            # dark current.  E[hot_add] = scale; Std[hot_add] = scale.
            # We use scale = (hot_max_e - hot_min_e) / 3 so ~95% of hot pixels fall
            # within [hot_min_e, hot_max_e] and clip the remainder at the limits.
            _hot_scale = max(1.0, (hot_max_e - hot_min_e) / 3.0)
            hot_add = rng.exponential(scale=_hot_scale, size=(h, w)).astype(np.float32)
            hot_add = np.clip(hot_add + hot_min_e, hot_min_e, hot_max_e)
        if persistent_map_npz is not None:
            persistent_map_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                persistent_map_npz,
                hot_mask=hot_mask.astype(np.uint8),
                stuck_high_mask=stuck_high_mask.astype(np.uint8),
                stuck_low_mask=stuck_low_mask.astype(np.uint8),
                hot_add_e=hot_add.astype(np.float32),
            )
            map_source = "saved_persistent_map"

    if np.any(hot_mask):
        if out.ndim == 2:
            out[hot_mask] += hot_add[hot_mask]
        else:
            out[hot_mask, :] += hot_add[hot_mask, None]

    if np.any(stuck_high_mask):
        if out.ndim == 2:
            out[stuck_high_mask] = stuck_high_value_e
        else:
            out[stuck_high_mask, :] = stuck_high_value_e
    if np.any(stuck_low_mask):
        if out.ndim == 2:
            out[stuck_low_mask] = stuck_low_value_e
        else:
            out[stuck_low_mask, :] = stuck_low_value_e

    out = np.clip(out, 0.0, None).astype(np.float32)
    return out, {
        "enabled": True,
        "hot_pixel_count": int(np.count_nonzero(hot_mask)),
        "stuck_high_count": int(np.count_nonzero(stuck_high_mask)),
        "stuck_low_count": int(np.count_nonzero(stuck_low_mask)),
        "hot_pixel_rate": hot_rate,
        "stuck_high_rate": stuck_high_rate,
        "stuck_low_rate": stuck_low_rate,
        "hot_dark_e_min": hot_min_e,
        "hot_dark_e_max": hot_max_e,
        "stuck_high_value_e": stuck_high_value_e,
        "stuck_low_value_e": stuck_low_value_e,
        "persistent_map_path": str(persistent_map_npz) if persistent_map_npz is not None else None,
        "persistent_map_source": map_source,
    }


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
        ir_i = np.clip(np.interp(wl, ir_wl, ir, left=0.0, right=0.0), 0.0, 1.0)
        q = q * ir_i
    q = np.clip(q, 0.0, 1.0)
    return np.interp(lambdas_nm, wl, q, left=0.0, right=0.0).astype(np.float32)


def integrate_exr_spectral_qe(
    exr_path: Path,
    repo: Path,
    qe_cfg: dict,
    sensor_cfg: dict,
    cal_cfg: dict,
    *,
    strict_qe_validation: bool = False,
    lens_cfg: dict | None = None,
) -> np.ndarray:
    """HxWx3 electrons via full photon-counting physics on a PBRT spectral EXR.

    Applies the same radiometric chain as ``pbrt_spectral_exr_to_electrons.py``:

        E_λ  = L_λ · (π / 4N²) · τ_optics · irr_scale · lux_scale   [W / m² / nm]
        φ(λ) = E_λ · λ / (h c)                                        [ph / m² / s / nm]
        e_c  = ∫ φ(λ) · QE_c(λ) dλ · (A_pixel · t_int · fill_factor) [electrons]

    Parameters
    ----------
    sensor_cfg:
        The ``sensor`` sub-dict from the camera model (provides ``f_number``,
        ``pixel_pitch_um``, ``integration_time_s``, ``fill_factor``).
    cal_cfg:
        The ``sensor_forward.model.calibration`` sub-dict (provides
        ``irradiance_scale_W_m2nm_per_unit``, ``optics_transmittance``,
        ``target_illuminance_lux``, ``illuminant_override_csv``,
        ``radiometric_autocalibration``).  Pass ``{}`` to use physical defaults.
    """
    # Deferred import to avoid a circular-import risk at module load time.
    from spectral_sensor_forward import illuminance_lux_from_irradiance  # noqa: PLC0415

    spec, lam = spectral_buckets_from_exr(exr_path.resolve())
    lam = lam.astype(np.float64)
    w = trapezoid_weights_nm(lam).astype(np.float64)

    # ---- Radiance → sensor-plane irradiance ----
    # For realistic cameras use the effective f-number derived from the lens prescription
    # (focal_length_mm / aperture_diameter_mm).  Falls back to sensor.f_number with a
    # warning when focal_length_mm is absent from the lens model.
    f_number = float(sensor_cfg.get("f_number", 2.8))
    if lens_cfg is not None and str(lens_cfg.get("camera", "pinhole")).lower() == "realistic":
        _fl = lens_cfg.get("focal_length_mm", None)
        _ap = lens_cfg.get("realistic_aperture_diameter_mm", None)
        if _fl is not None and _ap is not None and float(_ap) > 0:
            f_number = float(_fl) / float(_ap)
        else:
            print(
                "warning [integrate_qe]: realistic camera lens model missing focal_length_mm — "
                f"falling back to sensor.f_number={f_number}.",
                file=sys.stderr,
            )
    rad_to_irr = np.pi / (4.0 * max(1e-12, f_number**2))

    # ---- Scalar optics transmittance ----
    optics_t = float(cal_cfg.get("optics_transmittance", 1.0))

    # ---- Absolute irradiance scale (scene-unit → W/m²/nm) ----
    irr_scale = float(cal_cfg.get("irradiance_scale_W_m2nm_per_unit", 1.0e-3))

    # ---- Optional photometric calibration (match target_illuminance_lux) ----
    target_lux = cal_cfg.get("target_illuminance_lux", None)
    illum_csv = cal_cfg.get("illuminant_override_csv", None)
    auto_cal_mode = str(cal_cfg.get("radiometric_autocalibration", "off")).lower()
    lux_scale = 1.0

    if target_lux is not None and illum_csv:
        # Use illuminant CSV to normalise scene irradiance to target_lux.
        e_wl, e_v = read_csv_curve((repo / str(illum_csv)).resolve())
        ill_in = illuminance_lux_from_irradiance(e_wl, e_v * irr_scale)
        if ill_in > 0:
            lux_scale = float(target_lux) / ill_in
        else:
            print(
                "warning [integrate_qe]: photopic illuminance of illuminant CSV is <= 0; "
                "lux_scale left at 1",
                file=sys.stderr,
            )
    elif target_lux is not None and auto_cal_mode not in ("off", "none", "disabled", "false", "0"):
        # Autocalibrate: measure mean photopic lux of the rendered EXR and rescale.
        E_mean = (
            np.mean(spec.astype(np.float64), axis=(0, 1))
            * (rad_to_irr * optics_t * irr_scale)
        )
        scene_lux = illuminance_lux_from_irradiance(lam, E_mean)
        if scene_lux > 0:
            lux_scale = float(target_lux) / scene_lux
        else:
            print(
                "warning [integrate_qe]: EXR-derived scene illuminance <= 0; "
                "skipping autocalibration",
                file=sys.stderr,
            )

    global_scale = rad_to_irr * optics_t * irr_scale * lux_scale
    E = spec.astype(np.float64) * global_scale  # [W / m² / nm] per pixel

    # ---- Energy irradiance → spectral photon flux density ----
    phi = photon_flux_density_from_irradiance(E, lam)  # [ph / m² / s / nm]

    # ---- Geometry factor: pixel area × integration time × fill factor ----
    pixel_pitch_um = float(sensor_cfg.get("pixel_pitch_um", 1.4))
    t_int = float(sensor_cfg.get("integration_time_s", 0.01))
    fill_factor = float(sensor_cfg.get("fill_factor", 1.0))
    pixel_area = (pixel_pitch_um * 1e-6) ** 2
    geom = t_int * fill_factor * pixel_area

    # ---- QE integration ----
    ircf = qe_cfg.get("ircf_csv")
    ircf_path = (repo / ircf).resolve() if ircf else None
    q_r_src, q_g_src, q_b_src = load_qe_curves_rgb(
        repo,
        qe_cfg,
        strict_qe_validation=strict_qe_validation,
    )
    q_r = np.interp(lam, q_r_src[0], np.clip(q_r_src[1], 0.0, 1.0), left=0.0, right=0.0)
    q_g = np.interp(lam, q_g_src[0], np.clip(q_g_src[1], 0.0, 1.0), left=0.0, right=0.0)
    q_b = np.interp(lam, q_b_src[0], np.clip(q_b_src[1], 0.0, 1.0), left=0.0, right=0.0)
    if ircf_path is not None:
        ir_wl, ir = read_csv_curve(ircf_path)
        ir_i = np.interp(lam, ir_wl, np.clip(ir, 0.0, 1.0), left=0.0, right=0.0)
        q_r = np.clip(q_r * ir_i, 0.0, 1.0)
        q_g = np.clip(q_g * ir_i, 0.0, 1.0)
        q_b = np.clip(q_b * ir_i, 0.0, 1.0)

    acc_r = np.sum(phi * (q_r * w), axis=2) * geom
    acc_g = np.sum(phi * (q_g * w), axis=2) * geom
    acc_b = np.sum(phi * (q_b * w), axis=2) * geom
    return np.clip(np.stack([acc_r, acc_g, acc_b], axis=2), 0.0, None).astype(np.float32)


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
        "--preview-white-balance-enabled",
        type=str,
        choices=("true", "false"),
        default=None,
        help="Optional override for processing.preview_white_balance.enabled.",
    )
    ap.add_argument(
        "--preview-color-correction-enabled",
        type=str,
        choices=("true", "false"),
        default=None,
        help="Optional override for processing.preview_color_correction.enabled.",
    )
    ap.add_argument(
        "--integration-time-s",
        type=float,
        default=None,
        help="Optional integration time override in seconds (replaces sensor.integration_time_s).",
    )
    ap.add_argument(
        "--strict-qe-validation",
        action="store_true",
        help="Fail on QE wavelength-axis auto-remap or likely R/B swap detection.",
    )
    ap.add_argument(
        "--regenerate-defect-map",
        action="store_true",
        help="Regenerate and overwrite persistent defect-pixel map when configured.",
    )
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    cfg_path = None
    camera_model: dict = {}
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
    crosstalk_cfg = sensor.get("crosstalk", {}) or {}
    strict_qe_validation = bool(args.strict_qe_validation or qe_cfg.get("strict_validation", False))
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
    q_r_src, q_g_src, q_b_src = load_qe_curves_rgb(
        repo,
        qe_cfg,
        strict_qe_validation=strict_qe_validation,
    )
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

    if "full_well_e" not in adc:
        raise KeyError(
            "adc.full_well_e is required but not set in the sensor config. "
            "This value determines the ADC clipping point; there is no safe generic default."
        )
    full_well_e = float(adc["full_well_e"])
    K_e_per_DN = float(emva.get("overall_system_gain_K_e_per_DN", 0.08))
    sigma_d_e = float(emva.get("sigma_d_e", 2.0))
    sigma_amp_e = float(emva.get("sigma_amp_e", 0.0))
    dsnu_std_e = float(emva.get("dsnu_std_e", 0.3))
    prnu_std = float(emva.get("prnu_std_fraction", 0.005))
    prnu_std_r = float(emva.get("prnu_std_fraction_r", prnu_std))
    prnu_std_g = float(emva.get("prnu_std_fraction_g", prnu_std))
    prnu_std_b = float(emva.get("prnu_std_fraction_b", prnu_std))
    black_dn = float(emva.get("black_level_DN", 64.0))
    bit_depth = int(adc.get("bit_depth", 12))
    if bit_depth < 1:
        raise ValueError(f"adc.bit_depth must be >= 1, got {bit_depth!r}")
    full_well_e = _require_positive("adc.full_well_e", full_well_e)
    K_e_per_DN = _require_positive("emva.overall_system_gain_K_e_per_DN", K_e_per_DN)
    # ISO amplifier gain: scales the ADC path without touching electron-domain quantities.
    # K_eff = K_base / iso_gain  →  full_well_eff = full_well_base / iso_gain
    _iso_gain = float(emva.get("iso_gain_factor", 1.0))
    if _iso_gain <= 0:
        raise ValueError(f"emva.iso_gain_factor must be positive, got {_iso_gain!r}")
    _K_base = K_e_per_DN
    _full_well_base = full_well_e
    if _iso_gain != 1.0:
        K_e_per_DN = K_e_per_DN / _iso_gain
        full_well_e = full_well_e / _iso_gain
    # Amplifier noise: at high ISO, the analog gain stage adds its own noise in quadrature.
    # sigma_amp_e is the amplifier's input-referred noise floor at unit gain.  At iso_gain
    # G the amplifier contributes sigma_amp_e × G electrons, added in quadrature to the
    # base read noise.  Default 0 = no amplifier noise (backward-compatible).
    if sigma_amp_e > 0.0 and _iso_gain > 1.0:
        sigma_d_e = float(np.sqrt(sigma_d_e**2 + (sigma_amp_e * _iso_gain)**2))
    use_poisson = bool(emva.get("use_poisson_shot_noise", True))
    t_int_s = float(sensor.get("integration_time_s", 0.01))
    if args.integration_time_s is not None:
        t_int_s = float(args.integration_time_s)
    _require_positive("sensor.integration_time_s", t_int_s)
    dark_current_e_per_s = float(emva.get("dark_current_e_per_s", 0.0))
    dark_ref_temp_c = float(emva.get("dark_current_reference_temp_c", 20.0))
    dark_temp_c = float(emva.get("temperature_c", dark_ref_temp_c))
    dark_doubling_c = float(emva.get("dark_current_doubling_per_c", 6.0))
    dark_activation_energy_eV = float(emva.get("dark_activation_energy_eV", 0.0))
    row_fpn_std_e = float(emva.get("row_fpn_std_e", 0.0))
    col_fpn_std_e = float(emva.get("column_fpn_std_e", 0.0))
    row_fpn_fixed_std_e = float(emva.get("row_fpn_fixed_std_e", 0.0))
    col_fpn_fixed_std_e = float(emva.get("column_fpn_fixed_std_e", 0.0))
    prnu_lf_std = float(emva.get("prnu_lf_std_fraction", 0.0))
    prnu_lf_sigma_px = float(emva.get("prnu_lf_sigma_pixels", 100.0))
    ktc_cfg = emva.get("ktc_noise", {}) or {}
    ktc_enabled = bool(ktc_cfg.get("enabled", False))

    # Stable per-camera seed for fixed spatial patterns (PRNU, DSNU).
    # These represent physical properties of the sensor silicon that are the same
    # across every frame from the same camera unit.
    # If emva.spatial_noise_seed is set in YAML, use it directly.
    # Otherwise, derive a reproducible integer from the config file path so each
    # camera model gets its own unique but deterministic pattern.
    _spatial_seed_cfg = emva.get("spatial_noise_seed", None)
    if _spatial_seed_cfg is not None:
        _spatial_seed = int(_spatial_seed_cfg)
    else:
        _spatial_seed = (
            int(hashlib.sha256(str(cfg_path).encode()).hexdigest()[:16], 16) & 0x7FFFFFFF
        )
    spatial_rng = np.random.default_rng(_spatial_seed)

    adc_inl_quad_fraction = float(adc.get("inl_quadratic_fraction", 0.0))
    adc_dnl_std_lsb = float(adc.get("dnl_std_lsb", 0.0))
    _clipping_raw = adc.get("clipping", True)
    if isinstance(_clipping_raw, bool):
        adc_clipping = _clipping_raw
    elif isinstance(_clipping_raw, str):
        _cs = _clipping_raw.strip().lower()
        if _cs in ("hard", "true", "yes", "on", "1"):
            adc_clipping = True
        elif _cs in ("soft", "false", "no", "off", "0", "none", "disabled"):
            adc_clipping = False
        else:
            raise ValueError(
                f"adc.clipping must be a boolean or one of 'hard'/'soft', got {_clipping_raw!r}"
            )
    else:
        adc_clipping = bool(_clipping_raw)
    defect_cfg = emva.get("defect_pixels", {}) or {}
    persistent_map_npz = defect_cfg.get("persistent_map_npz", None)
    persistent_map_path = (repo / str(persistent_map_npz)).resolve() if persistent_map_npz else None

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
    if args.preview_white_balance_enabled is not None:
        wb_enabled = args.preview_white_balance_enabled == "true"
    wb_method = str(wb_cfg.get("method", "gray_world")).lower().strip()
    if wb_enabled and wb_method != "gray_world":
        raise ValueError('processing.preview_white_balance.method must be "gray_world"')
    ccm_cfg = proc_cfg.get("preview_color_correction", {}) or {}
    ccm_enabled = bool(ccm_cfg.get("enabled", True))
    if args.preview_color_correction_enabled is not None:
        ccm_enabled = args.preview_color_correction_enabled == "true"
    ccm_method = str(ccm_cfg.get("method", "diag_exr_reference")).lower().strip()
    if ccm_enabled and ccm_method not in ("diag_exr_reference", "lstsq_exr_reference"):
        raise ValueError(
            'processing.preview_color_correction.method must be "diag_exr_reference" or "lstsq_exr_reference"'
        )

    crosstalk_enabled = bool(crosstalk_cfg.get("enabled", False))
    crosstalk_matrix = np.eye(3, dtype=np.float32)
    if crosstalk_enabled:
        if "matrix_3x3" not in crosstalk_cfg:
            print(
                "warning: sensor.crosstalk.enabled is true but matrix_3x3 is absent; "
                "falling back to identity (no crosstalk applied). "
                "Set sensor.crosstalk.matrix_3x3 or disable crosstalk.",
                file=sys.stderr,
            )
        crosstalk_matrix = np.asarray(crosstalk_cfg.get("matrix_3x3", np.eye(3)), dtype=np.float32)
        if crosstalk_matrix.shape != (3, 3):
            raise ValueError("sensor.crosstalk.matrix_3x3 must be a 3x3 matrix")
        if bool(crosstalk_cfg.get("normalize_rows", True)):
            row_sum = np.sum(crosstalk_matrix, axis=1, keepdims=True)
            row_sum = np.where(np.abs(row_sum) < 1e-12, 1.0, row_sum)
            crosstalk_matrix = (crosstalk_matrix / row_sum).astype(np.float32)

    signal_source = "linear_exr"
    if electrons_npz is not None:
        signal_e = load_electrons_npz(electrons_npz)
        signal_source = "electrons_npz"
        if crosstalk_enabled:
            signal_e = apply_channel_crosstalk(signal_e, crosstalk_matrix)
        if auto_exposure:
            pctl = float(np.percentile(signal_e, args.target_fullwell_percentile))
            if pctl <= 0:
                raise RuntimeError("electrons_npz has non-positive signal for auto exposure")
            exposure_scale = float(args.target_fullwell_fraction) * full_well_e / pctl
        signal_e = signal_e * exposure_scale
    else:
        if exr_mode == "integrate_qe":
            signal_source = "linear_exr_integrate_qe"
            # Build calibration config for the full photon-counting physics.
            # When using --camera-model-config, read from sensor_forward.model.calibration;
            # otherwise fall back to sensible physics defaults.
            if camera_model:
                _sf_model = camera_model.get("sensor_forward", {}).get("model", {})
                _cal_cfg = dict(_sf_model.get("calibration", {}))
                _pbrt_block = _sf_model.get("pbrt_spectral_exr", {}) or {}
                _cal_cfg.setdefault(
                    "radiometric_autocalibration",
                    str(_pbrt_block.get("radiometric_autocalibration", "off")),
                )
            else:
                _cal_cfg = {}
            if exposure_scale != 1.0:
                print(
                    f"note [integrate_qe]: processing.exposure_scale_e_per_unit={exposure_scale} "
                    "is applied as a post-physics trim. After the radiometric fix the output "
                    "is already in electrons; set to 1.0 unless you need a manual correction.",
                    file=sys.stderr,
                )
            try:
                e_nominal = integrate_exr_spectral_qe(
                    exr_in,
                    repo,
                    qe_cfg,
                    sensor,
                    _cal_cfg,
                    strict_qe_validation=strict_qe_validation,
                    lens_cfg=camera_model.get("lens", {}) if camera_model else None,
                )
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

        if crosstalk_enabled:
            e_nominal = apply_channel_crosstalk(e_nominal, crosstalk_matrix)
        if auto_exposure:
            pctl = float(np.percentile(e_nominal, args.target_fullwell_percentile))
            if pctl <= 0:
                raise RuntimeError("input image has zero signal after QE weighting")
            exposure_scale = float(args.target_fullwell_fraction) * full_well_e / pctl
        signal_e = e_nominal * exposure_scale

    rng = np.random.default_rng(args.seed)
    signal_e, defect_stats = apply_hot_stuck_pixel_model(
        signal_e,
        rng,
        defect_cfg,
        full_well_e,
        persistent_map_npz=persistent_map_path,
        regenerate_persistent_map=bool(args.regenerate_defect_map),
    )

    bayer_on = bool(bayer_cfg.get("enabled", False))
    bayer_pat = str(bayer_cfg.get("pattern", "RGGB"))
    demosaic_on = bayer_on and demosaic_requested(bayer_cfg)
    demosaic_srgb = bool(bayer_cfg.get("demosaic_srgb", False))
    spatial_xtalk_cfg = bayer_cfg.get("spatial_crosstalk", {}) or {}

    if bayer_on:
        signal_e = bayer_sample_rgb(signal_e, bayer_pat)
        # Spatial inter-pixel diffusion crosstalk: must be applied after CFA sampling
        # so that the blur acts on the RAW mosaic (R leaking to G/B wells, etc.).
        # Applied before the noise chain so shot noise is drawn on the correct mean.
        signal_e = apply_cfa_spatial_crosstalk(signal_e, spatial_xtalk_cfg, bayer_pat)

    # Dark current mean: computed here (not in the temporal section) so that the DSNU
    # log-normal distribution can be correctly centred on it — ensuring that
    # dark_mean_e + dsnu_map[i] = D_i ≥ 0 for every pixel.
    if dark_activation_energy_eV > 0.0:
        _K_B_EV = 8.617333262e-5
        _T_K  = dark_temp_c + 273.15
        _T0_K = dark_ref_temp_c + 273.15
        dark_temp_scale = float(np.exp(
            (dark_activation_energy_eV / _K_B_EV) * (1.0 / _T0_K - 1.0 / _T_K)
        ))
    else:
        dark_temp_scale = 2.0 ** ((dark_temp_c - dark_ref_temp_c) / max(1e-6, dark_doubling_c))
    dark_mean_e = max(0.0, dark_current_e_per_s * t_int_s * dark_temp_scale)

    # ---- Fixed spatial patterns (same for every frame from the same camera) ----
    # PRNU: per-pixel multiplicative gain non-uniformity baked into the silicon.
    # DSNU: per-pixel additive dark-signal offset due to leakage current variation.
    # Both are sampled with spatial_rng, which is seeded from emva.spatial_noise_seed
    # or (by default) a hash of the config path.  This guarantees that different cameras
    # have different patterns while the same camera repeats the same pattern every run.
    #
    # In CFA mode each Bayer color has its own independent PRNU realisation drawn with
    # an optionally channel-specific sigma (emva.prnu_std_fraction_r/g/b).
    # The map is clipped to [0, ∞) so negative photosensitivity is always excluded.
    if signal_e.ndim == 2 and bayer_on:
        _prnu_phase_std = (prnu_std_r, prnu_std_g, prnu_std_b)  # R=0, G=1, B=2
        _cfa_phase_ch = _BAYER_PHASE[bayer_pat.upper()]
        prnu_map = np.empty(signal_e.shape, dtype=np.float32)
        for _pi in range(4):
            _row_off, _col_off = _pi >> 1, _pi & 1
            _ch_std = _prnu_phase_std[_cfa_phase_ch[_pi]]
            _sub = 1.0 + spatial_rng.normal(0.0, _ch_std, size=signal_e[_row_off::2, _col_off::2].shape)
            prnu_map[_row_off::2, _col_off::2] = _sub
        prnu_map = np.maximum(prnu_map, 0.0, out=prnu_map)
    else:
        prnu_map = np.maximum(
            (1.0 + spatial_rng.normal(0.0, prnu_std, size=signal_e.shape)).astype(np.float32),
            0.0,
        )

    # Low-frequency PRNU trend: models large-scale gain gradients (lens shading, process
    # variation across the die).  A smooth random field is added on top of the per-pixel
    # white-noise PRNU above.  Drawn from spatial_rng so it is fixed per camera unit.
    # Enabled only when prnu_lf_std_fraction > 0 in the emva config.
    if prnu_lf_std > 0.0 and prnu_lf_sigma_px > 0.0:
        from scipy.ndimage import gaussian_filter as _gf  # noqa: PLC0415
        _lf_raw = spatial_rng.normal(0.0, prnu_lf_std, size=signal_e.shape[:2]).astype(np.float32)
        _lf_smooth = _gf(_lf_raw, sigma=prnu_lf_sigma_px, mode="reflect").astype(np.float32)
        if signal_e.ndim == 2:
            prnu_map += _lf_smooth
        else:
            prnu_map += _lf_smooth[:, :, None]
        prnu_map = np.maximum(prnu_map, 0.0, out=prnu_map)

    # DSNU: per-pixel absolute dark-current variation follows a log-normal distribution.
    # Dark current arises from trap-mediated generation: most pixels are near the mean,
    # a long positive tail represents sites with elevated leakage.
    # The log-normal mean is dark_mean_e (computed above from dark_current_e_per_s × t_int),
    # ensuring dark_mean_e + dsnu_map[i] = D_i ≥ 0 for every pixel — consistent with the
    # global mean used in the Poisson draw.  dsnu_std_e is the absolute spatial std (electrons).
    # When dark_mean_e ≈ 0 (very short exposure / low dark current) dsnu_map is all-zeros.
    if dark_mean_e > 1e-9 and dsnu_std_e > 0.0:
        _v_ratio = dsnu_std_e / dark_mean_e
        _sigma_ln = float(np.sqrt(np.log1p(_v_ratio ** 2)))
        _mu_ln = np.log(dark_mean_e) - 0.5 * _sigma_ln ** 2
        if signal_e.ndim == 2:
            _dsnu_abs = spatial_rng.lognormal(_mu_ln, _sigma_ln, size=signal_e.shape).astype(np.float32)
            dsnu_map = (_dsnu_abs - dark_mean_e).astype(np.float32)
        else:
            _dsnu_abs = spatial_rng.lognormal(_mu_ln, _sigma_ln, size=signal_e.shape[:2]).astype(np.float32)
            dsnu_map = (_dsnu_abs - dark_mean_e)[:, :, None].astype(np.float32)
    else:
        if signal_e.ndim == 2:
            dsnu_map = np.zeros(signal_e.shape, dtype=np.float32)
        else:
            dsnu_map = np.zeros(signal_e.shape[:2] + (1,), dtype=np.float32)

    # ---- Temporal frame noise (varies legitimately per frame) ----
    # dark_mean_e was computed before the spatial-patterns section so the DSNU log-normal
    # could be correctly centred on it.  It is reused directly here.

    # Fixed row/column FPN: column offset and row gain variation baked into the readout
    # circuit — the same pattern every frame from the same camera unit.  Drawn from
    # spatial_rng (fixed seed) after DSNU so PRNU/DSNU patterns are unaffected.
    if signal_e.ndim == 2:
        row_fpn_fixed = spatial_rng.normal(0.0, row_fpn_fixed_std_e, size=(signal_e.shape[0], 1)).astype(np.float32)
        col_fpn_fixed = spatial_rng.normal(0.0, col_fpn_fixed_std_e, size=(1, signal_e.shape[1])).astype(np.float32)
    else:
        row_fpn_fixed = spatial_rng.normal(0.0, row_fpn_fixed_std_e, size=(signal_e.shape[0], 1, 1)).astype(np.float32)
        col_fpn_fixed = spatial_rng.normal(0.0, col_fpn_fixed_std_e, size=(1, signal_e.shape[1], 1)).astype(np.float32)

    # Row/column readout amplifier noise: temporal banding that changes every frame.
    if signal_e.ndim == 2:
        row_fpn = rng.normal(0.0, row_fpn_std_e, size=(signal_e.shape[0], 1)).astype(np.float32)
        col_fpn = rng.normal(0.0, col_fpn_std_e, size=(1, signal_e.shape[1])).astype(np.float32)
    else:
        row_fpn = rng.normal(0.0, row_fpn_std_e, size=(signal_e.shape[0], 1, 1)).astype(np.float32)
        col_fpn = rng.normal(0.0, col_fpn_std_e, size=(1, signal_e.shape[1], 1)).astype(np.float32)
    rc_fpn = row_fpn_fixed + col_fpn_fixed + row_fpn + col_fpn

    # ---- Poisson shot noise on the true mean ----
    # mean_e is the expected electron count per pixel:
    #   signal × PRNU (per-pixel gain)  +  dark_mean (temp-scaled)  +  DSNU (per-pixel dark offset)
    # PRNU and DSNU are spatially fixed; they shift the Poisson mean but do not add temporal variance.
    # dark_mean contributes both a mean shift AND Poisson (shot) variance — included here correctly.
    # rc_fpn and read_e are Gaussian readout noise processes added by the source-follower
    # and column amp DOWNSTREAM of the sense node.  The sense node saturates at full_well_e
    # before readout, so the clip must be applied to shot_e alone, not to (shot_e + readout).
    mean_e = signal_e * prnu_map + dark_mean_e + dsnu_map
    if use_poisson:
        shot_e = rng.poisson(np.maximum(mean_e, 0.0)).astype(np.float32)
    else:
        shot_e = np.maximum(mean_e, 0.0).astype(np.float32)

    # ---- kTC reset noise (sense-node sampling uncertainty) ----
    # Only relevant for sensors without correlated double sampling (CDS).
    # CDS (standard in rolling-shutter BSI CMOS) cancels kTC noise entirely;
    # disabled by default via noise.emva.ktc_noise.enabled: false.
    sigma_ktc_e = 0.0
    if ktc_enabled:
        _K_B_J = 1.380649e-23       # Boltzmann constant [J/K]
        _Q_E   = 1.602176634e-19    # elementary charge [C]
        _T_K   = dark_temp_c + 273.15
        if "node_capacitance_fF" in ktc_cfg:
            _C = float(ktc_cfg["node_capacitance_fF"]) * 1e-15  # fF → F
        else:
            # Estimate C from conversion gain assuming a supply reference voltage.
            # C = q · K[e/DN] · 2^bits / V_ref
            _vref = float(ktc_cfg.get("vref_V", 1.8))
            _C = K_e_per_DN * _Q_E * float(2 ** bit_depth) / _vref
        sigma_ktc_e = float(np.sqrt(_K_B_J * _T_K * _C) / _Q_E)

    # ---- Additive Gaussian readout noise (post-Poisson, includes kTC when enabled) ----
    read_e = rng.normal(0.0, sigma_d_e, size=signal_e.shape).astype(np.float32)
    if sigma_ktc_e > 0.0:
        read_e = read_e + rng.normal(0.0, sigma_ktc_e, size=signal_e.shape).astype(np.float32)
    if adc_clipping:
        total_e = np.clip(shot_e, 0.0, full_well_e) + rc_fpn + read_e
    else:
        total_e = np.clip(shot_e + rc_fpn + read_e, 0.0, None)

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
        "spatial_noise_seed": _spatial_seed,
        "signal_source": signal_source,
        "electrons_npz": str(electrons_npz) if electrons_npz is not None else None,
        "qe_relative_rgb": qe_vec.tolist(),
        "auto_exposure_enabled": auto_exposure,
        "exposure_scale_e_per_unit": exposure_scale,
        "bit_depth": bit_depth,
        "iso_gain_factor": _iso_gain,
        "K_base_e_per_DN": _K_base,
        "K_effective_e_per_DN": K_e_per_DN,
        "full_well_base_e": _full_well_base,
        "full_well_effective_e": full_well_e,
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
        "dark_current_model": "arrhenius" if dark_activation_energy_eV > 0.0 else "doubling_rule",
        "dark_activation_energy_eV": dark_activation_energy_eV if dark_activation_energy_eV > 0.0 else None,
        "dark_current_doubling_per_c": dark_doubling_c,
        "dark_temp_scale": dark_temp_scale,
        "dark_mean_e_per_pixel": dark_mean_e,
        "row_fpn_std_e": row_fpn_std_e,
        "column_fpn_std_e": col_fpn_std_e,
        "ktc_noise_enabled": ktc_enabled,
        "sigma_ktc_e": sigma_ktc_e if ktc_enabled else None,
        "adc_inl_quadratic_fraction": adc_inl_quad_fraction,
        "adc_dnl_std_lsb": adc_dnl_std_lsb,
        "adc_clipping": adc_clipping,
        "defect_pixels": defect_stats,
        "preview_no_normalize": preview_no_normalize,
        "crosstalk_enabled": crosstalk_enabled,
        "crosstalk_matrix_3x3": crosstalk_matrix.tolist(),
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
