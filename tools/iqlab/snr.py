"""Patch SNR, photon-transfer style SNR curves and SNR-threshold dynamic range (ISO 15739 / EMVA 1288 spirit)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PatchStats:
    mean: float
    std: float
    n: int
    saturated_fraction: float

    @property
    def snr(self) -> float:
        return float(self.mean / self.std) if self.std > 0 else float("inf")

    @property
    def snr_db(self) -> float:
        s = self.snr
        return 20.0 * float(np.log10(s)) if s > 0 else float("-inf")


def _plane_detrend(patch: np.ndarray) -> np.ndarray:
    """Remove a least-squares plane so illumination fall-off is not counted as noise (ISO 15739 practice)."""
    h, w = patch.shape
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.column_stack([np.ones(h * w), xx.ravel(), yy.ravel()])
    coef, *_ = np.linalg.lstsq(a, patch.ravel().astype(np.float64), rcond=None)
    trend = (a @ coef).reshape(h, w)
    return patch - trend + coef[0] + coef[1] * (w - 1) / 2 + coef[2] * (h - 1) / 2


def patch_stats(
    patch: np.ndarray, *, black_level: float = 0.0, saturation: float | None = None, detrend: bool = True
) -> PatchStats:
    """Mean / temporal+spatial std of one uniform patch (2-D), black level subtracted."""
    p = np.asarray(patch, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("patch_stats expects a 2-D (single-channel) patch")
    sat = float(np.mean(p >= saturation)) if saturation is not None else 0.0
    p = p - black_level
    if detrend and min(p.shape) >= 3:
        p = _plane_detrend(p)
    return PatchStats(float(np.mean(p)), float(np.std(p, ddof=1)), int(p.size), sat)


def temporal_patch_stats(frames: np.ndarray, *, black_level: float = 0.0) -> PatchStats:
    """Temporal noise from a stack ``(n_frames, h, w)`` of the same patch (EMVA 1288 two-frame idea generalised)."""
    f = np.asarray(frames, dtype=np.float64) - black_level
    if f.ndim != 3 or f.shape[0] < 2:
        raise ValueError("temporal_patch_stats expects (n_frames>=2, h, w)")
    var_t = float(np.mean(np.var(f, axis=0, ddof=1)))
    return PatchStats(float(np.mean(f)), float(np.sqrt(var_t)), int(f[0].size), 0.0)


def snr_threshold_signal(signal: np.ndarray, snr: np.ndarray, threshold: float) -> float:
    """Lowest signal at which the SNR curve reaches ``threshold`` (log-log interpolation).

    ``signal`` and ``snr`` are per-patch values (any order). Returns NaN if never reached.
    """
    s = np.asarray(signal, dtype=np.float64)
    r = np.asarray(snr, dtype=np.float64)
    ok = (s > 0) & (r > 0) & np.isfinite(r)
    s, r = s[ok], r[ok]
    order = np.argsort(s)
    s, r = s[order], r[order]
    if s.size == 0 or r.max() < threshold:
        return float("nan")
    if r[0] >= threshold:
        return float(s[0])
    i = int(np.argmax(r >= threshold))
    ls0, ls1 = np.log(s[i - 1]), np.log(s[i])
    lr0, lr1 = np.log(r[i - 1]), np.log(r[i])
    t = (np.log(threshold) - lr0) / (lr1 - lr0)
    return float(np.exp(ls0 + t * (ls1 - ls0)))


@dataclass(frozen=True)
class DynamicRange:
    max_signal: float
    min_signal: float
    snr_threshold: float

    @property
    def ratio(self) -> float:
        return self.max_signal / self.min_signal if self.min_signal > 0 else float("nan")

    @property
    def db(self) -> float:
        return 20.0 * float(np.log10(self.ratio))

    @property
    def stops(self) -> float:
        return float(np.log2(self.ratio))


def dynamic_range(
    signal: np.ndarray,
    snr: np.ndarray,
    *,
    snr_threshold: float = 1.0,
    saturated_fraction: np.ndarray | None = None,
    max_saturated_fraction: float = 0.001,
) -> DynamicRange:
    """Dynamic range = brightest unsaturated patch signal / signal where SNR first reaches the threshold.

    ``snr_threshold=1`` gives the EMVA-style (noise-floor) DR, ``10`` the ISO 15739 "SNR=10" DR.
    Signals can be in electrons, DN or scene luminance; the ratio is what is reported.
    """
    s = np.asarray(signal, dtype=np.float64)
    keep = np.ones_like(s, dtype=bool)
    if saturated_fraction is not None:
        keep = np.asarray(saturated_fraction) <= max_saturated_fraction
    if not keep.any():
        raise ValueError("every patch is saturated")
    s_max = float(np.max(s[keep]))
    s_min = snr_threshold_signal(s[keep], np.asarray(snr)[keep], snr_threshold)
    return DynamicRange(s_max, s_min, snr_threshold)


def shot_read_snr(signal_e: np.ndarray, read_noise_e: float, dark_e: float = 0.0) -> np.ndarray:
    """Analytic SNR = S / sqrt(S + dark + read^2) for a linear photon-counting pixel (reference curve)."""
    s = np.asarray(signal_e, dtype=np.float64)
    return s / np.sqrt(s + dark_e + read_noise_e**2)
