"""Dead-leaves texture target and texture MTF / texture acutance (CPIQ texture blur, ISO 19567-2 idea).

In simulation the ideal (pre-camera) target is known exactly, so the texture MTF is the ratio of the
captured to the ideal power spectrum, with the noise power spectrum from a flat patch subtracted:
    MTF_tex(f) = sqrt( (PSD_capture(f) - PSD_noise(f)) / PSD_ideal(f) )
"""

from __future__ import annotations

import numpy as np

from iqlab.cpiq import ViewingCondition, acutance


def dead_leaves(
    size: int = 512,
    *,
    r_min: float = 1.0,
    r_max: float = 100.0,
    n_disks: int | None = None,
    low: float = 0.25,
    high: float = 0.75,
    seed: int = 0,
) -> np.ndarray:
    """Grey-level dead-leaves image (occluding disks, radius pdf ~ 1/r^3 on [r_min, r_max])."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size), np.nan)
    yy, xx = np.mgrid[0:size, 0:size]
    n = n_disks or int(40 * size * size / (np.pi * r_min * r_max))
    # inverse-CDF sampling of p(r) ~ r^-3 on [r_min, r_max]
    u = rng.random(n)
    radii = 1.0 / np.sqrt(1.0 / r_min**2 - u * (1.0 / r_min**2 - 1.0 / r_max**2))
    cx = rng.uniform(-r_max, size + r_max, n)
    cy = rng.uniform(-r_max, size + r_max, n)
    val = rng.uniform(low, high, n)
    remaining = size * size
    for r, x, y, v in zip(radii, cx, cy, val, strict=True):
        x0, x1 = int(max(0, np.floor(x - r))), int(min(size, np.ceil(x + r) + 1))
        y0, y1 = int(max(0, np.floor(y - r))), int(min(size, np.ceil(y + r) + 1))
        if x0 >= x1 or y0 >= y1:
            continue
        sub = img[y0:y1, x0:x1]
        m = ((xx[y0:y1, x0:x1] - x) ** 2 + (yy[y0:y1, x0:x1] - y) ** 2 <= r * r) & np.isnan(sub)
        sub[m] = v
        remaining -= int(np.count_nonzero(m))
        if remaining == 0:
            break
    img[np.isnan(img)] = 0.5 * (low + high)
    return img


def radial_psd(
    img: np.ndarray, n_bins: int | None = None, px_scale: tuple[float, float] = (1.0, 1.0)
) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged power spectrum of a 2-D image (Hann-windowed, mean removed); freq in cy/px.

    *px_scale* = image pixels per output pixel along (y, x): frequencies are reported in cycles per
    output pixel, so a high-resolution reference can be binned on a coarser capture's frequency axis.
    """
    a = np.asarray(img, dtype=np.float64)
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    spec = np.abs(np.fft.fftshift(np.fft.fft2((a - a.mean()) * win))) ** 2 / np.sum(win**2)
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None] * px_scale[0]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :] * px_scale[1]
    fr = np.hypot(fx, fy)
    n_bins = n_bins or min(h, w) // 2
    edges = np.linspace(0.0, 0.5, n_bins + 1)
    idx = np.digitize(fr.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < n_bins)
    s = np.bincount(idx[ok], spec.ravel()[ok], n_bins)
    c = np.bincount(idx[ok], minlength=n_bins)
    return 0.5 * (edges[1:] + edges[:-1]), s / np.maximum(c, 1)


def texture_mtf(
    captured: np.ndarray, ideal: np.ndarray, noise_patch: np.ndarray | None = None, n_bins: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Texture MTF from captured vs ideal dead-leaves (linear units).

    *ideal* must cover the same field as *captured* but may be sampled more finely (e.g. the chart
    texture itself): its PSD is then evaluated on the capture's frequency axis, so the reference is the
    unfiltered, unaliased texture and the result includes the pixel aperture, like the edge SFR.
    Only the PSD shape matters; callers normalise the curve at low frequency.
    """
    n_bins = n_bins or min(captured.shape) // 2
    f, p_cap = radial_psd(captured, n_bins)
    scale = (ideal.shape[0] / captured.shape[0], ideal.shape[1] / captured.shape[1])
    _, p_ref = radial_psd(ideal, n_bins, scale)
    if noise_patch is not None:
        _, p_n = radial_psd(noise_patch, n_bins)
        p_cap = p_cap - p_n
    mtf = np.sqrt(np.clip(p_cap, 0.0, None) / np.maximum(p_ref, 1e-30))
    return f, mtf


def illumination_plane(img: np.ndarray) -> np.ndarray:
    """Least-squares plane through *img*, normalised to mean 1.

    Uneven chart lighting is multiplicative; dividing it out keeps its gradient from leaking power
    into the lowest PSD bins, which set the texture-MTF normalisation."""
    a = np.asarray(img, dtype=np.float64)
    yy, xx = np.mgrid[0 : a.shape[0], 0 : a.shape[1]]
    A = np.stack([np.ones(a.size), yy.ravel(), xx.ravel()], 1)
    plane = (A @ np.linalg.lstsq(A, a.ravel(), rcond=None)[0]).reshape(a.shape)
    plane /= plane.mean()
    return np.where(plane > 0.05, plane, 1.0)


def texture_acutance(
    captured: np.ndarray, ideal: np.ndarray, view: ViewingCondition, noise_patch: np.ndarray | None = None
) -> float:
    f, m = texture_mtf(captured, ideal, noise_patch)
    return acutance(f, m, view)
