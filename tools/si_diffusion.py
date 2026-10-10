"""Wavelength-dependent carrier-diffusion crosstalk from silicon absorption depth.

A photon of wavelength λ is absorbed at depth ``z`` below the illuminated surface with density
``α e^{-αz}`` (Green 2008 absorption, ``spectra/silicon/si_green2008_nk.csv``), truncated to the
epitaxial layer ``[0, d]``. Carriers generated inside the depletion region drift straight to their
own photodiode. Carriers generated in the field-free layer, at height ``h`` above the depletion
edge, diffuse; with no recombination (diffusion length ≫ epi thickness) and the field-free layer
bounded by the depletion edge, the point where a carrier reaches the depletion edge has the 2-D
Poisson-kernel density ``h / (2π (r² + h²)^{3/2})``; the far side of the field-free layer is
reflecting (passivated back surface for BSI, p/p+ epi-substrate junction for FSI), which the method
of images turns into an alternating series of Poisson kernels with exponentially decaying tails.

- FSI: junction at the illuminated side, depletion ``[0, w]``: ``h = max(0, z − w)``. Red goes
  deeper, so red spreads more.
- BSI: junction at the far side, depletion ``[d − w, d]``: ``h = max(0, d − w − z)``. Blue is
  absorbed near the back surface, so blue spreads more.

The kernel per wavelength integrates the Poisson kernel analytically over each neighbour pixel,
averages over generation positions inside the source pixel and over depth, and is truncated to
``radius_px`` (renormalised, so charge is conserved). The kernel of a CFA channel is the mixture over
wavelength weighted by that channel's spectral electron distribution. Not modelled: deep-trench
isolation, microlens / optical crosstalk, chief-ray angle, recombination, reflecting surfaces.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SI_CSV = REPO / "spectra" / "silicon" / "si_green2008_nk.csv"

DEFAULTS = {
    "architecture": "bsi",
    "epi_thickness_um": 3.0,
    "depletion_depth_um": 2.5,
    "radius_px": 4,
    "depth_samples": 64,
    "source_subsamples": 4,
}


@lru_cache(maxsize=1)
def _si_table() -> tuple[np.ndarray, np.ndarray]:
    rows = [ln.split(",") for ln in SI_CSV.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    d = np.array(rows[1:], float)
    lam = d[:, rows[0].index("wavelength_nm")]
    alpha_um = 4.0 * math.pi * d[:, rows[0].index("k")] / (lam * 1e-3)
    return lam, alpha_um


def absorption_coefficient_per_um(lam_nm) -> np.ndarray:
    """Silicon absorption coefficient α (1/µm), log-interpolated in the Green 2008 table."""
    lam, a = _si_table()
    return np.exp(np.interp(np.asarray(lam_nm, float), lam, np.log(a)))


def poisson_rect(x0, x1, y0, y1, h):
    """∫∫ of the Poisson kernel ``h / (2π (x²+y²+h²)^{3/2})`` over ``[x0,x1]×[y0,y1]`` (h > 0)."""

    def f(x, y):
        return np.arctan2(x * y, h * np.sqrt(x * x + y * y + h * h)) / (2.0 * math.pi)

    return f(x1, y1) - f(x0, y1) - f(x1, y0) + f(x0, y0)


def height_distribution(lam_nm: float, cfg: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """Diffusion heights above the depletion edge (µm), their probabilities, and the field-free thickness."""
    d = float(cfg["epi_thickness_um"])
    w = float(cfg["depletion_depth_um"])
    if not 0.0 <= w < d:
        raise ValueError("need 0 <= depletion_depth_um < epi_thickness_um")
    a = float(absorption_coefficient_per_um(lam_nm))
    n = int(cfg["depth_samples"])
    # Equal-probability depth bins of the truncated exponential, evaluated at their medians.
    norm = -math.expm1(-a * d)
    u = (np.arange(n) + 0.5) / n
    z = -np.log1p(-u * norm) / a
    arch = str(cfg["architecture"]).lower()
    if arch == "fsi":
        h = np.maximum(0.0, z - w)
    elif arch == "bsi":
        h = np.maximum(0.0, d - w - z)
    else:
        raise ValueError(f"architecture must be 'fsi' or 'bsi', got {arch!r}")
    return h, np.full(n, 1.0 / n), d - w


def pixel_kernel(h_px: float, slab_px: float, radius: int, subsamples: int, images: int = 24) -> np.ndarray:
    """Fraction of carriers from a uniformly lit source pixel collected by each neighbour.

    Field-free slab of thickness ``slab_px``: absorbing depletion edge below, reflecting surface
    (passivated back surface for BSI, p/p+ epi-substrate junction for FSI) above; method of images.
    """
    n = 2 * radius + 1
    k = np.zeros((n, n))
    if h_px <= 1e-9:
        k[radius, radius] = 1.0
        return k
    offs = (np.arange(subsamples) + 0.5) / subsamples - 0.5
    i = np.arange(-radius, radius + 1, dtype=float)
    sx = offs[:, None, None, None]
    sy = offs[None, :, None, None]
    x0, x1 = i[None, None, None, :] - 0.5 - sx, i[None, None, None, :] + 0.5 - sx
    y0, y1 = i[None, None, :, None] - 0.5 - sy, i[None, None, :, None] + 0.5 - sy
    period = 4.0 * slab_px
    acc = np.zeros(np.broadcast_shapes(x0.shape, y0.shape))
    for s in (h_px, 2.0 * slab_px - h_px):
        for m in range(-images, images + 1):
            for a, sign in ((s + m * period, 1.0), (-s + m * period, -1.0)):
                if abs(a) > 1e-12:
                    acc += sign * math.copysign(0.5, a) * poisson_rect(x0, x1, y0, y1, abs(a))
    return acc.mean(axis=(0, 1))


def wavelength_kernel(lam_nm: float, pitch_um: float, cfg: dict | None = None) -> np.ndarray:
    c = {**DEFAULTS, **(cfg or {})}
    h, p, slab = height_distribution(lam_nm, c)
    r, s = int(c["radius_px"]), int(c["source_subsamples"])
    slab_px = round(slab / pitch_um, 4)
    k = sum(pi * _cached_kernel(round(hi / pitch_um, 3), slab_px, r, s) for hi, pi in zip(h, p, strict=True))
    return k / k.sum()


@lru_cache(maxsize=8192)
def _cached_kernel(h_px: float, slab_px: float, radius: int, subsamples: int) -> np.ndarray:
    return pixel_kernel(h_px, slab_px, radius, subsamples)


def channel_kernels(lam_nm, channel_weights: np.ndarray, pitch_um: float, cfg: dict | None = None) -> list[np.ndarray]:
    """One kernel per channel: wavelength kernels mixed by ``channel_weights`` ``[K, C]``."""
    lam_nm = np.asarray(lam_nm, float)
    w = np.clip(np.asarray(channel_weights, float), 0.0, None)
    kl = [wavelength_kernel(float(lam), pitch_um, cfg) for lam in lam_nm]
    out = []
    for c in range(w.shape[1]):
        tot = w[:, c].sum()
        if tot <= 0:
            out.append(kl[len(kl) // 2])
            continue
        out.append(sum(w[k, c] * kl[k] for k in range(len(kl)) if w[k, c] > 0) / tot)
    return out


def nearest_neighbour_fraction(kernel: np.ndarray) -> float:
    """Mean charge fraction leaking to one of the 4 edge-adjacent pixels."""
    r = kernel.shape[0] // 2
    return float((kernel[r - 1, r] + kernel[r + 1, r] + kernel[r, r - 1] + kernel[r, r + 1]) / 4.0)


def apply_site_kernels(raw_e: np.ndarray, site_index: np.ndarray, kernels: list[np.ndarray]) -> np.ndarray:
    """Spread each site's charge with its channel kernel (routed by *source* site)."""
    from scipy.ndimage import convolve  # noqa: PLC0415

    x = np.asarray(raw_e, dtype=np.float64)
    out = np.zeros_like(x)
    for i, k in enumerate(kernels):
        part = np.where(site_index == i, x, 0.0)
        out += convolve(part, k[::-1, ::-1], mode="reflect")
    return np.clip(out, 0.0, None).astype(np.float32)
