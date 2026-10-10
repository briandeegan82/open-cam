#!/usr/bin/env python3
"""Surface micro-roughness scatter (veiling glare) for a traced lens prescription.

Each refracting surface scatters a total integrated scatter (TIS) of

    TIS_i(lambda) = (2 pi sigma_i |n_after - n_before| / lambda)^2

of the light it transmits. This is the smooth-surface (Rayleigh-Rice / scalar) limit for
transmission (Stover, *Optical Scattering*, 3rd ed., ch. 2; Fest, *Stray Light Analysis and
Control*, 2013, ch. 4). The angular shape is an ABg (Harvey) BSDF,

    BSDF(beta) = A / (B + beta^g),

in direction-cosine space, beta = |sin(theta) - sin(theta_0)|. It is shift-invariant (Harvey &
Shack 1978), and A is chosen so that the BSDF integrated over the unit disc equals TIS.

The scattered light passes through the rest of the lens. Paraxially, a direction-cosine change
dw = n dbeta just after surface i shifts the ray's film landing point by r = |B_i| dw. Here B_i is
the w->y element of the paraxial matrix from surface i to the film, so surface i adds the film PSF

    p_i(r) = TIS_i * f(r / k_i) / k_i^2,   k_i = n_i |B_i|.

The scatter cone is truncated where its central ray leaves a later clear aperture:
beta_max,i = min_j a_j / (n_i |C_ij|). Light scattered at larger angles hits the barrel and is
counted as absorbed. The total PSF is (1 - sum TIS_i) delta + sum_i p_i.

Not modelled: particulate contamination, edge and mount scatter, scatter of out-of-field light,
and the wavelength dependence of the BSDF *shape* (only TIS scales as lambda^-2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DEFAULTS = {
    # Conventional commercial polish on optical glass: 1-2 nm rms (Stover ch. 2; Fest ch. 4).
    "roughness_nm": 1.5,
    # ABg fall-off and shoulder; g = 2 gives the -2 Harvey slope typical of polished glass.
    "abg_g": 2.0,
    "abg_shoulder": 0.01,
    "reference_nm": 550.0,
}


@dataclass
class SurfaceScatter:
    index: int
    delta_n: float
    k_mm: float  # film radius per unit direction cosine
    beta_max: float  # scatter-cone truncation by later apertures

    def tis(self, lam_nm, roughness_nm: float) -> np.ndarray:
        return (2.0 * math.pi * roughness_nm * self.delta_n / np.asarray(lam_nm, float)) ** 2


def _after_matrices(lens, i: int) -> tuple[list[np.ndarray], np.ndarray]:
    """Paraxial (y, w) matrices from just after surface ``i`` to each later surface and to the film."""
    surfs = lens.surfaces
    n = surfs[i].n_after
    z = surfs[i].z
    m = np.eye(2)
    to_surf = []
    for s in surfs[i + 1 :]:
        m = np.array([[1.0, (s.z - z) / n], [0.0, 1.0]]) @ m
        z = s.z
        to_surf.append((s, m.copy()))
        if not s.is_stop:
            m = np.array([[1.0, 0.0], [-(s.n_after - n) / s.radius, 1.0]]) @ m
            n = s.n_after
    m = np.array([[1.0, (lens.z_film - z) / n], [0.0, 1.0]]) @ m
    return to_surf, m


def surface_scatter(lens) -> list[SurfaceScatter]:
    out = []
    for i, s in enumerate(lens.surfaces):
        if s.is_stop or s.n_after == s.n_before:
            continue
        to_surf, m_film = _after_matrices(lens, i)
        n_i = s.n_after
        w_max = math.inf
        for sj, mj in to_surf:
            c = abs(mj[0, 1])
            if c > 0:
                w_max = min(w_max, sj.ap_r / c)
        k = n_i * abs(m_film[0, 1])
        out.append(SurfaceScatter(i, abs(s.n_after - s.n_before), k, min(1.0, w_max / n_i)))
    return out


def abg_shape(beta, g: float, shoulder: float) -> np.ndarray:
    """ABg shape normalised to integrate to 1 over the unit disc (A/(B + beta^g), B = shoulder^g)."""
    b = shoulder**g
    beta = np.asarray(beta, float)
    return (1.0 / (b + beta**g)) / abg_norm(g, shoulder, 1.0)


def abg_norm(g: float, shoulder: float, beta_max: float) -> float:
    """Integral of 1/(B + beta^g) over the disc beta <= beta_max."""
    b = shoulder**g
    if abs(g - 2.0) < 1e-12:
        return math.pi * math.log1p(beta_max**2 / b)
    x = np.linspace(0.0, beta_max, 20001)
    y = 2 * math.pi * x / (b + x**g)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def scatter_kernel(
    lens, shape: tuple[int, int], pixel_mm: float, cfg: dict | None = None
) -> tuple[np.ndarray, float, list[dict]]:
    """Film scatter kernel at ``reference_nm`` on a (2H-1, 2W-1) grid, centred.

    Returns (kernel per unit input, total TIS at reference_nm, per-surface report). Kernel values
    are energy fractions per pixel; light that lands beyond the grid or is vignetted is dropped.
    """
    c = {**DEFAULTS, **(cfg or {})}
    h, w = shape
    yy, xx = np.mgrid[-(h - 1) : h, -(w - 1) : w]
    r_mm = np.hypot(yy, xx) * pixel_mm
    g, sh = float(c["abg_g"]), float(c["abg_shoulder"])
    kern = np.zeros(r_mm.shape)
    report = []
    tis_total = 0.0
    lam0 = float(c["reference_nm"])
    for s in surface_scatter(lens):
        tis = float(s.tis(lam0, float(c["roughness_nm"])))
        tis_total += tis
        beta = r_mm / s.k_mm if s.k_mm > 0 else np.full_like(r_mm, np.inf)
        dens = np.where(beta <= s.beta_max, abg_shape(beta, g, sh), 0.0)
        pk = dens * (pixel_mm / s.k_mm) ** 2 if s.k_mm > 0 else np.zeros_like(dens)
        # Centre pixel: integrate the disc of equal area analytically (the BSDF peaks there).
        r0 = min(pixel_mm / math.sqrt(math.pi) / max(s.k_mm, 1e-12), s.beta_max)
        pk[h - 1, w - 1] = abg_norm(g, sh, r0) / abg_norm(g, sh, 1.0)
        kern += tis * pk
        report.append(
            {
                "surface": s.index,
                "tis": tis,
                "k_mm": s.k_mm,
                "beta_max": s.beta_max,
                "reaches_film": abg_norm(g, sh, s.beta_max) / abg_norm(g, sh, 1.0),
            }
        )
    return kern, tis_total, report


def _fft_conv_same(img: np.ndarray, kern_f: np.ndarray, fshape: tuple[int, int], kshape) -> np.ndarray:
    h, w = img.shape
    full = np.fft.irfft2(np.fft.rfft2(img, fshape) * kern_f, fshape)
    oy, ox = (kshape[0] - 1) // 2, (kshape[1] - 1) // 2
    return full[oy : oy + h, ox : ox + w]


def apply_scatter(cube: np.ndarray, lams_nm, lens, pixel_mm: float, cfg: dict | None = None):
    """Apply surface scatter to an (H, W, L) cube; channels with a NaN wavelength scale like reference_nm."""
    c = {**DEFAULTS, **(cfg or {})}
    h, w, nl = cube.shape
    kern, tis0, report = scatter_kernel(lens, (h, w), pixel_mm, c)
    fshape = (h + kern.shape[0] - 1, w + kern.shape[1] - 1)
    kern_f = np.fft.rfft2(kern, fshape)
    lam0 = float(c["reference_nm"])
    out = np.empty_like(cube, dtype=np.float32)
    for k in range(nl):
        lam = lams_nm[k] if lams_nm[k] is not None and np.isfinite(lams_nm[k]) else lam0
        scale = (lam0 / lam) ** 2
        img = cube[:, :, k].astype(np.float64)
        out[:, :, k] = ((1.0 - tis0 * scale) * img + scale * _fft_conv_same(img, kern_f, fshape, kern.shape)).astype(
            np.float32
        )
    return out, {"tis_reference": tis0, "reference_nm": lam0, "surfaces": report, **{k: c[k] for k in DEFAULTS}}
