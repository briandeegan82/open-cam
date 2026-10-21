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

Field dependence (``field_zones`` > 1, opt-in): off axis the scatter starts from the paraxial
chief ray at height y_i on surface i, so a later aperture a_j truncates the cone at
(a_j - |y_j|) / (n_i |C_ij|), and the obliquity of the chief ray at the surface scales TIS by
((n2 cos t - n1 cos i) / (n2 - n1))^2. Kernels are built at ``field_zones`` image radii and
blended with linear (hat) weights in the source's radius: out = sum_z K_z * (w_z img).

Barrel scatter (``barrel_reflectance`` > 0, opt-in, nominal): the scatter that the truncation
sends into the barrel is re-emitted diffusely with that reflectance and spread uniformly over
the frame (an upper bound: all of it is taken to reach the frame).

Not modelled: particulate contamination, edge scatter, scatter of out-of-field light, the angular
distribution of the barrel light, and the wavelength dependence of the BSDF *shape* (only TIS
scales as lambda^-2).
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
    # Opt-in: number of image radii with their own kernel (1 = shift-invariant, the old model).
    "field_zones": 1,
    # Opt-in: diffuse reflectance of the barrel for the truncated scatter (black anodised ~0.02-0.05).
    "barrel_reflectance": 0.0,
}


@dataclass
class SurfaceScatter:
    index: int
    delta_n: float
    k_mm: float  # film radius per unit direction cosine
    beta_max: float  # scatter-cone truncation by later apertures
    obliquity: float = 1.0  # TIS factor of the chief ray's incidence angle

    def tis(self, lam_nm, roughness_nm: float) -> np.ndarray:
        return self.obliquity * (2.0 * math.pi * roughness_nm * self.delta_n / np.asarray(lam_nm, float)) ** 2


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


def chief_ray(lens, u0: float) -> list[tuple[float, float, float]]:
    """Paraxial chief ray for field slope ``u0``: (y, w before, w after) at every surface (w = n u)."""
    import lens_ghosts as lg

    m_stop = lg.paraxial_path_matrix_partial(lens, lg.primary_path(lens)[: lens.stop_index + 1])
    y = -m_stop[0, 1] * u0 / m_stop[0, 0]
    w, n, z = u0, 1.0, lens.surfaces[0].z
    out = []
    for s in lens.surfaces:
        y += (s.z - z) / n * w
        z = s.z
        w_in = w
        if not s.is_stop and math.isfinite(s.radius):
            w = w - y * (s.n_after - n) / s.radius
        n = s.n_after if not s.is_stop else n
        out.append((y, w_in, w))
    return out


def film_height_slope(lens) -> float:
    """d(film height) / d(field slope) of the paraxial chief ray [mm]."""
    import lens_ghosts as lg

    return lg.paraxial_chief_height(lens, lg.primary_path(lens), 1e-3) / 1e-3


def _obliquity(s, y: float, w_in: float, n_in: float) -> float:
    """Rayleigh-Rice transmission TIS factor ((n2 cos t - n1 cos i) / (n2 - n1))^2 at incidence i."""
    tilt = math.asin(max(-1.0, min(1.0, y / s.radius))) if math.isfinite(s.radius) else 0.0
    i = math.atan(w_in / n_in) + tilt
    st = n_in * math.sin(i) / s.n_after
    if abs(st) >= 1.0:
        return 0.0
    dn = s.n_after * math.sqrt(1.0 - st * st) - n_in * math.cos(i)
    return (dn / (s.n_after - s.n_before)) ** 2


def surface_scatter(lens, u0: float = 0.0) -> list[SurfaceScatter]:
    """Per-surface scatter geometry; ``u0`` != 0 starts the cones from the chief ray at that field slope."""
    chief = chief_ray(lens, u0) if u0 else None
    out = []
    for i, s in enumerate(lens.surfaces):
        if s.is_stop or s.n_after == s.n_before:
            continue
        to_surf, m_film = _after_matrices(lens, i)
        n_i = s.n_after
        y_i, w_in, w_i = chief[i] if chief else (0.0, 0.0, 0.0)
        w_max = math.inf
        for sj, mj in to_surf:
            c = abs(mj[0, 1])
            if c > 0:
                y_j = mj[0, 0] * y_i + mj[0, 1] * w_i
                w_max = min(w_max, max(sj.ap_r - abs(y_j), 0.0) / c)
        k = n_i * abs(m_film[0, 1])
        obl = _obliquity(s, y_i, w_in, s.n_before) if chief else 1.0
        out.append(SurfaceScatter(i, abs(s.n_after - s.n_before), k, min(1.0, w_max / n_i), obl))
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
    lens, shape: tuple[int, int], pixel_mm: float, cfg: dict | None = None, u0: float = 0.0
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
    for s in surface_scatter(lens, u0):
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


def _zone_weights(h: int, w: int, radii_px: np.ndarray) -> np.ndarray:
    """(Z, H, W) linear hat weights in image radius; they sum to 1 at every pixel."""
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - 0.5 * (h - 1), xx - 0.5 * (w - 1))
    out = np.zeros((radii_px.size, h, w))
    for z in range(radii_px.size):
        e = np.zeros(radii_px.size)
        e[z] = 1.0
        out[z] = np.interp(r, radii_px, e)
    return out


def apply_scatter(cube: np.ndarray, lams_nm, lens, pixel_mm: float, cfg: dict | None = None):
    """Apply surface scatter to an (H, W, L) cube; channels with a NaN wavelength scale like reference_nm."""
    c = {**DEFAULTS, **(cfg or {})}
    h, w, nl = cube.shape
    nz = max(1, int(c["field_zones"]))
    radii = np.linspace(0.0, math.hypot(0.5 * (h - 1), 0.5 * (w - 1)), nz) if nz > 1 else np.zeros(1)
    slope = film_height_slope(lens) if nz > 1 else 1.0
    zones = []
    for r_px in radii:
        u0 = r_px * pixel_mm / slope if nz > 1 else 0.0
        kern, tis0, report = scatter_kernel(lens, (h, w), pixel_mm, c, u0)
        lost = sum(s["tis"] * (1.0 - s["reaches_film"]) for s in report)
        zones.append((kern, tis0, report, lost, u0))
    kshape = zones[0][0].shape
    fshape = (h + kshape[0] - 1, w + kshape[1] - 1)
    kern_fs = [np.fft.rfft2(z[0], fshape) for z in zones]
    weights = _zone_weights(h, w, radii) if nz > 1 else np.ones((1, h, w))
    rho_b = float(c["barrel_reflectance"])
    lam0 = float(c["reference_nm"])
    out = np.empty_like(cube, dtype=np.float32)
    for k in range(nl):
        lam = lams_nm[k] if lams_nm[k] is not None and np.isfinite(lams_nm[k]) else lam0
        scale = (lam0 / lam) ** 2
        img = cube[:, :, k].astype(np.float64)
        acc = np.zeros((h, w))
        barrel = 0.0
        for (_kern, tis0, _rep, lost, _u0), kf, wz in zip(zones, kern_fs, weights, strict=True):
            part = img * wz if nz > 1 else img
            acc += (1.0 - tis0 * scale) * part + scale * _fft_conv_same(part, kf, fshape, kshape)
            barrel += rho_b * lost * scale * float(part.sum())
        out[:, :, k] = (acc + barrel / (h * w)).astype(np.float32)
    kern, tis0, report, _lost, _u0 = zones[0]
    info = {"tis_reference": tis0, "reference_nm": lam0, "surfaces": report, **{k: c[k] for k in DEFAULTS}}
    if nz > 1 or rho_b > 0:
        info["field"] = [
            {
                "radius_px": float(r),
                "field_slope": z[4],
                "tis_reference": z[1],
                "reaches_film": float(sum(s["tis"] * s["reaches_film"] for s in z[2]) / max(z[1], 1e-30)),
                "barrel_energy": rho_b * z[3],
            }
            for r, z in zip(radii, zones, strict=True)
        ]
    return out, info
