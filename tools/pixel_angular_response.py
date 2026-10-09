#!/usr/bin/env python3
"""Pixel angular response: chief-ray angle (CRA), microlens shift, lens and colour shading.

A pixel's quantum efficiency (QE) curve is measured with near-normal illumination (EMVA 1288
uses an f/8 cone on the optical axis). Behind a real lens each pixel sees a cone of rays whose
axis, the chief ray, tilts outward as image height grows. A microlens focuses that cone onto
the collection window at the silicon surface. When the cone is tilted the focused spot moves
sideways and part of it misses the window (lens shading) or lands in the neighbouring pixel
(optical crosstalk). The loss depends on wavelength, so it also shifts R/G and B/G across the
field (colour shading). Sensors shift each microlens toward the optical centre so the chief ray
of the lens they were designed for lands on the window centre. Pairing them with a lens that has
a different CRA profile leaves residual shading (CRA mismatch). See Agranov, Berezin & Tsai,
"Crosstalk and microlens study in a color CMOS image sensor", IEEE Trans. Electron Devices 50(1),
4-11 (2003), doi:10.1109/TED.2002.806473, and Catrysse & Wandell, "Optical efficiency of image
sensor pixels", J. Opt. Soc. Am. A 19(8), 1610-1620 (2002), doi:10.1364/JOSAA.19.001610.

Model, in three parts:

1. **Incidence-angle distribution.** For each film point we need the set of ray directions that
   reach it, weighted by projected solid angle ``cos θ dω`` (equivalently, uniform in the
   transverse direction cosines ``(a, b)``):

   * ``traced``: real skew rays are traced from the film point through a pbrt-v4 lens
     prescription (``lens_prescription.load_lens_file``) with every element aperture and the
     stop clipping, the same way pbrt's ``RealisticCamera`` does. This includes pupil
     vignetting and the real (aberrated) chief ray.
   * ``table``: a CRA-vs-image-height table (``lens.chief_ray_angle_table``) plus a circular
     cone of numerical aperture ``1/(2N)`` around the chief ray.
   * ``pinhole``: stop at the lens, so CRA equals the field angle from the render's ``fov``,
     with the same ``1/(2N)`` cone.

2. **Microlens + stack (geometric optics with a diffraction term).** An ideal square lenslet
   of aperture ``D`` (the pitch, for a gapless array) and in-stack focal length ``f`` sits at
   height ``d`` above the silicon. The stack index is ``n_s``. For a plane wave with
   transverse direction cosines ``(a, b)``, Snell's law in the planar stack gives
   ``a_s = a/n_s`` and ``tan θ_s = a_s/√(1−a_s²−b_s²)``. The lenslet maps the aperture onto a
   box of width ``D·|1 − d/f|`` centred at ``d·tan θ_s + c``, where ``c`` is the microlens shift.
   Wavelength enters in two places:

   * **Diffraction.** The box is blurred by a Gaussian of σ = 0.21 λ/NA, where NA is the
     lenslet's numerical aperture in the stack. This is the paraxial Airy-pattern fit of Zhang,
     Zerubia & Olivo-Marin, Appl. Opt. 46(10), 1819-1829 (2007), doi:10.1364/AO.46.001819.
   * **Penetration into silicon.** Photons are absorbed at depth ``z`` with density ∝ e^{−αz},
     truncated at the collection depth ``z_c``, with α = 4πk/λ from Green, Sol. Energy Mater.
     Sol. Cells 92, 1305-1310 (2008), doi:10.1016/j.solmat.2008.06.009 (CC0 data in
     ``spectra/silicon/si_green2008_nk.csv``). Below the surface the beam keeps travelling at
     the silicon refraction angle (``a_si = a/n_si``). Its centre therefore moves a further
     ``z·tan θ_si``, and the converging cone acts as if the stack were ``z·n_s/n_si`` thicker
     (paraxial). Red light is absorbed deep, so it walks off further: this is the main
     source of colour shading and colour crosstalk.

   The fraction collected by a square window of width ``w`` is a closed form (the separable
   box ⊛ Gaussian overlap). The same expression shifted by ±pitch gives what reaches the 8
   neighbouring windows (optical crosstalk). Light falling between windows is lost.

3. **Microlens shift (designed CRA profile).** ``c = −d·tan θ_s,design(h)`` points toward the
   optical centre. Modes: ``none`` (no shift), ``matched`` (designed for this lens: the
   centroid of its incidence distribution, i.e. the energy-weighted chief ray), ``linear``
   (CRA rises linearly to ``max_cra_deg`` at the frame corner, as sensor datasheets specify),
   and ``table``.

Responses are normalised to normal incidence on an unshifted pixel, which is the condition the
QE curve represents (``normalize: normal_incidence``). ``normalize: center`` instead makes the
centre pixel 1. The result is a per-pixel, per-wavelength weight ``R(x, y, λ)``. It multiplies
the spectral electron weights in ``pbrt_spectral_exr_to_electrons.spectral_radiance_to_electrons``.
Optional crosstalk moves each pixel's filtered light into its neighbours' CFA sites.

Approximations and things not modelled: the lenslet is ideal and thin (no microlens
aberrations, no curvature-dependent Fresnel loss). Diffraction is a Gaussian approximation; for
pitches ≲ 2 µm wave-optics (FDTD) effects matter, see Huo, Fesenmaier & Catrysse,
"Microlens performance limits in sub-2µm pixel CMOS image sensors", Opt. Express 18(6),
5861-5872 (2010), doi:10.1364/OE.18.005861. The colour filter is treated as sitting at the
microlens, so light is filtered by its own pixel's CFA. Metal-layer shadowing is lumped into
the window width. There is no polarisation, and carrier diffusion is left to
``cfa.spatial_crosstalk``. ``traced`` uses infinity focus; ``table`` and ``pinhole`` use a
circular cone in direction-cosine space. The default stack parameters are illustrative, not
measured; set them from the sensor's stack data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from lens_prescription import load_lens_file, paraxial_focal_lengths_mm
from scipy.special import ndtr

SILICON_NK_CSV = "spectra/silicon/si_green2008_nk.csv"
PBRT_DEFAULT_FILM_DIAGONAL_MM = 35.0  # pbrt-v4 film "diagonal" default (src/pbrt/film.cpp)
AIRY_GAUSSIAN_SIGMA_COEFF = 0.21  # σ = 0.21 λ/NA (Zhang et al. 2007, paraxial 2D widefield)
_BAYER_CHANNEL = {"R": 0, "G": 1, "B": 2}


# =====================================================================
# Pixel stack and silicon data
# =====================================================================
@dataclass(frozen=True)
class PixelStack:
    """Microlens/stack geometry of one pixel (lengths in µm). Defaults are illustrative."""

    pitch_um: float
    stack_height_um: float = 2.0
    microlens_focal_um: float | None = None
    microlens_aperture_um: float | None = None
    stack_index: float = 1.55
    window_fraction: float = 0.8
    collection_depth_um: float = 3.0
    diffraction: bool = True
    silicon_penetration: bool = True
    depth_samples: int = 8

    @property
    def focal_um(self) -> float:
        return float(self.microlens_focal_um or self.stack_height_um)

    @property
    def aperture_um(self) -> float:
        return float(self.microlens_aperture_um or self.pitch_um)

    @property
    def window_um(self) -> float:
        return float(self.window_fraction) * float(self.pitch_um)

    @property
    def numerical_aperture(self) -> float:
        return self.stack_index * math.sin(math.atan(0.5 * self.aperture_um / self.focal_um))

    @classmethod
    def from_config(cls, cfg: dict | None, pitch_um: float) -> PixelStack:
        cfg = dict(cfg or {})
        known = {f for f in cls.__dataclass_fields__ if f != "pitch_um"}
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(f"unknown pixel_angular_response.stack keys: {sorted(unknown)}")
        return cls(pitch_um=float(pitch_um), **cfg)


@lru_cache(maxsize=4)
def _silicon_nk(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", comments="#", skiprows=5)
    return data[:, 0], data[:, 1], data[:, 2]


def silicon_optical_constants(lambdas_nm: np.ndarray, repo: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(n_si, alpha_per_um)`` of crystalline silicon at 300 K (Green 2008) on ``lambdas_nm``."""
    path = Path(repo) / SILICON_NK_CSV if repo is not None else None
    if path is None or not path.is_file():
        path = Path(__file__).resolve().parents[1] / SILICON_NK_CSV  # data ships with the tools
    wl, n, k = _silicon_nk(str(path.resolve()))
    lam = np.asarray(lambdas_nm, dtype=np.float64)
    n_si = np.interp(lam, wl, n)
    k_si = np.interp(lam, wl, k)
    return n_si, 4.0 * math.pi * k_si / (lam * 1e-3)


# =====================================================================
# Closed-form collection fraction (box ⊛ Gaussian over a window)
# =====================================================================
def _int_phi(x: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Antiderivative of Φ(x/σ): x Φ(x/σ) + σ φ(x/σ)."""
    t = x / sigma
    return x * ndtr(t) + sigma * np.exp(-0.5 * t * t) / math.sqrt(2.0 * math.pi)


def window_fraction_1d(delta, spot_width, sigma, window) -> np.ndarray:
    """Fraction of a 1D box spot (width ``spot_width``, centre ``delta``) blurred by a Gaussian
    ``sigma`` that falls inside ``[-window/2, window/2]``. Exact; σ→0 gives the box overlap."""
    s = np.maximum(np.asarray(spot_width, dtype=np.float64), 1e-6)
    sig = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-9)
    hw = 0.5 * float(window)
    hi = hw - np.asarray(delta, dtype=np.float64)
    lo = -hw - np.asarray(delta, dtype=np.float64)
    return (
        _int_phi(hi + 0.5 * s, sig)
        - _int_phi(hi - 0.5 * s, sig)
        - _int_phi(lo + 0.5 * s, sig)
        + _int_phi(lo - 0.5 * s, sig)
    ) / s


def _depth_quantiles_um(alpha_per_um: np.ndarray, depth_um: float, n: int) -> np.ndarray:
    """Mid-quantile absorption depths [K, n] of e^{-αz} truncated to [0, depth_um]."""
    q = (np.arange(n) + 0.5) / n
    a = np.asarray(alpha_per_um, dtype=np.float64)[:, None]
    ad = a * depth_um
    with np.errstate(divide="ignore", invalid="ignore"):
        z = -np.log1p(-q[None, :] * (-np.expm1(-ad))) / a
    return np.where(ad < 1e-6, q[None, :] * depth_um, z)


def stack_tangents(a: np.ndarray, b: np.ndarray, index) -> tuple[np.ndarray, np.ndarray]:
    """Ray slopes (tan θx, tan θy) after refraction from air into a planar medium of ``index``."""
    a_s, b_s = np.asarray(a) / index, np.asarray(b) / index
    c_s = np.sqrt(np.clip(1.0 - a_s * a_s - b_s * b_s, 1e-12, None))
    return a_s / c_s, b_s / c_s


def pixel_response(
    a: np.ndarray,
    b: np.ndarray,
    weights: np.ndarray,
    shift_um: tuple[float, float],
    lambdas_nm: np.ndarray,
    stack: PixelStack,
    *,
    repo: Path | None = None,
) -> np.ndarray:
    """Collected fractions [3, 3, K] for incident directions ``(a, b)`` (air transverse direction
    cosines of the light, weighted by projected solid angle). Index ``[1 + ky, 1 + kx]`` is the
    window displaced by ``(kx, ky)`` pitches; ``[1, 1]`` is the pixel itself. Not normalised."""
    a = np.atleast_1d(np.asarray(a, dtype=np.float64))
    b = np.atleast_1d(np.asarray(b, dtype=np.float64))
    w = np.atleast_1d(np.asarray(weights, dtype=np.float64))
    lam = np.atleast_1d(np.asarray(lambdas_nm, dtype=np.float64))
    if a.size == 0 or float(np.sum(w)) <= 0.0:
        return np.zeros((3, 3, lam.size))
    d, f, n_s = stack.stack_height_um, stack.focal_um, stack.stack_index
    tx, ty = stack_tangents(a, b, n_s)
    if stack.silicon_penetration:
        n_si, alpha = silicon_optical_constants(lam, repo)
        z = _depth_quantiles_um(alpha, stack.collection_depth_um, max(1, int(stack.depth_samples)))
        sx, sy = stack_tangents(a[:, None], b[:, None], n_si[None, :])  # [N, K]
        d_eff = d + z * (n_s / n_si)[:, None]  # [K, Q]
    else:
        z = np.zeros((lam.size, 1))
        sx = sy = np.zeros((a.size, lam.size))
        d_eff = np.full((lam.size, 1), d)
    spot = stack.aperture_um * np.abs(1.0 - d_eff / f)  # [K, Q]
    sigma = AIRY_GAUSSIAN_SIGMA_COEFF * lam * 1e-3 / stack.numerical_aperture if stack.diffraction else 0.0 * lam
    sig = np.broadcast_to(np.asarray(sigma)[:, None], spot.shape)
    dx = d * tx[:, None, None] + z[None] * sx[:, :, None] + float(shift_um[0])  # [N, K, Q]
    dy = d * ty[:, None, None] + z[None] * sy[:, :, None] + float(shift_um[1])
    p, win = stack.pitch_um, stack.window_um
    fx = np.stack([window_fraction_1d(dx - k * p, spot, sig, win) for k in (-1, 0, 1)])
    fy = np.stack([window_fraction_1d(dy - k * p, spot, sig, win) for k in (-1, 0, 1)])
    norm = float(np.sum(w)) * z.shape[1]
    return np.einsum("xnkq,ynkq,n->yxk", fx, fy, w) / norm


def normal_incidence_response(lambdas_nm: np.ndarray, stack: PixelStack, *, repo: Path | None = None) -> np.ndarray:
    """Own-window collection [K] for a collimated normal beam on an unshifted pixel (QE reference)."""
    return pixel_response(np.zeros(1), np.zeros(1), np.ones(1), (0.0, 0.0), lambdas_nm, stack, repo=repo)[1, 1]


# =====================================================================
# Incidence-angle distribution from a lens prescription (3D trace)
# =====================================================================
def _eta(e: float) -> float:
    return 1.0 if e == 0.0 else e


def trace_from_film(rows, film_distance_mm, stop_diameter_mm, px, py, dx, dy, dz, *, clip: bool = True):
    """Trace rays from film points ``(px, py, 0)`` along unit directions ``(dx, dy, dz>0)`` toward the
    object through a pbrt lens prescription (film at z=0, lens toward +z, like pbrt's
    ``TraceLensesFromFilm``). Returns ``(ok, stop_x, stop_y, (vx, vy, vz))``."""
    px, py, dx, dy, dz = np.broadcast_arrays(*(np.asarray(v, dtype=np.float64) for v in (px, py, dx, dy, dz)))
    ox, oy, oz = px.copy(), py.copy(), np.zeros_like(px)
    vx, vy, vz = dx.copy(), dy.copy(), dz.copy()
    ok = np.ones(px.shape, dtype=bool)
    stop_x = stop_y = np.full(px.shape, np.nan)
    z_vertex = float(film_distance_mm)
    with np.errstate(invalid="ignore", divide="ignore"):
        for i in range(len(rows) - 1, -1, -1):
            radius, _t, eta, aperture = rows[i]
            if i < len(rows) - 1:
                z_vertex += rows[i][1]
            if radius == 0.0:
                s = (z_vertex - oz) / vz
                ox, oy, oz = ox + s * vx, oy + s * vy, oz + s * vz
                stop_x, stop_y = ox.copy(), oy.copy()
                if clip:
                    ok &= ox * ox + oy * oy <= (0.5 * min(stop_diameter_mm, aperture)) ** 2
                continue
            cz = z_vertex - radius
            qz = oz - cz
            bq = ox * vx + oy * vy + qz * vz
            disc = bq * bq - (ox * ox + oy * oy + qz * qz - radius * radius)
            ok &= disc >= 0.0
            sq = np.sqrt(np.maximum(disc, 0.0))
            t1, t2 = -bq - sq, -bq + sq
            z1, z2 = oz + t1 * vz, oz + t2 * vz
            pick1 = (t1 > 1e-9) & ((np.abs(z1 - z_vertex) <= np.abs(z2 - z_vertex)) | (t2 <= 1e-9))
            t = np.where(pick1, t1, t2)
            ok &= t > 1e-9
            ox, oy, oz = ox + t * vx, oy + t * vy, oz + t * vz
            if clip:
                ok &= ox * ox + oy * oy <= (0.5 * aperture) ** 2
            nx, ny, nz = ox / radius, oy / radius, (oz - cz) / radius
            flip = nx * vx + ny * vy + nz * vz > 0.0
            nx, ny, nz = np.where(flip, -nx, nx), np.where(flip, -ny, ny), np.where(flip, -nz, nz)
            ratio = _eta(eta) / (_eta(rows[i - 1][2]) if i > 0 else 1.0)
            cos_i = -(nx * vx + ny * vy + nz * vz)
            k = 1.0 - ratio * ratio * (1.0 - cos_i * cos_i)
            ok &= k >= 0.0
            g = ratio * cos_i - np.sqrt(np.maximum(k, 0.0))
            vx, vy, vz = ratio * vx + g * nx, ratio * vy + g * ny, ratio * vz + g * nz
            nrm = np.sqrt(vx * vx + vy * vy + vz * vz)
            vx, vy, vz = vx / nrm, vy / nrm, vz / nrm
    ok &= np.isfinite(vx) & np.isfinite(stop_x)
    return ok, stop_x, stop_y, (vx, vy, vz)


@dataclass(frozen=True)
class TracedLens:
    rows: tuple
    stop_diameter_mm: float
    film_distance_mm: float

    @classmethod
    def from_file(cls, lens_file: str | Path, aperture_diameter_mm: float | None = None) -> TracedLens:
        rows = load_lens_file(Path(lens_file))
        stop_max = next(r[3] for r in rows if r[0] == 0.0)
        stop = stop_max if aperture_diameter_mm is None else min(float(aperture_diameter_mm), stop_max)
        _efl, bfd = paraxial_focal_lengths_mm(rows)
        return cls(rows=rows, stop_diameter_mm=float(stop), film_distance_mm=float(bfd))

    def _trace(self, x_mm, y_mm, dl, dm, *, clip=True):
        n = np.sqrt(np.clip(1.0 - dl * dl - dm * dm, 0.0, None))
        return trace_from_film(
            self.rows, self.film_distance_mm, self.stop_diameter_mm, x_mm, y_mm, dl, dm, n, clip=clip
        )

    def incidence_directions(self, x_mm: float, y_mm: float, n_grid: int = 24):
        """Air transverse direction cosines ``(a, b)`` of the light reaching film point ``(x, y)`` and
        their projected-solid-angle weights (cell areas). Sum of weights = projected solid angle."""
        for coarse in (64, 256):
            c = (np.arange(coarse) + 0.5) / coarse * 2.0 - 1.0
            ll, mm = np.meshgrid(c, c, indexing="xy")
            inside = ll * ll + mm * mm < 0.995
            ok = self._trace(x_mm, y_mm, ll[inside], mm[inside])[0]
            if np.count_nonzero(ok) >= 4:
                break
        else:
            return np.zeros(0), np.zeros(0), np.zeros(0)
        cell = 2.0 / coarse
        lo_l, hi_l = ll[inside][ok].min() - cell, ll[inside][ok].max() + cell
        lo_m, hi_m = mm[inside][ok].min() - cell, mm[inside][ok].max() + cell
        gl = lo_l + (np.arange(n_grid) + 0.5) / n_grid * (hi_l - lo_l)
        gm = lo_m + (np.arange(n_grid) + 0.5) / n_grid * (hi_m - lo_m)
        ll, mm = np.meshgrid(gl, gm, indexing="xy")
        ll, mm = ll.ravel(), mm.ravel()
        keep = ll * ll + mm * mm < 0.9999
        ll, mm = ll[keep], mm[keep]
        ok = self._trace(x_mm, y_mm, ll, mm)[0]
        area = (hi_l - lo_l) * (hi_m - lo_m) / (n_grid * n_grid)
        # Light travels toward the film: its direction is the reverse of the traced ray.
        return -ll[ok], -mm[ok], np.full(np.count_nonzero(ok), area)

    def chief_ray_angle_deg(self, h_mm: float) -> float:
        """Real chief ray (through the stop centre) at image height ``h_mm`` in the meridional plane."""
        if h_mm == 0.0:
            return 0.0
        ls = np.linspace(-0.95, 0.95, 381)
        ok, sx, _sy, _v = self._trace(h_mm, 0.0, ls, np.zeros_like(ls), clip=False)
        sx = np.where(ok, sx, np.nan)
        idx = np.where(np.isfinite(sx[:-1]) & np.isfinite(sx[1:]) & (np.sign(sx[:-1]) != np.sign(sx[1:])))[0]
        if idx.size == 0:
            raise ValueError(f"no chief ray found at image height {h_mm} mm")
        lo, hi = float(ls[idx[0]]), float(ls[idx[0] + 1])
        f_lo = float(sx[idx[0]])
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            f_mid = float(self._trace(h_mm, 0.0, np.array([mid]), np.zeros(1), clip=False)[1][0])
            if np.sign(f_mid) == np.sign(f_lo):
                lo, f_lo = mid, f_mid
            else:
                hi = mid
        return math.degrees(math.asin(abs(0.5 * (lo + hi))))


def paraxial_exit_pupil_distance_mm(rows) -> float:
    """Distance from the paraxial infinity-focus image plane to the exit pupil (positive: pupil in
    front of the film). Paraxial CRA: ``tan θ = h / distance``."""
    _efl, bfd = paraxial_focal_lengths_mm(rows)
    i_stop = next(i for i, r in enumerate(rows) if r[0] == 0.0)
    n, y, u = 1.0, 0.0, 1.0
    y += rows[i_stop][1] * u
    for j in range(i_stop + 1, len(rows)):
        radius, thickness, eta, _ap = rows[j]
        n2 = _eta(eta)
        power = 0.0 if radius == 0.0 else (n2 - n) / radius
        u = (n * u - y * power) / n2
        n = n2
        if j < len(rows) - 1:
            y += thickness * u
    return bfd - (-y / u)


# =====================================================================
# Pupil cone model (CRA table / pinhole)
# =====================================================================
def cone_directions(cra_deg: float, azimuth_xy: tuple[float, float], na: float, n_grid: int = 24):
    """Uniform circular cone of numerical aperture ``na`` around the chief ray, in direction-cosine space."""
    ux, uy = azimuth_xy
    s0 = math.sin(math.radians(cra_deg))
    a0, b0 = s0 * ux, s0 * uy
    g = ((np.arange(n_grid) + 0.5) / n_grid * 2.0 - 1.0) * na
    ga, gb = np.meshgrid(g, g, indexing="xy")
    inside = ga * ga + gb * gb <= na * na
    a, b = a0 + ga[inside], b0 + gb[inside]
    keep = a * a + b * b < 0.9999
    area = (2.0 * na / n_grid) ** 2
    return a[keep], b[keep], np.full(np.count_nonzero(keep), area)


# =====================================================================
# Field maps
# =====================================================================
@dataclass
class AngularResponseMaps:
    """Relative responses on a coarse field grid (pixel coordinates)."""

    grid_rows: np.ndarray  # [Gy] pixel row coordinates
    grid_cols: np.ndarray  # [Gx]
    own: np.ndarray  # [Gy, Gx, K]
    neighbours: np.ndarray  # [Gy, Gx, 3, 3, K]; [.., 1+ky, 1+kx, :] film offset (kx, ky) pitches, +y up
    chief_ray_deg: np.ndarray  # [Gy, Gx] centroid CRA of the incidence distribution
    design_cra_deg: np.ndarray  # [Gy, Gx]
    projected_solid_angle: np.ndarray  # [Gy, Gx]
    image_height_norm: np.ndarray  # [Gy, Gx]
    meta: dict


def film_size_mm(diagonal_mm: float, xres: int, yres: int) -> tuple[float, float]:
    """pbrt-v4 film physical extent (RealisticCamera constructor)."""
    aspect = yres / xres
    x = math.sqrt(diagonal_mm**2 / (1.0 + aspect**2))
    return x, aspect * x


def _design_cra_deg(mode: str, cfg: dict, h_norm: float, lens_cra_deg: float) -> float:
    if mode == "none":
        return 0.0
    if mode == "matched":
        return lens_cra_deg
    if mode == "linear":
        return float(cfg.get("max_cra_deg", 25.0)) * h_norm
    if mode == "table":
        tab = np.asarray(cfg.get("table"), dtype=np.float64)
        if tab.ndim != 2 or tab.shape[1] != 2:
            raise ValueError("microlens_shift.table must be [[image_height_norm, cra_deg], ...]")
        return float(np.interp(h_norm, tab[:, 0], tab[:, 1]))
    raise ValueError(f"microlens_shift.mode must be none|matched|linear|table, got {mode!r}")


def incidence_source(cfg: dict, lens_cfg: dict | None, geometry: dict) -> str:
    src = str(cfg.get("incidence", "auto")).lower()
    lens = lens_cfg or {}
    if src != "auto":
        return src
    if geometry.get("lensfile") or (
        str(lens.get("camera", "")).lower() == "realistic" and lens.get("realistic_lensfile")
    ):
        return "traced"
    if lens.get("chief_ray_angle_table"):
        return "table"
    return "pinhole"


def compute_maps(
    xres: int,
    yres: int,
    lambdas_nm: np.ndarray,
    cfg: dict,
    *,
    pitch_um: float,
    f_number: float,
    lens_cfg: dict | None = None,
    geometry: dict | None = None,
    repo: Path | None = None,
) -> AngularResponseMaps:
    """Relative own/neighbour response on a field grid for an ``xres`` x ``yres`` frame."""
    repo = Path(repo) if repo is not None else Path(__file__).resolve().parents[1]
    geometry = dict(geometry or {})
    lens = lens_cfg or {}
    stack = PixelStack.from_config(cfg.get("stack"), pitch_um)
    shift_cfg = dict(cfg.get("microlens_shift") or {"mode": "matched"})
    shift_mode = str(shift_cfg.get("mode", "matched")).lower()
    source = incidence_source(cfg, lens, geometry)
    n_pupil = int(cfg.get("pupil_grid", 24))
    gy, gx = (int(v) for v in cfg.get("field_grid", (9, 13)))
    lam = np.asarray(lambdas_nm, dtype=np.float64)

    cx, cy = 0.5 * xres, 0.5 * yres
    rows_px = np.linspace(0.5, yres - 0.5, gy) if gy > 1 else np.array([0.5 * yres])
    cols_px = np.linspace(0.5, xres - 0.5, gx) if gx > 1 else np.array([0.5 * xres])
    r_corner_px = math.hypot(cx - 0.5, cy - 0.5) or 1.0
    meta: dict = {"incidence": source, "microlens_shift": shift_mode, "stack": dict(stack.__dict__)}

    if source == "traced":
        lensfile = geometry.get("lensfile") or lens.get("realistic_lensfile")
        if not lensfile:
            raise ValueError("pixel_angular_response incidence=traced needs lens.realistic_lensfile")
        lens_path = Path(lensfile) if Path(lensfile).is_absolute() else repo / str(lensfile)
        ap = geometry.get("aperture_diameter_mm", lens.get("realistic_aperture_diameter_mm"))
        tl = TracedLens.from_file(lens_path, None if ap is None else float(ap))
        diag = float(cfg.get("film_diagonal_mm") or geometry.get("film_diagonal_mm") or PBRT_DEFAULT_FILM_DIAGONAL_MM)
        fw, fh = film_size_mm(diag, xres, yres)
        mm_per_px_x, mm_per_px_y = fw / xres, fh / yres
        meta.update({"lensfile": str(lensfile), "film_diagonal_mm": diag, "stop_diameter_mm": tl.stop_diameter_mm})
    elif source == "table":
        table = np.asarray(lens.get("chief_ray_angle_table"), dtype=np.float64)
        if table.ndim != 2 or table.shape[1] != 2:
            raise ValueError("lens.chief_ray_angle_table must be [[image_height_norm, cra_deg], ...]")
    elif source == "pinhole":
        fov = cfg.get("fov_deg") or geometry.get("fov_deg")
        if fov is None:
            raise ValueError("pixel_angular_response incidence=pinhole needs fov_deg (config or scene manifest)")
        tan_half = math.tan(math.radians(float(fov)) * 0.5)
        meta["fov_deg"] = float(fov)
    else:
        raise ValueError(f"pixel_angular_response.incidence must be auto|traced|table|pinhole, got {source!r}")
    na = 1.0 / (2.0 * float(f_number))

    own = np.zeros((gy, gx, lam.size))
    nb = np.zeros((gy, gx, 3, 3, lam.size))
    cra = np.zeros((gy, gx))
    design = np.zeros((gy, gx))
    psa = np.zeros((gy, gx))
    hn = np.zeros((gy, gx))
    for iy, r in enumerate(rows_px):
        for ix, c in enumerate(cols_px):
            ux_px, uy_px = c - cx, cy - r  # +x right, +y up (film coordinates, up to a point flip)
            rad = math.hypot(ux_px, uy_px)
            az = (ux_px / rad, uy_px / rad) if rad > 0 else (1.0, 0.0)
            hn[iy, ix] = rad / r_corner_px
            if source == "traced":
                a, b, w = tl.incidence_directions(ux_px * mm_per_px_x, uy_px * mm_per_px_y, n_pupil)
            else:
                if source == "table":
                    cra_ij = float(np.interp(hn[iy, ix], table[:, 0], table[:, 1]))
                else:
                    cra_ij = math.degrees(math.atan(rad / (0.5 * min(xres, yres)) * tan_half))
                a, b, w = cone_directions(cra_ij, az, na, n_pupil)
            psa[iy, ix] = float(np.sum(w))
            if w.size == 0:
                continue
            a_c, b_c = float(np.average(a, weights=w)), float(np.average(b, weights=w))
            cra[iy, ix] = math.degrees(math.asin(min(1.0, math.hypot(a_c, b_c))))
            design[iy, ix] = _design_cra_deg(shift_mode, shift_cfg, hn[iy, ix], cra[iy, ix])
            s_air = math.sin(math.radians(design[iy, ix]))
            tdx, tdy = stack_tangents(np.array([s_air * az[0]]), np.array([s_air * az[1]]), stack.stack_index)
            shift = (-stack.stack_height_um * float(tdx[0]), -stack.stack_height_um * float(tdy[0]))
            resp = pixel_response(a, b, w, shift, lam, stack, repo=repo)
            own[iy, ix] = resp[1, 1]
            nb[iy, ix] = resp
    norm_mode = str(cfg.get("normalize", "normal_incidence")).lower()
    if norm_mode == "normal_incidence":
        ref = normal_incidence_response(lam, stack, repo=repo)
    elif norm_mode == "center":
        ref = _bilinear_point(own, rows_px, cols_px, cy, cx)
    else:
        raise ValueError(f"pixel_angular_response.normalize must be normal_incidence|center, got {norm_mode!r}")
    ref = np.where(ref > 0, ref, 1.0)
    own /= ref
    nb /= ref
    meta.update(
        {
            "normalize": norm_mode,
            "field_grid": [gy, gx],
            "pupil_grid": n_pupil,
            "cone_na": na if source != "traced" else None,
            "max_chief_ray_deg": float(cra.max()),
            "max_design_cra_deg": float(design.max()),
            "own_min": float(own.min()),
            "own_max": float(own.max()),
        }
    )
    return AngularResponseMaps(rows_px, cols_px, own, nb, cra, design, psa, hn, meta)


def _bilinear_point(grid: np.ndarray, rows: np.ndarray, cols: np.ndarray, r: float, c: float) -> np.ndarray:
    i0, i1, ty = _axis_weights(rows, np.array([r]))
    j0, j1, tx = _axis_weights(cols, np.array([c]))
    top = (1 - tx[0]) * grid[i0[0], j0[0]] + tx[0] * grid[i0[0], j1[0]]
    bot = (1 - tx[0]) * grid[i1[0], j0[0]] + tx[0] * grid[i1[0], j1[0]]
    return (1 - ty[0]) * top + ty[0] * bot


def _axis_weights(grid: np.ndarray, coords: np.ndarray):
    if grid.size == 1:
        z = np.zeros(coords.size, dtype=int)
        return z, z, np.zeros(coords.size)
    pos = np.clip(np.interp(coords, grid, np.arange(grid.size)), 0, grid.size - 1)
    i0 = np.minimum(np.floor(pos).astype(int), grid.size - 2)
    return i0, i0 + 1, pos - i0


def channel_response(maps: AngularResponseMaps, weights: np.ndarray) -> np.ndarray:
    """Own response per colour channel [Gy, Gx, C] for an equal-energy spectrum through ``weights`` [K, C]."""
    w = np.asarray(weights, dtype=np.float64)
    return np.einsum("yxk,kc->yxc", maps.own, w) / np.maximum(w.sum(axis=0), 1e-300)


# =====================================================================
# Application to spectral planes
# =====================================================================
def bayer_channel_index(pattern: str, yres: int, xres: int) -> np.ndarray:
    """Channel index (0=R, 1=G, 2=B) sampled at each pixel for a 2x2 pattern string, e.g. ``RGGB``."""
    p = str(pattern).upper()
    if len(p) != 4 or any(ch not in _BAYER_CHANNEL for ch in p):
        raise ValueError(f"crosstalk.cfa_pattern must be a 2x2 R/G/B pattern like RGGB, got {pattern!r}")
    tile = np.array([[_BAYER_CHANNEL[p[0]], _BAYER_CHANNEL[p[1]]], [_BAYER_CHANNEL[p[2]], _BAYER_CHANNEL[p[3]]]])
    return np.tile(tile, ((yres + 1) // 2, (xres + 1) // 2))[:yres, :xres]


def apply_pixel_angular_response(
    planes: np.ndarray,
    lambdas_nm: np.ndarray,
    weights: np.ndarray,
    cfg: dict,
    *,
    pitch_um: float,
    f_number: float,
    lens_cfg: dict | None = None,
    geometry: dict | None = None,
    repo: Path | None = None,
    maps: AngularResponseMaps | None = None,
) -> tuple[np.ndarray, dict]:
    """HxWxK planes, KxC electron weights → HxWxC electrons with per-pixel, per-wavelength angular
    response ``e_c = Σ_λ L·w_c·R(x, y, λ)``, plus optional crosstalk added to each pixel's own
    CFA channel (``crosstalk.enabled`` with ``crosstalk.cfa_pattern``; other planes at a pixel are
    discarded by CFA sampling)."""
    h, wdt, k = planes.shape
    wts = np.asarray(weights, dtype=np.float64)
    if maps is None:
        maps = compute_maps(
            wdt,
            h,
            lambdas_nm,
            cfg,
            pitch_um=pitch_um,
            f_number=f_number,
            lens_cfg=lens_cfg,
            geometry=geometry,
            repo=repo,
        )
    xt_cfg = dict(cfg.get("crosstalk") or {})
    xt_on = bool(xt_cfg.get("enabled", False))
    c_idx = bayer_channel_index(xt_cfg.get("cfa_pattern", "RGGB"), h, wdt) if xt_on else None
    j0, j1, tx = _axis_weights(maps.grid_cols, np.arange(wdt) + 0.5)
    i0, i1, ty = _axis_weights(maps.grid_rows, np.arange(h) + 0.5)
    out = np.empty((h, wdt, wts.shape[1]), dtype=np.float64)
    xt_src = np.zeros((h, wdt, 3, 3)) if xt_on else None
    block = max(1, int(4_000_000 // max(1, wdt * k * (10 if xt_on else 1))))
    own32 = maps.own.astype(np.float32)
    nb32 = maps.neighbours.astype(np.float32)
    for r0 in range(0, h, block):
        r1 = min(h, r0 + block)
        sl = slice(r0, r1)
        t_y = ty[sl, None, None].astype(np.float32)
        rows_g = (1 - t_y) * own32[i0[sl]] + t_y * own32[i1[sl]]  # [B, Gx, K]
        t_x = tx[None, :, None].astype(np.float32)
        own_px = (1 - t_x) * rows_g[:, j0] + t_x * rows_g[:, j1]  # [B, W, K]
        lb = planes[sl].astype(np.float32)
        out[sl] = np.einsum("bwk,kc->bwc", (lb * own_px).astype(np.float64), wts)
        if xt_on:
            t_y5 = ty[sl, None, None, None, None].astype(np.float32)
            nb_rows = (1 - t_y5) * nb32[i0[sl]] + t_y5 * nb32[i1[sl]]  # [B, Gx, 3, 3, K]
            t_x5 = tx[None, :, None, None, None].astype(np.float32)
            nb_px = (1 - t_x5) * nb_rows[:, j0] + t_x5 * nb_rows[:, j1]  # [B, W, 3, 3, K]
            lw = lb.astype(np.float64) * wts.T[c_idx[sl]]  # light filtered by each pixel's own CFA
            xt_src[sl] = np.einsum("bwk,bwyxk->bwyx", lw, nb_px.astype(np.float64))
    meta = dict(maps.meta)
    if xt_on:
        recv = np.zeros((h, wdt))
        for ky in (-1, 0, 1):
            for kx in (-1, 0, 1):
                if kx == 0 and ky == 0:
                    continue
                # Film offset (kx, ky) is image offset (row -ky, col +kx): P receives from N = P + (ky, -kx).
                src = xt_src[:, :, 1 + ky, 1 + kx]
                shifted = np.zeros_like(src)
                rs, cs = ky, -kx
                shifted[max(0, -rs) : h - max(0, rs), max(0, -cs) : wdt - max(0, cs)] = src[
                    max(0, rs) : h - max(0, -rs), max(0, cs) : wdt - max(0, -cs)
                ]
                recv += shifted
        rr, cc = np.indices((h, wdt))
        out[rr, cc, c_idx] += recv
        meta["crosstalk"] = {"enabled": True, "cfa_pattern": str(xt_cfg.get("cfa_pattern", "RGGB")).upper()}
    meta["channel_response_corner"] = channel_response(maps, wts)[0, 0].tolist()
    meta["channel_response_center"] = _bilinear_point(
        channel_response(maps, wts), maps.grid_rows, maps.grid_cols, 0.5 * h, 0.5 * wdt
    ).tolist()
    return out, meta


def pixel_angular_response_cfg(model: dict) -> dict:
    """``pixel_angular_response`` block from ``sensor_forward.model`` (or its ``calibration``)."""
    cal = model.get("calibration", {}) or {}
    return model.get("pixel_angular_response") or cal.get("pixel_angular_response") or {}
