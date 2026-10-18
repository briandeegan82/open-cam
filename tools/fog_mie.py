"""Mie scattering of fog droplets: spectral extinction and a tabulated phase function.

Fog droplets (radius 1-30 um) are comparable to or larger than visible and NIR wavelengths, so
their scattering is Mie, not Rayleigh or a smooth Henyey-Greenstein lobe: a narrow diffraction
peak a few degrees wide carries about half the scattered light, with a glory and a weak rainbow
in the back hemisphere. This module computes, for a droplet-size distribution:

* ``sigma_ext(lambda)``, ``sigma_s(lambda)`` per unit number density, from Lorenz-Mie efficiencies
  (Bohren & Huffman 1983, Absorption and Scattering of Light by Small Particles, app. A, BHMIE);
* the size-averaged phase function, tabulated over cos(theta), for pbrt patch 0013
  (``"float phasecos"`` / ``"float phasevalues"``).

Droplet-size distributions are modified gamma, ``n(r) ~ r^alpha exp(-b r^gamma)`` (r in um;
Deirmendjian 1969), with the fog models of Shettle & Fenn (1979, AFGL-TR-79-0214):
heavy advection fog (alpha 3, b 0.3, gamma 1, modal radius 10 um) and moderate radiation fog
(alpha 6, b 1.5, gamma 1, modal radius 4 um).

Water refractive index: real part and absorption index tabulated from Hale & Querry 1973
(Appl. Opt. 12, 555), rounded; absorption is negligible for droplets below ~1 um wavelength.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DropletDistribution:
    name: str
    alpha: float
    b: float
    gamma: float
    description: str = ""

    @property
    def modal_radius_um(self) -> float:
        return (self.alpha / (self.b * self.gamma)) ** (1.0 / self.gamma)


DROPLETS = {
    "advection": DropletDistribution("advection", 3.0, 0.3, 1.0, "heavy advection fog, modal radius 10 um"),
    "radiation": DropletDistribution("radiation", 6.0, 1.5, 1.0, "moderate radiation fog, modal radius 4 um"),
}

# Hale & Querry (1973), liquid water at 25 C (rounded).
_WATER_WL_NM = np.array([350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1100])
_WATER_N = np.array(
    [1.349, 1.339, 1.337, 1.335, 1.333, 1.332, 1.331, 1.331, 1.330, 1.329, 1.329, 1.328, 1.327, 1.327, 1.326]
)
_WATER_K = np.array(
    [
        3.5e-9,
        1.9e-9,
        1.0e-9,
        1.0e-9,
        2.0e-9,
        1.1e-8,
        1.6e-8,
        3.4e-8,
        1.6e-7,
        1.3e-7,
        2.9e-7,
        4.9e-7,
        2.9e-6,
        2.9e-6,
        1.9e-6,
    ]
)


def water_index(wl_nm: float) -> complex:
    n = float(np.interp(wl_nm, _WATER_WL_NM, _WATER_N))
    k = float(np.exp(np.interp(wl_nm, _WATER_WL_NM, np.log(_WATER_K))))
    return complex(n, k)


def _nstop(x: np.ndarray) -> np.ndarray:
    return np.ceil(x + 4.05 * np.cbrt(x) + 2.0).astype(int)


def mie_coefficients(m: complex, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lorenz-Mie a_n, b_n (shape len(x) x nmax, zero past each n_stop) for refractive index m."""
    x = np.asarray(x, dtype=np.float64)
    nst = _nstop(x)
    nmax = int(nst.max())
    mx = m * x
    nmx = int(max(nmax, np.abs(mx).max())) + 16
    d = np.zeros((x.size, nmx + 1), dtype=np.complex128)  # logarithmic derivative D_n(mx), downward
    for n in range(nmx, 0, -1):
        d[:, n - 1] = n / mx - 1.0 / (d[:, n] + n / mx)
    a = np.zeros((x.size, nmax), dtype=np.complex128)
    b = np.zeros_like(a)
    psi0, psi1 = np.cos(x), np.sin(x)
    chi0, chi1 = -np.sin(x), np.cos(x)
    xi1 = psi1 - 1j * chi1
    err = np.seterr(all="ignore")  # radii past their n_stop overflow; their terms are masked to 0
    for n in range(1, nmax + 1):
        psi = (2 * n - 1) / x * psi1 - psi0
        chi = (2 * n - 1) / x * chi1 - chi0
        xi = psi - 1j * chi
        dn = d[:, n]
        an = ((dn / m + n / x) * psi - psi1) / ((dn / m + n / x) * xi - xi1)
        bn = ((m * dn + n / x) * psi - psi1) / ((m * dn + n / x) * xi - xi1)
        live = n <= nst
        a[:, n - 1] = np.where(live, an, 0.0)
        b[:, n - 1] = np.where(live, bn, 0.0)
        psi0, psi1, chi0, chi1, xi1 = psi1, psi, chi1, chi, xi
    np.seterr(**err)
    return a, b, nst


def efficiencies(m: complex, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(Q_ext, Q_sca) per size parameter."""
    a, b, _ = mie_coefficients(m, x)
    n = np.arange(1, a.shape[1] + 1)
    x = np.asarray(x, dtype=np.float64)
    qext = 2.0 / x**2 * ((2 * n + 1) * (a.real + b.real)).sum(axis=1)
    qsca = 2.0 / x**2 * ((2 * n + 1) * (np.abs(a) ** 2 + np.abs(b) ** 2)).sum(axis=1)
    return qext, qsca


def _angular(mu: np.ndarray, nmax: int) -> tuple[np.ndarray, np.ndarray]:
    pi = np.zeros((nmax, mu.size))
    tau = np.zeros_like(pi)
    p0, p1 = np.zeros_like(mu), np.ones_like(mu)
    for n in range(1, nmax + 1):
        pi[n - 1] = p1
        tau[n - 1] = n * mu * p1 - (n + 1) * p0
        p0, p1 = p1, ((2 * n + 1) * mu * p1 - (n + 1) * p0) / n
    return pi, tau


def radius_grid(dist: DropletDistribution, n: int = 480) -> tuple[np.ndarray, np.ndarray]:
    """Log-spaced radii [um] and number weights (trapezoid in r) that capture >99.99 % of the area."""
    r = np.geomspace(0.05, 12.0 * dist.modal_radius_um + 10.0, n)
    nr = r**dist.alpha * np.exp(-dist.b * r**dist.gamma)
    w = nr * np.gradient(r)
    return r, w / w.sum()


@functools.lru_cache(maxsize=16)
def spectral_cross_sections(dist_name: str, wl_key: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Mean extinction and scattering cross-section per droplet [um^2] at each wavelength."""
    dist = DROPLETS[dist_name]
    r, w = radius_grid(dist)
    cext, csca = [], []
    for wl in wl_key:
        qe, qs = efficiencies(water_index(wl), 2.0 * np.pi * r / (wl * 1e-3))
        area = np.pi * r**2
        cext.append(float((w * area * qe).sum()))
        csca.append(float((w * area * qs).sum()))
    return np.array(cext), np.array(csca)


def phase_mu_grid(n_fwd: int = 400, n_rest: int = 300) -> np.ndarray:
    """Bin edges in cos(theta), ascending from -1 to 1, dense near the forward peak (theta < 20 deg)."""
    th_f = np.geomspace(0.01, 20.0, n_fwd)
    th_r = np.linspace(20.0, 180.0, n_rest + 1)[1:]
    th = np.concatenate([[0.0], th_f, th_r])
    return np.cos(np.radians(th))[::-1].copy()


@functools.lru_cache(maxsize=8)
def phase_table(
    dist_name: str, wl_key: tuple[float, ...] = (450.0, 500.0, 550.0, 600.0, 650.0)
) -> tuple[np.ndarray, np.ndarray]:
    """(cos edges, bin values) of the size- and wavelength-averaged Mie phase function.

    Values are bin means of p(cos theta) [1/sr], normalised so that sum(p * dcos) * 2 pi = 1.
    Scattering cosine convention: +1 is forward scattering. Wavelengths are weighted by their
    scattering cross-section.
    """
    dist = DROPLETS[dist_name]
    r, w = radius_grid(dist)
    edges = phase_mu_grid()
    # Sample each bin at 4 interior points (Gauss-Legendre in cos) for the bin mean.
    gx, gw = np.polynomial.legendre.leggauss(4)
    lo, hi = edges[:-1], edges[1:]
    mu = (0.5 * (hi - lo)[:, None] * gx[None, :] + 0.5 * (hi + lo)[:, None]).ravel()
    acc = np.zeros(mu.size)
    total_w = 0.0
    for wl in wl_key:
        x = 2.0 * np.pi * r / (wl * 1e-3)
        a, b, _ = mie_coefficients(water_index(wl), x)
        nmax = a.shape[1]
        pi, tau = _angular(mu, nmax)
        n = np.arange(1, nmax + 1)
        f = (2 * n + 1) / (n * (n + 1))
        s1 = (a * f) @ pi + (b * f) @ tau
        s2 = (a * f) @ tau + (b * f) @ pi
        # |S1|^2 + |S2|^2 summed over droplets: proportional to the scattered intensity.
        inten = (w[:, None] * (np.abs(s1) ** 2 + np.abs(s2) ** 2)).sum(axis=0) * (wl * 1e-3) ** 2
        acc += inten
        total_w += 1.0
    vals = (acc.reshape(lo.size, 4) * gw[None, :]).sum(axis=1) / 2.0  # bin mean
    vals /= 2.0 * np.pi * (vals * (hi - lo)).sum()
    return edges, vals


def pbrt_table(dist_name: str) -> tuple[np.ndarray, np.ndarray]:
    """(scattering-angle edges [deg] ascending 0..180, values) for ``"float phasetheta"/"phasevalues"``."""
    edges, vals = phase_table(dist_name)
    th = np.degrees(np.arccos(np.clip(edges[::-1], -1.0, 1.0)))
    th[0], th[-1] = 0.0, 180.0
    return th, vals[::-1].copy()


@functools.lru_cache(maxsize=8)
def _spectral_shape(dist_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wl = tuple(float(w) for w in np.arange(350.0, 1101.0, 25.0))
    ce, cs = spectral_cross_sections(dist_name, wl)
    i550 = wl.index(550.0)
    return np.array(wl), ce / ce[i550], cs / ce


def extinction_shape(dist_name: str, wl_nm) -> tuple[np.ndarray, np.ndarray]:
    """(sigma_ext(lambda) / sigma_ext(550 nm), single-scattering albedo) of the droplet population."""
    wl, rel, alb = _spectral_shape(dist_name)
    x = np.asarray(wl_nm, dtype=np.float64)
    return np.interp(x, wl, rel), np.interp(x, wl, alb)


def asymmetry(edges: np.ndarray, vals: np.ndarray) -> float:
    mid = 0.5 * (edges[1:] + edges[:-1])
    return float(2.0 * np.pi * (vals * mid * np.diff(edges)).sum())


def sample_cos(edges: np.ndarray, vals: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Scattering cosines drawn from the tabulated phase function (piecewise constant in cos)."""
    cdf = np.concatenate([[0.0], np.cumsum(vals * np.diff(edges))])
    cdf /= cdf[-1]
    i = np.clip(np.searchsorted(cdf, u, side="right") - 1, 0, vals.size - 1)
    t = (u - cdf[i]) / np.maximum(cdf[i + 1] - cdf[i], 1e-300)
    return edges[i] + t * (edges[i + 1] - edges[i])
