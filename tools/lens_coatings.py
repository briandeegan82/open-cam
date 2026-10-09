"""Spectral reflectance of coated and uncoated optical interfaces (thin-film characteristic matrix).

An interface between media ``n_inc`` and ``n_sub`` may carry a stack of homogeneous, non-absorbing
layers.  Reflectance and transmittance follow the characteristic-matrix method for stratified
media (Born & Wolf, *Principles of Optics*, 7th ed., Cambridge University Press 1999, sec. 1.6;
H. A. Macleod, *Thin-Film Optical Filters*, 4th ed., CRC Press 2010, ch. 2): for each layer of
index ``n_j`` and physical thickness ``d_j`` at angle ``theta_j`` (Snell invariant
``beta = n sin(theta)``),

    delta_j = 2 pi n_j d_j cos(theta_j) / lambda
    M_j     = [[cos delta_j, i sin delta_j / eta_j], [i eta_j sin delta_j, cos delta_j]]
    [B, C]^T = (prod_j M_j) [1, eta_sub]^T
    r = (eta_0 B - C) / (eta_0 B + C),  R = |r|^2,  T = 4 eta_0 Re(eta_sub) / |eta_0 B + C|^2

with tilted admittances ``eta = n cos(theta)`` (s) and ``eta = n / cos(theta)`` (p).  Light is
treated as unpolarised: ``R = (R_s + R_p) / 2``.  With no layers this reduces to the Fresnel
equations; beyond the critical angle ``cos(theta_sub)`` is imaginary and ``R = 1``.

Presets (all layer indices are non-dispersive; ``design_nm`` is the quarter-wave wavelength):

``uncoated``  bare Fresnel interface.
``mgf2``      single quarter-wave MgF2 layer, n = 1.38 (MgF2 ordinary index ~1.378 near 589 nm,
              M. J. Dodge, Appl. Opt. 23, 1980-1985 (1984)).  At ``design_nm`` and normal incidence
              R = ((n0 ns - n1^2) / (n0 ns + n1^2))^2.
``qhq``       quarter-half-quarter three-layer broadband AR (L. B. Lockhart and P. King,
              "Three-layered reflection reducing coatings", JOSA 37, 689 (1947); Macleod 2010, ch. 4):
              outer quarter-wave n = 1.38, half-wave n = 2.10 (representative high-index oxide), inner
              quarter-wave n = 1.38 sqrt(ns / n0).  The half-wave layer is absentee at ``design_nm``,
              so the inner index is the two-layer quarter-quarter zero-reflectance condition; the
              inner index is idealised per substrate (no real material catalogue).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

MGF2_N = 1.38
QHQ_HIGH_N = 2.10
DEFAULT_DESIGN_NM = 550.0


@dataclass(frozen=True)
class Coating:
    """Layer stack listed from the air (or lower-index ambient) side toward the glass."""

    name: str
    layers: tuple[tuple[float, float], ...] = ()  # (index, physical thickness nm)


def coating_for_interface(spec, n_air_side: float, n_glass_side: float, design_nm: float = DEFAULT_DESIGN_NM):
    """Resolve a coating spec (preset name or ``{"layers": [...]}``) for a given substrate."""
    if isinstance(spec, Coating):
        return spec
    if spec is None or spec == "uncoated":
        return Coating("uncoated")
    if isinstance(spec, dict):
        layers = []
        for layer in spec.get("layers", []):
            n = float(layer["n"])
            if "thickness_nm" in layer:
                d = float(layer["thickness_nm"])
            else:
                d = float(layer.get("qwot", 1.0)) * float(spec.get("design_nm", design_nm)) / (4.0 * n)
            layers.append((n, d))
        return Coating(str(spec.get("name", "custom")), tuple(layers))
    name = str(spec).lower()
    q = design_nm / 4.0
    if name == "mgf2":
        return Coating("mgf2", ((MGF2_N, q / MGF2_N),))
    if name in ("qhq", "multilayer"):
        n_inner = MGF2_N * np.sqrt(n_glass_side / n_air_side)
        return Coating("qhq", ((MGF2_N, q / MGF2_N), (QHQ_HIGH_N, 2.0 * q / QHQ_HIGH_N), (n_inner, q / n_inner)))
    raise ValueError(f"unknown coating {spec!r} (expected uncoated, mgf2, qhq or a layers dict)")


def _cos_theta(n: float, beta: np.ndarray) -> np.ndarray:
    c = np.sqrt((1.0 - (beta / n) ** 2).astype(np.complex128))
    # Evanescent branch: Im(cos) >= 0 so the field decays in +z (Born & Wolf sec. 1.5.4).
    return np.where(c.imag < 0, -c, c)


def stack_rt(lam_nm, beta, n_inc: float, n_sub: float, layers=()) -> tuple[np.ndarray, np.ndarray]:
    """Unpolarised (R, T) for incidence from ``n_inc`` through ``layers`` (incident-side first) into ``n_sub``.

    ``lam_nm`` and ``beta = n_inc sin(theta_inc)`` broadcast against each other.
    """
    lam = np.asarray(lam_nm, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    lam, beta = np.broadcast_arrays(lam, beta)
    c0 = _cos_theta(n_inc, beta)
    cs = _cos_theta(n_sub, beta)
    r_out, t_out = np.zeros(lam.shape), np.zeros(lam.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        for pol in ("s", "p"):
            _accumulate_pol(pol, lam, beta, n_inc, n_sub, layers, c0, cs, r_out, t_out)
    # Exactly grazing incidence (cos = 0) gives 0/0; its limit is R = 1, T = 0.
    bad = ~np.isfinite(r_out) | ~np.isfinite(t_out)
    r_out[bad], t_out[bad] = 1.0, 0.0
    return r_out, t_out


def _accumulate_pol(pol, lam, beta, n_inc, n_sub, layers, c0, cs, r_out, t_out) -> None:
    """Add one polarisation's (R, T) / 2 into ``r_out`` / ``t_out``."""

    def eta(n, c):
        return n * c if pol == "s" else n / c

    e0, es = eta(n_inc, c0), eta(n_sub, cs)
    m11 = np.ones(lam.shape, np.complex128)
    m12 = np.zeros(lam.shape, np.complex128)
    m21 = np.zeros(lam.shape, np.complex128)
    m22 = np.ones(lam.shape, np.complex128)
    for n_j, d_j in layers:
        cj = _cos_theta(n_j, beta)
        ej = eta(n_j, cj)
        delta = 2.0 * np.pi * n_j * d_j * cj / lam
        a, b = np.cos(delta), 1j * np.sin(delta)
        m11, m12, m21, m22 = (
            m11 * a + m12 * b * ej,
            m11 * b / ej + m12 * a,
            m21 * a + m22 * b * ej,
            m21 * b / ej + m22 * a,
        )
    bb = m11 + m12 * es
    cc = m21 + m22 * es
    den = e0 * bb + cc
    r = (e0 * bb - cc) / den
    r_out += 0.5 * np.abs(r) ** 2
    t_out += 0.5 * 4.0 * e0.real * es.real / np.abs(den) ** 2


def interface_rt(lam_nm, beta, n1: float, n2: float, coating: Coating) -> tuple[np.ndarray, np.ndarray]:
    """(R, T) for light in ``n1`` hitting the interface to ``n2``; the coating sits on the lower-index side."""
    layers = coating.layers if n1 <= n2 else tuple(reversed(coating.layers))
    return stack_rt(lam_nm, beta, n1, n2, layers)


@lru_cache(maxsize=512)
def reflectance_table(n1: float, n2: float, coating: Coating, lams: tuple[float, ...], n_beta: int = 2049):
    """Tabulated unpolarised R(beta, lambda) on ``beta in [0, n1]`` for fast per-ray lookup."""
    beta = np.linspace(0.0, n1, n_beta)
    r, _t = interface_rt(np.asarray(lams)[None, :], beta[:, None], n1, n2, coating)
    return beta, np.clip(r, 0.0, 1.0)


def lookup_reflectance(table, beta: np.ndarray) -> np.ndarray:
    """Linear interpolation of a :func:`reflectance_table` at per-ray ``beta`` -> (N, n_lambda)."""
    grid, r = table
    x = np.clip(beta, 0.0, grid[-1]) / grid[-1] * (grid.size - 1)
    i = np.minimum(x.astype(np.int64), grid.size - 2)
    f = (x - i)[:, None]
    return r[i] * (1.0 - f) + r[i + 1] * f
