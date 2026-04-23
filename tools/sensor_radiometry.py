"""Shared radiometry helpers for spectral sensor forward models."""

from __future__ import annotations

import numpy as np

# SI constants
H_PLANCK = 6.62607015e-34  # J·s
C_LIGHT = 299792458.0  # m/s


def photon_flux_density_from_irradiance(
    spectral_irradiance_W_m2nm: np.ndarray,
    wavelength_nm: np.ndarray,
) -> np.ndarray:
    """Convert spectral irradiance E_e(λ) [W/(m²·nm)] to photon flux density [photons/(s·m²·nm)].

    Φ_p(λ) = E_e(λ) · λ / (h c) with λ in meters.
    """
    lam_m = np.asarray(wavelength_nm, dtype=np.float64) * 1e-9
    return spectral_irradiance_W_m2nm * lam_m / (H_PLANCK * C_LIGHT)


def cosine_illuminance_factor(
    surface_normal: np.ndarray,
    direction_to_light_world: np.ndarray,
) -> float:
    """Lambert receiver: factor = max(0, n·omega) with omega unit vector toward the source."""
    n = np.asarray(surface_normal, dtype=np.float64)
    w = np.asarray(direction_to_light_world, dtype=np.float64)
    n = n / max(1e-15, np.linalg.norm(n))
    w = w / max(1e-15, np.linalg.norm(w))
    return float(max(0.0, np.dot(n, w)))


def cos4_vignetting_from_pinhole(
    xw: np.ndarray,
    yw: np.ndarray,
    cam_dist: float,
) -> np.ndarray:
    """Cos^4 vignetting vs chief ray angle for pinhole at (0,0,cam_dist) viewing z=0 plane."""
    r = np.sqrt(xw * xw + yw * yw + cam_dist * cam_dist)
    cos_t = np.abs(cam_dist) / np.maximum(1e-9, r)
    return cos_t**4
