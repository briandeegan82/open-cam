"""Optics/PSF adapter over ``tools/apply_spectral_psf.py``.

Every number here is produced by importing and calling the real functions
from that module — the same ones used by the actual post-render PSF pipeline
stage. This file adds no new optics physics; it only builds small test images
(a delta image, a radial test chart) so the reused functions have something to
operate on for a live plot, and does simple image bookkeeping (radial
averaging) around the results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opencam_gui.core.repo import import_tool


def _psf_module():
    return import_tool("apply_spectral_psf")


def rgb_center_wavelengths_nm() -> dict[str, float]:
    return dict(_psf_module()._RGB_CENTER_NM)


def chromatic_sigma_px(wavelength_nm: float, f_number: float, pixel_pitch_um: float, sigma_geometric_px: float) -> float:
    m = _psf_module()
    return m.psf_sigma_chromatic(wavelength_nm, f_number, pixel_pitch_um, sigma_geometric_px)


def airy_first_zero_px(wavelength_nm: float, f_number: float, pixel_pitch_um: float) -> float:
    m = _psf_module()
    return m._airy_first_zero_px(wavelength_nm, f_number, pixel_pitch_um)


@dataclass(frozen=True)
class PsfResult:
    kernel: np.ndarray  # normalised (peak = 1), 2-D
    sigma_diff_px: float
    sigma_geom_px: float
    sigma_total_px: float
    rho0_px: float  # Airy first-zero radius (0.0 for gaussian modes)


def compute_psf_kernel(
    *,
    mode: str,
    wavelength_nm: float,
    f_number: float,
    pixel_pitch_um: float,
    sigma_geometric_px: float,
    size: int = 65,
) -> PsfResult:
    """The PSF kernel actually used by the pipeline, via a delta-image convolution.

    Convolving a single-pixel delta image with the real ``airy_disk_convolve`` /
    ``separable_gaussian_blur_2d`` functions returns exactly their kernel
    (mod normalisation and edge clamping), so this is the true post-PSF stage
    output — not a re-derivation of it.
    """
    m = _psf_module()
    size = int(size) | 1  # force odd so there's a well-defined center pixel
    delta = np.zeros((size, size), dtype=np.float64)
    delta[size // 2, size // 2] = 1.0

    sigma_diff = m.psf_sigma_chromatic(wavelength_nm, f_number, pixel_pitch_um, 0.0)
    sigma_total = float(np.sqrt(sigma_diff**2 + sigma_geometric_px**2))
    rho0 = 0.0

    if mode == "airy_disk":
        rho0 = m._airy_first_zero_px(wavelength_nm, f_number, pixel_pitch_um)
        kernel = m.airy_disk_convolve(delta, rho0, sigma_geometric_px)
    else:  # gaussian / chromatic_gaussian
        kernel = m.separable_gaussian_blur_2d(delta, sigma_total)

    kernel = np.asarray(kernel, dtype=np.float64)
    peak = float(np.max(kernel)) or 1.0
    kernel = kernel / peak
    return PsfResult(
        kernel=kernel,
        sigma_diff_px=float(sigma_diff),
        sigma_geom_px=float(sigma_geometric_px),
        sigma_total_px=sigma_total,
        rho0_px=float(rho0),
    )


def radial_profile(kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthally-averaged radial profile (radius_px, value), center = kernel center."""
    h, w = kernel.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_int = r.astype(np.int32)
    max_r = int(r_int.max())
    sums = np.bincount(r_int.ravel(), weights=kernel.ravel(), minlength=max_r + 1)
    counts = np.bincount(r_int.ravel(), minlength=max_r + 1)
    counts = np.maximum(counts, 1)
    profile = sums / counts
    radii = np.arange(max_r + 1, dtype=np.float64)
    return radii, profile


def radial_test_chart(size: int = 256, n_rings: int = 14) -> np.ndarray:
    """A concentric-ring grayscale test chart for the lateral-CA fringing demo."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    cy, cx = (size - 1) / 2.0, (size - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rmax = 0.5 * size
    img = 0.5 + 0.5 * np.cos(2.0 * np.pi * n_rings * (r / rmax))
    img = np.clip(img, 0.0, 1.0)
    # Radial spokes so the fringe direction (radial vs tangential) is visible too.
    theta = np.arctan2(yy - cy, xx - cx)
    spokes = 0.5 + 0.5 * np.cos(24.0 * theta)
    img = np.where(r < rmax, 0.7 * img + 0.3 * spokes, 0.0)
    return img.astype(np.float64)


def lateral_ca_rgb_preview(
    gray: np.ndarray,
    lca_coefficient: float,
    lambda_reference_nm: float,
    center_wavelengths_nm: dict[str, float] | None = None,
) -> np.ndarray:
    """Apply the real per-channel lateral-CA magnification and stack into an RGB preview."""
    m = _psf_module()
    centers = center_wavelengths_nm or rgb_center_wavelengths_nm()
    channels = []
    for key in ("R", "G", "B"):
        shifted = m.apply_lateral_ca(gray, centers[key], lca_coefficient, lambda_reference_nm)
        channels.append(np.asarray(shifted, dtype=np.float64))
    rgb = np.stack(channels, axis=-1)
    return np.clip(rgb, 0.0, 1.0)
