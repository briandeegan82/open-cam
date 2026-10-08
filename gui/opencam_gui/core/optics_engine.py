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


def chromatic_sigma_px(
    wavelength_nm: float, f_number: float, pixel_pitch_um: float, sigma_geometric_px: float
) -> float:
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


# =====================================================================
# Stray light
# =====================================================================
#: Row through the test image where the high-contrast step edge lives.
EDGE_ROW_FRACTION = 0.75


def stray_light_test_image(size: int = 192, source_intensity: float = 200.0) -> np.ndarray:
    """A high-dynamic-range scene that makes every stray-light term visible.

    One very bright small source (the "sun"), a dark surround, and a
    high-contrast step edge in the lower half. The analytic stand-in for
    ``tools/build_straylight_test_scene.py``, which builds the same situation as
    a PBRT scene: flare only shows up when one part of the frame is orders of
    magnitude brighter than the rest.
    """
    img = np.full((size, size), 0.02, dtype=np.float64)

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    src_y, src_x, src_r = size * 0.28, size * 0.30, size * 0.055
    img[np.hypot(yy - src_y, xx - src_x) <= src_r] = source_intensity

    # A mid-grey block, so the ghost (a 180-degree rotated copy) has something
    # recognisable to land on.
    img[int(size * 0.18) : int(size * 0.34), int(size * 0.62) : int(size * 0.88)] = 0.45

    # High-contrast step edge for the contrast-destruction measurement.
    lo, hi = int(size * 0.60), int(size * 0.92)
    img[lo:hi, : size // 2] = 1.0
    img[lo:hi, size // 2 :] = 0.05
    return img


def stray_light_config(
    *,
    enabled: bool = True,
    veiling_glare_fraction: float = 0.0,
    halo_sigma_pixels: float = 0.0,
    halo_strength: float = 0.0,
    ghost_enabled: bool = False,
    ghost_strength: float = 0.02,
    aperture_diffraction_enabled: bool = False,
    n_blades: int = 6,
    diffraction_strength: float = 0.05,
    rotation_deg: float = 0.0,
    psf_kernel_size: int = 128,
) -> dict:
    """Build the ``lens.post_psf.stray_light`` block exactly as the YAML spells it."""
    return {
        "enabled": bool(enabled),
        "veiling_glare_fraction": float(veiling_glare_fraction),
        "halo_sigma_pixels": float(halo_sigma_pixels),
        "halo_strength": float(halo_strength),
        "ghost_reflections": {
            "enabled": bool(ghost_enabled),
            "ghost_strength": float(ghost_strength),
        },
        "aperture_diffraction": {
            "enabled": bool(aperture_diffraction_enabled),
            "n_blades": int(n_blades),
            "strength": float(diffraction_strength),
            "rotation_deg": float(rotation_deg),
            "psf_kernel_size": int(psf_kernel_size),
        },
    }


def apply_stray_light(img: np.ndarray, cfg: dict) -> np.ndarray:
    """Run the pipeline's real ``apply_stray_light`` over a preview image."""
    return np.asarray(_psf_module().apply_stray_light(np.asarray(img, dtype=np.float32), cfg), dtype=np.float64)


def aperture_diffraction_kernel(n_blades: int, size: int = 128, rotation_deg: float = 0.0) -> np.ndarray:
    """The N-blade iris starburst PSF itself, normalised to a peak of 1 for display."""
    psf = np.asarray(
        _psf_module()._aperture_diffraction_psf(int(n_blades), int(size), float(rotation_deg)),
        dtype=np.float64,
    )
    peak = float(psf.max()) or 1.0
    return psf / peak


def tone_for_display(img: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Log-then-gamma map an HDR preview into 0..1.

    A bright source is ~4 orders of magnitude above the background, so a linear
    stretch would show a white dot on black and hide the flare entirely -- the
    very thing the demo is about.
    """
    a = np.asarray(img, dtype=np.float64)
    a = np.log1p(np.maximum(a, 0.0) / 0.01)
    hi = float(a.max()) or 1.0
    return np.clip(a / hi, 0.0, 1.0) ** (1.0 / gamma)


@dataclass(frozen=True)
class EdgeContrast:
    position_px: np.ndarray
    clean: np.ndarray
    strayed: np.ndarray
    clean_contrast: float
    strayed_contrast: float

    @property
    def contrast_loss_percent(self) -> float:
        if self.clean_contrast <= 0:
            return 0.0
        return 100.0 * (1.0 - self.strayed_contrast / self.clean_contrast)


def _michelson_contrast(profile: np.ndarray) -> float:
    lo, hi = float(np.min(profile)), float(np.max(profile))
    return (hi - lo) / (hi + lo) if (hi + lo) > 0 else 0.0


def edge_contrast(clean: np.ndarray, strayed: np.ndarray) -> EdgeContrast:
    """Slice both images across the step edge and measure Michelson contrast.

    Veiling glare adds a constant to everything, which leaves the difference
    between black and white alone but raises their sum -- so contrast falls even
    though nothing got blurrier. This is why flare is a contrast problem before
    it is a sharpness problem.
    """
    row = int(clean.shape[0] * EDGE_ROW_FRACTION)
    a = np.asarray(clean, dtype=np.float64)[row]
    b = np.asarray(strayed, dtype=np.float64)[row]
    return EdgeContrast(
        position_px=np.arange(a.size, dtype=np.float64),
        clean=a,
        strayed=b,
        clean_contrast=_michelson_contrast(a),
        strayed_contrast=_michelson_contrast(b),
    )
