"""Resolution/MTF adapter over ``tools/sfr_analysis.py`` and ``tools/apply_spectral_psf.py``.

The demo has two ways to get an edge to measure: synthesise one and blur it
with the same PSF functions the pipeline uses (instant, and the ground truth is
known), or load a slanted-edge target actually rendered into ``out/``. Both end
up in the same ISO 12233 chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from opencam_gui.core.repo import import_tool, repo_root


def _sfr():
    return import_tool("sfr_analysis")


NYQUIST_CY_PER_PX = 0.5
DEFAULT_SPOKES = 72


# =====================================================================
# Sources
# =====================================================================
def synthetic_edge(
    *,
    size: int = 192,
    angle_deg: float = 5.0,
    mode: str = "chromatic_gaussian",
    f_number: float = 5.6,
    pixel_pitch_um: float = 4.3,
    sigma_geometric_px: float = 0.3,
    wavelength_nm: float = 550.0,
) -> np.ndarray:
    """An ideal slanted edge blurred by the pipeline's own PSF stage.

    Because the blur comes from ``apply_spectral_psf``'s functions rather than a
    re-implementation, the MTF measured here is the MTF of the real optics
    model -- which is what makes comparing it against theory meaningful.
    """
    m = import_tool("apply_spectral_psf")

    # Blur an oversized edge and crop back, so the returned ROI holds only valid
    # convolution output. Keeping the zero-padded border would put a second,
    # artificial edge in the ROI and corrupt the measurement.
    margin = max(16, size // 4)
    big = _sfr().synthetic_slanted_edge(size + 2 * margin, size + 2 * margin, angle_deg)

    if mode == "airy_disk":
        rho0 = m._airy_first_zero_px(wavelength_nm, f_number, pixel_pitch_um)
        blurred = m.airy_disk_convolve(big.astype(np.float32), rho0, sigma_geometric_px)
    else:
        sigma = m.psf_sigma_chromatic(wavelength_nm, f_number, pixel_pitch_um, sigma_geometric_px)
        blurred = m.separable_gaussian_blur_2d(big.astype(np.float32), sigma)

    cropped = np.asarray(blurred, dtype=np.float64)[margin : margin + size, margin : margin + size]
    return np.ascontiguousarray(cropped)


def rendered_edge_candidates() -> list[Path]:
    """Slanted-edge renders sitting in ``out/``, newest first."""
    out = repo_root() / "out"
    if not out.is_dir():
        return []
    found = [p for p in out.rglob("*.png") if "slanted" in p.name.lower() or "edge" in p.name.lower()]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def load_rendered_edge(path: Path, roi_size: int = 192) -> np.ndarray:
    """Load a rendered edge and crop a centred ROI around the strongest edge."""
    from PIL import Image

    img = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0
    h, w = img.shape
    half = min(roi_size, h, w) // 2
    # Centre the ROI on the column with the strongest mean gradient.
    col_energy = np.abs(np.diff(img, axis=1)).mean(axis=0)
    cx = int(np.argmax(col_energy)) if col_energy.size else w // 2
    cy = h // 2
    cx = int(np.clip(cx, half, w - half))
    cy = int(np.clip(cy, half, h - half))
    return img[cy - half : cy + half, cx - half : cx + half]


# =====================================================================
# Measurement
# =====================================================================
@dataclass(frozen=True)
class MtfMeasurement:
    roi: np.ndarray
    angle_deg: float
    esf_position_px: np.ndarray
    esf: np.ndarray
    lsf_position_px: np.ndarray
    lsf: np.ndarray
    frequency_cy_per_px: np.ndarray
    mtf: np.ndarray
    mtf50_cy_per_px: float
    mtf10_cy_per_px: float
    mtf_at_nyquist: float
    mtf50_cy_per_mm: float
    pixel_pitch_um: float


def measure(roi: np.ndarray, pixel_pitch_um: float) -> MtfMeasurement:
    s = _sfr()
    r = s.slanted_edge_sfr(roi)
    lsf_norm = r.lsf / (np.max(np.abs(r.lsf)) or 1.0)
    return MtfMeasurement(
        roi=np.asarray(roi, dtype=np.float64),
        angle_deg=r.angle_deg,
        esf_position_px=r.esf_position_px,
        esf=r.esf,
        lsf_position_px=r.esf_position_px,
        lsf=lsf_norm,
        frequency_cy_per_px=r.frequency_cy_per_px,
        mtf=r.mtf,
        mtf50_cy_per_px=r.mtf50_cy_per_px,
        mtf10_cy_per_px=r.mtf10_cy_per_px,
        mtf_at_nyquist=r.mtf_at_nyquist,
        mtf50_cy_per_mm=float(s.cycles_per_mm(r.mtf50_cy_per_px, pixel_pitch_um)),
        pixel_pitch_um=pixel_pitch_um,
    )


# =====================================================================
# Theory overlays
# =====================================================================
@dataclass(frozen=True)
class TheoryCurves:
    frequency_cy_per_px: np.ndarray
    diffraction: np.ndarray
    pixel_aperture: np.ndarray
    system: np.ndarray
    diffraction_cutoff_cy_per_px: float


def theory_curves(
    *,
    f_number: float,
    pixel_pitch_um: float,
    wavelength_nm: float = 550.0,
    fill_factor: float = 1.0,
    max_frequency: float = 1.0,
    n: int = 400,
) -> TheoryCurves:
    """Diffraction-limited and pixel-aperture MTFs, and their cascade."""
    s = _sfr()
    freq = np.linspace(0.0, max_frequency, int(n))
    diffraction = s.diffraction_mtf(freq, f_number, wavelength_nm, pixel_pitch_um)
    pixel = s.pixel_aperture_mtf(freq, fill_factor)
    return TheoryCurves(
        frequency_cy_per_px=freq,
        diffraction=diffraction,
        pixel_aperture=pixel,
        system=s.system_mtf(diffraction, pixel),
        diffraction_cutoff_cy_per_px=float(s.diffraction_cutoff_cy_per_px(f_number, wavelength_nm, pixel_pitch_um)),
    )


# =====================================================================
# Aliasing
# =====================================================================
@dataclass(frozen=True)
class AliasingPreview:
    reference: np.ndarray
    sampled: np.ndarray
    nyquist_radius_px: float
    spokes: int
    downsample: int


def aliasing_preview(
    *, size: int = 256, spokes: int = DEFAULT_SPOKES, downsample: int = 4, prefilter_sigma_px: float = 0.0
) -> AliasingPreview:
    """Sample a Siemens star coarsely to make aliasing visible.

    Inside the Nyquist radius the spokes are finer than the sampling grid can
    represent, so they fold back as moire running the wrong way. Raising the
    prefilter sigma is the optical anti-aliasing filter: it destroys the detail
    *before* sampling, which looks like blur but is honest, whereas aliasing
    invents structure that was never in the scene.
    """
    s = _sfr()
    star = s.siemens_star(size, spokes)
    if prefilter_sigma_px > 0:
        m = import_tool("apply_spectral_psf")
        star = np.asarray(m.separable_gaussian_blur_2d(star.astype(np.float32), prefilter_sigma_px), dtype=np.float64)
    # Point-sample on a coarse grid: no area averaging, so nothing suppresses
    # the frequencies above the new Nyquist limit.
    sampled = star[::downsample, ::downsample]
    return AliasingPreview(
        reference=s.siemens_star(size, spokes),
        sampled=sampled,
        nyquist_radius_px=float(s.star_nyquist_radius_px(spokes)) * downsample,
        spokes=spokes,
        downsample=downsample,
    )


def upsample_nearest(arr: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(np.asarray(arr), int(factor), axis=0), int(factor), axis=1)
