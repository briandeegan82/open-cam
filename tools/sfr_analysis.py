"""ISO 12233 slanted-edge SFR: turning a picture of a blur into a number.

``build_image_quality_targets.py`` can already render a slanted edge and a
Siemens star, but nothing measured anything from them. This module closes that
loop: given a region of interest containing a high-contrast slanted edge, it
recovers the edge spread function, differentiates it to the line spread
function, and transforms that to the modulation transfer function.

Why the edge has to be slanted
------------------------------
A perfectly vertical edge is sampled by every row at exactly the same pixel
phase, so it tells you about the system only at that one phase. Tilting it a
few degrees makes each row cross the edge at a slightly different sub-pixel
offset, and pooling the rows reconstructs the edge profile on a much finer grid
than the pixel pitch -- 4x finer by convention. That is what lets a sensor
measure detail beyond its own Nyquist limit.

Units
-----
Frequencies are in cycles per pixel unless a name says otherwise, so 0.5 is
always Nyquist. Use :func:`cycles_per_mm` to convert for a given pixel pitch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: ISO 12233 reconstructs the edge profile on a grid 4x finer than the pixels.
DEFAULT_OVERSAMPLING = 4

#: Nyquist frequency of the pixel grid, in cycles per pixel.
NYQUIST_CY_PER_PX = 0.5


# =====================================================================
# Synthetic targets
# =====================================================================
def synthetic_slanted_edge(
    height: int = 128,
    width: int = 128,
    angle_deg: float = 5.0,
    dark: float = 0.05,
    bright: float = 0.95,
    supersample: int = 8,
) -> np.ndarray:
    """An ideal slanted edge, area-sampled so each pixel gets true partial coverage.

    Supersampling matters: a hard-thresholded edge would quantise the sub-pixel
    crossing to whole pixels and destroy the very phase diversity the slanted
    edge method depends on.
    """
    s = int(supersample)
    yy, xx = np.mgrid[0 : height * s, 0 : width * s].astype(np.float64)
    # Sub-pixel centre coordinates in original pixel units.
    x = (xx + 0.5) / s
    y = (yy + 0.5) / s

    slope = math.tan(math.radians(angle_deg))
    edge_x = width / 2.0 + (y - height / 2.0) * slope
    fine = np.where(x < edge_x, bright, dark)
    return fine.reshape(height, s, width, s).mean(axis=(1, 3))


def siemens_star(size: int = 256, spokes: int = 72, radius_fraction: float = 0.9) -> np.ndarray:
    """A radial spoke target, matching the defaults of ``build_image_quality_targets.py``.

    Spatial frequency rises as you move inward, so a single image shows the
    system passing, then failing, then aliasing -- the spokes reverse contrast
    and swirl into moire once they pass Nyquist.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    c = (size - 1) / 2.0
    dy, dx = yy - c, xx - c
    r = np.hypot(dy, dx)
    theta = np.arctan2(dy, dx)
    pattern = 0.5 + 0.5 * np.sign(np.cos(spokes * theta / 2.0))
    return np.where(r <= radius_fraction * c, pattern, 0.5)


def star_frequency_cy_per_px(radius_px: np.ndarray | float, spokes: int) -> np.ndarray:
    """Local spatial frequency of a Siemens star at a given radius.

    One spoke pair spans ``2 pi r / (spokes/2)`` pixels of arc, so the frequency
    is ``spokes / (4 pi r)`` cycles per pixel. Setting this to 0.5 gives the
    radius at which the star crosses Nyquist and starts to alias.
    """
    r = np.maximum(np.asarray(radius_px, dtype=np.float64), 1e-9)
    return spokes / (4.0 * np.pi * r)


def star_nyquist_radius_px(spokes: int) -> float:
    """Radius inside which a Siemens star is past Nyquist and will alias."""
    return spokes / (4.0 * np.pi * NYQUIST_CY_PER_PX)


# =====================================================================
# Edge detection
# =====================================================================
def row_edge_positions(roi: np.ndarray, window_px: float | None = None) -> np.ndarray:
    """Sub-pixel horizontal edge location for every row, by derivative centroid.

    The centroid is taken only within a window around the dominant edge rather
    than across the whole row. A plain whole-row centroid is pulled off target by
    anything else with a gradient -- most easily by the border discontinuity a
    zero-padded convolution leaves behind, which can shift the estimate by tens
    of pixels and silently build the ESF around the wrong origin.
    """
    a = np.asarray(roi, dtype=np.float64)
    deriv = np.abs(np.diff(a, axis=1))
    # Derivative samples sit between pixels, i.e. at x = index + 0.5.
    x = np.arange(deriv.shape[1], dtype=np.float64) + 0.5

    if window_px is None:
        window_px = max(8.0, deriv.shape[1] / 6.0)

    # Locate the edge from the column-summed derivative, ignoring the outer
    # margin. A border step is a perfectly good edge as far as a gradient is
    # concerned, so the only reliable way to not lock onto one is to refuse to
    # look there -- ISO 12233 ROIs are cropped away from the frame edge anyway.
    profile = deriv.sum(axis=0)
    margin = min(int(round(0.1 * deriv.shape[1])), (deriv.shape[1] - 1) // 2)
    interior = profile[margin : deriv.shape[1] - margin]
    centre = float(x[margin + int(np.argmax(interior))])
    mask = np.abs(x - centre) <= window_px

    weights = deriv * mask
    total = weights.sum(axis=1)
    total[total == 0] = np.nan
    return (weights * x).sum(axis=1) / total


def find_edge_angle(roi: np.ndarray) -> float:
    """Edge tilt away from vertical, in degrees (positive leans right going down)."""
    positions = row_edge_positions(roi)
    rows = np.arange(positions.size, dtype=np.float64)
    valid = np.isfinite(positions)
    if valid.sum() < 2:
        raise ValueError("could not locate an edge: is there any contrast in the ROI?")
    slope = np.polyfit(rows[valid], positions[valid], 1)[0]
    return math.degrees(math.atan(slope))


# =====================================================================
# ESF / LSF / MTF
# =====================================================================
@dataclass(frozen=True)
class SfrResult:
    angle_deg: float
    esf_position_px: np.ndarray
    esf: np.ndarray
    lsf: np.ndarray
    frequency_cy_per_px: np.ndarray
    mtf: np.ndarray
    mtf50_cy_per_px: float
    mtf10_cy_per_px: float
    mtf_at_nyquist: float
    bin_width_px: float

    def mtf50_cy_per_mm(self, pixel_pitch_um: float) -> float:
        return cycles_per_mm(self.mtf50_cy_per_px, pixel_pitch_um)

    def mtf50_lp_per_mm(self, pixel_pitch_um: float) -> float:
        """Line pairs per mm is the same number as cycles per mm."""
        return self.mtf50_cy_per_mm(pixel_pitch_um)


def cycles_per_mm(cycles_per_px: float | np.ndarray, pixel_pitch_um: float) -> float | np.ndarray:
    """Convert cycles/pixel to cycles/mm for a given pixel pitch."""
    return np.asarray(cycles_per_px) * 1000.0 / pixel_pitch_um


def edge_spread_function(
    roi: np.ndarray, angle_deg: float | None = None, oversampling: int = DEFAULT_OVERSAMPLING
) -> tuple[np.ndarray, np.ndarray, float]:
    """Project every pixel onto the edge normal and bin at sub-pixel resolution.

    Returns ``(position_px, esf, bin_width_px)`` with position measured
    perpendicular to the edge and zero at the edge itself.
    """
    a = np.asarray(roi, dtype=np.float64)
    if angle_deg is None:
        angle_deg = find_edge_angle(a)

    positions = row_edge_positions(a)
    rows = np.arange(a.shape[0], dtype=np.float64)
    valid = np.isfinite(positions)
    slope, intercept = np.polyfit(rows[valid], positions[valid], 1)

    cols = np.arange(a.shape[1], dtype=np.float64)
    # Signed horizontal distance from the fitted edge, then projected onto the
    # edge normal so the ESF is a true perpendicular profile.
    distance = (cols[None, :] - (slope * rows[:, None] + intercept)) * math.cos(math.radians(angle_deg))

    bin_width = 1.0 / int(oversampling)
    d_flat = distance.ravel()
    v_flat = a.ravel()

    lo, hi = d_flat.min(), d_flat.max()
    n_bins = max(int(np.ceil((hi - lo) / bin_width)), 2)
    idx = np.clip(((d_flat - lo) / bin_width).astype(int), 0, n_bins - 1)

    total = np.bincount(idx, weights=v_flat, minlength=n_bins)
    count = np.bincount(idx, minlength=n_bins)
    occupied = count > 0
    esf = np.full(n_bins, np.nan)
    esf[occupied] = total[occupied] / count[occupied]

    centers = lo + (np.arange(n_bins) + 0.5) * bin_width
    # A slanted edge leaves occasional empty bins at the ends; fill by interpolation.
    if not occupied.all():
        esf = np.interp(centers, centers[occupied], esf[occupied])
    return centers, esf, bin_width


def line_spread_function(esf: np.ndarray, window: bool = True) -> np.ndarray:
    """Differentiate the ESF, optionally Hamming-windowed to suppress tail noise."""
    e = np.asarray(esf, dtype=np.float64)
    lsf = np.gradient(e)
    if window:
        lsf = lsf * np.hamming(lsf.size)
    return lsf


def mtf_from_lsf(
    lsf: np.ndarray, bin_width_px: float, correct_differentiation: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """FFT the LSF into a normalised MTF sampled in cycles per pixel.

    The centred-difference derivative is itself a low-pass filter with response
    ``sinc(f * bin_width)``; dividing it out is what lets the recovered MTF match
    theory instead of drooping at high frequency.
    """
    l = np.asarray(lsf, dtype=np.float64)
    n = l.size
    spectrum = np.abs(np.fft.rfft(l))
    dc = spectrum[0]
    if dc <= 0:
        raise ValueError("LSF has no DC component; the ROI probably contains no edge")
    mtf = spectrum / dc

    freq = np.fft.rfftfreq(n, d=bin_width_px)
    if correct_differentiation:
        correction = np.sinc(freq * bin_width_px)
        mtf = mtf / np.where(np.abs(correction) < 1e-6, 1.0, correction)
    return freq, mtf


def mtf_at(frequency: np.ndarray, mtf: np.ndarray, target_frequency: float) -> float:
    """MTF value at an arbitrary frequency, linearly interpolated."""
    return float(np.interp(target_frequency, frequency, mtf))


def frequency_at_mtf(frequency: np.ndarray, mtf: np.ndarray, level: float) -> float:
    """First frequency where the MTF falls through *level*, linearly interpolated.

    Returns ``nan`` if the curve never reaches that level within the sampled band.
    """
    f = np.asarray(frequency, dtype=np.float64)
    m = np.asarray(mtf, dtype=np.float64)
    below = np.flatnonzero(m < level)
    if below.size == 0 or below[0] == 0:
        return float("nan")
    i = below[0]
    m0, m1 = m[i - 1], m[i]
    if m0 == m1:
        return float(f[i])
    t = (m0 - level) / (m0 - m1)
    return float(f[i - 1] + t * (f[i] - f[i - 1]))


def mtf50(frequency: np.ndarray, mtf: np.ndarray) -> float:
    """The classic single-number sharpness score: where contrast halves."""
    return frequency_at_mtf(frequency, mtf, 0.5)


def slanted_edge_sfr(roi: np.ndarray, oversampling: int = DEFAULT_OVERSAMPLING, window: bool = True) -> SfrResult:
    """Full ISO 12233 chain: ROI -> edge angle -> ESF -> LSF -> MTF."""
    angle = find_edge_angle(roi)
    position, esf, bin_width = edge_spread_function(roi, angle, oversampling)
    lsf = line_spread_function(esf, window=window)
    freq, mtf = mtf_from_lsf(lsf, bin_width)

    # Report only up to the oversampled band's useful range.
    keep = freq <= NYQUIST_CY_PER_PX * oversampling
    freq, mtf = freq[keep], mtf[keep]

    return SfrResult(
        angle_deg=angle,
        esf_position_px=position,
        esf=esf,
        lsf=lsf,
        frequency_cy_per_px=freq,
        mtf=mtf,
        mtf50_cy_per_px=mtf50(freq, mtf),
        mtf10_cy_per_px=frequency_at_mtf(freq, mtf, 0.1),
        mtf_at_nyquist=mtf_at(freq, mtf, NYQUIST_CY_PER_PX),
        bin_width_px=bin_width,
    )


# =====================================================================
# Theoretical MTFs to compare against
# =====================================================================
def gaussian_mtf(frequency_cy_per_px: np.ndarray, sigma_px: float) -> np.ndarray:
    """MTF of a Gaussian PSF: ``exp(-2 pi^2 sigma^2 f^2)``."""
    f = np.asarray(frequency_cy_per_px, dtype=np.float64)
    return np.exp(-2.0 * np.pi**2 * sigma_px**2 * f**2)


def gaussian_sigma_from_mtf50(mtf50_cy_per_px: float) -> float:
    """Invert :func:`gaussian_mtf` at the half-contrast point.

    Useful as a sanity check: measure MTF50 on a Gaussian-blurred edge and this
    must return the sigma you blurred with.
    """
    if mtf50_cy_per_px <= 0:
        return float("inf")
    return math.sqrt(math.log(2.0)) / (math.pi * math.sqrt(2.0) * mtf50_cy_per_px)


def diffraction_cutoff_cy_per_px(f_number: float, wavelength_nm: float, pixel_pitch_um: float) -> float:
    """Incoherent diffraction cutoff ``1 / (lambda N)``, expressed per pixel.

    Beyond this frequency a diffraction-limited lens transmits no contrast at
    all, whatever the sensor does.
    """
    cutoff_cy_per_mm = 1.0e6 / (wavelength_nm * f_number)
    return cutoff_cy_per_mm * (pixel_pitch_um / 1000.0)


def diffraction_mtf(
    frequency_cy_per_px: np.ndarray,
    f_number: float,
    wavelength_nm: float = 550.0,
    pixel_pitch_um: float = 4.3,
) -> np.ndarray:
    """Diffraction-limited MTF of a circular aperture.

    ``(2/pi)(acos(v) - v sqrt(1 - v^2))`` with ``v = f / f_cutoff``.
    """
    f = np.asarray(frequency_cy_per_px, dtype=np.float64)
    cutoff = diffraction_cutoff_cy_per_px(f_number, wavelength_nm, pixel_pitch_um)
    v = np.clip(f / max(cutoff, 1e-12), 0.0, 1.0)
    return (2.0 / np.pi) * (np.arccos(v) - v * np.sqrt(np.maximum(1.0 - v**2, 0.0)))


def pixel_aperture_mtf(frequency_cy_per_px: np.ndarray, fill_factor: float = 1.0) -> np.ndarray:
    """MTF of the pixel's own square aperture: ``|sinc(f w)|``.

    The sensor is not a set of points -- each pixel integrates over its area,
    which is an additional low-pass the lens knows nothing about. At Nyquist a
    full-fill pixel has already lost 36% of the contrast on its own.
    """
    f = np.asarray(frequency_cy_per_px, dtype=np.float64)
    width = math.sqrt(max(fill_factor, 1e-9))
    return np.abs(np.sinc(f * width))


def system_mtf(*mtfs: np.ndarray) -> np.ndarray:
    """Cascade independent stages: MTFs multiply."""
    out = np.ones_like(np.asarray(mtfs[0], dtype=np.float64))
    for m in mtfs:
        out = out * np.asarray(m, dtype=np.float64)
    return out
