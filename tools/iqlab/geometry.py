"""Dot-grid geometry metrics: local geometric distortion and lateral chromatic displacement (CPIQ / ISO 17850 style)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


def dot_centroids(img: np.ndarray, *, dark_dots: bool = True, min_area: int = 4) -> np.ndarray:
    """Intensity-weighted centroids ``(N, 2)`` as (x, y) of the dots of a dot-grid chart (2-D image)."""
    a = np.asarray(img, dtype=np.float64)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    thr = 0.5 * (lo + hi)
    mask = a < thr if dark_dots else a > thr
    lab, n = ndimage.label(mask)
    if n == 0:
        return np.empty((0, 2))
    weight = (hi - a) if dark_dots else (a - lo)
    idx = np.arange(1, n + 1)
    area = ndimage.sum(mask, lab, idx)
    com = np.array(ndimage.center_of_mass(np.clip(weight, 0, None), lab, idx))
    keep = area >= min_area
    h, w = a.shape
    touching = np.array(
        [sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == h or sl[1].stop == w for sl in ndimage.find_objects(lab)]
    )
    keep &= ~touching
    return com[keep][:, ::-1]


@dataclass(frozen=True)
class GridDistortion:
    max_local_percent: float
    rms_local_percent: float
    radial_percent: np.ndarray
    field_radius: np.ndarray
    ideal: np.ndarray
    measured: np.ndarray


def _fit_central_grid(points: np.ndarray, centre: np.ndarray, n_fit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.linalg.norm(points - centre, axis=1)
    p0 = points[np.argmin(d)]
    rel = points - p0
    r = np.linalg.norm(rel, axis=1)
    nn = rel[np.argsort(r)[1:9]]
    e1 = nn[np.argmin(np.abs(np.arctan2(nn[:, 1], nn[:, 0])))]
    perp = nn[np.argmax(np.abs(e1[0] * nn[:, 1] - e1[1] * nn[:, 0]) / np.linalg.norm(nn, axis=1))]
    basis = np.column_stack([e1, perp])
    ij = np.rint(np.linalg.solve(basis, rel.T).T)
    near = np.argsort(r)[:n_fit]
    a = np.column_stack([np.ones(near.size), ij[near]])
    coef, *_ = np.linalg.lstsq(a, points[near], rcond=None)
    return coef[0], coef[1:].T, ij


def grid_distortion(points: np.ndarray, image_shape: tuple[int, int], n_fit: int = 9) -> GridDistortion:
    """Fit the ideal (undistorted) grid to the dots nearest the image centre, then report per-dot
    radial distortion (r_measured - r_ideal) / r_ideal x 100 and its max/RMS ("local geometric distortion")."""
    pts = np.asarray(points, dtype=np.float64)
    centre = np.array([(image_shape[1] - 1) / 2.0, (image_shape[0] - 1) / 2.0])
    origin, basis, ij = _fit_central_grid(pts, centre, n_fit)
    ideal = origin + ij @ basis.T
    r_ideal = np.linalg.norm(ideal - centre, axis=1)
    r_meas = np.linalg.norm(pts - centre, axis=1)
    sel = r_ideal > 0.15 * r_ideal.max()
    radial = (r_meas[sel] - r_ideal[sel]) / r_ideal[sel] * 100.0
    return GridDistortion(
        float(np.max(np.abs(radial))), float(np.sqrt(np.mean(radial**2))), radial, r_ideal[sel], ideal[sel], pts[sel]
    )


@dataclass(frozen=True)
class ChromaticDisplacement:
    max_px: float
    mean_px: float
    red_minus_green: np.ndarray
    blue_minus_green: np.ndarray
    field_radius: np.ndarray


def lateral_chromatic_displacement(
    rgb: np.ndarray, *, dark_dots: bool = True, max_match_px: float = 5.0
) -> ChromaticDisplacement:
    """Per-dot R-G and B-G centroid offsets (pixels) on a dot-grid image ``(h, w, 3)``."""
    cents = [dot_centroids(rgb[..., c], dark_dots=dark_dots) for c in range(3)]
    g = cents[1]

    def _match(other: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(other[:, None, :] - g[None, :, :], axis=2)
        j = np.argmin(d, axis=0)
        off = other[j] - g
        off[np.min(d, axis=0) > max_match_px] = np.nan
        return off

    rg, bg = _match(cents[0]), _match(cents[2])
    h, w = rgb.shape[:2]
    centre = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
    mag = np.nanmax(np.stack([np.linalg.norm(rg, axis=1), np.linalg.norm(bg, axis=1)]), axis=0)
    return ChromaticDisplacement(
        float(np.nanmax(mag)), float(np.nanmean(mag)), rg, bg, np.linalg.norm(g - centre, axis=1)
    )


def dot_grid_image(
    shape: tuple[int, int] = (480, 640), pitch: float = 40.0, radius: float = 6.0, k1: float = 0.0, supersample: int = 4
) -> np.ndarray:
    """Synthetic dark-dot grid (white background) with optional radial distortion r' = r (1 + k1 r^2), r normalised to the half-diagonal."""
    h, w = shape
    s = supersample
    yy, xx = np.mgrid[0 : h * s, 0 : w * s]
    x = (xx + 0.5) / s - 0.5
    y = (yy + 0.5) / s - 0.5
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    rn = np.hypot(cx, cy)
    dx, dy = (x - cx) / rn, (y - cy) / rn
    r2 = dx * dx + dy * dy
    # invert r' = r(1+k1 r^2) with two fixed-point steps (image -> undistorted object coordinates)
    scale = np.ones_like(r2)
    for _ in range(5):
        scale = 1.0 / (1.0 + k1 * r2 * scale**2)
    ux, uy = cx + dx * scale * rn, cy + dy * scale * rn
    gx = (ux - cx) / pitch
    gy = (uy - cy) / pitch
    d = np.hypot((gx - np.rint(gx)) * pitch, (gy - np.rint(gy)) * pitch)
    img = np.where(d <= radius, 0.1, 0.9)
    return img.reshape(h, s, w, s).mean(axis=(1, 3))
