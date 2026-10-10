"""Flare / stray-light metrics: veiling glare (black-hole target, ISO 18844 / ISO 9358 style) and
point-source flare (ghost peak and stray-light fraction around a bright source).

Both locate the target features in the image itself, so they work through distortion and on any
lens (pinhole, pbrt RealisticCamera, real captures). Inputs are linear images (radiance or electrons).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class HoleGlare:
    x: float
    y: float
    hole_mean: float
    surround_mean: float

    @property
    def glare_percent(self) -> float:
        return 100.0 * self.hole_mean / self.surround_mean if self.surround_mean > 0 else float("nan")


def black_hole_glare(
    img: np.ndarray,
    *,
    hole_true: float = 0.0,
    min_area: int = 16,
    core: float = 0.5,
    ring: tuple[float, float] = (0.4, 0.9),
) -> list[HoleGlare]:
    """Veiling glare of every dark "black hole" in a bright uniform field.

    For each hole (dark blob not touching the border): ``hole_mean`` is the mean of the central
    ``core`` fraction of the hole (minus ``hole_true``, the hole's own radiance), ``surround_mean`` the
    mean of a ring ``ring`` x equivalent-radius outside the hole edge. Glare % = 100 * hole / surround.
    """
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 3:
        a = a.mean(axis=2)
    thr = 0.5 * float(np.median(a))
    lab, n = ndimage.label(a < thr)
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == h or sl[1].stop == w:
            continue
        m = lab == k
        area = int(m.sum())
        if area < min_area:
            continue
        cy, cx = ndimage.center_of_mass(m)
        r_eq = np.sqrt(area / np.pi)
        dist_in = ndimage.distance_transform_edt(m)
        core_m = dist_in >= (1.0 - core) * dist_in.max()
        dist_out = ndimage.distance_transform_edt(~m)
        ring_m = (dist_out >= ring[0] * r_eq) & (dist_out <= ring[1] * r_eq) & (lab == 0)
        ring_m &= np.hypot(xx - cx, yy - cy) <= 3 * r_eq
        out.append(HoleGlare(float(cx), float(cy), float(a[core_m].mean() - hole_true), float(a[ring_m].mean())))
    return out


@dataclass(frozen=True)
class PointSourceFlare:
    x: float
    y: float
    source_energy: float
    stray_fraction: float
    ghost_peak_relative: float
    radius_px: np.ndarray
    radial_profile: np.ndarray


def point_source_flare(
    img: np.ndarray, *, exclude_radius_px: float = 10.0, background: float = 0.0, n_bins: int = 64
) -> PointSourceFlare:
    """Flare around the brightest source in a dark image.

    ``stray_fraction``: energy farther than ``exclude_radius_px`` from the source centroid / total energy.
    ``ghost_peak_relative``: brightest pixel outside the exclusion radius / source peak pixel (dimensionless
    ghost-to-source ratio). ``radial_profile``: mean signal in annuli, normalised to the source peak.
    """
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 3:
        a = a.mean(axis=2)
    a = a - background
    peak = float(a.max())
    lab, _ = ndimage.label(a >= 0.5 * peak)
    k = int(lab.flat[int(np.argmax(a))])
    cy, cx = ndimage.center_of_mass(np.clip(a, 0, None) * (lab == k))
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - cx, yy - cy)
    far = r > exclude_radius_px
    total = float(np.clip(a, 0, None).sum())
    edges = np.linspace(0, r.max(), n_bins + 1)
    idx = np.clip(np.digitize(r.ravel(), edges) - 1, 0, n_bins - 1)
    prof = np.bincount(idx, a.ravel(), n_bins) / np.maximum(np.bincount(idx, minlength=n_bins), 1)
    return PointSourceFlare(
        float(cx),
        float(cy),
        float(np.clip(a[~far], 0, None).sum()),
        float(np.clip(a[far], 0, None).sum() / total) if total > 0 else float("nan"),
        float(a[far].max() / peak) if far.any() and peak > 0 else 0.0,
        0.5 * (edges[1:] + edges[:-1]),
        prof / peak if peak > 0 else prof,
    )
