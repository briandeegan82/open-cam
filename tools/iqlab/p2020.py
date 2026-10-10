"""IEEE P2020 (automotive image quality) style metrics: contrast detection probability and colour separation.

CDP follows Geese et al., "Detection probabilities: performance prediction for sensors of autonomous
vehicles", IS&T Electronic Imaging 2018 (adopted by IEEE P2020): sample pixel pairs from a dark and a
bright patch, compute the per-pair Michelson contrast, and report the probability that it lies within
``epsilon`` (relative) of the nominal contrast. Wrong-sign pairs count as failures.

Colour separation scores how separable two colour patches are in the camera's output space using the
Euclidean and Mahalanobis distance between their pixel distributions (cf. the P2020 colour separation
work; github.com/briandeegan82/IEEE-P2020-Color-Separation).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np


def michelson_contrast(dark: float | np.ndarray, bright: float | np.ndarray) -> float | np.ndarray:
    d = np.asarray(dark, dtype=np.float64)
    b = np.asarray(bright, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (b - d) / (b + d)


def contrast_detection_probability(
    dark_pixels: np.ndarray,
    bright_pixels: np.ndarray,
    *,
    nominal_contrast: float | None = None,
    epsilon: float = 0.5,
    n_pairs: int = 200_000,
    seed: int = 0,
) -> float:
    """CDP for one patch pair (pixel values linear and black-level subtracted).

    ``nominal_contrast`` defaults to the Michelson contrast of the patch means; pass the scene
    (luminance) contrast to include tone-curve/flare errors in the score.
    """
    d = np.asarray(dark_pixels, dtype=np.float64).ravel()
    b = np.asarray(bright_pixels, dtype=np.float64).ravel()
    if d.size == 0 or b.size == 0:
        raise ValueError("empty patch")
    c_nom = float(michelson_contrast(d.mean(), b.mean())) if nominal_contrast is None else float(nominal_contrast)
    if not np.isfinite(c_nom) or c_nom <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    dk = d[rng.integers(0, d.size, n_pairs)]
    bk = b[rng.integers(0, b.size, n_pairs)]
    den = bk + dk
    ck = np.where(den > 0, (bk - dk) / np.where(den > 0, den, 1.0), -np.inf)
    return float(np.mean(np.abs(ck - c_nom) <= epsilon * c_nom))


def cdp_vs_level(
    patches: list[np.ndarray], *, contrast_pairs: list[tuple[int, int]], epsilon: float = 0.5, **kw
) -> np.ndarray:
    """CDP for a list of (dark_index, bright_index) patch pairs, e.g. adjacent HDR-chart steps."""
    return np.array(
        [contrast_detection_probability(patches[i], patches[j], epsilon=epsilon, **kw) for i, j in contrast_pairs]
    )


@dataclass(frozen=True)
class ColourSeparation:
    euclidean: float
    mahalanobis: float
    separation_probability: float
    empirical_accuracy: float


def _phi(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def colour_separation(a_pixels: np.ndarray, b_pixels: np.ndarray) -> ColourSeparation:
    """Separability of two colour patches given their pixels ``(N, C)``.

    ``separation_probability`` = Phi(D_M / 2): the probability that a pixel is assigned to the right
    patch by the optimal linear classifier under a shared-covariance Gaussian model. ``empirical_accuracy``
    is the measured nearest-mean (Mahalanobis) classification rate on the pixels themselves.
    """
    a = np.asarray(a_pixels, dtype=np.float64).reshape(-1, np.shape(a_pixels)[-1])
    b = np.asarray(b_pixels, dtype=np.float64).reshape(-1, np.shape(b_pixels)[-1])
    ma, mb = a.mean(0), b.mean(0)
    diff = mb - ma
    pooled = ((a.shape[0] - 1) * np.cov(a.T) + (b.shape[0] - 1) * np.cov(b.T)) / (a.shape[0] + b.shape[0] - 2)
    pooled = np.atleast_2d(pooled)
    inv = np.linalg.pinv(pooled)
    dm = float(np.sqrt(max(diff @ inv @ diff, 0.0)))

    def _d2(x: np.ndarray, m: np.ndarray) -> np.ndarray:
        r = x - m
        return np.einsum("ij,jk,ik->i", r, inv, r)

    correct = np.concatenate([_d2(a, ma) < _d2(a, mb), _d2(b, mb) < _d2(b, ma)])
    return ColourSeparation(float(np.linalg.norm(diff)), dm, _phi(dm / 2.0), float(np.mean(correct)))
