"""Dynamic defect pixel correction (DPC) on a raw CFA mosaic, as done on-sensor or early in an ISP.

Each pixel is compared with its ``neighbours`` nearest same-colour sites. It is replaced by their
median when it lies more than a threshold above their maximum (hot) or below their minimum (cold).
The threshold is ``threshold_sigma`` times the expected temporal noise at the neighbour median
(shot + read noise, in DN) plus ``threshold_relative`` times that median above black.

This is a camera processing step, not a renderer fix: it removes isolated real glints as well as
defects and Monte Carlo fireflies, and lowers the mean wherever those carry signal.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

DEFAULTS = {
    "enabled": False,
    "threshold_sigma": 6.0,
    "threshold_relative": 0.0,
    "neighbours": 8,
    "polarity": "both",
}


def bayer_index_tile(pattern: str) -> np.ndarray:
    """2x2 channel-index tile of a Bayer pattern string such as ``RGGB`` (both greens share an index)."""
    p = pattern.upper()
    if len(p) != 4 or set(p) != {"R", "G", "B"}:
        raise ValueError(f"unsupported Bayer pattern {pattern!r}")
    lut = {"R": 0, "G": 1, "B": 2}
    return np.array([[lut[p[0]], lut[p[1]]], [lut[p[2]], lut[p[3]]]], dtype=np.intp)


def same_colour_offsets(index_tile: np.ndarray, neighbours: int = 8) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """For each tile site, the ``neighbours`` nearest (dy, dx) offsets that carry the same channel."""
    rows, cols = index_tile.shape
    reach = max(rows, cols) * 2
    out = {}
    for r in range(rows):
        for c in range(cols):
            ch = index_tile[r, c]
            cand = [
                (dy, dx)
                for dy in range(-reach, reach + 1)
                for dx in range(-reach, reach + 1)
                if (dy or dx) and index_tile[(r + dy) % rows, (c + dx) % cols] == ch
            ]
            cand.sort(key=lambda o: (o[0] ** 2 + o[1] ** 2, o))
            if len(cand) < neighbours:
                raise ValueError(f"tile site {(r, c)} has only {len(cand)} same-colour neighbours in reach")
            out[(r, c)] = cand[:neighbours]
    return out


def correct_defects(
    raw_dn: np.ndarray,
    index_tile: np.ndarray,
    cfg: Mapping | None,
    *,
    black_dn: float,
    K_e_per_DN: float,
    sigma_d_e: float,
) -> tuple[np.ndarray, dict]:
    """Return the corrected mosaic (float32) and stats. ``raw_dn`` is a 2-D mosaic in DN incl. black."""
    c = DEFAULTS | dict(cfg or {})
    if not c["enabled"]:
        return raw_dn, {"enabled": False}
    polarity = str(c["polarity"]).lower()
    if polarity not in ("both", "hot", "cold"):
        raise ValueError(f"defect_correction.polarity must be both|hot|cold, got {polarity!r}")
    raw = np.asarray(raw_dn, dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError(f"defect correction needs a 2-D mosaic, got shape {raw.shape}")
    h, w = raw.shape
    rows, cols = index_tile.shape
    offsets = same_colour_offsets(index_tile, int(c["neighbours"]))
    pad = max(max(abs(dy), abs(dx)) for offs in offsets.values() for dy, dx in offs)
    padded = np.pad(raw, pad, mode="constant", constant_values=np.nan)
    out = raw.copy()
    hot = np.zeros((h, w), dtype=bool)
    cold = np.zeros((h, w), dtype=bool)
    k_sigma = float(c["threshold_sigma"])
    rel = float(c["threshold_relative"])
    for (r, cc), offs in offsets.items():
        ys = slice(r, h, rows)
        xs = slice(cc, w, cols)
        centre = raw[ys, xs]
        ny, nx = centre.shape
        nb = np.stack(
            [
                padded[pad + r + dy : pad + r + dy + ny * rows : rows, pad + cc + dx : pad + cc + dx + nx * cols : cols]
                for dy, dx in offs
            ]
        )
        med = np.nanmedian(nb, axis=0)
        sig = np.maximum(med - black_dn, 0.0)
        noise_dn = np.sqrt(sig / K_e_per_DN + (sigma_d_e / K_e_per_DN) ** 2)
        thr = k_sigma * noise_dn + rel * sig
        is_hot = centre > np.nanmax(nb, axis=0) + thr if polarity != "cold" else np.zeros_like(centre, bool)
        is_cold = centre < np.nanmin(nb, axis=0) - thr if polarity != "hot" else np.zeros_like(centre, bool)
        bad = is_hot | is_cold
        out[ys, xs] = np.where(bad, med, centre)
        hot[ys, xs] = is_hot
        cold[ys, xs] = is_cold
    n = int(hot.sum() + cold.sum())
    stats = {
        "enabled": True,
        "corrected_count": n,
        "corrected_fraction": n / float(h * w),
        "hot_count": int(hot.sum()),
        "cold_count": int(cold.sum()),
        "signal_removed_dn": float(np.sum(raw - out)),
        "threshold_sigma": k_sigma,
        "threshold_relative": rel,
        "neighbours": int(c["neighbours"]),
        "polarity": polarity,
    }
    return out, stats
