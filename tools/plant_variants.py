"""Split Poly Haven plant files that hold a row of 2-7 versions of a plant into separate plants.

Several Poly Haven vegetation models (fir tree and saplings, Searsia, wild rooibos, shrubs, grass
clumps) are a row of differently sized versions side by side in one file, so instancing the file
plants the whole row. ``split`` groups the file's shapes into plants by their ground footprint:
two shapes belong to the same plant when the median (x, z) point of one lies inside the other's
footprint (2-98 % percentile box), which joins each trunk, bark and foliage mesh to its own crown.
Clusters lower than ``MIN_HEIGHT_FRACTION`` of the tallest are dropped (stray sprigs).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fetch_highway_assets import read_ply_positions

# Sparse sprig patches kept as one plant (splitting them gives single 5 cm grass blades).
WHOLE = {"veg_grass_bermuda_01"}
MIN_HEIGHT_FRACTION = 0.3


def _plant(idx: list[int], pts: list[np.ndarray]) -> dict:
    p = np.concatenate([pts[k] for k in idx])
    lo, hi = np.percentile(p, 0.05, 0), np.percentile(p, 99.95, 0)
    return {
        "shapes": sorted(idx),
        "centre": [round(float((lo[0] + hi[0]) / 2), 4), 0.0, round(float((lo[2] + hi[2]) / 2), 4)],
        "height_m": round(float(hi[1] - min(lo[1], 0.0)), 4),
        "lo": lo.round(4).tolist(),
        "hi": hi.round(4).tolist(),
    }


def split(vid: str, info: dict, prep: Path) -> list[dict]:
    """Plants in a prepared vegetation asset, sorted along x; one entry when the file is a single plant."""
    pts = []
    for s in info["shapes"]:
        p = read_ply_positions(prep / s["ply"])
        pts.append(p[np.isfinite(p).all(1)])
    n = len(pts)
    if vid in WHOLE or n < 2:
        return [_plant(list(range(n)), pts)]
    box = [(np.percentile(p[:, [0, 2]], 2, 0), np.percentile(p[:, [0, 2]], 98, 0)) for p in pts]
    med = [np.median(p[:, [0, 2]], 0) for p in pts]
    parent = list(range(n))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def inside(i: int, j: int) -> bool:
        return bool(np.all(med[i] >= box[j][0]) and np.all(med[i] <= box[j][1]))

    for i in range(n):
        for j in range(i + 1, n):
            if inside(i, j) or inside(j, i):
                parent[root(i)] = root(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    plants = [_plant(g, pts) for g in groups.values()]
    tallest = max(p["height_m"] for p in plants)
    plants = [p for p in plants if p["height_m"] >= MIN_HEIGHT_FRACTION * tallest]
    return sorted(plants, key=lambda p: (p["centre"][0], p["centre"][2]))
