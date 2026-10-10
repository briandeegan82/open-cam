"""Even aspheric surfaces for traced lens prescriptions (pbrt ``RealisticCamera`` lens files).

A pbrt lens file has spherical interfaces only.  A sidecar ``<lens>.asphere.yaml`` next to the
``.dat`` gives a row the even-asphere sag used by phone-lens patents (Zemax "even asphere")

    z(r) = c r^2 / (1 + sqrt(1 - (1 + k) c^2 r^2)) + sum_i A_i r^i,   c = 1 / R,  i = 4, 6, ..., 16,

with z positive toward the film for R > 0 (the same sign convention as the lens file radius).
``R`` stays in the lens file, so paraxial helpers (EFL, pupils, pbrt focusing) are unchanged:
the conic and polynomial terms are all O(r^4).  The same eight numbers per row (k, A4..A16)
go to the patched pbrt camera as ``"float asphere"`` (third_party/patches/
0004-realistic-camera-asphere.patch), and :func:`intersect` gives the Python tracers the
identical surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ORDERS = (4, 6, 8, 10, 12, 14, 16)
N_PBRT = 1 + len(ORDERS)


@dataclass(frozen=True)
class Asphere:
    k: float = 0.0
    coeffs: tuple[float, ...] = (0.0,) * len(ORDERS)

    def sag(self, radius: float, r) -> np.ndarray:
        r = np.asarray(r, dtype=np.float64)
        with np.errstate(all="ignore"):
            return self._sag(radius, r)

    def _sag(self, radius: float, r: np.ndarray) -> np.ndarray:
        c = 1.0 / radius
        r2 = r * r
        arg = 1.0 - (1.0 + self.k) * c * c * r2
        z = c * r2 / (1.0 + np.sqrt(np.clip(arg, 0.0, None)))
        for a, i in zip(self.coeffs, ORDERS):
            z = z + a * r**i
        return np.where(arg >= 0.0, z, np.nan)

    def slope(self, radius: float, r) -> np.ndarray:
        """dz/dr."""
        r = np.asarray(r, dtype=np.float64)
        with np.errstate(all="ignore"):
            return self._slope(radius, r)

    def _slope(self, radius: float, r: np.ndarray) -> np.ndarray:
        c = 1.0 / radius
        arg = 1.0 - (1.0 + self.k) * c * c * r * r
        s = c * r / np.sqrt(np.clip(arg, 1e-300, None))
        for a, i in zip(self.coeffs, ORDERS):
            s = s + i * a * r ** (i - 1)
        return np.where(arg > 0.0, s, np.nan)


def asphere_map_path(lens_path: str | Path) -> Path:
    """Sidecar path: ``phone.4mm.dat`` -> ``phone.4mm.asphere.yaml``."""
    return Path(lens_path).with_suffix(".asphere.yaml")


def load_aspheres(lens_path: str | Path) -> list[Asphere | None] | None:
    """Per lens-file row asphere (None = spherical), or None if the lens has no sidecar."""
    import yaml
    from lens_prescription import load_lens_file

    lens_path = Path(lens_path)
    side = asphere_map_path(lens_path)
    if not side.is_file():
        return None
    cfg = yaml.safe_load(side.read_text()) or {}
    rows = load_lens_file(lens_path)
    out: list[Asphere | None] = [None] * len(rows)
    for key, entry in (cfg.get("rows") or {}).items():
        i = int(key)
        if not 0 <= i < len(rows) or rows[i][0] == 0.0:
            raise ValueError(f"{side}: row {i} is not a curved interface of {lens_path.name}")
        a = {int(o): float(v) for o, v in (entry.get("A") or {}).items()}
        bad = set(a) - set(ORDERS)
        if bad:
            raise ValueError(f"{side}: row {i}: unsupported orders {sorted(bad)} (even 4..16 only)")
        out[i] = Asphere(float(entry.get("k", 0.0)), tuple(a.get(o, 0.0) for o in ORDERS))
    return out


def pbrt_asphere_values(lens_path: str | Path) -> list[float] | None:
    """Eight floats per lens-file row (k, A4..A16; zeros = sphere) for pbrt."""
    asph = load_aspheres(lens_path)
    if asph is None:
        return None
    vals: list[float] = []
    for a in asph:
        vals += [0.0] * N_PBRT if a is None else [a.k, *a.coeffs]
    return vals


def pbrt_camera_lines(lens_path: str | Path) -> list[str]:
    """``"float asphere"`` line for a ``Camera "realistic"`` block ([] without a sidecar)."""
    vals = pbrt_asphere_values(lens_path)
    if not vals:
        return []
    return [f'    "float asphere" [{" ".join(f"{v:.9g}" for v in vals)}]']


def intersect(o: np.ndarray, d: np.ndarray, z_vertex: float, radius: float, asph: Asphere, iters: int = 40):
    """Ray/asphere intersection by Newton on z(t) - z_v - sag(r(t)); returns (t, normal, ok)."""
    t = (z_vertex - o[:, 2]) / d[:, 2]
    for _ in range(iters):
        p = o + t[:, None] * d
        r = np.hypot(p[:, 0], p[:, 1])
        f = p[:, 2] - z_vertex - asph.sag(radius, r)
        rs = np.where(r > 0, r, 1.0)
        df = d[:, 2] - asph.slope(radius, r) * (p[:, 0] * d[:, 0] + p[:, 1] * d[:, 1]) / rs
        step = np.where(np.isfinite(f / df), f / df, 0.0)
        t = t - step
        if np.all(np.abs(step) < 1e-12):
            break
    p = o + t[:, None] * d
    r = np.hypot(p[:, 0], p[:, 1])
    f = p[:, 2] - z_vertex - asph.sag(radius, r)
    s = asph.slope(radius, r)
    rs = np.where(r > 0, r, 1.0)
    nrm = np.stack([-s * p[:, 0] / rs, -s * p[:, 1] / rs, np.ones_like(r)], axis=1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    ok = np.isfinite(f) & (np.abs(f) < 1e-9) & np.isfinite(nrm).all(axis=1)
    return t, nrm, ok
