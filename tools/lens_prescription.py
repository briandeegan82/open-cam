"""On-axis light-gathering f-number of a pbrt-v4 ``RealisticCamera`` lens prescription.

A lens file lists interfaces front (object) to back: ``radius thickness eta aperture`` in mm,
with ``radius 0`` marking the aperture stop. pbrt clamps a requested ``aperturediameter`` to
the stop's listed diameter, focuses by placing the film behind the last element, and weights
film rays by ``cos θ dω``. The on-axis film irradiance is therefore ``π L sin²θ_max``, where
``θ_max`` is the widest film-side ray that clears every element. The working f-number at
infinity focus is ``N = 1 / (2 sin θ_max)``.

``focal_length / aperture_diameter`` is not this: the stop sits inside the lens, so the
entrance pupil differs from the stop, and pupil aberrations change the real marginal ray.
For ``wide_22mm.dat`` at full stop, 22 / 8.756 = f/2.51, but the traced lens collects light
like f/2.79.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path


def load_lens_file(path: Path) -> tuple[tuple[float, float, float, float], ...]:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        vals = [float(v) for v in line.split()[:4]]
        if len(vals) != 4:
            raise ValueError(f"{path}: expected 4 columns (radius thickness eta aperture), got {line!r}")
        rows.append(tuple(vals))
    if not any(r[0] == 0.0 for r in rows):
        raise ValueError(f"{path}: no aperture stop (radius 0) in lens prescription")
    return tuple(rows)


def _eta(e: float) -> float:
    return 1.0 if e == 0.0 else e


def paraxial_focal_lengths_mm(rows) -> tuple[float, float]:
    """Return (effective focal length, back focal distance) from a paraxial y-nu trace."""
    n, y, u = 1.0, 1.0, 0.0
    for i, (radius, thickness, eta, _ap) in enumerate(rows):
        n2 = _eta(eta)
        power = 0.0 if radius == 0.0 else (n2 - n) / radius
        u = (n * u - y * power) / n2
        n = n2
        if i < len(rows) - 1:
            y += thickness * u
    return -1.0 / u, -y / u


def _ray_clears_lens(rows, film_distance_mm: float, theta: float, stop_diameter_mm: float) -> bool:
    """Trace a meridional ray from the on-axis film point toward the object through every interface."""
    # Coordinates (y, z): film at z = 0, z increasing toward the object.
    py, pz = 0.0, 0.0
    vy, vz = math.sin(theta), math.cos(theta)
    z_vertex = film_distance_mm
    for i in range(len(rows) - 1, -1, -1):
        radius, _t, eta, aperture = rows[i]
        if i < len(rows) - 1:
            z_vertex += rows[i][1]
        if radius == 0.0:
            s = (z_vertex - pz) / vz
            py, pz = py + s * vy, pz + s * vz
            if abs(py) > 0.5 * min(stop_diameter_mm, aperture):
                return False
            continue
        cz = z_vertex - radius
        oy, oz = py, pz - cz
        b = oy * vy + oz * vz
        disc = b * b - (oy * oy + oz * oz - radius * radius)
        if disc < 0.0:
            return False
        roots = [s for s in (-b - math.sqrt(disc), -b + math.sqrt(disc)) if s > 1e-9]
        if not roots:
            return False
        s = min(roots, key=lambda r: abs(pz + r * vz - z_vertex))
        py, pz = py + s * vy, pz + s * vz
        if abs(py) > 0.5 * aperture:
            return False
        ny, nz = py / radius, (pz - cz) / radius
        if ny * vy + nz * vz > 0.0:
            ny, nz = -ny, -nz
        n_from = _eta(eta)
        n_to = _eta(rows[i - 1][2]) if i > 0 else 1.0
        ratio = n_from / n_to
        cos_i = -(ny * vy + nz * vz)
        k = 1.0 - ratio * ratio * (1.0 - cos_i * cos_i)
        if k < 0.0:
            return False
        vy = ratio * vy + (ratio * cos_i - math.sqrt(k)) * ny
        vz = ratio * vz + (ratio * cos_i - math.sqrt(k)) * nz
        norm = math.hypot(vy, vz)
        vy, vz = vy / norm, vz / norm
    return True


@lru_cache(maxsize=64)
def traced_f_number(lens_file: str, aperture_diameter_mm: float | None = None) -> float:
    """Infinity-focus working f-number ``1 / (2 sin θ_max)`` of the traced on-axis film point.

    ``aperture_diameter_mm`` is pbrt's ``aperturediameter`` (stop diameter); like pbrt it is clamped
    to the stop's listed diameter, and ``None`` means fully open.
    """
    rows = load_lens_file(Path(lens_file))
    stop_max = next(r[3] for r in rows if r[0] == 0.0)
    stop = stop_max if aperture_diameter_mm is None else min(float(aperture_diameter_mm), stop_max)
    if stop <= 0.0:
        raise ValueError(f"aperture diameter must be positive, got {aperture_diameter_mm}")
    _efl, bfd = paraxial_focal_lengths_mm(rows)
    if not _ray_clears_lens(rows, bfd, 1e-6, stop):
        raise ValueError(f"{lens_file}: axial ray does not pass the lens")
    lo, hi = 0.0, 0.5 * math.pi * 0.999
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _ray_clears_lens(rows, bfd, mid, stop):
            lo = mid
        else:
            hi = mid
    return 1.0 / (2.0 * math.sin(lo))
