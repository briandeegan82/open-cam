#!/usr/bin/env python3
"""Physically traced lens ghosts (two-reflection flare paths) for pbrt SpectralFilm EXRs.

pbrt-v4's ``RealisticCamera`` refracts camera rays through the lens prescription with unit
transmission at every interface: no Fresnel loss and no inter-reflections.  This module adds the
missing ghost light as a post-render step that runs *before* the radiance -> electrons conversion,
so every spectral bucket is weighted by the sensor QE afterwards.

Model (after M. B. Hullin, E. Eisemann, H.-P. Seidel and S. Lee, "Physically-based real-time lens
flare rendering", ACM Trans. Graph. 30(4), 108 (SIGGRAPH 2011), doi:10.1145/2010324.1965003):

* The sequential prescription (``config/lenses/*.dat``, pbrt format, see :mod:`lens_prescription`)
  has K refracting interfaces.  A ghost is light that reflects once at interface ``j`` and once
  more at an earlier interface ``i < j`` before reaching the film: K(K-1)/2 two-reflection paths
  (66 for ``wide_22mm.dat``).  An optional reflective sensor plane adds K more (``sensor_reflectance``).
* For each bright source (pixels above a threshold, grouped into ``cluster_px`` cells) a collimated
  beam at the source's field direction is sampled on a regular grid over the front element and traced
  in 3-D through each path with exact sphere intersections, vector Snell refraction/reflection,
  every element's clear aperture and the aperture stop (circular, or an ``iris_blades``-gon).
  Geometry is wavelength independent (pbrt lens files have one index per glass, as in pbrt).
* Each ray carries the product of the interface reflectances/transmittances along its path, from
  the thin-film model in :mod:`lens_coatings` evaluated at the ray's own incidence angle for every
  spectral bucket (Fresnel, single-layer MgF2, quarter-half-quarter multilayer; cemented glass-glass
  interfaces are uncoated Fresnel).
* Energy normalisation: the rendered source flux (sum of its pixels per bucket) is what pbrt passed
  through the primary path with unit transmission, i.e. ``N_primary`` grid rays.  A ghost ray adds
  ``flux * weight / N_primary`` - so a ghost's total energy relative to the primary image is exactly
  the traced ratio, independent of the scene units (film irradiance or radiance).
* The ray grid is rasterised as bilinear patches (each grid cell's energy spread over the quad
  spanned by its four traced corners, as in Hullin et al. 2011), giving smooth, energy-conserving
  ghost footprints whose shape is the (vignetted, defocused) image of the stop.

The source -> direction and film -> pixel mappings use the same traced lens: for ``camera: realistic``
renders the film is pbrt's physical extent (``film_diagonal_mm``, pbrt's default 35 mm); for pinhole /
thin-lens renders a pixel's field angle comes from the pinhole ``fov_deg`` and film heights are mapped
back through the traced primary image height h(theta).

Validation (tests/test_lens_ghosts.py, docs/LENS_GHOSTS.md): R + T = 1 and the analytic quarter-wave
results for the coatings; the single-surface-pair singlet energy T R R T; ghost positions against
the paraxial ghost matrices of S. Lee and E. Eisemann, "Practical real-time lens-flare rendering",
Comput. Graph. Forum 32(4), 1-6 (EGSR 2013) and an independent scalar ray trace; per-ghost energies
against a Monte-Carlo non-sequential trace; energy conservation of the splat.

Not modelled: diffraction and interference between ghosts (geometric optics only), glass and coating
dispersion, glass absorption, ghosts with four or more reflections, barrel/mount scatter, sensor
cover glass / IR filter (unless approximated by ``sensor_reflectance``), polarisation state (the
average of s and p is used at each interface), sources at finite distance (beams are collimated),
and the ghosts of pixels below the source threshold (use the parametric veiling glare in
``apply_spectral_psf.py`` for that diffuse component; ISO 18844:2017 "Photography - Image flare
measurement" is the measurement standard to calibrate it against).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from lens_coatings import (
    DEFAULT_DESIGN_NM,
    coating_for_interface,
    interface_rt,
    lookup_reflectance,
    reflectance_table,
)
from lens_prescription import load_lens_file

_RGB_CENTER_NM = {"R": 620.0, "G": 540.0, "B": 460.0}
PBRT_DEFAULT_FILM_DIAGONAL_MM = 35.0


# ---------------------------------------------------------------------------
# Lens system
# ---------------------------------------------------------------------------


@dataclass
class Surface:
    z: float  # vertex position [mm], z increases toward the film
    radius: float  # [mm], > 0: centre of curvature on the film side; 0: aperture stop
    ap_r: float  # clear semi-aperture [mm]
    n_before: float  # index on the object side
    n_after: float  # index on the film side
    coating: object = None

    @property
    def is_stop(self) -> bool:
        return self.radius == 0.0


@dataclass
class Lens:
    surfaces: list[Surface]
    z_film: float
    stop_index: int
    iris_blades: int = 0
    iris_rotation_deg: float = 0.0
    sensor_reflectance: float = 0.0
    lens_file: str = ""
    efl_mm: float = float("nan")
    listed_film_z: float = 0.0  # film position implied by the lens file's last thickness

    @property
    def refracting(self) -> list[int]:
        return [k for k, s in enumerate(self.surfaces) if not s.is_stop]

    @property
    def sensor_index(self) -> int:
        return len(self.surfaces)


def _eta(e: float) -> float:
    return 1.0 if e == 0.0 else float(e)


def load_lens(
    lens_file,
    *,
    aperture_diameter_mm: float | None = None,
    focus_distance_m: float | None = None,
    coating="mgf2",
    surface_coatings: dict | None = None,
    design_nm: float = DEFAULT_DESIGN_NM,
    iris_blades: int = 0,
    iris_rotation_deg: float = 0.0,
    sensor_reflectance: float = 0.0,
    rows=None,
) -> Lens:
    """Build a :class:`Lens` from a pbrt lens file (or explicit ``rows``), focused like pbrt.

    Air-glass interfaces get ``coating`` (``surface_coatings`` maps row index -> coating to override);
    glass-glass (cemented) interfaces are always uncoated.  ``aperture_diameter_mm`` is clamped to the
    stop's listed diameter as pbrt does; ``focus_distance_m`` (from the film, pbrt's ``focusdistance``)
    ``None``/inf focuses at infinity.
    """
    rows = rows if rows is not None else load_lens_file(Path(lens_file))
    surface_coatings = {int(k): v for k, v in (surface_coatings or {}).items()}
    surfs: list[Surface] = []
    z = 0.0
    stop_index = -1
    for k, (radius, thickness, eta, aperture) in enumerate(rows):
        n_before = 1.0 if k == 0 else _eta(rows[k - 1][2])
        n_after = _eta(eta)
        ap_r = 0.5 * float(aperture)
        if radius == 0.0:
            stop_index = k
            if aperture_diameter_mm is not None:
                ap_r = min(ap_r, 0.5 * float(aperture_diameter_mm))
        s = Surface(z, float(radius), ap_r, n_before, n_after)
        if radius != 0.0:
            if min(n_before, n_after) == 1.0 and n_before != n_after:
                spec = surface_coatings.get(k, coating)
                s.coating = coating_for_interface(spec, 1.0, max(n_before, n_after), design_nm)
            else:
                s.coating = coating_for_interface("uncoated", 1.0, 1.0)
        surfs.append(s)
        z += float(thickness)
    if stop_index < 0:
        raise ValueError("lens prescription has no aperture stop (radius 0)")
    lens = Lens(
        surfs,
        z_film=0.0,
        stop_index=stop_index,
        iris_blades=int(iris_blades),
        iris_rotation_deg=float(iris_rotation_deg),
        sensor_reflectance=float(sensor_reflectance),
        lens_file=str(lens_file or ""),
    )
    lens.listed_film_z = z
    lens.z_film, lens.efl_mm = _focus_film_z(lens, focus_distance_m)
    return lens


def _parallel_ray_cardinal(lens: Lens, h: float, from_film: bool) -> tuple[float, float]:
    """(principal plane z, focal point z) from a ray parallel to the axis at height ``h`` (pbrt-style)."""
    if not from_film:
        o = np.array([[h, 0.0, lens.surfaces[0].z - 1.0]])
        d = np.array([[0.0, 0.0, 1.0]])
        res = trace(lens, primary_path(lens), o, d, to_film=False)
    else:
        last = lens.surfaces[-1].z
        o = np.array([[h, 0.0, last + 1.0]])
        d = np.array([[0.0, 0.0, -1.0]])
        steps = [(k, "T") for k in range(len(lens.surfaces) - 1, -1, -1)]
        res = trace(lens, steps, o, d, to_film=False, start_forward=False)
    if not res.alive[0]:
        raise ValueError("paraxial ray does not pass the lens (aperture stop too small?)")
    p, v = res.pos3[0], res.dir3[0]
    t_f = -p[0] / v[0]
    z_f = p[2] + t_f * v[2]
    t_p = (h - p[0]) / v[0]
    z_p = p[2] + t_p * v[2]
    return z_p, z_f


def _focus_film_z(lens: Lens, focus_distance_m: float | None) -> tuple[float, float]:
    """Film z and effective focal length, reproducing pbrt's ``RealisticCamera::FocusThickLens``.

    pbrt's lens space has the film at z = 0 and the scene at negative z; ours is shifted by the
    listed film distance ``z0`` (sum of all thicknesses).  pbrt moves the film by ``delta``.
    """
    z0 = lens.listed_film_z
    h = 1e-3 * PBRT_DEFAULT_FILM_DIAGONAL_MM
    zp_img, zf_img = _parallel_ray_cardinal(lens, h, from_film=False)
    zp_obj, _zf_obj = _parallel_ray_cardinal(lens, h, from_film=True)
    f = zf_img - zp_img
    if focus_distance_m is None or not math.isfinite(float(focus_distance_m)):
        return zf_img, f
    pz0, pz1 = zp_img - z0, zp_obj - z0
    z = -1000.0 * float(focus_distance_m)
    c = (pz1 - z - pz0) * (pz1 - z - 4.0 * f - pz0)
    if c <= 0:
        raise ValueError(f"focus distance {focus_distance_m} m is too short for this lens")
    delta = 0.5 * (pz1 - z + pz0 - math.sqrt(c))
    return z0 + delta, f


# ---------------------------------------------------------------------------
# Vectorised 3-D sequential trace
# ---------------------------------------------------------------------------


@dataclass
class TraceResult:
    pos3: np.ndarray  # (N, 3) final positions
    dir3: np.ndarray  # (N, 3) final directions
    alive: np.ndarray  # (N,) bool
    events: list = field(default_factory=list)  # (surface, kind 'R'|'T', n1, n2, beta (N,))

    @property
    def film_xy(self) -> np.ndarray:
        return self.pos3[:, :2]


def primary_path(lens: Lens) -> list[tuple[int, str]]:
    return [(k, "T") for k in range(len(lens.surfaces))]


def ghost_path(lens: Lens, i: int, j: int) -> list[tuple[int, str]]:
    """Reflect at ``j`` (later; ``lens.sensor_index`` = sensor plane) then at ``i < j``, end on the film."""
    if not i < j:
        raise ValueError("ghost path needs i < j")
    n = len(lens.surfaces)
    steps = [(k, "T") for k in range(min(j, n))]
    steps.append((j, "R"))
    steps += [(k, "T") for k in range(min(j, n) - 1, i, -1)]
    steps.append((i, "R"))
    steps += [(k, "T") for k in range(i + 1, n)]
    return steps


def ghost_pairs(lens: Lens) -> list[tuple[int, int]]:
    ref = lens.refracting
    pairs = [(a, b) for ia, a in enumerate(ref) for b in ref[ia + 1 :]]
    if lens.sensor_reflectance > 0.0:
        pairs += [(a, lens.sensor_index) for a in ref]
    return pairs


def _iris_ok(lens: Lens, x: np.ndarray, y: np.ndarray, r: float) -> np.ndarray:
    if lens.iris_blades < 3:
        return x * x + y * y <= r * r
    n = lens.iris_blades
    phi0 = math.radians(lens.iris_rotation_deg) + math.pi / n  # edge normals between vertices
    apothem = r * math.cos(math.pi / n)
    ok = np.ones(x.shape, bool)
    for k in range(n):
        a = phi0 + 2.0 * math.pi * k / n
        ok &= x * math.cos(a) + y * math.sin(a) <= apothem
    return ok


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2]


@dataclass
class _State:
    p: np.ndarray
    v: np.ndarray
    alive: np.ndarray
    events: list
    forward: bool

    @classmethod
    def new(cls, o, d, forward: bool = True) -> _State:
        p = np.array(o, dtype=np.float64, copy=True)
        v = np.array(d, dtype=np.float64, copy=True)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return cls(p, v, np.ones(p.shape[0], bool), [], forward)

    def _to_plane(self, z: float):
        vz = np.where(self.v[:, 2] == 0, 1e-300, self.v[:, 2])
        t = (z - self.p[:, 2]) / vz
        return self.p + t[:, None] * self.v, t > 0

    def step(self, lens: Lens, k: int, kind: str) -> _State:
        """Intersect surface ``k`` and transmit ('T') or reflect ('R'); returns a new state."""
        nsurf = len(lens.surfaces)
        events = list(self.events)
        if k == nsurf:  # flat reflective sensor at the film plane
            p, ok = self._to_plane(lens.z_film)
            cos_i = np.abs(self.v[:, 2])
            beta = np.sqrt(np.clip(1.0 - cos_i**2, 0.0, None))
            events.append((k, "R", 1.0, 1.0, beta))
            v = self.v * np.array([1.0, 1.0, -1.0])
            return _State(p, v, self.alive & ok, events, not self.forward)
        s = lens.surfaces[k]
        if s.is_stop:
            if kind == "R":
                raise ValueError("the aperture stop cannot be a reflecting surface")
            p, ok = self._to_plane(s.z)
            ok &= _iris_ok(lens, p[:, 0], p[:, 1], s.ap_r)
            return _State(p, self.v, self.alive & ok, events, self.forward)
        p0, v = self.p, self.v
        cz = s.z + s.radius
        oc = p0 - np.array([0.0, 0.0, cz])
        b = _dot(oc, v)
        c = _dot(oc, oc) - s.radius * s.radius
        disc = b * b - c
        alive = self.alive & (disc >= 0)
        sq = np.sqrt(np.clip(disc, 0.0, None))
        best = np.full(p0.shape[0], np.inf)
        # The lens surface is the half of the sphere facing the vertex (as pbrt's closer/farther choice).
        for t_c in (-b - sq, -b + sq):
            zc = p0[:, 2] + t_c * v[:, 2] - cz
            ok = (t_c > 1e-9) & (zc * s.radius < 0) & (t_c < best)
            best = np.where(ok, t_c, best)
        alive &= np.isfinite(best)
        best = np.where(np.isfinite(best), best, 0.0)
        p = p0 + best[:, None] * v
        alive &= p[:, 0] ** 2 + p[:, 1] ** 2 <= s.ap_r * s.ap_r
        nrm = (p - np.array([0.0, 0.0, cz])) / s.radius
        nrm = np.where((_dot(v, nrm) > 0)[:, None], -nrm, nrm)  # face the incoming ray
        cos_i = np.clip(-_dot(v, nrm), 0.0, 1.0)
        n1, n2 = (s.n_before, s.n_after) if self.forward else (s.n_after, s.n_before)
        beta = n1 * np.sqrt(np.clip(1.0 - cos_i**2, 0.0, None))
        events.append((k, kind, n1, n2, beta))
        if kind == "R":
            return _State(p, v + 2.0 * cos_i[:, None] * nrm, alive, events, not self.forward)
        eta = n1 / n2
        k2 = 1.0 - eta * eta * (1.0 - cos_i**2)
        alive &= k2 >= 0  # total internal reflection removes the ray from a transmitting step
        vt = eta * v + (eta * cos_i - np.sqrt(np.clip(k2, 0.0, None)))[:, None] * nrm
        vt /= np.linalg.norm(vt, axis=1, keepdims=True)
        return _State(p, vt, alive, events, self.forward)

    def finish(self, lens: Lens, to_film: bool = True) -> TraceResult:
        p, alive = self.p, self.alive
        if to_film:
            p, ok = self._to_plane(lens.z_film)
            alive = alive & ok & (self.v[:, 2] > 0)
        p = np.where(alive[:, None], p, np.nan)
        return TraceResult(p, self.v, alive, self.events)


def trace(lens: Lens, steps, o, d, *, to_film: bool = True, start_forward: bool = True) -> TraceResult:
    """Trace rays ``o + t d`` through ``steps`` [(surface, 'T'|'R')]; optionally end on the film plane."""
    st = _State.new(o, d, start_forward)
    for k, kind in steps:
        st = st.step(lens, k, kind)
    return st.finish(lens, to_film)


def iter_ghost_traces(lens: Lens, o, d):
    """Yield ``((i, j), TraceResult)`` for every two-reflection path, sharing common path prefixes.

    Equivalent to ``trace(lens, ghost_path(lens, i, j), o, d)`` for each pair of :func:`ghost_pairs`.
    """
    nsurf = len(lens.surfaces)
    reflectors = set(lens.refracting)
    js = sorted(reflectors) + ([nsurf] if lens.sensor_reflectance > 0.0 else [])
    fwd = _State.new(o, d)
    k_fwd = 0
    for j in js:
        while k_fwd < min(j, nsurf):
            fwd = fwd.step(lens, k_fwd, "T")
            k_fwd += 1
        if not fwd.alive.any():
            return
        back = fwd.step(lens, j, "R")
        for i in range(min(j, nsurf) - 1, -1, -1):
            if not back.alive.any():
                break
            if i in reflectors:
                br = back.step(lens, i, "R")
                for k in range(i + 1, nsurf):
                    if not br.alive.any():
                        break
                    br = br.step(lens, k, "T")
                yield (i, j), br.finish(lens)
            back = back.step(lens, i, "T")


def path_weights(lens: Lens, res: TraceResult, lams_nm) -> np.ndarray:
    """Per-ray spectral throughput (N, n_lambda): product of R or T over all interface events."""
    lams = tuple(float(x) for x in np.atleast_1d(lams_nm))
    w = np.ones((res.alive.size, len(lams)))
    for k, kind, n1, n2, beta in res.events:
        if k == lens.sensor_index:
            w *= lens.sensor_reflectance
            continue
        r = lookup_reflectance(reflectance_table(n1, n2, lens.surfaces[k].coating, lams), beta)
        w *= r if kind == "R" else (1.0 - r)
    w[~res.alive] = 0.0
    return w


def path_weights_exact(lens: Lens, res: TraceResult, lams_nm) -> np.ndarray:
    """As :func:`path_weights` but evaluating the thin-film formula per ray (slow; for tests)."""
    lams = np.atleast_1d(np.asarray(lams_nm, float))
    w = np.ones((res.alive.size, lams.size))
    for k, kind, n1, n2, beta in res.events:
        if k == lens.sensor_index:
            w *= lens.sensor_reflectance
            continue
        r, t = interface_rt(lams[None, :], beta[:, None], n1, n2, lens.surfaces[k].coating)
        w *= r if kind == "R" else t
    w[~res.alive] = 0.0
    return w


# ---------------------------------------------------------------------------
# Beams, paraxial ghost matrices, Monte-Carlo reference
# ---------------------------------------------------------------------------


def direction_from_angles(theta_rad: float, azimuth_rad: float = 0.0) -> np.ndarray:
    st = math.sin(theta_rad)
    return np.array([st * math.cos(azimuth_rad), st * math.sin(azimuth_rad), math.cos(theta_rad)])


def collimated_beam(lens: Lens, direction, n_grid: int) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Regular ``n_grid x n_grid`` grid of parallel rays covering the front element.

    Returns (origins, directions, cell area on the vertex plane [mm^2], grid (iy, ix) shape).
    """
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    s0 = lens.surfaces[0]
    r = s0.ap_r
    sag = abs(s0.radius) - math.sqrt(max(s0.radius**2 - r * r, 0.0)) if s0.radius else 0.0
    tan_t = math.hypot(d[0], d[1]) / d[2]
    rg = r + sag * tan_t + 1e-6
    u = np.linspace(-rg, rg, n_grid)
    gx, gy = np.meshgrid(u, u)
    z0 = s0.z - sag - 1.0
    p0 = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, s0.z)], axis=1)
    o = p0 + ((z0 - s0.z) / d[2]) * d[None, :]
    dirs = np.broadcast_to(d, o.shape).copy()
    cell = (u[1] - u[0]) ** 2 if n_grid > 1 else math.pi * r * r
    return o, dirs, cell, np.array([n_grid, n_grid])


def paraxial_path_matrix(lens: Lens, steps) -> np.ndarray:
    """2x2 paraxial matrix from the front vertex plane to the film for a step path.

    State (y, omega = n u) with u = dy/dz; signed index n < 0 while travelling toward the object;
    refraction/reflection omega' = omega - y (n' - n) / R with n' = -n on reflection; transfer
    y' = y + (dz / n) omega (Lee & Eisemann 2013 use the equivalent ray-transfer matrices).
    """
    m = np.eye(2)
    z = lens.surfaces[0].z
    n = 1.0
    nsurf = len(lens.surfaces)
    for k, kind in steps:
        zk = lens.z_film if k == nsurf else lens.surfaces[k].z
        m = np.array([[1.0, (zk - z) / n], [0.0, 1.0]]) @ m
        z = zk
        if k == nsurf or lens.surfaces[k].is_stop:
            if kind == "R":
                n = -n
            continue
        s = lens.surfaces[k]
        forward = n > 0
        n_to = (s.n_after if forward else s.n_before) * (1 if forward else -1)
        if kind == "R":
            n_to = -n
        m = np.array([[1.0, 0.0], [-(n_to - n) / s.radius, 1.0]]) @ m
        n = n_to
    m = np.array([[1.0, (lens.z_film - z) / n], [0.0, 1.0]]) @ m
    return m


def paraxial_chief_height(lens: Lens, steps, u0: float) -> float:
    """Paraxial film height of the path's ray through the stop centre for field slope ``u0``."""
    # Chief ray of the incident beam: through the stop centre on the primary (transmitted) path.
    m_stop = paraxial_path_matrix_partial(lens, primary_path(lens)[: lens.stop_index + 1])
    y0 = -m_stop[0, 1] * u0 / m_stop[0, 0]
    m = paraxial_path_matrix(lens, steps)
    return float(m[0, 0] * y0 + m[0, 1] * u0)


def paraxial_path_matrix_partial(lens: Lens, steps) -> np.ndarray:
    """Matrix from the front vertex plane up to the last step's surface (no final transfer)."""
    m = np.eye(2)
    z = lens.surfaces[0].z
    n = 1.0
    for k, kind in steps:
        s = lens.surfaces[k]
        m = np.array([[1.0, (s.z - z) / n], [0.0, 1.0]]) @ m
        z = s.z
        if s.is_stop:
            continue
        forward = n > 0
        n_to = (s.n_after if forward else s.n_before) * (1 if forward else -1)
        if kind == "R":
            n_to = -n
        m = np.array([[1.0, 0.0], [-(n_to - n) / s.radius, 1.0]]) @ m
        n = n_to
    return m


def monte_carlo_nonsequential(lens: Lens, direction, lam_nm: float, n_rays: int, seed: int = 0, max_events: int = 64):
    """Non-sequential Monte-Carlo reference: at every interface reflect with probability R, else refract.

    Independent of the path enumeration and of the weight products (Russian-roulette estimator of the
    same geometric-optics transport).  Rays start uniformly over the front element's clear aperture.
    Returns the film hits ``(x, y, reflection-surface tuple)`` and the energy budget fractions
    (film, back out of the front, vignetted/absorbed, truncated) - these sum to 1.
    """
    rng = np.random.default_rng(seed)
    s0 = lens.surfaces[0]
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    rr = s0.ap_r * np.sqrt(rng.random(n_rays))
    ph = 2.0 * np.pi * rng.random(n_rays)
    p0 = np.stack([rr * np.cos(ph), rr * np.sin(ph), np.full(n_rays, s0.z)], 1)
    sag = abs(s0.radius) - math.sqrt(max(s0.radius**2 - s0.ap_r**2, 0.0)) if s0.radius else 0.0
    o = p0 + ((-sag - 1.0) / d[2]) * d[None, :]
    nsurf = len(lens.surfaces)
    hits = []
    budget = {"film": 0, "front": 0, "absorbed": 0, "truncated": 0}
    for ray in range(n_rays):
        st = _State.new(o[ray : ray + 1], d[None, :])
        idx, refl = 0, []
        for _ in range(max_events):
            if idx < 0:
                budget["front"] += 1
                break
            if idx == nsurf:
                if lens.sensor_reflectance > 0 and rng.random() < lens.sensor_reflectance:
                    st = st.step(lens, nsurf, "R")
                    refl.append(nsurf)
                    idx = nsurf - 1
                    continue
                res = st.finish(lens)
                if res.alive[0]:
                    budget["film"] += 1
                    hits.append((res.pos3[0, 0], res.pos3[0, 1], tuple(refl)))
                else:
                    budget["absorbed"] += 1
                break
            s = lens.surfaces[idx]
            if s.is_stop:
                st = st.step(lens, idx, "T")
                if not st.alive[0]:
                    budget["absorbed"] += 1
                    break
                idx += 1 if st.forward else -1
                continue
            st_r = st.step(lens, idx, "R")
            if not st_r.alive[0]:  # missed the surface or outside its clear aperture
                budget["absorbed"] += 1
                break
            st_t = st.step(lens, idx, "T")
            _k, _kind, n1, n2, beta = st_r.events[-1]
            r_val = 1.0 if not st_t.alive[0] else float(interface_rt(lam_nm, beta, n1, n2, s.coating)[0][0])
            if rng.random() < r_val:
                st = st_r
                refl.append(idx)
            else:
                st = st_t
            idx += 1 if st.forward else -1
        else:
            budget["truncated"] += 1
    return {"hits": hits, "budget": {k: v / n_rays for k, v in budget.items()}, "n_rays": n_rays}


# ---------------------------------------------------------------------------
# Per-direction ghost evaluation
# ---------------------------------------------------------------------------


def ghost_energy_fractions(lens: Lens, direction, lams_nm, n_grid: int = 96) -> dict:
    """Energy of every two-reflection ghost relative to the unit-transmission primary image."""
    o, d, _cell, _shape = collimated_beam(lens, direction, n_grid)
    prim = trace(lens, primary_path(lens), o, d)
    n_p = int(prim.alive.sum())
    out = {"n_primary": n_p, "primary_transmittance": None, "ghosts": {}}
    if n_p == 0:
        return out
    out["primary_transmittance"] = path_weights(lens, prim, lams_nm).sum(0) / n_p
    for (i, j), res in iter_ghost_traces(lens, o, d):
        w = path_weights(lens, res, lams_nm)
        if res.alive.any():
            xy = res.film_xy[res.alive]
            wm = w[res.alive].mean(1)
            cen = (xy * wm[:, None]).sum(0) / max(wm.sum(), 1e-300)
        else:
            cen = np.array([np.nan, np.nan])
        out["ghosts"][(i, j)] = {"energy": w.sum(0) / n_p, "centroid_mm": cen, "n_rays": int(res.alive.sum())}
    return out


# ---------------------------------------------------------------------------
# Camera / film mapping
# ---------------------------------------------------------------------------


class FilmMapping:
    """Pixel <-> lens-film coordinates and source direction for a rendered EXR."""

    def __init__(self, lens: Lens, width: int, height: int, camera: str, film_diagonal_mm=None, fov_deg=None):
        self.lens, self.w, self.h = lens, int(width), int(height)
        self.camera = "realistic" if camera == "realistic" else "pinhole"
        diag = float(film_diagonal_mm or PBRT_DEFAULT_FILM_DIAGONAL_MM)
        aspect = self.h / self.w
        self.xw = math.sqrt(diag * diag / (1.0 + aspect * aspect))
        self.yh = aspect * self.xw
        self.fov = None if fov_deg is None else math.radians(float(fov_deg))
        if self.camera == "pinhole" and self.fov is None:
            raise ValueError("pinhole/perspective mapping needs fov_deg")
        self._build_height_table()

    def _build_height_table(self) -> None:
        thetas, heights = [0.0], [0.0]
        for th in np.radians(np.arange(0.25, 89.0, 0.25)):
            o, d, _c, _s = collimated_beam(self.lens, direction_from_angles(th), 48)
            res = trace(self.lens, primary_path(self.lens), o, d)
            if not res.alive.any():
                break
            x = float(np.mean(res.film_xy[res.alive, 0]))
            if x <= heights[-1]:
                break
            thetas.append(float(th))
            heights.append(x)
        # A beam travelling toward +x (source at -x) images at +x: the image is inverted.
        self.theta_tab = np.array(thetas)
        self.r_tab = np.array(heights)
        if self.r_tab.size < 3:
            raise ValueError("lens passes no off-axis primary rays")

    def theta_for_film_radius(self, r):
        r = np.asarray(r, float)
        th = np.interp(r, self.r_tab, self.theta_tab, right=np.nan)
        return th

    def pixel_to_film(self, px, py):
        """Lens-film (X, Y) [mm] of continuous pixel-centre coordinates."""
        px, py = np.asarray(px, float), np.asarray(py, float)
        if self.camera == "realistic":
            x2 = -0.5 * self.xw + (px + 0.5) / self.w * self.xw
            y2 = -0.5 * self.yh + (py + 0.5) / self.h * self.yh
            return -x2, y2
        s = 0.5 * min(self.w, self.h)
        dx, dy = (px + 0.5 - 0.5 * self.w), (py + 0.5 - 0.5 * self.h)
        rho = np.hypot(dx, dy)
        th = np.arctan(rho / s * math.tan(0.5 * self.fov))
        r = np.interp(th, self.theta_tab, self.r_tab, right=np.nan)
        scale = np.where(rho > 0, r / np.where(rho > 0, rho, 1.0), 0.0)
        return -dx * scale, dy * scale

    def film_to_pixel(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if self.camera == "realistic":
            x2, y2 = -x, y
            px = (x2 + 0.5 * self.xw) / self.xw * self.w - 0.5
            py = (y2 + 0.5 * self.yh) / self.yh * self.h - 0.5
            return px, py
        r = np.hypot(x, y)
        th = self.theta_for_film_radius(r)
        s = 0.5 * min(self.w, self.h)
        rho = s * np.tan(th) / math.tan(0.5 * self.fov)
        scale = np.where(r > 0, rho / np.where(r > 0, r, 1.0), 0.0)
        return 0.5 * self.w - 0.5 - x * scale, 0.5 * self.h - 0.5 + y * scale

    def direction_for_film(self, x: float, y: float) -> np.ndarray | None:
        r = math.hypot(x, y)
        if r == 0.0:
            return np.array([0.0, 0.0, 1.0])
        th = float(self.theta_for_film_radius(r))
        if not math.isfinite(th):
            return None
        # A beam travelling toward +e images at +e (the source itself is at -e: inverted image).
        ex, ey = x / r, y / r
        return np.array([math.sin(th) * ex, math.sin(th) * ey, math.cos(th)])


# ---------------------------------------------------------------------------
# Splatting
# ---------------------------------------------------------------------------


def _bilinear_splat(px, py, weights, h, w) -> np.ndarray:
    """Bilinear splat of points with (N, r) weights into an (r, h, w) array (pixel-centre coords)."""
    out = np.zeros((weights.shape[1], h * w))
    x0 = np.floor(px).astype(np.int64)
    y0 = np.floor(py).astype(np.int64)
    fx, fy = px - x0, py - y0
    for dx, dy, f in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + dx, y0 + dy
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        idx = yi[ok] * w + xi[ok]
        for c in range(weights.shape[1]):
            out[c] += np.bincount(idx, weights=weights[ok, c] * f[ok], minlength=h * w)
    return out.reshape(weights.shape[1], h, w)


# Cap on the smoothing blur [px] for strongly magnified ray-grid cells (caustic neighbourhoods).
MAX_BLUR_PX = 8.0


def _log2_bin(v: np.ndarray) -> np.ndarray:
    """0 for v <= 1, else 1 + floor(log2 v) (groups blur scales within a factor of 2)."""
    v = np.asarray(v, float)
    return np.where(v > 1.0, 1 + np.floor(np.log2(np.maximum(v, 1.0))), 0).astype(int)


def rasterize_ray_grid(px, py, alive, wspec, grid_n: int, max_sub: int = 8, spacing_px: float = 0.7):
    """Spread each grid cell's energy over the bilinear patch of its four traced corners.

    ``px, py`` (N,) pixel coords, ``wspec`` (N, n_lambda) per-ray energy.  Returns a list of
    point sets ``(x, y, spectral_weights, blur_sigma_px)``; total energy equals ``wspec.sum(0)``
    up to rays on the grid border (which never pass a lens and carry no energy).
    """
    n = grid_n
    X, Y = px.reshape(n, n), py.reshape(n, n)
    A = alive.reshape(n, n)
    W = wspec.reshape(n, n, -1)
    a00, a10, a01, a11 = A[:-1, :-1], A[:-1, 1:], A[1:, :-1], A[1:, 1:]
    full = a00 & a10 & a01 & a11
    out = []
    # Partially vignetted cells: each alive corner contributes w/4 as a point.
    corners = (
        (A[:-1, :-1], slice(None, -1), slice(None, -1)),
        (A[:-1, 1:], slice(None, -1), slice(1, None)),
        (A[1:, :-1], slice(1, None), slice(None, -1)),
        (A[1:, 1:], slice(1, None), slice(1, None)),
    )
    edge_scale = 0.0
    if full.any():
        ex = np.maximum(
            np.ptp(np.stack([X[:-1, :-1][full], X[:-1, 1:][full], X[1:, :-1][full], X[1:, 1:][full]], 1), 1),
            np.ptp(np.stack([Y[:-1, :-1][full], Y[:-1, 1:][full], Y[1:, :-1][full], Y[1:, 1:][full]], 1), 1),
        )
        edge_scale = float(np.median(ex))
    if full.any():
        cx = np.stack([X[:-1, :-1][full], X[:-1, 1:][full], X[1:, :-1][full], X[1:, 1:][full]], 1)
        cy = np.stack([Y[:-1, :-1][full], Y[:-1, 1:][full], Y[1:, :-1][full], Y[1:, 1:][full]], 1)
        cw = 0.25 * (W[:-1, :-1][full] + W[:-1, 1:][full] + W[1:, :-1][full] + W[1:, 1:][full])
        ext = np.maximum(np.ptp(cx, 1), np.ptp(cy, 1))
        # Fill each patch at <= spacing_px; cells beyond max_sub get a blur matched to the residual
        # sample spacing (grouped in log2 bins of that spacing).
        sub = np.clip(np.ceil(ext / spacing_px), 1, max_sub).astype(int)
        rbin = np.where(sub == max_sub, _log2_bin(ext / (max_sub * spacing_px)), 0)
        for s, b in sorted(set(zip(sub.tolist(), rbin.tolist()))):
            m = (sub == s) & (rbin == b)
            t = (np.arange(s) + 0.5) / s
            uu, vv = np.meshgrid(t, t)
            uu, vv = uu.ravel()[None, :], vv.ravel()[None, :]
            c = cx[m]
            gx = (
                (1 - uu) * (1 - vv) * c[:, :1]
                + uu * (1 - vv) * c[:, 1:2]
                + (1 - uu) * vv * c[:, 2:3]
                + uu * vv * c[:, 3:]
            )
            c = cy[m]
            gy = (
                (1 - uu) * (1 - vv) * c[:, :1]
                + uu * (1 - vv) * c[:, 1:2]
                + (1 - uu) * vv * c[:, 2:3]
                + uu * vv * c[:, 3:]
            )
            ww = np.repeat(cw[m] / (s * s), s * s, axis=0)
            resid = float(np.median(ext[m])) / s
            sigma = 0.5 * resid if resid > spacing_px else 0.0
            out.append((gx.ravel(), gy.ravel(), ww, sigma))
    part = ~full
    if part.any():
        # Partially vignetted cells: each alive corner carries w/4, spread with the blur scale of the
        # ghost's full cells (a lone corner has no extent of its own; a point splat would leave a
        # one-pixel spike at the vignetting edge).
        sig_p = min(0.5 * edge_scale, MAX_BLUR_PX) if edge_scale > 1.0 else 0.0
        for a, sy, sx in corners:
            m = part & a
            if m.any():
                out.append((X[sy, sx][m], Y[sy, sx][m], 0.25 * W[sy, sx][m], sig_p))
    return out


def _low_rank(wspec: np.ndarray, tol: float = 1e-4, max_rank: int = 4):
    """Truncated SVD (N, L) ~ U (N, r) @ S (r, L) with relative Frobenius error <= tol where possible."""
    if wspec.shape[0] == 0:
        return np.zeros((0, 0)), np.zeros((0, wspec.shape[1]))
    u, s, vt = np.linalg.svd(wspec, full_matrices=False)
    tot = float((s * s).sum())
    if tot == 0:
        return u[:, :0], vt[:0]
    resid = 1.0 - np.cumsum(s * s) / tot
    r = int(min(max_rank, np.searchsorted(-resid, -tol * tol) + 1, s.size))
    return u[:, :r] * s[:r], vt[:r]


# ---------------------------------------------------------------------------
# Image application
# ---------------------------------------------------------------------------


DEFAULTS = {
    "enabled": False,
    "lens_file": None,
    "aperture_diameter_mm": None,
    "focus_distance_m": None,
    "camera": None,
    "film_diagonal_mm": None,
    "fov_deg": None,
    "coating": "mgf2",
    "coating_design_wavelength_nm": DEFAULT_DESIGN_NM,
    "surface_coatings": {},
    "sensor_reflectance": 0.0,
    "iris_blades": 0,
    "iris_rotation_deg": 0.0,
    "pupil_samples": 64,
    "source_threshold_relative": 0.01,
    "source_threshold_abs": 0.0,
    "cluster_px": 8,
    "max_sources": 256,
    "min_ghost_fraction": 1e-7,
    "apply_primary_transmittance": False,
}


def resolve_config(ghost_cfg: dict | None, lens_cfg: dict | None = None) -> dict:
    """Merge ``lens.traced_ghosts`` with defaults; fall back to the lens model's realistic-camera keys."""
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in (ghost_cfg or {}).items() if v is not None})
    lens_cfg = lens_cfg or {}
    fallbacks = {
        "lens_file": lens_cfg.get("realistic_lensfile"),
        "aperture_diameter_mm": lens_cfg.get("realistic_aperture_diameter_mm"),
        "focus_distance_m": lens_cfg.get("realistic_focus_distance"),
        "camera": lens_cfg.get("camera"),
        "fov_deg": lens_cfg.get("pinhole_fov_deg"),
    }
    for k, v in fallbacks.items():
        if cfg.get(k) is None and v is not None:
            cfg[k] = v
    if cfg.get("camera") is None:
        cfg["camera"] = "realistic"
    return cfg


def lens_from_config(cfg: dict, repo: Path) -> Lens:
    lf = cfg.get("lens_file")
    if not lf:
        raise ValueError("traced_ghosts needs lens_file (or lens.realistic_lensfile in the lens model)")
    path = Path(lf)
    if not path.is_absolute():
        path = (repo / path).resolve()
    return load_lens(
        path,
        aperture_diameter_mm=cfg.get("aperture_diameter_mm"),
        focus_distance_m=cfg.get("focus_distance_m"),
        coating=cfg.get("coating", "mgf2"),
        surface_coatings=cfg.get("surface_coatings") or {},
        design_nm=float(cfg.get("coating_design_wavelength_nm", DEFAULT_DESIGN_NM)),
        iris_blades=int(cfg.get("iris_blades", 0)),
        iris_rotation_deg=float(cfg.get("iris_rotation_deg", 0.0)),
        sensor_reflectance=float(cfg.get("sensor_reflectance", 0.0)),
    )


def _channel_wavelengths(names) -> list[float]:
    from exr_multispectral import parse_s0_wavelength_nm  # noqa: PLC0415

    lams = []
    for n in names:
        lam = parse_s0_wavelength_nm(n)
        if lam is None:
            lam = _RGB_CENTER_NM.get(n.upper(), 550.0)
        lams.append(float(lam))
    return lams


def find_sources(cube: np.ndarray, spectral_mask: np.ndarray, cfg: dict) -> tuple[list[dict], dict]:
    """Bright-source cells: pixels above threshold summed per ``cluster_px`` block."""
    h, w, _ = cube.shape
    sig = cube[:, :, spectral_mask].sum(2) if spectral_mask.any() else cube.sum(2)
    peak = float(sig.max()) if sig.size else 0.0
    thr = max(float(cfg["source_threshold_abs"]), float(cfg["source_threshold_relative"]) * peak)
    if peak <= 0 or thr <= 0:
        return [], {"threshold": thr, "n_pixels": 0, "kept_flux_fraction": 1.0}
    mask = sig >= thr
    ys, xs = np.nonzero(mask)
    cp = max(1, int(cfg["cluster_px"]))
    cell = (ys // cp) * ((w + cp - 1) // cp) + xs // cp
    order = np.argsort(cell, kind="stable")
    ys, xs, cell = ys[order], xs[order], cell[order]
    starts = np.flatnonzero(np.r_[True, cell[1:] != cell[:-1]])
    ends = np.r_[starts[1:], cell.size]
    sources = []
    for a, b in zip(starts, ends):
        yy, xx = ys[a:b], xs[a:b]
        flux = cube[yy, xx, :].sum(0).astype(np.float64)
        wsig = sig[yy, xx].astype(np.float64)
        tot = float(wsig.sum())
        sources.append(
            {
                "x": float((xx * wsig).sum() / tot),
                "y": float((yy * wsig).sum() / tot),
                "flux": flux,
                "signal": tot,
            }
        )
    sources.sort(key=lambda s: -s["signal"])
    total = sum(s["signal"] for s in sources)
    kept = sources[: int(cfg["max_sources"])]
    info = {
        "threshold": thr,
        "n_pixels": int(mask.sum()),
        "n_cells": len(sources),
        "n_sources": len(kept),
        "kept_flux_fraction": (sum(s["signal"] for s in kept) / total) if total > 0 else 1.0,
    }
    return kept, info


def _gauss_blur_local(maps: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter  # noqa: PLC0415

    return np.stack([gaussian_filter(m, sigma, mode="constant", truncate=3.0) for m in maps])


def _splat_ghost(ghost: np.ndarray, gx, gy, alive, wsp: np.ndarray, n_grid: int) -> None:
    """Rasterise one ghost's ray grid into ``ghost`` (H, W, L) in place.

    The per-ray spectra are factored once, ``wsp ~ coeff @ spectra`` (rank <= 4, since they vary only
    through the angle dependence of the coatings); rasterisation is linear, so only the rank-r
    coefficient maps are splatted and expanded to the L buckets at the end.
    """
    h, w, _nl = ghost.shape
    coeff, spectra = _low_rank(wsp)
    if coeff.shape[1] == 0:
        return
    sets = rasterize_ray_grid(gx, gy, alive, coeff, n_grid)
    sets = [st for st in sets if st[0].size]
    if not sets:
        return
    pad = int(math.ceil(3 * min(max(st[3] for st in sets), MAX_BLUR_PX))) + 2
    x0 = max(int(math.floor(min(st[0].min() for st in sets))) - pad, 0)
    x1 = min(int(math.ceil(max(st[0].max() for st in sets))) + pad + 1, w)
    y0 = max(int(math.floor(min(st[1].min() for st in sets))) - pad, 0)
    y1 = min(int(math.ceil(max(st[1].max() for st in sets))) + pad + 1, h)
    if x1 <= x0 or y1 <= y0:
        return
    # One blur per distinct (capped, rounded) sigma: point sets sharing a blur are splatted together.
    by_sigma: dict[float, np.ndarray] = {}
    for x, y, ws, sigma in sets:
        key = round(min(sigma, MAX_BLUR_PX), 1)
        m = _bilinear_splat(x - x0, y - y0, ws, y1 - y0, x1 - x0)
        by_sigma[key] = by_sigma[key] + m if key in by_sigma else m
    maps = sum(_gauss_blur_local(m, k) if k > 0 else m for k, m in by_sigma.items())
    flat = maps.reshape(maps.shape[0], -1).T @ spectra
    ghost[y0:y1, x0:x1, :] += flat.reshape(y1 - y0, x1 - x0, -1)


def render_ghosts(cube: np.ndarray, lams: list[float], spectral_mask: np.ndarray, lens: Lens, cfg: dict):
    """Return (ghost cube (H, W, L), primary transmittance map or None, report dict)."""
    t0 = time.time()
    h, w, nl = cube.shape
    mapping = FilmMapping(lens, w, h, str(cfg["camera"]), cfg.get("film_diagonal_mm"), cfg.get("fov_deg"))
    sources, info = find_sources(cube, spectral_mask, cfg)
    ghost = np.zeros((h, w, nl), np.float64)
    pairs = ghost_pairs(lens)
    n_grid = int(cfg["pupil_samples"])
    min_frac = float(cfg["min_ghost_fraction"])
    lam_t = tuple(lams)
    ghost_energy = np.zeros(nl)
    src_energy = np.zeros(nl)
    skipped = 0
    per_pair = {p: 0.0 for p in pairs}
    for src in sources:
        fx, fy = mapping.pixel_to_film(src["x"], src["y"])
        d = mapping.direction_for_film(float(fx), float(fy))
        if d is None:
            skipped += 1
            continue
        o, dirs, _cell, _shape = collimated_beam(lens, d, n_grid)
        prim = trace(lens, primary_path(lens), o, dirs)
        n_p = int(prim.alive.sum())
        if n_p == 0:
            skipped += 1
            continue
        scale = src["flux"] / n_p
        src_energy += src["flux"]
        for pair, res in iter_ghost_traces(lens, o, dirs):
            if not res.alive.any():
                continue
            wsp = path_weights(lens, res, lam_t)
            frac = float(wsp.sum(0).max()) / n_p
            if frac < min_frac:
                continue
            wsp = wsp * scale[None, :]
            gx, gy = mapping.film_to_pixel(res.film_xy[:, 0], res.film_xy[:, 1])
            alive = res.alive & np.isfinite(gx) & np.isfinite(gy)
            gx, gy = np.where(alive, gx, 0.0), np.where(alive, gy, 0.0)
            wsp[~alive] = 0.0
            _splat_ghost(ghost, gx, gy, alive, wsp, n_grid)
            e = wsp.sum(0)
            ghost_energy += e
            per_pair[pair] += float(e[spectral_mask].sum() if spectral_mask.any() else e.sum())
    trans_map = None
    if bool(cfg.get("apply_primary_transmittance", False)):
        trans_map = primary_transmittance_map(lens, mapping, lams)
    top = sorted(per_pair.items(), key=lambda kv: -kv[1])[:10]
    sel = spectral_mask if spectral_mask.any() else np.ones(nl, bool)
    report = {
        "sources": info,
        "sources_outside_lens_field": skipped,
        "n_ghost_paths": len(pairs),
        "ghost_to_source_energy": float(ghost_energy[sel].sum() / max(src_energy[sel].sum(), 1e-300)),
        "ghost_energy_in_frame": float(ghost[:, :, sel].sum() / max(ghost_energy[sel].sum(), 1e-300)),
        "brightest_ghosts": [
            {"surfaces": list(p), "energy_fraction_of_sources": v / max(src_energy[sel].sum(), 1e-300)}
            for p, v in top
            if v > 0
        ],
        "film_z_mm": lens.z_film,
        "efl_mm": lens.efl_mm,
        "runtime_s": time.time() - t0,
    }
    return ghost.astype(np.float32), trans_map, report


def primary_transmittance_map(lens: Lens, mapping: FilmMapping, lams) -> np.ndarray:
    """Field-dependent primary-path transmittance (H, W, L) from the coating model."""
    th_tab = mapping.theta_tab
    tab = []
    for th in th_tab:
        o, d, _c, _s = collimated_beam(lens, direction_from_angles(float(th)), 48)
        res = trace(lens, primary_path(lens), o, d)
        n = max(int(res.alive.sum()), 1)
        tab.append(path_weights(lens, res, tuple(lams)).sum(0) / n)
    tab = np.array(tab)
    yy, xx = np.mgrid[0 : mapping.h, 0 : mapping.w]
    fx, fy = mapping.pixel_to_film(xx, yy)
    th = np.nan_to_num(mapping.theta_for_film_radius(np.hypot(fx, fy)), nan=float(th_tab[-1]))
    return np.stack([np.interp(th, th_tab, tab[:, k]) for k in range(tab.shape[1])], 2).astype(np.float32)


def apply_traced_ghosts(chans: dict[str, np.ndarray], cfg: dict, repo: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Add traced ghosts to every channel of a SpectralFilm EXR channel dict."""
    from exr_multispectral import parse_s0_wavelength_nm  # noqa: PLC0415

    names = [n for n in chans if np.asarray(chans[n]).ndim == 2]
    lams = _channel_wavelengths(names)
    spectral_mask = np.array([parse_s0_wavelength_nm(n) is not None for n in names])
    cube = np.stack([np.asarray(chans[n], np.float32) for n in names], 2)
    lens = lens_from_config(cfg, repo)
    ghost, trans, report = render_ghosts(cube, lams, spectral_mask, lens, cfg)
    base = cube * trans if trans is not None else cube
    out_cube = base + ghost
    out = dict(chans)
    for k, n in enumerate(names):
        out[n] = out_cube[:, :, k].astype(np.float32)
    report["coating"] = cfg.get("coating")
    report["camera"] = cfg.get("camera")
    report["apply_primary_transmittance"] = trans is not None
    return out, report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import yaml  # noqa: PLC0415
    from camera_model import load_camera_model  # noqa: PLC0415
    from exr_multispectral import read_separate_exr_channels, write_separate_channels_exr  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    repo_default = Path(__file__).resolve().parent.parent
    ap.add_argument("--repo-root", type=Path, default=repo_default)
    ap.add_argument("--exr-in", type=Path, required=True)
    ap.add_argument("--exr-out", type=Path, default=None, help="default: overwrite --exr-in")
    ap.add_argument("--camera-model-config", type=Path, default=None, help="reads lens.traced_ghosts")
    ap.add_argument("--config", type=Path, default=None, help="YAML with a traced_ghosts block")
    ap.add_argument("--scene-manifest-json", type=Path, default=None, help="highway manifest: camera settings")
    ap.add_argument("--enable", action="store_true", help="force traced_ghosts.enabled = true")
    ap.add_argument("--lens-file", default=None)
    ap.add_argument("--coating", default=None, help="uncoated | mgf2 | qhq")
    ap.add_argument("--camera", default=None, help="realistic | pinhole (perspective/thinlens)")
    ap.add_argument("--aperture-diameter-mm", type=float, default=None)
    ap.add_argument("--focus-distance-m", type=float, default=None)
    ap.add_argument("--film-diagonal-mm", type=float, default=None)
    ap.add_argument("--fov-deg", type=float, default=None)
    ap.add_argument("--iris-blades", type=int, default=None)
    ap.add_argument("--sensor-reflectance", type=float, default=None)
    ap.add_argument("--source-threshold-relative", type=float, default=None)
    ap.add_argument("--pupil-samples", type=int, default=None)
    ap.add_argument("--report-json", type=Path, default=None)
    args = ap.parse_args()
    repo = args.repo_root.resolve()

    ghost_cfg: dict = {}
    lens_cfg: dict = {}
    if args.camera_model_config is not None:
        cm = load_camera_model(args.camera_model_config.resolve())
        lens_cfg = cm.get("lens", {}) or {}
        ghost_cfg = dict(lens_cfg.get("traced_ghosts", {}) or {})
        stray = ((lens_cfg.get("post_psf") or {}).get("stray_light") or {}).get("ghost_reflections") or {}
        if bool(stray.get("enabled", False)) and bool(ghost_cfg.get("enabled", False)):
            print(
                "warning: post_psf.stray_light.ghost_reflections (parametric) and traced_ghosts are both "
                "enabled; ghosts are counted twice",
                file=sys.stderr,
            )
    if args.config is not None:
        doc = yaml.safe_load(args.config.read_text()) or {}
        doc_lens = doc.get("lens", {}) if isinstance(doc.get("lens"), dict) else {}
        lens_cfg = {**lens_cfg, **doc_lens}
        ghost_cfg.update(doc.get("traced_ghosts") or doc_lens.get("traced_ghosts") or {})
    if args.scene_manifest_json is not None:
        cam = json.loads(args.scene_manifest_json.read_text()).get("camera") or {}
        kind = cam.get("type")
        lens_cfg.setdefault("camera", "realistic" if kind == "realistic" else "pinhole")
        if cam.get("fov_deg") is not None:
            lens_cfg.setdefault("pinhole_fov_deg", cam["fov_deg"])
        if kind == "realistic":
            lens_cfg.setdefault("realistic_lensfile", cam.get("lensfile"))
            lens_cfg.setdefault("realistic_aperture_diameter_mm", cam.get("aperture_diameter_mm"))
            lens_cfg.setdefault("realistic_focus_distance", cam.get("focus_distance"))
            ghost_cfg.setdefault("film_diagonal_mm", cam.get("film_diagonal_mm"))
    overrides = {
        "lens_file": args.lens_file,
        "coating": args.coating,
        "camera": args.camera,
        "aperture_diameter_mm": args.aperture_diameter_mm,
        "focus_distance_m": args.focus_distance_m,
        "film_diagonal_mm": args.film_diagonal_mm,
        "fov_deg": args.fov_deg,
        "iris_blades": args.iris_blades,
        "sensor_reflectance": args.sensor_reflectance,
        "source_threshold_relative": args.source_threshold_relative,
        "pupil_samples": args.pupil_samples,
    }
    ghost_cfg.update({k: v for k, v in overrides.items() if v is not None})
    if args.enable:
        ghost_cfg["enabled"] = True
    cfg = resolve_config(ghost_cfg, lens_cfg)
    if not bool(cfg.get("enabled", False)):
        print("traced_ghosts.enabled is false; nothing to do.", file=sys.stderr)
        sys.exit(0)

    exr_in = args.exr_in if args.exr_in.is_absolute() else (repo / args.exr_in).resolve()
    exr_out = (args.exr_out or exr_in).resolve()
    chans = read_separate_exr_channels(exr_in)
    out, report = apply_traced_ghosts(chans, cfg, repo)
    exr_out.parent.mkdir(parents=True, exist_ok=True)
    write_separate_channels_exr(exr_out, out)
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2, default=float))
    print(
        f"wrote {exr_out} (traced ghosts: {report['n_ghost_paths']} paths, "
        f"{report['sources'].get('n_sources', 0)} sources, coating={cfg.get('coating')}, "
        f"ghost/source energy={report['ghost_to_source_energy']:.3e}, {report['runtime_s']:.1f} s)"
    )


if __name__ == "__main__":
    main()
