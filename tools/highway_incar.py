#!/usr/bin/env python3
"""In-car (windscreen-mounted ADAS) camera effects for the highway scene.

Opt-in pieces used by tools/build_highway_scene.py (all off by default):

* **Windscreen**: a raked, optionally curved laminated-glass shell (glass / PVB / glass) a few
  centimetres in front of the lens, as a ``dielectric`` with an absorbing ``homogeneous``
  medium inside (green-tinted, IR-attenuating iron-bearing glass), optional dirt film and
  raindrops. Attached to the ego car, so it moves with the camera.
* **Exposure / motion**: pbrt shutter + ``TransformTimes``/``ActiveTransform`` animated
  transforms for ego motion (speed, yaw rate) and per-car velocities. The shutter is the
  sensor integration time recorded in the manifest (``exposure.integration_time_s``), which
  tools/pbrt_spectral_exr_to_electrons.py uses as its default integration time.
* **Rolling shutter / LED flicker**: the manifest records the line time and every
  PWM-driven emitter; tools/render_time_slices.py renders row bands x PWM on/off intervals
  and composites them. Emitters follow a generic contract (:func:`write_emitter`), so any
  emissive object (e.g. vehicle lights) can be tagged for flicker.

World units are metres, y up, the road along +z, +x image-right (as in the builder). The
"car frame" is the world frame translated to the camera position (no pitch).
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from colour_science import cmf_on_grid
from highway_spectra import reflectance

KMH = 1.0 / 3.6

# ---------------------------------------------------------------------------------------------
# Windscreen glass optics
# ---------------------------------------------------------------------------------------------
# Laminated windscreen: 2.1 mm float glass / 0.76 mm PVB / 2.1 mm float glass (the common
# automotive build-up, e.g. Pilkington/AGC/Saint-Gobain product literature). Refractive indices:
# soda-lime glass n_d ~ 1.52, PVB ~ 1.48 (index matched within 0.04, so the laminate is
# treated as one dielectric with the glass index; the PVB/glass interface reflection is ~2e-4).
GLASS_ETA = 1.52
LAMINATE_MM = (2.1, 0.76, 2.1)

# Absorption of green-tinted (iron-bearing) soda-lime glass. Iron gives the colour and the IR
# attenuation (Bamford, "Colour Generation and Control in Glass", Elsevier 1977, ch. 4;
# Volotinen et al., J. Non-Cryst. Solids 354 (2008) 4084): Fe2+ has a broad band centred at
# ~1050 nm (~9500 cm^-1) whose short-wavelength wing absorbs the red/NIR, Fe3+ a charge-transfer
# edge in the near UV. Parameters below are a smooth fit giving the typical behaviour of a
# green solar-control windscreen: luminous transmittance (CIE illuminant A, ISO 9050) ~0.80,
# above the 70 % legal minimum (UN ECE R43 / ANSI Z26.1 / FMVSS 205), ~0.45 at 800 nm and
# ~0.2 at 1000 nm. The PVB interlayer carries a UV absorber (cut-off ~380 nm).
FE2_PEAK_CM1, FE2_SIGMA_CM1 = 9500.0, 2100.0


def glass_absorption_per_m(
    wl_nm: np.ndarray, fe2_peak_per_m: float = 390.0, fe3_uv_per_m: float = 60.0, base_per_m: float = 30.0
) -> np.ndarray:
    """Napierian absorption coefficient [1/m] of green-tinted float glass (see module notes)."""
    wl = np.asarray(wl_nm, dtype=np.float64)
    nu = 1.0e7 / wl
    fe2 = fe2_peak_per_m * np.exp(-0.5 * ((nu - FE2_PEAK_CM1) / FE2_SIGMA_CM1) ** 2)
    fe3 = fe3_uv_per_m * np.exp(-(wl - 380.0) / 30.0)
    return base_per_m + fe2 + fe3


def pvb_absorption_per_m(wl_nm: np.ndarray, cutoff_nm: float = 385.0) -> np.ndarray:
    """UV-absorbing PVB interlayer: internal transmittance ~0.995 in the visible, ~0 below 375 nm."""
    wl = np.asarray(wl_nm, dtype=np.float64)
    t = 0.995 / (1.0 + np.exp(-(wl - cutoff_nm) / 3.5))
    return -np.log(np.clip(t, 1e-12, 1.0)) / (LAMINATE_MM[1] * 1e-3)


def laminate_absorption_per_m(wl_nm: np.ndarray, laminate_mm: tuple[float, float, float] = LAMINATE_MM) -> np.ndarray:
    """Thickness-weighted absorption of the whole laminate, applied as one homogeneous medium."""
    g, p, g2 = laminate_mm
    d = (g + p + g2) * 1e-3
    a = (glass_absorption_per_m(wl_nm) * (g + g2) + pvb_absorption_per_m(wl_nm) * p) * 1e-3 / d
    return np.minimum(a, 1.0e5)


def fresnel_unpolarised(cos_i: float, eta: float = GLASS_ETA) -> tuple[float, float]:
    """(Rs, Rp) of an air/dielectric interface."""
    sin_t = math.sqrt(max(0.0, 1.0 - cos_i * cos_i)) / eta
    cos_t = math.sqrt(max(0.0, 1.0 - sin_t * sin_t))
    rs = ((cos_i - eta * cos_t) / (cos_i + eta * cos_t)) ** 2
    rp = ((eta * cos_i - cos_t) / (eta * cos_i + cos_t)) ** 2
    return rs, rp


def windscreen_transmittance(
    wl_nm: np.ndarray, incidence_deg: float = 0.0, thickness_m: float | None = None, alpha_per_m=None
) -> np.ndarray:
    """Direct (regular) transmittance of the slab: Fresnel at two faces, incoherent multiple
    reflections per polarisation, internal absorption along the refracted path."""
    wl = np.asarray(wl_nm, dtype=np.float64)
    d = sum(LAMINATE_MM) * 1e-3 if thickness_m is None else float(thickness_m)
    a = laminate_absorption_per_m(wl) if alpha_per_m is None else np.asarray(alpha_per_m, dtype=np.float64)
    cos_i = math.cos(math.radians(incidence_deg))
    cos_t = math.sqrt(1.0 - (math.sin(math.radians(incidence_deg)) / GLASS_ETA) ** 2)
    ti = np.exp(-a * d / cos_t)
    out = np.zeros_like(wl)
    for r in fresnel_unpolarised(cos_i):
        out += 0.5 * (1 - r) ** 2 * ti / (1 - (r * ti) ** 2)
    return out


def luminous_transmittance(wl_nm: np.ndarray, t: np.ndarray, cct_k: float = 2856.0) -> float:
    """ISO 9050 / ECE R43 luminous transmittance: CIE illuminant A (Planck 2856 K) x V(lambda)."""
    wl = np.asarray(wl_nm, dtype=np.float64)
    lam = wl * 1e-9
    planck = 1.0 / (lam**5 * (np.exp(1.4388e-2 / (lam * cct_k)) - 1.0))
    w = planck * cmf_on_grid(wl)[1]
    return float(np.trapezoid(w * t, wl) / np.trapezoid(w, wl))


def alpha_from_transmittance_csv(path: Path, wl_nm: np.ndarray, thickness_m: float) -> np.ndarray:
    """Absorption [1/m] reproducing a measured normal-incidence total transmittance T(lambda)."""
    data = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    t = np.interp(wl_nm, data[:, 0], data[:, 1])
    r = fresnel_unpolarised(1.0)[0]
    t_int = np.clip(t * (1 - r * r) / (1 - r) ** 2, 1e-6, 1.0)  # single-pass approximation
    return -np.log(t_int) / thickness_m


# ---------------------------------------------------------------------------------------------
# Windscreen geometry (car frame: origin at the lens, x right, y up, z forward)
# ---------------------------------------------------------------------------------------------
@dataclass
class Windscreen:
    rake_deg: float = 27.0  # glass angle from horizontal (modern cars ~25-30 deg)
    axis_distance_m: float = 0.07  # lens -> inner surface along the forward axis
    thickness_m: float = sum(LAMINATE_MM) * 1e-3
    width_m: float = 1.4
    below_m: float = 0.7  # slope length below/ahead of the axis point
    above_m: float = 0.15  # slope length above/behind it (towards the roof)
    radius_h_m: float = 0.0  # horizontal radius of curvature (0 = flat)
    radius_v_m: float = 0.0  # vertical radius of curvature (0 = flat)

    @property
    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(P0, ex, es, n): axis point, lateral and up-slope tangents, outward normal."""
        r = math.radians(self.rake_deg)
        return (
            np.array([0.0, 0.0, self.axis_distance_m]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, math.sin(r), -math.cos(r)]),
            np.array([0.0, math.cos(r), math.sin(r)]),
        )

    def surface(self, u: np.ndarray, s: np.ndarray, offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Points and outward unit normals of the inner surface (+ offset along the normal)."""
        p0, ex, es, n = self.basis
        u, s = np.asarray(u, dtype=np.float64)[..., None], np.asarray(s, dtype=np.float64)[..., None]
        kh = 1.0 / self.radius_h_m if self.radius_h_m > 0 else 0.0
        kv = 1.0 / self.radius_v_m if self.radius_v_m > 0 else 0.0
        sag = 0.5 * kh * u**2 + 0.5 * kv * s**2
        p = p0 + u * ex + s * es - sag * n
        du = ex - kh * u * n
        ds = es - kv * s * n
        nn = np.cross(du, ds)
        nn /= np.linalg.norm(nn, axis=-1, keepdims=True)
        return p + offset * nn, nn

    def grid(self, nu: int = 41, ns: int = 25) -> tuple[np.ndarray, np.ndarray]:
        return np.linspace(-self.width_m / 2, self.width_m / 2, nu), np.linspace(-self.below_m, self.above_m, ns)

    def shell(self, nu: int = 41, ns: int = 25) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Closed glass shell as (P, tri, N) parts with outward normals: inner, outer, 4 edges."""
        us, ss = self.grid(nu, ns)
        U, S = np.meshgrid(us, ss, indexing="ij")
        pi, ni = self.surface(U, S)
        po, no = self.surface(U, S, self.thickness_m)
        parts = [_grid_mesh(pi, -ni), _grid_mesh(po, no)]
        # Edge strips: boundary of the inner and outer grids, outward in-plane direction.
        for sl, sign, along in ((np.s_[0, :], -1, 0), (np.s_[-1, :], 1, 0), (np.s_[:, 0], -1, 1), (np.s_[:, -1], 1, 1)):
            a, b = pi[sl], po[sl]
            t = np.gradient(pi, axis=along)[sl]
            t = sign * t / np.linalg.norm(t, axis=-1, keepdims=True)
            parts.append(_grid_mesh(np.stack([a, b], axis=1), np.stack([t, t], axis=1)))
        return parts

    def footprint(self, dirs: np.ndarray, margin_m: float) -> tuple[float, float, float, float]:
        """(u0, u1, s0, s1) region of the flat inner plane hit by rays from the lens (+ margin)."""
        p0, ex, es, n = self.basis
        t = (p0 @ n) / (dirs @ n)
        hit = dirs * t[:, None] - p0
        u, s = hit @ ex, hit @ es
        ok = t > 0
        u0, u1 = float(u[ok].min()) - margin_m, float(u[ok].max()) + margin_m
        s0, s1 = float(s[ok].min()) - margin_m, float(s[ok].max()) + margin_m
        w2 = self.width_m / 2
        return max(u0, -w2), min(u1, w2), max(s0, -self.below_m), min(s1, self.above_m)


def _grid_mesh(p: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate an (a, b, 3) vertex grid; triangles wound to agree with the vertex normals."""
    a, b = p.shape[:2]
    idx = np.arange(a * b).reshape(a, b)
    q = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
    tri = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    return orient(p.reshape(-1, 3), tri, n.reshape(-1, 3))


def orient(p: np.ndarray, tri: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fn = np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])
    flip = np.einsum("ij,ij->i", fn, n[tri].sum(1)) < 0
    tri = tri.copy()
    tri[flip] = tri[flip][:, [0, 2, 1]]
    return p, tri, n


def raindrops(
    ws: Windscreen,
    region: tuple[float, float, float, float],
    coverage: float,
    rng: np.random.Generator,
    contact_angle_deg: float = 45.0,
    median_radius_mm: float = 0.8,
) -> list[tuple[np.ndarray, float]]:
    """Non-overlapping drops (centre (u, s), base radius [m]) covering ``coverage`` of ``region``.

    Base radii are log-normal (median 0.8 mm, sigma_ln 0.5, clipped 0.15-3 mm): the range of
    sessile rain drops on glass (e.g. Garg & Nayar, IJCV 75 (2007) 3). Water on clean glass has a
    contact angle of ~20-50 deg; hydrophobic coatings exceed 90 deg.
    """
    u0, u1, s0, s1 = region
    area = max(u1 - u0, 0.0) * max(s1 - s0, 0.0)
    target = coverage * area
    drops: list[tuple[np.ndarray, float]] = []
    covered, tries = 0.0, 0
    while covered < target and tries < 200_000:
        tries += 1
        a = float(np.clip(median_radius_mm * math.exp(0.5 * rng.standard_normal()), 0.15, 3.0)) * 1e-3
        c = np.array([rng.uniform(u0, u1), rng.uniform(s0, s1)])
        if any(np.hypot(*(c - c2)) < a + a2 for c2, a2 in drops):
            continue
        drops.append((c, a))
        covered += math.pi * a * a
    return drops


def drop_cap_mesh(
    ws: Windscreen, drops: list[tuple[np.ndarray, float]], contact_angle_deg: float, n_ring: int = 6, n_az: int = 16
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spherical-cap water surfaces sitting on the outer glass surface (one merged mesh)."""
    th = math.radians(contact_angle_deg)
    ps, ts, ns = [], [], []
    base = 0
    for (u, s), a in drops:
        o, nrm = ws.surface(np.array(u), np.array(s), ws.thickness_m)
        t1 = np.cross(nrm, [1.0, 0.0, 0.0])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(nrm, t1)
        r = a / math.sin(th)
        c = o - nrm * r * math.cos(th)
        phis = np.linspace(0, th, n_ring + 1)[1:]
        psis = np.linspace(0, 2 * math.pi, n_az, endpoint=False)
        dirs = [nrm]
        for ph in phis:
            for ps_ in psis:
                dirs.append(math.sin(ph) * (math.cos(ps_) * t1 + math.sin(ps_) * t2) + math.cos(ph) * nrm)
        dirs = np.array(dirs)
        tri = [(0, 1 + i, 1 + (i + 1) % n_az) for i in range(n_az)]
        for k in range(n_ring - 1):
            r0, r1 = 1 + k * n_az, 1 + (k + 1) * n_az
            for i in range(n_az):
                j = (i + 1) % n_az
                tri += [(r0 + i, r1 + i, r1 + j), (r0 + i, r1 + j, r0 + j)]
        ps.append(c + r * dirs)
        ns.append(dirs)
        ts.append(np.array(tri) + base)
        base += len(dirs)
    if not ps:
        return np.zeros((0, 3)), np.zeros((0, 3), int), np.zeros((0, 3))
    return orient(np.concatenate(ps), np.concatenate(ts), np.concatenate(ns))


def dirt_mask(path: Path, coverage: float, seed: int, size: int = 1024) -> None:
    """Tileable dirt coverage map (8-bit, linear): dust specks + a low-frequency grime film.

    Mean value = ``coverage``; pbrt uses it as a stochastic alpha so that fraction of rays hit
    the scattering dirt layer.
    """
    from PIL import Image
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    film = gaussian_filter(rng.standard_normal((size, size)), 40, mode="wrap")
    film = np.clip(0.5 + film / (4 * film.std() + 1e-12), 0, 1)
    specks = np.zeros((size, size))
    yy, xx = np.mgrid[-4:5, -4:5]
    n = int(coverage * size * size / 12.0)
    for y, x, r in zip(rng.integers(0, size, n), rng.integers(0, size, n), rng.uniform(0.8, 3.5, n)):
        disc = (xx**2 + yy**2 <= r * r).astype(float)
        specks[np.ix_((y + np.arange(-4, 5)) % size, (x + np.arange(-4, 5)) % size)] += disc
    m = 0.5 * np.clip(specks, 0, 1) + 0.5 * film * coverage * 2
    m *= coverage / max(float(m.mean()), 1e-9)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(m * 255 + 0.5, 0, 255).astype(np.uint8)).save(path)


# ---------------------------------------------------------------------------------------------
# PWM-driven emitters and the generic emitter contract
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PWM:
    """On for ``duty / frequency`` at the start of each period, starting at ``phase_s``."""

    frequency_hz: float
    duty: float
    phase_s: float = 0.0

    def on_intervals(self, t0: float, t1: float) -> list[tuple[float, float]]:
        if self.frequency_hz <= 0 or self.duty >= 1.0:
            return [(t0, t1)]
        if self.duty <= 0.0:
            return []
        per = 1.0 / self.frequency_hz
        k = math.floor((t0 - self.phase_s) / per)
        out = []
        while True:
            a = self.phase_s + k * per
            if a >= t1:
                break
            lo, hi = max(a, t0), min(a + self.duty * per, t1)
            if hi > lo:
                out.append((lo, hi))
            k += 1
        return out

    def on_fraction(self, t0: float, t1: float) -> float:
        if t1 <= t0:
            return float(self.is_on(t0))
        return sum(b - a for a, b in self.on_intervals(t0, t1)) / (t1 - t0)

    def is_on(self, t: float) -> bool:
        if self.frequency_hz <= 0 or self.duty >= 1.0:
            return True
        return ((t - self.phase_s) * self.frequency_hz) % 1.0 < self.duty

    def edges(self, t0: float, t1: float) -> list[float]:
        return sorted({x for iv in self.on_intervals(t0, t1) for x in iv if t0 < x < t1})

    def as_dict(self) -> dict:
        return {"frequency_hz": self.frequency_hz, "duty": self.duty, "phase_s": self.phase_s}


def write_emitter(
    out_dir: Path,
    eid: str,
    lines_for_level,
    *,
    pwm: PWM | None,
    exposure_window_s: tuple[float, float] | None,
    meta: dict | None = None,
) -> tuple[str, dict]:
    """Write a tagged emitter and return its ``Include`` line and manifest entry.

    ``lines_for_level(level)`` returns the pbrt lines of the object with its emission scaled by
    ``level`` (1 = PWM "on" peak, 0 = off: keep the geometry, drop the ``AreaLightSource``).
    Three files are written under ``emitters/``: ``<id>.pbrt`` (included by the scene: the
    frame-averaged level, i.e. the on-fraction over ``exposure_window_s`` or the duty cycle when
    there is no exposure), ``<id>.on.pbrt`` and ``<id>.off.pbrt`` (swapped in per time slice by
    tools/render_time_slices.py). The manifest entry goes into ``manifest["emitters"]``.
    """
    d = out_dir / "emitters"
    d.mkdir(parents=True, exist_ok=True)
    if pwm is None:
        level = 1.0
    elif exposure_window_s is None:
        level = min(max(pwm.duty, 0.0), 1.0) if pwm.frequency_hz > 0 else 1.0
    else:
        level = pwm.on_fraction(*exposure_window_s)
    files = {"default": f"emitters/{eid}.pbrt", "on": f"emitters/{eid}.on.pbrt", "off": f"emitters/{eid}.off.pbrt"}
    for key, lv in (("default", level), ("on", 1.0), ("off", 0.0)):
        (out_dir / files[key]).write_text("\n".join(lines_for_level(lv)) + "\n")
    entry = {
        "id": eid,
        "include": files["default"],
        "states": {"on": files["on"], "off": files["off"]},
        "default_level": level,
        "pwm": pwm.as_dict() if pwm else None,
        **(meta or {}),
    }
    return f'Include "{files["default"]}"', entry


def led_spectrum(wl_nm: np.ndarray, peak_nm: float = 592.0, fwhm_nm: float = 17.0) -> np.ndarray:
    """AlInGaP amber LED: near-Gaussian line. The thermal linewidth is ~1.8 kT in energy
    (Schubert, "Light-Emitting Diodes", 2nd ed., CUP 2006, ch. 5), ~13 nm at 590 nm and 300 K;
    alloy broadening gives the 15-20 nm FWHM of commercial amber LEDs."""
    s = fwhm_nm / (2 * math.sqrt(2 * math.log(2)))
    return np.exp(-0.5 * ((np.asarray(wl_nm, dtype=np.float64) - peak_nm) / s) ** 2)


# Classic 5x7 dot-matrix font (column bytes, bit 0 = top row).
FONT5X7 = {
    " ": (0, 0, 0, 0, 0), "!": (0, 0, 0x5F, 0, 0), "-": (8, 8, 8, 8, 8), ".": (0, 0x60, 0x60, 0, 0),
    "0": (0x3E, 0x51, 0x49, 0x45, 0x3E), "1": (0, 0x42, 0x7F, 0x40, 0), "2": (0x42, 0x61, 0x51, 0x49, 0x46),
    "3": (0x21, 0x41, 0x45, 0x4B, 0x31), "4": (0x18, 0x14, 0x12, 0x7F, 0x10), "5": (0x27, 0x45, 0x45, 0x45, 0x39),
    "6": (0x3C, 0x4A, 0x49, 0x49, 0x30), "7": (0x01, 0x71, 0x09, 0x05, 0x03), "8": (0x36, 0x49, 0x49, 0x49, 0x36),
    "9": (0x06, 0x49, 0x49, 0x29, 0x1E), "A": (0x7E, 0x11, 0x11, 0x11, 0x7E), "B": (0x7F, 0x49, 0x49, 0x49, 0x36),
    "C": (0x3E, 0x41, 0x41, 0x41, 0x22), "D": (0x7F, 0x41, 0x41, 0x22, 0x1C), "E": (0x7F, 0x49, 0x49, 0x49, 0x41),
    "F": (0x7F, 0x09, 0x09, 0x09, 0x01), "G": (0x3E, 0x41, 0x49, 0x49, 0x7A), "H": (0x7F, 8, 8, 8, 0x7F),
    "I": (0, 0x41, 0x7F, 0x41, 0), "J": (0x20, 0x40, 0x41, 0x3F, 0x01), "K": (0x7F, 0x08, 0x14, 0x22, 0x41),
    "L": (0x7F, 0x40, 0x40, 0x40, 0x40), "M": (0x7F, 0x02, 0x0C, 0x02, 0x7F), "N": (0x7F, 0x04, 0x08, 0x10, 0x7F),
    "O": (0x3E, 0x41, 0x41, 0x41, 0x3E), "P": (0x7F, 0x09, 0x09, 0x09, 0x06), "Q": (0x3E, 0x41, 0x51, 0x21, 0x5E),
    "R": (0x7F, 0x09, 0x19, 0x29, 0x46), "S": (0x46, 0x49, 0x49, 0x49, 0x31), "T": (1, 1, 0x7F, 1, 1),
    "U": (0x3F, 0x40, 0x40, 0x40, 0x3F), "V": (0x1F, 0x20, 0x40, 0x20, 0x1F), "W": (0x3F, 0x40, 0x38, 0x40, 0x3F),
    "X": (0x63, 0x14, 0x08, 0x14, 0x63), "Y": (0x07, 0x08, 0x70, 0x08, 0x07), "Z": (0x61, 0x51, 0x49, 0x45, 0x43),
}  # fmt: skip


def dot_matrix(lines: list[str]) -> np.ndarray:
    """Boolean LED matrix (rows top-down) for centred text lines, 1 dot between chars, 2 between lines."""
    width = max(len(s) for s in lines) * 6 - 1
    rows = []
    for k, text in enumerate(lines):
        m = np.zeros((7, width), bool)
        x0 = (width - (len(text) * 6 - 1)) // 2
        for i, ch in enumerate(text.upper()):
            if ch not in FONT5X7:
                raise ValueError(f"VMS text: unsupported character {ch!r}")
            for c, col in enumerate(FONT5X7[ch]):
                for r in range(7):
                    m[r, x0 + 6 * i + c] = bool(col >> r & 1)
        if k:
            rows.append(np.zeros((2, width), bool))
        rows.append(m)
    return np.concatenate(rows)


# ---------------------------------------------------------------------------------------------
# pbrt text helpers (local copies: the builder imports this module)
# ---------------------------------------------------------------------------------------------
def _f(x: float) -> str:
    return f"{x:.7g}"


def _pts(a) -> str:
    return " ".join(_f(v) for v in np.asarray(a, dtype=np.float64).ravel())


def mesh_lines(p: np.ndarray, tri: np.ndarray, n: np.ndarray | None = None, uv: np.ndarray | None = None) -> list[str]:
    out = ['Shape "trianglemesh"', f'    "point3 P" [ {_pts(p)} ]', f'    "integer indices" [ {_pts(tri)} ]']
    if n is not None:
        out.append(f'    "normal N" [ {_pts(n)} ]')
    if uv is not None:
        out.append(f'    "point2 uv" [ {_pts(uv)} ]')
    return out


def write_spd(path: Path, wl: np.ndarray, val: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # pbrt's float parser rejects denormal-range values: flush anything below 1e-30 to zero.
    path.write_text("\n".join(f"{w:.1f} {v if abs(v) > 1e-30 else 0.0:.6g}" for w, v in zip(wl, val)) + "\n")


def rot_y(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# ---------------------------------------------------------------------------------------------
# CLI options and the builder hook
# ---------------------------------------------------------------------------------------------
LANE_SPEEDS_KMH = {0: 125.0, 1: 110.0, 2: 95.0}  # own carriageway, fast lane = 0
ONCOMING_SPEED_KMH = 105.0


def add_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("in-car camera effects (tools/highway_incar.py; all off by default)")
    g.add_argument("--exposure-s", type=float, default=None, help="Integration time = pbrt shutter (enables motion).")
    g.add_argument("--rolling-shutter-line-time-us", type=float, default=0.0, help="Row readout time; 0 = global.")
    g.add_argument("--ego-speed-kmh", type=float, default=0.0)
    g.add_argument("--ego-yaw-rate-deg-s", type=float, default=0.0, help="+ = turning right.")
    g.add_argument(
        "--traffic-speed-kmh",
        default="0",
        help='"lanes" (125/110/95 km/h by lane, oncoming 105), one value for all cars, or a comma list per car.',
    )
    g.add_argument("--windscreen", action="store_true", help="Laminated glass in front of the lens (volpath).")
    g.add_argument("--windscreen-rake-deg", type=float, default=27.0)
    g.add_argument(
        "--windscreen-distance-m", type=float, default=None, help="Camera to glass along the axis (default: auto)."
    )
    g.add_argument("--windscreen-radius-h-m", type=float, default=0.0, help="Horizontal curvature radius; 0 = flat.")
    g.add_argument("--windscreen-radius-v-m", type=float, default=0.0, help="Vertical curvature radius; 0 = flat.")
    g.add_argument("--windscreen-transmittance-csv", type=Path, default=None, help="Measured T(lambda) (nm,T).")
    g.add_argument("--windscreen-dirt", type=float, default=0.0, help="Dirt coverage fraction (e.g. 0.05).")
    g.add_argument("--windscreen-rain", type=float, default=0.0, help="Raindrop coverage fraction (e.g. 0.2).")
    g.add_argument("--rain-contact-angle-deg", type=float, default=45.0)
    g.add_argument("--windscreen-seed", type=int, default=1)
    g.add_argument("--windscreen-outside-medium", default="", help="Medium outside the glass (e.g. a haze medium).")
    g.add_argument("--vms", action="store_true", help="Roadside LED variable-message sign (PWM, flicker-tagged).")
    g.add_argument("--vms-text", default="QUEUE AHEAD|SLOW DOWN", help="Lines separated by |.")
    g.add_argument("--vms-distance-m", type=float, default=45.0)
    g.add_argument("--vms-luminance-cd-m2", type=float, default=6000.0, help="Time-averaged face luminance.")
    g.add_argument("--vms-pwm-hz", type=float, default=100.0)
    g.add_argument("--vms-duty", type=float, default=0.25)
    g.add_argument("--vms-phase-ms", type=float, default=0.0)


class InCarEffects:
    """Small hooks for build_highway_scene.py; every method is a no-op when its option is off."""

    def __init__(self, args: argparse.Namespace, repo: Path):
        self.a = args
        self.repo = Path(repo)
        self.t_int = args.exposure_s
        if self.t_int is not None and self.t_int <= 0:
            raise ValueError("--exposure-s must be > 0")
        self.line_time = args.rolling_shutter_line_time_us * 1e-6
        if self.line_time and self.t_int is None:
            raise ValueError("--rolling-shutter-line-time-us needs --exposure-s")
        self.span = None if self.t_int is None else self.t_int + max(args.yres - 1, 0) * self.line_time
        self.ego_v = args.ego_speed_kmh * KMH
        self.yaw_rate = args.ego_yaw_rate_deg_s
        self.windscreen = (
            Windscreen(
                rake_deg=args.windscreen_rake_deg,
                axis_distance_m=args.windscreen_distance_m or self.auto_windscreen_distance_m(),
                radius_h_m=args.windscreen_radius_h_m,
                radius_v_m=args.windscreen_radius_v_m,
            )
            if args.windscreen
            else None
        )
        if not args.windscreen and (args.windscreen_dirt or args.windscreen_rain):
            raise ValueError("--windscreen-dirt/--windscreen-rain need --windscreen")
        self.emitters: list[dict] = []
        self.meta: dict = {}
        self.warnings: list[str] = []

    def auto_windscreen_distance_m(self) -> float:
        """Axis distance that keeps the raked glass ~10 mm clear of the front lens element.

        pbrt's realistic camera puts the film at the camera origin and the front element at
        ~(back focal distance + lens length) ahead; the paraxial focal length stands in for the
        back focal distance."""
        a = self.a
        if a.camera != "realistic":
            return 0.05 + a.thinlens_lens_radius / math.tan(math.radians(a.windscreen_rake_deg))
        from lens_prescription import load_lens_file, paraxial_focal_lengths_mm

        rows = load_lens_file(self.repo / a.lensfile)
        front_mm = sum(r[1] for r in rows) + paraxial_focal_lengths_mm(rows)[0]
        r_mm = rows[0][3] / 2
        return (front_mm + r_mm / math.tan(math.radians(a.windscreen_rake_deg)) + 10.0) * 1e-3

    @property
    def animated(self) -> bool:
        return self.span is not None

    def integrator(self, default: str) -> str:
        return "volpath" if self.windscreen is not None else default

    def film_lines(self) -> list[str]:
        """pbrt-v4 scales film values by exposure time x ISO/100 (PixelSensor imagingRatio);
        ISO = 100 / exposure keeps the EXR in radiance (irradiance) units for any shutter."""
        return [f'    "float iso" [{_f(100.0 / self.t_int)}]'] if self.animated else []

    def shutter_params(self) -> str:
        if not self.animated:
            return ""
        return f' "float shutteropen" [0] "float shutterclose" [{_f(self.t_int)}]'

    def _ego_end(self, eye: np.ndarray) -> tuple[np.ndarray, float]:
        yaw = self.yaw_rate * self.span
        return eye + self.ego_v * self.span * (rot_y(0.5 * yaw) @ np.array([0.0, 0.0, 1.0])), yaw

    def camera_transform_lines(self, eye: np.ndarray, target: np.ndarray) -> list[str]:
        look = f"LookAt {_pts(eye)}  {_pts(target)}  0 1 0"
        if not self.animated:
            return [look]
        eye1, yaw = self._ego_end(eye)
        target1 = eye1 + rot_y(yaw) @ (target - eye)
        return [
            f"TransformTimes 0 {_f(self.span)}",
            "ActiveTransform StartTime",
            look,
            "ActiveTransform EndTime",
            f"LookAt {_pts(eye1)}  {_pts(target1)}  0 1 0",
            "ActiveTransform All",
        ]

    def car_speeds_kmh(self, lanes: list[int]) -> list[float]:
        spec = str(self.a.traffic_speed_kmh).strip()
        if spec == "lanes":
            return [LANE_SPEEDS_KMH.get(ln, 100.0) if ln >= 0 else ONCOMING_SPEED_KMH for ln in lanes]
        vals = [float(v) for v in spec.split(",")]
        if len(vals) == 1:
            return vals * len(lanes)
        if len(vals) != len(lanes):
            raise ValueError(f"--traffic-speed-kmh: {len(vals)} values for {len(lanes)} cars")
        return vals

    def car_lines(self, inst: str, lane: int, speed_kmh: float, include_text: str) -> tuple[list[str], dict]:
        """Lines to put right after the car's AttributeBegin (before its placement transforms)."""
        vz = speed_kmh * KMH * (1.0 if lane >= 0 else -1.0)
        meta = {"speed_kmh": speed_kmh, "velocity_world_mps": [0.0, 0.0, vz]}
        if not self.animated or vz == 0.0:
            return [], meta
        if "AreaLightSource" in include_text:
            # pbrt-v4 has no animated area lights ("Animated area lights are not supported").
            self.warnings.append(f"{inst}: has area lights, rendered without motion (pbrt limitation)")
            meta["velocity_world_mps"] = [0.0, 0.0, 0.0]
            return [], meta
        return ["ActiveTransform EndTime", f"Translate 0 0 {_f(vz * self.span)}", "ActiveTransform All"], meta

    def camera_dirs(self, n: int = 9) -> np.ndarray:
        """Unit ray directions (car frame) spanning the image with a 15 % margin."""
        a = self.a
        aspect = a.xres / a.yres
        if a.camera == "realistic":
            from lens_prescription import load_lens_file, paraxial_focal_lengths_mm

            f_mm = paraxial_focal_lengths_mm(load_lens_file(self.repo / a.lensfile))[0]
            h = a.film_diagonal_mm / math.hypot(aspect, 1.0)
            tan_v = 0.5 * h / f_mm
        else:
            tan_v = math.tan(math.radians(a.fov) / 2) * (1.0 if aspect >= 1 else 1.0 / aspect)
        tan_h = tan_v * aspect
        g = np.linspace(-1.15, 1.15, n)
        X, Y = np.meshgrid(g * tan_h, g * tan_v)
        d = np.stack([X.ravel(), Y.ravel(), np.ones(X.size)], -1)
        c, s = math.cos(math.radians(a.cam_pitch)), math.sin(math.radians(a.cam_pitch))
        d = d @ np.array([[1, 0, 0], [0, c, s], [0, -s, c]])  # pitch about x (negative = down)
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def windscreen_lines(self, out_dir: Path, wl: np.ndarray, eye: np.ndarray) -> list[str]:
        ws = self.windscreen
        if ws is None:
            return []
        a = self.a
        if a.windscreen_transmittance_csv is not None:
            alpha = alpha_from_transmittance_csv(a.windscreen_transmittance_csv, wl, ws.thickness_m)
            src = str(a.windscreen_transmittance_csv)
        else:
            alpha = laminate_absorption_per_m(wl)
            src = "analytic green laminate (tools/highway_incar.py)"
        write_spd(out_dir / "spd" / "windscreen_sigma_a.spd", wl, alpha)
        t0 = windscreen_transmittance(wl, 0.0, ws.thickness_m, alpha)
        p0, _, _, n = ws.basis
        pitch = math.radians(a.cam_pitch)
        axis = np.array([0.0, math.sin(pitch), math.cos(pitch)])
        axis_inc = math.degrees(math.acos(abs(float(axis @ n))))
        lines = [
            "# Windscreen (tools/highway_incar.py): laminated glass attached to the ego car",
            'MakeNamedMedium "windscreen_laminate" "string type" "homogeneous"',
            '    "spectrum sigma_a" "spd/windscreen_sigma_a.spd" "spectrum sigma_s" [300 0 900 0] "float scale" [1]',
            f'MakeNamedMaterial "windscreen_glass" "string type" "dielectric" "float eta" [{_f(GLASS_ETA)}]',
            'MakeNamedMaterial "windscreen_water" "string type" "dielectric" "float eta" [1.333]',
            "AttributeBegin",
            *self._ego_attach_lines(eye),
            "AttributeBegin",
            f'MediumInterface "windscreen_laminate" "{a.windscreen_outside_medium}"',
            'NamedMaterial "windscreen_glass"',
        ]
        for p, t, nn in ws.shell():
            lines += mesh_lines(p, t, nn)
        lines.append("AttributeEnd")
        rng = np.random.default_rng(a.windscreen_seed)
        meta = {
            "rake_deg": ws.rake_deg,
            "axis_distance_m": ws.axis_distance_m,
            "thickness_m": ws.thickness_m,
            "laminate_mm": list(LAMINATE_MM),
            "eta": GLASS_ETA,
            "radius_h_m": ws.radius_h_m,
            "radius_v_m": ws.radius_v_m,
            "absorption": src,
            "luminous_transmittance_normal": luminous_transmittance(wl, t0),
            "optical_axis_incidence_deg": axis_inc,
            "luminous_transmittance_axis": luminous_transmittance(
                wl, windscreen_transmittance(wl, axis_inc, ws.thickness_m, alpha)
            ),
        }
        if a.windscreen_dirt > 0:
            tile = 0.2
            mask = out_dir / "textures" / f"windscreen_dirt_{a.windscreen_seed}_{a.windscreen_dirt:g}.png"
            dirt_mask(mask, a.windscreen_dirt, a.windscreen_seed)
            # Road dust: soil-like reflectance; ~40 % diffuse forward transmission (veiling glare).
            write_spd(out_dir / "spd" / "windscreen_dirt.spd", wl, 0.8 * reflectance("soil", wl))
            us, ss = ws.grid(41, 25)
            U, S = np.meshgrid(us, ss, indexing="ij")
            p, nn = ws.surface(U, S, ws.thickness_m + 5e-5)
            p, t, nn = _grid_mesh(p, nn)
            uv = np.stack([U.ravel(), S.ravel()], -1) / tile
            lines += [
                f'Texture "windscreen_dirt_mask" "float" "imagemap" "string filename" "{os.path.relpath(mask, out_dir)}"',
                '    "string encoding" "linear"',
                'MakeNamedMaterial "windscreen_dirt" "string type" "diffusetransmission"',
                '    "spectrum reflectance" "spd/windscreen_dirt.spd" "spectrum transmittance" [300 0.4 900 0.4]',
                'NamedMaterial "windscreen_dirt"',
                *mesh_lines(p, t, nn, uv)[:-1],
                f'    "point2 uv" [ {_pts(uv)} ] "texture alpha" "windscreen_dirt_mask"',
            ]
            meta["dirt"] = {"coverage": a.windscreen_dirt, "mask": os.path.relpath(mask, out_dir), "tile_m": tile}
        if a.windscreen_rain > 0:
            margin = 0.005 + (a.aperture_diameter_mm * 1e-3 if a.camera == "realistic" else 0.0)
            margin += a.thinlens_lens_radius if a.camera == "thinlens" else 0.0
            region = ws.footprint(self.camera_dirs(), margin)
            drops = raindrops(ws, region, a.windscreen_rain, rng, a.rain_contact_angle_deg)
            p, t, nn = drop_cap_mesh(ws, drops, a.rain_contact_angle_deg)
            if len(drops):
                lines += ['NamedMaterial "windscreen_water"', *mesh_lines(p, t, nn)]
            meta["rain"] = {
                "coverage": a.windscreen_rain,
                "drops": len(drops),
                "contact_angle_deg": a.rain_contact_angle_deg,
                "region_uv_m": list(region),
            }
        lines += ["AttributeEnd", ""]
        self.meta["windscreen"] = meta
        return lines

    def _ego_attach_lines(self, eye: np.ndarray) -> list[str]:
        if not self.animated:
            return [f"Translate {_pts(eye)}"]
        eye1, yaw = self._ego_end(eye)
        return [
            "ActiveTransform StartTime",
            f"Translate {_pts(eye)}",
            "ActiveTransform EndTime",
            f"Translate {_pts(eye1)}",
            f"Rotate {_f(yaw)} 0 1 0",
            "ActiveTransform All",
        ]

    def vms_lines(self, out_dir: Path, wl: np.ndarray, roadside_x: float) -> list[str]:
        a = self.a
        if not a.vms:
            return []
        write_spd(out_dir / "spd" / "led_amber.spd", wl, led_spectrum(wl))
        dots = dot_matrix([s for s in a.vms_text.split("|") if s])
        pitch, dot = 0.04, 0.024  # m; 24 mm LED clusters on a 40 mm pitch
        rows, cols = dots.shape
        w, h = cols * pitch + 0.3, rows * pitch + 0.3
        cx, cz, y0 = roadside_x + 0.5 + w / 2, float(a.vms_distance_m), 2.2
        face_z = cz - 0.15
        fill = dot * dot / (pitch * pitch)
        # EN 12966 luminance is the time-averaged luminance over the character area with all
        # pixels lit, so each LED dot runs at L / (fill factor * duty) while on.
        duty = a.vms_duty if a.vms_pwm_hz > 0 else 1.0
        peak = a.vms_luminance_cd_m2 / (fill * max(duty, 1e-6))
        ys, xs = np.nonzero(dots)
        px = cx - (cols - 1) * pitch / 2 + xs * pitch
        py = y0 + h - 0.15 - pitch / 2 - ys * pitch
        q = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)]) * dot / 2
        P = np.concatenate([np.stack([px + dx, py + dy, np.full_like(px, face_z - 0.005)], -1) for dx, dy in q], 0)
        P = P.reshape(4, -1, 3).transpose(1, 0, 2).reshape(-1, 3)
        k = np.arange(len(px))[:, None] * 4
        tri = np.concatenate([k + [0, 2, 1], k + [0, 3, 2]])
        N = np.tile([0.0, 0.0, -1.0], (len(P), 1))
        P, tri, N = orient(P, tri, N)
        housing = [
            (cx - w / 2, y0, face_z),
            (cx + w / 2, y0, face_z),
            (cx + w / 2, y0 + h, face_z),
            (cx - w / 2, y0 + h, face_z),
        ]

        def lines_for_level(level: float) -> list[str]:
            out = ["AttributeBegin", 'NamedMaterial "sheet_black"']
            out += mesh_lines(np.array(housing), np.array([(0, 1, 2), (0, 2, 3)]), np.tile([0, 0, -1.0], (4, 1)))
            out += ['NamedMaterial "galvanized"']
            for sx in (cx - w * 0.35, cx + w * 0.35):
                bx = np.array([sx - 0.07, sx + 0.07])
                out += mesh_lines(
                    np.array([(bx[0], 0, cz), (bx[1], 0, cz), (bx[1], y0, cz), (bx[0], y0, cz)]),
                    np.array([(0, 1, 2), (0, 2, 3)]),
                    np.tile([0, 0, -1.0], (4, 1)),
                )
            out.append(
                'MakeNamedMaterial "vms_led_lens" "string type" "diffuse" "spectrum reflectance" [300 0.05 900 0.05]'
            )
            out.append('NamedMaterial "vms_led_lens"')
            if level > 0:
                out.append(
                    f'AreaLightSource "diffuse" "spectrum L" "spd/led_amber.spd" "float scale" [{_f(peak * level)}]'
                )
            out += mesh_lines(P, tri, N)
            out.append("AttributeEnd")
            return out

        pwm = PWM(a.vms_pwm_hz, a.vms_duty, a.vms_phase_ms * 1e-3) if a.vms_pwm_hz > 0 else None
        window = (0.0, self.t_int) if self.animated else None
        inc, entry = write_emitter(
            out_dir,
            "vms0",
            lines_for_level,
            pwm=pwm,
            exposure_window_s=window,
            meta={
                "kind": "led_vms",
                "text": a.vms_text,
                "luminance_cd_m2_time_averaged": a.vms_luminance_cd_m2,
                "led_peak_luminance_cd_m2": peak,
                "spectrum": "spd/led_amber.spd",
                "centre_world": [cx, y0 + h / 2, face_z],
                "size_m": [w, h],
            },
        )
        self.emitters.append(entry)
        return [f"# LED variable-message sign at {cz:.0f} m (PWM, flicker-tagged)", inc, ""]

    def manifest(self) -> dict:
        out: dict = dict(self.meta)
        if self.animated:
            out["exposure"] = {
                "integration_time_s": self.t_int,
                "shutter_open_s": 0.0,
                "rolling_shutter_line_time_s": self.line_time,
                "readout_rows": self.a.yres,
                "frame_time_span_s": self.span,
                "row_window": "row r integrates [r * line_time, r * line_time + integration_time]",
            }
            out["ego_motion"] = {
                "speed_kmh": self.a.ego_speed_kmh,
                "yaw_rate_deg_s": self.yaw_rate,
                "velocity_world_mps": [0.0, 0.0, self.ego_v],
            }
        if self.emitters:
            out["emitters"] = self.emitters
        if self.warnings:
            out["incar_warnings"] = self.warnings
        return out
