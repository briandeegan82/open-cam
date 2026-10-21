#!/usr/bin/env python3
"""Build a spectral pbrt-v4 road-tunnel scene as an HDR test (black hole / white hole).

A two-lane, one-way road tunnel bored through a hillside: concrete headwall portals at both
ends, a lined tube (vertical walls, elliptical vault, Lambertian asphalt with lane markings)
and two rows of LED line luminaires. Daylight outside is the highway scene's absolute
clear-sky model (spectral sun + analytic CIE clear sky). Inside, the luminaires are solved
so that the *direct* road luminance follows a stepped zone profile: threshold
(``L_th = k * L20``, then 0.5 L_th), transition (0.15 / 0.05 / 0.02 L_th, never below the
interior level), interior, and exit (3x then 5x the interior level). The zones follow the CIE 88
definitions and the step ratios follow the Chinese 2014 highway-tunnel guideline, both as
summarised in the sources listed in docs/TUNNEL_SCENES.md (``--zone-profile two-step`` gives
the earlier threshold -> interior step).

Two views (``--view``), each written as its own pbrt file and manifest:

* ``outside_in``: camera on the approach road in daylight looking into the entrance portal
  (black-hole effect: bright exterior vs dark tube, a car in the threshold zone);
* ``inside_out``: camera in the interior zone looking at the exit portal through the exit zone
  (white-hole effect: lit interior vs bright exit, a car silhouetted against the exit).

World units are metres, y up, the road runs along +z. Each manifest holds raster ROIs (pinhole
projection; realistic-lens runs re-derive them from the ``world_box`` with a gbuffer render),
predicted luminances [cd/m^2] and the scene dynamic range in dB. Score the renders with
tools/run_tunnel_hdr_test.py.

    venv/bin/python tools/build_tunnel_scene.py --out-dir scenes/generated/tunnel
    (cd scenes/generated/tunnel && mkdir -p out && ../../../third_party/pbrt-v4/build/pbrt tunnel_outside_in.pbrt)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import road_rtable as rt  # noqa: E402
import tunnel_interreflection as ti  # noqa: E402
from build_highway_scene import (  # noqa: E402
    DASH,
    GAP,
    LANE_W,
    _f,
    _pts,
    add_camera_args,
    add_film_args,
    camera_lines,
    extrude_profile,
    film_lines,
    mesh,
    proxy_car,
    quads_mesh,
    sun_world,
    write_spd,
)
from colour_science import cmf_on_grid  # noqa: E402
from highway_night import LAMP_SPECTRA, LIGHT_WL, _quad_light  # noqa: E402
from highway_sky import (  # noqa: E402
    analytic_clear_sky,
    clear_sky_illuminance_lux,
    solar_direct_spectrum,
    write_rgb_exr,
)
from highway_spectra import SURFACES, reflectance  # noqa: E402
from pbrt_spectral_exr_to_electrons import PBRT_CIE_Y_INTEGRAL  # noqa: E402

VIEWS = ("outside_in", "inside_out")
N_LANES = 2
ROAD_W = N_LANES * LANE_W
WALL_GAP = 1.25  # walkway / kerb between the outer lane edge and the wall [m]
WALL_X = (-WALL_GAP, ROAD_W + WALL_GAP)
WALL_H = 4.8  # vertical wall height before the vault springs [m]
CROWN_H = 7.0  # tunnel crown height [m]
HEADWALL_HALF_W = 20.0
HEADWALL_H = 12.0
HILL_RISE = 25.0  # extra hill height well inside the hillside [m]
HILL_HALF_W = 250.0
LAMP_SIZE = (0.12, 1.5)  # LED line luminaire: width across, length along the road [m]
LAMP_Y = 5.4
LAMP_X = (-0.45, ROAD_W + 0.45)  # one row above each kerb
GROUND_FAR = 1500.0
CAR_PAINT = "carpaint_gray"


def tube_profile(n_arch: int = 24) -> list[tuple[float, float]]:
    """(x, y) cross-section from the left wall foot over the vault to the right wall foot."""
    cx, a, b = 0.5 * (WALL_X[0] + WALL_X[1]), 0.5 * (WALL_X[1] - WALL_X[0]), CROWN_H - WALL_H
    pts = [(WALL_X[0], 0.0), (WALL_X[0], WALL_H)]
    for t in np.linspace(math.pi, 0.0, n_arch + 1)[1:-1]:
        pts.append((cx + a * math.cos(t), WALL_H + b * math.sin(t)))
    return pts + [(WALL_X[1], WALL_H), (WALL_X[1], 0.0)]


def _ray_polyline(c: np.ndarray, d: np.ndarray, poly: list[tuple[float, float]]) -> float:
    best = math.inf
    for p, q in zip(poly[:-1], poly[1:]):
        p, q = np.asarray(p), np.asarray(q)
        e = q - p
        den = d[0] * (-e[1]) - d[1] * (-e[0])
        if abs(den) < 1e-12:
            continue
        w = p - c
        t = (w[0] * (-e[1]) - w[1] * (-e[0])) / den
        s = (d[0] * w[1] - d[1] * w[0]) / den
        if t > 1e-9 and -1e-9 <= s <= 1 + 1e-9:
            best = min(best, t)
    return best


def headwall(z: float) -> tuple[np.ndarray, np.ndarray]:
    """Rectangular headwall at ``z`` with the tube cross-section cut out (polar strip from the road centre)."""
    prof = tube_profile()
    cx = 0.5 * (WALL_X[0] + WALL_X[1])
    c = np.array([cx, 0.0])
    rect = [
        (cx + HEADWALL_HALF_W, 0.0),
        (cx + HEADWALL_HALF_W, HEADWALL_H),
        (cx - HEADWALL_HALF_W, HEADWALL_H),
        (cx - HEADWALL_HALF_W, 0.0),
    ]
    corner = math.atan2(HEADWALL_H, HEADWALL_HALF_W)
    thetas = np.unique(np.concatenate([np.linspace(0.0, math.pi, 97), [corner, math.pi - corner]]))
    p, t = [], []
    for i, th in enumerate(thetas):
        d = np.array([math.cos(th), math.sin(th)])
        rh, ro = _ray_polyline(c, d, prof), _ray_polyline(c, d, rect)
        for r in (rh, ro):
            q = c + r * d
            p.append((q[0], max(q[1], 0.0), z))
        if i:
            a, b, cc, dd = 2 * i - 2, 2 * i - 1, 2 * i, 2 * i + 1
            t += [(a, b, dd), (a, dd, cc)]
    return np.array(p), np.array(t)


def hill_lateral(x: np.ndarray) -> np.ndarray:
    cx = 0.5 * (WALL_X[0] + WALL_X[1])
    u = np.clip((np.abs(x - cx) - HEADWALL_HALF_W) / (HILL_HALF_W - HEADWALL_HALF_W), 0.0, 1.0)
    return 0.05 + 0.95 * 0.5 * (1.0 + np.cos(math.pi * u))


def hill_mesh(z0: float, z1: float) -> tuple[np.ndarray, np.ndarray]:
    """Hillside above the tube (z0..z1) plus vertical soil faces beside each headwall."""
    cx = 0.5 * (WALL_X[0] + WALL_X[1])
    xs = cx + np.linspace(-HILL_HALF_W, HILL_HALF_W, 101)
    zs = np.unique(
        np.concatenate([np.linspace(z0, z0 + 60, 13), np.linspace(z0, z1, 41), np.linspace(z1 - 60, z1, 13)])
    )
    X, Z = np.meshgrid(xs, zs)
    dmin = np.minimum(Z - z0, z1 - Z)
    Y = hill_lateral(X) * (HEADWALL_H + HILL_RISE * (1.0 - np.exp(-dmin / 50.0)) * (0.8 + 0.2 * np.sin(X / 37.0)))
    P = [np.stack([X, Y, Z], -1).reshape(-1, 3)]
    nx, nz = len(xs), len(zs)
    T = [(j * nx + i, (j + 1) * nx + i, j * nx + i + 1) for j in range(nz - 1) for i in range(nx - 1)] + [
        ((j + 1) * nx + i, (j + 1) * nx + i + 1, j * nx + i + 1) for j in range(nz - 1) for i in range(nx - 1)
    ]
    T = [np.array(T)]
    base = nx * nz
    for z in (z0, z1):  # soil faces outside the headwall: ground -> hill edge
        for side in (-1, 1):
            sx = xs[side * (xs - cx) >= HEADWALL_HALF_W - 1e-9]
            top = hill_lateral(sx) * HEADWALL_H
            q = np.concatenate([np.stack([sx, np.zeros_like(sx), np.full_like(sx, z)], -1),
                                np.stack([sx, top, np.full_like(sx, z)], -1)])  # fmt: skip
            m = len(sx)
            T.append(np.array([(base + i, base + i + 1, base + m + i + 1) for i in range(m - 1)]
                              + [(base + i, base + m + i + 1, base + m + i) for i in range(m - 1)]))  # fmt: skip
            P.append(q)
            base += 2 * m
    return np.concatenate(P), np.concatenate(T)


def marking_rects(z0: float, z1: float) -> list[tuple[float, float, float, float]]:
    """Solid white edge lines and a dashed lane line (split into DASH + GAP segments, see the highway builder)."""
    seg = DASH + GAP
    rects = []
    for x0, x1 in ((0.0, 0.15), (ROAD_W - 0.15, ROAD_W)):
        rects += [(x0, x1, z, min(z + seg, z1)) for z in np.arange(z0, z1, seg)]
    for k in range(1, N_LANES):
        xc = k * LANE_W
        rects += [(xc - 0.075, xc + 0.075, z, min(z + DASH, z1)) for z in np.arange(z0 + 2.0, z1, seg)]
    return rects


# Stepped zone profile (ratios of L_th for threshold/transition, of L_in for exit), as
# summarised from the Chinese 2014 highway-tunnel lighting guideline in [S23] (docs/TUNNEL_SCENES.md).
THRESHOLD_STEPS = (1.0, 0.5)
TRANSITION_STEPS = (0.15, 0.05, 0.02)
EXIT_STEPS = (3.0, 5.0)


# Zone lengths from the design speed: [S23]'s example lengths (2 x 42 m threshold, 72 / 89 / 133 m transition,
# 30 + 30 m exit) at its 80 km/h, scaled with speed. The threshold scales with the stopping distance
# v t_r + v^2 / (2 a) (it is at least the stopping distance [H24]); the other zones scale with v (same time to drive).
SPEED_REF_KMH = 80.0
SPEED_REF_LENGTHS = {"threshold": 84.0, "transition": (72.0, 89.0, 133.0), "exit": (30.0, 30.0)}
SSD_REACTION_S = 2.5
SSD_DECEL_M_S2 = 3.4


def stopping_distance(speed_kmh: float) -> float:
    v = speed_kmh / 3.6
    return v * SSD_REACTION_S + v * v / (2.0 * SSD_DECEL_M_S2)


def zone_lengths_for_speed(speed_kmh: float) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    """(threshold length, transition step lengths, exit step lengths) [m] for a design speed [km/h]."""
    if speed_kmh <= 0:
        raise SystemExit("--design-speed-kmh must be positive")
    f_ssd = stopping_distance(speed_kmh) / stopping_distance(SPEED_REF_KMH)
    f_v = speed_kmh / SPEED_REF_KMH
    ref = SPEED_REF_LENGTHS
    return (
        ref["threshold"] * f_ssd,
        tuple(ln * f_v for ln in ref["transition"]),
        tuple(ln * f_v for ln in ref["exit"]),
    )


def zone_profile(
    z_in: float,
    z_out: float,
    l_th: float,
    l_in: float,
    threshold_len: float,
    transition_lens: tuple[float, ...],
    exit_lens: tuple[float, ...],
    spacing: dict[str, float],
    profile: str = "stepped",
) -> list[dict]:
    """Lighting zones along the tube, entrance to exit: name, [z0, z1), design road luminance, lamp spacing.

    ``stepped``: two threshold halves (L_th, 0.5 L_th), transition steps at 0.15 / 0.05 / 0.02 L_th
    (a step at or below the interior level L_in is interior, so the transition ends there),
    interior, then the exit zone at 3 L_in and 5 L_in. ``two-step``: threshold L_th, then interior.
    """
    if profile == "two-step":
        steps = [("threshold", threshold_len, l_th, spacing["threshold"])]
        exits: list[tuple[str, float, float, float]] = []
    else:
        steps = [
            (f"threshold_{i + 1}", 0.5 * threshold_len, r * l_th, spacing["threshold"])
            for i, r in enumerate(THRESHOLD_STEPS)
        ]
        for i, (r, ln) in enumerate(zip(TRANSITION_STEPS, transition_lens)):
            if r * l_th <= l_in:
                break
            steps.append((f"transition_{i + 1}", ln, r * l_th, spacing["transition"]))
        exits = [
            (f"exit_{i + 1}", ln, r * l_in, spacing["exit"]) for i, (r, ln) in enumerate(zip(EXIT_STEPS, exit_lens))
        ]
    zones, z = [], z_in
    for name, ln, lum, sp in steps:
        zones.append({"zone": name, "z0": z, "z1": z + ln, "road_luminance": lum, "spacing": sp})
        z += ln
    z_exit = z_out - sum(e[1] for e in exits)
    if z_exit <= z:
        raise SystemExit("entrance + exit zones are longer than the tunnel: shorten them or use --tunnel-length")
    zones.append({"zone": "interior", "z0": z, "z1": z_exit, "road_luminance": l_in, "spacing": spacing["interior"]})
    z = z_exit
    for name, ln, lum, sp in exits:
        zones.append({"zone": name, "z0": z, "z1": z + ln, "road_luminance": lum, "spacing": sp})
        z += ln
    return zones


def zone_at(zones: list[dict], z: float) -> str:
    for zn in zones:
        if zn["z0"] <= z < zn["z1"]:
            return zn["zone"]
    return zones[-1]["zone"] if z >= zones[-1]["z1"] else zones[0]["zone"]


def luminaire_positions(zones: list[dict]):
    """(x, z_centre, zone) for both rows, from the entrance portal; spacing follows the zone of each lamp."""
    out = []
    z = zones[0]["z0"] + 0.5 * LAMP_SIZE[1] + 0.5
    z_end = zones[-1]["z1"] - 0.5 * LAMP_SIZE[1]
    by_name = {zn["zone"]: zn for zn in zones}
    while z < z_end:
        zone = zone_at(zones, z)
        out += [(x, z, zone) for x in LAMP_X]
        z += by_name[zone]["spacing"]
    return out


def zone_road_points(zn: dict) -> np.ndarray:
    """7 x 13 road grid at the zone centre (+-6 m, or +-0.4 of a shorter zone) where the lamps are solved."""
    zc, h = 0.5 * (zn["z0"] + zn["z1"]), min(6.0, 0.4 * (zn["z1"] - zn["z0"]))
    return np.array([(x, 0.0, zc + dz) for x in np.linspace(0.5, ROAD_W - 0.5, 7) for dz in np.linspace(-h, h, 13)])


def solve_lamp_luminance(
    zones: list[dict],
    lamps: list[tuple[float, float, str]],
    rho: float,
    table: rt.RTable | None = None,
    k_tint: float = 1.0,
    indirect: np.ndarray | None = None,
) -> dict[str, float]:
    """Lamp luminance per zone so the direct road luminance (all zones' lamps) hits each zone's design value.

    Lambertian road: solves ``rho / pi * sum_j E_ij L_j = L_road,i`` where E_ij is the mean horizontal
    illuminance at zone i's centre grid from unit-luminance lamps of zone j (spill across zone boundaries
    included). r-table road: E_ij is replaced by the r-table luminance (``direct_road_luminance``) times
    ``k_tint``, the luminous factor of the road tint under the lamp. ``indirect[i, j]`` (optional) adds
    zone j's interreflected road luminance at zone i per unit lamp luminance (``interreflected_road_luminance``).
    """
    names = [zn["zone"] for zn in zones]
    if table is None:
        a = np.array([[direct_road_illuminance(zone_road_points(zi), lamps, j).mean() for j in names] for zi in zones])
        b = np.array([math.pi * zn["road_luminance"] / rho for zn in zones])
    else:
        a = k_tint * np.array(
            [[direct_road_luminance(zone_road_points(zi), lamps, j, table).mean() for j in names] for zi in zones]
        )
        b = np.array([zn["road_luminance"] for zn in zones])
    if indirect is not None:
        a = a + indirect * (b / np.array([zn["road_luminance"] for zn in zones]))[:, None]
    lum = np.linalg.solve(a, b)
    if np.any(lum <= 0):
        raise SystemExit(f"no non-negative lamp solution for zones {names}: lengthen the short zones")
    return {n: float(v) for n, v in zip(names, lum)}


def interreflected_road_luminance(
    zones: list[dict],
    lamps: list[tuple[float, float, str]],
    z_in: float,
    z_out: float,
    rho_road: float,
    rho_lining: float,
    road_q0: float | None = None,
) -> tuple[np.ndarray, ti.TubeRadiosity]:
    """``M[i, j]``: interreflected road luminance [cd/m^2] at zone i's centre grid from unit-luminance lamps of zone j.

    Radiosity of the lit tube (``tunnel_interreflection``): lamp light on the road, walls and vault, then every
    further bounce. The road reflects interreflected light as ``rho_road / pi`` (Lambertian) or ``road_q0``
    (r-table road: the light arrives from all directions, so its average luminance coefficient Q0)."""
    tube = ti.TubeRadiosity(tube_profile(), lambda x: rho_road, rho_lining, z_out - z_in)
    names = [zn["zone"] for zn in zones]
    k = rho_road / math.pi if road_q0 is None else road_q0
    m = np.zeros((len(zones), len(zones)))
    for j, name in enumerate(names):
        e = tube.bounce(tube.direct_from_lamps([(x, z - z_in) for x, z, zz in lamps if zz == name], LAMP_Y, LAMP_SIZE))
        for i, zn in enumerate(zones):
            zc, h = 0.5 * (zn["z0"] + zn["z1"]) - z_in, min(6.0, 0.4 * (zn["z1"] - zn["z0"]))
            m[i, j] = k * tube.floor_mean(e, 0.5, ROAD_W - 0.5, zc - h, zc + h)
    return m, tube


def direct_road_illuminance(points: np.ndarray, lamps: list[tuple[float, float, str]], zone: str) -> np.ndarray:
    """Horizontal illuminance [lx] at ``points`` from unit-luminance downward Lambertian lamps of ``zone``."""
    w, ln = LAMP_SIZE
    sub = np.linspace(-0.5 + 1 / 12, 0.5 - 1 / 12, 6) * ln
    e = np.zeros(len(points))
    da = w * ln / len(sub)
    for x, z, zz in lamps:
        if zz != zone:
            continue
        for dz in sub:
            v = np.array([x, LAMP_Y, z + dz]) - points
            d2 = np.einsum("ij,ij->i", v, v)
            cos = v[:, 1] / np.sqrt(d2)  # same angle at lamp (normal -y) and road (normal +y)
            e += da * cos * cos / d2
    return e


def direct_road_luminance(
    points: np.ndarray, lamps: list[tuple[float, float, str]], zone: str, table: rt.RTable
) -> np.ndarray:
    """r-table road luminance [cd/m^2] at ``points`` from unit-luminance lamps of ``zone`` (neutral tint).

    CIE observer convention: 1 deg above the road looking along +z (the driving direction), so beta = 0
    for a lamp further down the road than the point.
    """
    w, ln = LAMP_SIZE
    sub = np.linspace(-0.5 + 1 / 12, 0.5 - 1 / 12, 6) * ln
    out = np.zeros(len(points))
    da = w * ln / len(sub)
    for x, z, zz in lamps:
        if zz != zone:
            continue
        for dz in sub:
            v = np.array([x, LAMP_Y, z + dz]) - points
            d2 = np.einsum("ij,ij->i", v, v)
            cos = v[:, 1] / np.sqrt(d2)
            h = np.hypot(v[:, 0], v[:, 2])
            beta = np.degrees(np.arccos(np.clip(v[:, 2] / np.maximum(h, 1e-12), -1, 1)))
            out += da * cos * cos / d2 * table.q(h / v[:, 1], beta)
    return out


def luminous_reflectance(rho: np.ndarray, spd: np.ndarray, wl: np.ndarray) -> float:
    y = cmf_on_grid(wl)[1]
    return float(np.sum(rho * spd * y) / np.sum(spd * y))


class Projector:
    """Pinhole projection matching pbrt-v4's LookAt + perspective camera (fov on the shorter axis)."""

    def __init__(self, eye: np.ndarray, target: np.ndarray, fov: float, xres: int, yres: int):
        self.eye, self.xres, self.yres = eye, xres, yres
        f = (target - eye) / np.linalg.norm(target - eye)
        r = np.cross([0.0, 1.0, 0.0], f)
        self.f, self.r = f, r / np.linalg.norm(r)
        self.u = np.cross(f, self.r)
        self.s = 1.0 / math.tan(math.radians(fov) / 2.0)
        self.ax, self.ay = (xres / yres, 1.0) if xres >= yres else (1.0, yres / xres)

    def __call__(self, p: np.ndarray) -> np.ndarray:
        d = np.atleast_2d(p) - self.eye
        zc = d @ self.f
        sx, sy = self.s * (d @ self.r) / zc, self.s * (d @ self.u) / zc
        return np.stack([(sx + self.ax) / (2 * self.ax) * self.xres, (self.ay - sy) / (2 * self.ay) * self.yres], -1)

    def roi(self, box: list[float], shrink: float = 0.15) -> list[int] | None:
        x0, x1, y0, y1, z0, z1 = box
        c = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
        if np.any((c - self.eye) @ self.f <= 0.1):
            return None
        q = self(c)
        (a, b), (cc, d) = q.min(0), q.max(0)
        mx, my = shrink * (cc - a), shrink * (d - b)
        r = [math.ceil(a + mx), math.ceil(b + my), math.floor(cc - mx), math.floor(d - my)]
        r = [max(r[0], 0), max(r[1], 0), min(r[2], self.xres), min(r[3], self.yres)]
        return r if r[2] - r[0] >= 2 and r[3] - r[1] >= 2 else None


def _lengths(text: str) -> tuple[float, ...]:
    vals = tuple(float(v) for v in text.split(",") if v.strip())
    if not vals or min(vals) <= 0:
        raise argparse.ArgumentTypeError("expected positive comma-separated lengths [m]")
    return vals


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=Path, default=root)
    ap.add_argument("--out-dir", type=Path, default=None, help="default: scenes/generated/tunnel")
    ap.add_argument("--view", choices=(*VIEWS, "both"), default="both")
    ap.add_argument("--sun-elevation", type=float, default=45.0, help="Sun elevation [deg] (analytic clear sky).")
    ap.add_argument(
        "--sun-azimuth", type=float, default=140.0, help="Sun azimuth [deg] from the road direction, + = right."
    )
    ap.add_argument("--global-illuminance-lux", type=float, default=None, help="Override sun + sky horizontal lux.")
    ap.add_argument("--tunnel-length", type=float, default=400.0)
    ap.add_argument("--portal-distance", type=float, default=40.0, help="outside_in: camera to entrance portal [m].")
    ap.add_argument("--exit-distance", type=float, default=60.0, help="inside_out: camera to exit portal [m].")
    ap.add_argument(
        "--sky-portal-guide",
        type=float,
        default=0.5,
        help="Fraction of sky light samples aimed through the two portal openings for points inside the tube "
        "(patched pbrt 'guideportals'; unbiased, cuts sky noise on the interior road). 0 = off.",
    )
    ap.add_argument("--l20", type=float, default=3000.0, help="Design access-zone luminance L20 [cd/m^2].")
    ap.add_argument("--k", type=float, default=0.025, help="Threshold-zone ratio k = L_th / L20.")
    ap.add_argument("--interior-luminance", type=float, default=6.0, help="Interior-zone road luminance [cd/m^2].")
    ap.add_argument(
        "--zone-profile",
        choices=("stepped", "two-step"),
        default="stepped",
        help="stepped: threshold, transition, interior and exit zones; two-step: threshold then interior only.",
    )
    ap.add_argument(
        "--design-speed-kmh",
        type=float,
        default=None,
        help="Derive the threshold, transition and exit lengths from this design speed (overrides "
        "--threshold-length, --transition-lengths and --exit-lengths; see docs/TUNNEL_SCENES.md).",
    )
    ap.add_argument("--threshold-length", type=float, default=80.0, help="Threshold-zone length (both halves) [m].")
    ap.add_argument(
        "--transition-lengths",
        type=_lengths,
        default=(72.0, 89.0, 133.0),
        help="Transition step lengths [m], comma-separated (0.15 / 0.05 / 0.02 L_th).",
    )
    ap.add_argument(
        "--exit-lengths", type=_lengths, default=(30.0, 30.0), help="Exit step lengths [m] (3x / 5x interior)."
    )
    ap.add_argument("--spacing-threshold", type=float, default=3.0, help="Luminaire spacing, threshold zone [m].")
    ap.add_argument("--spacing-transition", type=float, default=4.5, help="Luminaire spacing, transition zone [m].")
    ap.add_argument("--spacing-interior", type=float, default=9.0, help="Luminaire spacing, interior zone [m].")
    ap.add_argument("--spacing-exit", type=float, default=4.5, help="Luminaire spacing, exit zone [m].")
    ap.add_argument("--lamp", choices=("led_4000k", "hps"), default="led_4000k")
    ap.add_argument(
        "--luminaire-solve",
        choices=("interreflection", "direct"),
        default="interreflection",
        help="Size the luminaires for the road luminance including light interreflected by the road and lining "
        "(radiosity of the tube, default) or for the direct lamp light only (earlier behaviour).",
    )
    ap.add_argument("--target", action=argparse.BooleanOptionalAction, default=True, help="Place the target cars.")
    add_camera_args(ap, fov=30.0)
    ap.add_argument("--cam-height", type=float, default=1.35, help="Camera height above the road [m].")
    ap.add_argument("--cam-lane", type=int, default=1, help="Ego lane (0 = left).")
    ap.add_argument("--cam-pitch", type=float, default=0.0, help="Camera pitch [deg], negative = down.")
    add_film_args(ap, xres=960, yres=640, pixelsamples=256)
    ap.add_argument("--maxdepth", type=int, default=5)
    ap.add_argument(
        "--road-reflection",
        choices=("lambertian", "rtable"),
        default="lambertian",
        help="Road surface: Lambertian asphalt (default) or a road r-table (pbrt roadrtable material).",
    )
    ap.add_argument(
        "--road-rtable",
        default="placeholder_dry_model",
        help="r-table CSV or name in config/road_rtables (default: placeholder, not CIE data).",
    )
    ap.add_argument("--sampler-seed", type=int, default=0, help="In-scene sampler seed (overrides pbrt --seed).")
    return ap.parse_args(argv)


def sky_guide_lines(fraction: float, z_in: float, z_out: float) -> list[str]:
    """pbrt ``guideportals`` for the sky light: the tube's bounding rectangle at each portal, normals outward."""
    if fraction == 0:
        return []
    if not 0 < fraction < 1:
        raise ValueError("--sky-portal-guide must be 0 (off) or in (0, 1)")
    x0, x1 = WALL_X
    rects = (
        [(x1, 0.0, z_in), (x0, 0.0, z_in), (x0, CROWN_H, z_in), (x1, CROWN_H, z_in)],  # normal -z
        [(x0, 0.0, z_out), (x1, 0.0, z_out), (x1, CROWN_H, z_out), (x0, CROWN_H, z_out)],  # normal +z
    )
    pts = " ".join(_f(v) for r in rects for q in r for v in q)
    return [f'        "point3 guideportals" [{pts}]', f'        "float guidefraction" [{_f(fraction)}]']


def build(args: argparse.Namespace) -> dict:  # noqa: C901 (linear scene assembly)
    repo = args.repo_root.resolve()
    out_dir = (args.out_dir or repo / "scenes" / "generated" / "tunnel").resolve()
    spd = out_dir / "spd"
    spd.mkdir(parents=True, exist_ok=True)
    z_in = float(args.portal_distance)
    z_out = z_in + float(args.tunnel_length)

    # ---- spectra
    wl = np.arange(args.spectral_lambda_min, args.spectral_lambda_max + 1e-9, args.step_nm)
    for name in SURFACES:
        write_spd(spd / f"{name}.spd", wl, reflectance(name, wl))
    lamp_spd = LAMP_SPECTRA[args.lamp](LIGHT_WL)
    write_spd(spd / f"lamp_{args.lamp}.spd", LIGHT_WL, lamp_spd)

    # ---- daylight (same clear-sky model as tools/build_highway_scene.py --sky analytic)
    elev = float(args.sun_elevation)
    zs = math.radians(90.0 - elev)
    sky_rot, sun_w = sun_world(np.array([math.sin(zs), 0.0, math.cos(zs)]), args.sun_azimuth)
    sky_file = out_dir / "textures" / f"sky_cie12_{elev:.1f}.exr"
    if not sky_file.is_file():
        write_rgb_exr(sky_file, analytic_clear_sky(512, elev))
    e_dn, e_sky_h = clear_sky_illuminance_lux(elev)
    sin_el = max(math.sin(math.radians(elev)), 1e-3)
    e_ref = e_dn * sin_el + e_sky_h
    if args.global_illuminance_lux is not None:
        k = float(args.global_illuminance_lux) / e_ref
        e_dn, e_sky_h, e_ref = e_dn * k, e_sky_h * k, float(args.global_illuminance_lux)
    sun_spd = solar_direct_spectrum(wl, elev)
    write_spd(spd / "sun.spd", wl, sun_spd)

    # ---- photometry: luminaires solved for the direct road luminance per zone
    rho_road = reflectance("asphalt_aged", wl)
    rho_sun = luminous_reflectance(rho_road, sun_spd, wl)
    rho_lamp = luminous_reflectance(rho_road, np.interp(wl, LIGHT_WL, lamp_spd), wl)
    table, road_refl = None, {"model": "lambertian"}
    if args.road_reflection == "rtable":
        table = rt.load(rt.resolve(args.road_rtable))
        rt.warn_if_placeholder(table)
        tint = rt.tint_spectrum(rho_road, wl)
        write_spd(spd / "asphalt_aged_rtint.spd", wl, tint)
        k_sun = luminous_reflectance(tint, sun_spd, wl)
        k_lamp = luminous_reflectance(tint, np.interp(wl, LIGHT_WL, lamp_spd), wl)
        road_refl = {
            "model": "rtable",
            "table": table.summary(),
            "table_file": str(rt.resolve(args.road_rtable)),
            "tint_spd": "spd/asphalt_aged_rtint.spd (asphalt_aged shape, luminous reflectance 1)",
            "tint_luminous_factor": {"daylight": k_sun, "lamp": k_lamp},
            "observation_blend_deg": [rt.OBSERVATION_MIN_DEG, rt.OBSERVATION_MAX_DEG],
            "design_observer": "1 deg above the road looking along +z (CIE r-table geometry)",
            "exterior_sky_term": "Q0 x horizontal sky illuminance (sky treated as uniform)",
        }
    l_th = float(args.k) * float(args.l20)
    l_in = float(args.interior_luminance)
    spacing = {
        "threshold": args.spacing_threshold,
        "transition": args.spacing_transition,
        "interior": args.spacing_interior,
        "exit": args.spacing_exit,
    }
    if args.design_speed_kmh is None:
        lengths = (float(args.threshold_length), tuple(args.transition_lengths), tuple(args.exit_lengths))
    else:
        lengths = zone_lengths_for_speed(float(args.design_speed_kmh))
    zones = zone_profile(
        z_in,
        z_out,
        l_th,
        l_in,
        *lengths,
        spacing,
        args.zone_profile,
    )
    lamps = luminaire_positions(zones)
    indirect = None
    if args.luminaire_solve == "interreflection":
        rho_lining = luminous_reflectance(reflectance("concrete", wl), np.interp(wl, LIGHT_WL, lamp_spd), wl)
        q0 = None if table is None else k_lamp * table.q0()
        indirect, _ = interreflected_road_luminance(zones, lamps, z_in, z_out, rho_lamp, rho_lining, q0)
    if table is None:
        lamp_l = solve_lamp_luminance(zones, lamps, rho_lamp, indirect=indirect)
        l_ext = rho_sun * e_ref / math.pi
    else:
        lamp_l = solve_lamp_luminance(zones, lamps, rho_lamp, table, k_lamp, indirect)
        hs = math.hypot(sun_w[0], sun_w[2])
        beta_sun = math.degrees(math.acos(max(-1.0, min(1.0, sun_w[2] / max(hs, 1e-12)))))
        q_sun = float(table.q(1.0 / math.tan(math.radians(max(elev, 1e-3))), beta_sun))
        l_ext = k_sun * (q_sun * e_dn * sin_el + table.q0() * e_sky_h)
        road_refl["sun_q"] = q_sun

    # ---- shared world
    W = [
        "",
        "WorldBegin",
        "",
        f"# Sun: elevation {elev:.2f} deg, azimuth {args.sun_azimuth:.1f} deg, {e_dn:.0f} lux direct-normal",
        'LightSource "distant" "spectrum L" "spd/sun.spd"',
        f'    "float illuminance" [{_f(e_dn)}]',
        f'    "point3 from" [{_pts(sun_w)}] "point3 to" [0 0 0]',
        f"# Sky: analytic CIE standard clear sky (type 12), {e_sky_h:.0f} lux on a horizontal plane",
        "AttributeBegin",
        f"    Rotate {_f(sky_rot)} 0 1 0",
        "    Rotate -90 1 0 0",
        f'    LightSource "infinite" "string filename" "textures/{sky_file.name}"',
        f'        "float illuminance" [{_f(e_sky_h)}]',
        *sky_guide_lines(args.sky_portal_guide, z_in, z_out),
        "AttributeEnd",
        "",
        *(
            [
                # Lambertian asphalt (no CIE r-table specular lobe) so road luminance = rho * E / pi exactly.
                'MakeNamedMaterial "asphalt" "string type" "diffuse" "spectrum reflectance" "spd/asphalt_aged.spd"',
            ]
            if table is None
            else rt.material_lines("asphalt", table, '"spectrum tint" "spd/asphalt_aged_rtint.spd"')
        ),
        'MakeNamedMaterial "paint_white" "string type" "coateddiffuse"'
        ' "spectrum reflectance" "spd/paint_road_white.spd" "float roughness" [0.3]',
        'MakeNamedMaterial "concrete" "string type" "diffuse" "spectrum reflectance" "spd/concrete.spd"',
        'MakeNamedMaterial "grass" "string type" "diffuse" "spectrum reflectance" "spd/grass.spd"',
        'MakeNamedMaterial "soil" "string type" "diffuse" "spectrum reflectance" "spd/soil.spd"',
        'MakeNamedMaterial "galvanized" "string type" "coateddiffuse"'
        ' "spectrum reflectance" "spd/galvanized.spd" "float roughness" [0.3]',
        "",
    ]
    z_road0, z_road1 = -50.0, z_out + GROUND_FAR
    p, t = quads_mesh([(WALL_X[0], WALL_X[1], z_road0, z_road1)], 0.0)
    W += ['NamedMaterial "asphalt"', *mesh(p, t), ""]
    p, t = quads_mesh(marking_rects(z_road0, z_road1), 0.004)
    W += ['NamedMaterial "paint_white"', *mesh(p, t), ""]
    cx = 0.5 * (WALL_X[0] + WALL_X[1])
    ground = [
        (cx - 600.0, WALL_X[0], z_road0 - 50.0, z_in),
        (WALL_X[1], cx + 600.0, z_road0 - 50.0, z_in),
        (cx - 2000.0, WALL_X[0], z_out, z_road1),
        (WALL_X[1], cx + 2000.0, z_out, z_road1),
    ]
    p, t = quads_mesh(ground, -0.01)
    W += ['NamedMaterial "grass"', *mesh(p, t), ""]
    p, t = extrude_profile(tube_profile(), 0.0, z_in, z_out)
    W += ["# Tunnel lining (walls + vault)", 'NamedMaterial "concrete"', *mesh(p, t), ""]
    for z in (z_in, z_out):
        W += [f"# Headwall at z = {z:g}", *mesh(*headwall(z)), ""]
    p, t = hill_mesh(z_in, z_out)
    W += ["# Hillside (grass) and soil faces beside the headwalls", 'NamedMaterial "soil"', *mesh(p, t), ""]
    W += [f"# {len(lamps)} LED line luminaires ({args.lamp}), emitting downwards", 'NamedMaterial "galvanized"']
    for x, z, zone in lamps:
        W += _quad_light(
            (x, LAMP_Y, z),
            (LAMP_SIZE[0] / 2, 0, 0),
            (0, 0, LAMP_SIZE[1] / 2),
            f"spd/lamp_{args.lamp}.spd",
            lamp_l[zone],
        )
    targets = {
        "outside_in": {"x": 0.5 * LANE_W, "z": z_in + 25.0, "zone": zone_at(zones, z_in + 25.0)},
        "inside_out": {
            "x": 0.5 * LANE_W,
            "z": z_out - 15.0,
            "zone": f"{zone_at(zones, z_out - 15.0)} (silhouetted against the exit)",
        },
    }

    common = {
        "generator": "tools/build_tunnel_scene.py",
        "units": {"length": "m", "luminance": "cd/m^2", "illuminance": "lx"},
        "tunnel": {
            "portal_entrance_z": z_in,
            "portal_exit_z": z_out,
            "length_m": float(args.tunnel_length),
            "lanes": N_LANES,
            "lane_width_m": LANE_W,
            "wall_x": list(WALL_X),
            "wall_height_m": WALL_H,
            "crown_height_m": CROWN_H,
            "headwall": {"half_width_m": HEADWALL_HALF_W, "height_m": HEADWALL_H},
        },
        "luminaires": {
            "count": len(lamps),
            "count_by_zone": {zn["zone"]: sum(1 for lp in lamps if lp[2] == zn["zone"]) for zn in zones},
            "lamp_spectrum": args.lamp,
            "size_m": list(LAMP_SIZE),
            "mount_height_m": LAMP_Y,
            "rows_x_m": list(LAMP_X),
            "spacing_m": {zn["zone"]: zn["spacing"] for zn in zones},
            "luminance_cd_m2": lamp_l,
            "solve": args.luminaire_solve,
            "interreflected_road_luminance_cd_m2": None
            if indirect is None
            else {zn["zone"]: float(indirect[i] @ [lamp_l[z["zone"]] for z in zones]) for i, zn in enumerate(zones)},
        },
        "photometry": {
            "l20_cd_m2": float(args.l20),
            "k": float(args.k),
            "threshold_road_luminance_cd_m2": l_th,
            "design_speed_kmh": args.design_speed_kmh,
            "stopping_distance_m": None if args.design_speed_kmh is None else stopping_distance(args.design_speed_kmh),
            "interior_road_luminance_cd_m2": l_in,
            "threshold_length_m": float(args.threshold_length),
            "zone_profile": args.zone_profile,
            "zones": [
                {
                    "zone": zn["zone"],
                    "z_start_m": zn["z0"],
                    "z_end_m": zn["z1"],
                    "length_m": zn["z1"] - zn["z0"],
                    "road_luminance_cd_m2": zn["road_luminance"],
                    "luminaire_spacing_m": zn["spacing"],
                }
                for zn in zones
            ],
            "exterior_road_luminance_cd_m2": l_ext,
            "road_luminous_reflectance": {"daylight": rho_sun, "lamp": rho_lamp},
            "road_reflection": road_refl,
            "note": "tunnel road luminances are design values for the lamp light: direct only with luminaires.solve "
            "'direct'; direct + interreflected in the lit tube with 'interreflection' (daylight entering through "
            "the portals adds to them in both cases)",
        },
        "lighting": {
            "sun": {"elevation_deg": elev, "azimuth_deg": args.sun_azimuth, "direct_normal_lux": e_dn},
            "sky": {"model": "analytic CIE clear sky (type 12)", "horizontal_lux": e_sky_h},
            "reference_illuminance_lux": e_ref,
            "reference_illuminance_exr_lux": 683.0 * PBRT_CIE_Y_INTEGRAL * e_ref,
            "reference": "horizontal illuminance (sun + sky) at road level, unoccluded",
        },
        "approximations": [
            (
                "Lambertian asphalt (no road-surface r-table / specular lobe)"
                if table is None
                else f"r-table asphalt ({table.name}{', PLACEHOLDER not CIE data' if table.placeholder else ''}):"
                " design luminances for a 1 deg observer; steeper views blend to Q0"
            ),
            (
                "step from threshold to interior luminance: no transition or exit-zone lighting"
                if args.zone_profile == "two-step"
                else "stepped zone luminances (constant per step), not a continuous adaptation curve"
            ),
            "zone lengths are fixed in metres, not derived from design speed / stopping distance"
            if args.design_speed_kmh is None
            else f"zone lengths scaled from [S23]'s 80 km/h example to {args.design_speed_kmh:g} km/h "
            "(threshold with the stopping distance, the rest with speed); not a CIE 88 design",
            "luminaire solve ('interreflection'): radiosity of the lit tube only (Lambertian road and lining, "
            "no cars, no daylight entering through the portals)",
            "lamp output solved for direct road luminance only; walls and daylight spill add to it",
            "box proxy cars as targets",
        ],
    }

    written = {}
    views = VIEWS if args.view == "both" else (args.view,)
    for view in views:
        cam_x = 0.5 * LANE_W + args.cam_lane * LANE_W
        cam_z = 0.0 if view == "outside_in" else z_out - float(args.exit_distance)
        eye = np.array([cam_x, args.cam_height, cam_z])
        target = eye + 100.0 * np.array([0.0, math.tan(math.radians(args.cam_pitch)), 1.0])
        film_out = f"out/tunnel_{view}{'_spectral' if args.film == 'spectral' else ''}.exr"
        L = [
            "# Generated by tools/build_tunnel_scene.py — do not hand-edit.",
            'ColorSpace "srgb"',
            f'Sampler "zsobol" "integer pixelsamples" [{args.pixelsamples}] "integer seed" [{args.sampler_seed}]',
            f'Integrator "path" "integer maxdepth" [{args.maxdepth}] "bool regularize" true',
            'PixelFilter "box" "float xradius" [0.5] "float yradius" [0.5]',
            *film_lines(args, film_out),
            f"LookAt {_pts(eye)}  {_pts(target)}  0 1 0",
            *camera_lines(args, repo, out_dir),
            *W,
        ]
        tg = targets[view]
        if args.target:
            L += [
                f"# Target car ({tg['zone']})",
                "AttributeBegin",
                f"Translate {_f(tg['x'])} 0 {_f(tg['z'])}",
                *proxy_car(f"spd/{CAR_PAINT}.spd", "target"),
                "AttributeEnd",
            ]
        L.append("")
        scene = out_dir / f"tunnel_{view}.pbrt"
        scene.write_text("\n".join(L))

        rear = tg["z"] - 2.3
        if view == "outside_in":
            boxes = {
                "exterior_road": ([0.4, ROAD_W - 0.4, 0.0, 0.0, 22.0, 34.0], "exterior", l_ext),
                "headwall": ([cx - 17.0, cx - 8.0, 2.0, 9.0, z_in, z_in], "exterior", None),
                "threshold_road": ([LANE_W + 0.4, ROAD_W - 0.4, 0.0, 0.0, z_in + 8.0, z_in + 30.0], "dark", l_th),
                "threshold_wall": ([WALL_X[1], WALL_X[1], 0.8, 3.5, z_in + 10.0, z_in + 40.0], "dark", None),
            }
            dark, bright = "threshold_road", "exterior_road"
        else:
            # Dark reference: the road 12-30 m ahead, in whichever zone the camera looks at
            # (interior, or the exit zone with the default stepped profile and --exit-distance).
            rz = (cam_z + 12.0, cam_z + 30.0)
            kind = zone_at(zones, 0.5 * (rz[0] + rz[1]))
            if zone_at(zones, rz[0]) != kind or zone_at(zones, rz[1] - 1e-6) != kind:
                raise SystemExit(
                    f"inside_out: road 12-30 m ahead straddles a zone boundary ({kind}); change --exit-distance"
                )
            dark = (
                "interior_road"
                if kind == "interior"
                else "exit_zone_road"
                if kind.startswith("exit")
                else f"{kind}_road"
            )
            l_dark = next(zn["road_luminance"] for zn in zones if zn["zone"] == kind)
            boxes = {
                dark: ([LANE_W + 0.4, ROAD_W - 0.4, 0.0, 0.0, *rz], "dark", l_dark),
                "interior_wall": ([WALL_X[1], WALL_X[1], 0.8, 3.5, cam_z + 8.0, cam_z + 30.0], "dark", None),
                "exit_opening": ([LANE_W + 0.3, ROAD_W, 2.6, 6.0, z_out, z_out], "exterior", None),
                "exit_road": ([0.4, ROAD_W - 0.4, 0.0, 0.0, z_out + 3.0, z_out + 40.0], "exterior", l_ext),
            }
            last = zones[-1]
            if dark == "interior_road" and last["zone"].startswith("exit") and last["z1"] - last["z0"] > 8.0:
                boxes["exit_zone_road"] = (
                    [LANE_W + 0.4, ROAD_W - 0.4, 0.0, 0.0, last["z0"] + 2.0, last["z1"] - 3.0],
                    "dark",
                    last["road_luminance"],
                )
            bright = "exit_road"
        boxes["target"] = ([tg["x"] - 0.75, tg["x"] + 0.75, 0.45, 1.2, rear, rear], "target", None)
        proj = Projector(eye, target, args.fov, args.xres, args.yres)
        rois = {}
        for name, (box, region, lum) in boxes.items():
            rois[name] = {"world_box": box, "region": region, "roi_xyxy": proj.roi(box)}
            if lum is not None:
                rois[name]["predicted_luminance_cd_m2"] = lum
        if view == "outside_in":
            crest = np.array(
                [(x, float(hill_lateral(np.array([x]))[0]) * HEADWALL_H, z_in) for x in (cx - 15, cx + 15)]
            )
            y_top = float(proj(crest)[:, 1].min())
            if y_top > 0.08 * args.yres:
                x0, x1 = int(0.35 * args.xres), int(0.65 * args.xres)
                rois["sky"] = {"world_box": None, "region": "exterior", "roi_xyxy": [x0, 2, x1, int(0.6 * y_top)]}
        dr = 20.0 * math.log10(rois[bright]["predicted_luminance_cd_m2"] / rois[dark]["predicted_luminance_cd_m2"])
        manifest = {
            "scene": scene.name,
            "film_output": film_out,
            "view": view,
            **common,
            "film": {"type": args.film, "xres": args.xres, "yres": args.yres, "pixelsamples": args.pixelsamples},
            "camera": {
                "type": "pinhole" if args.camera == "perspective" else args.camera,
                "lookat": {"eye": eye.tolist(), "target": target.tolist(), "up": [0.0, 1.0, 0.0]},
                "fov_deg": None if args.camera == "realistic" else float(args.fov),
                **(
                    {
                        "lensfile": args.lensfile,
                        "aperture_diameter_mm": args.aperture_diameter_mm,
                        "focus_distance": args.focus_distance,
                        "film_diagonal_mm": args.film_diagonal_mm,
                        "lens_dispersion": bool(args.lens_dispersion),
                        "lens_dispersion_split": bool(args.lens_dispersion_split),
                    }
                    if args.camera == "realistic"
                    else {}
                ),
            },
            "target": {
                "kind": "proxy car (box), rear face towards the camera",
                "paint": CAR_PAINT,
                "position_m": [tg["x"], 0.0, tg["z"]],
                "zone": tg["zone"],
                "roi": "target",
            },
            "rois": rois,
            "rois_valid_for": "pinhole/thinlens projection; realistic: derive from world_box (gbuffer)",
            "dynamic_range": {
                "bright_roi": bright,
                "dark_roi": dark,
                "ratio": 10.0 ** (dr / 20.0),
                "db": dr,
            },
        }
        man = out_dir / f"tunnel_{view}_manifest.json"
        man.write_text(json.dumps(manifest, indent=2))
        written[view] = {"scene": scene, "manifest": man, "meta": manifest}
    return written


def main(argv: list[str] | None = None) -> None:
    for view, w in build(parse_args(argv)).items():
        d = w["meta"]["dynamic_range"]
        print(f"{view}: {w['scene']}  ({d['bright_roi']} / {d['dark_roi']} = {d['db']:.1f} dB)")


if __name__ == "__main__":
    main()
