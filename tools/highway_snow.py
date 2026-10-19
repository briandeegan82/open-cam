#!/usr/bin/env python3
"""Winter scene options for the highway builder: snow-covered verges, plough banks, snow on the
road and falling snow (opt-in, ``--snow``).

Snow reflectance is the spherical albedo of a semi-infinite layer from asymptotic radiative
transfer (Kokhanovsky & Zege 2004, Appl. Opt. 43, 1589):

    r(lambda) = exp(-sqrt(alpha(lambda) * L)),  alpha = 4 pi m_im / lambda + alpha_dirt,  L = 13 d

with the ice absorption index ``m_im`` of Warren & Brandt (2008) (spectra/ice/), ``d`` the optical
grain diameter and an absorbing-impurity term ``alpha_dirt = a550 (lambda / 550 nm)^-1``. It is
used as a Lambertian reflectance: the forward-scattering snow BRDF and sub-surface blur are not
modelled.

Falling snow: static flakes (spheres) inside the camera frustum out to ``--snowfall-range-m``,
melted-diameter distribution N(D) = N0 exp(-Lambda D) of Gunn & Marshall (1958),
N0 = 3.8e3 R^-0.87 m^-3 mm^-1, Lambda = 2.55 R^-0.48 mm^-1 (R = melted snowfall rate, mm/h); a
flake of bulk density rho has diameter D (rho_w / rho)^(1/3). The equivalent extinction 2 x the
geometric cross-section gives the meteorological optical range reported in the manifest, which
can be applied beyond the flake range with ``--haze mist --visibility-m``.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ICE_CSV = REPO / "spectra" / "ice" / "warren_brandt_2008.csv"

STATES = ("none", "verges", "slush", "packed")
L_OVER_D = 13.0  # absorption path / grain diameter (Kokhanovsky & Zege 2004)
DIRT_A550 = {"clean": 0.0, "light": 3.4, "heavy": 13.0}  # m^-1, nominal (verge ~0.9, slush ~0.6 visible albedo)
GRAIN_MM = {"verge": 0.25, "packed": 0.5, "slush": 1.5, "flake": 0.1}  # nominal optical grain diameters
ROAD_DIRT = {"packed": "light", "slush": "heavy"}
TRACK_HALF_SEP, TRACK_W = 0.85, 0.6  # wheel tracks at lane centre +- 0.85 m, 0.6 m wide (nominal)
FLAKE_DMIN_MM = 0.3  # smallest melted diameter rendered as geometry
MAX_FLAKES = 2_000_000


def add_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("winter / snow (tools/highway_snow.py)")
    g.add_argument(
        "--snow",
        choices=STATES,
        default="none",
        help="verges: snow-covered verges and plough banks, clear road; slush: + slush between the "
        "wheel tracks (wet road in the tracks); packed: + compacted snow over the whole road.",
    )
    g.add_argument(
        "--snow-grain-mm", type=float, default=None, help=f"Verge grain diameter (default {GRAIN_MM['verge']})."
    )
    g.add_argument("--snow-dirt", choices=tuple(DIRT_A550), default="clean", help="Impurities in the verge snow.")
    g.add_argument("--no-snowbanks", dest="snowbanks", action="store_false", help="No plough banks at the road edges.")
    g.add_argument("--snowfall-mmh", type=float, default=0.0, help="Falling snow, melted-water rate [mm/h].")
    g.add_argument("--snowfall-range-m", type=float, default=15.0, help="Flakes are geometry out to this range [m].")
    g.add_argument("--snowflake-density", type=float, default=0.1, help="Flake bulk density [g/cm^3].")


def enabled(args: argparse.Namespace) -> bool:
    return getattr(args, "snow", "none") != "none" or getattr(args, "snowfall_mmh", 0.0) > 0


def check_args(args: argparse.Namespace) -> list[str]:
    """Validate; a slushy road is wet in the wheel tracks. Returns notes for the manifest."""
    notes = []
    if args.snowfall_mmh < 0 or args.snowfall_range_m <= 0.5 or not 0.01 <= args.snowflake_density <= 0.917:
        raise SystemExit("--snowfall-mmh >= 0, --snowfall-range-m > 0.5, --snowflake-density in [0.01, 0.917]")
    if args.snow == "slush" and getattr(args, "road_wetness", "dry") in ("dry", "damp"):
        args.road_wetness = "wet"
        notes.append("--snow slush sets --road-wetness wet (water in the wheel tracks)")
    if args.snow == "packed" and getattr(args, "road_wetness", "dry") != "dry":
        raise SystemExit("--snow packed covers the road: use a dry --road-wetness")
    return notes


# ------------------------------------------------------------------ optics
def ice_m_im(wl_nm: np.ndarray) -> np.ndarray:
    """Warren & Brandt 2008 imaginary index, log-log interpolated (their recommended scheme)."""
    d = np.loadtxt(ICE_CSV, delimiter=",", comments="#", skiprows=5)
    lw = np.log(np.asarray(wl_nm, dtype=np.float64))
    return np.exp(np.interp(lw, np.log(d[:, 0]), np.log(d[:, 2])))


def snow_albedo(wl_nm: np.ndarray, grain_mm: float, dirt: str = "clean") -> np.ndarray:
    wl = np.asarray(wl_nm, dtype=np.float64)
    alpha = 4.0 * math.pi * ice_m_im(wl) / (wl * 1e-9) + DIRT_A550[dirt] * (wl / 550.0) ** -1.0
    return np.exp(-np.sqrt(alpha * L_OVER_D * grain_mm * 1e-3))


def gunn_marshall(rate_mmh: float) -> tuple[float, float]:
    """(N0 [m^-3 mm^-1], Lambda [mm^-1]) for a melted snowfall rate."""
    return 3.8e3 * rate_mmh**-0.87, 2.55 * rate_mmh**-0.48


def snowfall_extinction(rate_mmh: float, flake_density: float) -> float:
    """Geometric-optics extinction [m^-1] = 2 * integral N(D) pi D_f^2 / 4 dD, D_f = D (1 / rho)^(1/3)."""
    if rate_mmh <= 0:
        return 0.0
    n0, lam = gunn_marshall(rate_mmh)
    k = (1.0 / flake_density) ** (1.0 / 3.0)
    return 2.0 * math.pi / 4.0 * k**2 * n0 * 2.0 / lam**3 * 1e-6


# ------------------------------------------------------------------ pbrt
def _write_spd(path: Path, wl: np.ndarray, v: np.ndarray) -> None:
    path.write_text("".join(f"{a:g} {b:.6g}\n" for a, b in zip(wl, v, strict=True)))


def _blocks(lines: list[str]):
    i = 0
    while i < len(lines):
        j = i + 1
        while j < len(lines) and lines[j].startswith("    "):
            j += 1
        yield i, j
        i = j


def replace_materials(lines: list[str], names: dict[str, str]) -> list[str]:
    """Replace the MakeNamedMaterial blocks in ``names`` (name -> new single-line definition)."""
    out = []
    for i, j in _blocks(lines):
        head = lines[i]
        hit = next((n for n in names if head.startswith(f'MakeNamedMaterial "{n}" ')), None)
        out += [names[hit]] if hit else lines[i:j]
    return out


def materials(args, wl: np.ndarray, spd_dir: Path) -> tuple[list[str], dict]:
    grain = args.snow_grain_mm if args.snow_grain_mm is not None else GRAIN_MM["verge"]
    spec = {"verge": (grain, args.snow_dirt), "flake": (GRAIN_MM["flake"], "clean")}
    if args.snow in ROAD_DIRT:
        spec["road"] = (GRAIN_MM[args.snow], ROAD_DIRT[args.snow])
    meta, lines = {}, []
    for tag, (d, dirt) in spec.items():
        r = snow_albedo(wl, d, dirt)
        _write_spd(spd_dir / f"snow_{tag}.spd", wl, r)
        vis = (wl >= 450) & (wl <= 650)
        meta[tag] = {"grain_mm": d, "dirt": dirt, "albedo_450_650nm": round(float(r[vis].mean()), 4)}
        if tag == "road" and args.snow == "slush":  # water-saturated: smooth water film over the grains
            lines.append(
                'MakeNamedMaterial "snow:road" "string type" "coateddiffuse" "spectrum reflectance" '
                '"spd/snow_road.spd" "float roughness" [0.08] "float eta" [1.333] "float thickness" [0.001]'
            )
        else:
            lines.append(
                f'MakeNamedMaterial "snow:{tag}" "string type" "diffuse" "spectrum reflectance" "spd/snow_{tag}.spd"'
            )
    return lines, meta


def road_rects(geo: dict, state: str, rng: np.random.Generator, z0: float, z1: float) -> list[tuple]:
    """Snow-covered rectangles on the road (x0, x1, za, zb): everything, or all but the wheel tracks."""
    if state == "packed":
        return [(a, b, z0, z1) for a, b in geo["paved"]]
    rects = []
    tracks = sorted(c + s * TRACK_HALF_SEP for c in geo["lane_centres"] for s in (-1, 1))
    zs = list(np.arange(z0, 120.0, 1.5)) + list(np.arange(120.0, z1, 6.0)) + [z1]
    for za, zb in zip(zs[:-1], zs[1:], strict=True):
        for a, b in geo["paved"]:
            x = a
            for t in tracks:
                if not a < t < b:
                    continue
                w = TRACK_W * rng.uniform(0.8, 1.25)
                c = t + rng.uniform(-0.08, 0.08)
                if c - w / 2 > x:
                    rects.append((x, c - w / 2, za, zb))
                x = c + w / 2
            if x < b:
                rects.append((x, b, za, zb))
    return rects


def bank_profile(rng: np.random.Generator, width: float) -> list[tuple[float, float]]:
    h = rng.uniform(0.3, 0.45)
    xs = np.linspace(0.0, width, 7)
    ys = h * np.clip(np.sin(np.pi * xs / width), 0.0, None) ** 0.7
    return [(float(a), float(b)) for a, b in zip(xs, ys, strict=True)]


def flakes(args, eye: np.ndarray, fov_deg: float, aspect: float, rng: np.random.Generator) -> np.ndarray:
    """(x, y, z, radius) of flakes in the view pyramid (axis +z) from 0.5 m to the range."""
    if args.snowfall_mmh <= 0:
        return np.zeros((0, 4))
    n0, lam = gunn_marshall(args.snowfall_mmh)
    dens = n0 / lam * math.exp(-lam * FLAKE_DMIN_MM)  # m^-3 with D >= Dmin
    th = math.tan(math.radians(min(fov_deg, 150.0)) / 2) * 1.05
    tw = th * max(aspect, 1.0)
    th = th * min(aspect, 1.0) if aspect < 1 else th
    r = args.snowfall_range_m
    vol = 4.0 * tw * th * (r**3 - 0.5**3) / 3.0
    n = int(dens * vol)
    if n > MAX_FLAKES:
        raise SystemExit(f"{n} flakes > {MAX_FLAKES}: lower --snowfall-range-m or --snowfall-mmh")
    z = (0.5**3 + rng.random(n) * (r**3 - 0.5**3)) ** (1.0 / 3.0)
    x = (2 * rng.random(n) - 1) * tw * z
    y = (2 * rng.random(n) - 1) * th * z
    d_melt = FLAKE_DMIN_MM + rng.exponential(1.0 / lam, n)
    rad = 0.5e-3 * d_melt * (1.0 / args.snowflake_density) ** (1.0 / 3.0)
    p = np.stack([eye[0] + x, eye[1] + y, eye[2] + z, rad], -1)
    return p[p[:, 1] > p[:, 3]]  # above the road


def scene_lines(
    args, wl: np.ndarray, spd_dir: Path, out_dir: Path, geo: dict, eye: np.ndarray, mesh, quads_mesh, extrude_profile
) -> tuple[list[str], list[str], dict]:
    """Returns (material lines, geometry lines, manifest). ``geo``: paved [(x0, x1)], lane_centres,
    banks [(x_edge, side)], z0, z1, fov_deg, aspect."""
    rng = np.random.default_rng([7 if args.seed is None else args.seed, 0x5A0])
    mats, meta = materials(args, wl, spd_dir)
    geom: list[str] = []
    m: dict = {
        "state": args.snow,
        "albedo_model": "ART spherical albedo, Kokhanovsky & Zege 2004; ice m_im Warren & Brandt 2008; Lambertian",
        "surfaces": meta,
    }
    if args.snow != "none" and args.snowbanks:
        geom.append('NamedMaterial "snow:verge"')
        for x_edge, side in geo["banks"]:
            for za in np.arange(geo["z0"], 600.0, 25.0):
                prof = bank_profile(rng, rng.uniform(0.45, 0.6))
                p, t = extrude_profile(prof, x_edge + side * 0.02, float(za), float(za) + 25.0, flip=side < 0)
                geom += mesh(p, t)
        m["banks"] = {"height_m": [0.3, 0.45], "between": "paved edge and guard rail"}
    if args.snow in ROAD_DIRT:
        rects = road_rects(geo, args.snow, rng, geo["z0"], geo["z1"])
        p, t = quads_mesh(rects, 0.008)
        geom += ['NamedMaterial "snow:road"', *mesh(p, t)]
        m["road"] = {
            "cover": "whole paved surface (markings hidden)" if args.snow == "packed" else "outside the wheel tracks",
            **(
                {"wheel_tracks": {"offset_from_lane_centre_m": TRACK_HALF_SEP, "width_m": TRACK_W}}
                if args.snow == "slush"
                else {}
            ),
        }
    if args.snowfall_mmh > 0:
        f = flakes(args, eye, geo["fov_deg"], geo["aspect"], rng)
        inc = out_dir / "snowfall.pbrt"
        inc.write_text(
            "".join(
                f'AttributeBegin Translate {x:.4f} {y:.4f} {z:.4f} Shape "sphere" "float radius" [{r:.5f}] AttributeEnd\n'
                for x, y, z, r in f
            )
        )
        beta = snowfall_extinction(args.snowfall_mmh, args.snowflake_density)
        geom += ['NamedMaterial "snow:flake"', f'Include "{os.path.relpath(inc, out_dir)}"']
        m["snowfall"] = {
            "rate_mm_per_h_melted": args.snowfall_mmh,
            "flakes": int(len(f)),
            "range_m": args.snowfall_range_m,
            "min_melted_diameter_mm": FLAKE_DMIN_MM,
            "flake_density_g_cm3": args.snowflake_density,
            "size_distribution": "Gunn & Marshall 1958 (melted diameter)",
            "extinction_per_m": round(beta, 6),
            "equivalent_visibility_m": round(2.996 / beta, 1),
            "static": True,
        }
    m["approximations"] = [
        "Snow is Lambertian with the ART spherical albedo: no forward-scattering BRDF, no sub-surface blur.",
        "Trees, cars and signs carry no snow; plough banks are smooth extrusions.",
        "Falling flakes are static spheres (no motion streaks); beyond the flake range only the optional haze.",
    ]
    return mats, geom, m
