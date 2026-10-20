#!/usr/bin/env python3
"""Falling rain for the highway builder (opt-in, ``--rain-mmh``).

Drops are water spheres (``dielectric``, eta 1.333) falling at terminal velocity during the pbrt
shutter, so each one is motion-blurred into a streak whose length and faintness follow from the
exposure time (Garg & Nayar 2007, Int. J. Comput. Vis. 75, 3). Rain needs a shutter: without
``--exposure-s`` it uses ``DEFAULT_EXPOSURE_S``.

* Size distribution: Marshall & Palmer (1948, J. Meteorol. 5, 165),
  N(D) = 8000 exp(-Lambda D) m^-3 mm^-1, Lambda = 4.1 R^-0.21 mm^-1 (R in mm/h). Only drops above
  ``--rain-dmin-mm`` are placed; smaller ones are sub-pixel and add only haze.
* Terminal velocity: Atlas, Srivastava & Sekhon (1973, Rev. Geophys. 11, 1),
  v = 9.65 - 10.3 exp(-0.6 D) m/s (D in mm); ``--rain-wind-ms`` adds a horizontal drift.
* Extinction (manifest): geometric optics, 2 x cross-section integrated over the full
  distribution; beyond ``--rain-range-m`` add ``--haze mist --visibility-m`` for it.

Approximations: spherical drops (no oblate shape, no oscillation), no splashes or ripples on the
road (the road is wet, ``--road-wetness wet``), drops start at random heights inside the view
pyramid.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

DEFAULT_EXPOSURE_S = 0.004
MAX_DROPS = 1_500_000
WATER_ETA = 1.333


def add_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("falling rain (tools/highway_rain.py)")
    g.add_argument(
        "--rain-mmh", type=float, default=0.0, help="Rain rate [mm/h]; 0 = off. 2 light, 10 heavy, 50 cloudburst."
    )
    g.add_argument("--rain-range-m", type=float, default=12.0, help="Drops are placed from 0.5 m out to this range.")
    g.add_argument("--rain-dmin-mm", type=float, default=0.5, help="Smallest drop placed [mm].")
    g.add_argument("--rain-wind-ms", type=float, default=0.0, help="Crosswind drift of the drops [m/s] (+x).")
    g.add_argument("--rain-seed", type=int, default=7)


def enabled(args: argparse.Namespace) -> bool:
    return getattr(args, "rain_mmh", 0.0) > 0


def check_args(args: argparse.Namespace) -> list[str]:
    """Validate; rain needs a shutter and a wet road. Returns notes for the manifest."""
    if args.rain_mmh < 0 or args.rain_range_m <= 0.5 or args.rain_dmin_mm <= 0:
        raise SystemExit("--rain-mmh >= 0, --rain-range-m > 0.5, --rain-dmin-mm > 0")
    notes = []
    if not enabled(args):
        return notes
    if getattr(args, "exposure_s", None) is None:
        args.exposure_s = DEFAULT_EXPOSURE_S
        notes.append(f"--rain-mmh sets --exposure-s {DEFAULT_EXPOSURE_S} (drops streak during the shutter)")
    if getattr(args, "road_wetness", "dry") in ("dry", "damp"):
        args.road_wetness = "wet"
        notes.append("--rain-mmh sets --road-wetness wet")
    return notes


def marshall_palmer(rate_mmh: float) -> tuple[float, float]:
    """(N0 [m^-3 mm^-1], Lambda [mm^-1])."""
    return 8000.0, 4.1 * rate_mmh**-0.21


def terminal_velocity(d_mm: np.ndarray | float) -> np.ndarray | float:
    return np.maximum(9.65 - 10.3 * np.exp(-0.6 * np.asarray(d_mm)), 0.0)


def rain_extinction(rate_mmh: float) -> float:
    """[m^-1] = 2 * integral N(D) pi D^2 / 4 dD."""
    if rate_mmh <= 0:
        return 0.0
    n0, lam = marshall_palmer(rate_mmh)
    return 2.0 * math.pi / 4.0 * n0 * 2.0 / lam**3 * 1e-6


def drops(args, eye: np.ndarray, fov_deg: float, aspect: float, rng: np.random.Generator) -> np.ndarray:
    """(x, y, z, radius, v_fall) of drops in the view pyramid (axis +z) from 0.5 m to the range."""
    n0, lam = marshall_palmer(args.rain_mmh)
    dens = n0 / lam * math.exp(-lam * args.rain_dmin_mm)
    t = math.tan(math.radians(min(fov_deg, 150.0)) / 2) * 1.1
    tw, th = (t * aspect, t) if aspect >= 1 else (t, t / aspect)
    r, r0 = args.rain_range_m, 0.5
    n = int(dens * 4.0 * tw * th * (r**3 - r0**3) / 3.0)
    if n > MAX_DROPS:
        raise SystemExit(f"{n} drops > {MAX_DROPS}: lower --rain-range-m or raise --rain-dmin-mm")
    z = (r0**3 + rng.random(n) * (r**3 - r0**3)) ** (1.0 / 3.0)
    x = (2 * rng.random(n) - 1) * tw * z
    y = (2 * rng.random(n) - 1) * th * z
    d = args.rain_dmin_mm + rng.exponential(1.0 / lam, n)
    p = np.stack([eye[0] + x, eye[1] + y, eye[2] + z, 0.5e-3 * d, terminal_velocity(d)], -1)
    return p[p[:, 1] - p[:, 4] * args.exposure_s > p[:, 3]]


def scene_lines(
    args, out_dir: Path, eye: np.ndarray, fov_deg: float, aspect: float, notes: list[str]
) -> tuple[list[str], dict]:
    """World-block lines (material + Include) and the manifest entry."""
    t = args.exposure_s
    p = drops(args, eye, fov_deg, aspect, np.random.default_rng(args.rain_seed))
    with open(out_dir / "rain.pbrt", "w") as f:
        for x, y, z, rad, v in p:
            f.write(
                f"AttributeBegin Translate {x:.4f} {y:.4f} {z:.4f} ActiveTransform EndTime "
                f"Translate {args.rain_wind_ms * t:.5f} {-v * t:.5f} 0 ActiveTransform All "
                f'Shape "sphere" "float radius" [{rad:.6f}] AttributeEnd\n'
            )
    lines = [
        "# falling rain (tools/highway_rain.py)",
        "AttributeBegin",
        f'MakeNamedMaterial "rain:water" "string type" "dielectric" "float eta" [{WATER_ETA}]',
        'NamedMaterial "rain:water"',
        'Include "rain.pbrt"',
        "AttributeEnd",
    ]
    beta = rain_extinction(args.rain_mmh)
    n0, lam = marshall_palmer(args.rain_mmh)
    meta = {
        "rate_mm_per_h": args.rain_mmh,
        "drops": len(p),
        "range_m": args.rain_range_m,
        "min_diameter_mm": args.rain_dmin_mm,
        "size_distribution": "Marshall & Palmer 1948",
        "lambda_per_mm": round(lam, 4),
        "terminal_velocity": "Atlas et al. 1973",
        "exposure_s": t,
        "streak_length_m_at_2mm": round(float(terminal_velocity(2.0)) * t, 4),
        "wind_ms": args.rain_wind_ms,
        "extinction_per_m": round(beta, 6),
        "equivalent_visibility_m": round(2.996 / beta, 1),
        "approximations": [
            "Spherical drops, no oscillation; no splashes or ripples on the road.",
            "Beyond the drop range only the optional haze.",
        ],
        "notes": notes,
    }
    return lines, meta
