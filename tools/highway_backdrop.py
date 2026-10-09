#!/usr/bin/env python3
"""Distant terrain for the highway scenes: a ring of hills out to the horizon.

Without it the ground stops at a few km and the horizon is a sharp line against the sky,
so aerial perspective (haze, see tools/highway_atmosphere.py) has nothing to act on. The
backdrop is a polar heightfield centred on the ego position, from ``r0`` (2.5 km, where it
blends under the near terrain) to ``r1`` (30 km): low rolling hills a few km out rising to
~1 km ridges far away, with a valley around the road direction. Shapes come from a sum of
seeded sinusoids (deterministic for a given ``--seed``).

Cover is a stochastic mix (pbrt ``fbm`` texture, ~0.5 km features) of forest canopy and the
scene's grass spectrum. The canopy reflectance is a smooth fit to closed-canopy spectra
(visible 0.02-0.06 with a green peak, red edge to ~0.3 in the NIR; e.g. MODIS forest
albedo, Moody et al. 2005; ECOSTRESS/ASTER canopy library), darker than single leaves
because of inter-crown shadowing.
"""

from __future__ import annotations

import argparse

import numpy as np

R0_M, R1_M = 2_500.0, 30_000.0
R_EFF_EARTH_M = 7.0 / 6.0 * 6.371e6


def add_cli_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--distant-terrain",
        choices=("auto", "none", "hills"),
        default="auto",
        help="Hills out to the horizon (auto: on when --haze is set).",
    )


def enabled(args: argparse.Namespace) -> bool:
    mode = getattr(args, "distant_terrain", "auto")
    return mode == "hills" or (mode == "auto" and getattr(args, "haze", "none") != "none")


def forest_canopy_reflectance(wl_nm) -> np.ndarray:
    wl = np.asarray(wl_nm, dtype=np.float64)
    green = 0.035 * np.exp(-0.5 * ((wl - 552.0) / 30.0) ** 2)
    red_well = 0.008 * np.exp(-0.5 * ((wl - 672.0) / 15.0) ** 2)
    edge = 0.28 / (1.0 + np.exp(-(wl - 718.0) / 12.0))
    return np.clip(0.025 + green - red_well + edge, 0.01, 1.0)


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def hills_height(r: np.ndarray, phi: np.ndarray, seed: int) -> np.ndarray:
    """Height [m] at radius r and azimuth phi (0 = road direction +z, + towards +x).

    Ridges are specified by the elevation angle they subtend from the road (peaks ~0.8 deg
    at 3 km rising to ~2.2 deg at 30 km, i.e. ~40 m hills to ~1.1 km ridges), so successive
    ranges stay visible behind each other, minus the Earth-curvature drop r^2 / 2R_eff with
    standard refraction (R_eff = 7/6 R_earth).
    """
    rng = np.random.default_rng(seed + 9173)
    n = np.zeros(np.broadcast(r, phi).shape)
    amp_sum = 0.0
    for octave, m in enumerate((3, 5, 9, 16, 29, 53)):
        a = 0.6**octave
        kr = rng.uniform(0.4, 1.6) * (octave + 1) / 4000.0
        n = n + a * np.sin(m * phi + kr * r + rng.uniform(0, 2 * np.pi))
        amp_sum += a
    n = 0.5 + 0.5 * n / amp_sum  # 0..1
    peak_deg = 0.8 + 1.4 * _smoothstep(R0_M, R1_M, r)
    valley = 0.55 + 0.45 * _smoothstep(0.03, 0.3, np.abs(np.angle(np.exp(1j * phi))))
    relief = r * np.tan(np.radians(peak_deg)) * n**1.3 * valley
    return -0.3 + _smoothstep(R0_M, R0_M + 1500.0, r) * relief - r**2 / (2.0 * R_EFF_EARTH_M)


def hills_mesh(cx: float, cz: float, seed: int, n_r: int = 64, n_phi: int = 720):
    """(points, triangles, normals) of the backdrop ring."""
    r = np.geomspace(R0_M, R1_M, n_r)
    phi = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)
    R, PHI = np.meshgrid(r, phi, indexing="ij")  # (n_r, n_phi)
    Y = hills_height(R, PHI, seed)
    P = np.stack([cx + R * np.sin(PHI), Y, cz + R * np.cos(PHI)], -1).reshape(-1, 3)
    i = np.arange(n_r - 1)[:, None]
    j = np.arange(n_phi)[None, :]
    jn = (j + 1) % n_phi
    a, b, c, d = i * n_phi + j, i * n_phi + jn, (i + 1) * n_phi + j, (i + 1) * n_phi + jn
    tri = np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3), np.stack([b, c, d], -1).reshape(-1, 3)])
    fn = np.cross(P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]])
    fn *= np.sign(fn[:, 1:2] + 1e-30)  # face up
    N = np.zeros_like(P)
    for k in range(3):
        np.add.at(N, tri[:, k], fn)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    return P, tri, N


def pbrt_lines(mesh_fn, cx: float, cz: float, seed: int, canopy_spd: str, grass_spd: str) -> list[str]:
    """pbrt block for the backdrop (``mesh_fn`` = build_highway_scene.mesh)."""
    P, tri, N = hills_mesh(cx, cz, seed)
    return [
        f"# Distant terrain: hills {R0_M / 1000:g}-{R1_M / 1000:g} km (tools/highway_backdrop.py)",
        "AttributeBegin",
        f'    MakeNamedMaterial "hills:forest" "string type" "diffuse" "spectrum reflectance" "{canopy_spd}"',
        f'    MakeNamedMaterial "hills:open" "string type" "diffuse" "spectrum reflectance" "{grass_spd}"',
        "    AttributeBegin",
        "        Scale 0.002 0.002 0.002",
        '        Texture "hills:cover" "float" "fbm" "integer octaves" [5] "float roughness" [0.55]',
        "    AttributeEnd",
        '    MakeNamedMaterial "hills" "string type" "mix" "string materials" ["hills:forest" "hills:open"]',
        '        "texture amount" "hills:cover"',
        '    NamedMaterial "hills"',
        *("    " + ln for ln in mesh_fn(P, tri, None, N)),
        "AttributeEnd",
    ]
