"""Wet road, puddles, wet retroreflective markings and vehicle spray for the highway scenes.

Opt-in (``--road-wetness damp|wet|flooded``, ``--puddles``, ``--spray``); with the defaults
(``dry``, off) nothing here touches the scene. Hooks in tools/build_highway_scene.py:
``apply_materials`` (rewrites the road/marking materials after they are written) and
``spray_lines`` (one participating-medium volume per moving vehicle).

Wet asphalt
    A water film on a rough porous surface has two effects (Lekner & Dorf 1988, "Why some
    things are darker when wet", Appl. Opt. 27, 1278-1280, doi:10.1364/AO.27.001278, extending
    Angstrom 1925): (1) light diffusely reflected by the substrate is trapped by total internal
    reflection at the water/air surface and gets further chances of absorption, so the body
    (off-specular) albedo drops to

        A_w = (1 - r_e)(1 - r_i) A / (1 - r_i A),   r_i = 1 - (1 - r_e) / n^2,

    with r_e the hemispherical (cosine-weighted) Fresnel reflectance of water (n = 1.333:
    r_e = 0.0664, r_i = 0.475); (2) the water surface is a smooth dielectric (n = 1.333) whose
    Fresnel specular lobe dominates at the grazing driver geometry. The model is applied per
    wavelength to the scene's spectral asphalt (and paint) reflectance: the wet base reflectance
    is chosen so that the rendered wet body reflectance equals the dry material's rendered body
    reflectance times A_w / A, i.e. the published darkening ratio relative to the scene's dry
    look (the dry material keeps its pbrt ``coateddiffuse`` n = 1.5 binder sheen). The wet
    material is pbrt ``coateddiffuse`` (layered random walk == the Angstrom/Lekner-Dorf
    geometry) with ``eta`` 1.333 and a water-surface roughness per wetness level.

    Water-surface roughness is calibrated so the CIE average luminance coefficient Q0 of the
    model falls in the range of the CIE wet-road W classes (CIE 47-1979 "Road lighting for wet
    conditions": W1..W4 Q0 = 0.11, 0.15, 0.21, 0.25; dry R/N/C classes Q0 = 0.07-0.10, see
    ``CIE_Q0``); ``q0`` integrates the model's luminance coefficient q = f_r over the CIE r-table
    incidence solid angle (tan(gamma) <= 12) at the 1 deg observation angle.

Puddles
    Water ponds in the wheel-path ruts: the mask is the road-wear wheel-path map of
    tools/highway_road_wear.py (lateral Gaussian wheel tracks with longitudinal wander) above a
    wetness-dependent level, modulated by low-frequency longitudinal unevenness. Puddle surfaces
    are optically flat water (roughness 0) over the darkened asphalt; the puddle surface is flush
    with the road (no depression geometry).

Wet markings
    Glass-bead retroreflection collapses when water covers the beads. EN 1436:2018 specifies
    R_L in the wet-recovery condition (RW classes, measured 60 s after flooding; = ASTM E2177
    "condition of wet recovery") and in continuous rain (RR classes; ASTM E2176). Wet/damp use
    RW, flooded uses RR; class minima RW1/RR1 = 25, RW2/RR2 = 35, RW3/RR3 = 50 mcd m^-2 lx^-1.
    FHWA's pavement-marking synthesis (FHWA-SA-... ch. 5) reports that beads-on-paint wet-recovery
    and rain detection distances drop to 28 % and 17 % of dry.

Vehicle spray
    Spray clouds measured behind heavy vehicles on wet asphalt (Otxoterena Drake, Willstrand,
    Andersson & Biswanger 2021, J. Wind Eng. Ind. Aerodyn. 217, 104734,
    doi:10.1016/j.jweia.2021.104734): mean droplet diameter 100-400 um, number density
    20-40 cm^-3, light extinction up to 0.2 m^-1. Droplets >> wavelength: extinction efficiency
    2, spectrally flat, single-scattering albedo ~1, strongly forward-peaked phase function;
    modelled as Henyey-Greenstein with g = 0.85 (geometric-optics asymmetry of water drops,
    Hansen & Travis 1974, Space Sci. Rev. 16, 527). Spray volume per vehicle: a ``uniformgrid``
    medium behind the rear axle, two wheel-track plumes decaying downstream and rising.
    Speed/wetness scaling (FHWA-HRT-15-062: water film thickness and speed are the main
    factors) is an ASSUMED linear law above a 30 km/h onset; the peak extinction 0.2 m^-1 is
    assigned to a heavy vehicle at 90 km/h on a ``wet`` road, passenger cars get half.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np

N_WATER = 1.333
N_COAT_DRY = 1.5  # pbrt coateddiffuse default eta used by the dry road/paint materials
LEVELS = ("dry", "damp", "wet", "flooded")
# water-surface roughness (pbrt "roughness", alpha = sqrt) -> model Q0 (see q0): damp 0.07,
# wet 0.14 (~W2), flooded 0.24 (~W4)
ROUGHNESS = {"damp": 0.15, "wet": 0.05, "flooded": 0.02}
# pbrt LayeredBxDF attenuates by exp(-thickness/|cos|) even with albedo 0: keep the clear water lossless
THICKNESS = 1e-4
PUDDLE_LEVEL = {
    "wet": 0.8,
    "flooded": 0.6,
}  # on the rut map normalised to its maximum  # wheel-path map level above which water ponds
SPRAY_WATER = {"damp": 0.0, "wet": 1.0, "flooded": 2.0}  # assumed spray ~ free water film
# CIE 47-1979 / CIE 144 average luminance coefficients Q0 (cd m^-2 lx^-1)
CIE_Q0 = {"R1": 0.10, "R2": 0.07, "R3": 0.07, "R4": 0.08, "W1": 0.11, "W2": 0.15, "W3": 0.21, "W4": 0.25}
# EN 1436:2018 wet R_L class minima [mcd m^-2 lx^-1]; RW = wet recovery, RR = rain
EN1436_WET_CLASSES = {"RW1": 25.0, "RW2": 35.0, "RW3": 50.0, "RW4": 75.0, "RW5": 100.0, "RW6": 150.0}
EN1436_RAIN_CLASSES = {"RR1": 25.0, "RR2": 35.0, "RR3": 50.0, "RR4": 75.0, "RR5": 100.0, "RR6": 150.0}
WET_MARKING_CLASS = {
    "damp": {"white": "RW2", "yellow": "RW1"},
    "wet": {"white": "RW2", "yellow": "RW1"},
    "flooded": {"white": "RR1", "yellow": "RR1"},
}
SPRAY_EXT_REF = 0.2  # m^-1, Otxoterena Drake et al. 2021 (heavy vehicles)
SPRAY_V_REF_KMH, SPRAY_V_ONSET_KMH = 90.0, 30.0  # assumed
SPRAY_G = 0.85
SPRAY_CAR_FACTOR = 0.5
SPRAY_SEGMENTS = 1  # homogeneous slabs along the plume (see spray_lines)
SPRAY_FLOOR = 0.5  # box floor below the road (see spray_lines)
WET_SPDS = ("asphalt_aged", "asphalt_new", "asphalt_patch", "crack_sealant", "paint_road_white", "paint_road_yellow")
ROAD_MATERIALS = ("asphalt", "rw:asphaltA", "rw:asphaltB", "rw:sealant", "paint_white", "paint_yellow")


# ------------------------------------------------------------------ optics
def fresnel(cos_i, n: float) -> np.ndarray:
    """Unpolarised Fresnel reflectance from air into a dielectric of index ``n``."""
    c = np.clip(np.asarray(cos_i, dtype=np.float64), 0.0, 1.0)
    ct = np.sqrt(np.clip(1.0 - (1.0 - c**2) / n**2, 0.0, 1.0))
    rs = (c - n * ct) / (c + n * ct)
    rp = (n * c - ct) / (n * c + ct)
    return 0.5 * (rs**2 + rp**2)


def diffuse_reflectances(n: float, m: int = 20000) -> tuple[float, float]:
    """(r_e, r_i): external and internal hemispherical reflectance for diffuse light."""
    c = (np.arange(m) + 0.5) / m
    r_e = float(np.mean(fresnel(c, n) * 2.0 * c))
    return r_e, 1.0 - (1.0 - r_e) / n**2


def body_albedo(a, n: float) -> np.ndarray:
    """Angstrom / Lekner-Dorf body albedo of a Lambertian substrate ``a`` under a smooth film ``n``."""
    r_e, r_i = diffuse_reflectances(n)
    a = np.asarray(a, dtype=np.float64)
    return (1.0 - r_e) * (1.0 - r_i) * a / (1.0 - r_i * a)


def darkening_ratio(a, n: float = N_WATER) -> np.ndarray:
    """Wet / dry body albedo of a Lambertian surface (Lekner & Dorf 1988)."""
    a = np.asarray(a, dtype=np.float64)
    return body_albedo(a, n) / np.maximum(a, 1e-12)


def _invert_body(target, n: float) -> np.ndarray:
    r_e, r_i = diffuse_reflectances(n)
    t = (1.0 - r_e) * (1.0 - r_i)
    return target / (t + r_i * target)


def wet_base_reflectance(a, coated: bool = True) -> np.ndarray:
    """Base reflectance for the wet material (water coat n = 1.333) from the dry one.

    ``coated``: the dry material is pbrt coateddiffuse (n = 1.5 coat); the wet body reflectance
    is the dry rendered body x the Lekner-Dorf ratio. Uncoated (Lambertian, e.g. the diffuse part
    of the retroreflective markings): base = Lekner-Dorf wet albedo directly.
    """
    a = np.asarray(a, dtype=np.float64)
    if not coated:
        return body_albedo(a, N_WATER)
    return _invert_body(body_albedo(a, N_COAT_DRY) * darkening_ratio(a), N_WATER)


def body_brdf(a, n: float, cos_i, cos_o) -> np.ndarray:
    _, r_i = diffuse_reflectances(n)
    return a * (1 - fresnel(cos_i, n)) * (1 - fresnel(cos_o, n)) / (math.pi * n**2 * (1 - r_i * a))


def ggx_reflection(cos_i, cos_o, cos_phi, n: float, roughness: float) -> np.ndarray:
    """Microfacet (Trowbridge-Reitz, pbrt alpha = sqrt(roughness)) dielectric reflection BRDF."""
    ci, co, cp = np.broadcast_arrays(*(np.asarray(v, dtype=np.float64) for v in (cos_i, cos_o, cos_phi)))
    a2 = max(roughness, 1e-8)  # alpha^2 = roughness
    si, so = np.sqrt(1 - ci**2), np.sqrt(1 - co**2)
    wi = np.stack([si, np.zeros_like(si), ci])
    wo = np.stack([so * cp, so * np.sqrt(np.clip(1 - cp**2, 0, 1)), co])
    h = wi + wo
    h /= np.linalg.norm(h, axis=0)
    d = a2 / (math.pi * (h[2] ** 2 * (a2 - 1) + 1) ** 2)

    def lam(c):
        return (-1 + np.sqrt(1 + a2 * (1 - c**2) / c**2)) / 2

    g = 1 / (1 + lam(ci) + lam(co))
    return fresnel((wi * h).sum(0), n) * d * g / (4 * ci * co)


def luminance_coefficient(a, n, roughness, cos_i, cos_o, cos_phi, body_scale: float = 1.0) -> np.ndarray:
    """q = L / E_horizontal = f_r for the layered model (``cos_phi`` = -1: light behind the point)."""
    return body_scale * body_brdf(a, n, cos_i, cos_o) + ggx_reflection(cos_i, cos_o, cos_phi, n, roughness)


def q0(level: str, a: float = 0.12) -> float:
    """CIE average luminance coefficient Q0 of the modelled surface (``dry`` = the scene's dry asphalt).

    Mean of q over incidence directions tan(gamma) <= 12, all azimuths, observer 1 deg above the
    surface (CIE 30.2 / CIE 144 r-table geometry); light azimuth beta = 0 is the observer's side.
    """
    co = math.sin(math.radians(1.0))
    gam = np.linspace(0, math.atan(12.0), 400)[:, None]
    beta = np.linspace(0, math.pi, 361)[None, :]
    ci = np.cos(gam) + 0 * beta
    if level == "dry":
        q = luminance_coefficient(a, N_COAT_DRY, 0.35, ci, co, -np.cos(beta))
    else:
        scale = float(darkening_ratio(a) * body_albedo(a, N_COAT_DRY) / body_albedo(a, N_WATER))
        q = luminance_coefficient(a, N_WATER, ROUGHNESS[level], ci, co, -np.cos(beta), body_scale=scale)
    w = np.sin(gam) + 0 * beta
    return float((q * w).sum() / w.sum())


# ------------------------------------------------------------------ markings
def wet_marking_rl(level: str, override: float | None = None) -> dict[str, float]:
    if override is not None:
        return {"white": float(override), "yellow": float(override)}
    cls = WET_MARKING_CLASS[level]
    table = {**EN1436_WET_CLASSES, **EN1436_RAIN_CLASSES}
    return {c: table[k] for c, k in cls.items()}


# ------------------------------------------------------------------ spray
def spray_extinction(speed_kmh: float, level: str, heavy: bool) -> float:
    """Peak spray extinction coefficient [m^-1] behind a vehicle (see module docstring)."""
    v = max(0.0, (abs(speed_kmh) - SPRAY_V_ONSET_KMH) / (SPRAY_V_REF_KMH - SPRAY_V_ONSET_KMH))
    return SPRAY_EXT_REF * SPRAY_WATER.get(level, 0.0) * v * (1.0 if heavy else SPRAY_CAR_FACTOR)


def nominal_speed_kmh(lane: int) -> float:
    """Spray speed for vehicles of a static scene: the lane speeds of tools/highway_incar.py."""
    import highway_incar

    return highway_incar.LANE_SPEEDS_KMH.get(lane, 100.0) if lane >= 0 else highway_incar.ONCOMING_SPEED_KMH


def spray_length_m(speed_kmh: float) -> float:
    """Plume length: ~0.6 s of travel (assumed), at least 3 m."""
    return max(3.0, 0.6 * abs(speed_kmh) / 3.6)


def spray_density(nx: int, ny: int, nz: int, width: float, height: float, length: float, y0: float = 0.0) -> np.ndarray:
    """Relative density (peak 1) on the grid, index (z * ny + y) * nx + x; z = 0 at the rear axle.

    The grid spans ``y0 .. height``; cells below the road surface (y < 0) are empty.
    """
    x = (np.arange(nx) + 0.5) / nx * width - width / 2
    y = y0 + (np.arange(ny) + 0.5) / ny * (height - y0)
    z = (np.arange(nz) + 0.5) / nz * length
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    track = width / 2 - 0.45
    sx = 0.25 + 0.08 * Z
    sy = 0.3 + 0.1 * Z
    lat = np.exp(-0.5 * ((np.abs(X) - track) / sx) ** 2)
    rho = lat * np.exp(-0.5 * (Y / sy) ** 2) * (Y >= 0) * np.exp(-Z / (0.35 * length)) * (0.3 + 0.7 * np.exp(-Z / 2.0))
    return rho / rho.max()


# ------------------------------------------------------------------ CLI
def add_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("wet road / spray (tools/highway_wet.py)")
    g.add_argument("--road-wetness", choices=LEVELS, default="dry", help="Water on the road surface.")
    g.add_argument("--puddles", action="store_true", help="Water ponding in the wheel-path ruts (wet/flooded).")
    g.add_argument("--spray", action="store_true", help="Spray volumes behind moving vehicles (wet/flooded).")
    g.add_argument(
        "--wet-marking-rl", type=float, default=None, help="Wet marking R_L [mcd/m^2/lx] (default: EN 1436 class)."
    )


def enabled(args: argparse.Namespace) -> bool:
    return getattr(args, "road_wetness", "dry") != "dry"


def check_args(args: argparse.Namespace, has_wear: bool) -> None:
    lvl = getattr(args, "road_wetness", "dry")
    if (args.puddles or args.spray) and lvl in ("dry", "damp"):
        raise SystemExit("--puddles/--spray need --road-wetness wet or flooded")
    if args.puddles and not has_wear:
        raise SystemExit("--puddles uses the road-wear wheel-path map: needs --road-wear != none")


# ------------------------------------------------------------------ pbrt materials
def _write_spd(path: Path, wl: np.ndarray, v: np.ndarray) -> None:
    path.write_text("\n".join(f"{w:.1f} {x:.6g}" for w, x in zip(wl, v)) + "\n")


def _blocks(lines: list[str]):
    """Yield (start, end) index ranges of statements (first line + indented continuations)."""
    i = 0
    while i < len(lines):
        j = i + 1
        while j < len(lines) and lines[j].startswith("    "):
            j += 1
        yield i, j
        i = j


def _set_param(block: str, decl: str, value: str) -> str:
    """Replace (or append) a ``"<type> <name>" [..]``/``"texture <name>" ".."`` parameter."""
    import re

    name = decl.split()[1]
    block = re.sub(rf'\s*"(float|texture) {name}" (\[[^\]]*\]|"[^"]*")', "", block)
    return f'{block} "{decl}" {value}'


def apply_materials(
    lines: list[str], args: argparse.Namespace, wl: np.ndarray, spd_dir: Path, out_dir: Path, wear=None
) -> tuple[list[str], dict]:
    """Rewrite the road, sealant and marking materials for a wet road; returns (lines, manifest)."""
    import re

    import highway_night as night

    lvl = args.road_wetness
    rough = ROUGHNESS[lvl]
    for name in WET_SPDS:
        src = spd_dir / f"{name}.spd"
        if src.is_file():
            d = np.loadtxt(src)
            _write_spd(spd_dir / f"wet_{name}.spd", d[:, 0], d[:, 1] * 0 + wet_base_reflectance(d[:, 1]))
            _write_spd(spd_dir / f"wetd_{name}.spd", d[:, 0], body_albedo(d[:, 1], N_WATER))
    spd_re = re.compile(r'"spd/(' + "|".join(WET_SPDS) + r')\.spd"')
    road_mats = set(ROAD_MATERIALS)
    out: list[str] = []
    for i, j in _blocks(lines):
        blk = lines[i:j]
        head = blk[0]
        m = re.match(r'(Texture|MakeNamedMaterial) "([^"]+)"', head)
        if m is None:
            out += blk
            continue
        kind, name = m.groups()
        is_road = name in road_mats or name.startswith("rw:") and not name.startswith("rw:rpm")
        if kind == "Texture" and (name == "asphalt" or name.startswith("rw:")):
            out += [spd_re.sub(r'"spd/wet_\1.spd"', ln) for ln in blk]
            continue
        if kind != "MakeNamedMaterial" or not is_road:
            out += blk
            continue
        if '"retroreflective"' in head:
            blk = [spd_re.sub(r'"spd/wetd_\1.spd"', ln) for ln in blk]
        elif '"coateddiffuse"' in head:
            blk = [spd_re.sub(r'"spd/wet_\1.spd"', ln) for ln in blk]
            blk[0] = _set_param(blk[0], "float roughness", f"[{rough:g}]")
            blk[0] = _set_param(blk[0], "float eta", f"[{N_WATER}]")
            blk[0] = _set_param(blk[0], "float thickness", f"[{THICKNESS}]")
            if lvl != "damp":  # the water surface hides the texture's micro-normals
                blk = [ln for ln in blk if '"string normalmap"' not in ln]
        out += blk
    meta: dict = {
        "level": lvl,
        "model": "Lekner & Dorf 1988 water-film darkening + smooth dielectric water surface (n = 1.333), "
        "pbrt coateddiffuse; roughness calibrated to CIE W-class Q0",
        "water_ior": N_WATER,
        "surface_roughness": rough,
        "darkening_ratio_asphalt_aged": round(
            float(np.mean(darkening_ratio(np.loadtxt(spd_dir / "asphalt_aged.spd")[:, 1]))), 4
        )
        if (spd_dir / "asphalt_aged.spd").is_file()
        else None,
        "q0_model": {"dry": round(q0("dry"), 4), lvl: round(q0(lvl), 4)},
        "q0_cie": CIE_Q0,
    }
    # wet markings at night (retroreflective paint): EN 1436 wet/rain classes
    rl = wet_marking_rl(lvl, getattr(args, "wet_marking_rl", None))
    if any('"retroreflective"' in ln for ln in out):
        for name, colour in (("paint_white", "white"), ("paint_yellow", "yellow")):
            f = spd_dir / f"ra_wet_{name}.spd"
            surface = night.RETRO_MATERIALS[name][0]
            _write_spd(f, wl, night.retro_ra_spectrum(wl, surface, night.marking_ra_param(rl[colour])))
            out = [ln.replace(f'"spd/ra_{name}.spd"', f'"{os.path.relpath(f, out_dir)}"') for ln in out]
        meta["markings"] = {
            "rl_mcd_m2_lx": rl,
            "classes": WET_MARKING_CLASS[lvl] if getattr(args, "wet_marking_rl", None) is None else "override",
            "standard": "EN 1436:2018 RW (wet recovery, = ASTM E2177) / RR (rain, = ASTM E2176)",
        }
    if args.puddles:
        out, meta["puddles"] = _puddle_lines(out, wear, lvl)
    return out, meta


def puddle_mask(wheel: np.ndarray, level: float, rng: np.random.Generator, texel_zx=(0.02, 0.02)) -> np.ndarray:
    """Ponding mask from the wheel-path (rut) map: water where rut depth proxy exceeds ``level``."""
    from highway_road_wear import _smoothstep, periodic_noise

    h = wheel.shape[0]
    uneven = periodic_noise(rng, (h, 2), (texel_zx[0], 1.0), (6.0, 1.0))[:, :1]  # longitudinal unevenness
    depth = wheel / max(float(wheel.max()), 1e-6) * (0.85 + 0.15 * uneven)
    return np.clip(_smoothstep(level - 0.04, level + 0.04, depth), 0.0, 1.0).astype(np.float32)


def _puddle_lines(lines: list[str], wear, lvl: str) -> tuple[list[str], dict]:
    from highway_road_wear import write_float_exr

    rng = np.random.default_rng([wear.seed, 0x9DD1E])
    mask = puddle_mask(wear.wheel_map, PUDDLE_LEVEL[lvl], rng, wear.texel_zx)
    f = wear.tex_dir / f"puddle_{wear.level}_s{wear.seed}_{lvl}.exr"
    write_float_exr(f, mask)
    g = wear.geom
    w_m = float(g["x1"] - g["x0"])
    planar = (
        f'"string mapping" "uv" "float uscale" [{wear.tile_m / w_m:.8g}] '
        f'"float vscale" [{-wear.tile_m / g["period"]:.8g}] "float udelta" [{-g["x0"] / w_m:.8g}]'
    )
    refl = (
        '"texture reflectance" "rw:A:refl"'
        if any(ln.startswith('Texture "rw:A:refl"') for ln in lines)
        else ('"spectrum reflectance" "spd/wet_asphalt_aged.spd"')
    )
    new = [
        f'Texture "wet:puddle" "float" "imagemap" "string filename" "{os.path.relpath(f, wear.out_dir)}" {planar}',
        f'MakeNamedMaterial "wet:puddle" "string type" "coateddiffuse" {refl}',
        f'    "float roughness" [0] "float eta" [{N_WATER}] "float thickness" [{THICKNESS}]',
        'MakeNamedMaterial "asphalt" "string type" "mix" "string materials" ["wet:film" "wet:puddle"]'
        ' "texture amount" "wet:puddle"',
    ]
    out: list[str] = []
    for i, j in _blocks(lines):
        blk = lines[i:j]
        if blk[0].startswith('MakeNamedMaterial "asphalt" '):
            out += [blk[0].replace('"asphalt"', '"wet:film"', 1), *blk[1:], *new]
        else:
            out += blk
    return out, {
        "area_fraction": round(float(mask.mean()), 4),
        "level": PUDDLE_LEVEL[lvl],
        "model": "wheel-path rut map (tools/highway_road_wear.py) above a water level; flat water, flush",
    }


# ------------------------------------------------------------------ spray volumes
def spray_lines(
    cars: list[dict], speeds: list[float], heavy: list[bool], args, outside: str, place, eye_z: float = 0.0
) -> tuple[list[str], list[dict]]:
    """Spray media behind every moving vehicle: ``SPRAY_SEGMENTS`` homogeneous slab(s) along the plume.

    Each slab carries the mean extinction of ``spray_density`` over its volume, so the spatial
    structure (downstream decay, wheel tracks, height) is averaged out; one box per vehicle is the
    default (abutting slabs need small gaps between their faces). With spray on, the road is split
    into short strips (``road_rects``).

    The box bottom sits ``SPRAY_FLOOR`` below the road (the road plane then lies inside the medium,
    avoiding a near-coplanar interface face that pbrt's ray offsets on the large road triangles
    cross inconsistently), and plumes are clipped so the camera is never inside one (pbrt has no
    per-camera medium here).
    """
    lvl = args.road_wetness
    nx, ny, nz = 10, 8, 24
    L: list[str] = [f"# Vehicle spray ({lvl} road): HG g = {SPRAY_G}, sigma_a = 0 (tools/highway_wet.py)"]
    meta = []
    for c, v, hv in zip(cars, speeds, heavy):
        ext = spray_extinction(v, lvl, hv)
        if ext <= 0:
            continue
        fwd = 1.0 if c["heading_deg"] % 360 < 90 or c["heading_deg"] % 360 > 270 else -1.0
        rear = c["distance_m"] - fwd * c["length_m"] / 2
        length = spray_length_m(v)
        behind = [
            o["distance_m"] + fwd * o["length_m"] / 2
            for o in cars
            if o is not c
            and abs(o["x"] - c["x"]) < 0.5 * (o["width_m"] + c["width_m"]) + 0.3
            and fwd * (rear - (o["distance_m"] + fwd * o["length_m"] / 2)) > 0
        ]
        if behind:
            length = min(length, min(fwd * (rear - b) for b in behind) - 0.3)
        z0 = rear - fwd * 0.15  # leave a gap to the body
        if fwd * (z0 - eye_z) > 0:
            length = min(length, fwd * (z0 - eye_z) - 0.5)
        if length < 1.0:
            continue
        width = c["width_m"] + 0.6
        height = min(2.5, 0.7 * c["height_m"] + 0.6)
        rho = spray_density(nx, ny, nz, width, height, length)
        seg = rho.reshape(SPRAY_SEGMENTS, -1).mean(axis=1) * ext  # mean extinction per z segment
        dz = length / SPRAY_SEGMENTS
        L += ["AttributeBegin", *place(c["x"], 0.0, z0), *(["Rotate 180 0 1 0"] if fwd < 0 else [])]
        for k, e in enumerate(seg):
            name = f"spray:{c['id']}:{k}"
            L += [
                f'MakeNamedMedium "{name}" "string type" "homogeneous"',
                f'    "spectrum sigma_a" [300 0 900 0] "spectrum sigma_s" [300 1 900 1] "float scale" [{e:.6g}]',
                f'    "float g" [{SPRAY_G}]',
                f'MediumInterface "{name}" "{outside}"',
                'Material "interface"',
                *_box(-width / 2, -SPRAY_FLOOR, -(k + 1) * dz + 0.005, width / 2, height, -k * dz - 0.005),
            ]
        L.append("AttributeEnd")
        meta.append(
            {
                "id": c["id"],
                "speed_kmh": v,
                "heavy": hv,
                "peak_extinction_m": ext,
                "length_m": length,
                "segment_extinction_m": [round(float(e), 5) for e in seg],
            }
        )
    return L + [""], meta


def _box(x0, y0, z0, x1, y1, z1) -> list[str]:
    p = [(x, y, z) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]
    q = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    idx = [i for a, b, c, d in q for i in (a, c, b, a, d, c)]  # outward normals (pbrt: +n = outside)
    pts = " ".join(f"{v:.6g}" for pt in p for v in pt)
    return [f'Shape "trianglemesh" "point3 P" [ {pts} ] "integer indices" [ {" ".join(map(str, idx))} ]']


def road_rects(rects: list[tuple], args) -> list[tuple]:
    """Split the paved road (x0, x1, z0, z1) rectangles into short z strips when spray is on.

    On the default two 1.5 km road triangles pbrt-v4 loses track of the spray medium for camera
    rays that leave a spray box and then hit the road (the road there renders black); 2 m strips
    near the camera, 25 m beyond 100 m, remove it. Without spray the road is left untouched.
    """
    if not getattr(args, "spray", False):
        return rects
    zs = np.r_[np.arange(-40.0, 100.0, 2.0), np.arange(100.0, 1500.0, 25.0)]
    out = []
    for x0, x1, z0, z1 in rects:
        zz = np.unique(np.r_[z0, zs[(zs > z0) & (zs < z1)], z1])
        out += [(x0, x1, float(a), float(b)) for a, b in zip(zz[:-1], zz[1:], strict=True)]
    return out


def manifest_spray(meta: list[dict]) -> dict:
    return {
        "vehicles": meta,
        "phase": f"Henyey-Greenstein g = {SPRAY_G} (water drops >> wavelength; Hansen & Travis 1974)",
        "reference": "Otxoterena Drake et al. 2021, J. Wind Eng. Ind. Aerodyn. 217, 104734: extinction <= 0.2 m^-1",
        "scaling": f"linear in speed above {SPRAY_V_ONSET_KMH:g} km/h, {SPRAY_EXT_REF} m^-1 for a heavy vehicle at "
        f"{SPRAY_V_REF_KMH:g} km/h on a wet road (assumed), cars x {SPRAY_CAR_FACTOR}, flooded x 2",
    }
