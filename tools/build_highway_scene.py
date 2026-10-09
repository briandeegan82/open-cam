#!/usr/bin/env python3
"""Build a spectral pbrt-v4 highway scene (road, markings, cars, barriers, signs, vegetation, sun + sky).

A three-lane-per-direction divided highway (US-style markings: yellow left edge, dashed
white lane lines, solid white right edge) with a concrete median barrier, W-beam guard rails,
guide/speed signs, rolling grass terrain with trees and shrubs, and traffic. The camera
sits at windscreen/ADAS height looking down the road. World units are metres, y up, the
road runs along +z and +x is image-right.

Lighting is absolute: a spectral ``distant`` sun (air-mass dependent solar SPD) plus an
equal-area sky (Poly Haven HDRI with the sun removed, or an analytic CIE clear sky), both
set with pbrt's photometric ``illuminance``. The manifest records the resulting horizontal
illuminance at the road (``lighting.reference_illuminance_lux``) and its EXR-unit value, so
tools/pbrt_spectral_exr_to_electrons.py converts to electrons in absolute units.

Third-party assets come from tools/fetch_highway_assets.py. ``--allow-missing-assets``
falls back to proxy cars, analytic sky and untextured surfaces (used by the tests).

    venv/bin/python tools/fetch_highway_assets.py
    venv/bin/python tools/build_highway_scene.py --camera realistic
    third_party/pbrt-v4/build/pbrt scenes/generated/highway/highway.pbrt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import highway_atmosphere as atmosphere  # noqa: E402
import highway_backdrop as backdrop  # noqa: E402
import highway_night as night  # noqa: E402
from fetch_highway_assets import cache_root, load_asset_manifest  # noqa: E402
from highway_sky import (  # noqa: E402
    analytic_clear_sky,
    clear_sky_illuminance_lux,
    read_rgb_exr,
    solar_direct_spectrum,
    write_rgb_exr,
)
from highway_spectra import CAR_PAINTS, SURFACES, reflectance  # noqa: E402
from pbrt_spectral_exr_to_electrons import PBRT_CIE_Y_INTEGRAL  # noqa: E402

DEFAULT_REALISTIC_LENSFILE = "config/lenses/wide_22mm.dat"
SKIES = {
    "kloofendal_43d_clear": "sky_kloofendal_43d_clear",
    "kloofendal_48d_partly_cloudy": "sky_kloofendal_48d_partly_cloudy",
    "syferfontein_6d_clear": "sky_syferfontein_6d_clear",
    "analytic": None,
}
CAR_MODELS = ("car_bmw_m6", "car_pontiac_gto", "car_vintage")
LANE_W = 3.66  # 12 ft
N_LANES = 3
LEFT_SHOULDER, RIGHT_SHOULDER = 1.2, 3.0
MEDIAN_W = 2.0
DASH, GAP = 3.05, 9.14  # MUTCD broken line: 10 ft dash, 30 ft gap
ROAD_Z0, ROAD_Z1 = -40.0, 1500.0

# (lane index 0 = leftmost/fast, distance ahead [m], model, paint, yaw jitter [deg]); oncoming: lane < 0.
DEFAULT_TRAFFIC = [
    (1, 16.0, "car_bmw_m6", "red", 0.0),
    (2, 27.0, "car_pontiac_gto", "silver", 0.4),
    (0, 41.0, "car_bmw_m6", "white", -0.3),
    (1, 64.0, "car_vintage", "blue", 0.0),
    (2, 90.0, "car_bmw_m6", "black", 0.2),
    (0, 125.0, "car_pontiac_gto", "darkgreen", 0.0),
    (-1, 58.0, "car_bmw_m6", "gray", 0.0),
    (-2, 110.0, "car_bmw_m6", "silver", 0.0),
]


def _rel(repo: Path, p: Path) -> str:
    try:
        return str(p.resolve().relative_to(repo))
    except ValueError:
        return str(p.resolve())


def _f(x: float) -> str:
    return f"{x:.6g}"


def _pts(a: np.ndarray) -> str:
    return " ".join(_f(v) for v in np.asarray(a, dtype=np.float64).ravel())


def mesh(p: np.ndarray, tri: np.ndarray, uv: np.ndarray | None = None, n: np.ndarray | None = None) -> list[str]:
    out = ['Shape "trianglemesh"', f'    "point3 P" [ {_pts(p)} ]', f'    "integer indices" [ {_pts(tri)} ]']
    if uv is not None:
        out.append(f'    "point2 uv" [ {_pts(uv)} ]')
    if n is not None:
        out.append(f'    "normal N" [ {_pts(n)} ]')
    return out


def quads_mesh(rects: list[tuple[float, float, float, float]], y: float) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal rectangles (x0, x1, z0, z1) at height y, facing up."""
    p, t = [], []
    for i, (x0, x1, z0, z1) in enumerate(rects):
        p += [(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)]
        b = 4 * i
        t += [(b, b + 2, b + 1), (b, b + 3, b + 2)]
    return np.array(p), np.array(t)


def extrude_profile(profile: list[tuple[float, float]], x0: float, z0: float, z1: float, flip: bool = False):
    """Extrude a (lateral offset, height) polyline along z at lateral position x0."""
    sgn = -1.0 if flip else 1.0
    p, t = [], []
    n = len(profile)
    for z in (z0, z1):
        for dx, h in profile:
            p.append((x0 + sgn * dx, h, z))
    for i in range(n - 1):
        a, b, c, d = i, i + 1, n + i, n + i + 1
        t += [(a, c, b), (b, c, d)]
    return np.array(p), np.array(t)


def box(cx: float, y0: float, cz: float, sx: float, sy: float, sz: float):
    x0, x1, y1, z0, z1 = cx - sx / 2, cx + sx / 2, y0 + sy, cz - sz / 2, cz + sz / 2
    p = np.array(
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    )
    t = np.array(
        [
            (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
            (3, 7, 6), (3, 6, 2), (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5),
        ]
    )  # fmt: skip
    return p, t


def write_spd(path: Path, wl: np.ndarray, val: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"{w:.1f} {v:.6g}" for w, v in zip(wl, val)) + "\n")


def luminance_texture(src: Path, dst: Path, size: int = 1024) -> None:
    """Linear luminance of an sRGB colour map, normalised to mean 1 (float EXR, single channel)."""
    if dst.is_file():
        return
    import OpenEXR
    from PIL import Image

    im = np.asarray(Image.open(src).convert("RGB").resize((size, size), Image.LANCZOS), dtype=np.float64) / 255.0
    lin = np.where(im <= 0.04045, im / 12.92, ((im + 0.055) / 1.055) ** 2.4)
    y = lin @ np.array([0.2126, 0.7152, 0.0722])
    y = y / max(float(y.mean()), 1e-9)
    dst.parent.mkdir(parents=True, exist_ok=True)
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    with OpenEXR.File(header, {"Y": np.ascontiguousarray(y, dtype=np.float32)}) as f:
        f.write(str(dst))


def sign_legend(path: Path, lines: list[str], size: tuple[int, int], border: int) -> None:
    """White legend + border mask (1 = white sheeting) for a guide sign."""
    from PIL import Image, ImageDraw, ImageFont

    w, h = size
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([border, border, w - border, h - border], radius=3 * border, outline=255, width=border)
    font_h = int(h * 0.8 / (len(lines) + 0.6))
    try:
        font = ImageFont.load_default(size=font_h)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    y = (h - font_h * len(lines) * 1.15) / 2
    for ln in lines:
        tw = d.textlength(ln, font=font)
        d.text(((w - tw) / 2, y), ln, fill=255, font=font)
        y += font_h * 1.15
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def rot_x(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class Layout:
    """Lateral positions (x) of the carriageways; our lanes span [0, 3*LANE_W]."""

    right_edge = N_LANES * LANE_W
    right_paved = right_edge + RIGHT_SHOULDER
    rail_right = right_paved + 0.6
    median_l = -LEFT_SHOULDER - MEDIAN_W
    opp_inner = median_l - LEFT_SHOULDER
    opp_edge = opp_inner - N_LANES * LANE_W
    opp_paved = opp_edge - RIGHT_SHOULDER
    rail_left = opp_paved - 0.6

    @staticmethod
    def lane_center(lane: int) -> float:
        if lane >= 0:
            return (lane + 0.5) * LANE_W
        return Layout.opp_inner - (-lane - 0.5) * LANE_W


def terrain_height(x: np.ndarray, z: np.ndarray, edge: float, side: float) -> np.ndarray:
    """Flat 6 m verge -> shallow cut slope -> rolling hills, 0 at the paved edge."""
    d = np.clip(side * (x - edge) - 6.0, 0.0, None)
    hills = 14.0 * (1.0 - np.exp(-d / 160.0)) * (0.65 + 0.35 * np.sin(z / 140.0 + 1.3 * side))
    return 0.06 * np.minimum(d, 12.0) + hills


def terrain_mesh(edge: float, side: float, far: float, tile: float, seed: int):
    xs = edge + side * np.concatenate([np.linspace(0, 46, 24), np.geomspace(50, far, 30)])
    zs = np.concatenate([np.linspace(ROAD_Z0 - 60, 400, 70), np.geomspace(410, 4000, 40)])
    X, Z = np.meshgrid(xs, zs)
    Y = terrain_height(X, Z, edge, side) - 0.02
    P = np.stack([X, Y, Z], -1).reshape(-1, 3)
    nx, nz = len(xs), len(zs)
    t = []
    for j in range(nz - 1):
        for i in range(nx - 1):
            a, b, c, d = j * nx + i, j * nx + i + 1, (j + 1) * nx + i, (j + 1) * nx + i + 1
            t += [(a, c, b), (b, c, d)] if side > 0 else [(a, b, c), (b, d, c)]
    uv = P[:, [0, 2]] / tile
    return P, np.array(t), uv


def car_include(info: dict, raw_dir: Path, out_dir: Path, inst: str, paint_spd: str, length_m: float) -> list[str]:
    """pbrt lines for one car instance (materials renamed per instance, roles overridden)."""
    roles = info.get("roles", {})
    role_of = {m: r for r, ms in roles.items() for m in ms}
    lines = []
    for name, body in info["materials"].items():
        role = role_of.get(name)
        mn = f"{inst}:{name}"
        if role == "paint":
            lines += [
                f'MakeNamedMaterial "{mn}"',
                '    "string type" "coateddiffuse"',
                f'    "spectrum reflectance" "{paint_spd}"',
                '    "float roughness" [0.008]',
                '    "float thickness" [0.04]',
                '    "bool remaproughness" false',
            ]
        elif role == "glass":
            lines += [f'MakeNamedMaterial "{mn}"', '    "string type" "dielectric"', '    "float eta" [1.52]']
        elif role == "tire":
            lines += [
                f'MakeNamedMaterial "{mn}"',
                '    "string type" "coateddiffuse"',
                '    "spectrum reflectance" "spd/rubber.spd"',
                '    "float roughness" [0.45]',
            ]
        else:
            body_lines = body.splitlines()
            for i, ln in enumerate(body_lines):
                if '"string materials"' in ln:
                    for other in info["materials"]:
                        ln = ln.replace(f'"{other}"', f'"{inst}:{other}"')
                    body_lines[i] = ln
            lines += [f'MakeNamedMaterial "{mn}"'] + ["    " + ln for ln in body_lines]
    lo, hi = np.array(info["bbox_min"]), np.array(info["bbox_max"])
    s = length_m / float(hi[0] - lo[0])
    cx, cz = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[2] + hi[2])
    lines += [f"Scale {_f(s)} {_f(s)} {_f(s)}", f"Translate {_f(-cx)} {_f(-lo[1])} {_f(-cz)}"]
    for sh in info["shapes"]:
        ply = os.path.relpath(raw_dir / sh["ply"], out_dir)
        lines += [f'NamedMaterial "{inst}:{sh["material"]}"', f'Shape "plymesh" "string filename" "{ply}"']
    return lines


def proxy_car(paint_spd: str, inst: str) -> list[str]:
    """Box-and-cylinder stand-in (front towards +z, ground at y=0) when car assets are absent."""
    out = [
        f'MakeNamedMaterial "{inst}:paint" "string type" "coateddiffuse" "spectrum reflectance" "{paint_spd}"'
        ' "float roughness" [0.01]',
        f'MakeNamedMaterial "{inst}:glass" "string type" "dielectric" "float eta" [1.52]',
        f'MakeNamedMaterial "{inst}:tire" "string type" "diffuse" "spectrum reflectance" "spd/rubber.spd"',
        f'NamedMaterial "{inst}:paint"',
        *mesh(*box(0, 0.3, 0, 1.8, 0.75, 4.6)),
        f'NamedMaterial "{inst}:glass"',
        *mesh(*box(0, 1.05, -0.2, 1.6, 0.45, 2.4)),
        f'NamedMaterial "{inst}:tire"',
    ]
    for wx in (-0.78, 0.78):
        for wz in (-1.45, 1.45):
            out += [
                "AttributeBegin",
                f"Translate {wx - 0.11} 0.32 {wz}",
                "Rotate 90 0 1 0",
                'Shape "cylinder" "float radius" [0.32] "float zmin" [0] "float zmax" [0.22]',
                "AttributeEnd",
            ]
    return out


def vegetation_object(name: str, info: dict, raw_dir: Path, out_dir: Path) -> list[str]:
    out = [f'ObjectBegin "{name}"']
    for k, m in enumerate(info["materials"]):
        mn = f"{name}:{k}"
        tex = m.get("base_color_texture")
        leafy = any(w in m["name"].lower() for w in ("leaf", "leaves", "shrub", "grass"))
        if tex and leafy:
            lum = out_dir / "textures" / f"{name}_{k}_lum.exr"
            luminance_texture(raw_dir / tex, lum, size=512)
            out += [
                f'Texture "{mn}:lum" "float" "imagemap" "string filename" "{os.path.relpath(lum, out_dir)}"',
                f'Texture "{mn}:r" "spectrum" "scale" "spectrum tex" "spd/leaf_reflectance.spd"'
                f' "texture scale" "{mn}:lum"',
                f'Texture "{mn}:t" "spectrum" "scale" "spectrum tex" "spd/leaf_transmittance.spd"'
                f' "texture scale" "{mn}:lum"',
                f'MakeNamedMaterial "{mn}" "string type" "diffusetransmission"'
                f' "texture reflectance" "{mn}:r" "texture transmittance" "{mn}:t"',
            ]
        elif tex:
            rel = os.path.relpath(raw_dir / tex, out_dir)
            out += [
                f'Texture "{mn}:c" "spectrum" "imagemap" "string filename" "{rel}" "string encoding" "sRGB"',
                f'MakeNamedMaterial "{mn}" "string type" "diffuse" "texture reflectance" "{mn}:c"',
            ]
        else:
            out.append(f'MakeNamedMaterial "{mn}" "string type" "diffuse" "spectrum reflectance" "spd/bark.spd"')
    for sh in info["shapes"]:
        out += [f'NamedMaterial "{name}:{sh["material"]}"', f'Shape "plymesh" "string filename" "{sh["ply_rel"]}"']
    out.append("ObjectEnd")
    return out


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=Path, default=root)
    ap.add_argument("--out-dir", type=Path, default=None, help="default: scenes/generated/highway")
    ap.add_argument("--asset-manifest", type=Path, default=None, help="default: config/highway_assets.yaml")
    ap.add_argument("--allow-missing-assets", action="store_true", help="Fall back to proxies when not fetched.")
    ap.add_argument("--sky", choices=sorted(SKIES), default="kloofendal_43d_clear")
    ap.add_argument("--sun-elevation", type=float, default=45.0, help="Sun elevation [deg] for --sky analytic.")
    ap.add_argument(
        "--sun-azimuth", type=float, default=140.0, help="Sun azimuth [deg] from the road direction, + = right."
    )
    ap.add_argument(
        "--global-illuminance-lux",
        type=float,
        default=None,
        help="Override the horizontal (sun + sky) illuminance at the road; default: clear-sky model.",
    )
    ap.add_argument("--road", choices=("asphalt031", "asphalt026c", "plain"), default="asphalt031")
    ap.add_argument("--traffic", choices=("default", "none"), default="default")
    ap.add_argument("--proxy-cars", action="store_true", help="Use box proxies instead of the CC0 car models.")
    ap.add_argument("--vegetation", choices=("full", "shrubs", "none"), default="full")
    ap.add_argument("--seed", type=int, default=7, help="Vegetation placement seed.")
    ap.add_argument("--camera", choices=("perspective", "pinhole", "thinlens", "realistic"), default="pinhole")
    ap.add_argument("--lensfile", default=DEFAULT_REALISTIC_LENSFILE)
    ap.add_argument("--aperture-diameter-mm", type=float, default=4.0)
    ap.add_argument("--focus-distance", type=float, default=25.0, help="Focus distance [m] (realistic/thinlens).")
    ap.add_argument("--film-diagonal-mm", type=float, default=25.0, help="Realistic camera film diagonal.")
    ap.add_argument("--fov", type=float, default=31.0, help="Pinhole/thinlens fov [deg] (shorter image axis).")
    ap.add_argument("--thinlens-lens-radius", type=float, default=0.0)
    ap.add_argument("--cam-height", type=float, default=1.35, help="Camera height above the road [m].")
    ap.add_argument("--cam-lane", type=int, default=1, help="Ego lane (0 = leftmost).")
    ap.add_argument("--cam-pitch", type=float, default=-1.5, help="Camera pitch [deg], negative = down.")
    ap.add_argument("--xres", type=int, default=1280)
    ap.add_argument("--yres", type=int, default=720)
    ap.add_argument("--pixelsamples", type=int, default=256)
    ap.add_argument("--maxdepth", type=int, default=8)
    ap.add_argument("--film", choices=("spectral", "rgb"), default="spectral")
    ap.add_argument("--film-output", default=None, help="default: out/highway_spectral.exr (or out/highway.exr)")
    ap.add_argument("--spectral-nbuckets", type=int, default=16)
    ap.add_argument("--spectral-lambda-min", type=float, default=360.0)
    ap.add_argument("--spectral-lambda-max", type=float, default=830.0)
    ap.add_argument("--step-nm", type=float, default=5.0)
    night.add_night_args(ap)
    atmosphere.add_cli_args(ap)
    backdrop.add_cli_args(ap)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:  # noqa: C901 (linear scene assembly)
    args = parse_args(argv)
    repo = args.repo_root.resolve()
    out_dir = (args.out_dir or repo / "scenes" / "generated" / "highway").resolve()
    spd = out_dir / "spd"
    out_dir.mkdir(parents=True, exist_ok=True)
    amanifest = load_asset_manifest(args.asset_manifest or repo / "config" / "highway_assets.yaml")
    aroot = cache_root(repo, amanifest)
    used_assets: dict[str, dict] = {}
    fallbacks: list[str] = []

    def asset(aid: str | None) -> tuple[dict, Path] | None:
        if aid is None:
            return None
        p = aroot / "prepared" / aid / "prepared.json"
        if not p.is_file():
            if not args.allow_missing_assets:
                sys.exit(
                    f"error: asset {aid} not prepared ({p}); run tools/fetch_highway_assets.py {aid} "
                    "or pass --allow-missing-assets"
                )
            fallbacks.append(aid)
            return None
        info = json.loads(p.read_text())
        a = amanifest["assets"][aid]
        used_assets[aid] = {k: a.get(k) for k in ("title", "source", "author", "license", "license_note") if a.get(k)}
        return info, aroot / "raw" / aid

    # ---- spectra
    wl = np.arange(args.spectral_lambda_min, args.spectral_lambda_max + 1e-9, args.step_nm)
    for name in SURFACES:
        write_spd(spd / f"{name}.spd", wl, reflectance(name, wl))

    # ---- sun & sky
    sky_aid = SKIES[args.sky]
    sky_asset = asset(sky_aid)
    if sky_asset is not None:
        info, _ = sky_asset
        sun_l = np.array(info["sun_dir_light"])
        elev = float(info["sun_elevation_deg"])
        sky_ratio = float(info["sky_to_sun_horizontal_ratio"])
        sky_file = aroot / "prepared" / sky_aid / info["sky_equiarea"]
        sky_desc = f"Poly Haven HDRI {args.sky} (sun removed; RGB sky upsampled by pbrt)"
    else:
        if sky_aid is not None:
            print(f"warning: {sky_aid} not fetched, using analytic sky", file=sys.stderr)
        elev = float(args.sun_elevation)
        zs = math.radians(90.0 - elev)
        sun_l = np.array([math.sin(zs), 0.0, math.cos(zs)])
        sky_file = out_dir / "textures" / f"sky_cie12_{elev:.1f}.exr"
        if not sky_file.is_file():
            write_rgb_exr(sky_file, analytic_clear_sky(512, elev))
        sky_ratio = None
        sky_desc = "analytic CIE standard clear sky (type 12)"
    to_world = rot_x(-90.0)
    s0 = to_world @ sun_l
    az0 = math.degrees(math.atan2(s0[0], s0[2]))
    sky_rot = args.sun_azimuth - az0
    sun_w = rot_y(sky_rot) @ s0
    e_dn, e_diffuse = clear_sky_illuminance_lux(elev)
    sin_el = max(math.sin(math.radians(elev)), 1e-3)
    e_sky_h = sky_ratio * e_dn * sin_el if sky_ratio is not None else e_diffuse
    e_ref = e_dn * sin_el + e_sky_h
    if args.global_illuminance_lux is not None:
        k = float(args.global_illuminance_lux) / e_ref
        e_dn, e_sky_h, e_ref = e_dn * k, e_sky_h * k, float(args.global_illuminance_lux)
    write_spd(spd / "sun.spd", wl, solar_direct_spectrum(wl, elev))
    haze = atmosphere.params_from_args(args)
    if haze is not None and args.time_of_day != "day":
        raise SystemExit(
            "--haze is not supported with --time-of-day dusk/night yet (night reference illuminance has no medium model)"
        )
    atm = None
    if haze is not None:
        atm = atmosphere.setup_scene(
            haze,
            args,
            wl=wl,
            spd_dir=spd,
            sky_rgb_equal_area=read_rgb_exr(sky_file),
            sun_spd=solar_direct_spectrum(wl, elev),
            elev=elev,
            e_dn=e_dn,
            e_sky_h=e_sky_h,
            e_ref=e_ref,
        )
        e_dn, e_sky_h, e_ref = atm["e_dn"], atm["e_sky_h"], atm["e_ref"]

    # ---- film & camera
    film_out = args.film_output or ("out/highway_spectral.exr" if args.film == "spectral" else "out/highway.exr")
    film = [
        f'Film "{args.film}"',
        f'    "string filename" ["{film_out}"]',
        f'    "integer xresolution" [{args.xres}]',
        f'    "integer yresolution" [{args.yres}]',
        '    "bool savefp16" false',
    ]
    if args.film == "spectral":
        film += [
            f'    "integer nbuckets" [{args.spectral_nbuckets}]',
            f'    "float lambdamin" [{_f(args.spectral_lambda_min)}]',
            f'    "float lambdamax" [{_f(args.spectral_lambda_max)}]',
        ]
    camera_kind = "pinhole" if args.camera == "perspective" else args.camera
    if camera_kind == "realistic":
        film.append(f'    "float diagonal" [{_f(args.film_diagonal_mm)}]')
    cam_x = Layout.lane_center(args.cam_lane)
    eye = np.array([cam_x, args.cam_height, 0.0])
    target = eye + 100.0 * np.array([0.0, math.tan(math.radians(args.cam_pitch)), 1.0])
    integrator, maxdepth = ("volpath", atm["maxdepth"]) if atm else ("path", args.maxdepth)
    L = [
        "# Generated by tools/build_highway_scene.py — do not hand-edit.",
        'Option "seed" 0',
        'ColorSpace "srgb"',
        f'Sampler "zsobol" "integer pixelsamples" [{args.pixelsamples}]',
        f'Integrator "{integrator}" "integer maxdepth" [{maxdepth}] "bool regularize" true',
        'PixelFilter "gaussian"',
        *film,
        f"LookAt {_pts(eye)}  {_pts(target)}  0 1 0",
        *(atm["camera_lines"] if atm else []),
    ]
    if camera_kind == "pinhole":
        L.append(f'Camera "perspective" "float fov" [{_f(args.fov)}]')
    elif camera_kind == "thinlens":
        L.append(
            f'Camera "perspective" "float fov" [{_f(args.fov)}] "float lensradius" [{_f(args.thinlens_lens_radius)}]'
            f' "float focaldistance" [{_f(args.focus_distance)}]'
        )
    else:
        lens = (repo / args.lensfile).resolve()
        if not lens.is_file():
            raise FileNotFoundError(f"realistic camera: lens file not found: {lens}")
        L.append(
            f'Camera "realistic" "string lensfile" ["{os.path.relpath(lens, out_dir)}"]'
            f' "float aperturediameter" [{_f(args.aperture_diameter_mm)}]'
            f' "float focusdistance" [{_f(args.focus_distance)}]'
        )
    if atm:
        L += atm["medium_lines"]

    L += ["", "WorldBegin", ""]
    day_lights = [
        f"# Sun: elevation {elev:.2f} deg, azimuth {args.sun_azimuth:.1f} deg, {e_dn:.0f} lux direct-normal",
        'LightSource "distant" "spectrum L" "spd/sun.spd"',
        f'    "float illuminance" [{_f(e_dn)}]',
        f'    "point3 from" [{_pts(sun_w)}] "point3 to" [0 0 0]',
        f"# Sky: {sky_desc}, {e_sky_h:.0f} lux on a horizontal plane",
        "AttributeBegin",
        f"    Rotate {_f(sky_rot)} 0 1 0",
        "    Rotate -90 1 0 0",
        f'    LightSource "infinite" "string filename" "{os.path.relpath(sky_file, out_dir)}"',
        f'        "float illuminance" [{_f(e_sky_h)}]',
        "AttributeEnd",
        "",
    ]
    nopts = night.resolve(args)
    night_lighting = None
    if args.time_of_day == "day":
        L += day_lights
    else:
        natural, night_lighting = night.natural_light(args, out_dir, spd)
        L += natural

    # ---- road surface
    road_tex = None if args.road == "plain" else asset(f"tex_{args.road}")
    road_spd = "spd/asphalt_aged.spd"
    tile = 1.0
    if road_tex is not None:
        info, raw = road_tex
        tile = float(info["tile_size_m"])
        lum = out_dir / "textures" / f"{args.road}_lum.exr"
        luminance_texture(raw / info["maps"]["color"], lum)
        L += [
            f'Texture "asphalt:lum" "float" "imagemap" "string filename" "{os.path.relpath(lum, out_dir)}"',
            f'Texture "asphalt" "spectrum" "scale" "spectrum tex" "{road_spd}" "texture scale" "asphalt:lum"',
            'MakeNamedMaterial "asphalt" "string type" "coateddiffuse" "texture reflectance" "asphalt"',
            '    "float roughness" [0.35] "float thickness" [0.001]',
            f'    "string normalmap" "{os.path.relpath(raw / info["maps"]["normal"], out_dir)}"',
        ]
    else:
        L.append(
            f'MakeNamedMaterial "asphalt" "string type" "coateddiffuse" "spectrum reflectance" "{road_spd}"'
            ' "float roughness" [0.35] "float thickness" [0.001]'
        )
    grass_tex = asset("tex_grass004")
    gtile = 1.5
    if grass_tex is not None:
        info, raw = grass_tex
        gtile = float(info["tile_size_m"])
        lum = out_dir / "textures" / "grass004_lum.exr"
        luminance_texture(raw / info["maps"]["color"], lum)
        L += [
            f'Texture "grass:lum" "float" "imagemap" "string filename" "{os.path.relpath(lum, out_dir)}"',
            'Texture "grass" "spectrum" "scale" "spectrum tex" "spd/grass.spd" "texture scale" "grass:lum"',
            'MakeNamedMaterial "grass" "string type" "diffuse" "texture reflectance" "grass"',
            f'    "string normalmap" "{os.path.relpath(raw / info["maps"]["normal"], out_dir)}"',
        ]
    else:
        L.append('MakeNamedMaterial "grass" "string type" "diffuse" "spectrum reflectance" "spd/grass.spd"')
    L += [
        'MakeNamedMaterial "grass_far" "string type" "diffuse" "spectrum reflectance" "spd/grass.spd"',
        'MakeNamedMaterial "concrete" "string type" "diffuse" "spectrum reflectance" "spd/concrete.spd"',
        # Weathered hot-dip galvanised steel: zinc-oxide patina, i.e. light grey with a rough sheen.
        'MakeNamedMaterial "galvanized" "string type" "coateddiffuse" "spectrum reflectance" "spd/galvanized.spd"'
        ' "float roughness" [0.3]',
        # Glass-bead road paint: diffuse binder under a rough clear interface (daytime look).
        'MakeNamedMaterial "paint_white" "string type" "coateddiffuse"'
        ' "spectrum reflectance" "spd/paint_road_white.spd" "float roughness" [0.3]',
        'MakeNamedMaterial "paint_yellow" "string type" "coateddiffuse"'
        ' "spectrum reflectance" "spd/paint_road_yellow.spd" "float roughness" [0.3]',
        'MakeNamedMaterial "sheet_white" "string type" "diffuse" "spectrum reflectance" "spd/sheeting_white.spd"',
        'MakeNamedMaterial "sheet_green" "string type" "diffuse" "spectrum reflectance" "spd/sheeting_green.spd"',
        'MakeNamedMaterial "sheet_black" "string type" "diffuse" "spectrum reflectance" "spd/sheeting_black.spd"',
        "",
    ]
    if nopts["retroreflective"]:
        L = night.replace_named_materials(L, night.retro_material_lines(spd, out_dir, wl))
    if nopts["dark"]:
        L = night.strip_normal_map(L)
    paved = [
        (Layout.median_l, Layout.right_paved, ROAD_Z0, ROAD_Z1),
        (Layout.opp_paved, Layout.median_l, ROAD_Z0, ROAD_Z1),
    ]
    p, t = quads_mesh(paved, 0.0)
    L += ['NamedMaterial "asphalt"', *mesh(p, t, p[:, [0, 2]] / tile), ""]

    # ---- markings (US: yellow left edge, broken white lane lines, solid white right edge)
    def dashes(xc: float, w: float) -> list[tuple[float, float, float, float]]:
        z, out = ROAD_Z0 + 2.0, []
        while z < ROAD_Z1:
            out.append((xc - w / 2, xc + w / 2, z, min(z + DASH, ROAD_Z1)))
            z += DASH + GAP
        return out

    white, yellow = [], []
    for side, inner, edge in ((1, 0.0, Layout.right_edge), (-1, Layout.opp_inner, Layout.opp_edge)):
        yellow.append((min(inner, inner + side * 0.15), max(inner, inner + side * 0.15), ROAD_Z0, ROAD_Z1))
        white.append((min(edge, edge - side * 0.2), max(edge, edge - side * 0.2), ROAD_Z0, ROAD_Z1))
        for k in (1, 2):
            white += dashes(inner + side * k * LANE_W, 0.15)
    for mat, rects in (("paint_white", white), ("paint_yellow", yellow)):
        p, t = quads_mesh(rects, 0.004)
        L += [f'NamedMaterial "{mat}"', *mesh(p, t), ""]

    # ---- concrete median barrier (single-slope ~ Jersey profile, 0.81 m tall)
    jersey = [(-0.30, 0.0), (-0.28, 0.08), (-0.18, 0.33), (-0.08, 0.81), (0.08, 0.81), (0.18, 0.33), (0.28, 0.08)]
    jersey.append((0.30, 0.0))
    p, t = extrude_profile(jersey, Layout.median_l + MEDIAN_W / 2 + LEFT_SHOULDER / 2 - 0.6, ROAD_Z0, ROAD_Z1)
    L += ['NamedMaterial "concrete"', *mesh(p, t), ""]

    # ---- W-beam guard rails with posts every 1.905 m
    wbeam = [(0.0, 0.43), (0.05, 0.47), (0.08, 0.53), (0.05, 0.58), (0.08, 0.64), (0.05, 0.70), (0.0, 0.74)]
    L.append('NamedMaterial "galvanized"')
    for x0, flip in ((Layout.rail_right, True), (Layout.rail_left, False)):
        p, t = extrude_profile(wbeam, x0, ROAD_Z0, ROAD_Z1, flip=flip)
        L += mesh(p, t)
        posts_p, posts_t = [], []
        for k, z in enumerate(np.arange(ROAD_Z0, 400.0, 1.905)):  # posts behind the beam
            bp, bt = box(x0 + (0.09 if flip else -0.09), -0.1, z, 0.15, 0.85, 0.1)
            posts_p.append(bp)
            posts_t.append(bt + 8 * k)
        L += mesh(np.concatenate(posts_p), np.concatenate(posts_t))
    L.append("")

    # ---- signs
    legend = out_dir / "textures" / "guide_sign.png"
    sign_legend(legend, ["Galway  12", "Athlone  48", "Dublin  135"], (1000, 520), 14)
    speed = out_dir / "textures" / "speed_sign.png"
    sign_legend(speed, ["SPEED", "LIMIT", "65"], (360, 480), 0)
    gx, gz = Layout.rail_right + 3.2, 140.0
    gw, gh, gy = 5.0, 2.6, 2.4
    L += [
        f'Texture "guide_legend" "float" "imagemap" "string filename" "{os.path.relpath(legend, out_dir)}"',
        'MakeNamedMaterial "guide_sign" "string type" "mix" "string materials" ["sheet_green" "sheet_white"]',
        '    "texture amount" "guide_legend"',
        f'Texture "speed_legend" "float" "imagemap" "string filename" "{os.path.relpath(speed, out_dir)}"',
        'MakeNamedMaterial "speed_sign" "string type" "mix" "string materials" ["sheet_white" "sheet_black"]',
        '    "texture amount" "speed_legend"',
    ]
    for mat, cx, cz, w, h, y0 in (
        ("guide_sign", gx, gz, gw, gh, gy),
        ("speed_sign", Layout.rail_right + 1.6, 60.0, 0.9, 1.2, 1.6),
    ):
        p = np.array([(cx - w / 2, y0, cz), (cx + w / 2, y0, cz), (cx + w / 2, y0 + h, cz), (cx - w / 2, y0 + h, cz)])
        uv = np.array([(0, 0), (1, 0), (1, 1), (0, 1)])
        L += [f'NamedMaterial "{mat}"', *mesh(p, np.array([(0, 2, 1), (0, 3, 2)]), uv)]
        L += ['NamedMaterial "galvanized"']
        for px in (cx - w * 0.3, cx + w * 0.3) if w > 2 else (cx,):
            L += mesh(*box(px, -0.1, cz + 0.08, 0.12, y0 + h * 0.9, 0.12))
    L.append("")

    # ---- verges + terrain (finely tessellated from the paved edge; no long thin triangles)
    L.append('NamedMaterial "grass"')
    for edge, side in ((Layout.right_paved, 1.0), (Layout.opp_paved, -1.0)):
        p, t, uv = terrain_mesh(edge, side, 3000.0, gtile, args.seed)
        L += mesh(p, t, uv)
    L += [
        'NamedMaterial "grass_far"',
        *mesh(*quads_mesh([(-8000, 8000, ROAD_Z1, 12000)], -0.15)),
        "",
    ]
    if backdrop.enabled(args):
        write_spd(spd / "forest_canopy.spd", wl, backdrop.forest_canopy_reflectance(wl))
        L += [*backdrop.pbrt_lines(mesh, cam_x, 0.0, args.seed, "spd/forest_canopy.spd", "spd/grass.spd"), ""]

    # ---- vegetation (instanced)
    rng = np.random.default_rng(args.seed)
    veg_meta = []
    for vid, kind in (("veg_island_tree_02", "tree"), ("veg_shrub_02", "shrub")):
        if args.vegetation == "none" or (args.vegetation == "shrubs" and kind == "tree"):
            continue
        va = asset(vid)
        if va is None:
            continue
        info, raw = va
        prep = aroot / "prepared" / vid
        for sh in info["shapes"]:
            sh["ply_rel"] = os.path.relpath(prep / sh["ply"], out_dir)
        L += vegetation_object(vid, info, raw, out_dir)
        n = 0
        for side, edge in ((1.0, Layout.right_paved), (-1.0, Layout.opp_paved)):
            z = 8.0 + rng.uniform(0, 10)
            while z < 900.0:
                if kind == "tree":
                    d = rng.uniform(10.0, 36.0) + (z / 900.0) * 30.0
                    s = rng.uniform(2.6, 3.8)
                    z += rng.uniform(9.0, 26.0)
                else:
                    d = rng.uniform(6.5, 12.0)
                    s = rng.uniform(0.8, 1.4)
                    z += rng.uniform(6.0, 22.0)
                x = edge + side * d
                y = float(terrain_height(np.array(x), np.array(z), edge, side)) - 0.15
                L += [
                    "AttributeBegin",
                    f"Translate {_f(x)} {_f(y)} {_f(z)}",
                    f"Rotate {_f(rng.uniform(0, 360))} 0 1 0",
                    f"Scale {_f(s)} {_f(s)} {_f(s)}",
                    f'ObjectInstance "{vid}"',
                    "AttributeEnd",
                ]
                n += 1
        veg_meta.append({"asset": vid, "instances": n})
    L.append("")

    # ---- traffic
    cars_meta = []
    if args.traffic != "none":
        car_dir = out_dir / "cars"
        car_dir.mkdir(exist_ok=True)
        for k, (lane, z, model, paint, yaw) in enumerate(DEFAULT_TRAFFIC):
            if paint not in CAR_PAINTS:
                raise ValueError(paint)
            inst = f"car{k:02d}"
            paint_spd = f"spd/carpaint_{paint}.spd"
            ca = None if args.proxy_cars else asset(model)
            heading = 0.0 if lane >= 0 else 180.0
            if ca is not None:
                info, raw = ca
                body = car_include(info, raw, out_dir, inst, paint_spd, amanifest["assets"][model]["length_m"])
                yaw_model = 90.0 + heading  # model front is -x
            else:
                body = proxy_car(paint_spd, inst)
                yaw_model = heading
            inc = car_dir / f"{inst}.pbrt"
            inc.write_text("\n".join(body) + "\n")
            x = Layout.lane_center(lane)
            L += [
                "AttributeBegin",
                f"Translate {_f(x)} 0 {_f(z)}",
                f"Rotate {_f(yaw_model + yaw)} 0 1 0",
                f'Include "{os.path.relpath(inc, out_dir)}"',
                "AttributeEnd",
            ]
            cars_meta.append(
                {
                    "id": inst,
                    "model": model if ca else "proxy",
                    "paint": paint,
                    "lane": lane,
                    "distance_m": z,
                    **night.car_geometry(
                        info if ca else None, amanifest["assets"][model]["length_m"], x, heading + yaw
                    ),
                }
            )

    night_lines, night_meta = night.scene_lights(
        args,
        nopts,
        out_dir,
        spd,
        cars_meta,
        eye,
        Layout.median_l + MEDIAN_W / 2 + LEFT_SHOULDER / 2 - 0.6,
        (0.0, Layout.right_edge),
        (ROAD_Z0 + 8.0, 900.0),
    )
    L += night_lines

    scene_path = out_dir / "highway.pbrt"
    scene_path.write_text("\n".join(L) + "\n")

    manifest = {
        "scene": _rel(repo, scene_path),
        "generator": "tools/build_highway_scene.py",
        "units": "metres; y up; road along +z; +x image-right",
        "film": {
            "type": args.film,
            "filename": film_out,
            "xresolution": args.xres,
            "yresolution": args.yres,
            **(
                {
                    "nbuckets": args.spectral_nbuckets,
                    "lambdamin": args.spectral_lambda_min,
                    "lambdamax": args.spectral_lambda_max,
                }
                if args.film == "spectral"
                else {}
            ),
        },
        "camera": {
            "type": camera_kind,
            "height_m": args.cam_height,
            "lane": args.cam_lane,
            "pitch_deg": args.cam_pitch,
            "lookat": {"eye": eye.tolist(), "target": target.tolist(), "up": [0.0, 1.0, 0.0]},
            "fov_deg": None if camera_kind == "realistic" else float(args.fov),
            **(
                {
                    "lensfile": args.lensfile,
                    "aperture_diameter_mm": args.aperture_diameter_mm,
                    "film_diagonal_mm": args.film_diagonal_mm,
                    "focus_distance": args.focus_distance,
                }
                if camera_kind == "realistic"
                else {}
            ),
            **(
                {"lens_radius": args.thinlens_lens_radius, "focal_distance": args.focus_distance}
                if camera_kind == "thinlens"
                else {}
            ),
        },
        "lighting": {
            "sun": {
                "elevation_deg": elev,
                "azimuth_deg": args.sun_azimuth,
                "direction_world": sun_w.tolist(),
                "illuminance_normal_lux": e_dn,
                "illuminance_horizontal_lux": e_dn * sin_el,
                "spectrum": "spd/sun.spd",
            },
            "sky": {
                "source": args.sky,
                "description": sky_desc,
                "illuminance_horizontal_lux": e_sky_h,
                "rotate_y_deg": sky_rot,
            },
            "reference_illuminance_lux": e_ref,
            "reference_illuminance_exr_lux": 683.0 * PBRT_CIE_Y_INTEGRAL * e_ref,
            "reference": "horizontal illuminance (sun + sky) at road level, unoccluded",
            "model": "clear-sky direct-normal 128 klux*exp(-0.21 m_KY); sky from HDRI ratio or IESNA fit",
        },
        "road": {
            "lanes_per_direction": N_LANES,
            "lane_width_m": LANE_W,
            "markings": "MUTCD: yellow left edge, broken white 3.05/9.14 m, solid white right edge",
            "surface": args.road,
            "length_m": ROAD_Z1 - ROAD_Z0,
        },
        "cars": cars_meta,
        "vegetation": veg_meta,
        "atmosphere": atm["manifest"] if atm else None,
        "distant_terrain": "hills (tools/highway_backdrop.py)" if backdrop.enabled(args) else None,
        "assets": used_assets,
        "missing_assets_fallback": fallbacks,
        "approximations": [
            "Road-marking glass-bead retroreflection is not modelled (pbrt has no retroreflective BSDF); "
            "markings and sign sheeting use their daytime diffuse/sheen appearance.",
            "Sky radiance is RGB (HDRI or analytic) upsampled by pbrt; the sun is spectral.",
            "Asphalt/grass colour maps modulate the analytic spectral reflectance by luminance only.",
            "No atmospheric scattering medium (aerial perspective comes only from the sky map)."
            if atm is None
            else "Atmosphere: horizontally uniform grid medium, single HG phase (Rayleigh folded into g); "
            "lights are at the top of the layer, road illuminance from a plane-parallel MC model.",
        ],
    }
    night.update_manifest(manifest, nopts, night_lighting, night_meta)
    (out_dir / "highway_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"wrote {_rel(repo, scene_path)} (+ highway_manifest.json); horizontal illuminance "
        f"{manifest['lighting']['reference_illuminance_lux']:.4g} lux"
    )
    if fallbacks:
        print(f"warning: missing assets replaced by fallbacks: {fallbacks}", file=sys.stderr)


if __name__ == "__main__":
    main()
