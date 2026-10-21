#!/usr/bin/env python3
"""Underground parking garage seen by a surround-view fisheye camera.

    venv/bin/python tools/build_garage_scene.py --out-dir scenes/generated/garage --view front
    pbrt scenes/generated/garage/garage.pbrt

A concrete deck with a central aisle, perpendicular bays both sides, pillars and walls, lit only by
ceiling luminaires (`--lighting fluorescent|led|mixed`, `--lights-off-fraction` for dead fittings)
and green exit signs. The ego car stands in the aisle; `--view front|rear|left|right` picks the
surround-view camera (bumper or mirror mount). The default camera is the realistic
`fisheye.10mm.dat` lens (155.9 deg) on a 4:3 film sized so the image circle is cropped top and
bottom, like a production surround-view camera. The manifest records every bay's corners and
occupancy (parking-slot ground truth) and the pillar positions.
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

import highway_night as night  # noqa: E402
from build_highway_scene import (  # noqa: E402
    _f,
    _pts,
    _rel,
    add_camera_args,
    add_film_args,
    box,
    camera_lines,
    car_include,
    film_lines,
    mesh,
    proxy_car,
    quads_mesh,
    write_spd,
)
from fetch_highway_assets import cache_root, load_asset_manifest  # noqa: E402
from highway_incar import MainsRipple, write_emitter  # noqa: E402
from highway_spectra import SURFACES, reflectance  # noqa: E402

AISLE_W = 6.5
BAY_W, BAY_D = 2.5, 5.0
N_BAYS = 16  # per side
BAY_Z0 = -10.0
CEIL_H = 2.6
WALL_X = AISLE_W / 2 + BAY_D + 0.5
Z_BACK, Z_END = -16.0, BAY_Z0 + N_BAYS * BAY_W + 6.0
PILLAR = 0.5
PILLAR_EVERY = 3  # bays
LUM_PITCH = 5.0
EGO_Z = 12.0
EGO_L, EGO_W = 4.6, 1.85
# Nominal luminaires: twin 58 W T8 (2 x 5200 lm, 80 % LOR) and an LED batten [lm, length, width]
LUMINAIRE = {"fluorescent": (8300.0, 1.5, 0.12), "led": (4400.0, 1.2, 0.08)}
EXIT_SIGN_LUMINANCE = 200.0  # cd/m^2, green ISO 7010 E001 panel (nominal)
CAR_MODELS = ("car_bmw_m6", "car_pontiac_gto", "car_vintage")
PAINTS = ("silver", "red", "blue", "black", "white", "gray", "darkgreen")
# Surround-view mounts relative to the ego centre: (offset xyz, view direction, pitch down [deg])
VIEWS = {
    "front": ((0.0, 0.65, EGO_L / 2 + 0.03), (0.0, 1.0), 20.0),
    "rear": ((0.0, 0.95, -EGO_L / 2 - 0.03), (0.0, -1.0), 25.0),
    "left": ((-EGO_W / 2 - 0.12, 1.0, 0.9), (-1.0, 0.0), 45.0),
    "right": ((EGO_W / 2 + 0.12, 1.0, 0.9), (1.0, 0.0), 45.0),
}


BALLAST_PERCENT_FLICKER = {"magnetic": 35.0, "electronic": 5.0}  # nominal percent flicker at 2 x mains


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=Path, default=root)
    ap.add_argument("--out-dir", type=Path, default=None, help="default: scenes/generated/garage")
    ap.add_argument("--asset-manifest", type=Path, default=None)
    ap.add_argument("--allow-missing-assets", action="store_true")
    ap.add_argument("--proxy-cars", action="store_true")
    ap.add_argument("--seed", type=int, default=3, help="Bay occupancy, paints and dead luminaires.")
    ap.add_argument("--view", choices=tuple(VIEWS), default="front")
    ap.add_argument("--lighting", choices=("fluorescent", "led", "mixed"), default="mixed")
    ap.add_argument("--fluorescent-type", choices=("f2", "f11"), default="f2", help="CIE F2 cool white or F11 triband.")
    ap.add_argument("--lights-off-fraction", type=float, default=0.15)
    ap.add_argument(
        "--mains-flicker",
        choices=("on", "off"),
        default="on",
        help="Luminaire light output ripples at twice --mains-hz; each row is on one phase of a three-phase "
        "supply. The frame average is unchanged; tools/render_time_slices.py renders the ripple.",
    )
    ap.add_argument("--mains-hz", type=float, choices=(50.0, 60.0), default=50.0)
    ap.add_argument(
        "--mains-phases",
        type=int,
        choices=(1, 3),
        default=3,
        help="3 = rows on L1/L2/L3 (their ripples largely cancel in the ambient light); 1 = all on one phase.",
    )
    ap.add_argument(
        "--fluorescent-ballast",
        choices=tuple(BALLAST_PERCENT_FLICKER),
        default="magnetic",
        help="magnetic = 35 %% percent flicker, electronic = 5 %% (nominal).",
    )
    ap.add_argument("--led-percent-flicker", type=float, default=10.0, help="LED batten driver ripple [%%], nominal.")
    ap.add_argument("--exposure-s", type=float, default=None, help="Integration time; records the exposure.")
    ap.add_argument("--rolling-line-time-s", type=float, default=15e-6, help="With --exposure-s.")
    ap.add_argument("--occupancy", type=float, default=0.7, help="Fraction of bays with a car.")
    ap.add_argument("--ego-body", action=argparse.BooleanOptionalAction, default=True)
    add_camera_args(ap, fov=120.0, focus_distance=2.0)
    add_film_args(ap, xres=1280, yres=960, pixelsamples=256)
    ap.add_argument("--maxdepth", type=int, default=8)
    ap.set_defaults(
        camera="realistic", lensfile="config/lenses/fisheye.10mm.dat", film_diagonal_mm=34.0, aperture_diameter_mm=2.0
    )
    args = ap.parse_args(argv)
    if not 0.0 <= args.lights_off_fraction <= 1.0 or not 0.0 <= args.occupancy <= 1.0:
        ap.error("--lights-off-fraction and --occupancy must be in [0, 1]")
    if not 0.0 <= args.led_percent_flicker <= 100.0:
        ap.error("--led-percent-flicker must be in [0, 100]")
    if args.exposure_s is not None and args.exposure_s <= 0:
        ap.error("--exposure-s must be > 0")
    return args


def bays() -> list[dict]:
    """Bay rectangles both sides of the aisle: corners (x, z) on the floor, near edge first."""
    out = []
    for side in (1, -1):
        x0, x1 = side * AISLE_W / 2, side * (AISLE_W / 2 + BAY_D)
        for i in range(N_BAYS):
            z0 = BAY_Z0 + i * BAY_W
            out.append(
                {
                    "id": f"{'R' if side > 0 else 'L'}{i:02d}",
                    "side": side,
                    "centre": [(x0 + x1) / 2, z0 + BAY_W / 2],
                    "corners_xz": [[x0, z0], [x0, z0 + BAY_W], [x1, z0 + BAY_W], [x1, z0]],
                }
            )
    return out


def bay_lines_rects() -> list[tuple[float, float, float, float]]:
    w = 0.12
    rects = []
    for side in (1, -1):
        xa, xb = sorted((side * AISLE_W / 2, side * (AISLE_W / 2 + BAY_D)))
        rects += [(xa, xb, BAY_Z0 + i * BAY_W - w / 2, BAY_Z0 + i * BAY_W + w / 2) for i in range(N_BAYS + 1)]
        xr = side * (AISLE_W / 2 + BAY_D) - (w if side > 0 else 0.0)
        rects.append((xr, xr + w, BAY_Z0, BAY_Z0 + N_BAYS * BAY_W))
    return rects


def pillar_positions() -> list[tuple[float, float]]:
    xs = (AISLE_W / 2 + PILLAR / 2 + 0.05, -(AISLE_W / 2 + PILLAR / 2 + 0.05))
    zs = [BAY_Z0 + i * BAY_W for i in range(0, N_BAYS + 1, PILLAR_EVERY)]
    return [(x, z) for z in zs for x in xs]


def luminaire_rows(lighting: str) -> list[tuple[float, str]]:
    bay_x = AISLE_W / 2 + BAY_D / 2
    aisle = "led" if lighting == "mixed" else lighting
    bay = "fluorescent" if lighting == "mixed" else lighting
    return [(0.0, aisle), (bay_x, bay), (-bay_x, bay)]


def mains_ripple(args, kind: str, row: int) -> MainsRipple | None:
    """Light-output ripple of a luminaire in row ``row`` (phase L1/L2/L3 = row mod 3), or None."""
    if args.mains_flicker != "on":
        return None
    pf = BALLAST_PERCENT_FLICKER[args.fluorescent_ballast] if kind == "fluorescent" else args.led_percent_flicker
    # a 120 deg mains phase shift is 1 / (3 f) in time; the ripple has period 1 / (2 f)
    return MainsRipple(2.0 * args.mains_hz, pf / 100.0, phase_s=_phase(args, row) / (3.0 * args.mains_hz))


def _phase(args, row: int) -> int:
    return row % 3 if getattr(args, "mains_phases", 3) == 3 else 0


def luminaire_lines(args, rng: np.random.Generator, out_dir: Path | None = None) -> tuple[list, list, list]:
    """Ceiling battens: pbrt lines, per-luminaire manifest entries and flicker emitters (one per row)."""
    L, meta, emitters = ["# ---- ceiling luminaires"], [], []
    zs = np.arange(Z_BACK + 3.0, Z_END - 2.0, LUM_PITCH)
    window = (0.0, args.exposure_s) if getattr(args, "exposure_s", None) else None
    for row, (x, kind) in enumerate(luminaire_rows(args.lighting)):
        flux, length, width = LUMINAIRE[kind]
        lum = flux / (math.pi * length * width)
        spd = f"spd/lamp_{'fl_' + args.fluorescent_type if kind == 'fluorescent' else 'led_batten'}.spd"
        ripple = mains_ripple(args, kind, row) if out_dir is not None else None
        peak = lum / ripple.duty if ripple else lum  # time average = nominal luminance
        lit: list[tuple[float, float, float]] = []
        for z in zs:
            on = bool(rng.random() >= args.lights_off_fraction)
            L += ['NamedMaterial "galvanized"', *mesh(*box(x, CEIL_H - 0.08, z, width + 0.06, 0.07, length + 0.06))]
            c = (x, CEIL_H - 0.085, float(z))
            if on and ripple is None:
                L += night._quad_light(c, (width / 2, 0, 0), (0, 0, length / 2), spd, lum)
            elif on:
                lit.append(c)
            else:
                p, t = quads_mesh([(x - width / 2, x + width / 2, z - length / 2, z + length / 2)], CEIL_H - 0.086)
                L += ['NamedMaterial "diffuser_off"', *mesh(p, t)]
            meta.append(
                {"x": x, "z": float(z), "kind": kind, "on": on, "flux_lm": flux, "luminance_cd_m2": lum, "row": row}
            )
        if ripple and lit:

            def lines_for_level(level, lit=lit, width=width, length=length, spd=spd, peak=peak):
                out = []
                for c in lit:
                    if level > 0:
                        out += night._quad_light(c, (width / 2, 0, 0), (0, 0, length / 2), spd, peak * level)
                    else:
                        x0, z0 = c[0], c[2]
                        p, t = quads_mesh([(x0 - width / 2, x0 + width / 2, z0 - length / 2, z0 + length / 2)], c[1])
                        out += ['NamedMaterial "diffuser_off"', *mesh(p, t)]
                return out

            inc, entry = write_emitter(
                out_dir,
                f"luminaire_row{row}",
                lines_for_level,
                pwm=ripple,
                exposure_window_s=window,
                meta={
                    "kind": f"luminaire_{kind}",
                    "row_x_m": x,
                    "mains_phase": f"L{_phase(args, row) + 1}",
                    "peak_luminance_cd_m2": peak,
                    "luminaires": len(lit),
                },
            )
            L.append(inc)
            emitters.append(entry)
    return L + [""], meta, emitters


def exit_sign_lines() -> tuple[list[str], list[dict]]:
    L, meta = ["# ---- exit signs (green panels on the end walls)"], []
    for z, facing in ((Z_END - 0.02, -1.0), (Z_BACK + 0.02, 1.0)):
        c = (0.0, 2.25, z)
        right = (0.2 * -facing, 0.0, 0.0)
        L += night._quad_light(c, right, (0.0, 0.1, 0.0), "spd/lamp_exit_green.spd", EXIT_SIGN_LUMINANCE)
        meta.append({"centre": list(c), "size_m": [0.4, 0.2], "luminance_cd_m2": EXIT_SIGN_LUMINANCE})
    return L + [""], meta


def ego_body_lines() -> list[str]:
    return [
        "# ---- ego car body (simple box, so the fisheye sees its own bumper/flank)",
        'NamedMaterial "ego_paint"',
        *mesh(*box(0.0, 0.25, EGO_Z, EGO_W, 0.75, EGO_L)),
        *mesh(*box(0.0, 1.0, EGO_Z - 0.3, EGO_W - 0.25, 0.45, 2.4)),
        'NamedMaterial "rubber"',
        *[
            ln
            for sx in (1, -1)
            for sz in (1, -1)
            for ln in mesh(*box(sx * (EGO_W / 2 - 0.12), 0.0, EGO_Z + sz * 1.4, 0.22, 0.62, 0.62))
        ],
        "",
    ]


def camera_pose(view: str) -> tuple[np.ndarray, np.ndarray]:
    off, (dx, dz), pitch = VIEWS[view]
    eye = np.array([off[0], off[1], EGO_Z + off[2]])
    p = math.radians(pitch)
    d = np.array([dx * math.cos(p), -math.sin(p), dz * math.cos(p)])
    return eye, eye + 10.0 * d


def lamp_spectra(repo: Path, wl: np.ndarray) -> dict[str, np.ndarray]:
    def csv(name: str) -> np.ndarray:
        a = np.loadtxt(repo / "spectra" / "illuminant" / "interpolated" / f"{name}.csv", delimiter=",")
        return np.interp(wl, a[:, 0], a[:, 1], left=0.0, right=0.0)

    g = np.exp(-4 * math.log(2) * ((wl - 525.0) / 30.0) ** 2)
    return {
        "fl_f2": csv("F2_CWF"),
        "fl_f11": csv("F11"),
        "led_batten": csv("LED_B4"),
        "exit_green": np.where(g < 1e-12, 0.0, g),
    }


def main(argv: list[str] | None = None) -> None:  # noqa: C901 (linear scene assembly)
    args = parse_args(argv)
    repo = args.repo_root.resolve()
    out_dir = (args.out_dir or repo / "scenes" / "generated" / "garage").resolve()
    spd = out_dir / "spd"
    (out_dir / "out").mkdir(parents=True, exist_ok=True)
    amanifest = load_asset_manifest(args.asset_manifest or repo / "config" / "highway_assets.yaml")
    aroot = cache_root(repo, amanifest)
    rng = np.random.default_rng(args.seed)
    used_assets: dict[str, dict] = {}
    fallbacks: list[str] = []

    def asset(aid: str):
        p = aroot / "prepared" / aid / "prepared.json"
        if not p.is_file():
            if not args.allow_missing_assets:
                sys.exit(f"error: asset {aid} not prepared ({p}); fetch it or pass --allow-missing-assets")
            fallbacks.append(aid)
            return None
        a = amanifest["assets"][aid]
        used_assets[aid] = {k: a.get(k) for k in ("title", "source", "author", "license") if a.get(k)}
        return json.loads(p.read_text()), aroot / "raw" / aid

    wl = np.arange(args.spectral_lambda_min, args.spectral_lambda_max + 1e-9, args.step_nm)
    for name in SURFACES:
        write_spd(spd / f"{name}.spd", wl, reflectance(name, wl))
    for name, v in lamp_spectra(repo, wl).items():
        write_spd(spd / f"lamp_{name}.spd", wl, v)

    eye, target = camera_pose(args.view)
    film_out = "out/garage_spectral.exr" if args.film == "spectral" else "out/garage.exr"
    L = [
        "# Generated by tools/build_garage_scene.py — do not hand-edit.",
        'ColorSpace "srgb"',
        f'Sampler "zsobol" "integer pixelsamples" [{args.pixelsamples}]',
        f'Integrator "volpath" "integer maxdepth" [{args.maxdepth}] "bool regularize" true',
        'PixelFilter "box" "float xradius" [0.5] "float yradius" [0.5]',
        *film_lines(args, film_out),
        f"LookAt {_pts(eye)} {_pts(target)} 0 1 0",
        *camera_lines(args, repo, out_dir),
        "",
        "WorldBegin",
        "",
        'MakeNamedMaterial "concrete" "string type" "diffuse" "spectrum reflectance" "spd/concrete.spd"',
        'MakeNamedMaterial "floor" "string type" "coateddiffuse" "spectrum reflectance" "spd/concrete.spd"'
        ' "float roughness" [0.25] "float thickness" [0.001]',
        'MakeNamedMaterial "wall_paint" "string type" "diffuse" "spectrum reflectance" "spd/paint_road_white.spd"',
        'MakeNamedMaterial "paint_white" "string type" "coateddiffuse" "spectrum reflectance"'
        ' "spd/paint_road_white.spd" "float roughness" [0.3]',
        'MakeNamedMaterial "paint_yellow" "string type" "diffuse" "spectrum reflectance" "spd/paint_road_yellow.spd"',
        'MakeNamedMaterial "paint_black" "string type" "diffuse" "spectrum reflectance" "spd/sheeting_black.spd"',
        'MakeNamedMaterial "galvanized" "string type" "conductor" "spectrum eta" "metal-Al-eta"'
        ' "spectrum k" "metal-Al-k" "float roughness" [0.3]',
        'MakeNamedMaterial "diffuser_off" "string type" "diffuse" "spectrum reflectance" "spd/paint_road_white.spd"',
        'MakeNamedMaterial "rubber" "string type" "diffuse" "spectrum reflectance" "spd/rubber.spd"',
        'MakeNamedMaterial "ego_paint" "string type" "coateddiffuse" "spectrum reflectance"'
        ' "spd/carpaint_silver.spd" "float roughness" [0.05]',
        "",
    ]
    x0, x1 = -WALL_X, WALL_X
    p, t = quads_mesh([(x0, x1, Z_BACK, Z_END)], 0.0)
    L += ['NamedMaterial "floor"', *mesh(p, t)]
    p, t = quads_mesh(bay_lines_rects(), 0.003)
    L += ['NamedMaterial "paint_white"', *mesh(p, t)]
    p, t = quads_mesh([(-0.06, 0.06, Z_BACK + 1.0, Z_END - 1.0)], 0.003)  # aisle centre line
    L += ['NamedMaterial "paint_yellow"', *mesh(p, t)]
    L += [
        'NamedMaterial "concrete"',
        *mesh(*box(0.0, CEIL_H, (Z_BACK + Z_END) / 2, 2 * WALL_X + 0.4, 0.3, Z_END - Z_BACK + 0.4)),
        'NamedMaterial "wall_paint"',
        *mesh(*box(WALL_X + 0.1, 0.0, (Z_BACK + Z_END) / 2, 0.2, CEIL_H, Z_END - Z_BACK)),
        *mesh(*box(-WALL_X - 0.1, 0.0, (Z_BACK + Z_END) / 2, 0.2, CEIL_H, Z_END - Z_BACK)),
        *mesh(*box(0.0, 0.0, Z_END + 0.1, 2 * WALL_X, CEIL_H, 0.2)),
        *mesh(*box(0.0, 0.0, Z_BACK - 0.1, 2 * WALL_X, CEIL_H, 0.2)),
        "",
        "# ---- pillars (hazard band at the base)",
    ]
    pillars = pillar_positions()
    for x, z in pillars:
        L += ['NamedMaterial "concrete"', *mesh(*box(x, 0.5, z, PILLAR, CEIL_H - 0.5, PILLAR))]
        L += ['NamedMaterial "paint_yellow"', *mesh(*box(x, 0.0, z, PILLAR + 0.004, 0.25, PILLAR + 0.004))]
        L += ['NamedMaterial "paint_black"', *mesh(*box(x, 0.25, z, PILLAR + 0.004, 0.25, PILLAR + 0.004))]
    L.append("")

    lum, lum_meta, lum_emitters = luminaire_lines(args, rng, out_dir)
    L += lum
    ex, ex_meta = exit_sign_lines()
    L += ex
    if args.ego_body:
        L += ego_body_lines()

    bay_meta = bays()
    cars = []
    for k, b in enumerate(bay_meta):
        b["occupied"] = bool(rng.random() < args.occupancy)
        if not b["occupied"]:
            continue
        model, paint = CAR_MODELS[k % 3], PAINTS[int(rng.integers(len(PAINTS)))]
        ca = None if args.proxy_cars else asset(model)
        length = amanifest["assets"][model].get("length_m", 4.6)
        paint_spd = f"spd/carpaint_{paint}.spd"
        body = car_include(*ca, out_dir, b["id"], paint_spd, length) if ca else proxy_car(paint_spd, b["id"])
        inc = out_dir / "cars" / f"{b['id']}.pbrt"
        inc.parent.mkdir(exist_ok=True)
        inc.write_text("\n".join(body) + "\n")
        heading = 90.0 if b["side"] > 0 else -90.0  # nose to the wall
        x = b["side"] * (AISLE_W / 2 + BAY_D - min(length, BAY_D - 0.2) / 2 - 0.15)
        z = b["centre"][1]
        L += [
            "AttributeBegin",
            f"Translate {_f(x)} 0 {_f(z)}",
            f"Rotate {_f(heading + (90.0 if ca else 0.0))} 0 1 0",
            f'Include "{os.path.relpath(inc, out_dir)}"',
            "AttributeEnd",
        ]
        cars.append(
            {
                "id": b["id"],
                "bay": b["id"],
                "model": model if ca else "proxy",
                "paint": paint,
                **night.car_geometry(ca[0] if ca else None, length, x, heading),
            }
        )

    scene_path = out_dir / "garage.pbrt"
    scene_path.write_text("\n".join(L) + "\n")
    lit = [m for m in lum_meta if m["on"]]
    manifest = {
        "scene": _rel(repo, scene_path),
        "generator": "tools/build_garage_scene.py",
        "units": "metres; y up; aisle along +z; +x right of the ego car",
        "film": {"type": args.film, "filename": film_out, "xresolution": args.xres, "yresolution": args.yres},
        "camera": {
            "view": args.view,
            "eye": eye.tolist(),
            "target": target.tolist(),
            "type": args.camera,
            "lensfile": args.lensfile if args.camera == "realistic" else None,
            "film_diagonal_mm": args.film_diagonal_mm if args.camera == "realistic" else None,
        },
        "ego": {"centre_xz": [0.0, EGO_Z], "length_m": EGO_L, "width_m": EGO_W, "body": args.ego_body},
        "layout": {
            "aisle_width_m": AISLE_W,
            "bay_width_m": BAY_W,
            "bay_depth_m": BAY_D,
            "ceiling_height_m": CEIL_H,
            "wall_x_m": WALL_X,
            "z_range_m": [Z_BACK, Z_END],
        },
        "bays": bay_meta,
        "pillars_xz": [list(p) for p in pillars],
        "lighting": {
            "mode": args.lighting,
            "fluorescent_spectrum": {"f2": "CIE F2 (cool white)", "f11": "CIE F11 (triband)"}[args.fluorescent_type],
            "led_spectrum": "CIE LED-B4",
            "luminaires": lum_meta,
            "luminaires_on": len(lit),
            "total_flux_lm": float(sum(m["flux_lm"] for m in lit)),
            "exit_signs": ex_meta,
            "daylight": "none (enclosed deck)",
        },
        "cars": cars,
        "assets": used_assets,
        "missing_assets_fallback": fallbacks,
        "approximations": [
            "Luminaires are flat Lambertian diffusers (no housing optics); fluxes are nominal catalogue values.",
            "Mains flicker: a sinusoidal ripple at twice the mains frequency with nominal percent flicker "
            "(magnetic ballast 35 %, electronic 5 %, LED driver --led-percent-flicker); real waveforms are "
            "not sinusoidal and lamps on the same phase ripple in step.",
            "Concrete, paint and the ego body are analytic spectra and simple boxes.",
        ],
    }
    manifest["emitters"] = lum_emitters
    manifest["lighting"]["mains_flicker"] = (
        None
        if args.mains_flicker != "on"
        else {
            "mains_hz": args.mains_hz,
            "ripple_hz": 2.0 * args.mains_hz,
            "fluorescent_ballast": args.fluorescent_ballast,
            "percent_flicker": {
                "fluorescent": BALLAST_PERCENT_FLICKER[args.fluorescent_ballast],
                "led": args.led_percent_flicker,
            },
            "phases": "row r on phase L(r mod 3 + 1), 120 deg apart" if args.mains_phases == 3 else "all on L1",
        }
    )
    if args.exposure_s:
        manifest["exposure"] = {
            "integration_time_s": args.exposure_s,
            "shutter_open_s": 0.0,
            "rolling_shutter_line_time_s": args.rolling_line_time_s,
            "readout_rows": args.yres,
            "row_window": "row r integrates [r * line_time, r * line_time + integration_time]",
        }
    (out_dir / "garage_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"wrote {_rel(repo, scene_path)} (+ garage_manifest.json); {len(cars)}/{len(bay_meta)} bays occupied, "
        f"{len(lit)}/{len(lum_meta)} luminaires on"
    )


if __name__ == "__main__":
    main()
