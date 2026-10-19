#!/usr/bin/env python3
"""Unlit two-lane rural road at night: high-beam glare, reflector posts, a deer crossing.

    venv/bin/python tools/build_rural_scene.py --out-dir scenes/generated/rural
    pbrt scenes/generated/rural/rural.pbrt

No street lighting: the only light is the night sky (optionally the moon), the ego headlamps
(`--ego-beam low|high`) and an oncoming car (`--oncoming-beam`, default high: the glare case).
Reflector posts (delineators) stand 0.5 m beyond each paved edge every 50 m with retroreflective
reflectors. `--deer roe|red` puts a procedural deer across the ego lane, with retroreflective
eyes (tapetum lucidum eyeshine). Reuses highway_night (sky, lamps, retroreflection), highway_wet
and the highway car assets.
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
import highway_wet as wet  # noqa: E402
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
from highway_spectra import SURFACES, reflectance  # noqa: E402

LANE_W = 3.25
EDGE_STRIP = 0.5  # paved strip outside the edge line [m]
PAVED = LANE_W + EDGE_STRIP
Z0, Z1 = -20.0, 1200.0
DASH, GAP = 6.0, 12.0  # rural broken centre line
POST_SPACING = 50.0
POST_OFFSET = 0.5  # beyond the paved edge
POST_H = 1.0
# Nominal deer sizes (shoulder height, body length) [m]
DEER = {"roe": (0.70, 1.10), "red": (1.20, 1.90)}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=Path, default=root)
    ap.add_argument("--out-dir", type=Path, default=None, help="default: scenes/generated/rural")
    ap.add_argument("--asset-manifest", type=Path, default=None)
    ap.add_argument("--allow-missing-assets", action="store_true")
    ap.add_argument("--proxy-cars", action="store_true")
    ap.add_argument("--seed", type=int, default=5, help="Tree placement.")
    ap.add_argument("--ego-beam", choices=("low", "high"), default="low")
    ap.add_argument("--oncoming", choices=("car", "none"), default="car")
    ap.add_argument("--oncoming-beam", choices=("low", "high"), default="high")
    ap.add_argument("--oncoming-distance-m", type=float, default=120.0)
    ap.add_argument("--deer", choices=("none", *DEER), default="roe")
    ap.add_argument("--deer-distance-m", type=float, default=55.0)
    ap.add_argument("--deer-offset-m", type=float, default=1.2, help="Lateral position (+ = right of centre line).")
    ap.add_argument("--deer-heading-deg", type=float, default=-90.0, help="-90 = walking right-to-left.")
    add_camera_args(ap, fov=30.0, focus_distance=40.0)
    ap.add_argument("--cam-height", type=float, default=1.3)
    ap.add_argument("--cam-pitch", type=float, default=-0.5)
    add_film_args(ap, xres=1280, yres=720, pixelsamples=256)
    ap.add_argument("--maxdepth", type=int, default=8)
    night.add_night_args(ap)
    wet.add_args(ap)
    ap.set_defaults(time_of_day="night", streetlights="none", brake_lights="none")
    args = ap.parse_args(argv)
    if args.time_of_day == "day":
        ap.error("the rural scene is night/dusk only (use tools/build_highway_scene.py for daytime)")
    if args.streetlights != "none":
        ap.error("the rural road is unlit (--streetlights none)")
    if args.puddles or args.spray:
        ap.error("--puddles/--spray need the highway road-wear map")
    return args


def post_positions() -> list[tuple[float, float, int]]:
    """(x, z, side) of the reflector posts on both edges, every 50 m from 25 m."""
    zs = np.arange(25.0, Z1 - 10.0, POST_SPACING)
    x = PAVED + POST_OFFSET + 0.05
    return [(side * x, float(z), side) for z in zs for side in (1, -1)]


def post_lines() -> tuple[list[str], int]:
    """White 1 m posts with a black band; right: one 40x180 mm reflector, left: two 60 mm reflectors."""
    pts = post_positions()
    white, band, refl = [], [], []
    for x, z, side in pts:
        white.append(box(x, 0.0, z, 0.10, POST_H, 0.10))
        band.append(box(x, 0.68, z - 0.001, 0.102, 0.25, 0.102))
        zf = z - 0.052
        if side > 0:
            refl.append((x - 0.02, x + 0.02, 0.74, 0.92, zf))
        else:
            refl += [(x - 0.03, x + 0.03, 0.82, 0.88, zf), (x - 0.03, x + 0.03, 0.73, 0.79, zf)]
    L = ["# ---- reflector posts (delineators)", 'NamedMaterial "post_white"']
    for b in white:
        L += mesh(*b)
    L.append('NamedMaterial "sheet_black"')
    for b in band:
        L += mesh(*b)
    P = np.array([p for x0, x1, y0, y1, z in refl for p in ((x0, y0, z), (x0, y1, z), (x1, y1, z), (x1, y0, z))])
    T = np.array([t for i in range(len(refl)) for t in ((4 * i, 4 * i + 1, 4 * i + 2), (4 * i, 4 * i + 2, 4 * i + 3))])
    L += ['NamedMaterial "sheet_white"', *mesh(P, T), ""]
    return L, len(pts)


def road_marks() -> list[tuple[float, float, float, float]]:
    out = [(-LANE_W - 0.12, -LANE_W, Z0, Z1), (LANE_W, LANE_W + 0.12, Z0, Z1)]
    out += [(-0.06, 0.06, z, z + DASH) for z in np.arange(Z0, Z1, DASH + GAP)]
    return out


def _ellipsoid(c, r, mat: str) -> list[str]:
    return [
        "AttributeBegin",
        f'NamedMaterial "{mat}"',
        f"Translate {_pts(c)}",
        f"Scale {_f(r[0])} {_f(r[1])} {_f(r[2])}",
        'Shape "sphere" "float radius" [1]',
        "AttributeEnd",
    ]


def deer_lines(kind: str, x: float, z: float, heading_deg: float) -> tuple[list[str], dict]:
    """Procedural deer in a local frame (+x = forward), rotated about y by the heading."""
    sh, length = DEER[kind]
    s = sh / 0.70
    leg_h = 0.62 * sh
    body_c = (0.0, leg_h + 0.17 * s, 0.0)
    parts = _ellipsoid(body_c, (length / 2, 0.19 * s, 0.15 * s), "deer_fur")
    neck = (0.42 * length, leg_h + 0.38 * s, 0.0)
    parts += _ellipsoid(neck, (0.09 * s, 0.22 * s, 0.07 * s), "deer_fur")
    head = (0.52 * length + 0.06 * s, leg_h + 0.56 * s, 0.0)
    parts += _ellipsoid(head, (0.13 * s, 0.07 * s, 0.065 * s), "deer_fur")
    eyes = [(head[0] + 0.05 * s, head[1] + 0.03 * s, side * 0.055 * s) for side in (1, -1)]
    for e in eyes:
        parts += _ellipsoid(e, (0.012 * s, 0.012 * s, 0.012 * s), "deer_eye")
    parts.append('NamedMaterial "deer_fur"')
    for lx in (0.36 * length, -0.36 * length):
        for lz in (0.08 * s, -0.08 * s):
            parts += mesh(*box(lx, 0.0, lz, 0.05 * s, leg_h + 0.05, 0.05 * s))
    lines = [
        f"# ---- deer ({kind}, procedural)",
        "AttributeBegin",
        f"Translate {_f(x)} 0 {_f(z)}",
        f"Rotate {_f(heading_deg)} 0 1 0",
        *parts,
        "AttributeEnd",
        "",
    ]
    th = math.radians(heading_deg)
    fx, fz = math.cos(th), -math.sin(th)  # local +x after pbrt Rotate about +y
    ext = np.array([[length / 2 + 0.2 * s, 0.0, 0.0], [0.0, 0.0, 0.2 * s]])
    hx = abs(ext[0, 0] * fx) + abs(ext[1, 2] * fz)
    hz = abs(ext[0, 0] * fz) + abs(ext[1, 2] * fx)
    world_eyes = [[x + e[0] * fx + e[2] * fz, e[1], z + e[0] * fz - e[2] * fx] for e in eyes]
    meta = {
        "kind": kind,
        "shoulder_height_m": sh,
        "body_length_m": length,
        "position_world": [x, 0.0, z],
        "heading_deg": heading_deg,
        "bbox_world": {"min": [x - hx, 0.0, z - hz], "max": [x + hx, leg_h + 0.66 * s, z + hz]},
        "eyes_world": world_eyes,
        "eyeshine": "retroreflective (nominal sheeting lobe), yellow-green tint",
        "model": "procedural ellipsoids; nominal sizes, fur reflectance analytic",
    }
    return lines, meta


def fur_reflectance(wl: np.ndarray) -> np.ndarray:
    """Brown summer coat: dark in the blue, rising through red into the NIR (nominal)."""
    return 0.05 + 0.25 / (1.0 + np.exp(-(wl - 640.0) / 45.0)) + 0.15 / (1.0 + np.exp(-(wl - 760.0) / 30.0))


def eyeshine_reflectance(wl: np.ndarray) -> np.ndarray:
    return 0.05 + 0.6 * np.exp(-(((wl - 545.0) / 60.0) ** 2))


def main(argv: list[str] | None = None) -> None:  # noqa: C901 (linear scene assembly)
    args = parse_args(argv)
    repo = args.repo_root.resolve()
    out_dir = (args.out_dir or repo / "scenes" / "generated" / "rural").resolve()
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
    write_spd(spd / "deer_fur.spd", wl, fur_reflectance(wl))
    write_spd(spd / "eyeshine.spd", wl, eyeshine_reflectance(wl))
    nopts = night.resolve(args)

    eye = np.array([LANE_W / 2, args.cam_height, 0.0])
    target = eye + 100.0 * np.array([0.0, math.tan(math.radians(args.cam_pitch)), 1.0])
    film_out = "out/rural_spectral.exr" if args.film == "spectral" else "out/rural.exr"
    L = [
        "# Generated by tools/build_rural_scene.py — do not hand-edit.",
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
    ]
    nat, lighting = night.natural_light(args, out_dir, spd)
    L += nat
    L += [
        'MakeNamedMaterial "asphalt" "string type" "coateddiffuse" "spectrum reflectance" "spd/asphalt_aged.spd"'
        ' "float roughness" [0.35] "float thickness" [0.001]',
        'MakeNamedMaterial "paint_white" "string type" "coateddiffuse" "spectrum reflectance"'
        ' "spd/paint_road_white.spd" "float roughness" [0.3]',
        'MakeNamedMaterial "grass" "string type" "diffuse" "spectrum reflectance" "spd/grass.spd"',
        'MakeNamedMaterial "bark" "string type" "diffuse" "spectrum reflectance" "spd/bark.spd"',
        'MakeNamedMaterial "leaf" "string type" "diffuse" "spectrum reflectance" "spd/leaf_reflectance.spd"',
        'MakeNamedMaterial "post_white" "string type" "diffuse" "spectrum reflectance" "spd/paint_road_white.spd"',
        'MakeNamedMaterial "sheet_white" "string type" "diffuse" "spectrum reflectance" "spd/sheeting_white.spd"',
        'MakeNamedMaterial "sheet_black" "string type" "diffuse" "spectrum reflectance" "spd/sheeting_black.spd"',
        'MakeNamedMaterial "deer_fur" "string type" "diffuse" "spectrum reflectance" "spd/deer_fur.spd"',
        'MakeNamedMaterial "deer_eye" "string type" "diffuse" "spectrum reflectance" "spd/eyeshine.spd"',
        "",
    ]
    if nopts["retroreflective"]:
        retro = night.retro_material_lines(spd, out_dir, wl)
        eye_retro = [
            ln.replace('"sheet_white"', '"deer_eye"').replace("sheeting_white.spd", "eyeshine.spd")
            for ln in retro["sheet_white"]
        ]
        L = night.replace_named_materials(L, {**retro, "deer_eye": eye_retro})

    p, t = quads_mesh([(-PAVED, PAVED, Z0, Z1)], 0.0)
    L += ['NamedMaterial "asphalt"', *mesh(p, t)]
    p, t = quads_mesh(road_marks(), 0.004)
    L += ['NamedMaterial "paint_white"', *mesh(p, t)]
    p, t = quads_mesh([(-400.0, -PAVED, Z0, Z1 + 800), (PAVED, 400.0, Z0, Z1 + 800)], -0.03)
    L += ['NamedMaterial "grass"', *mesh(p, t), ""]
    pl, n_posts = post_lines()
    L += pl

    trees = []
    for side in (1, -1):
        z = 8.0 + 10.0 * rng.random()
        while z < 600.0:
            x = side * (PAVED + 4.0 + 6.0 * rng.random())
            h = 6.0 + 6.0 * rng.random()
            trees.append((x, z, h))
            z += 9.0 + 16.0 * rng.random()
    L.append("# ---- trees (procedural trunk + crown)")
    for x, z, h in trees:
        L += ['NamedMaterial "bark"', *mesh(*box(x, 0.0, z, 0.3, 0.55 * h, 0.3))]
        L += _ellipsoid((x, 0.62 * h, z), (0.22 * h, 0.3 * h, 0.22 * h), "leaf")
    L.append("")

    cars = []
    if args.oncoming == "car":
        model = "car_bmw_m6"
        ca = None if args.proxy_cars else asset(model)
        length = amanifest["assets"][model].get("length_m", 4.6)
        body = (
            car_include(*ca, out_dir, "oncoming", "spd/carpaint_silver.spd", length)
            if ca
            else proxy_car("spd/carpaint_silver.spd", "oncoming")
        )
        inc = out_dir / "cars" / "oncoming.pbrt"
        inc.parent.mkdir(exist_ok=True)
        inc.write_text("\n".join(body) + "\n")
        x, z = -LANE_W / 2, args.oncoming_distance_m
        L += [
            "AttributeBegin",
            f"Translate {_f(x)} 0 {_f(z)}",
            f"Rotate {_f(180.0 + (90.0 if ca else 0.0))} 0 1 0",
            f'Include "{os.path.relpath(inc, out_dir)}"',
            "AttributeEnd",
            "",
        ]
        cars.append(
            {
                "id": "oncoming",
                "model": model if ca else "proxy",
                "lane": -1,
                "z": z,
                "distance_m": z,
                "beam": args.oncoming_beam,
                **night.car_geometry(ca[0] if ca else None, length, x, 180.0),
            }
        )

    assets = night.write_light_assets(out_dir, spd, nopts, high_beam=True)

    def beam_assets(beam: str) -> dict:
        return {k: assets[f"{k}_high"] if beam == "high" else assets[k] for k in ("halogen", "led")}

    vopts = dict(nopts, brake="none", ego_lane=0)
    lamp_meta: dict = {"ego_beam": args.ego_beam}
    vl, vm = night.vehicle_light_lines([], (float(eye[0]), float(eye[2])), vopts, beam_assets(args.ego_beam))
    L += vl
    lamp_meta["ego"] = vm["headlamps"]
    if cars:
        vl, vm = night.vehicle_light_lines(
            cars, (float(eye[0]), float(eye[2])), dict(vopts, ego_lamps=False), beam_assets(args.oncoming_beam)
        )
        L += vl
        lamp_meta["oncoming"] = vm["headlamps"]
    lamp_meta["high_beam_peak_cd_nominal"] = night.HIGH_BEAM_PEAK_CD

    deer_meta = None
    if args.deer != "none":
        dl, deer_meta = deer_lines(args.deer, args.deer_offset_m, args.deer_distance_m, args.deer_heading_deg)
        L += dl

    wet_meta = None
    if wet.enabled(args):
        L, wet_meta = wet.apply_materials(L, args, wl, spd, out_dir)

    scene_path = out_dir / "rural.pbrt"
    scene_path.write_text("\n".join(L) + "\n")
    manifest = {
        "scene": _rel(repo, scene_path),
        "generator": "tools/build_rural_scene.py",
        "units": "metres; y up; road along +z; +x image-right; right-hand traffic",
        "film": {"type": args.film, "filename": film_out, "xresolution": args.xres, "yresolution": args.yres},
        "camera": {"eye": eye.tolist(), "target": target.tolist(), "type": args.camera, "fov_deg": args.fov},
        "layout": {
            "lane_width_m": LANE_W,
            "paved_half_width_m": PAVED,
            "centre_line": f"{DASH:g} m dash / {GAP:g} m gap",
        },
        "lighting": {**lighting, "street_lighting": "none", **lamp_meta},
        "delineators": {
            "count": n_posts,
            "spacing_m": POST_SPACING,
            "offset_from_paved_edge_m": POST_OFFSET,
            "height_m": POST_H,
            "reflectors": "right: one 40x180 mm, left: two 60x60 mm, retroreflective sheeting model (nominal)",
        },
        "trees": len(trees),
        "cars": cars,
        "deer": deer_meta,
        "road": {"wet": wet_meta},
        "assets": used_assets,
        "missing_assets_fallback": fallbacks,
        "approximations": [
            "Delineator reflectors and deer eyeshine use the traffic-sign sheeting lobe, not measured cube-corner or tapetum data.",
            "High beam is an analytic H-V centred driving beam (70 kcd per lamp, nominal) added to the passing beam.",
            "Deer and trees are procedural; deer sizes and fur reflectance are nominal.",
        ],
    }
    (out_dir / "rural_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {_rel(repo, scene_path)} (+ rural_manifest.json); {n_posts} reflector posts, {len(trees)} trees")


if __name__ == "__main__":
    main()
