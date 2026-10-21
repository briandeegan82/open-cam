#!/usr/bin/env python3
"""Urban four-way intersection (night + wet by default): signals, shop LED signage, facades.

    venv/bin/python tools/build_urban_scene.py --out-dir scenes/generated/urban
    pbrt scenes/generated/urban/urban.pbrt

The ego camera waits on the near approach of a two-lane street (one lane per direction, a parking
lane and a 3 m sidewalk each side) facing a signalised crossing with zebra crossings on every arm.
Reuses the highway modules: spectra and car assets (build_highway_scene), dusk/night sky, lamps
and retroreflection (highway_night), wet road (highway_wet), sign artwork (highway_signs) and
AEB targets (highway_vru). Traffic-signal aspects and shop LED signs are PWM emitters
(highway_incar.write_emitter), so tools/render_time_slices.py renders their flicker.
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

import facade_kits as fk  # noqa: E402
import highway_night as night  # noqa: E402
import highway_signs as sign_scene  # noqa: E402
import highway_vru as vru  # noqa: E402
import highway_wet as wet  # noqa: E402
import street_furniture as sf  # noqa: E402
import two_wheelers as tw  # noqa: E402
from build_highway_scene import (  # noqa: E402
    CAR_MODELS,
    _f,
    _pts,
    _rel,
    add_camera_args,
    add_film_args,
    box,
    camera_lines,
    car_include,
    film_lines,
    luminance_texture,
    mesh,
    proxy_car,
    quads_mesh,
    sun_world,
    write_spd,
)
from fetch_highway_assets import cache_root, load_asset_manifest  # noqa: E402
from highway_incar import PWM, write_emitter  # noqa: E402
from highway_incar import led_spectrum as _led  # noqa: E402
from highway_sky import analytic_clear_sky, clear_sky_illuminance_lux, solar_direct_spectrum, write_rgb_exr  # noqa: E402
from highway_spectra import SURFACES, reflectance  # noqa: E402

LANE_W = 3.5
PARK_W = 2.3
WALK_W = 3.0
KERB_H = 0.15
HALF = LANE_W + PARK_W  # kerb line from the street centre [m]
BUILD_X = HALF + WALK_W  # facade line
Z_INT = 32.0  # intersection centre ahead of the camera [m]
Z0, Z1 = -30.0, 160.0
X_FAR = 120.0
ZEBRA_W = 4.0  # crossing width along the approach
ZEBRA_SET = 1.5  # crossing setback from the corner kerb line
STOP_SET = 2.0  # stop line before the crossing

# Nominal signal photometry: 200 mm EN 12368 aspects ~ 200-400 cd on axis; L = I / lens area.
ASPECT_D = 0.2
SIGNAL_CD = {"red": 300.0, "amber": 300.0, "green": 300.0}
SIGNAL_LED = {"red": (626.0, 17.0), "amber": (592.0, 15.0), "green": (505.0, 30.0)}
SHOP_LED = {"red": (626.0, 17.0), "blue": (465.0, 22.0), "green": (525.0, 30.0)}
SHOP_LUMINANCE = 400.0  # time-averaged [cd/m^2], nominal (EN 12966-like advertising panel)
WINDOW_LUMINANCE = 15.0  # lit office/flat window at night [cd/m^2], nominal


def led_spectrum(wl: np.ndarray, peak_nm: float, fwhm_nm: float) -> np.ndarray:
    """LED spectrum with denormal tails zeroed (pbrt cannot parse 1e-295)."""
    v = _led(wl, peak_nm, fwhm_nm)
    return np.where(v < 1e-12 * v.max(), 0.0, v)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=Path, default=root)
    ap.add_argument("--out-dir", type=Path, default=None, help="default: scenes/generated/urban")
    ap.add_argument("--asset-manifest", type=Path, default=None, help="default: config/highway_assets.yaml")
    ap.add_argument("--allow-missing-assets", action="store_true", help="Fall back to proxies when not fetched.")
    ap.add_argument("--proxy-cars", action="store_true", help="Box proxies instead of the CC0 car models.")
    ap.add_argument("--seed", type=int, default=3, help="Facade heights, lit windows, shop sign colours.")
    ap.add_argument("--sun-elevation", type=float, default=35.0, help="Daytime analytic sky.")
    ap.add_argument("--sun-azimuth", type=float, default=140.0)
    ap.add_argument("--signal-state", choices=("red", "amber", "green"), default="red", help="Ego-facing signals.")
    ap.add_argument(
        "--signal-pwm-hz", type=float, default=0.0, help="Traffic-signal LED PWM [Hz] (0 = DC, the default)."
    )
    ap.add_argument("--signal-duty", type=float, default=0.5)
    ap.add_argument(
        "--signal-model",
        choices=("procedural", *sf.SIGNAL_MODELS),
        default="procedural",
        help="Signal-head housing: procedural box or an imported Objaverse CC-BY mesh (aspects stay LED emitters).",
    )
    ap.add_argument(
        "--street-furniture",
        choices=("none", "pavement"),
        default="none",
        help="Hydrants, bins, benches and utility cabinets on both pavements (Poly Haven CC0).",
    )
    ap.add_argument(
        "--work-zone",
        choices=("none", "cones", "drums"),
        default="none",
        help="Close the near-right parking lane (z 8-20 m) with a channelizer taper and concrete barriers.",
    )
    ap.add_argument(
        "--two-wheelers",
        choices=("none", "parked"),
        default="none",
        help="Parked bicycles and kick e-scooters on the pavements, motorbikes and scooters in empty parking bays "
        "(Objaverse CC-BY).",
    )
    ap.add_argument("--shop-signs", type=int, default=4, help="Shop LED signs on the ground floors.")
    ap.add_argument("--shop-pwm-hz", type=float, default=100.0, help="Shop LED PWM [Hz] (0 = DC).")
    ap.add_argument("--shop-duty", type=float, default=0.3)
    ap.add_argument("--lit-windows", type=float, default=0.35, help="Fraction of windows lit at dusk/night.")
    ap.add_argument(
        "--facade-style",
        choices=("procedural", "kits"),
        default="procedural",
        help="Street fronts: procedural boxes with window quads, or assembled Poly Haven CC0 apartment/factory kits.",
    )
    ap.add_argument(
        "--facade-textures",
        choices=("none", "mixed"),
        default="none",
        help="ambientCG CC0 brick/plaster/concrete maps on the street lots (spectral base kept; texture adds detail).",
    )
    ap.add_argument("--exposure-s", type=float, default=None, help="Integration time; records the exposure.")
    ap.add_argument("--rolling-line-time-s", type=float, default=15e-6, help="With --exposure-s.")
    add_camera_args(ap, fov=40.0, focus_distance=20.0)
    ap.add_argument("--cam-height", type=float, default=1.35)
    ap.add_argument("--cam-pitch", type=float, default=0.0)
    add_film_args(ap, xres=1280, yres=720, pixelsamples=256)
    ap.add_argument("--maxdepth", type=int, default=8)
    night.add_night_args(ap)
    wet.add_args(ap)
    vru.add_args(ap)
    sign_scene.add_args(ap)
    ap.set_defaults(
        time_of_day="night",
        road_wetness="wet",
        brake_lights="none",
        vru_distance_m=Z_INT - HALF - ZEBRA_SET - ZEBRA_W / 2,
        vru_offset_m=0.5,
        sign_start_m=Z_INT + 12.0,
        sign_spacing_m=9.0,
        sign_offset_m=0.6,
    )
    args = ap.parse_args(argv)
    if args.puddles or args.spray:
        ap.error("--puddles/--spray need the highway road-wear map; not available in the urban scene")
    if not 0.0 <= args.lit_windows <= 1.0:
        ap.error("--lit-windows must be in [0, 1]")
    vru.check_args(args)
    sign_scene.check_args(args)
    return args


def cross_half() -> float:
    return HALF


def zebra_rects() -> list[tuple[float, float, float, float]]:
    """0.5 m bars / 0.5 m gaps across all four arms (x0, x1, z0, z1)."""
    out = []
    zc = Z_INT - HALF - ZEBRA_SET - ZEBRA_W / 2
    bars = np.arange(-HALF + 0.25, HALF - 0.25, 1.0)
    for z_mid in (zc, 2 * Z_INT - zc):  # near and far approaches
        out += [(x, x + 0.5, z_mid - ZEBRA_W / 2, z_mid + ZEBRA_W / 2) for x in bars]
    for x_mid in (-(HALF + ZEBRA_SET + ZEBRA_W / 2), HALF + ZEBRA_SET + ZEBRA_W / 2):  # side arms
        out += [(x_mid - ZEBRA_W / 2, x_mid + ZEBRA_W / 2, Z_INT + z, Z_INT + z + 0.5) for z in bars]
    return out


def lane_markings() -> list[tuple[float, float, float, float]]:
    """Broken centre lines on the approaches, stop lines on each approach (right-hand traffic)."""
    out = []
    near_end = Z_INT - HALF - ZEBRA_SET - ZEBRA_W
    for z0, z1 in ((Z0, near_end - STOP_SET), (2 * Z_INT - near_end + STOP_SET, Z1)):
        out += [(-0.06, 0.06, z, min(z + 3.0, z1)) for z in np.arange(z0, z1, 9.0)]
    out.append((0.0, LANE_W, near_end - STOP_SET - 0.3, near_end - STOP_SET))  # ego stop line
    out.append((-LANE_W, 0.0, 2 * Z_INT - near_end + STOP_SET, 2 * Z_INT - near_end + STOP_SET + 0.3))
    for x0, x1 in ((-X_FAR, -(HALF + ZEBRA_SET + ZEBRA_W + STOP_SET)), (HALF + ZEBRA_SET + ZEBRA_W + STOP_SET, X_FAR)):
        out += [(x, min(x + 3.0, x1), Z_INT - 0.06, Z_INT + 0.06) for x in np.arange(x0, x1, 9.0)]
    return out


def corner_blocks() -> list[tuple[float, float, float, float]]:
    """Building footprints (x0, x1, z0, z1) on the four corners behind the sidewalks."""
    zn, zf = Z_INT - BUILD_X, Z_INT + BUILD_X
    return [(-X_FAR, -BUILD_X, Z0, zn), (BUILD_X, X_FAR, Z0, zn), (-X_FAR, -BUILD_X, zf, Z1), (BUILD_X, X_FAR, zf, Z1)]


def building_lots(rng: np.random.Generator) -> list[dict]:
    """Facades along the main street, split into 8-16 m lots of 3-6 storeys."""
    lots = []
    for side in (-1, 1):
        for z0, z1 in ((Z0, Z_INT - BUILD_X), (Z_INT + BUILD_X, Z1)):
            z = z0
            while z < z1 - 4.0:
                w = float(min(rng.uniform(8.0, 16.0), z1 - z))
                lots.append({"side": side, "z0": z, "z1": z + w, "storeys": int(rng.integers(3, 7)), "depth": 14.0})
                z += w
    return lots


def _quad(p0, du, dv) -> tuple[np.ndarray, np.ndarray]:
    p0, du, dv = (np.asarray(v, float) for v in (p0, du, dv))
    return np.array([p0, p0 + du, p0 + du + dv, p0 + dv]), np.array([(0, 1, 2), (0, 2, 3)])


FACADE_TEX = (
    "tex_bricks059",
    "tex_bricks085",
    "tex_bricks092",
    "tex_bricks060",
    "tex_bricks093",
    "tex_concrete034",
    "tex_paintedplaster017",
    "tex_paintedplaster004",
)


def textured_box(cx: float, cz: float, sx: float, h: float, sz: float, tile: float):
    """Lot box (walls + roof) with metric UVs: u along the wall, v up, both divided by the tile size."""
    x0, x1, z0, z1 = cx - sx / 2, cx + sx / 2, cz - sz / 2, cz + sz / 2
    faces = [
        ((x0, 0, z0), (0, 0, sz), (0, h, 0), (z0, 0), "z"),
        ((x1, 0, z0), (0, 0, sz), (0, h, 0), (z0, 0), "z"),
        ((x0, 0, z0), (sx, 0, 0), (0, h, 0), (x0, 0), "x"),
        ((x0, 0, z1), (sx, 0, 0), (0, h, 0), (x0, 0), "x"),
        ((x0, h, z0), (sx, 0, 0), (0, 0, sz), (x0, z0), "xz"),
    ]
    P, T, UV = [], [], []
    for i, (p0, du, dv, (u0, v0), _) in enumerate(faces):
        p, t = _quad(p0, du, dv)
        lu, lv = float(np.linalg.norm(du)), float(np.linalg.norm(dv))
        P.append(p)
        T.append(t + 4 * i)
        UV.append(np.array([(u0, v0), (u0 + lu, v0), (u0 + lu, v0 + lv), (u0, v0 + lv)]) / tile)
    return np.concatenate(P), np.concatenate(T), np.concatenate(UV)


def facade_texture_lines(
    args, asset, out_dir: Path, n_lots: int, meta: dict
) -> tuple[list[str], list | None, list[dict]]:
    """Opt-in: one textured material per ambientCG map; each lot draws one from its own RNG."""
    pool = [(aid, a) for aid in FACADE_TEX if (a := asset(aid)) is not None] if args.facade_textures != "none" else []
    if not pool:
        return [], None, []
    L = ["# ---- facade textures (ambientCG CC0)"]
    for aid, (info, raw) in pool:
        lum = out_dir / "textures" / f"{aid}_lum.exr"
        luminance_texture(raw / info["maps"]["color"], lum)
        normal = os.path.relpath(raw / info["maps"]["normal"], out_dir)
        L += [
            f'Texture "{aid}:lum" "float" "imagemap" "string filename" "{os.path.relpath(lum, out_dir)}"',
            f'Texture "{aid}" "spectrum" "scale" "spectrum tex" "spd/{meta[aid]["spd"]}.spd" "texture scale" "{aid}:lum"',
            f'MakeNamedMaterial "{aid}" "string type" "diffuse" "texture reflectance" "{aid}" "string normalmap" "{normal}"',
        ]
    trng = np.random.default_rng([args.seed, 11])
    picks = [pool[int(trng.integers(len(pool)))] for _ in range(n_lots)]
    tex = [(aid, float(meta[aid]["tile_size_m"])) for aid, (info, _) in picks]
    return L + [""], tex, [{"lot": k, "texture": aid} for k, (aid, _) in enumerate(tex)]


def facade_lines(
    lots: list[dict], rng: np.random.Generator, dark: bool, lit: float, tex: list | None = None, kit: bool = False
) -> tuple[list[str], dict]:
    """Lot boxes plus a window grid on the street face (lit windows emit at dusk/night)."""
    L = ["# ---- facades (procedural)"]
    lit_n = dark_n = 0
    win_lit, win_dark = [], []
    for k, lot in enumerate(lots):
        h = 0.8 + 3.2 * lot["storeys"]
        s = lot["side"]
        x_face = s * BUILD_X
        cx = x_face + s * lot["depth"] / 2
        depth = lot["depth"]
        if kit:  # box behind the kit wall plane, kit storey height
            h, depth = fk.STOREY * lot["storeys"] + 0.3, lot["depth"] - fk.RECESS
            cx = x_face + s * (fk.RECESS + depth / 2)
        mat = ("facade_render", "facade_brick", "concrete")[k % 3]
        cz, sz = (lot["z0"] + lot["z1"]) / 2, lot["z1"] - lot["z0"]
        if tex:
            L += [f'NamedMaterial "{tex[k][0]}"', *mesh(*textured_box(cx, cz, depth, h, sz, tex[k][1]))]
        else:
            L += [f'NamedMaterial "{mat}"', *mesh(*box(cx, 0.0, cz, depth, h, sz))]
        for fl in range(1, lot["storeys"]):
            y = 0.8 + 3.2 * fl + 0.9
            for z in np.arange(lot["z0"] + 1.2, lot["z1"] - 1.6, 2.6):
                p, t = _quad((x_face - s * 0.01, y, z), (0, 0, 1.4), (0, 1.5, 0))
                if s > 0:
                    t = t[:, ::-1]
                on = dark and rng.random() < lit  # drawn in kit mode too, so the rest of the scene is unchanged
                if kit:
                    continue
                (win_lit if on else win_dark).append((p, t))
                lit_n += on
                dark_n += not on
    for name, quads in (("window_dark", win_dark), ("window_lit", win_lit)):
        if not quads:
            continue
        P = np.concatenate([p for p, _ in quads])
        T = np.concatenate([t + 4 * i for i, (_, t) in enumerate(quads)])
        L += ["AttributeBegin", f'NamedMaterial "{name}"']
        if name == "window_lit":
            L.append(
                f'AreaLightSource "diffuse" "spectrum L" "spd/lamp_window.spd" "float scale" [{_f(WINDOW_LUMINANCE)}]'
            )
        L += [*mesh(P, T), "AttributeEnd"]
    return L + [""], {"lots": len(lots), "windows_lit": int(lit_n), "windows_dark": int(dark_n)}


def emissive_panel(centre, size, normal_x: float, spd: str, luminance: float, mat: str) -> list[str]:
    """Rectangle in a plane of constant x (normal_x = +-1) or z (normal_x = 0, facing -z)."""
    cx, cy, cz = centre
    w, h = size
    if normal_x:
        p, t = _quad((cx, cy - h / 2, cz - w / 2), (0, 0, w), (0, h, 0))
        if normal_x > 0:
            t = t[:, ::-1]
    else:
        p, t = _quad((cx - w / 2, cy - h / 2, cz), (w, 0, 0), (0, h, 0))
        t = t[:, ::-1]
    out = ["AttributeBegin", f'NamedMaterial "{mat}"']
    if luminance > 0:
        out.append(f'AreaLightSource "diffuse" "spectrum L" "{spd}" "float scale" [{_f(luminance)}]')
    return out + [*mesh(p, t), "AttributeEnd"]


def signal_heads(args) -> list[dict]:
    """Ego approach: near-right primary and far-right secondary heads; cross arms show the opposite."""
    zs = Z_INT - HALF - 0.6
    cross = {"red": "green", "amber": "red", "green": "red"}[args.signal_state]
    return [
        {"id": "sig_near_right", "pos": (HALF + 0.6, 2.6, zs), "face": "-z", "state": args.signal_state},
        {"id": "sig_far_right", "pos": (HALF + 0.6, 2.6, 2 * Z_INT - zs), "face": "-z", "state": args.signal_state},
        {"id": "sig_far_left", "pos": (-HALF - 0.6, 2.6, 2 * Z_INT - zs), "face": "-z", "state": args.signal_state},
        {"id": "sig_cross_right", "pos": (HALF + 0.6, 2.6, Z_INT + HALF + 0.6), "face": "-x", "state": cross},
        {"id": "sig_cross_left", "pos": (-HALF - 0.6, 2.6, zs), "face": "+x", "state": cross},
    ]


def disc_emitter(centre, yaw_deg: float, spd: str, luminance: float, mat: str) -> list[str]:
    """Round aspect (diameter ASPECT_D) facing +z rotated by yaw; same area as the procedural square."""
    return [
        "AttributeBegin",
        f"Translate {_pts(centre)}",
        f"Rotate {_f(yaw_deg)} 0 1 0",
        f'NamedMaterial "{mat}"',
        f'AreaLightSource "diffuse" "spectrum L" "{spd}" "float scale" [{_f(luminance)}]',
        f'Shape "disk" "float radius" [{_f(ASPECT_D / 2)}]',
        "AttributeEnd",
    ]


def signal_lines(args, out_dir: Path, heads: list[dict], window, head_obj=None) -> tuple[list[str], list[dict]]:
    """Pole + housing + three aspects; the lit aspect is a PWM-tagged LED emitter.

    With head_obj = (object name, SIGNAL_MODELS entry) the housing is the imported mesh, scaled so its
    aspect pitch is 0.3 m and its lenses sit 1 cm behind the (unchanged) emitter positions.
    """
    L, emitters = ["# ---- traffic signals"], []
    area = math.pi * (ASPECT_D / 2) ** 2
    pwm = PWM(args.signal_pwm_hz, args.signal_duty) if args.signal_pwm_hz > 0 else None
    duty = args.signal_duty if pwm else 1.0
    for hd in heads:
        x, y, z = hd["pos"]
        nx = {"-z": 0.0, "-x": -1.0, "+x": 1.0}[hd["face"]]
        hw = 0.32
        if nx:
            housing = box(x - nx * 0.15, y, z, 0.25, 0.95, hw)
        else:
            housing = box(x, y, z + 0.15, hw, 0.95, 0.25)
        L += ['NamedMaterial "galvanized"', *mesh(*box(x, 0.0, z + (0.3 if not nx else 0.0), 0.12, y, 0.12))]
        yaw = sf.FACE_YAW[hd["face"]]
        if head_obj:
            _, (_, lx, ly, pitch, lz) = head_obj
            lens = (x + nx * 0.02, y + 0.49, z) if nx else (x, y + 0.49, z + 0.005)
            L += sf.instance_lines(head_obj[0], lens, yaw, 0.3 / pitch, (lx, ly, lz))
        else:
            L += ['NamedMaterial "signal_housing"', *mesh(*housing)]
        for k, colour in enumerate(("red", "amber", "green")):
            cy = y + 0.95 - 0.16 - 0.3 * k
            c = (x + nx * 0.03, cy, z) if nx else (x, cy, z + 0.025 - 0.03)
            if colour != hd["state"]:
                if not head_obj:  # imported heads show their own unlit lenses
                    L += emissive_panel(c, (ASPECT_D * 0.886, ASPECT_D * 0.886), nx, "", 0.0, f"lens_{colour}")
                continue
            lum = SIGNAL_CD[colour] / area / duty  # peak while on (time average = I / area)

            def lines_for_level(level, c=c, colour=colour, lum=lum, nx=nx, yaw=yaw):
                if head_obj:
                    return disc_emitter(c, yaw, f"spd/led_{colour}.spd", lum * level, f"lens_{colour}")
                return emissive_panel(
                    c, (ASPECT_D * 0.886, ASPECT_D * 0.886), nx, f"spd/led_{colour}.spd", lum * level, f"lens_{colour}"
                )

            inc, entry = write_emitter(
                out_dir,
                f"{hd['id']}_{colour}",
                lines_for_level,
                pwm=pwm,
                exposure_window_s=window,
                meta={
                    "kind": "traffic_signal",
                    "aspect": colour,
                    "intensity_cd_nominal": SIGNAL_CD[colour],
                    "led_peak_luminance_cd_m2": lum,
                    "centre_world": list(c),
                    "faces": hd["face"],
                },
            )
            L.append(inc)
            emitters.append(entry)
    return L + [""], emitters


def furniture_lines(args, asset, out_dir: Path, zone) -> tuple[list[str], list[dict]]:
    """Opt-in pavement props and a work-zone lane closure; own RNG so the default scene is unchanged."""
    rng = np.random.default_rng([args.seed, 7])
    L, meta, defined = ["# ---- street furniture"], [], {}

    def place(aid: str, x: float, y: float, z: float, yaw: float, vk: int = 0) -> None:
        if aid not in defined:
            a = asset(aid)
            defined[aid] = None
            if a is not None:
                info, raw = a
                prep = raw.parent.parent / "prepared" / aid
                defined[aid] = sf.variants(aid, info, prep)
                for v in defined[aid]:
                    L.extend(sf.object_lines(v["name"], info, prep, raw, out_dir, v["shapes"]))
        if not defined[aid]:
            return
        v = defined[aid][vk % len(defined[aid])]
        s, origin = sf.placement(v, sf.PROPS[aid][1])
        L.extend(sf.instance_lines(v["name"], (x, y, z), yaw, s, origin))
        meta.append(
            {
                "asset": aid,
                "variant": v["name"],
                "role": sf.PROPS[aid][0],
                "position": [round(x, 3), round(y, 3), round(z, 3)],
                "yaw_deg": round(yaw, 2),
                "scale": round(s, 5),
                "box_world": sf.world_box(v, (x, y, z), yaw, s, origin),
            }
        )

    if args.street_furniture == "pavement":
        kerb = [a for a, r in sf.PROPS.items() if r[0] == "kerb"]
        facade = [a for a, r in sf.PROPS.items() if r[0] == "facade"]
        ranges = [(3.0, Z_INT - BUILD_X - 2.0), (Z_INT + BUILD_X + 2.0, Z_INT + 45.0)]
        for side in (1, -1):
            avoid = [p[2] for p in streetlight_positions() if p[3] == side]
            avoid += [h["pos"][2] for h in signal_heads(args) if h["pos"][0] * side > 0]
            front = -90.0 if side > 0 else 90.0  # model +z (front) faces the street
            for z in sf.pavement_slots(rng, ranges, avoid):
                if rng.random() < 0.5:
                    aid, x, yaw = kerb[int(rng.integers(len(kerb)))], side * (HALF + 0.45), front + rng.uniform(-10, 10)
                else:
                    aid, x, yaw = facade[int(rng.integers(len(facade)))], side * (BUILD_X - 0.35), front
                place(aid, x, KERB_H, z, float(yaw), int(rng.integers(2)))
    if zone:
        z0, z1 = zone
        chan = [a for a, r in sf.PROPS.items() if r[0] == args.work_zone[:-1]]
        for k, (x, z) in enumerate(sf.work_zone_layout(HALF - 0.5, HALF - PARK_W + 0.45, z0, z1)):
            place(chan[k % len(chan)], x, 0.0, z, float(rng.uniform(0.0, 360.0)))
        barriers = [a for a, r in sf.PROPS.items() if r[0] == "barrier"]
        for k, z in enumerate(np.arange(z0 + 7.0, z1, 1.6)):
            place(barriers[k % len(barriers)], HALF - 0.5, 0.0, float(z), 90.0)
    return L + [""], meta


def two_wheeler_lines(args, asset, out_dir: Path, free_bays, furniture) -> tuple[list[str], list[dict]]:
    """Opt-in parked two-wheelers; own RNG so the rest of the scene is unchanged."""
    rng = np.random.default_rng([args.seed, 13])
    p = tw.Placer(asset, out_dir)
    for side, z in free_bays:  # 2-3 per empty bay, front to the street, angled 30 deg towards the traffic
        n = int(rng.integers(2, 4))
        for j in range(n):
            aid = tw.POWERED[int(rng.integers(len(tw.POWERED)))]
            pos = (side * (HALF - PARK_W / 2), 0.0, z + (j - (n - 1) / 2) * 1.2)
            p.place(aid, pos, -90.0 * side + 30.0 + float(rng.uniform(-8, 8)), "parking_bay")
    ranges = [(3.0, Z_INT - BUILD_X - 2.0), (Z_INT + BUILD_X + 2.0, Z_INT + 45.0)]
    for side in (1, -1):  # bicycle groups across the pavement next to the kerb, front to the facades
        avoid = [q[2] for q in streetlight_positions() if q[3] == side]
        avoid += [h["pos"][2] for h in signal_heads(args) if h["pos"][0] * side > 0]
        avoid += [f["position"][2] for f in furniture if f["position"][0] * side > 0]
        for z in sf.pavement_slots(rng, ranges, [], spacing=9.0):
            n = int(rng.integers(2, 5))
            zs = [z + (j - (n - 1) / 2) * 0.75 for j in range(n)]
            if any(abs(zb - a) < 1.2 for zb in zs for a in avoid) or not (ranges[0][0] <= min(zs)):
                continue
            for zb in zs:
                aid = tw.BICYCLES[int(rng.integers(len(tw.BICYCLES)))]
                p.place(aid, (side * (HALF + 1.1), KERB_H, zb), 90.0 * side, "pavement")
            if tw.KICK and rng.random() < 0.4:  # dockless kick e-scooter at the kerb, parallel to it
                zk = max(zs) + 1.0
                if all(abs(zk - a) > 1.2 for a in avoid):
                    p.place(tw.KICK[0], (side * (HALF + 0.4), KERB_H, zk), 0.0 if side > 0 else 180.0, "pavement")
    return (["# ---- parked two-wheelers", *p.lines, ""] if p.meta else []), p.meta


def shop_lines(args, out_dir: Path, lots: list[dict], rng: np.random.Generator, window) -> tuple[list[str], list[dict]]:
    L, emitters = ["# ---- shop LED signs (PWM)"], []
    pwm = PWM(args.shop_pwm_hz, args.shop_duty) if args.shop_pwm_hz > 0 else None
    duty = args.shop_duty if pwm else 1.0
    near = sorted(
        (lot for lot in lots if lot["z1"] - lot["z0"] > 6.0), key=lambda lot: abs((lot["z0"] + lot["z1"]) / 2 - Z_INT)
    )
    for k, lot in enumerate(near[: args.shop_signs]):
        colour = ("red", "blue", "green")[int(rng.integers(0, 3))]
        s = lot["side"]
        w = min(5.0, lot["z1"] - lot["z0"] - 1.5)
        c = (s * (BUILD_X - 0.06), 3.4, (lot["z0"] + lot["z1"]) / 2)
        peak = SHOP_LUMINANCE / duty
        L += ['NamedMaterial "signal_housing"', *mesh(*box(c[0] + s * 0.03, 3.0, c[2], 0.06, 0.8, w + 0.2))]

        def lines_for_level(level, c=c, w=w, colour=colour, peak=peak, s=s):
            return emissive_panel(c, (w, 0.6), -s, f"spd/led_shop_{colour}.spd", peak * level, "led_lens")

        inc, entry = write_emitter(
            out_dir,
            f"shop{k}",
            lines_for_level,
            pwm=pwm,
            exposure_window_s=window,
            meta={
                "kind": "shop_led_sign",
                "colour": colour,
                "luminance_cd_m2_time_averaged": SHOP_LUMINANCE,
                "led_peak_luminance_cd_m2": peak,
                "centre_world": list(c),
                "size_m": [w, 0.6],
            },
        )
        L.append(inc)
        emitters.append(entry)
    return L + [""], emitters


def streetlight_positions() -> list[tuple[float, float, float, int]]:
    """Single-arm 8 m columns on both kerbs, 25 m apart, staggered; (x, y, z, side)."""
    out = []
    for side, offset in ((1, 0.0), (-1, 12.5)):
        for z in np.arange(Z0 + 10.0 + offset, Z1, 25.0):
            if abs(z - Z_INT) < BUILD_X:
                continue
            out.append((side * (HALF + 0.5), 8.0, float(z), side))
    return out


def streetlight_lines(opts: dict, assets: dict) -> tuple[list[str], dict]:
    kind = opts["streetlights"]
    if kind == "none":
        return [], {}
    sl = night.STREETLIGHT[kind]
    spd = f"spd/lamp_{sl['spectrum']}.spd"
    lens = (0.6, 0.26)
    lens_l = night.LENS_FLUX_FRACTION * sl["flux_lm"] / (math.pi * lens[0] * lens[1])
    L = ["# ---- street lights (single-arm columns)"]
    pos = streetlight_positions()
    for x, y, z, side in pos:
        arm = 1.5
        L += ['NamedMaterial "galvanized"', *mesh(*box(x, 0.0, z, 0.16, y + 0.1, 0.16))]
        L += [*mesh(*box(x - side * arm / 2, y + 0.1, z, arm, 0.08, 0.08))]
        lx = x - side * arm
        L += ['NamedMaterial "luminaire"', *mesh(*box(lx, y - 0.1, z, 0.7, 0.14, 0.32))]
        L += [
            "AttributeBegin",
            f"Translate {_f(lx)} {_f(y - 0.102)} {_f(z)}",
            *night._quad_light((0, 0, 0), (0, 0, lens[1] / 2), (lens[0] / 2, 0, 0), spd, lens_l),
            *night._gonio((0, -0.03, 0), 180.0 if side > 0 else 0.0, assets["streetlight"]["map"], spd),
            "AttributeEnd",
        ]
    return L + [""], {
        "kind": kind,
        "spectrum": spd,
        "luminaire_flux_lm": sl["flux_lm"],
        "columns": len(pos),
        "mounting_height_m": 7.9,
        "spacing_m": 25.0,
        "distribution": "analytic full-cutoff IES Type III medium",
    }


def main(argv: list[str] | None = None) -> None:  # noqa: C901 (linear scene assembly)
    args = parse_args(argv)
    repo = args.repo_root.resolve()
    out_dir = (args.out_dir or repo / "scenes" / "generated" / "urban").resolve()
    spd = out_dir / "spd"
    (out_dir / "out").mkdir(parents=True, exist_ok=True)
    amanifest = load_asset_manifest(args.asset_manifest or repo / "config" / "highway_assets.yaml")
    aroot = cache_root(repo, amanifest)
    used_assets: dict[str, dict] = {}
    fallbacks: list[str] = []
    rng = np.random.default_rng(args.seed)

    def asset(aid: str):
        p = aroot / "prepared" / aid / "prepared.json"
        if not p.is_file():
            if not args.allow_missing_assets:
                sys.exit(
                    f"error: asset {aid} not prepared ({p}); run tools/fetch_highway_assets.py {aid} or pass --allow-missing-assets"
                )
            fallbacks.append(aid)
            return None
        a = amanifest["assets"][aid]
        used_assets[aid] = {k: a.get(k) for k in ("title", "source", "author", "license") if a.get(k)}
        return json.loads(p.read_text()), aroot / "raw" / aid

    wl = np.arange(args.spectral_lambda_min, args.spectral_lambda_max + 1e-9, args.step_nm)
    for name in SURFACES:
        write_spd(spd / f"{name}.spd", wl, reflectance(name, wl))
    write_spd(
        spd / "facade_render.spd",
        wl,
        0.55 * reflectance("concrete", wl) / max(reflectance("concrete", wl).max(), 1e-6) + 0.1,
    )
    write_spd(spd / "facade_brick.spd", wl, 0.08 + 0.32 / (1 + np.exp(-(wl - 590.0) / 25.0)))
    for colour, (pk, fw) in SIGNAL_LED.items():
        write_spd(spd / f"led_{colour}.spd", wl, led_spectrum(wl, pk, fw))
        write_spd(
            spd / f"lens_{colour}.spd",
            wl,
            0.02 + 0.10 * led_spectrum(wl, pk, 3 * fw) / led_spectrum(wl, pk, 3 * fw).max(),
        )
    for colour, (pk, fw) in SHOP_LED.items():
        write_spd(spd / f"led_shop_{colour}.spd", wl, led_spectrum(wl, pk, fw))
    write_spd(spd / "lamp_window.spd", wl, night.pc_led(wl, 3000.0, red_phosphor=0.45))
    nopts = night.resolve(args)
    dark = nopts["dark"]
    window = (0.0, args.exposure_s) if args.exposure_s else None

    camera_x = LANE_W / 2
    eye = np.array([camera_x, args.cam_height, 0.0])
    target = eye + 100.0 * np.array([0.0, math.tan(math.radians(args.cam_pitch)), 1.0])
    film_out = "out/urban_spectral.exr" if args.film == "spectral" else "out/urban.exr"
    L = [
        "# Generated by tools/build_urban_scene.py — do not hand-edit.",
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
    lighting: dict
    if dark:
        nat, lighting = night.natural_light(args, out_dir, spd)
        L += nat
    else:
        elev = float(args.sun_elevation)
        zs = math.radians(90.0 - elev)
        sky_rot, sun_w = sun_world(np.array([math.sin(zs), 0.0, math.cos(zs)]), args.sun_azimuth)
        sky_file = out_dir / "textures" / f"sky_cie12_{elev:.1f}.exr"
        if not sky_file.is_file():
            write_rgb_exr(sky_file, analytic_clear_sky(512, elev))
        e_dn, e_sky = clear_sky_illuminance_lux(elev)
        write_spd(spd / "sun.spd", wl, solar_direct_spectrum(wl, elev))
        L += [
            'LightSource "distant" "spectrum L" "spd/sun.spd"',
            f'    "float illuminance" [{_f(e_dn)}] "point3 from" [{_pts(sun_w)}] "point3 to" [0 0 0]',
            "AttributeBegin",
            f"    Rotate {_f(sky_rot)} 0 1 0",
            "    Rotate -90 1 0 0",
            f'    LightSource "infinite" "string filename" "{os.path.relpath(sky_file, out_dir)}" "float illuminance" [{_f(e_sky)}]',
            "AttributeEnd",
            "",
        ]
        lighting = {
            "time_of_day": "day",
            "sun_elevation_deg": elev,
            "illuminance_normal_lux": e_dn,
            "sky_horizontal_lux": e_sky,
        }

    L += [
        'MakeNamedMaterial "asphalt" "string type" "coateddiffuse" "spectrum reflectance" "spd/asphalt_aged.spd"'
        ' "float roughness" [0.35] "float thickness" [0.001]',
        'MakeNamedMaterial "paint_white" "string type" "coateddiffuse" "spectrum reflectance" "spd/paint_road_white.spd" "float roughness" [0.3]',
        'MakeNamedMaterial "concrete" "string type" "diffuse" "spectrum reflectance" "spd/concrete.spd"',
        'MakeNamedMaterial "galvanized" "string type" "coateddiffuse" "spectrum reflectance" "spd/galvanized.spd" "float roughness" [0.3]',
        'MakeNamedMaterial "facade_render" "string type" "diffuse" "spectrum reflectance" "spd/facade_render.spd"',
        'MakeNamedMaterial "facade_brick" "string type" "diffuse" "spectrum reflectance" "spd/facade_brick.spd"',
        'MakeNamedMaterial "window_dark" "string type" "coateddiffuse" "rgb reflectance" [0.03 0.035 0.04] "float roughness" [0.0]',
        'MakeNamedMaterial "window_lit" "string type" "diffuse" "rgb reflectance" [0.3 0.3 0.3]',
        'MakeNamedMaterial "signal_housing" "string type" "diffuse" "rgb reflectance" [0.03 0.03 0.03]',
        'MakeNamedMaterial "led_lens" "string type" "diffuse" "rgb reflectance" [0.05 0.05 0.05]',
        'MakeNamedMaterial "luminaire" "string type" "diffuse" "rgb reflectance" [0.08 0.08 0.08]',
        'MakeNamedMaterial "sheet_white" "string type" "diffuse" "spectrum reflectance" "spd/sheeting_white.spd"',
        *[
            f'MakeNamedMaterial "lens_{c}" "string type" "coateddiffuse" "spectrum reflectance" "spd/lens_{c}.spd" "float roughness" [0.05]'
            for c in SIGNAL_LED
        ],
        "",
    ]
    if nopts["retroreflective"]:
        L = night.replace_named_materials(L, night.retro_material_lines(spd, out_dir, wl))
    if sign_scene.enabled(args):
        L += sign_scene.material_lines(args, spd, out_dir, wl, nopts["retroreflective"])

    # ---- ground: streets (asphalt), raised sidewalks (concrete) inside the corner blocks
    streets = [
        (-HALF, HALF, Z0, Z1),
        (-X_FAR, -HALF, Z_INT - HALF, Z_INT + HALF),
        (HALF, X_FAR, Z_INT - HALF, Z_INT + HALF),
    ]
    p, t = quads_mesh(streets, 0.0)
    L += ['NamedMaterial "asphalt"', *mesh(p, t), ""]
    for x0, x1, z0, z1 in (
        (-X_FAR, -HALF, Z0, Z_INT - HALF),
        (HALF, X_FAR, Z0, Z_INT - HALF),
        (-X_FAR, -HALF, Z_INT + HALF, Z1),
        (HALF, X_FAR, Z_INT + HALF, Z1),
    ):
        L += ['NamedMaterial "concrete"', *mesh(*box((x0 + x1) / 2, 0.0, (z0 + z1) / 2, x1 - x0, KERB_H, z1 - z0))]
    p, t = quads_mesh(zebra_rects() + lane_markings(), 0.004)
    L += ['NamedMaterial "paint_white"', *mesh(p, t), ""]

    lots = building_lots(rng)
    kit_lines, kit_meta, kit_win = [], [], {}
    if args.facade_style == "kits":
        kits = {aid: a for aid in fk.KITS if (a := asset(aid)) is not None}
        if kits:
            kit_lines, kit_meta, kit_win = fk.street_fronts(
                lots, kits, BUILD_X, args.seed, dark, args.lit_windows, out_dir, WINDOW_LUMINANCE
            )
    tl, tex, tex_meta = facade_texture_lines(args, asset, out_dir, len(lots), amanifest["assets"])
    L += tl
    fl, facade_meta = facade_lines(lots, rng, dark, args.lit_windows, tex, kit=bool(kit_meta))
    L += fl + kit_lines
    if tex:
        facade_meta = {**facade_meta, "textures": tex_meta}
    if kit_meta:
        facade_meta = {**facade_meta, **kit_win, "kits": kit_meta}
    cross_lots = []
    for sx in (-1, 1):
        for sz in (-1, 1):
            zc = Z_INT + sz * (HALF + WALK_W + 25.0)
            inset = fk.RECESS if kit_meta else 0.0  # keep the kit recess behind the wall plane clear
            cross_lots.append(
                box(sx * (BUILD_X + 30.0 + inset / 2), 0.0, zc, 60.0 - inset, 12.0 + 4.0 * rng.random(), 50.0)
            )
    L += ['NamedMaterial "facade_render"', *[m for b in cross_lots for m in mesh(*b)], ""]

    emitters: list[dict] = []
    heads = signal_heads(args)
    head_obj = None
    if args.signal_model != "procedural":
        spec = sf.SIGNAL_MODELS[args.signal_model]
        ha = asset(spec[0])
        if ha is not None:
            info, raw = ha
            name = f"signal_head_{args.signal_model}"
            L += sf.object_lines(
                name, info, raw.parent.parent / "prepared" / spec[0], raw, out_dir, list(range(len(info["shapes"])))
            )
            head_obj = (name, spec)
    sl, em = signal_lines(args, out_dir, heads, window, head_obj)
    L += sl
    emitters += em
    sh, em = shop_lines(args, out_dir, lots, rng, window)
    L += sh
    emitters += em
    zone = (8.0, 20.0) if args.work_zone != "none" else None
    furniture_meta: list[dict] = []
    if args.street_furniture != "none" or zone:
        fl, furniture_meta = furniture_lines(args, asset, out_dir, zone)
        L += fl

    # ---- cars: parked along both kerbs, one waiting on the cross street, one oncoming
    cars_meta: list[dict] = []
    paints = ("silver", "red", "blue", "black", "white", "gray", "darkgreen")

    def car(inst: str, model: str, paint: str, x: float, z: float, heading: float, lane: int) -> None:
        paint_spd = f"spd/carpaint_{paint}.spd"
        ca = None if args.proxy_cars else asset(model)
        length = amanifest["assets"][model].get("length_m", 4.6)
        body = car_include(*ca, out_dir, inst, paint_spd, length) if ca else proxy_car(paint_spd, inst)
        yaw = 90.0 if ca else 0.0
        inc = out_dir / "cars" / f"{inst}.pbrt"
        inc.parent.mkdir(exist_ok=True)
        inc.write_text("\n".join(body) + "\n")
        nonlocal L
        L += [
            "AttributeBegin",
            f"Translate {_f(x)} 0 {_f(z)}",
            f"Rotate {_f(heading + yaw)} 0 1 0",
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
                "z": z,
                **night.car_geometry(ca[0] if ca else None, length, x, heading),
            }
        )

    k = 0
    free_bays: list[tuple[int, float]] = []
    for side in (1, -1):
        for z in np.arange(4.0, Z_INT - HALF - 6.0, 6.5):
            if vru.enabled(args) and side > 0 and abs(z - args.vru_distance_m) < 4.0:
                continue
            keep = rng.random() < 0.75
            if zone and side > 0 and zone[0] - 3.0 < z < zone[1] + 3.0:
                k += keep  # closed by the work zone; later cars keep their model and paint
                continue
            if keep:
                car(
                    f"park{k}",
                    CAR_MODELS[k % 3],
                    paints[k % len(paints)],
                    side * (HALF - PARK_W / 2),
                    float(z),
                    0.0 if side > 0 else 180.0,
                    9,
                )
                k += 1
            else:
                free_bays.append((side, float(z)))
    car("oncoming", "car_bmw_m6", "white", -LANE_W / 2, Z_INT + HALF + 14.0, 180.0, -1)
    car(
        "cross_wait",
        "car_pontiac_gto",
        "red",
        HALF + ZEBRA_SET + ZEBRA_W + STOP_SET + 3.0,
        Z_INT + LANE_W / 2,
        -90.0,
        5,
    )

    two_wheeler_meta: list[dict] = []
    if args.two_wheelers != "none":
        tl, two_wheeler_meta = two_wheeler_lines(args, asset, out_dir, free_bays, furniture_meta)
        L += tl

    vru_meta = None
    if vru.enabled(args):
        vru_spectra = vru.write_spectra(args, spd, wl, write_spd, reflectance)
        vlines, vru_meta = vru.scene_lines(
            args,
            LANE_W,
            lambda x, y, z: [f"Translate {_f(x)} {_f(y)} {_f(z)}"],
            lambda *a: (_ for _ in ()).throw(SystemExit("--vru-obstruction is highway-only")),
        )
        vru_meta["spectra"] = vru_spectra
        L += vlines

    sign_meta = {}
    if sign_scene.enabled(args):
        slines, sign_meta = sign_scene.scene_lines(args, amanifest, asset, aroot, out_dir, HALF, mesh, box, lambda q: q)
        L += slines

    light_meta: dict = {}
    if dark or nopts["headlamps"] != "off" or nopts["streetlights"] != "none":
        assets = night.write_light_assets(out_dir, spd, nopts)
        moving = [c for c in cars_meta if c["lane"] != 9]  # parked cars are dark
        vopts = dict(nopts, brake=args.brake_lights, ego_lane=0)
        vl, vmeta = night.vehicle_light_lines(moving, (float(eye[0]), float(eye[2])), vopts, assets)
        L += vl
        sll, smeta = streetlight_lines(nopts, assets)
        L += sll
        light_meta = {**vmeta, **({"streetlights": smeta} if smeta else {})}

    wet_meta = None
    if wet.enabled(args):
        L, wet_meta = wet.apply_materials(L, args, wl, spd, out_dir)

    scene_path = out_dir / "urban.pbrt"
    scene_path.write_text("\n".join(L) + "\n")
    manifest = {
        "scene": _rel(repo, scene_path),
        "generator": "tools/build_urban_scene.py",
        "units": "metres; y up; ego street along +z; +x image-right",
        "film": {"type": args.film, "filename": film_out, "xresolution": args.xres, "yresolution": args.yres},
        "camera": {"eye": eye.tolist(), "target": target.tolist(), "type": args.camera, "fov_deg": args.fov},
        "layout": {
            "lane_width_m": LANE_W,
            "parking_lane_m": PARK_W,
            "sidewalk_m": WALK_W,
            "kerb_height_m": KERB_H,
            "intersection_centre_z_m": Z_INT,
            "zebra_width_m": ZEBRA_W,
            "traffic": "right-hand",
        },
        "lighting": {**lighting, **light_meta},
        "signals": [{k2: v for k2, v in h.items()} for h in heads],
        "signal_model": args.signal_model,
        "signal_photometry": "nominal: 300 cd per 200 mm aspect (EN 12368 order of magnitude), LED spectra analytic",
        "facades": facade_meta,
        "cars": cars_meta,
        "road": {"wet": wet_meta},
        "assets": used_assets,
        "missing_assets_fallback": fallbacks,
        "emitters": emitters,
        "approximations": [
            "Facades are procedural boxes with flat window quads; lit windows are diffuse 3000 K emitters.",
            "Signal and shop-sign luminances and PWM settings are nominal, not measured.",
            "Puddles and spray are highway-only (they use the road-wear wheel-path map).",
        ],
    }
    if args.exposure_s:
        manifest["exposure"] = {
            "integration_time_s": args.exposure_s,
            "shutter_open_s": 0.0,
            "rolling_shutter_line_time_s": args.rolling_line_time_s,
            "readout_rows": args.yres,
            "row_window": "row r integrates [r * line_time, r * line_time + integration_time]",
        }
    if vru_meta is not None:
        manifest["vru"] = vru_meta
    if sign_meta:
        manifest["signs"] = sign_meta
    if furniture_meta or args.street_furniture != "none" or zone:
        manifest["street_furniture"] = {
            "pavement": args.street_furniture,
            "work_zone": args.work_zone,
            "work_zone_z_m": list(zone) if zone else None,
            "items": furniture_meta,
        }
    if args.two_wheelers != "none":
        manifest["two_wheelers"] = {"mode": args.two_wheelers, "items": two_wheeler_meta}
        manifest["approximations"].append(
            "Two-wheelers use sRGB base-colour textures (no measured spectra) and nominal overall lengths; "
            "they carry no lamps and their reflectors are not retroreflective."
        )
    if furniture_meta or head_obj:
        manifest["approximations"].append(
            "Street furniture and imported signal heads use sRGB base-colour textures (no measured spectra); "
            "cone and drum sleeves are diffuse, not retroreflective."
        )
    (out_dir / "urban_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {_rel(repo, scene_path)} (+ urban_manifest.json); {len(emitters)} PWM-tagged emitters")


if __name__ == "__main__":
    main()
