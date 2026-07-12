#!/usr/bin/env python3
"""Generate a parameterized autonomous-driving highway scene for pbrt-v4.

Multi-lane road with MUTCD-style markings, guard rails, road signs, cars
(fetched PLY assets with procedural fallback), spectral distant sun and an
optional Hosek-Wilkie sky dome. Emits a spectral-film .pbrt scene plus a
manifest JSON satisfying the pbrt_spectral_exr_to_electrons.py contract.

All surface reflectances are measured-style SPD curves
(spectra/surfaces/highway/); the sun uses a direct-normal solar SPD. Sign
legends (when textured) are the only RGB surfaces. Retroreflection is
approximated as bright diffuse (recorded in the manifest).

Calibration note: for this 3D scene use
``model.pbrt_spectral_exr.radiometric_autocalibration: mean_photopic_lux``
with ``--target-illuminance-lux`` at the electrons stage; the chart-style
illuminant-CSV lux normalization assumes uniform illumination and does not
apply here.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import highway_layout as hl  # noqa: E402
import highway_materials as hm  # noqa: E402
from spectral_sensor_forward import illuminance_lux_from_irradiance, read_csv_curve  # noqa: E402

DEFAULT_SOLAR_SPD = "spectra/illuminant/interpolated/solar_direct_am15.csv"
SURFACE_DIR = "spectra/surfaces/highway"


def _rel(repo: Path, p: Path) -> str:
    try:
        return str(p.relative_to(repo))
    except ValueError:
        return str(p)


def subsample_for_spd(wl: np.ndarray, val: np.ndarray, step_nm: float) -> tuple[np.ndarray, np.ndarray]:
    wmin, wmax = float(wl[0]), float(wl[-1])
    grid = np.arange(wmin, wmax + 1e-9, step_nm, dtype=np.float64)
    if grid[-1] < wmax - 1e-6:
        grid = np.append(grid, wmax)
    return grid, np.interp(grid, wl, val)


def write_spd(path: Path, wl: np.ndarray, val: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"{float(w):.6g}\t{float(v):.12g}" for w, v in zip(wl, val))
    path.write_text("# wavelength_nm\tvalue\n" + body + "\n")


def sun_dir_world(elevation_deg: float, azimuth_deg: float) -> np.ndarray:
    """Unit vector pointing TO the sun. Azimuth 0 = behind the camera (+Z),
    90 = sun on the camera's right (world -X, since image-left = +X),
    measured clockwise as seen by the driver."""
    el, az = np.deg2rad(elevation_deg), np.deg2rad(azimuth_deg)
    return np.array([-np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])


# ---------------------------------------------------------------------------
# Sky dome (Hosek-Wilkie via imgtool makesky)
# ---------------------------------------------------------------------------

def ensure_sky_exr(imgtool: Path, sky_dir: Path, elevation_deg: float,
                   turbidity: float, albedo: float, resolution: int) -> Path:
    """Generate (or reuse cached) Hosek-Wilkie sky environment EXR."""
    name = f"sky_e{elevation_deg:g}_t{turbidity:g}_a{albedo:g}_r{resolution}.exr"
    out = sky_dir / name
    if out.exists():
        return out
    sky_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(imgtool), "makesky",
           "--elevation", f"{elevation_deg:g}",
           "--turbidity", f"{turbidity:g}",
           "--albedo", f"{albedo:g}",
           "--resolution", str(resolution),
           "--outfile", str(out)]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return out


def _equal_area_square_to_dir(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Inverse of pbrt's EqualAreaSphereToSquare ([0,1]^2 -> unit sphere).

    Vectorized; returns array of shape u.shape + (3,).
    """
    u = 2.0 * np.asarray(u, dtype=np.float64) - 1.0
    v = 2.0 * np.asarray(v, dtype=np.float64) - 1.0
    up, vp = np.abs(u), np.abs(v)
    sd = 1.0 - (up + vp)
    r = 1.0 - np.abs(sd)
    ratio = np.divide(vp - up, r, out=np.zeros_like(r), where=r != 0)
    phi = (np.pi / 4.0) * (ratio + 1.0)
    z = (1.0 - r * r) * np.sign(sd)
    s = r * np.sqrt(2.0 - r * r)
    return np.stack([np.cos(phi) * np.sign(u) * s,
                     np.sin(phi) * np.sign(v) * s, z], axis=-1)


def _read_exr_rgb(path: Path) -> np.ndarray:
    import OpenEXR
    import Imath

    f = OpenEXR.InputFile(str(path))
    dw = f.header()["dataWindow"]
    w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    return np.stack([np.frombuffer(f.channel(c, pt), dtype=np.float32).reshape(h, w)
                     for c in ("R", "G", "B")], axis=-1).copy()


def _write_exr_rgb(path: Path, img: np.ndarray) -> None:
    import OpenEXR
    import Imath

    h, w = img.shape[:2]
    hdr = OpenEXR.Header(w, h)
    ft = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    hdr["channels"] = {c: ft for c in ("R", "G", "B")}
    out = OpenEXR.OutputFile(str(path), hdr)
    out.writePixels({c: img[..., i].astype(np.float32).tobytes()
                     for i, c in enumerate(("R", "G", "B"))})
    out.close()


def _texel_dirs(w: int, h: int) -> np.ndarray:
    ys, xs = np.meshgrid((np.arange(h) + 0.5) / h, (np.arange(w) + 0.5) / w, indexing="ij")
    return _equal_area_square_to_dir(xs, ys)


def preprocess_sky_exr(src: Path, dst: Path, sun_cone_deg: float = 4.0) -> Path:
    """Make a makesky dome usable as pure diffuse fill.

    1. Remove the baked solar disk (makesky includes solar radiance; the
       builder's spectral distant light is the sole sun, so leaving the disk
       would double-count direct sun and skew the sun/sky ratio). Disk texels
       are replaced by the median of sky texels in the same elevation band.
    2. Fill the lower hemisphere by extending horizon-band sky color downward
       per azimuth. This removes the dark 'sawtooth' fringe that bilinear
       filtering otherwise bleeds across the octahedral equator seam.
    """
    if dst.exists():
        return dst
    img = _read_exr_rgb(src)
    h, w = img.shape[:2]
    d = _texel_dirs(w, h)                      # map space: z = up
    lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]

    # --- 1. excise the solar disk ------------------------------------------
    iy, ix = np.unravel_index(int(np.argmax(np.where(d[..., 2] > 0, lum, -np.inf))),
                              lum.shape)
    sun_d = d[iy, ix]
    cosang = d @ sun_d
    disk = (cosang > np.cos(np.deg2rad(sun_cone_deg))) & (d[..., 2] > 0)
    band = (np.abs(d[..., 2] - sun_d[2]) < 0.05) & ~disk & (d[..., 2] > 0)
    if band.any():
        img[disk] = np.median(img[band], axis=0)

    # --- 2. extend horizon color into the lower hemisphere ------------------
    az = np.arctan2(d[..., 1], d[..., 0])
    nbins = 720
    bins = np.clip(((az + np.pi) / (2 * np.pi) * nbins).astype(int), 0, nbins - 1)
    horizon = (d[..., 2] > 0.0) & (d[..., 2] < 0.08)
    ring = np.zeros((nbins, 3), dtype=np.float64)
    counts = np.zeros(nbins, dtype=np.int64)
    np.add.at(ring, bins[horizon], img[horizon])
    np.add.at(counts, bins[horizon], 1)
    filled = counts > 0
    ring[filled] /= counts[filled][:, None]
    if (~filled).any():  # fill any empty azimuth bins by nearest filled bin
        idx = np.where(filled)[0]
        for b in np.where(~filled)[0]:
            ring[b] = ring[idx[np.argmin(np.minimum(np.abs(idx - b), nbins - np.abs(idx - b)))]]
    below = d[..., 2] <= 0.0
    img[below] = ring[bins[below]]

    _write_exr_rgb(dst, img.astype(np.float32))
    return dst


def detect_baked_sun_dir(sky_exr: Path) -> np.ndarray:
    """Find the sun direction baked into a makesky EXR (max-luminance texel).

    pbrt's ImageInfiniteLight treats the equal-area image as a light-space
    octahedral map; makesky places the sun at a fixed azimuth. Returning the
    world direction (y-up convention) lets the builder rotate the dome so the
    baked sun matches the requested one.
    """
    img = _read_exr_rgb(sky_exr)
    h, w = img.shape[:2]
    lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    iy, ix = np.unravel_index(int(np.argmax(lum)), lum.shape)
    d = _equal_area_square_to_dir((ix + 0.5) / w, (iy + 0.5) / h)
    # World direction for an unrotated dome under the builder's fixed
    # 'Rotate -90 1 0 0' (z-up map -> y-up world): (x, y, z) -> (x, z, -y).
    return np.array([d[0], d[2], -d[1]])


def hemisphere_illuminance_rel(sky_exr: Path) -> float:
    """Cosine-weighted photopic illuminance of the upper hemisphere, relative
    units (mean luminance x pi in the equal-area map, upper half only)."""
    import OpenEXR
    import Imath

    f = OpenEXR.InputFile(str(sky_exr))
    hdr = f.header()
    dw = hdr["dataWindow"]
    w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    chans = [np.frombuffer(f.channel(c, pt), dtype=np.float32).reshape(h, w)
             for c in ("R", "G", "B")]
    lum = 0.2126 * chans[0] + 0.7152 * chans[1] + 0.0722 * chans[2]

    # Upper hemisphere = texels whose direction has positive up-component.
    ys, xs = np.meshgrid((np.arange(h) + 0.5) / h, (np.arange(w) + 0.5) / w, indexing="ij")
    up = _equal_area_square_to_dir(xs, ys)[..., 2]
    mask = up > 0
    if not mask.any():
        return 0.0
    # Equal-area map: each texel = equal solid angle; E_h = integral(L cos) domega.
    return float(np.mean(lum[mask] * up[mask]) * 2.0 * np.pi)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--out-dir", default="scenes/generated/highway")
    ap.add_argument("--name", default="default", help="Scene slug.")

    g = ap.add_argument_group("road")
    g.add_argument("--lanes", type=int, default=3)
    g.add_argument("--lane-width", type=float, default=3.7)
    g.add_argument("--road-length", type=float, default=400.0)
    g.add_argument("--guardrails", dest="guardrails", action="store_true", default=True)
    g.add_argument("--no-guardrails", dest="guardrails", action="store_false")

    g = ap.add_argument_group("objects")
    g.add_argument("--cars", default=None,
                   help='Explicit placements "model:lane:dist_m:paint,..." (overrides --num-cars).')
    g.add_argument("--num-cars", type=int, default=4)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--car-paints", default="silver,white,red,black")
    g.add_argument("--procedural-cars", action="store_true",
                   help="Force procedural cars even if fetched assets exist.")
    g.add_argument("--assets-dir", default="scenes/assets")
    g.add_argument("--signs", default="speed_limit:120,exit:300",
                   help='"type:distance_m,..." (types: speed_limit, exit).')
    g.add_argument("--spectral-signs", action="store_true",
                   help="Plain SPD sign panels (no RGB legend texture).")
    g.add_argument("--opaque-glass", action="store_true",
                   help="Dark coated-diffuse car glass instead of dielectric (less noise).")

    g = ap.add_argument_group("lighting")
    g.add_argument("--sun-elevation", type=float, default=45.0)
    g.add_argument("--sun-azimuth", type=float, default=135.0)
    g.add_argument("--turbidity", type=float, default=3.0)
    g.add_argument("--sky-albedo", type=float, default=0.15)
    g.add_argument("--sky-resolution", type=int, default=1024)
    g.add_argument("--no-sky", action="store_true", help="Distant sun only (no HW dome).")
    g.add_argument("--sun-sky-ratio", type=float, default=5.0,
                   help="Direct-horizontal : diffuse-horizontal illuminance ratio.")
    g.add_argument("--solar-spd", default=DEFAULT_SOLAR_SPD)
    g.add_argument("--illuminant", default=None,
                   help="Override the sun SPD CSV (e.g. D65 for comparison runs).")
    g.add_argument("--sun-scale", type=float, default=None,
                   help="Override computed distant-light scale.")
    g.add_argument("--visible-sun-disk", action="store_true",
                   help="Replace the delta distant light with an emissive disk of the "
                        "real 0.53 deg angular size (same SPD/irradiance): the sun "
                        "becomes visible in-frame (glare scenarios) and shadows get "
                        "the correct ~0.5 deg penumbra.")

    g = ap.add_argument_group("camera")
    g.add_argument("--camera", choices=("pinhole", "perspective", "thinlens", "realistic"),
                   default="realistic")
    g.add_argument("--lensfile", default="config/lenses/wide_22mm.dat")
    g.add_argument("--aperture-diameter-mm", type=float, default=8.0)
    g.add_argument("--focus-distance", type=float, default=20.0)
    g.add_argument("--fov", type=float, default=60.0)
    g.add_argument("--thinlens-lens-radius", type=float, default=0.004)
    g.add_argument("--cam-height", type=float, default=1.4)
    g.add_argument("--cam-lane", type=int, default=0,
                   help="Lane the camera occupies (0 = rightmost).")
    g.add_argument("--look-distance", type=float, default=60.0)

    g = ap.add_argument_group("film")
    g.add_argument("--film", choices=("rgb", "spectral"), default="spectral")
    g.add_argument("--xres", type=int, default=960)
    g.add_argument("--yres", type=int, default=640)
    g.add_argument("--pixelsamples", type=int, default=4096)
    g.add_argument("--spectral-nbuckets", type=int, default=64)
    g.add_argument("--spectral-lambda-min", type=float, default=360.0)
    g.add_argument("--spectral-lambda-max", type=float, default=830.0)
    g.add_argument("--film-output", default=None,
                   help="EXR path (default out/highway_<name>_spectral.exr).")
    g.add_argument("--step-nm", type=float, default=5.0)
    g.add_argument("--imgtool", default="third_party/pbrt-v4/build/imgtool")
    g.add_argument("--preset", choices=("smoke", "parse"), default=None,
                   help="smoke: 480x320/128spp/200m; parse: 160x107/4spp CPU sanity.")
    return ap.parse_args()


def _flag_given(name: str) -> bool:
    return any(a == name or a.startswith(name + "=") for a in sys.argv[1:])


def apply_preset(args: argparse.Namespace) -> None:
    """Presets supply defaults; explicitly-passed flags win."""
    if args.preset is None:
        return
    values = {
        "smoke": {"xres": 480, "yres": 320, "pixelsamples": 128,
                  "road_length": 200.0, "sky_resolution": 512},
        "parse": {"xres": 160, "yres": 107, "pixelsamples": 4,
                  "road_length": 200.0, "sky_resolution": 256},
    }[args.preset]
    for attr, val in values.items():
        if not _flag_given("--" + attr.replace("_", "-")):
            setattr(args, attr, val)
    args.opaque_glass = True


# ---------------------------------------------------------------------------
# Scene assembly
# ---------------------------------------------------------------------------

def spd_for_surface(repo: Path, spd_dir: Path, surface: str, step_nm: float) -> str:
    """Resample a committed surface CSV into the per-scene spd/ dir."""
    wl, val = read_csv_curve(repo / SURFACE_DIR / f"{surface}.csv")
    g, v = subsample_for_spd(wl, val, step_nm)
    name = f"{surface}.spd"
    write_spd(spd_dir / name, g, v)
    return f"spd/{name}"


def load_car_assets(repo: Path, assets_dir: Path, model: str) -> dict[str, Path] | None:
    """Return slot->PLY mapping for a fetched car model, or None if unavailable."""
    d = repo / assets_dir / "cars" / model
    if not (d / "meta.json").is_file():
        return None
    parts = {p.stem: p for p in d.glob("*.ply")}
    return parts if "paint" in parts else None


def car_object_blocks(models_used: dict[tuple[str, str], dict],
                      spd_paths: dict[str, str], scene_dir: Path, repo: Path,
                      opaque_glass: bool) -> list[str]:
    """ObjectBegin blocks, one per (model, paint) combination."""
    lines: list[str] = []
    glass_mat = (hm.coateddiffuse_spd(spd_paths["car_glass_tint"], roughness=0.08)
                 if opaque_glass else hm.dielectric_glass())
    slot_mats = {
        "glass": glass_mat,
        "tire": hm.diffuse_spd(spd_paths["rubber_tire"]),
        "trim": hm.conductor_metal("Ag", roughness=0.15),
    }
    for (model, paint), parts in sorted(models_used.items()):
        lines += [f'ObjectBegin "car_{model}_{paint}"']
        for slot in ("paint", "glass", "tire", "trim"):
            if slot not in parts:
                continue
            mat = (hm.coateddiffuse_spd(spd_paths[f"car_paint_{paint}"])
                   if slot == "paint" else slot_mats[slot])
            lines += ["    AttributeBegin"] + [f"    {m}" for m in mat]
            part = parts[slot]
            if isinstance(part, hl.Mesh):
                lines += hl.emit_trianglemesh(part, indent="        ")
            else:  # Path to PLY
                ply_rel = os.path.relpath(str(part), str(scene_dir))
                lines += [f'        Shape "plymesh" "string filename" ["{ply_rel}"]']
            lines += ["    AttributeEnd"]
        lines += ["ObjectEnd", ""]
    return lines


def main() -> None:
    args = parse_args()
    apply_preset(args)
    repo = args.repo_root.resolve()
    scene_dir = (repo / args.out_dir / args.name).resolve()
    spd_dir = scene_dir / "spd"
    scene_dir.mkdir(parents=True, exist_ok=True)

    film_output = args.film_output or f"out/highway_{args.name}_spectral.exr"
    spec = hl.RoadSpec(lanes=args.lanes, lane_width=args.lane_width, length=args.road_length)

    # --- SPDs -------------------------------------------------------------
    surfaces = ["asphalt_aged", "road_paint_white", "road_paint_yellow", "grass_green",
                "rubber_tire", "car_glass_tint", "car_paint_gray",
                "sign_sheeting_white", "sign_sheeting_green"]
    paints_pool = tuple(p.strip() for p in args.car_paints.split(",") if p.strip())
    for p in paints_pool:
        if f"car_paint_{p}" not in hm.SURFACE_CURVES:
            raise SystemExit(f"unknown car paint '{p}' (known: {hm.CAR_PAINTS})")
        surfaces.append(f"car_paint_{p}")
    spd_paths = {s: spd_for_surface(repo, spd_dir, s, args.step_nm) for s in set(surfaces)}

    # Sun SPD
    sun_csv = repo / (args.illuminant or args.solar_spd)
    wl_sun, v_sun = read_csv_curve(sun_csv)
    g, v = subsample_for_spd(wl_sun, v_sun, args.step_nm)
    write_spd(spd_dir / "sun.spd", g, v)

    # --- Sky --------------------------------------------------------------
    sun_dir = sun_dir_world(args.sun_elevation, args.sun_azimuth)
    sky_info: dict = {"enabled": not args.no_sky}
    sun_scale = args.sun_scale if args.sun_scale is not None else 1.0
    sky_rotate_deg = 0.0
    sky_exr = None
    if not args.no_sky:
        sky_raw = ensure_sky_exr(repo / args.imgtool, repo / args.assets_dir / "sky",
                                 args.sun_elevation, args.turbidity,
                                 args.sky_albedo, args.sky_resolution)
        # Detect the baked sun BEFORE the disk is excised, then use the
        # diffuse-only dome (sun disk removed, lower hemisphere horizon-filled).
        baked = detect_baked_sun_dir(sky_raw)
        sky_exr = preprocess_sky_exr(sky_raw, sky_raw.with_name(sky_raw.stem + "_diffuse.exr"))
        # Invert sun_dir_world's azimuth convention: x = -sin(az), z = cos(az).
        baked_az = float(np.rad2deg(np.arctan2(-baked[0], baked[2])))
        # pbrt 'Rotate theta 0 1 0' maps azimuth az -> az - theta, so to move
        # the baked sun onto the requested azimuth: theta = baked - target.
        sky_rotate_deg = baked_az - args.sun_azimuth
        e_sky = hemisphere_illuminance_rel(sky_exr)
        # pbrt upsamples the dome's RGB texels to illuminant spectra; the
        # photopic lux of the upsampled dome per unit Rec.709 map luminance
        # (as integrated by hemisphere_illuminance_rel) differs from the
        # photopic lux of our SPD-based sun by this constant. Measured
        # empirically against pbrt renders (sun-only vs sky-only road
        # radiance): 72.2k at el=10/t=5, 84.5k at el=35/t=3 -> 78k, +/-8%
        # residual chromaticity dependence. Re-derive with the kver recipe in
        # docs/guides/highway_scenes.md if pbrt's upsampling ever changes.
        SKY_DOME_PHOTOMETRIC_K = 78000.0
        # Distant-light horizontal illuminance for scale=1: photopic lux of the
        # SPD x sin(elevation). Only the sun:sky ratio matters here because the
        # electrons stage re-normalizes the absolute level.
        e_dir_unit = float(illuminance_lux_from_irradiance(g, v)
                           * np.sin(np.deg2rad(args.sun_elevation)))
        if args.sun_scale is None and e_dir_unit > 0 and e_sky > 0:
            sun_scale = args.sun_sky_ratio * e_sky * SKY_DOME_PHOTOMETRIC_K / e_dir_unit
        sky_info.update({
            "exr": _rel(repo, sky_exr), "exr_raw": _rel(repo, sky_raw),
            "sun_disk": "excised", "turbidity": args.turbidity,
            "albedo": args.sky_albedo, "resolution": args.sky_resolution,
            "baked_sun_azimuth_deg": baked_az, "rotate_deg": sky_rotate_deg,
            "E_sky_horizontal_rel": e_sky, "E_direct_horizontal_rel": e_dir_unit * sun_scale,
        })

    # --- Cars ---------------------------------------------------------------
    if args.cars is not None:  # "" means an empty road
        placements = hl.parse_car_spec(args.cars)
    else:
        rng = np.random.default_rng(args.seed)
        placements = hl.place_cars_random(spec, args.num_cars, rng, paints=paints_pool)

    models_used: dict[tuple[str, str], dict] = {}
    car_meta = []
    for p in placements:
        key = (p.model, p.paint)
        if key not in models_used:
            parts = None if args.procedural_cars else load_car_assets(
                repo, Path(args.assets_dir), p.model)
            source = "plymesh_assets" if parts else "procedural"
            models_used[key] = parts if parts else hl.procedural_car_meshes()
            car_meta.append({"model": p.model, "paint": p.paint, "source": source})
        if f"car_paint_{p.paint}" not in spd_paths:
            spd_paths[f"car_paint_{p.paint}"] = spd_for_surface(
                repo, spd_dir, f"car_paint_{p.paint}", args.step_nm)

    # --- Signs --------------------------------------------------------------
    signs = []
    default_text = {"speed_limit": "65", "exit": "12"}
    for item in (args.signs or "").split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) not in (2, 3):
            raise SystemExit(f"bad sign spec '{item}' (want type:distance[:text])")
        stype = parts[0].strip()
        signs.append((stype, float(parts[1]),
                      parts[2] if len(parts) == 3 else default_text.get(stype, "")))

    # --- Camera -------------------------------------------------------------
    cam_x = spec.lane_center(args.cam_lane)
    eye = (cam_x, args.cam_height, 0.0)
    target = (cam_x, 0.9, -args.look_distance)

    # --- PBRT text ----------------------------------------------------------
    L: list[str] = [
        f"# Highway scene '{args.name}' generated by tools/build_highway_scene.py",
        f"# lanes={args.lanes} length={args.road_length}m cars={len(placements)} "
        f"sun=({args.sun_elevation},{args.sun_azimuth})deg turbidity={args.turbidity}",
        'Option "disablepixeljitter" false',
        f'Option "seed" {args.seed}',
        "",
        'ColorSpace "srgb"',
        "",
        'Sampler "zsobol" "integer pixelsamples" [%d]' % args.pixelsamples,
        'Integrator "path" "integer maxdepth" [6]',
        'PixelFilter "gaussian"',
        "",
    ]
    if args.film == "spectral":
        if not film_output.lower().endswith(".exr"):
            raise SystemExit("spectral film requires .exr output")
        L += [
            'Film "spectral"',
            f'    "string filename" ["{film_output}"]',
            f'    "integer xresolution" [{args.xres}]',
            f'    "integer yresolution" [{args.yres}]',
            '    "bool savefp16" false',
            f'    "integer nbuckets" [{args.spectral_nbuckets}]',
            f'    "float lambdamin" [{args.spectral_lambda_min:.6g}]',
            f'    "float lambdamax" [{args.spectral_lambda_max:.6g}]',
        ]
    else:
        L += [
            'Film "rgb"',
            f'    "string filename" ["{film_output}"]',
            f'    "integer xresolution" [{args.xres}]',
            f'    "integer yresolution" [{args.yres}]',
        ]
    L += [
        "",
        "LookAt",
        "    %.6g %.6g %.6g" % eye,
        "    %.6g %.6g %.6g" % target,
        "    0 1 0",
    ]
    camera_kind = "pinhole" if args.camera == "perspective" else args.camera
    if camera_kind == "pinhole":
        L += ['Camera "perspective" "float fov" [%g]' % args.fov]
    elif camera_kind == "thinlens":
        L += [
            'Camera "perspective"',
            f'    "float fov" [{args.fov:.6g}]',
            f'    "float lensradius" [{args.thinlens_lens_radius:.6g}]',
            f'    "float focaldistance" [{args.focus_distance:.6g}]',
        ]
    else:
        lens_repo = (repo / args.lensfile).resolve()
        if not lens_repo.is_file():
            raise SystemExit(f"realistic camera: lens file not found: {lens_repo}")
        lens_rel = os.path.relpath(str(lens_repo), str(scene_dir))
        L += [
            'Camera "realistic"',
            f'    "string lensfile" ["{lens_rel}"]',
            f'    "float aperturediameter" [{args.aperture_diameter_mm:.6g}]',
            f'    "float focusdistance" [{args.focus_distance:.6g}]',
        ]

    L += ["", "WorldBegin", ""]

    # Sun: delta distant light by default; --visible-sun-disk swaps in an
    # emissive disk of the true angular size delivering the same irradiance.
    SUN_ANGULAR_RADIUS_DEG = 0.2665
    if args.visible_sun_disk:
        dist = 1000.0
        radius = dist * np.tan(np.deg2rad(SUN_ANGULAR_RADIUS_DEG))
        omega = np.pi * np.sin(np.deg2rad(SUN_ANGULAR_RADIUS_DEG)) ** 2
        # Distant light irradiance E = scale * SPD; disk of solid angle omega
        # needs radiance scale/omega for the same E on a sun-facing surface.
        disk_scale = sun_scale / omega
        pos = sun_dir * dist
        # Rotate the disk's +Z normal onto -sun_dir (facing the scene).
        n = -sun_dir
        axis = np.cross([0.0, 0.0, 1.0], n)
        s = np.linalg.norm(axis)
        angle = float(np.rad2deg(np.arctan2(s, n[2])))
        rot = (f"    Rotate {angle:.4f} {axis[0]/s:.6f} {axis[1]/s:.6f} {axis[2]/s:.6f}"
               if s > 1e-9 else "    # disk already faces -Z" if n[2] > 0
               else "    Rotate 180 1 0 0")
        L += [
            f"# Visible spectral sun disk ({2*SUN_ANGULAR_RADIUS_DEG:.3g} deg diameter, "
            f"same irradiance as the distant light it replaces)",
            "AttributeBegin",
            "    Translate %.4f %.4f %.4f" % tuple(pos),
            rot,
            '    AreaLightSource "diffuse"',
            '        "spectrum L" "spd/sun.spd"',
            f'        "float scale" [{disk_scale:.6g}]',
            '        "bool twosided" true',
            f'    Shape "disk" "float radius" [{radius:.4f}]',
            "AttributeEnd",
            "",
        ]
    else:
        sun_from = sun_dir * 1000.0
        L += [
            "# Spectral distant sun",
            'LightSource "distant"',
            '    "spectrum L" "spd/sun.spd"',
            f'    "float scale" [{sun_scale:.6g}]',
            '    "point3 from" [%.4f %.4f %.4f]' % tuple(sun_from),
            '    "point3 to" [0 0 0]',
            "",
        ]
    # Sky dome
    if sky_exr is not None:
        sky_rel = os.path.relpath(str(sky_exr), str(scene_dir))
        L += [
            "# Hosek-Wilkie sky dome (RGB, upsampled to spectra by pbrt)",
            "AttributeBegin",
            f"    Rotate {sky_rotate_deg:.4f} 0 1 0",
            "    Rotate -90 1 0 0   # makesky is z-up; scene is y-up",
            f'    LightSource "infinite" "string filename" ["{sky_rel}"]',
            "AttributeEnd",
            "",
        ]

    # Static geometry
    def attr(mat_lines: list[str], mesh: hl.Mesh, comment: str) -> None:
        L.append(f"# {comment}")
        L.append("AttributeBegin")
        L.extend(f"    {m}" for m in mat_lines)
        L.extend(hl.emit_trianglemesh(mesh))
        L.append("AttributeEnd")
        L.append("")

    attr(hm.diffuse_spd(spd_paths["grass_green"]), hl.terrain_plane(spec), "Terrain")
    attr(hm.diffuse_spd(spd_paths["asphalt_aged"]), hl.road_slab(spec), "Road slab")
    markings = hl.lane_marking_meshes(spec)
    attr(hm.diffuse_spd(spd_paths["road_paint_white"]), markings["white"], "White markings")
    attr(hm.diffuse_spd(spd_paths["road_paint_yellow"]), markings["yellow"], "Yellow edge line")

    if args.guardrails:
        rail, posts = hl.guardrail_meshes(spec)
        attr(hm.conductor_metal("Al", roughness=0.25), rail, "Guard rails (W-beam)")
        L += ['ObjectBegin "guardrail_post"', "    AttributeBegin"]
        L += [f"    {m}" for m in hm.conductor_metal("Al", roughness=0.35)]
        L += hl.emit_trianglemesh(hl.box((0, 0, 0), hl.GUARDRAIL_POST_SIZE), indent="        ")
        L += ["    AttributeEnd", "ObjectEnd", ""]
        for t in posts:
            L += ["AttributeBegin",
                  "    Translate %.4f %.4f %.4f" % tuple(t),
                  '    ObjectInstance "guardrail_post"',
                  "AttributeEnd"]
        L.append("")

    # Signs: textured legend faces by default; --spectral-signs uses plain SPD
    # panels (the legend texture is the only RGB-defined surface in the scene).
    sign_meta = []
    for si, (stype, dist, text) in enumerate(signs):
        face_spd = spd_paths["sign_sheeting_white" if stype == "speed_limit"
                             else "sign_sheeting_green"]
        if not args.spectral_signs:
            png = scene_dir / "textures" / f"sign{si}_{stype}.png"
            hm.render_sign_face_png(stype, text, png)
            tex_name = f"sign{si}_tex"
            L += [f'Texture "{tex_name}" "spectrum" "imagemap"',
                  f'    "string filename" ["textures/{png.name}"]', ""]
        for slot, mesh in hl.sign_meshes(stype, dist, spec):
            if slot == "face":
                mat = (hm.diffuse_spd(face_spd) if args.spectral_signs
                       else hm.sign_face_textured(tex_name))
                attr(mat, mesh, f"Sign {stype}@{dist:g}m face")
            elif slot == "back":
                attr(hm.diffuse_spd(spd_paths["car_paint_gray"]), mesh,
                     f"Sign {stype}@{dist:g}m back")
            else:
                attr(hm.conductor_metal("Al", roughness=0.35), mesh,
                     f"Sign {stype}@{dist:g}m post")
        sign_meta.append({"type": stype, "distance_m": dist, "text": text,
                          "legend": ("spectral_panel" if args.spectral_signs
                                     else "rgb_texture"),
                          "retroreflection": "approximated_diffuse"})

    # Cars
    L += car_object_blocks(models_used, spd_paths, scene_dir, repo, args.opaque_glass)
    cars_meta = []
    for p in placements:
        tx, ty, tz, yaw = hl.car_world_transform(p, spec)
        L += ["AttributeBegin",
              "    Translate %.4f %.4f %.4f" % (tx, ty, tz)]
        if yaw:
            L += ["    Rotate %.4f 0 1 0" % yaw]
        L += [f'    ObjectInstance "car_{p.model}_{p.paint}"', "AttributeEnd"]
        cars_meta.append({"model": p.model, "lane": p.lane, "s_m": p.s_m,
                          "lateral_offset_m": p.lateral_offset_m, "paint": p.paint,
                          "heading_deg": p.heading_deg,
                          "source": next(c["source"] for c in car_meta
                                         if c["model"] == p.model and c["paint"] == p.paint)})
    L.append("")

    scene_path = scene_dir / f"highway_{args.name}.pbrt"
    scene_path.write_text("\n".join(L) + "\n")

    # --- Manifest -----------------------------------------------------------
    manifest = {
        "scene": _rel(repo, scene_path),
        "generator": "tools/build_highway_scene.py",
        "seed": args.seed,
        "film": {
            "type": args.film, "filename": film_output,
            "xresolution": args.xres, "yresolution": args.yres,
            **({"nbuckets": args.spectral_nbuckets,
                "lambda_min_nm": args.spectral_lambda_min,
                "lambda_max_nm": args.spectral_lambda_max} if args.film == "spectral" else {}),
        },
        "camera": {
            "type": camera_kind,
            "lookat": {"eye": list(eye), "target": list(target), "up": [0, 1, 0]},
            "fov_deg": None if camera_kind == "realistic" else args.fov,
            **({"lensfile": args.lensfile,
                "aperture_diameter_mm": args.aperture_diameter_mm,
                "focus_distance": args.focus_distance} if camera_kind == "realistic" else {}),
            "cam_height_m": args.cam_height, "cam_lane": args.cam_lane,
            "look_distance_m": args.look_distance,
        },
        "lighting": {
            "sun": {"elevation_deg": args.sun_elevation, "azimuth_deg": args.sun_azimuth,
                    "spd_csv": _rel(repo, sun_csv), "spd": "spd/sun.spd",
                    "scale": sun_scale,
                    "mode": "visible_disk" if args.visible_sun_disk else "distant"},
            "sky": sky_info,
            "sun_sky_ratio": args.sun_sky_ratio,
            "calibration_recommendation": {
                "mode": "mean_photopic_lux",
                "note": ("3D scene: use model.pbrt_spectral_exr.radiometric_autocalibration; "
                         "target is mean sensor-plane photopic lux (E = pi*L/(4 N^2) tau)"),
                "suggested_target_lux": 500,
            },
        },
        "road": {"lanes": args.lanes, "lane_width_m": args.lane_width,
                 "length_m": args.road_length,
                 "shoulders_m": {"right": spec.shoulder_right, "median": spec.shoulder_median},
                 "markings": {"dash_m": hl.DASH_LENGTH_M, "gap_m": hl.DASH_GAP_M,
                              "width_m": hl.MARKING_WIDTH_M},
                 "guardrails": args.guardrails},
        "objects": {"cars": cars_meta, "signs": sign_meta},
        "spd_step_nm": args.step_nm,
        "cli_args": sys.argv[1:],
    }
    manifest_path = scene_dir / f"highway_{args.name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {_rel(repo, scene_path)}")
    print(f"wrote {_rel(repo, manifest_path)}")
    print(f"  cars={len(placements)} signs={len(signs)} sun_scale={sun_scale:.4g} "
          f"sky={'off' if args.no_sky else _rel(repo, sky_exr)}")


if __name__ == "__main__":
    main()
