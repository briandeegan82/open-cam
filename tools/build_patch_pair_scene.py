#!/usr/bin/env python3
"""Render two ColorChecker patches side by side under an angled light.

Unlike the flat-field single-patch scenes, this places a *point* (or spot) light off to
one side at a chosen angle so the target is physically unevenly illuminated: inverse-square
falloff plus the incidence-angle cosine give a real brightness gradient across the pair.

Note: a pbrt "distant" light is a parallel-ray source and illuminates a flat frontal target
uniformly (no gradient) — use "point" or "spot" to actually get uneven illumination.

Reuses the spectral-reflectance SPD machinery from build_colorchecker_scene.py.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_colorchecker_scene import (  # noqa: E402
    load_csv_spectrum, resample_clip, subsample_for_spd, write_spd,
    patch_paths, resolve_lensfile,
)

DEFAULT_REALISTIC_LENSFILE = "config/lenses/wide_22mm.dat"


def _camera_block(args, out_dir: Path, repo: Path) -> list[str]:
    focus_d = float(args.focus_distance) if args.focus_distance is not None else float(args.cam_dist)
    if args.camera in ("perspective", "pinhole"):
        return [f'Camera "perspective" "float fov" [{float(args.fov):.6g}]']
    lens_repo = resolve_lensfile(repo, args.lensfile)
    if not lens_repo.is_file():
        raise FileNotFoundError(f"lens file not found: {lens_repo}")
    lens_for_scene = os.path.relpath(str(lens_repo), str(out_dir.resolve()))
    return [
        'Camera "realistic"',
        f'    "string lensfile" ["{lens_for_scene}"]',
        f'    "float aperturediameter" [{float(args.aperture_diameter_mm):.6g}]',
        f'    "float focusdistance" [{focus_d:.6g}]',
    ]


def _light_position(args) -> tuple[float, float, float]:
    """Light position at ``light_angle_deg`` from the target normal (+Z), offset toward
    ``light_direction``.  Image orientation (camera up = +Y, LookAt mirrors X):
      north -> +Y (top),  south -> -Y (bottom),  west -> +X (image left),  east -> -X (right).
    """
    theta = math.radians(float(args.light_angle_deg))
    d = float(args.light_distance)
    horiz, depth = d * math.sin(theta), d * math.cos(theta)
    offsets = {
        "north": (0.0, horiz, depth),
        "south": (0.0, -horiz, depth),
        "west": (horiz, 0.0, depth),
        "east": (-horiz, 0.0, depth),
    }
    return offsets[args.light_direction]


def _light_block(args, ill_spd_name: str) -> list[str]:
    """Angled light creating an illumination gradient across the target."""
    fx, fy, fz = _light_position(args)
    if args.light_type == "distant":
        return [
            "# Distant (parallel-ray) light — uniform on a flat frontal target (no gradient).",
            'LightSource "distant"',
            f'    "spectrum L" "spd/{ill_spd_name}"',
            f'    "float scale" [{float(args.light_scale):.6g}]',
            f'    "point3 from" [{fx:.4g} {fy:.4g} {fz:.4g}]',
            '    "point3 to" [0 0 0]',
        ]
    if args.light_type == "spot":
        return [
            f"# Spot light at {args.light_angle_deg:g} deg, aimed at target centre.",
            'LightSource "spot"',
            f'    "spectrum I" "spd/{ill_spd_name}"',
            f'    "float scale" [{float(args.light_scale):.6g}]',
            f'    "point3 from" [{fx:.4g} {fy:.4g} {fz:.4g}]',
            '    "point3 to" [0 0 0]',
            f'    "float coneangle" [{float(args.spot_cone_deg):.6g}]',
            f'    "float conedeltaangle" [{float(args.spot_delta_deg):.6g}]',
        ]
    return [
        f"# Point light at {args.light_angle_deg:g} deg from normal — inverse-square gradient.",
        'LightSource "point"',
        f'    "spectrum I" "spd/{ill_spd_name}"',
        f'    "float scale" [{float(args.light_scale):.6g}]',
        f'    "point3 from" [{fx:.4g} {fy:.4g} {fz:.4g}]',
    ]


def _ambient_block(args, ill_spd_name: str) -> list[str]:
    """Uniform ambient diffuse fill (infinite/environment light).  Adds a flat
    illumination floor so shadowed regions of the gradient are not near-black."""
    if float(args.ambient_scale) <= 0.0:
        return []
    return [
        f"# Ambient diffuse fill (uniform infinite light), scale {args.ambient_scale:g}.",
        'LightSource "infinite"',
        f'    "spectrum L" "spd/{ill_spd_name}"',
        f'    "float scale" [{float(args.ambient_scale):.6g}]',
    ]


def _patch_quad(spd_ref: str, xl: float, xr: float, yb: float, yt: float) -> list[str]:
    # Mirror X to match pbrt LookAt (camera +X is world -X), like build_colorchecker_scene.
    return [
        "AttributeBegin",
        f'    Material "diffuse" "spectrum reflectance" "{spd_ref}"',
        '    Shape "bilinearmesh"',
        f'        "point3 P" [ {-xr:.6g} {yb:.6g} 0.001   {-xl:.6g} {yb:.6g} 0.001   '
        f'{-xr:.6g} {yt:.6g} 0.001   {-xl:.6g} {yt:.6g} 0.001 ]',
        '        "point2 uv" [ 0 0   1 0   0 1   1 1 ]',
        "AttributeEnd",
        "",
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    repo_default = Path(__file__).resolve().parent.parent
    ap.add_argument("--repo-root", type=Path, default=repo_default)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--left-patch", type=int, default=3, help="Left patch index (1..24).")
    ap.add_argument("--right-patch", type=int, default=8, help="Right patch index (1..24).")
    ap.add_argument("--illuminant", type=Path, default=None,
                    help="Illuminant CSV (default D65).")
    ap.add_argument("--film-output", type=str, default="out/patch_pair.exr")
    # Light
    ap.add_argument("--light-type", choices=("point", "spot", "distant"), default="point")
    ap.add_argument("--light-direction", choices=("north", "south", "east", "west"),
                    default="north", help="Side the light sits on (north=above the pair).")
    ap.add_argument("--light-angle-deg", type=float, default=45.0)
    ap.add_argument("--light-distance", type=float, default=1.8)  # locked: 59% falloff, mirror-symmetric
    ap.add_argument("--light-scale", type=float, default=40.0)
    ap.add_argument("--ambient-scale", type=float, default=4.0,  # locked: ~27% falloff with fill
                    help="Uniform ambient diffuse fill (infinite-light scale). 0 = off.")
    ap.add_argument("--spot-cone-deg", type=float, default=35.0)
    ap.add_argument("--spot-delta-deg", type=float, default=8.0)
    # Geometry
    ap.add_argument("--patch-size", type=float, default=1.3)
    ap.add_argument("--gap", type=float, default=0.12)
    ap.add_argument("--cam-dist", type=float, default=3.75)
    ap.add_argument("--camera", choices=("perspective", "pinhole", "realistic"),
                    default="perspective")
    ap.add_argument("--fov", type=float, default=42.0)
    ap.add_argument("--lensfile", type=str, default=DEFAULT_REALISTIC_LENSFILE)
    ap.add_argument("--aperture-diameter-mm", type=float, default=8.756)
    ap.add_argument("--focus-distance", type=float, default=None)
    # Film
    ap.add_argument("--film", choices=("rgb", "spectral"), default="rgb")
    ap.add_argument("--xres", type=int, default=960)
    ap.add_argument("--yres", type=int, default=640)
    ap.add_argument("--pixelsamples", type=int, default=128)
    ap.add_argument("--step-nm", type=float, default=5.0)
    ap.add_argument("--spectral-nbuckets", type=int, default=32)
    ap.add_argument("--spectral-lambda-min", type=float, default=360.0)
    ap.add_argument("--spectral-lambda-max", type=float, default=830.0)
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    out_dir = (args.out_dir or (repo / "scenes" / "generated")).resolve()
    spd_dir = out_dir / "spd"
    spd_dir.mkdir(parents=True, exist_ok=True)
    (repo / "out").mkdir(parents=True, exist_ok=True)

    ill_path = (args.illuminant or (repo / "spectra/illuminant/interpolated/D65.csv")).resolve()
    xrite_dir = repo / "spectra" / "xrite"
    wl_lo, wl_hi = 360.0, 830.0

    # Write illuminant + the two patch SPDs.
    grid_hi = np.arange(wl_lo, wl_hi + 1e-6, 1.0)
    iwl, ival = load_csv_spectrum(ill_path)
    iw, iv = subsample_for_spd(*resample_clip(iwl, ival, wl_lo, wl_hi, 1.0), args.step_nm)
    ill_spd_name = f"illuminant_{ill_path.stem}.spd"
    write_spd(spd_dir / ill_spd_name, iw, iv)

    files = patch_paths(xrite_dir)
    refs = {}
    for side, idx in (("left", args.left_patch), ("right", args.right_patch)):
        if not (1 <= idx <= 24):
            raise ValueError(f"{side} patch index must be 1..24, got {idx}")
        rwl, rval = load_csv_spectrum(files[idx - 1])
        r_hi = np.interp(grid_hi, rwl, rval, left=rval[0], right=rval[-1])
        sw, sv = subsample_for_spd(grid_hi, r_hi, args.step_nm)
        name = f"patch_{idx:02d}.spd"
        write_spd(spd_dir / name, sw, sv)
        refs[side] = (idx, f"spd/{name}", files[idx - 1].stem)

    # Film block
    film_out = args.film_output
    if args.film == "spectral":
        film_block = [
            'Film "spectral"',
            f'    "string filename" ["{film_out}"]',
            f'    "integer xresolution" [{int(args.xres)}]',
            f'    "integer yresolution" [{int(args.yres)}]',
            '    "bool savefp16" false',
            f'    "integer nbuckets" [{int(args.spectral_nbuckets)}]',
            f'    "float lambdamin" [{float(args.spectral_lambda_min):.6g}]',
            f'    "float lambdamax" [{float(args.spectral_lambda_max):.6g}]',
        ]
    else:
        film_block = [
            'Film "rgb"',
            f'    "string filename" ["{film_out}"]',
            f'    "integer xresolution" [{int(args.xres)}]',
            f'    "integer yresolution" [{int(args.yres)}]',
            '    "bool savefp16" false',
        ]

    ps = float(args.patch_size)
    g = float(args.gap)
    # Two patches centred about x=0: left occupies [-(ps+g/2), -g/2], right [g/2, ps+g/2].
    left_xl, left_xr = -(ps + g / 2.0), -g / 2.0
    right_xl, right_xr = g / 2.0, ps + g / 2.0
    yb, yt = -ps / 2.0, ps / 2.0

    lines: list[str] = [
        "# Generated by tools/build_patch_pair_scene.py — do not hand-edit.",
        f"# Patches: left={refs['left'][2]} right={refs['right'][2]} | "
        f"light={args.light_type} {args.light_direction}@{args.light_angle_deg:g}deg "
        f"scale={args.light_scale:g}",
        'Option "seed" 0',
        "",
        'ColorSpace "srgb"',
        f'Sampler "zsobol" "integer pixelsamples" [{int(args.pixelsamples)}]',
        'Integrator "path" "integer maxdepth" [6]',
        'PixelFilter "gaussian"',
        "",
        *film_block,
        "",
        "LookAt",
        f"0 0 {float(args.cam_dist):.6f}",
        "    0 0 0",
        "    0 1 0",
        *_camera_block(args, out_dir, repo),
        "",
        "WorldBegin",
        "",
        *_light_block(args, ill_spd_name),
        "",
        *_ambient_block(args, ill_spd_name),
        "",
        "# Dark surround so the illumination gradient is read off the patches only.",
        "AttributeBegin",
        '    Material "diffuse" "rgb reflectance" [0.04 0.04 0.04]',
        '    Shape "bilinearmesh"',
        '        "point3 P" [ -6 -4.5 -0.05   6 -4.5 -0.05   -6 4.5 -0.05   6 4.5 -0.05 ]',
        '        "point2 uv" [ 0 0   1 0   0 1   1 1 ]',
        "AttributeEnd",
        "",
        *_patch_quad(refs["left"][1], left_xl, left_xr, yb, yt),
        *_patch_quad(refs["right"][1], right_xl, right_xr, yb, yt),
    ]

    scene_path = out_dir / f"patch_pair_{refs['left'][0]:02d}_{refs['right'][0]:02d}.pbrt"
    scene_path.write_text("\n".join(lines) + "\n")
    print(f"left  patch {refs['left'][0]:02d} ({refs['left'][2]})")
    print(f"right patch {refs['right'][0]:02d} ({refs['right'][2]})")
    print(f"light: {args.light_type} {args.light_direction} @ {args.light_angle_deg:g} deg, "
          f"dist {args.light_distance:g}, scale {args.light_scale:g}")
    print(f"Wrote scene: {scene_path}")
    print(f"Film output: {film_out}")


if __name__ == "__main__":
    main()
