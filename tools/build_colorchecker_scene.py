#!/usr/bin/env python3
"""Generate a pbrt-v4 scene for the 24-patch ColorChecker (measured reflectance + D55).

Patch quads are mirrored in world X so patch 01 matches the usual top-left X-Rite layout:
pbrt LookAt uses right = cross(up, view), so camera +X is world -X for the default camera.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np


def _rel(repo: Path, p: Path) -> str:
    try:
        return str(p.relative_to(repo))
    except ValueError:
        return str(p)


def load_csv_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    wl: list[float] = []
    val: list[float] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r",\s*", line, maxsplit=1)
        if len(parts) != 2:
            continue
        wl.append(float(parts[0]))
        val.append(float(parts[1]))
    if not wl:
        raise ValueError(f"no spectral data in {path}")
    w = np.asarray(wl, dtype=np.float64)
    v = np.asarray(val, dtype=np.float64)
    order = np.argsort(w)
    return w[order], v[order]


def resample_clip(
    wl: np.ndarray,
    val: np.ndarray,
    lam_min: float,
    lam_max: float,
    step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform grid in [lam_min, lam_max] with linear interpolation; flat extrapolation."""
    grid = np.arange(lam_min, lam_max + 1e-9, step, dtype=np.float64)
    v = np.interp(grid, wl, val, left=val[0], right=val[-1])
    return grid, v


def subsample_for_spd(wl: np.ndarray, val: np.ndarray, step_nm: float) -> tuple[np.ndarray, np.ndarray]:
    wmin, wmax = float(wl[0]), float(wl[-1])
    grid = np.arange(wmin, wmax + 1e-9, step_nm, dtype=np.float64)
    if grid[-1] < wmax - 1e-6:
        grid = np.append(grid, wmax)
    v = np.interp(grid, wl, val)
    return grid, v


def write_spd(path: Path, wl: np.ndarray, val: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"{float(w):.6g}\t{float(v):.12g}" for w, v in zip(wl, val))
    path.write_text("# wavelength_nm\tvalue\n" + body + "\n")


def patch_paths(xrite_dir: Path) -> list[Path]:
    files = sorted(xrite_dir.glob("*.csv"), key=lambda p: p.name)
    out: list[Path] = []
    for p in files:
        m = re.match(r"^(\d+)_", p.name)
        if not m:
            continue
        out.append((int(m.group(1)), p))
    out.sort(key=lambda t: t[0])
    paths = [p for _, p in out]
    if len(paths) != 24:
        raise RuntimeError(f"expected 24 xrite CSVs, found {len(paths)} in {xrite_dir}")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--repo-root", type=Path, default=root)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Generated scene directory (default: <repo>/scenes/generated)",
    )
    ap.add_argument(
        "--illuminant",
        type=Path,
        default=None,
        help="Illuminant CSV (default: spectra/illuminant/interpolated/D55.csv)",
    )
    ap.add_argument("--step-nm", type=float, default=5.0, help="SPD file wavelength step (nm)")
    ap.add_argument("--lambda-min", type=float, default=360.0)
    ap.add_argument("--lambda-max", type=float, default=830.0)
    ap.add_argument("--patch-width", type=float, default=0.45)
    ap.add_argument("--patch-height", type=float, default=0.45)
    ap.add_argument("--gap", type=float, default=0.04)
    ap.add_argument("--light-scale", type=float, default=2.0)
    ap.add_argument("--cam-dist", type=float, default=4.25)
    ap.add_argument(
        "--camera",
        choices=("perspective", "pinhole", "thinlens", "realistic"),
        default="perspective",
        help=(
            'PBRT camera: "pinhole"/"perspective" (lensradius=0), '
            '"thinlens" (perspective + lensradius/focaldistance), '
            'or "realistic" (traced lens; needs --lensfile).'
        ),
    )
    ap.add_argument(
        "--lensfile",
        type=str,
        default="scenes/lenses/wide_22mm.dat",
        help="Repo-relative lens description for RealisticCamera (pbrt ReadFloatFile format).",
    )
    ap.add_argument(
        "--aperture-diameter-mm",
        type=float,
        default=4.0,
        help="RealisticCamera aperture diameter [mm] (clamped by lens prescription).",
    )
    ap.add_argument(
        "--focus-distance",
        type=float,
        default=None,
        help="RealisticCamera focus distance in scene units (default: same as --cam-dist).",
    )
    ap.add_argument("--fov", type=float, default=35.0)
    ap.add_argument(
        "--thinlens-lens-radius",
        type=float,
        default=0.0,
        help="Thin-lens lens radius in scene units (PerspectiveCamera lensradius).",
    )
    ap.add_argument(
        "--thinlens-focal-distance",
        type=float,
        default=None,
        help="Thin-lens focal distance in scene units (default: --cam-dist).",
    )
    ap.add_argument("--xres", type=int, default=960)
    ap.add_argument("--yres", type=int, default=640)
    ap.add_argument("--pixelsamples", type=int, default=64)
    ap.add_argument(
        "--film",
        choices=("rgb", "spectral"),
        default="rgb",
        help='pbrt-v4 Film type: "rgb" (default) or "spectral" (OpenEXR with RGB + wavelength buckets).',
    )
    ap.add_argument(
        "--film-output",
        type=str,
        default=None,
        help='EXR path relative to repo root (default: out/colorchecker.exr or out/colorchecker_spectral.exr).',
    )
    ap.add_argument(
        "--spectral-nbuckets",
        type=int,
        default=16,
        help='SpectralFilm only: number of wavelength buckets (pbrt "nbuckets").',
    )
    ap.add_argument(
        "--spectral-lambda-min",
        type=float,
        default=360.0,
        help='SpectralFilm only: lambdamin (nm), must be >= 360 (pbrt Lambda_min).',
    )
    ap.add_argument(
        "--spectral-lambda-max",
        type=float,
        default=830.0,
        help='SpectralFilm only: lambdamax (nm), must be <= 830 (pbrt Lambda_max).',
    )
    args = ap.parse_args()

    repo: Path = args.repo_root
    out_dir = args.out_dir or (repo / "scenes" / "generated")
    spd_dir = out_dir / "spd"
    ill_path = args.illuminant or (repo / "spectra" / "illuminant" / "interpolated" / "D55.csv")
    xrite_dir = repo / "spectra" / "xrite"

    out_dir.mkdir(parents=True, exist_ok=True)
    (repo / "out").mkdir(parents=True, exist_ok=True)

    wl_lo, wl_hi = args.lambda_min, args.lambda_max
    grid_hi = np.arange(wl_lo, wl_hi + 1e-9, 1.0, dtype=np.float64)

    ill_wl, ill_val = load_csv_spectrum(ill_path)
    ill_hi = np.interp(grid_hi, ill_wl, ill_val, left=ill_val[0], right=ill_val[-1])

    ill_spd_wl, ill_spd_val = subsample_for_spd(*resample_clip(ill_wl, ill_val, wl_lo, wl_hi, 1.0), args.step_nm)
    write_spd(spd_dir / "illuminant_D55.spd", ill_spd_wl, ill_spd_val)

    patches_meta: list[dict] = []
    patch_files = patch_paths(xrite_dir)

    for idx, pth in enumerate(patch_files, start=1):
        rwl, rval = load_csv_spectrum(pth)
        r_hi = np.interp(grid_hi, rwl, rval, left=rval[0], right=rval[-1])
        swl, sval = subsample_for_spd(grid_hi, r_hi, args.step_nm)
        spd_name = f"patch_{idx:02d}.spd"
        write_spd(spd_dir / spd_name, swl, sval)
        patches_meta.append(
            {
                "index": idx,
                "file": pth.name,
                "spd": f"spd/{spd_name}",
            }
        )

    pw, ph, g = args.patch_width, args.patch_height, args.gap
    board_w = 6 * pw + 5 * g
    board_h = 4 * ph + 3 * g
    x0 = -board_w / 2.0
    y_top = board_h / 2.0

    film_type = str(args.film).lower()
    if film_type == "rgb":
        film_filename = args.film_output or "out/colorchecker.exr"
    else:
        film_filename = args.film_output or "out/colorchecker_spectral.exr"
        if not str(film_filename).lower().endswith(".exr"):
            raise ValueError(f'SpectralFilm requires an .exr filename, got "{film_filename}"')
        lo, hi = float(args.spectral_lambda_min), float(args.spectral_lambda_max)
        if lo < 360.0 or hi > 830.0 or lo >= hi:
            raise ValueError(
                "spectral lambda range must satisfy 360 <= lambdamin < lambdamax <= 830 (pbrt-v4 spectrum range)"
            )
        if int(args.spectral_nbuckets) < 1:
            raise ValueError("spectral-nbuckets must be >= 1")

    film_block: list[str]
    if film_type == "rgb":
        film_block = [
            'Film "rgb"',
            f'    "string filename" ["{film_filename}"]',
            '    "integer xresolution" [%d]' % int(args.xres),
            '    "integer yresolution" [%d]' % int(args.yres),
            '    "bool savefp16" false',
            '    "float iso" [100]',
        ]
    else:
        film_block = [
            'Film "spectral"',
            f'    "string filename" ["{film_filename}"]',
            '    "integer xresolution" [%d]' % int(args.xres),
            '    "integer yresolution" [%d]' % int(args.yres),
            '    "bool savefp16" false',
            '    "integer nbuckets" [%d]' % int(args.spectral_nbuckets),
            '    "float lambdamin" [%.6g]' % float(args.spectral_lambda_min),
            '    "float lambdamax" [%.6g]' % float(args.spectral_lambda_max),
        ]

    pbrt_lines: list[str] = [
        "# Generated by tools/build_colorchecker_scene.py — do not hand-edit.",
        f"# Illuminant: {_rel(repo, ill_path)}",
        f"# Film: {film_type} -> {film_filename}",
        'Option "disablepixeljitter" false',
        'Option "seed" 0',
        "",
        'ColorSpace "srgb"',
        "",
        'Sampler "zsobol" "integer pixelsamples" [%d]' % int(args.pixelsamples),
        'Integrator "path" "integer maxdepth" [6]',
        'PixelFilter "gaussian"',
        "",
        *film_block,
        "",
        "LookAt",
        "0 0 %.6f" % args.cam_dist,
        "    0 0 0",
        "    0 1 0",
    ]

    focus_d = float(args.focus_distance) if args.focus_distance is not None else float(args.cam_dist)
    camera_kind = "pinhole" if args.camera == "perspective" else args.camera
    if camera_kind == "pinhole":
        pbrt_lines.append('Camera "perspective" "float fov" [%s]' % args.fov)
    elif camera_kind == "thinlens":
        thin_focal_d = (
            float(args.thinlens_focal_distance)
            if args.thinlens_focal_distance is not None
            else float(args.cam_dist)
        )
        pbrt_lines.extend(
            [
                'Camera "perspective"',
                f'    "float fov" [{float(args.fov):.6g}]',
                f'    "float lensradius" [{float(args.thinlens_lens_radius):.6g}]',
                f'    "float focaldistance" [{thin_focal_d:.6g}]',
            ]
        )
    else:
        lens_repo = (repo / args.lensfile).resolve()
        if not lens_repo.is_file():
            raise FileNotFoundError(f'realistic camera: lens file not found: {lens_repo} (from --lensfile {args.lensfile!r})')
        lens_for_scene = os.path.relpath(str(lens_repo), str(out_dir.resolve()))
        pbrt_lines.extend(
            [
                'Camera "realistic"',
                f'    "string lensfile" ["{lens_for_scene}"]',
                f'    "float aperturediameter" [{float(args.aperture_diameter_mm):.6g}]',
                f'    "float focusdistance" [{focus_d:.6g}]',
            ]
        )

    pbrt_lines.extend(
        [
            "",
            "WorldBegin",
            "",
            "# D55 distant light (spectrum scaled for exposure)",
            'LightSource "distant"',
            '    "spectrum L" "spd/illuminant_D55.spd"',
            '    "float scale" [%s]' % args.light_scale,
            '    "point3 from" [0.12 0.55 2.9]',
            '    "point3 to" [0 0 0]',
            "",
            "# Neutral surround (reduces edge color bleeding)",
            "AttributeBegin",
            '    Material "diffuse" "rgb reflectance" [0.22 0.22 0.22]',
            '    Shape "bilinearmesh"',
            '        "point3 P" [ -3.5 -2.8 -0.08   3.5 -2.8 -0.08   -3.5 2.8 -0.08   3.5 2.8 -0.08 ]',
            '        "point2 uv" [ 0 0   1 0   0 1   1 1 ]',
            "AttributeEnd",
            "",
        ]
    )

    for row in range(4):
        for col in range(6):
            idx = row * 6 + col + 1
            meta = patches_meta[idx - 1]
            xL = x0 + col * (pw + g)
            xR = xL + pw
            y_max = y_top - row * (ph + g)
            y_min = y_max - ph
            # Match pbrt-v4 LookAt: camera +X is world -X; mirror board in X so patch 01 is image-left.
            p00 = (-xR, y_min, 0.0)
            p10 = (-xL, y_min, 0.0)
            p01 = (-xR, y_max, 0.0)
            p11 = (-xL, y_max, 0.0)
            pbrt_lines.append("AttributeBegin")
            pbrt_lines.append(f'    Material "diffuse" "spectrum reflectance" "{meta["spd"]}"')
            pbrt_lines.append('    Shape "bilinearmesh"')
            pbrt_lines.append(
                '        "point3 P" [ %s %s %s   %s %s %s   %s %s %s   %s %s %s ]'
                % (
                    p00[0],
                    p00[1],
                    p00[2],
                    p10[0],
                    p10[1],
                    p10[2],
                    p01[0],
                    p01[1],
                    p01[2],
                    p11[0],
                    p11[1],
                    p11[2],
                )
            )
            pbrt_lines.append('        "point2 uv" [ 0 0   1 0   0 1   1 1 ]')
            pbrt_lines.append("AttributeEnd")
            pbrt_lines.append("")

    scene_path = out_dir / "colorchecker.pbrt"
    scene_path.write_text("\n".join(pbrt_lines) + "\n")

    manifest = {
        "scene": str(scene_path.relative_to(repo)),
        "illuminant_csv": _rel(repo, ill_path),
        "patches": patches_meta,
        "geometry": {
            "patch_width": pw,
            "patch_height": ph,
            "gap": g,
            "board_size": [board_w, board_h],
        },
        "camera": {
            "type": camera_kind,
            "cam_dist": args.cam_dist,
            "fov_deg": args.fov,
            "lookat": {"eye": [0.0, 0.0, args.cam_dist], "target": [0.0, 0.0, 0.0], "up": [0.0, 1.0, 0.0]},
            **(
                {
                    "lens_radius": float(args.thinlens_lens_radius),
                    "focal_distance": (
                        float(args.thinlens_focal_distance)
                        if args.thinlens_focal_distance is not None
                        else float(args.cam_dist)
                    ),
                }
                if camera_kind == "thinlens"
                else {}
            ),
            **(
                {
                    "lensfile": _rel(repo, (repo / args.lensfile).resolve()),
                    "aperture_diameter_mm": float(args.aperture_diameter_mm),
                    "focus_distance": focus_d,
                }
                if camera_kind == "realistic"
                else {}
            ),
        },
        "film": {
            "type": film_type,
            "filename": film_filename,
            "xresolution": args.xres,
            "yresolution": args.yres,
            **(
                {
                    "nbuckets": int(args.spectral_nbuckets),
                    "lambda_min_nm": float(args.spectral_lambda_min),
                    "lambda_max_nm": float(args.spectral_lambda_max),
                }
                if film_type == "spectral"
                else {}
            ),
        },
        "spectrum_clip_nm": [wl_lo, wl_hi],
        "spd_step_nm": args.step_nm,
        # Matches LightSource "distant" in this file (for cosine shading in sensor_forward).
        "lighting": {
            "distant": {
                "from": [0.12, 0.55, 2.9],
                "to": [0.0, 0.0, 0.0],
                "scale": float(args.light_scale),
            }
        },
    }
    (out_dir / "colorchecker_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # High-res grids for validation tooling (1 nm)
    refl_rows = []
    for pth in patch_files:
        rwl, rval = load_csv_spectrum(pth)
        refl_rows.append(np.interp(grid_hi, rwl, rval, left=rval[0], right=rval[-1]))
    np.savez_compressed(
        out_dir / "spectral_reference_1nm.npz",
        wavelength_nm=grid_hi,
        illuminant=ill_hi,
        reflectance=np.stack(refl_rows, axis=0),
    )


if __name__ == "__main__":
    main()
