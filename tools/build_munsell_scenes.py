#!/usr/bin/env python3
"""Generate PBRT scenes for Munsell patches grouped by hue family.

Input data: Joensuu matte Munsell MAT dataset (380-800 nm, 1 nm).
Output: one scene per hue family with per-patch SPD files, manifest JSON, and spectral NPZ.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from munsell_mat import load_joensuu_mat, parse_munsell_label, sanitize_filename

HUE_FAMILY_ORDER = ("R", "YR", "Y", "GY", "G", "BG", "B", "PB", "P", "RP", "N")
HUE_STEP_ORDER = (2.5, 5.0, 7.5, 10.0)


@dataclass(frozen=True)
class Chip:
    index_1based: int
    label: str
    hue_family: str
    hue_step: float | None
    value: float | None
    chroma: float | None
    reflectance: np.ndarray


@dataclass(frozen=True)
class HueGroup:
    slug: str
    family: str
    step: float | None
    label: str


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
    v = np.interp(grid, wl, val)
    return grid, v


def write_spd(path: Path, wl: np.ndarray, val: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"{float(w):.6g}\t{float(v):.12g}" for w, v in zip(wl, val))
    path.write_text("# wavelength_nm\tvalue\n" + body + "\n")


def _hue_family_from_token(hue: str | None) -> str:
    if hue is None:
        return "N"
    s = str(hue).strip().upper().replace(" ", "")
    if s.startswith("N"):
        return "N"
    letters = re.sub(r"[^A-Z]", "", s)
    if letters in HUE_FAMILY_ORDER:
        return letters
    for fam in sorted(HUE_FAMILY_ORDER, key=len, reverse=True):
        if fam != "N" and letters.endswith(fam):
            return fam
    return "N"


def _hue_step_from_token(hue: str | None) -> float | None:
    if hue is None:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", hue)
    if m is None:
        return None
    return float(m.group(1))


def _hue_step_bucket(h: float | None) -> float | None:
    if h is None:
        return None
    return min(HUE_STEP_ORDER, key=lambda t: abs(float(h) - t))


def build_chip_table(wavelength_nm: np.ndarray, reflectance: np.ndarray, labels: list[str]) -> list[Chip]:
    chips: list[Chip] = []
    for idx in range(reflectance.shape[1]):
        label = labels[idx]
        parsed = parse_munsell_label(label)
        hue_token = parsed.get("hue")
        hue_family = _hue_family_from_token(hue_token)
        hue_step = _hue_step_from_token(hue_token)
        value = parsed.get("value")
        chroma = parsed.get("chroma")
        chips.append(
            Chip(
                index_1based=idx + 1,
                label=label,
                hue_family=hue_family,
                hue_step=float(hue_step) if hue_step is not None else None,
                value=float(value) if value is not None else None,
                chroma=float(chroma) if chroma is not None else None,
                reflectance=reflectance[:, idx].astype(np.float64, copy=False),
            )
        )
    return chips


def _chip_sort_key(chip: Chip) -> tuple[float, float, float, int]:
    return (
        chip.value if chip.value is not None else math.inf,
        chip.chroma if chip.chroma is not None else math.inf,
        chip.hue_step if chip.hue_step is not None else math.inf,
        chip.index_1based,
    )


def _value_group_key(value: float | None) -> float:
    if value is None:
        return -math.inf
    return float(value)


def _chroma_sort_key(chip: Chip) -> tuple[float, float, int]:
    return (
        chip.chroma if chip.chroma is not None else -math.inf,
        chip.hue_step if chip.hue_step is not None else math.inf,
        chip.index_1based,
    )


def layout_value_chroma(chips: list[Chip], min_columns: int) -> tuple[list[tuple[int, int, Chip]], int, int]:
    value_to_chips: dict[float | None, list[Chip]] = {}
    for chip in chips:
        value_to_chips.setdefault(chip.value, []).append(chip)

    value_levels_asc = sorted(value_to_chips.keys(), key=_value_group_key)
    rows = len(value_levels_asc)
    max_row_len = max(len(value_to_chips[v]) for v in value_levels_asc)
    cols = max(1, min_columns, max_row_len)

    placements: list[tuple[int, int, Chip]] = []
    for row_from_bottom, value_level in enumerate(value_levels_asc):
        # Geometry loop uses row 0 as top, so invert value row index.
        row_from_top = rows - 1 - row_from_bottom
        row_chips = sorted(value_to_chips[value_level], key=_chroma_sort_key)
        for col, chip in enumerate(row_chips):
            placements.append((row_from_top, col, chip))
    return placements, rows, cols


def _make_film_block(args: argparse.Namespace, film_type: str, film_filename: str) -> list[str]:
    if film_type == "rgb":
        return [
            'Film "rgb"',
            f'    "string filename" ["{film_filename}"]',
            '    "integer xresolution" [%d]' % int(args.xres),
            '    "integer yresolution" [%d]' % int(args.yres),
            '    "bool savefp16" false',
            '    "float iso" [100]',
        ]

    if not str(film_filename).lower().endswith(".exr"):
        raise ValueError(f'SpectralFilm requires an .exr filename, got "{film_filename}"')
    lo, hi = float(args.spectral_lambda_min), float(args.spectral_lambda_max)
    if lo < 360.0 or hi > 830.0 or lo >= hi:
        raise ValueError("spectral lambda range must satisfy 360 <= lambdamin < lambdamax <= 830")
    if int(args.spectral_nbuckets) < 1:
        raise ValueError("spectral-nbuckets must be >= 1")
    return [
        'Film "spectral"',
        f'    "string filename" ["{film_filename}"]',
        '    "integer xresolution" [%d]' % int(args.xres),
        '    "integer yresolution" [%d]' % int(args.yres),
        '    "bool savefp16" false',
        '    "integer nbuckets" [%d]' % int(args.spectral_nbuckets),
        '    "float lambdamin" [%.6g]' % float(args.spectral_lambda_min),
        '    "float lambdamax" [%.6g]' % float(args.spectral_lambda_max),
    ]


def _board_geometry(n: int, columns: int, patch_width: float, patch_height: float, gap: float) -> tuple[int, float, float]:
    cols = max(1, columns)
    rows = int(math.ceil(float(n) / float(cols)))
    board_w = cols * patch_width + max(0, cols - 1) * gap
    board_h = rows * patch_height + max(0, rows - 1) * gap
    return rows, board_w, board_h


def emit_hue_scene(
    repo: Path,
    group: HueGroup,
    chips: list[Chip],
    wavelength_nm: np.ndarray,
    illuminant_csv: Path,
    args: argparse.Namespace,
) -> None:
    hue_slug = sanitize_filename(group.slug)
    scene_dir = args.out_dir / hue_slug
    spd_dir = scene_dir / "spd"
    scene_dir.mkdir(parents=True, exist_ok=True)
    spd_dir.mkdir(parents=True, exist_ok=True)

    # Illuminant SPD.
    ill_wl: list[float] = []
    ill_val: list[float] = []
    for line in illuminant_csv.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r",\s*", line, maxsplit=1)
        if len(parts) != 2:
            continue
        ill_wl.append(float(parts[0]))
        ill_val.append(float(parts[1]))
    if not ill_wl:
        raise ValueError(f"no illuminant spectral data in {illuminant_csv}")
    iw = np.asarray(ill_wl, dtype=np.float64)
    iv = np.asarray(ill_val, dtype=np.float64)
    iorder = np.argsort(iw)
    iw, iv = iw[iorder], iv[iorder]
    iw_clip = np.interp(wavelength_nm, iw, iv, left=iv[0], right=iv[-1])
    iw_spd, iv_spd = subsample_for_spd(wavelength_nm, iw_clip, args.step_nm)
    write_spd(spd_dir / "illuminant_D55.spd", iw_spd, iv_spd)

    # Reflectance SPDs + value/chroma layout:
    # - lightness(value) increases bottom -> top
    # - chroma increases left -> right
    ordered = sorted(chips, key=_chip_sort_key)
    if not ordered:
        return

    placements, rows, cols = layout_value_chroma(ordered, args.columns)
    if args.max_patches_per_scene is not None:
        placements = placements[: max(0, int(args.max_patches_per_scene))]
    if not placements:
        return

    # Recompute effective board shape after optional patch cap.
    rows = max(r for r, _, _ in placements) + 1
    cols = max(c for _, c, _ in placements) + 1

    patch_meta: list[dict] = []
    refl_rows: list[np.ndarray] = []
    placements_sorted = sorted(placements, key=lambda t: (t[0], t[1], t[2].index_1based))
    for patch_idx, (_, _, chip) in enumerate(placements_sorted, start=1):
        swl, sval = subsample_for_spd(wavelength_nm, chip.reflectance, args.step_nm)
        spd_name = f"patch_{patch_idx:04d}_{sanitize_filename(chip.label)}.spd"
        write_spd(spd_dir / spd_name, swl, sval)
        patch_meta.append(
            {
                "patch_index": patch_idx,
                "chip_index_1based": chip.index_1based,
                "label": chip.label,
                "hue_family": chip.hue_family,
                "hue_step": chip.hue_step,
                "value": chip.value,
                "chroma": chip.chroma,
                "spd": f"spd/{spd_name}",
            }
        )
        refl_rows.append(chip.reflectance)

    pw, ph, gap = args.patch_width, args.patch_height, args.gap
    board_w = cols * pw + max(0, cols - 1) * gap
    board_h = rows * ph + max(0, rows - 1) * gap
    x0 = -board_w / 2.0
    y_top = board_h / 2.0

    film_type = str(args.film).lower()
    if args.film_output_template:
        film_filename = args.film_output_template.format(hue=hue_slug)
    elif film_type == "rgb":
        film_filename = f"out/munsell_{hue_slug}.exr"
    else:
        film_filename = f"out/munsell_{hue_slug}_spectral.exr"

    film_block = _make_film_block(args, film_type, film_filename)

    scene_name = f"munsell_{hue_slug}.pbrt"
    scene_path = scene_dir / scene_name
    pbrt_lines: list[str] = [
        "# Generated by tools/build_munsell_scenes.py - do not hand-edit.",
        f"# MAT source: {_rel(repo, args.mat)}",
        f"# Hue group: {group.label}",
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
    if args.camera == "perspective":
        pbrt_lines.append('Camera "perspective" "float fov" [%s]' % args.fov)
    else:
        lens_repo = (repo / args.lensfile).resolve()
        if not lens_repo.is_file():
            raise FileNotFoundError(f"realistic camera: lens file not found: {lens_repo}")
        lens_for_scene = os.path.relpath(str(lens_repo), str(scene_dir.resolve()))
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
            "# D55 distant light",
            'LightSource "distant"',
            '    "spectrum L" "spd/illuminant_D55.spd"',
            '    "float scale" [%s]' % args.light_scale,
            '    "point3 from" [0.12 0.55 2.9]',
            '    "point3 to" [0 0 0]',
            "",
            "# Neutral surround",
            "AttributeBegin",
            '    Material "diffuse" "rgb reflectance" [0.22 0.22 0.22]',
            '    Shape "bilinearmesh"',
            '        "point3 P" [ -3.5 -2.8 -0.08   3.5 -2.8 -0.08   -3.5 2.8 -0.08   3.5 2.8 -0.08 ]',
            '        "point2 uv" [ 0 0   1 0   0 1   1 1 ]',
            "AttributeEnd",
            "",
        ]
    )

    for (row, col, _), meta in zip(placements_sorted, patch_meta):
        xL = x0 + col * (pw + gap)
        xR = xL + pw
        y_max = y_top - row * (ph + gap)
        y_min = y_max - ph
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

    scene_path.write_text("\n".join(pbrt_lines) + "\n")

    manifest = {
        "scene": _rel(repo, scene_path),
        "source_mat": _rel(repo, args.mat),
        "hue_family": group.family,
        "hue_step_group": group.step,
        "hue_group_label": group.label,
        "ordering": {
            "group": "hue_family",
            "sort": ["value_asc", "chroma_asc", "hue_step_asc", "chip_index_asc"],
            "equivalent_axes": {"x": "chroma", "y": "value"},
            "layout_constraints": {
                "value_direction": "bottom_to_top",
                "chroma_direction": "left_to_right",
            },
        },
        "patches": patch_meta,
        "layout": {"columns": cols, "rows": rows},
        "geometry": {
            "patch_width": pw,
            "patch_height": ph,
            "gap": gap,
            "board_size": [board_w, board_h],
        },
        "camera": {
            "type": args.camera,
            "cam_dist": args.cam_dist,
            "fov_deg": args.fov,
            "lookat": {"eye": [0.0, 0.0, args.cam_dist], "target": [0.0, 0.0, 0.0], "up": [0.0, 1.0, 0.0]},
            **(
                {
                    "lensfile": _rel(repo, (repo / args.lensfile).resolve()),
                    "aperture_diameter_mm": float(args.aperture_diameter_mm),
                    "focus_distance": focus_d,
                }
                if args.camera == "realistic"
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
        "spectrum_clip_nm": [float(wavelength_nm[0]), float(wavelength_nm[-1])],
        "spd_step_nm": args.step_nm,
        "lighting": {"distant": {"from": [0.12, 0.55, 2.9], "to": [0.0, 0.0, 0.0], "scale": float(args.light_scale)}},
    }
    (scene_dir / f"munsell_{hue_slug}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    np.savez_compressed(
        scene_dir / f"munsell_{hue_slug}_spectral_reference_1nm.npz",
        wavelength_nm=wavelength_nm,
        illuminant=iw_clip,
        reflectance=np.stack(refl_rows, axis=0),
    )


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=root)
    ap.add_argument("--mat", type=Path, default=None, help="Path to munsell380_800_1.mat")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory root (default: scenes/generated/munsell)")
    ap.add_argument("--illuminant", type=Path, default=None, help="Illuminant CSV (default: spectra/illuminant/interpolated/D55.csv)")
    ap.add_argument("--hues", type=str, default="all", help='Comma-separated hue families (e.g. "R,YR,Y"), or "all"')
    ap.add_argument("--max-patches-per-scene", type=int, default=None, help="Optional cap per hue family scene")
    ap.add_argument("--columns", type=int, default=12, help="Grid columns per hue-family scene")
    ap.add_argument("--step-nm", type=float, default=5.0, help="SPD file wavelength step (nm)")
    ap.add_argument("--patch-width", type=float, default=0.28)
    ap.add_argument("--patch-height", type=float, default=0.28)
    ap.add_argument("--gap", type=float, default=0.025)
    ap.add_argument("--light-scale", type=float, default=2.0)
    ap.add_argument("--cam-dist", type=float, default=6.0)
    ap.add_argument("--camera", choices=("perspective", "realistic"), default="perspective")
    ap.add_argument("--lensfile", type=str, default="scenes/lenses/wide_22mm.dat")
    ap.add_argument("--aperture-diameter-mm", type=float, default=4.0)
    ap.add_argument("--focus-distance", type=float, default=None)
    ap.add_argument("--fov", type=float, default=35.0)
    ap.add_argument("--xres", type=int, default=1600)
    ap.add_argument("--yres", type=int, default=1200)
    ap.add_argument("--pixelsamples", type=int, default=64)
    ap.add_argument("--film", choices=("rgb", "spectral"), default="rgb")
    ap.add_argument(
        "--film-output-template",
        type=str,
        default=None,
        help='Optional output template with "{hue}" placeholder, e.g. "out/munsell_{hue}.exr"',
    )
    ap.add_argument("--spectral-nbuckets", type=int, default=16)
    ap.add_argument("--spectral-lambda-min", type=float, default=360.0)
    ap.add_argument("--spectral-lambda-max", type=float, default=830.0)
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    args.mat = (args.mat or (repo / "spectra" / "munsell" / "munsell380_800_1.mat")).resolve()
    args.out_dir = (args.out_dir or (repo / "scenes" / "generated" / "munsell")).resolve()
    illuminant_csv = (args.illuminant or (repo / "spectra" / "illuminant" / "interpolated" / "D55.csv")).resolve()

    bundle = load_joensuu_mat(args.mat)
    chips = build_chip_table(bundle.wavelength_nm, bundle.reflectance, bundle.labels)
    by_hue: dict[str, list[Chip]] = {}
    for chip in chips:
        step_bucket = _hue_step_bucket(chip.hue_step)
        if chip.hue_family == "N" or step_bucket is None:
            key = chip.hue_family
        else:
            key = f"{chip.hue_family}_{step_bucket:g}"
        by_hue.setdefault(key, []).append(chip)

    if args.hues.strip().lower() == "all":
        hue_list: list[str] = []
        for fam in HUE_FAMILY_ORDER:
            if fam == "N":
                if "N" in by_hue:
                    hue_list.append("N")
                continue
            for step in HUE_STEP_ORDER:
                k = f"{fam}_{step:g}"
                if k in by_hue:
                    hue_list.append(k)
        hue_list.extend(h for h in sorted(by_hue.keys()) if h not in hue_list)
    else:
        hue_list = []
        requested = [h.strip().upper() for h in args.hues.split(",") if h.strip()]
        for token in requested:
            if token in by_hue:
                hue_list.append(token)
                continue
            # Family token expands to all available step groups for that family.
            fam_matches = [k for k in by_hue if k == token or k.startswith(f"{token}_")]
            if fam_matches:
                # Stable step-first order for family expansions.
                fam_ordered = []
                for step in HUE_STEP_ORDER:
                    k = f"{token}_{step:g}"
                    if k in by_hue:
                        fam_ordered.append(k)
                if token in by_hue and token not in fam_ordered:
                    fam_ordered.append(token)
                fam_ordered.extend(k for k in sorted(fam_matches) if k not in fam_ordered)
                hue_list.extend(fam_ordered)

    for hue in hue_list:
        hue_chips = by_hue.get(hue, [])
        if not hue_chips:
            continue
        family = hue.split("_", 1)[0]
        step = None
        label = family
        if "_" in hue:
            _, step_s = hue.split("_", 1)
            try:
                step = float(step_s)
                label = f"{family} {step:g}"
            except ValueError:
                step = None
        group = HueGroup(slug=hue, family=family, step=step, label=label)
        emit_hue_scene(repo, group, hue_chips, bundle.wavelength_nm, illuminant_csv, args)

    index = {
        "source_mat": _rel(repo, args.mat),
        "out_dir": _rel(repo, args.out_dir),
        "hues_requested": hue_list,
        "hues_generated": [h for h in hue_list if (args.out_dir / sanitize_filename(h)).is_dir()],
    }
    (args.out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")


if __name__ == "__main__":
    main()
