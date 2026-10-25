"""pbrt scene-text helpers shared by the scene builders (film blocks, SPD files, repo-relative paths)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def spectral_film_lines(
    filename: str, xres: int, yres: int, nbuckets: int, lambda_min: float, lambda_max: float
) -> list[str]:
    """``Film "spectral"`` block (fp32 EXR, ``nbuckets`` buckets over ``lambda_min``..``lambda_max`` nm)."""
    return [
        'Film "spectral"',
        f'    "string filename" ["{filename}"]',
        f'    "integer xresolution" [{int(xres)}]',
        f'    "integer yresolution" [{int(yres)}]',
        '    "bool savefp16" false',
        f'    "integer nbuckets" [{int(nbuckets)}]',
        f'    "float lambdamin" [{float(lambda_min):.6g}]',
        f'    "float lambdamax" [{float(lambda_max):.6g}]',
    ]


def rgb_film_lines(filename: str, xres: int, yres: int, iso: float | None = 100) -> list[str]:
    """``Film "rgb"`` block (fp32 EXR); ``iso=None`` leaves pbrt's default."""
    lines = [
        'Film "rgb"',
        f'    "string filename" ["{filename}"]',
        f'    "integer xresolution" [{int(xres)}]',
        f'    "integer yresolution" [{int(yres)}]',
        '    "bool savefp16" false',
    ]
    return lines + ([f'    "float iso" [{iso:g}]'] if iso is not None else [])


def write_spd_table(path: Path, wl: np.ndarray, val: np.ndarray) -> None:
    """Tab-separated SPD file with a ``# wavelength_nm<TAB>value`` header (12 significant digits)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"{float(w):.6g}\t{float(v):.12g}" for w, v in zip(wl, val))
    path.write_text("# wavelength_nm\tvalue\n" + body + "\n")


def rel_to(root: Path, p: Path) -> str:
    """``p`` relative to ``root`` when below it (no symlink resolution), else ``p`` unchanged."""
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)
