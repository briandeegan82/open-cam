#!/usr/bin/env python3
"""Road-surface reflection tables (CIE reduced luminance coefficients r) for the pbrt ``roadrtable`` material.

An r-table gives ``r(tan gamma, beta) = q cos^3(gamma)`` where q = L / E_h [cd m^-2 lx^-1] is the luminance
coefficient for an observer about 1 deg above the road (CIE 30.2 / CIE 144 geometry): gamma is the light's
incidence angle from the road normal and beta the angle between the vertical planes of observation and
incidence (beta = 0: light beyond the point, facing the observer). Tables are stored as CSV
(``config/road_rtables/*.csv``) with ``# key: value`` header lines (name, source, placeholder, scale), one
header row of beta [deg] and one row per tan(gamma); entries are multiplied by ``scale`` (tables print
``10^4 r``, so ``scale: 1e-4``).

The functions here mirror the C++ evaluation in ``third_party/patches/0007-road-rtable-material.patch``
(bilinear in r, tan(gamma) and beta clamped to the table; Q0 by the same midpoint rule; Lambertian blend
with Q0 for observer elevations above ``observationmin``), so scene builders can solve photometry with
exactly what pbrt renders.

No CIE table ships with the repo: ``placeholder_dry_model.csv`` is generated from the open-cam dry-asphalt
layered model (tools/highway_wet.py) on a CIE-style grid and is NOT CIE data. Supply the standard tables as
CSV (see docs/ROAD_RTABLE.md).
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
TABLE_DIR = REPO / "config" / "road_rtables"
PLACEHOLDER = TABLE_DIR / "placeholder_dry_model.csv"
OBSERVATION_MIN_DEG = 1.5
OBSERVATION_MAX_DEG = 15.0
Q0_NG, Q0_NB = 512, 180  # midpoint grid for Q0, same as the C++ material


@dataclass
class RTable:
    name: str
    tan_gamma: np.ndarray  # (n,)
    beta_deg: np.ndarray  # (m,)
    r: np.ndarray  # (n, m) reduced luminance coefficients [cd m^-2 lx^-1]
    source: str = ""
    placeholder: bool = False
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tan_gamma = np.asarray(self.tan_gamma, dtype=np.float64)
        self.beta_deg = np.asarray(self.beta_deg, dtype=np.float64)
        self.r = np.asarray(self.r, dtype=np.float64)
        t, b = self.tan_gamma, self.beta_deg
        if t.ndim != 1 or b.ndim != 1 or len(t) < 2 or len(b) < 2:
            raise ValueError(f"{self.name}: need >= 2 tan(gamma) rows and >= 2 beta columns")
        if self.r.shape != (len(t), len(b)):
            raise ValueError(f"{self.name}: r has shape {self.r.shape}, expected {(len(t), len(b))}")
        if np.any(np.diff(t) <= 0) or np.any(np.diff(b) <= 0):
            raise ValueError(f"{self.name}: tan(gamma) and beta must be strictly increasing")
        if t[0] < 0 or b[0] < 0 or b[-1] > 180:
            raise ValueError(f"{self.name}: need tan(gamma) >= 0 and 0 <= beta <= 180 deg")
        if np.any(~np.isfinite(self.r)) or np.any(self.r < 0):
            raise ValueError(f"{self.name}: r must be finite and non-negative")

    # ---- evaluation (mirrors RoadRTable::Q)
    def q(self, tan_gamma, beta_deg) -> np.ndarray:
        """Luminance coefficient q = r / cos^3(gamma) [cd m^-2 lx^-1]; inputs clamped to the table range."""
        t = np.clip(np.asarray(tan_gamma, dtype=np.float64), self.tan_gamma[0], self.tan_gamma[-1])
        b = np.clip(np.asarray(beta_deg, dtype=np.float64), self.beta_deg[0], self.beta_deg[-1])
        t, b = np.broadcast_arrays(t, b)
        i = np.clip(np.searchsorted(self.tan_gamma, t, side="right") - 1, 0, len(self.tan_gamma) - 2)
        j = np.clip(np.searchsorted(self.beta_deg, b, side="right") - 1, 0, len(self.beta_deg) - 2)
        ft = np.clip((t - self.tan_gamma[i]) / (self.tan_gamma[i + 1] - self.tan_gamma[i]), 0, 1)
        fb = np.clip((b - self.beta_deg[j]) / (self.beta_deg[j + 1] - self.beta_deg[j]), 0, 1)
        r = self.r
        rv = (1 - ft) * ((1 - fb) * r[i, j] + fb * r[i, j + 1]) + ft * ((1 - fb) * r[i + 1, j] + fb * r[i + 1, j + 1])
        return rv * (1 + t * t) ** 1.5

    def q0(self) -> float:
        """Average luminance coefficient Q0: solid-angle mean of q over the table's tan(gamma) range, all beta."""
        g = (np.arange(Q0_NG) + 0.5) / Q0_NG * math.atan(self.tan_gamma[-1])
        b = (np.arange(Q0_NB) + 0.5) / Q0_NB * 180.0
        q = self.q(np.tan(g)[:, None], b[None, :])
        w = np.sin(g)[:, None]
        return float((q * w).sum() / (w.sum() * Q0_NB))

    def s1(self) -> float:
        """Specular factor S1 = r(tan gamma = 2, beta = 0) / r(0, 0)."""
        return float(self.q(2.0, 0.0) * 5.0**-1.5 / self.q(0.0, 0.0))

    def brdf(self, cos_i, cos_o, cos_phi, alpha0_deg=OBSERVATION_MIN_DEG, alpha1_deg=OBSERVATION_MAX_DEG):
        """Material BRDF (mirrors RoadRTableBxDF::Q). ``cos_phi`` = -1: light beyond the point (beta = 0)."""
        ci, co, cp = np.broadcast_arrays(*(np.asarray(v, dtype=np.float64) for v in (cos_i, cos_o, cos_phi)))
        alpha = np.degrees(np.arcsin(np.clip(co, 0, 1)))
        w = smoothstep(alpha, alpha0_deg, alpha1_deg)
        t = np.sqrt(np.clip(1 - ci * ci, 0, None)) / np.maximum(ci, 1e-12)
        beta = 180.0 - np.degrees(np.arccos(np.clip(cp, -1, 1)))
        f = (1 - w) * self.q(t, beta) + w * self.q0()
        return np.where((ci > 0) & (co > 0), f, 0.0)

    def albedo(self, alpha_deg: float, n: int = 400) -> float:
        """Directional-hemispherical reflectance for an observer ``alpha_deg`` above the surface (reciprocity)."""
        th = (np.arange(n) + 0.5) / n * (math.pi / 2)
        ph = (np.arange(2 * n) + 0.5) / (2 * n) * (2 * math.pi)
        ci = np.cos(th)[:, None] + 0 * ph
        f = self.brdf(ci, math.sin(math.radians(alpha_deg)), np.cos(ph)[None, :])
        return float((f * ci * np.sin(th)[:, None]).sum() * (math.pi / 2 / n) * (math.pi / n))

    def summary(self) -> dict:
        return {
            "name": self.name,
            "source": self.source,
            "placeholder": self.placeholder,
            "Q0": self.q0(),
            "S1": self.s1(),
            "shape": list(self.r.shape),
        }


def smoothstep(x, a: float, b: float):
    """pbrt's SmoothStep (step at ``a`` when a == b)."""
    x = np.asarray(x, dtype=np.float64)
    if a == b:
        return np.where(x < a, 0.0, 1.0)
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


# ---- I/O
def load(path: str | Path) -> RTable:
    """Read an r-table CSV (see module docstring)."""
    path = Path(path)
    meta: dict[str, str] = {}
    rows: list[list[float]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            k, sep, v = line[1:].partition(":")
            if sep:
                meta.setdefault(k.strip().lower(), v.strip())
            continue
        cells = [c.strip() for c in line.split(",")]
        if not rows and not _is_float(cells[0]):
            cells = cells[1:]  # header row with a corner label
            rows.append([float("nan"), *map(float, cells)])
            continue
        rows.append([float(c) for c in cells])
    if len(rows) < 3:
        raise ValueError(f"{path}: need a beta header row and >= 2 tan(gamma) rows")
    beta = np.array(rows[0][1:])
    body = np.array(rows[1:])
    if body.shape[1] != len(beta) + 1:
        raise ValueError(f"{path}: every row needs tan(gamma) + {len(beta)} values")
    scale = float(meta.get("scale", "1e-4"))
    return RTable(
        name=meta.get("name", path.stem),
        tan_gamma=body[:, 0],
        beta_deg=beta,
        r=body[:, 1:] * scale,
        source=meta.get("source", ""),
        placeholder=meta.get("placeholder", "false").lower() in ("1", "true", "yes"),
        meta=meta,
    )


def resolve(name_or_path: str | Path) -> Path:
    """A table file path, or a name in config/road_rtables (``R3`` -> ``config/road_rtables/R3.csv``)."""
    p = Path(name_or_path)
    if p.is_file():
        return p.resolve()
    for cand in (TABLE_DIR / f"{name_or_path}.csv", TABLE_DIR / str(name_or_path)):
        if cand.is_file():
            return cand
    known = ", ".join(sorted(f.stem for f in TABLE_DIR.glob("*.csv")))
    raise SystemExit(f"r-table {name_or_path!r} not found (files or names in {TABLE_DIR}: {known})")


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def write_csv(table: RTable, path: Path, scale: float = 1e-4, header: list[str] | None = None) -> None:
    lines = [f"# name: {table.name}", f"# source: {table.source}", f"# placeholder: {str(table.placeholder).lower()}"]
    lines += [f"# {h}" for h in (header or [])]
    lines += [f"# scale: {scale:g}", "tan_gamma\\beta_deg," + ",".join(f"{b:g}" for b in table.beta_deg)]
    for t, row in zip(table.tan_gamma, table.r / scale):
        lines.append(f"{t:g}," + ",".join(f"{v:.4g}" for v in row))
    path.write_text("\n".join(lines) + "\n")


# ---- pbrt
def material_lines(name: str, table: RTable, tint: str, alpha0_deg=OBSERVATION_MIN_DEG, alpha1_deg=OBSERVATION_MAX_DEG):
    """``MakeNamedMaterial`` block; ``tint`` is a pbrt parameter, e.g. ``'"spectrum tint" "spd/x.spd"'``."""

    def arr(v) -> str:
        return " ".join(f"{x:.7g}" for x in np.ravel(v))

    src = (table.source or "unspecified").replace('"', "'")
    return [
        f"# road r-table {table.name} ({'PLACEHOLDER, not CIE data' if table.placeholder else src}),"
        f" Q0 = {table.q0():.4f}, S1 = {table.s1():.3f}",
        f'MakeNamedMaterial "{name}" "string type" "roadrtable" {tint}',
        f'    "float observationmin" [{alpha0_deg:g}] "float observationmax" [{alpha1_deg:g}] "float rscale" [1]',
        f'    "float tangamma" [{arr(table.tan_gamma)}]',
        f'    "float beta" [{arr(table.beta_deg)}]',
        f'    "float r" [{arr(table.r)}]',
    ]


def tint_spectrum(rho: np.ndarray, wl: np.ndarray) -> np.ndarray:
    """Spectral shape of the road, rho(lambda) / rho_Y (equal-energy luminous reflectance = 1)."""
    from colour_science import cmf_on_grid

    y = cmf_on_grid(wl)[1]
    return rho * (np.sum(y) / np.sum(rho * y))


def warn_if_placeholder(table: RTable) -> None:
    if table.placeholder:
        print(
            f"warning: road r-table {table.name!r} is a placeholder (open-cam dry-asphalt model), not CIE data",
            file=sys.stderr,
        )


# ---- placeholder
PLACEHOLDER_TAN = np.concatenate([np.arange(0, 2.0001, 0.25), np.arange(2.5, 12.0001, 0.5)])
PLACEHOLDER_BETA = np.array([0, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180.0])


def placeholder_table(a: float = 0.12) -> RTable:
    """r-table of the open-cam dry-asphalt layered model (body albedo ``a`` + rough coat), observer at 1 deg."""
    sys.path.insert(0, str(TOOLS))
    from highway_wet import N_COAT_DRY, luminance_coefficient

    t = PLACEHOLDER_TAN[:, None]
    ci = 1 / np.sqrt(1 + t * t) + 0 * PLACEHOLDER_BETA[None, :]
    q = luminance_coefficient(
        a, N_COAT_DRY, 0.35, ci, math.sin(math.radians(1.0)), -np.cos(np.radians(PLACEHOLDER_BETA))
    )
    return RTable(
        name="placeholder_dry_model",
        tan_gamma=PLACEHOLDER_TAN,
        beta_deg=PLACEHOLDER_BETA,
        r=q * ci**3,
        source="open-cam dry-asphalt layered model (tools/highway_wet.py), NOT a CIE table",
        placeholder=True,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write-placeholder", action="store_true", help=f"regenerate {PLACEHOLDER.relative_to(REPO)}")
    ap.add_argument("tables", nargs="*", help="r-table CSVs or names to summarise (Q0, S1, albedo)")
    args = ap.parse_args(argv)
    if args.write_placeholder:
        write_csv(
            placeholder_table(),
            PLACEHOLDER,
            header=[
                "note: PLACEHOLDER generated by tools/road_rtable.py --write-placeholder; not CIE 144 / EN 13201 data"
            ],
        )
        print(f"wrote {PLACEHOLDER}")
    for name in args.tables:
        tb = load(resolve(name))
        s = tb.summary()
        s["albedo_1deg"] = tb.albedo(1.0)
        s["albedo_60deg"] = tb.albedo(60.0)
        print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
