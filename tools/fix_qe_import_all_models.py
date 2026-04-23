#!/usr/bin/env python3
"""Repair imported QE CSVs for all camera models.

Repairs two known import artifacts:
- wavelength axis in normalized units (0..1-ish) instead of nanometers
- red/blue channel inversion inferred from peak ordering

Usage:
  python tools/fix_qe_import_all_models.py --repo-root . --write
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class Curve:
    path: Path
    w: np.ndarray
    v: np.ndarray
    normalized_axis: bool


def read_curve(path: Path) -> Curve:
    wl: list[float] = []
    val: list[float] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) != 2:
            continue
        try:
            wl.append(float(parts[0]))
            val.append(float(parts[1]))
        except ValueError:
            continue
    if not wl:
        raise ValueError(f"no numeric data in {path}")
    w = np.asarray(wl, dtype=np.float64)
    v = np.asarray(val, dtype=np.float64)
    ok = np.isfinite(w) & np.isfinite(v)
    w = w[ok]
    v = v[ok]
    if w.size == 0:
        raise ValueError(f"no finite samples in {path}")
    idx = np.argsort(w)
    w = w[idx]
    v = v[idx]
    return Curve(path=path, w=w, v=v, normalized_axis=float(np.max(w)) <= 10.0)


def map_axis_to_nm(curve: Curve) -> Curve:
    if not curve.normalized_axis:
        return curve
    wmin = float(np.min(curve.w))
    wmax = float(np.max(curve.w))
    if wmax - wmin <= 1e-12:
        raise ValueError(f"cannot map near-constant normalized axis: {curve.path}")
    w_nm = 380.0 + (curve.w - wmin) * (400.0 / (wmax - wmin))
    idx = np.argsort(w_nm)
    return Curve(path=curve.path, w=w_nm[idx], v=curve.v[idx], normalized_axis=False)


def peak_nm(curve: Curve) -> float:
    i = int(np.argmax(curve.v))
    return float(curve.w[i])


def write_curve(path: Path, w: np.ndarray, v: np.ndarray) -> None:
    lines = [f"{float(x):.6f},{float(y):.10f}" for x, y in zip(w, v)]
    path.write_text("\n".join(lines) + "\n")


def load_model_qe_paths(repo: Path, model_path: Path) -> tuple[Path, Path, Path]:
    cfg = yaml.safe_load(model_path.read_text()) or {}
    qe = ((cfg.get("sensor") or {}).get("quantum_efficiency") or {})
    return (
        (repo / qe["red_csv"]).resolve(),
        (repo / qe["green_csv"]).resolve(),
        (repo / qe["blue_csv"]).resolve(),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    repo_default = Path(__file__).resolve().parent.parent
    ap.add_argument("--repo-root", type=Path, default=repo_default)
    ap.add_argument("--models-dir", type=Path, default=Path("config/camera_models"))
    ap.add_argument("--write", action="store_true", help="Write repaired CSVs in place")
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    models_dir = (repo / args.models_dir).resolve()
    model_files = sorted(models_dir.glob("*.yaml"))

    # Operate per unique QE triple to avoid double-writing shared files.
    triples: dict[tuple[Path, Path, Path], set[str]] = {}
    for model in model_files:
        try:
            triple = load_model_qe_paths(repo, model)
        except Exception:
            continue
        triples.setdefault(triple, set()).add(model.stem)

    axis_fixed = 0
    swap_fixed = 0
    inspected = 0

    for (r_path, g_path, b_path), owners in sorted(triples.items(), key=lambda x: tuple(str(p) for p in x[0])):
        # Only repair extracted per-camera QE sets; leave global interpolated defaults untouched.
        if "/spectra/QE/cameras/" not in str(r_path):
            continue
        inspected += 1

        r_in = read_curve(r_path)
        g_in = read_curve(g_path)
        b_in = read_curve(b_path)
        had_axis_issue = r_in.normalized_axis or g_in.normalized_axis or b_in.normalized_axis
        r = map_axis_to_nm(r_in)
        g = map_axis_to_nm(g_in)
        b = map_axis_to_nm(b_in)

        # If any original axis was normalized, count once per triple.
        if had_axis_issue:
            axis_fixed += 1

        r_pk = peak_nm(r)
        g_pk = peak_nm(g)
        b_pk = peak_nm(b)
        need_swap = (r_pk < g_pk) and (b_pk > g_pk)
        if need_swap:
            swap_fixed += 1
            r_w, r_v = b.w, b.v
            b_w, b_v = r.w, r.v
        else:
            r_w, r_v = r.w, r.v
            b_w, b_v = b.w, b.v

        if args.write:
            write_curve(r_path, r_w, r_v)
            write_curve(g_path, g.w, g.v)
            write_curve(b_path, b_w, b_v)

        owners_list = ",".join(sorted(owners))
        print(
            f"{owners_list}: axis_fixed={1 if had_axis_issue else 0} "
            f"rb_swap={1 if need_swap else 0} peaks_nm=({float(r_w[int(np.argmax(r_v))]):.1f},{peak_nm(g):.1f},{float(b_w[int(np.argmax(b_v))]):.1f})"
        )

    print("")
    print(f"QE triples inspected (camera extracted): {inspected}")
    print(f"Triples needing axis repair: {axis_fixed}")
    print(f"Triples needing red/blue swap: {swap_fixed}")
    if not args.write:
        print("Dry run only. Re-run with --write to apply fixes.")


if __name__ == "__main__":
    main()
