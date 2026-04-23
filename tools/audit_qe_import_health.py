#!/usr/bin/env python3
"""Audit camera-model QE CSV health (wavelength axis + channel ordering).

Flags likely import issues seen in extracted QE files:
- normalized/non-nm wavelength axis (e.g. 0..1 instead of 380..780/830 nm)
- likely red/blue channel inversion based on peak ordering
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class CurveInfo:
    path: Path
    wl_min: float
    wl_max: float
    peak_nm: float
    peak_val: float
    normalized_axis: bool


def read_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
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
        raise ValueError(f"no data in {path}")
    w = np.asarray(wl, dtype=np.float64)
    v = np.asarray(val, dtype=np.float64)
    ok = np.isfinite(w) & np.isfinite(v)
    w = w[ok]
    v = v[ok]
    if w.size == 0:
        raise ValueError(f"no finite samples in {path}")
    idx = np.argsort(w)
    return w[idx], v[idx]


def summarize_curve(path: Path) -> CurveInfo:
    w, v = read_curve(path)
    normalized_axis = float(np.max(w)) <= 10.0
    wn = w
    if normalized_axis:
        # Match runtime repair mapping used in pipeline tools.
        wmin = float(np.min(w))
        wmax = float(np.max(w))
        if wmax <= wmin:
            raise ValueError(f"invalid normalized wavelength axis in {path}")
        wn = 380.0 + (w - wmin) * (400.0 / (wmax - wmin))
    i = int(np.argmax(v))
    return CurveInfo(
        path=path,
        wl_min=float(np.min(w)),
        wl_max=float(np.max(w)),
        peak_nm=float(wn[i]),
        peak_val=float(v[i]),
        normalized_axis=normalized_axis,
    )


def model_qe_paths(repo: Path, model_path: Path) -> tuple[Path, Path, Path]:
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
    ap.add_argument("--csv-out", type=Path, default=None, help="Optional CSV report path")
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    models_dir = (repo / args.models_dir).resolve()
    model_files = sorted(p for p in models_dir.glob("*.yaml") if p.name != "default_cmy.yaml")

    rows: list[dict[str, object]] = []
    for model in model_files:
        try:
            r_path, g_path, b_path = model_qe_paths(repo, model)
            r = summarize_curve(r_path)
            g = summarize_curve(g_path)
            b = summarize_curve(b_path)
            rb_inversion = (r.peak_nm < g.peak_nm) and (b.peak_nm > g.peak_nm)
            rows.append(
                {
                    "model": model.stem,
                    "normalized_axis": int(r.normalized_axis or g.normalized_axis or b.normalized_axis),
                    "rb_inversion": int(rb_inversion),
                    "red_peak_nm": round(r.peak_nm, 2),
                    "green_peak_nm": round(g.peak_nm, 2),
                    "blue_peak_nm": round(b.peak_nm, 2),
                    "red_axis_max": round(r.wl_max, 6),
                    "green_axis_max": round(g.wl_max, 6),
                    "blue_axis_max": round(b.wl_max, 6),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model": model.stem,
                    "normalized_axis": -1,
                    "rb_inversion": -1,
                    "red_peak_nm": "ERR",
                    "green_peak_nm": "ERR",
                    "blue_peak_nm": "ERR",
                    "red_axis_max": "ERR",
                    "green_axis_max": "ERR",
                    "blue_axis_max": f"{exc}",
                }
            )

    affected_axis = sum(1 for r in rows if r["normalized_axis"] == 1)
    affected_rb = sum(1 for r in rows if r["rb_inversion"] == 1)
    print(f"Models scanned: {len(rows)}")
    print(f"Normalized/non-nm wavelength axis: {affected_axis}")
    print(f"Likely red/blue inversion: {affected_rb}")
    print("")
    print("model,axis_issue,rb_swap,red_peak_nm,green_peak_nm,blue_peak_nm")
    for r in rows:
        print(
            f'{r["model"]},{r["normalized_axis"]},{r["rb_inversion"]},'
            f'{r["red_peak_nm"]},{r["green_peak_nm"]},{r["blue_peak_nm"]}'
        )

    if args.csv_out is not None:
        out = args.csv_out if args.csv_out.is_absolute() else (repo / args.csv_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "model",
                    "normalized_axis",
                    "rb_inversion",
                    "red_peak_nm",
                    "green_peak_nm",
                    "blue_peak_nm",
                    "red_axis_max",
                    "green_axis_max",
                    "blue_axis_max",
                ],
            )
            w.writeheader()
            for row in rows:
                w.writerow(row)
        print(f"\nWrote report: {out}")


if __name__ == "__main__":
    main()
