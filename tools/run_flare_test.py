#!/usr/bin/env python3
"""Lens-flare test runner: traced-ghost energy vs field angle per coating, and EXR scoring.

* ``--ghost-sweep`` (no render needed): for a lens prescription, total two-reflection ghost energy
  relative to the primary image (``lens_ghosts.ghost_energy_fractions``) vs field angle, for each
  coating in ``--coatings`` -- plus the brightest ghost path per angle.
* ``--veiling-exr`` / ``--point-exr``: score rendered images from ``build_flare_test_scene.py``
  (after ``lens_ghosts.py``): veiling glare % per black hole, ghost peak / stray fraction per source.

    venv/bin/python tools/run_flare_test.py --ghost-sweep --out-dir out/iq_lab/flare
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import lens_ghosts as lg
import numpy as np
from iqlab import flare

REPO = Path(__file__).resolve().parent.parent
VISIBLE_NM = (450.0, 550.0, 650.0)


def ghost_sweep(
    lens_file: str, coatings: list[str], angles_deg: list[float], *, aperture_mm: float | None = 4.0, n_grid: int = 48
) -> list[dict]:
    rows = []
    for coating in coatings:
        lens = lg.load_lens(REPO / lens_file, aperture_diameter_mm=aperture_mm, coating=coating)
        for ang in angles_deg:
            res = lg.ghost_energy_fractions(lens, lg.direction_from_angles(math.radians(ang)), VISIBLE_NM, n_grid)
            ghosts = res["ghosts"]
            tot = sum(float(np.mean(g["energy"])) for g in ghosts.values())
            top = max(ghosts.items(), key=lambda kv: float(np.mean(kv[1]["energy"])), default=(None, None))
            rows.append(
                {
                    "coating": coating,
                    "field_deg": ang,
                    "total_ghost_fraction": tot,
                    "brightest_path": list(top[0]) if top[0] else None,
                    "brightest_fraction": float(np.mean(top[1]["energy"])) if top[1] else 0.0,
                    "n_paths": len(ghosts),
                }
            )
    return rows


def _load(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    from exr_multispectral import linear_rgb_from_exr  # noqa: PLC0415

    return linear_rgb_from_exr(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ghost-sweep", action="store_true")
    ap.add_argument("--lens-file", default="config/lenses/wide_22mm.dat")
    ap.add_argument("--aperture-diameter-mm", type=float, default=4.0)
    ap.add_argument("--coatings", nargs="+", default=["uncoated", "mgf2", "qhq"])
    ap.add_argument("--angles-deg", type=float, nargs="+", default=[0, 5, 10, 15, 20, 25, 30])
    ap.add_argument("--pupil-samples", type=int, default=48)
    ap.add_argument("--veiling-exr", type=Path, nargs="*", default=[])
    ap.add_argument("--point-exr", type=Path, nargs="*", default=[])
    ap.add_argument("--exclude-radius-px", type=float, default=10.0)
    ap.add_argument("--out-dir", type=Path, default=REPO / "out" / "iq_lab" / "flare")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    if args.ghost_sweep:
        report["ghost_sweep"] = ghost_sweep(
            args.lens_file,
            args.coatings,
            args.angles_deg,
            aperture_mm=args.aperture_diameter_mm,
            n_grid=args.pupil_samples,
        )
        for r in report["ghost_sweep"]:
            print(f"{r['coating']:9s} {r['field_deg']:5.1f} deg  ghost/primary {r['total_ghost_fraction']:.3e}")
    for p in args.veiling_exr:
        holes = flare.black_hole_glare(_load(p))
        report.setdefault("veiling_glare", {})[str(p)] = [
            {"x": h.x, "y": h.y, "glare_percent": h.glare_percent} for h in holes
        ]
        print(f"{p.name}: veiling glare " + ", ".join(f"{h.glare_percent:.3f}%" for h in holes))
    for p in args.point_exr:
        f = flare.point_source_flare(_load(p), exclude_radius_px=args.exclude_radius_px)
        report.setdefault("point_source", {})[str(p)] = {
            "x": f.x,
            "y": f.y,
            "stray_fraction": f.stray_fraction,
            "ghost_peak_relative": f.ghost_peak_relative,
        }
        print(f"{p.name}: stray {f.stray_fraction:.3e}, ghost peak {f.ghost_peak_relative:.3e}")
    (args.out_dir / "flare_results.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
