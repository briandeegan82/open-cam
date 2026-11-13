"""Merge independent renders of one scene (different ``pbrt --seed``) and take the speckle out (display/demo renders only).

A path tracer leaves rare, very bright samples in a night render: on a wet road, the ray reflected by the
road lands where a car's clear coat mirrors a headlamp (docs/HIGHWAY_SCENES.md, "Lamp regularization"). One
render cannot tell such a sample from a small bright object. Several renders can: an object is in all of
them, a speckle in one.

For K >= 3 renders of the same scene, per pixel and per channel (every ``S0.<lambda>nm`` bucket, and
``R,G,B``)::

    cap    = ratio * median over the renders + floor * (5x5 median of that median)   (for positive medians)
    kept   = mean over the renders of min(value, cap)
    moved  = mean over the renders of max(value - cap, 0)
    merged = kept + Gaussian blur of moved (sigma pixels, same channel)

Nothing is discarded: the image total of every channel is the total of the plain mean. Light above the cap
turns from dots into a glow ``sigma`` pixels wide, which is where a converged render has it to within the
size of the highlight's reflection. The merged image is not a converged render: its noise is no longer
independent from pixel to pixel, so do not measure noise, SNR or texture MTF on it. Other channels are
averaged. A ``*.merge.json`` sidecar records the inputs and how much light was moved. K renders at spp / K
cost the same as one at spp. See docs/DENOISING.md.

    for s in 1 2 3 4; do pbrt --seed $s --outfile out/x_s$s.exr x.pbrt; done
    venv/bin/python tools/merge_seeds.py --exr-in out/x_s1.exr out/x_s2.exr out/x_s3.exr out/x_s4.exr --exr-out out/x.exr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS_WARNING = (
    "Merged seeds (tools/merge_seeds.py): outlier light is blurred; do not use for noise or texture metrics."
)
MIN_RENDERS = 3


def robust_merge(
    stack: np.ndarray, ratio: float = 3.0, sigma: float = 2.0, floor: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Merge ``stack`` (K, H, W, C) as in the module docstring; returns ``(merged, moved)``, both (H, W, C)."""
    from scipy.ndimage import gaussian_filter, median_filter  # noqa: PLC0415

    x = np.asarray(stack, dtype=np.float64)
    if x.ndim != 4 or x.shape[0] < MIN_RENDERS:
        raise ValueError(f"need a (K >= {MIN_RENDERS}, H, W, C) stack, got shape {x.shape}")
    if ratio < 1 or sigma < 0 or floor < 0:
        raise ValueError("ratio must be >= 1, sigma and floor >= 0")
    ref = np.median(x, axis=0)
    level = np.stack([median_filter(ref[..., c], size=5, mode="nearest") for c in range(ref.shape[-1])], axis=-1)
    # ref + (ratio - 1) |ref|: the same cap for positive values, and still above the median where the
    # median is negative (sensor RGB of a saturated colour), so only values above the cap move.
    cap = ref + (ratio - 1.0) * np.abs(ref) + floor * np.abs(level)
    kept = np.minimum(x, cap[None])
    moved = (x - kept).mean(axis=0)
    if sigma > 0:
        glow = np.stack(
            [_blur_conserving(moved[..., c], sigma, gaussian_filter) for c in range(moved.shape[-1])], axis=-1
        )
    else:
        glow = moved
    return kept.mean(axis=0) + glow, moved


def _blur_conserving(plane: np.ndarray, sigma: float, gaussian_filter) -> np.ndarray:
    """Gaussian blur that keeps the plane's total (light blurred past the frame edge is folded back)."""
    total = float(plane.sum())
    if total <= 0:
        return plane
    out = gaussian_filter(plane, sigma, mode="reflect")
    s = float(out.sum())
    return out * (total / s) if s > 0 else plane


def merge_channels(
    renders: list[dict[str, np.ndarray]], ratio: float = 3.0, sigma: float = 2.0, floor: float = 0.5
) -> tuple[dict[str, np.ndarray], dict]:
    """Merge channel dicts (name -> HxW) of K renders. Spectral buckets and R,G,B are merged robustly."""
    from exr_multispectral import parse_s0_wavelength_nm  # noqa: PLC0415

    names = list(renders[0])
    for r in renders[1:]:
        if list(r) != names:
            raise ValueError("renders have different channels")
    spectral = [n for n in names if parse_s0_wavelength_nm(n) is not None]
    rgb = [n for n in ("R", "G", "B") if n in names]
    out: dict[str, np.ndarray] = {}
    report: dict = {}
    for label, group in (("spectral", spectral), ("rgb", rgb)):
        if not group:
            continue
        stack = np.stack([np.stack([r[n] for n in group], axis=-1) for r in renders])
        merged, moved = robust_merge(stack, ratio, sigma, floor)
        plain = stack.mean(axis=0, dtype=np.float64)
        total = float(plain.sum())
        report[f"{label}_energy_ratio"] = float(merged.sum() / total) if total else 1.0
        report[f"{label}_moved_fraction"] = float(moved.sum() / total) if total else 0.0
        report[f"{label}_pixels_capped_fraction"] = float(np.mean(np.any(moved > 0, axis=-1)))
        for i, n in enumerate(group):
            out[n] = merged[..., i].astype(np.float32)
    for n in names:
        if n not in out:
            out[n] = np.mean([r[n] for r in renders], axis=0, dtype=np.float64).astype(np.float32)
    return {n: out[n] for n in names}, report


def merge_exrs(
    exr_in: list[Path], exr_out: Path, *, ratio: float = 3.0, sigma: float = 2.0, floor: float = 0.5
) -> dict:
    """Merge the EXRs ``exr_in`` into ``exr_out`` and write the ``.merge.json`` sidecar."""
    from exr_multispectral import read_separate_exr_channels, write_separate_channels_exr  # noqa: PLC0415

    if len(exr_in) < MIN_RENDERS:
        raise ValueError(f"need at least {MIN_RENDERS} renders to tell a speckle from an object, got {len(exr_in)}")
    renders = [read_separate_exr_channels(Path(p)) for p in exr_in]
    out, report = merge_channels(renders, ratio, sigma, floor)
    exr_out = Path(exr_out)
    write_separate_channels_exr(exr_out, out)
    report.update(
        {
            "merged_seeds": True,
            "warning": METRICS_WARNING,
            "n_renders": len(exr_in),
            "ratio": ratio,
            "sigma_px": sigma,
            "floor": floor,
            "exr_in": [str(p) for p in exr_in],
            "exr_out": str(exr_out),
        }
    )
    Path(f"{exr_out}.merge.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exr-in", type=Path, nargs="+", required=True, help="renders of one scene, different seeds")
    ap.add_argument("--exr-out", type=Path, required=True)
    ap.add_argument("--ratio", type=float, default=3.0, help="cap = ratio x median over the renders (+ floor)")
    ap.add_argument(
        "--sigma", type=float, default=2.0, help="Gaussian sigma [px] the light above the cap is spread with"
    )
    ap.add_argument("--floor", type=float, default=0.5, help="cap floor, as a fraction of the 5x5 median level")
    args = ap.parse_args(argv)
    report = merge_exrs(args.exr_in, args.exr_out, ratio=args.ratio, sigma=args.sigma, floor=args.floor)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
