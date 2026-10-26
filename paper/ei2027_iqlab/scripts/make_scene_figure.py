"""Figure of the four IQ-lab test scenes, from the scorecard's own pbrt spectral renders.

Each panel is the spectral EXR integrated against the CIE 1931 observer and shown in sRGB. The display
exposure is set per panel (stated in the caption); nothing else is done to the render. The HDR chart spans
120 dB, so it is shown as log luminance.

    venv/bin/python paper/ei2027_iqlab/scripts/make_scene_figure.py --work-dir out/iq_lab/scorecard/work
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import OpenEXR

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from colour_science import SRGB_FROM_XYZ, cmf_on_grid  # noqa: E402

HERE = Path(__file__).resolve().parents[1]

# (label, path under the scorecard work dir, display)
PANELS = [
    ("(a) HDR chart", "hdr/hdr_dr_chart_spectral.exr", "log"),
    ("(b) Diorama", "pinhole_45deg/diorama/iq_diorama_spectral_roi.exr", "linear"),
    ("(c) Skin chart (D65)", "pinhole_45deg/skin/skin_tones_D65_spectral.exr", "linear"),
    ("(d) Flare target", "pinhole_45deg/flare/veiling_glare_spectral.exr", "linear"),
]


def spectral_xyz(path: Path) -> np.ndarray:
    """``(H, W, 3)`` CIE XYZ (arbitrary scale) of a pbrt SpectralFilm EXR (``S0.<nm>`` channels)."""
    ch = OpenEXR.File(str(path)).channels()
    keys = [k for k in ch if k.startswith("S0.") and k.endswith("nm")]
    wl = np.array([float(k[3:-2].replace(",", ".")) for k in keys])
    order = np.argsort(wl)
    wl = wl[order]
    cube = np.stack([np.asarray(ch[keys[i]].pixels, np.float64) for i in order], axis=-1)
    dw = np.gradient(wl)
    return np.einsum("hwn,nc->hwc", cube, cmf_on_grid(wl).T * dw[:, None])


def srgb_encode(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def display(xyz: np.ndarray, mode: str, percentile: float) -> np.ndarray:
    if mode == "log":
        y = np.maximum(xyz[..., 1], 0.0)
        hi = float(y.max())
        v = np.clip((np.log10(np.maximum(y, hi * 1e-6)) - np.log10(hi * 1e-6)) / 6.0, 0.0, 1.0)
        return np.repeat(v[..., None], 3, axis=-1)
    rgb = xyz @ SRGB_FROM_XYZ.T
    scale = float(np.percentile(xyz[..., 1], percentile))
    return srgb_encode(rgb / max(scale, 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work-dir", type=Path, required=True, help="run_recipe_scorecard.py out-dir/work")
    ap.add_argument("--percentile", type=float, default=99.0, help="Y percentile shown at white (linear panels)")
    ap.add_argument("--out", type=Path, default=HERE / "figures" / "fig_scenes.pdf")
    args = ap.parse_args()
    fig, axes = plt.subplots(1, len(PANELS), figsize=(7.0, 1.45), dpi=300)
    for ax, (label, rel, mode) in zip(axes, PANELS, strict=True):
        ax.imshow(display(spectral_xyz(args.work_dir / rel), mode, args.percentile), interpolation="nearest")
        ax.set_title(label, fontsize=7, pad=2)
        ax.set_axis_off()
    fig.subplots_adjust(left=0.005, right=0.995, top=0.86, bottom=0.01, wspace=0.03)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
