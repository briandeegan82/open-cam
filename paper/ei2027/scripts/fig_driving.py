"""Driving-scene figure: four highway renders from render_driving.sh through the sensor model.

Each spectral EXR goes through tools/apply_emva_noise.py (per-render auto-exposure: 99th percentile
at 80 % of full well). Visible panels use the default research camera and show its noisy demosaiced
sRGB preview; the SWIR panel uses the representative InGaAs recipe and shows its noisy grey preview. The
wet panels use the Open Image Denoise output of render_driving.sh (the pbrt render is denoised before
the sensor model, so the sensor noise is still simulated). Run from the repository root after render_driving.sh:
    venv/bin/python paper/ei2027/scripts/fig_driving.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
BUILD = REPO / "paper" / "ei2027" / "build" / "driving"
OUT = REPO / "paper" / "ei2027" / "figures"
# name: (title, recipe, preview PNG, render)
PANELS = {
    "day_dry": ("(a) day, dry, visible", "default", "noisy_demosaic_rgb8.png", "highway_spectral.exr"),
    "day_wet": (
        "(b) day, wet road, puddles, spray, visible",
        "default",
        "noisy_demosaic_rgb8.png",
        "highway_spectral_denoised.exr",
    ),
    "swir_900_1600": (
        "(c) day, dry, SWIR 900-1600 nm",
        "swir_ingaas_vga_15um",
        "noisy_rgb8.png",
        "highway_spectral.exr",
    ),
    "night_wet": (
        "(d) night, wet road, puddles, visible",
        "default",
        "noisy_demosaic_rgb8.png",
        "highway_spectral_denoised.exr",
    ),
}


def main() -> None:
    stats = {}
    for name, (_, recipe, _, exr) in PANELS.items():
        d = BUILD / name
        subprocess.run(
            [
                sys.executable,
                str(REPO / "tools" / "apply_emva_noise.py"),
                "--repo-root",
                str(REPO),
                "--camera-model-config",
                str(REPO / "config" / "camera_recipes" / f"{recipe}.yaml"),
                "--linear-exr",
                str(d / "out" / exr),
                "--scene-manifest-json",
                str(d / "highway_manifest.json"),
                "--auto-exposure",
                "--target-fullwell-percentile",
                "99",
                "--target-fullwell-fraction",
                "0.8",
                "--raw-out",
                str(d / "raw.raw16"),
                "--seed",
                "0",
            ],  # fmt: skip
            check=True,
            cwd=REPO,
            stdout=subprocess.DEVNULL,
        )
        s = json.loads((d / "raw_png" / "run_stats.json").read_text())
        stats[name] = {"recipe": recipe, "render": exr} | {
            k: s.get(k) for k in ("exposure_scale_e_per_unit", "signal_e_mean_mono", "demosaic")
        }
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 2.25 * 2))
    for ax, (name, (title, _, png, _)) in zip(axes.ravel(), PANELS.items(), strict=True):
        ax.imshow(mpimg.imread(BUILD / name / "raw_png" / png), cmap="gray")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.tight_layout(pad=0.2, h_pad=0.4, w_pad=0.3)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_driving.jpg", dpi=300, pil_kwargs={"quality": 88})
    (OUT / "driving_summary.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
