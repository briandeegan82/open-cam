"""ColorChecker through the full open-cam pipeline (iphone_8 recipe), D65 vs A.

Composes the pipeline previews written by tools/apply_emva_noise.py
(noisy Malvar demosaic, grey-world preview white balance, no CCM) for the
D65 and CIE A renders produced by render_scenes.sh / make_figures.sh.

Run from the repository root:
    venv/bin/python paper/ei2027/scripts/fig_colorchecker.py
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
BUILD = REPO / "paper" / "ei2027" / "build"
OUT = REPO / "paper" / "ei2027" / "figures"
CROP = (slice(120, 520), slice(190, 770))


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45))
    stats = {}
    for ax, ill, label in zip(axes, ("D65", "A"), ("CIE D65", "CIE A (2856 K)"), strict=True):
        img = iio.imread(BUILD / f"pipeline_{ill}" / "noisy_demosaic_rgb8.png")
        ax.imshow(img[CROP])
        ax.set_title(
            f"{label}: PBRT spectral $\\rightarrow$ e$^-$ $\\rightarrow$ EMVA $\\rightarrow$ GBRG $\\rightarrow$ Malvar",
            fontsize=7,
        )
        ax.axis("off")
        rs = json.loads((BUILD / f"pipeline_{ill}" / "run_stats.json").read_text())
        stats[ill] = {
            k: rs.get(k) for k in ("bayer_pattern", "demosaic", "signal_e_mean_mono", "preview_white_balance_gains_rgb")
        }
    fig.tight_layout(pad=0.3)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_colorchecker.pdf", dpi=300, bbox_inches="tight")
    (OUT / "colorchecker_summary.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
