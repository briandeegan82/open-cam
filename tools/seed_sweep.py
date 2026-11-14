"""Night wet-road seed sweep: plain mean vs ``merge_seeds`` of N independent renders against an independent reference.

Uses the renders of ``tools/firefly_experiment.py`` (``<work>/out/<variant>_s<k>.exr``, 256 spp each). Test sets are
seeds 1..N for N in ``--ns`` (nested); the reference is the plain mean of seeds ``--ref-first``.. ``--ref-first +
--ref-count - 1``, which share no samples with any test set.

Per N and method (``mean``, ``merge`` = tools/merge_seeds.py defaults, ratio 3, sigma 2, floor 0.5, run per spectral
bucket), on four bands (400-475, 475-550, 550-625, 625-700 nm):

* ``nmse``: sum (test - ref)^2 / sum ref^2 over the frame without the brightest 0.5 % of reference pixels (the
  lamps), mean over the bands; ``nmse_corr`` subtracts the reference's own noise, estimated from its two halves
  (E|A - B|^2 = 4 x the noise of the full reference for a plain mean)
* ``speckle_px``: road pixels (rows 140-270) above 3 x the reference plus a quarter of its median
* ``behind_lead``: mean of the road behind the lead car over the reference's; ``total``: image total over the reference's

    venv/bin/python tools/seed_sweep.py --work out/firefly_exp --variant path_lampreg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from firefly_experiment import ROAD_ROWS, ROIS  # noqa: E402
from merge_seeds import robust_merge  # noqa: E402

BANDS = ((400, 475), (475, 550), (550, 625), (625, 700))


def load_bands(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(H, W, K) spectral buckets in 400-700 nm and their band index (0-3) per bucket."""
    from exr_multispectral import spectral_buckets_from_exr  # noqa: PLC0415

    a, lam = spectral_buckets_from_exr(path)
    m = (lam > BANDS[0][0]) & (lam < BANDS[-1][1])
    idx = np.searchsorted([b[1] for b in BANDS], lam[m], side="right")
    return a[..., m].astype(np.float32), np.minimum(idx, len(BANDS) - 1)


def to_bands(cube: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.stack([cube[..., idx == b].sum(-1) for b in range(len(BANDS))], axis=-1).astype(np.float64)


def merge_buckets(stack: np.ndarray, chunk: int = 4) -> np.ndarray:
    """robust_merge per spectral bucket (it is per channel), a few buckets at a time to bound memory."""
    out = np.empty(stack.shape[1:], dtype=np.float64)
    for c in range(0, stack.shape[-1], chunk):
        out[..., c : c + chunk], _ = robust_merge(stack[..., c : c + chunk])
    return out


def nmse(test: np.ndarray, ref: np.ndarray, keep: np.ndarray) -> float:
    return float(np.mean([((test[..., b] - ref[..., b])[keep] ** 2).sum() / (ref[..., b][keep] ** 2).sum() for b in range(ref.shape[-1])]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", type=Path, default=REPO / "out" / "firefly_exp")
    ap.add_argument("--variant", default="path_lampreg")
    ap.add_argument("--ns", default="4,8,16,32")
    ap.add_argument("--ref-first", type=int, default=33)
    ap.add_argument("--ref-count", type=int, default=32)
    a = ap.parse_args(argv)
    ns = [int(n) for n in a.ns.split(",")]
    exr = lambda s: a.work / "out" / f"{a.variant}_s{s}.exr"  # noqa: E731

    ref_seeds = list(range(a.ref_first, a.ref_first + a.ref_count))
    half = len(ref_seeds) // 2
    sums = []
    idx = None
    for part in (ref_seeds[:half], ref_seeds[half:]):
        acc = None
        for s in part:
            cube, idx = load_bands(exr(s))
            b = to_bands(cube, idx)
            acc = b if acc is None else acc + b
        sums.append(acc / len(part))
    ref = (sums[0] + sums[1]) / 2
    ref_y = ref.sum(-1)
    keep = ref_y <= np.percentile(ref_y, 99.5)
    floor = nmse(sums[0], sums[1], keep) / 4.0
    road_ref = ref_y[ROAD_ROWS]
    y0, y1, x0, x1 = ROIS["behind_lead"]

    cubes = [load_bands(exr(s))[0] for s in range(1, max(ns) + 1)]
    rows = []
    imgs = {"reference": ref_y}
    for n in ns:
        stack = np.stack(cubes[:n])
        for method in ("mean", "merge"):
            img = to_bands(stack.mean(0) if method == "mean" else merge_buckets(stack), idx)
            y = img.sum(-1)
            e = nmse(img, ref, keep)
            rows.append({
                "n": n, "spp_total": 256 * n, "method": method, "nmse": e, "nmse_corr": e - floor,
                "speckle_px": int((y[ROAD_ROWS] > 3 * road_ref + 0.25 * np.median(road_ref)).sum()),
                "behind_lead": float(y[y0:y1, x0:x1].mean() / ref_y[y0:y1, x0:x1].mean()),
                "total": float(y.sum() / ref_y.sum()),
            })
            imgs[f"{method} N={n}"] = y
            r = rows[-1]
            print(f"N={n:2d} {method:5s} nmse={e:.5f} corr={r['nmse_corr']:.5f} speckle={r['speckle_px']:5d} "
                  f"behind_lead={r['behind_lead']:.4f} total={r['total']:.5f}", flush=True)
    res = {"variant": a.variant, "ref_seeds": [ref_seeds[0], ref_seeds[-1]], "ref_noise_nmse": floor, "rows": rows}
    (a.work / f"seed_sweep_{a.variant}.json").write_text(json.dumps(res, indent=2) + "\n")
    figure(a.work / "figs" / f"seed_sweep_{a.variant}.png", imgs, rows, ns, floor)
    return 0


def figure(out: Path, imgs: dict, rows: list[dict], ns: list[int], floor: float) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    out.parent.mkdir(exist_ok=True)
    scale = float(np.percentile(imgs["reference"][ROAD_ROWS], 99.0))
    fig = plt.figure(figsize=(15, 2.0 * (len(ns) + 1) + 0.5), facecolor="#fcfcfb")
    gs = fig.add_gridspec(len(ns) + 1, 3, width_ratios=(1, 1, 0.9))
    show = lambda ax, y, t: (ax.imshow(np.clip(y[ROAD_ROWS] / scale, 0, 1) ** (1 / 2.2), cmap="gray", interpolation="nearest"),  # noqa: E731
                             ax.set_title(t, fontsize=9, loc="left", color="#0b0b0b"), ax.axis("off"))
    show(fig.add_subplot(gs[0, 0]), imgs["reference"], "reference: mean of 32 independent seeds (8192 spp)")
    for i, n in enumerate(ns, start=1):
        for j, m in enumerate(("mean", "merge")):
            r = next(r for r in rows if r["n"] == n and r["method"] == m)
            show(fig.add_subplot(gs[i, j]), imgs[f"{m} N={n}"], f"{m}, N={n} ({256 * n} spp)  |  speckle px {r['speckle_px']}")
    ax = fig.add_subplot(gs[1:, 2])
    ax.set_facecolor("#fcfcfb")
    for m, c, mk in (("mean", "#2a78d6", "o"), ("merge", "#eb6834", "s")):
        pts = [r for r in rows if r["method"] == m]
        ax.loglog([r["n"] for r in pts], [max(r["nmse_corr"], 1e-7) for r in pts], color=c, marker=mk, lw=1.8, label=m)
    m0 = next(r for r in rows if r["method"] == "mean")
    ax.loglog(ns, [m0["nmse_corr"] * ns[0] / n for n in ns], ":", color="#52514e", lw=1, label="1/N")
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlabel("seeds N (256 spp each)", color="#52514e", fontsize=9)
    ax.set_ylabel("NMSE vs reference (reference noise removed)", color="#52514e", fontsize=9)
    ax.grid(True, color="#e4e3df", which="both")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Night wet highway (lamp-regularize 0.1), road rows 140-270: plain mean vs merge_seeds", fontsize=11, color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor="#fcfcfb")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
