"""Systematic sweep of the OIDN denoiser parameters against a converged reference (docs/DENOISE_SWEEP.md).

One-at-a-time variations around a base configuration (guides, guide_spp, quality, firefly_ratio, blend, input spp)
on the IQ-lab diorama, scored on the cards' ROIs from ``diorama.json``:

* ``flat_noise``  : std(Y - Gaussian_3px(Y)) / mean(Y) on the flat back wall: noise left in the image, measured without the
                    reference because even 2048 spp is too noisy to serve as one there (lower is better)
* ``texture_rmse``: rms(Y - reference) / mean(reference) in the texture ROI: detail lost plus noise left, together
                    (the reference's own noise sets a floor, so only compare settings with each other)
* ``tail_err``    : 99.9th percentile of |denoised - reference| / reference over the lit pixels (fireflies, ringing)
* ``texture_gain``: regression slope of the 1-3 px band (DoG) of the dead-leaves card on the reference's band
                    (1 = detail kept, <1 = blurred; noise does not raise it)
* ``edge_gain``   : the same on the slanted-edge card
* ``global_err``  : median |denoised - reference| / reference over the lit pixels

``blend`` is applied after one OIDN run (``blend_planes``), so that axis costs nothing. Only R,G,B are denoised
here (one OIDN call per variant; the spectral buckets blur the same way). Writes ``results.json`` and three
figures (overview, ROI crops, metric curves) into ``<work>/figs``.

    venv/bin/python tools/denoise_sweep.py --work out/denoise_sweep
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import denoise_spectral_exr as dn  # noqa: E402
from exr_multispectral import read_separate_exr_channels  # noqa: E402

BASE = {"guides": "albedo_normal", "guide_spp": 16, "quality": "high", "firefly_ratio": 20.0, "blend": 0.0}
AXES = {
    "blend": [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0],
    "guide_spp": [1, 4, 16, 64],
    "guides": ["none", "albedo", "albedo_normal"],
    "quality": ["fast", "balanced", "high"],
    "firefly_ratio": [0.0, 5.0, 20.0, 100.0],
}
# Light-mode categorical slots 1,2,3,5,7 (dataviz palette), one per axis; markers carry identity as well.
COLOURS = {"blend": "#2a78d6", "guide_spp": "#eb6834", "guides": "#1baf7a", "quality": "#e87ba4", "firefly_ratio": "#6250d6"}
MARKERS = {"blend": "o", "guide_spp": "s", "guides": "^", "quality": "D", "firefly_ratio": "v"}
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def crop(img: np.ndarray, roi: list[int]) -> np.ndarray:
    x0, y0, x1, y1 = roi
    return img[y0:y1, x0:x1]


def gauss(img: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter  # noqa: PLC0415

    return gaussian_filter(img, sigma, mode="nearest")


def band(img: np.ndarray) -> np.ndarray:
    """1-3 px detail band of a full luminance image (cropped afterwards so ROI edges do not ring)."""
    return gauss(img, 1.0) - gauss(img, 2.0)


def gain(test_band: np.ndarray, ref_band: np.ndarray) -> float:
    r = ref_band - ref_band.mean()
    t = test_band - test_band.mean()
    return float((t * r).sum() / max((r * r).sum(), 1e-20))


def score(y: np.ndarray, y_ref: np.ndarray, rois: dict) -> dict:
    lit = y_ref > 0.05 * np.median(y_ref)
    rel = np.abs(y[lit] - y_ref[lit]) / y_ref[lit]
    flat = crop(y, rois["flat"])
    flat_hp = flat - crop(gauss(y, 3.0), rois["flat"])
    yb, rb = band(y), band(y_ref)
    return {
        "flat_noise": float(np.std(flat_hp) / flat.mean()),
        "texture_gain": gain(crop(yb, rois["texture"]), crop(rb, rois["texture"])),
        "edge_gain": gain(crop(yb, rois["edge"]), crop(rb, rois["edge"])),
        "texture_rmse": float(np.sqrt(np.mean((crop(y, rois["texture"]) - crop(y_ref, rois["texture"])) ** 2)) / crop(y_ref, rois["texture"]).mean()),
        "global_err": float(np.median(rel)),
        "tail_err": float(np.percentile(rel, 99.9)),
    }


def load_rgb(path: Path) -> np.ndarray:
    ch = read_separate_exr_channels(path)
    return np.stack([ch["R"], ch["G"], ch["B"]], axis=2).astype(np.float32)


def run_oidn(rgb, runner, firefly_ratio):
    ch = {c: rgb[..., i] for i, c in enumerate("RGB")}
    out, _ = dn.denoise_channels(ch, runner, 0.0, firefly_ratio)
    return np.stack([out[c] for c in "RGB"], axis=2)


def variants() -> list[tuple[str, str, dict]]:
    """``(axis, label, params)`` for every point of the sweep; the base point appears once per axis."""
    return [(axis, str(v), {**BASE, axis: v}) for axis, vals in AXES.items() for v in vals]


def setup(work: Path, ref_name: str, oidn: Path | None, scene: Path | None = None, rois: dict | None = None):
    """Load ROIs and the reference; returns ``(rois, ref, denoise)`` where ``denoise(rgb, params)`` runs OIDN (blend 0).

    Default scene: the IQ-lab diorama in ``<work>/scene`` with ROIs from its ``diorama.json``; otherwise pass the pbrt
    file and ``{"flat": xyxy, "texture": xyxy, "edge": xyxy}`` (raster pixels).
    """
    pbrt = REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"
    if scene is None:
        scene_dir = work / "scene"
        scene = scene_dir / "iq_diorama.pbrt"
        info = json.loads((scene_dir / "diorama.json").read_text())
        rois = {
            "flat": info["shadow_reference"]["roi_xyxy"],
            "texture": info["cards"]["dead_leaves"]["roi_xyxy"],
            "edge": info["cards"]["slanted_edge"]["roi_xyxy"],
        }
    elif rois is None:
        raise ValueError("--scene needs --rois")
    ref = load_rgb(work / ref_name)
    h, w = ref.shape[:2]
    binary = dn.find_oidn(oidn)
    guide_cache: dict[int, tuple] = {}

    def guides_for(gspp):
        if gspp not in guide_cache:
            guide_cache[gspp] = dn.render_guides(scene, pbrt, w, h, gspp)
        return guide_cache[gspp]

    def denoise(rgb, p):
        alb = nrm = None
        if p["guides"] != "none":
            alb, nrm = guides_for(p["guide_spp"])
            nrm = nrm if p["guides"] == "albedo_normal" else None
        runner = dn.OidnRunner(binary, alb, nrm, quality=p["quality"])
        try:
            return run_oidn(rgb, runner, p["firefly_ratio"])
        finally:
            runner.close()

    return rois, ref, denoise


def sweep(work: Path, noisy_name: str, ref_name: str, oidn: Path | None, scene=None, rois=None) -> dict:
    rois, ref, denoise = setup(work, ref_name, oidn, scene, rois)
    y_ref = luminance(ref)
    images: dict[str, np.ndarray] = {"reference": ref}
    rows: list[dict] = []

    def record(spp_name, axis, label, p, noisy, den):
        img = dn.blend_planes(noisy, den, p["blend"])
        key = f"{spp_name}|{axis}={label}"
        images[key] = img
        rows.append({"input": spp_name, "axis": axis, "value": label, "params": p, **score(luminance(img), y_ref, rois)})
        print(f"{key:40s} " + " ".join(f"{k}={v:.3f}" for k, v in rows[-1].items() if k.endswith(("noise", "err", "gain"))), flush=True)

    for spp_name, path in (("spp256", work / noisy_name.format(spp=256)), ("spp64", work / noisy_name.format(spp=64))):
        noisy = load_rgb(path)
        images[f"{spp_name}|noisy"] = noisy
        rows.append({"input": spp_name, "axis": "noisy", "value": "-", "params": None, **score(luminance(noisy), y_ref, rois)})
        cache: dict[tuple, np.ndarray] = {}
        for axis, label, p in variants():
            if spp_name == "spp64" and axis != "blend":
                continue
            k = tuple((a, p[a]) for a in ("guides", "guide_spp", "quality", "firefly_ratio"))
            if k not in cache:
                t0 = time.time()
                cache[k] = denoise(noisy, p)
                print(f"  OIDN {spp_name} {dict(k)}: {time.time() - t0:.1f}s", flush=True)
            record(spp_name, axis, label, p, noisy, cache[k])
    return {"rois": rois, "rows": rows, "images": images}


# ---------------------------------------------------------------- figures


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8)


def srgb(lin: np.ndarray, scale: float) -> np.ndarray:
    x = np.clip(lin / scale, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def fig_overview(res: dict, out: Path, best_blend: str):
    import matplotlib.pyplot as plt  # noqa: PLC0415

    im = res["images"]
    panels = [
        ("reference (2048 spp)", im["reference"]),
        ("noisy 256 spp", im["spp256|noisy"]),
        ("OIDN base (blend 0)", im["spp256|blend=0.0"]),
        (f"OIDN base + blend {best_blend}", im[f"spp256|blend={best_blend}"]),
    ]
    scale = float(np.percentile(luminance(im["reference"]), 99.0))
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.2), facecolor=SURFACE)
    for ax, (t, img) in zip(axs.ravel(), panels):
        ax.imshow(srgb(img, scale))
        ax.set_title(t, color=INK, fontsize=11, loc="left")
        ax.axis("off")
        for name, c in (("texture", "#eb6834"), ("edge", "#2a78d6"), ("flat", "#1baf7a")):
            x0, y0, x1, y1 = res["rois"][name]
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=c, lw=1.4))
    fig.suptitle("Full frame (boxes: texture / edge / flat ROIs)", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def fig_crops(res: dict, out: Path, roi_name: str = "texture", zoom: int = 4):
    import matplotlib.pyplot as plt  # noqa: PLC0415

    im, roi = res["images"], res["rois"][roi_name]
    scale = float(np.percentile(luminance(crop(im["reference"], roi)), 99.5))
    rows = [("blend, 256 spp", "spp256", "blend"), ("blend, 64 spp", "spp64", "blend")] + [
        (f"{a}, 256 spp", "spp256", a) for a in ("guide_spp", "guides", "quality", "firefly_ratio")
    ]
    ncol = 2 + max(len(AXES[a]) for _, _, a in rows)
    fig, axs = plt.subplots(len(rows), ncol, figsize=(1.9 * ncol, 2.05 * len(rows)), facecolor=SURFACE, squeeze=False)
    for r, (title, spp, axis) in enumerate(rows):
        entries = [("reference", im["reference"]), (f"{spp} noisy", im[f"{spp}|noisy"])]
        entries += [(f"{axis}={v}", im[f"{spp}|{axis}={v}"]) for v in map(str, AXES[axis])]
        for c in range(ncol):
            ax = axs[r, c]
            ax.axis("off")
            if c < len(entries):
                lab, img = entries[c]
                ax.imshow(srgb(crop(img, roi), scale), interpolation="nearest")
                ax.set_title(lab, fontsize=7.5, color=INK2, pad=2)
        axs[r, 0].text(-0.06, 0.5, title, transform=axs[r, 0].transAxes, rotation=90, ha="right", va="center", fontsize=8.5, color=INK)
    # reference + noisy occupy columns 0,1; shift so every row starts the same way
    fig.suptitle(f"{roi_name} ROI crops ({zoom}x nearest): reference, noisy input, then the swept values", color=INK, fontsize=11)
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def fig_tradeoff(res: dict, out: Path):
    import matplotlib.pyplot as plt  # noqa: PLC0415

    rows = res["rows"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.4), facecolor=SURFACE)
    for ax, (gk, gl) in zip(axs, (("texture_gain", "texture detail kept (dead-leaves, 1 = reference)"), ("edge_gain", "edge detail kept (slanted edge, 1 = reference)"))):
        _style(ax)
        for spp, ls, alpha in (("spp256", "-", 1.0), ("spp64", "--", 0.55)):
            pts = [r for r in rows if r["input"] == spp and r["axis"] == "blend"]
            ax.plot([r["flat_noise"] for r in pts], [r[gk] for r in pts], ls, color=COLOURS["blend"], marker="o", ms=5, lw=1.8, alpha=alpha, label=f"blend sweep, {spp[3:]} spp")
            for r in pts:
                ax.annotate(r["value"], (r["flat_noise"], r[gk]), textcoords="offset points", xytext=(4, 4), fontsize=7, color=INK2, alpha=alpha)
        for axis in ("guide_spp", "guides", "quality", "firefly_ratio"):
            pts = [r for r in rows if r["input"] == "spp256" and r["axis"] == axis]
            ax.plot([r["flat_noise"] for r in pts], [r[gk] for r in pts], color=COLOURS[axis], marker=MARKERS[axis], ms=6, lw=0, mec=SURFACE, mew=1, label=axis)
            base = next(r for r in rows if r["input"] == "spp256" and r["axis"] == "blend" and r["value"] == "0.0")
            for r in pts:
                if abs(r[gk] - base[gk]) > 0.02 or r["flat_noise"] > 0.01:  # skip labels piled on the base point
                    ax.annotate(f'{axis}={r["value"]}', (r["flat_noise"], r[gk]), textcoords="offset points", xytext=(5, -9), fontsize=7, color=INK2)
        for spp, m in (("spp256", "x"), ("spp64", "+")):
            n = next(r for r in rows if r["input"] == spp and r["axis"] == "noisy")
            ax.plot(n["flat_noise"], n[gk], m, color=INK, ms=8, label=f"noisy input {spp[3:]} spp")
        ax.set_xlabel("noise left on the flat wall (std / mean; lower = cleaner)", color=INK2, fontsize=9)
        ax.set_ylabel(gl, color=INK2, fontsize=9)
        ax.axhline(1.0, color=INK2, lw=0.8, ls=":")
    axs[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("Noise left vs detail kept: up and to the left is better", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def fig_axes(res: dict, out: Path):
    import matplotlib.pyplot as plt  # noqa: PLC0415

    rows = res["rows"]
    metrics = [("flat_noise", "flat-wall noise (lower better)"), ("tail_err", "99.9th pct rel. error (lower better)"), ("texture_gain", "texture detail kept"), ("edge_gain", "edge detail kept")]
    fig, axs = plt.subplots(len(metrics), len(AXES), figsize=(3.0 * len(AXES), 2.6 * len(metrics)), facecolor=SURFACE, squeeze=False, sharey="row")
    for j, axis in enumerate(AXES):
        for i, (mk, ml) in enumerate(metrics):
            ax = axs[i, j]
            _style(ax)
            for spp, ls in (("spp256", "-"), ("spp64", "--")):
                pts = [r for r in rows if r["input"] == spp and r["axis"] == axis]
                if not pts:
                    continue
                xs = list(range(len(pts))) if axis != "blend" else [float(r["value"]) for r in pts]
                ax.plot(xs, [r[mk] for r in pts], ls, color=COLOURS[axis], marker=MARKERS[axis], ms=5, lw=1.8, label=spp[3:] + " spp")
            if axis != "blend":
                ax.set_xticks(range(len(AXES[axis])))
                ax.set_xticklabels([str(v) for v in AXES[axis]], fontsize=8)
            if mk not in ("flat_noise", "tail_err"):
                ax.axhline(1.0, color=INK2, lw=0.8, ls=":")
            if i == 0:
                ax.set_title(axis, color=INK, fontsize=10, loc="left")
            if j == 0:
                ax.set_ylabel(ml, color=INK2, fontsize=8.5)
    axs[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("One parameter at a time around the base (albedo_normal, guide_spp 16, high, firefly 20, blend 0)", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


GRID_AXES = {"guides": ["albedo", "albedo_normal"], "guide_spp": [16, 64], "blend": [0.0, 0.1, 0.2, 0.35, 0.5]}
GRID_STYLE = {("albedo", 16): ("#2a78d6", "o", "-"), ("albedo", 64): ("#eb6834", "s", "-"),
              ("albedo_normal", 16): ("#2a78d6", "o", "--"), ("albedo_normal", 64): ("#eb6834", "s", "--")}


def grid(work: Path, noisy_name: str, ref_name: str, oidn: Path | None, scene=None, rois=None) -> dict:
    """guides x guide_spp x blend on the 256 spp input; one OIDN run per (guides, guide_spp), blend is post-hoc."""
    rois, ref, denoise = setup(work, ref_name, oidn, scene, rois)
    y_ref = luminance(ref)
    noisy = load_rgb(work / noisy_name.format(spp=256))
    rows: list[dict] = [{"guides": "noisy", "guide_spp": None, "blend": None, **score(luminance(noisy), y_ref, rois)}]
    for g in GRID_AXES["guides"]:
        for gs in GRID_AXES["guide_spp"]:
            den = denoise(noisy, {**BASE, "guides": g, "guide_spp": gs})
            for b in GRID_AXES["blend"]:
                img = dn.blend_planes(noisy, den, b)
                rows.append({"guides": g, "guide_spp": gs, "blend": b, **score(luminance(img), y_ref, rois)})
                print(f"{g:14s} gspp={gs:<3d} blend={b:<4} " + " ".join(f"{k}={v:.3f}" for k, v in rows[-1].items() if k.endswith(("noise", "err", "gain"))), flush=True)
    return {"rois": rois, "rows": rows}


def fig_grid(res: dict, out: Path):
    import matplotlib.pyplot as plt  # noqa: PLC0415

    rows = [r for r in res["rows"] if r["guides"] != "noisy"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), facecolor=SURFACE)
    for ax, (mk, ml) in zip(axs, (("texture_gain", "texture detail kept (1 = reference)"), ("edge_gain", "edge detail kept (1 = reference)"), ("tail_err", "99.9th pct rel. error (lower better)"))):
        _style(ax)
        for (g, gs), (c, m, ls) in GRID_STYLE.items():
            pts = [r for r in rows if r["guides"] == g and r["guide_spp"] == gs]
            ax.plot([r["flat_noise"] for r in pts], [r[mk] for r in pts], ls, color=c, marker=m, ms=5, lw=1.8, mec=SURFACE, label=f"{g}, guide_spp {gs}")
            for r in pts[::2]:
                ax.annotate(f'{r["blend"]}', (r["flat_noise"], r[mk]), textcoords="offset points", xytext=(4, 4), fontsize=7, color=INK2)
        ax.set_xlabel("noise left on the flat wall (std / mean; lower = cleaner)", color=INK2, fontsize=9)
        ax.set_ylabel(ml, color=INK2, fontsize=9)
    axs[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("guides x guide_spp x blend (256 spp input; labels = blend 0, 0.2, 0.5)", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", type=Path, default=REPO / "out" / "denoise_sweep")
    ap.add_argument("--noisy", default="noisy_spp{spp}.exr", help="noisy render name pattern in --work")
    ap.add_argument("--reference", default="reference_spp2048.exr")
    ap.add_argument("--oidn", type=Path, default=None)
    ap.add_argument("--best-blend", default="0.2", help="blend value shown in the overview figure")
    ap.add_argument("--grid", action="store_true", help="run the guides x guide_spp x blend grid instead of the one-at-a-time sweep")
    ap.add_argument("--scene", type=Path, default=None, help="pbrt scene of the inputs (default: the diorama in --work/scene)")
    ap.add_argument("--rois", default=None, help='JSON {"flat": [x0,y0,x1,y1], "texture": [...], "edge": [...]}; needed with --scene')
    ap.add_argument("--tag", default="", help="suffix for results/figure names, e.g. _day")
    a = ap.parse_args(argv)
    rois = json.loads(a.rois) if a.rois else None
    if a.grid:
        res = grid(a.work, a.noisy, a.reference, a.oidn, a.scene, rois)
        (a.work / "figs").mkdir(exist_ok=True)
        (a.work / f"results_grid{a.tag}.json").write_text(json.dumps({"axes": GRID_AXES, **res}, indent=2) + "\n")
        fig_grid(res, a.work / "figs" / f"grid{a.tag}.png")
        return 0
    res = sweep(a.work, a.noisy, a.reference, a.oidn, a.scene, rois)
    figs = a.work / "figs"
    figs.mkdir(exist_ok=True)
    (a.work / f"results{a.tag}.json").write_text(json.dumps({"base": BASE, "axes": AXES, "rois": res["rois"], "rows": res["rows"]}, indent=2) + "\n")
    fig_overview(res, figs / f"overview{a.tag}.png", a.best_blend)
    fig_crops(res, figs / f"crops_texture{a.tag}.png", "texture")
    fig_crops(res, figs / f"crops_edge{a.tag}.png", "edge")
    fig_tradeoff(res, figs / f"tradeoff{a.tag}.png")
    fig_axes(res, figs / f"axes{a.tag}.png")
    print(f"wrote {a.work / f'results{a.tag}.json'} and figures in {figs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
