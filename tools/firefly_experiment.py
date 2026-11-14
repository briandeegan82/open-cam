"""Night wet-road firefly experiment: integrator and lamp-size variants against the unbiased path tracer.

Variants of one night wet highway scene (480x270, ``--lamp-regularize 0``), all with the same samples per pixel:

* ``path``          : unbiased path tracer, point (goniometric) headlamps: the reference estimator
* ``path_lampreg``  : ``"float lampregularize" [0.1]`` (the builder's night default; biased)
* ``bdpt``          : bidirectional path tracer (unbiased)
* ``sphere_3cm``    : the two ego low beams as emitting spheres of 3 cm radius with the same beam
* ``sphere_10cm``   : the same with 10 cm (the scene changes: a real lamp has a size, a point light has none)

The sphere lamps are diffuse area lights with an ``angularfilename`` map (pbrt patch 0016): the low-beam
goniometric map resampled into area-light space (goniometric lights swap y and z, area lights do not),
divided by its peak, with ``scale`` = peak cd / (pi r^2), so the intensity in every direction equals the
point lamp's. A sphere's projected area is pi r^2 from every side, so no cosine correction is needed.

Measured (band sum 400-700 nm, as out/wet_night/scripts/fireflies.py):

* road part of the frame (rows 140-270) over the seeds: median per-pixel variance, outlier pixels per render
  (above 3x the per-pixel median over the seeds plus a quarter of the frame median), render time
* ``efficiency`` = median variance x render time, relative to ``path`` (lower = less noise for the same time)
* ROI means: road behind the lead car (where the speckle is) and the ego-lit near road (checks the sphere lamps)
* high-spp crop renders of the road behind the lead car (mean +- standard error over seeds) for the bias

    venv/bin/python tools/firefly_experiment.py --work out/firefly_exp build
    venv/bin/python tools/firefly_experiment.py --work out/firefly_exp render
    venv/bin/python tools/firefly_experiment.py --work out/firefly_exp crops
    venv/bin/python tools/firefly_experiment.py --work out/firefly_exp analyze
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from pbrt_retry import run_pbrt  # noqa: E402

PBRT = REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"
# every render here takes < 1 min; pbrt now and then hangs (tools/pbrt_retry.py)
RENDER_TIMEOUT_S = 600
VARIANTS = ("path", "path_lampreg", "bdpt", "sphere_3cm", "sphere_10cm")
SPHERE_RADIUS = {"sphere_3cm": 0.03, "sphere_10cm": 0.10}
# ROIs at 480x270 as (y0, y1, x0, x1); behind_lead as in docs/HIGHWAY_SCENES.md
ROIS = {"behind_lead": (168, 215, 205, 290), "near_road": (230, 268, 100, 380)}
ROAD_ROWS = slice(140, 270)
ANGULAR_MAP = "textures/lowbeam_led_angular.exr"


def build(work: Path, spp: int) -> None:
    subprocess.run(
        [sys.executable, str(REPO / "tools" / "build_highway_scene.py"), "--xres", "480", "--yres", "270",
         "--pixelsamples", str(spp), "--time-of-day", "night", "--road-wetness", "wet", "--puddles",
         "--lamp-regularize", "0", "--out-dir", str(work)],
        check=True,
    )
    base = (work / "highway.pbrt").read_text()
    integ = re.search(r'^Integrator "path" "integer maxdepth" \[(\d+)\]\n', base, re.M)
    if not integ:
        raise RuntimeError("expected the builder's unbiased path Integrator line")
    depth = integ.group(1)
    peak = write_angular_map(work)
    for name in VARIANTS:
        txt = base
        if name == "path_lampreg":
            txt = txt.replace(integ.group(0), f'Integrator "path" "integer maxdepth" [{depth}] "float lampregularize" [0.1]\n')
        elif name == "bdpt":
            txt = txt.replace(integ.group(0), f'Integrator "bdpt" "integer maxdepth" [{depth}]\n')
        elif name in SPHERE_RADIUS:
            txt = sphere_ego_lamps(txt, SPHERE_RADIUS[name], peak)
        (work / f"highway_{name}.pbrt").write_text(txt)
    print(f"wrote {len(VARIANTS)} variants in {work} (low-beam peak {peak:.0f} cd)")


def write_angular_map(work: Path) -> float:
    """Resample the ego low-beam goniometric map into area-light space; returns its peak [cd]."""
    from exr_multispectral import read_separate_exr_channels  # noqa: PLC0415
    from highway_night import map_intensity, write_y_exr  # noqa: PLC0415
    from highway_sky import equal_area_directions  # noqa: PLC0415

    gonio = read_separate_exr_channels(work / "textures" / "lowbeam_led.exr")["Y"].astype(np.float64)
    d = equal_area_directions(gonio.shape[0])
    # pbrt's GoniometricLight looks its image up at (x, z, y) of the light-space direction
    m = map_intensity(gonio, d[..., [0, 2, 1]])
    peak = float(gonio.max())
    write_y_exr(work / ANGULAR_MAP, (m / peak).astype(np.float32))
    return peak


def sphere_ego_lamps(txt: str, radius: float, peak_cd: float) -> str:
    """Replace the two ego low beams (the first goniometric lights after the vehicle-lamps marker)."""
    head, sep, tail = txt.partition("# ---- vehicle lamps")
    if not sep:
        raise RuntimeError("vehicle-lamps marker not found")
    pat = re.compile(r'^(\s*)LightSource "goniometric" "spectrum I" "(spd/lamp_led_headlamp\.spd)" "string filename" "textures/lowbeam_led\.exr" "float scale" \[1\]\n', re.M)
    scale = peak_cd / (np.pi * radius**2)

    def repl(m):
        ind, spd = m.group(1), m.group(2)
        return (f'{ind}AreaLightSource "diffuse" "spectrum L" "{spd}" "float scale" [{scale:.6g}] '
                f'"string angularfilename" "{ANGULAR_MAP}"\n{ind}Shape "sphere" "float radius" [{radius}]\n')

    tail, n = pat.subn(repl, tail, count=2)
    if n != 2:
        raise RuntimeError(f"expected 2 ego low beams, replaced {n}")
    return head + sep + tail


def _render(work: Path, name: str, seed: int, out: str, spp: int | None = None, crop: str | None = None) -> float:
    cmd = [str(PBRT), "--quiet", "--seed", str(seed), "--outfile", out]
    if spp:
        cmd += ["--spp", str(spp)]
    if crop:
        cmd += ["--cropwindow", crop]
    t0 = time.time()
    run_pbrt(cmd + [f"highway_{name}.pbrt"], cwd=work, timeout=RENDER_TIMEOUT_S)
    return time.time() - t0


def render(work: Path, seeds: int, variants) -> None:
    (work / "out").mkdir(exist_ok=True)
    times = _load(work / "times.json")
    for name in variants:
        for s in range(1, seeds + 1):
            key = f"{name}_s{s}"
            if (work / "out" / f"{key}.exr").exists() and key in times:
                continue
            times[key] = _render(work, name, s, f"out/{key}.exr")
            print(f"{key}: {times[key]:.1f}s", flush=True)
            _save(work / "times.json", times)


def crop_window() -> str:
    y0, y1, x0, x1 = ROIS["behind_lead"]
    return f"{x0 / 480:.6f},{x1 / 480:.6f},{y0 / 270:.6f},{y1 / 270:.6f}"


def crops(work: Path, seeds: int, spp: int, variants) -> None:
    (work / "out").mkdir(exist_ok=True)
    times = _load(work / "times.json")
    for name in variants:
        for s in range(1, seeds + 1):
            key = f"crop_{name}_spp{spp}_s{s}"
            if (work / "out" / f"{key}.exr").exists() and key in times:
                continue
            times[key] = _render(work, name, 100 + s, f"out/{key}.exr", spp, crop_window())
            print(f"{key}: {times[key]:.1f}s", flush=True)
            _save(work / "times.json", times)


def band_sum(path: Path) -> np.ndarray:
    from exr_multispectral import spectral_buckets_from_exr  # noqa: PLC0415

    a, lam = spectral_buckets_from_exr(path)
    m = (lam > 400) & (lam < 700)
    return a[..., m].sum(-1).astype(np.float64)


def analyze(work: Path, seeds: int, crop_spp: int) -> dict:
    times = _load(work / "times.json")
    imgs = {n: np.array([band_sum(work / "out" / f"{n}_s{s}.exr") for s in range(1, seeds + 1)]) for n in VARIANTS
            if all((work / "out" / f"{n}_s{s}.exr").exists() for s in range(1, seeds + 1))}
    ref_mean = imgs["path"].mean(0)
    keep = ref_mean <= np.percentile(ref_mean, 99.5)  # whole-frame comparison without the lamps themselves
    rows = {}
    for n, L in imgs.items():
        road = L[:, ROAD_ROWS]
        med = np.median(road, 0)
        var = road.var(0, ddof=1)
        out = road > 3 * med[None] + 0.25 * np.percentile(med, 50)
        t = float(np.mean([times[f"{n}_s{s}"] for s in range(1, seeds + 1)]))
        r = {
            "time_s": t,
            "median_px_var": float(np.median(var)),
            "outliers_per_render": float(out.sum() / len(L)),
            "frame_ratio": float(L.mean(0)[keep].sum() / ref_mean[keep].sum()),
        }
        for roi, (y0, y1, x0, x1) in ROIS.items():
            m = L[:, y0:y1, x0:x1].mean((1, 2))
            r[f"{roi}_mean"], r[f"{roi}_se"] = float(m.mean()), float(m.std(ddof=1) / np.sqrt(len(m)))
        cm = [band_sum(p).mean() for p in sorted((work / "out").glob(f"crop_{n}_spp{crop_spp}_s*.exr"))]
        if len(cm) >= 2:
            r["crop_mean"], r["crop_se"], r["crop_n"] = float(np.mean(cm)), float(np.std(cm, ddof=1) / np.sqrt(len(cm))), len(cm)
        rows[n] = r
    p = rows["path"]
    for r in rows.values():
        r["efficiency_vs_path"] = (r["median_px_var"] * r["time_s"]) / (p["median_px_var"] * p["time_s"])
        if "crop_mean" in r and "crop_mean" in p:
            r["crop_vs_path"] = r["crop_mean"] / p["crop_mean"]
        r["near_road_vs_path"] = r["near_road_mean"] / p["near_road_mean"]
    res = {"seeds": seeds, "crop_spp": crop_spp, "rois": ROIS, "variants": rows}
    (work / "results.json").write_text(json.dumps(res, indent=2) + "\n")
    figure(work, imgs, rows)
    return res


def figure(work: Path, imgs: dict, rows: dict) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    names = list(imgs)
    scale = float(np.percentile(imgs["path"].mean(0)[ROAD_ROWS], 99.0))
    fig, axs = plt.subplots(len(names), 2, figsize=(12, 2.1 * len(names)), facecolor="#fcfcfb", squeeze=False)
    for i, n in enumerate(names):
        for j, (img, lab) in enumerate(((imgs[n][0], "seed 1"), (imgs[n].mean(0), f"mean of {len(imgs[n])} seeds"))):
            ax = axs[i, j]
            ax.imshow(np.clip(img[ROAD_ROWS] / scale, 0, 1) ** (1 / 2.2), cmap="gray", interpolation="nearest")
            ax.axis("off")
            r = rows[n]
            ax.set_title(f"{n}, {lab}  |  outliers/render {r['outliers_per_render']:.0f}, efficiency {r['efficiency_vs_path']:.2f}",
                         fontsize=8.5, color="#0b0b0b", loc="left")
            y0, y1, x0, x1 = ROIS["behind_lead"]
            ax.add_patch(plt.Rectangle((x0, y0 - ROAD_ROWS.start), x1 - x0, y1 - y0, fill=False, ec="#eb6834", lw=1))
    fig.suptitle("Night wet highway, road rows 140-270 (box: road behind the lead car)", fontsize=11, color="#0b0b0b")
    fig.tight_layout()
    (work / "figs").mkdir(exist_ok=True)
    fig.savefig(work / "figs" / "variants.png", dpi=120, facecolor="#fcfcfb")
    plt.close(fig)


def _load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def _save(p: Path, d: dict) -> None:
    p.write_text(json.dumps(d, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", type=Path, default=REPO / "out" / "firefly_exp")
    ap.add_argument("--spp", type=int, default=256, help="samples per pixel of the full-frame renders")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--crop-spp", type=int, default=4096)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("stage", choices=("build", "render", "crops", "analyze"))
    a = ap.parse_args(argv)
    variants = [v for v in a.variants.split(",") if v]
    if a.stage == "build":
        build(a.work, a.spp)
    elif a.stage == "render":
        render(a.work, a.seeds, variants)
    elif a.stage == "crops":
        crops(a.work, a.seeds, a.crop_spp, variants)
    else:
        res = analyze(a.work, a.seeds, a.crop_spp)
        print(json.dumps(res["variants"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
