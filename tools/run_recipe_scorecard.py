#!/usr/bin/env python3
"""Recipe scorecard: every IQ-lab scene through the full pipeline for every camera recipe.

Scenes (one pbrt render per *optics group*, reused by every recipe in the group, because the
spectral EXR depends only on scene + optics):

* ``hdr``     -- emissive HDR chart (``build_hdr_dr_chart.py``, pinhole): SNR, DR, CDP
* ``diorama`` -- tabletop diorama (``build_iq_diorama.py``): slanted-edge MTF50 / acutance, dead-leaves acutance
* ``skin``    -- skin-tone chart under D65 (``build_skin_tone_chart.py``): mean / max skin ΔE00
* ``flare``   -- black-hole veiling-glare scene (``build_flare_test_scene.py``): veiling glare %

Per recipe, each EXR goes through ``apply_emva_noise`` with the recipe's own camera model (QE, IRCF,
EMVA noise, defects, CFA, HDR architecture, ADC), auto-exposed so the scene's ``exposure_percentile``
pixel sits at ``exposure_fraction`` of full well (HDR chart: brightest patch at 90 % of the recipe's
HDR saturation). The raw is demosaiced (``cfa_mosaic.demosaic``) and colour-corrected with the recipe's
spectral ColorChecker CCM under D65 (``cfa_mosaic.colorchecker_spectral_ccm``), so the skin chart is
scored with a CCM that was *not* fitted on it. SNR / DR / CDP use the recipe's electrons at the guide
channel (G or C) sites only, so there is no demosaic noise correlation.

Optics: ``lens.camera`` = pinhole / thinlens / realistic with the recipe's FOV, lens radius, lens file
and aperture; every camera is focused on the target plane and the camera distance is scaled with
tan(FOV/2) so each test chart fills the same part of the frame. Realistic-lens ROIs come from a
4-spp ``gbuffer`` render in world coordinates (pixels whose world point lies inside the target), so
they follow the lens' real projection and distortion. ``lens.post_psf`` is not applied.

    venv/bin/python tools/run_recipe_scorecard.py --out-dir out/iq_lab/scorecard --figure
    venv/bin/python tools/run_recipe_scorecard.py --recipes default default_hdr_dcg iphone_8 \\
        --xres 360 --yres 240 --pixelsamples 16

Writes ``scorecard.csv``, ``scorecard.json``, ``scorecard.md`` (and ``scorecard.png``).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import json
import math
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / "tools"
sys.path.insert(0, str(TOOLS))

import apply_emva_noise  # noqa: E402
import cfa_mosaic as cm  # noqa: E402
import lens_prescription as lp  # noqa: E402
import run_hdr_dr_test as hdr  # noqa: E402
import run_skin_tone_test as skin  # noqa: E402
import sfr_analysis as sfr  # noqa: E402
from camera_model import load_camera_model, noise_config_from_camera_model  # noqa: E402
from exr_multispectral import read_separate_exr_channels  # noqa: E402
from iqlab import cpiq, flare  # noqa: E402
from iqlab.dead_leaves import dead_leaves, texture_mtf  # noqa: E402

RECIPE_DIR = REPO / "config" / "camera_recipes"
PBRT = REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"
FILM_DIAGONAL_MM = 35.0  # pbrt-v4 RealisticCamera default
VIEW = "monitor_100pct"
CC_WHITE = 18  # ColorChecker "white 9.5"
HDR_MIN_SPP = 128  # the chart is rendered once; MC noise must stay well below sensor noise for CDP
HDR_MIN_XRES = 960  # HDR-chart patches are ~2 % of the width; 4x4 CFA tiles (quad Bayer) need >= 4 guide sites

# builder, default fov, default cam_dist, target-plane z, exposure: white-patch fraction of full well
# (spot metering) or (percentile, fraction) auto-exposure
SCENES = {
    "hdr": ("build_hdr_dr_chart.py", 30.0, 3.0, 0.0, None),
    "diorama": ("build_iq_diorama.py", 40.0, 1.6, -0.3, 0.5),
    "skin": ("build_skin_tone_chart.py", 40.0, 1.5, 0.0, 0.6),
    "flare": ("build_flare_test_scene.py", 60.0, 2.0, None, (95.0, 0.5)),
}
METRICS = (
    ("dr_snr1_db", "DR (SNR=1) dB", "{:.1f}"),
    ("dr_snr10_db", "DR (SNR=10) dB", "{:.1f}"),
    ("snr_at_18pct_db", "SNR @ 18 % sat. dB", "{:.1f}"),
    ("cdp_min_level_db_at_0p9", "CDP≥0.9 from dB", "{:.1f}"),
    ("edge_mtf50_cy_px", "MTF50 cy/px", "{:.3f}"),
    ("edge_acutance", "Edge acutance", "{:.3f}"),
    ("texture_acutance", "Texture acutance", "{:.3f}"),
    ("skin_de00_mean", "Skin ΔE00 mean", "{:.2f}"),
    ("skin_de00_max", "Skin ΔE00 max", "{:.2f}"),
    ("veiling_glare_pct", "Veiling glare %", "{:.2f}"),
)


# ------------------------------------------------------------------------------------------ recipes
def discover_recipes(names: list[str] | None = None) -> list[str]:
    found = sorted(p.stem for p in RECIPE_DIR.glob("*.yaml"))
    if not names:
        return found
    missing = [n for n in names if n not in found]
    if missing:
        raise SystemExit(f"unknown recipe(s): {missing}")
    return list(names)


def optics_of(model: dict) -> dict:
    """The part of a recipe that changes the pbrt render."""
    lens = model.get("lens") or {}
    cam = str(lens.get("camera", "pinhole")).lower()
    if cam == "realistic":
        lensfile = str(lens["realistic_lensfile"])
        efl, _ = lp.paraxial_focal_lengths_mm(lp.load_lens_file(REPO / lensfile))
        return {
            "camera": "realistic",
            "lensfile": lensfile,
            "aperture_diameter_mm": float(lens["realistic_aperture_diameter_mm"]),
            "efl_mm": round(float(efl), 4),
        }
    if cam == "thinlens":
        return {
            "camera": "thinlens",
            "fov": float(lens.get("thinlens_fov_deg", 45.0)),
            "lens_radius": float(lens.get("thinlens_lens_radius", 0.0)),
        }
    return {"camera": "perspective", "fov": float(lens.get("pinhole_fov_deg", 45.0))}


def optics_key(optics: dict) -> str:
    if optics["camera"] == "realistic":
        return f"realistic:{Path(optics['lensfile']).stem}:f{optics['aperture_diameter_mm']:g}mm"
    if optics["camera"] == "thinlens":
        return f"thinlens:{optics['fov']:g}deg:r{optics['lens_radius']:g}"
    return f"pinhole:{optics['fov']:g}deg"


def vertical_fov(optics: dict, xres: int, yres: int) -> float:
    if optics["camera"] != "realistic":
        return optics["fov"]
    h = FILM_DIAGONAL_MM * min(xres, yres) / math.hypot(xres, yres)
    return 2 * math.degrees(math.atan(h / 2 / optics["efl_mm"]))


def group_recipes(recipes: list[str]) -> dict[str, dict]:
    groups: dict[str, dict] = {}
    for r in recipes:
        o = optics_of(load_camera_model(RECIPE_DIR / f"{r}.yaml"))
        groups.setdefault(optics_key(o), {"optics": o, "recipes": []})["recipes"].append(r)
    return groups


# ------------------------------------------------------------------------------------------ render
def builder_args(scene: str, optics: dict, xres: int, yres: int, spp: int) -> list[str]:
    _, fov0, d0, z_t, _ = SCENES[scene]
    args = ["--xres", str(xres), "--yres", str(yres), "--pixelsamples", str(spp)]
    if scene == "hdr":
        return args
    fov = vertical_fov(optics, xres, yres)
    args += ["--fov", f"{fov:.6g}"]
    if z_t is not None:
        d = (d0 - z_t) * math.tan(math.radians(fov0 / 2)) / math.tan(math.radians(fov / 2)) + z_t
        args += ["--cam-dist", f"{d:.6g}", "--focus-distance", f"{d - z_t:.6g}"]
        focus = d - z_t
    else:
        focus = d0
    cam = optics["camera"]
    args += ["--camera", cam]
    if cam == "realistic":
        args += ["--lensfile", optics["lensfile"], "--aperture-diameter-mm", f"{optics['aperture_diameter_mm']:g}"]
    elif cam == "thinlens":
        args += ["--thinlens-lens-radius", f"{optics['lens_radius']:g}", "--thinlens-focal-distance", f"{focus:.6g}"]
    if scene == "skin":
        args += ["--illuminants", "D65"]
    if scene == "flare":
        args += ["--targets", "veiling_glare"]
    return args


def _pbrt(scene_file: Path, *extra: str) -> None:
    subprocess.run([str(PBRT), "--quiet", *extra, scene_file.name], cwd=scene_file.parent, check=True)


def gbuffer_world_positions(scene_file: Path, xres: int, yres: int) -> np.ndarray:
    """``[H, W, 3]`` world-space hit point per pixel from a 4-spp gbuffer render of ``scene_file``."""
    s = scene_file.read_text()
    film = (
        f'Film "gbuffer"\n    "string filename" ["gbuffer.exr"]\n    "integer xresolution" [{xres}]\n'
        f'    "integer yresolution" [{yres}]\n    "string coordinatesystem" "world"\n    "bool savefp16" false'
    )
    s, n = re.subn(r'Film "[a-z]+".*?(?=\n\S)', film, s, count=1, flags=re.S)
    if n != 1:
        raise RuntimeError(f"no Film block in {scene_file}")
    s = re.sub(r'"integer pixelsamples" \[\d+\]', '"integer pixelsamples" [4]', s)
    gb = scene_file.with_name(scene_file.stem + "_gbuffer.pbrt")
    gb.write_text(s)
    _pbrt(gb)
    ch = read_separate_exr_channels(str(gb.parent / "gbuffer.exr"))
    return np.stack([np.asarray(ch[f"P.{a}"], dtype=np.float64) for a in "XYZ"], axis=-1)


def world_roi(P: np.ndarray, rect: list[float], z: float, margin: float = 0.15) -> list[int] | None:
    """Raster bbox of the pixels that see the inner ``1 - 2*margin`` of world rectangle ``rect`` at depth ``z``."""
    x0, y0, x1, y1 = rect[:4]
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    sel = (
        (P[..., 0] > x0 + mx)
        & (P[..., 0] < x1 - mx)
        & (P[..., 1] > y0 + my)
        & (P[..., 1] < y1 - my)
        & (np.abs(P[..., 2] - z) < 0.01)
    )
    if sel.sum() < 4:
        return None
    ys, xs = np.nonzero(sel)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def render_scene(scene: str, optics: dict, out: Path, xres: int, yres: int, spp: int) -> dict:
    """Build + render ``scene`` for one optics group; returns EXR path, manifest and raster ROIs."""
    out.mkdir(parents=True, exist_ok=True)
    if scene == "hdr":
        spp = max(spp, HDR_MIN_SPP)
        if xres < HDR_MIN_XRES:
            xres, yres = HDR_MIN_XRES, round(yres * HDR_MIN_XRES / xres)
    builder = SCENES[scene][0]
    cmd = [sys.executable, str(TOOLS / builder), "--out-dir", str(out), *builder_args(scene, optics, xres, yres, spp)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    if scene == "hdr":
        meta = json.loads((out / "chart.json").read_text())
        scene_file, film = out / "hdr_dr_chart.pbrt", meta.get("film_output", "hdr_dr_chart_spectral.exr")
    elif scene == "diorama":
        meta = json.loads((out / "diorama.json").read_text())
        scene_file, film = out / "iq_diorama.pbrt", meta["film_output"]
    elif scene == "skin":
        meta = json.loads((out / "chart.json").read_text())
        scene_file, film = out / meta["scenes"]["D65"]["scene"], meta["scenes"]["D65"]["film_output"]
    else:
        meta = json.loads((out / "flare_manifest.json").read_text())
        sc = meta["scenes"][0]
        scene_file, film = out / sc["scene"], sc["film_output"]
    t0 = time.time()
    _pbrt(scene_file)
    res = {"exr": out / film, "meta": meta, "render_s": time.time() - t0, "rois": {}, "shape": (yres, xres)}
    if scene == "diorama":
        # Second render with another sampler seed: the twin capture then differs in MC noise too,
        # so the texture metric's noise PSD removes render noise as well as sensor noise.
        # pbrt's --seed leaves the image unchanged (README), so set the sampler's in-scene seed.
        twin = out / f"{Path(film).stem}_seed1{Path(film).suffix}"
        seeded = scene_file.with_name(scene_file.stem + "_seed1.pbrt")
        text, n = re.subn(r'(Sampler "\w+")', r'\1 "integer seed" [1]', scene_file.read_text(), count=1)
        if n != 1:
            raise RuntimeError(f"no Sampler in {scene_file}")
        seeded.write_text(text)
        _pbrt(seeded, "--outfile", twin.name)
        res["exr_twin"] = twin
    if scene in ("diorama", "skin") and optics["camera"] == "realistic":
        P = gbuffer_world_positions(scene_file, xres, yres)
        if scene == "diorama":
            for name in ("slanted_edge", "dead_leaves"):
                w = meta["cards"][name]["world"]
                res["rois"][name] = world_roi(P, w, w[4], 0.05)
            w = meta["colorchecker"][CC_WHITE]["world"]
            res["rois"]["white"] = world_roi(P, w, w[4], 0.2)
        else:
            res["rois"]["patches"] = [world_roi(P, p["world"], 0.0, 0.2) for p in meta["patches"]]
    elif scene == "diorama":
        res["rois"] = {k: meta["cards"][k]["roi_xyxy"] for k in ("slanted_edge", "dead_leaves")}
        res["rois"]["white"] = meta["colorchecker"][CC_WHITE]["roi_xyxy"]
    elif scene == "skin":
        res["rois"]["patches"] = [p["roi_xyxy"] for p in meta["patches"]]
    if scene == "skin":
        res["rois"]["white"] = res["rois"]["patches"][meta["white_index"]]
    radiometry = {
        "camera": {"type": optics["camera"], **({"lensfile": optics["lensfile"]} if "lensfile" in optics else {})}
    }
    if optics["camera"] == "realistic" and scene != "hdr":
        radiometry["camera"]["aperture_diameter_mm"] = optics["aperture_diameter_mm"]
    if scene == "hdr":
        radiometry["camera"]["type"] = "perspective"
    (out / "scorecard_manifest.json").write_text(json.dumps(radiometry))
    res["manifest"] = out / "scorecard_manifest.json"
    return res


# ------------------------------------------------------------------------------------------ sensor
def recipe_layout(cfg: dict) -> cm.CfaLayout:
    bayer = cfg.get("bayer") or {}
    lay = cm.resolve_layout(bayer)
    if lay is not None:
        return lay
    lay = cm.resolve_layout({"layout": str(bayer.get("pattern", "RGGB"))})
    qe = cfg["sensor"]["quantum_efficiency"]
    csv_for = {"R": qe["red_csv"], "G": qe["green_csv"], "B": qe["blue_csv"]}
    return dataclasses.replace(lay, qe_csv={c: csv_for[c] for c in lay.channels})


@contextlib.contextmanager
def _patched(model: dict, exr: Path, raw_out: Path):
    orig = apply_emva_noise.load_camera_model, apply_emva_noise.noise_config_from_camera_model
    apply_emva_noise.load_camera_model = lambda _p: model
    apply_emva_noise.noise_config_from_camera_model = lambda cam, **_kw: noise_config_from_camera_model(
        cam, str(exr), str(raw_out)
    )
    argv = sys.argv
    try:
        yield
    finally:
        sys.argv = argv
        apply_emva_noise.load_camera_model, apply_emva_noise.noise_config_from_camera_model = orig


def run_sensor(
    recipe: str,
    render: dict,
    tmp: Path,
    *,
    seed: int,
    percentile: float = 99.0,
    fraction: float = 0.8,
    exposure_scale: float | None = None,
) -> dict:
    """``apply_emva_noise`` with the recipe's camera model; returns the electron mosaic and run stats."""
    path = RECIPE_DIR / f"{recipe}.yaml"
    model = load_camera_model(path)
    tmp.mkdir(parents=True, exist_ok=True)
    raw_out = tmp / "raw.raw16"
    with _patched(model, render["exr"], raw_out), contextlib.redirect_stdout(sys.stderr):
        sys.argv = [
            "apply_emva_noise.py",
            "--repo-root", str(REPO),
            "--camera-model-config", str(path),
            "--scene-manifest-json", str(render["manifest"]),
            *(
                ["--exposure-scale", f"{exposure_scale:.9g}"]
                if exposure_scale is not None
                else [
                    "--auto-exposure",
                    "--target-fullwell-percentile", f"{percentile:g}",
                    "--target-fullwell-fraction", f"{fraction:.6g}",
                ]
            ),
            "--preview-color-correction-enabled", "false",
            "--seed", str(seed),
        ]  # fmt: skip
        apply_emva_noise.main()
    stats = json.loads((tmp / "raw_png" / "run_stats.json").read_text())
    if "hdr" in stats:
        e = np.load(Path(stats["hdr"]["outputs"]["dir"]) / "hdr_linear.npz")["hdr_e"].astype(np.float64)
    else:
        raw = np.fromfile(raw_out, dtype=np.uint16).astype(np.float64)
        h = int(round(math.sqrt(raw.size * render_shape(render)[0] / render_shape(render)[1])))
        e = (raw.reshape(h, -1) - float(stats["black_level_DN"])) * float(stats["K_effective_e_per_DN"])
    return {"e": e, "stats": stats}


def render_shape(render: dict) -> tuple[int, int]:
    return tuple(render["shape"])


def to_srgb_linear(e: np.ndarray, layout: cm.CfaLayout, ccm: np.ndarray, white_roi: list[int] | None) -> np.ndarray:
    """Demosaic, white-balance on ``white_roi`` (per-channel gains), then the recipe CCM."""
    chans = cm.demosaic(np.clip(e, 0.0, None), layout, "gradient")
    if white_roi is not None:
        x0, y0, x1, y1 = white_roi
        w = chans[y0:y1, x0:x1].reshape(-1, chans.shape[2]).mean(axis=0)
        chans = chans * (w.mean() / np.maximum(w, 1e-12))
    return cm.apply_channel_ccm(chans, ccm)


def white_signal_e(e: np.ndarray, layout: cm.CfaLayout, roi: list[int]) -> float:
    """Brightest channel's mean (e-) over ``roi`` after demosaicing, so metering on it clips no channel.

    Metering on the densest (guide) channel let the panchromatic W of RGBW clip, and a guide-site
    ROI of a binned quad-Bayer raster was empty at low resolution.
    """
    x0, y0, x1, y1 = roi
    chans = cm.demosaic(np.clip(e, 0.0, None), layout, "gradient")[y0:y1, x0:x1]
    if chans.size == 0:
        raise ValueError(f"white ROI {roi} is empty on a {e.shape} raster")
    return float(chans.reshape(-1, chans.shape[2]).mean(axis=0).max())


def saturation_e(stats: dict) -> float:
    """Output saturation in electrons: full well, or the ADC ceiling if lower (a 2x2 charge-binned
    quad Bayer keeps the 4800 e- photodiode's conversion gain, so its ADC clips well below 19200 e-)."""
    fw = float(stats["full_well_effective_e"])
    if "hdr" in stats or "bit_depth" not in stats:
        return fw
    adc = (2 ** int(stats["bit_depth"]) - 1 - float(stats["black_level_DN"])) * float(stats["K_effective_e_per_DN"])
    return min(fw, adc)


def meter_signal_e(
    e: np.ndarray, layout: cm.CfaLayout, white_roi: list[int], card_rois: list[list[int]] = (), pct: float = 90.0
) -> float:
    """Metering signal: the larger of the white patch's brightest-channel mean and the ``pct``-th
    percentile of the brightest channel in each card ROI (90th: robust to MC fireflies), so neither the
    white patch nor the card's white side clip or cross an HDR readout transition (the slanted-edge card is brighter than the
    ColorChecker white in the diorama)."""
    v = white_signal_e(e, layout, white_roi)
    if card_rois:
        chans = cm.demosaic(np.clip(e, 0.0, None), layout, "gradient")
        for x0, y0, x1, y1 in card_rois:
            c = chans[y0:y1, x0:x1]
            if c.size:
                v = max(v, float(np.percentile(c.max(axis=2), pct)))
    return v


def guide_sites(e: np.ndarray, layout: cm.CfaLayout) -> tuple[np.ndarray, int, int]:
    g = layout.channels[cm.guide_channel(layout)]
    for r, row in enumerate(layout.tile):
        for c, ch in enumerate(row):
            if ch == g:
                th, tw = len(layout.tile), len(layout.tile[0])
                return e[r::th, c::tw], th, tw
    raise RuntimeError("guide channel not in tile")


def binned_layout(layout: cm.CfaLayout, e_shape: tuple, shape: tuple) -> tuple[cm.CfaLayout, int]:
    """CFA of a ``b x b`` charge-binned readout (e.g. quad Bayer -> Bayer) and ``b``; ``b = 1`` if unbinned."""
    b = round(shape[0] / e_shape[0])
    if b <= 1:
        return layout, 1
    return dataclasses.replace(layout, tile=tuple(tuple(row[::b]) for row in layout.tile[::b])), b


def _scale_roi(roi: list[int], th: int, tw: int) -> list[int]:
    x0, y0, x1, y1 = roi
    return [math.ceil(x0 / tw), math.ceil(y0 / th), x1 // tw, y1 // th]


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.2126, 0.7152, 0.0722])


# ------------------------------------------------------------------------------------------ scoring
def score_hdr(
    e: np.ndarray, layout: cm.CfaLayout, chart: dict, saturation_e: float, *, b: int = 1, e2: np.ndarray | None = None
) -> dict:
    """``e2``: a second capture (other noise seed) so SNR uses temporal noise, free of pbrt MC and PRNU."""
    sites, th, tw = guide_sites(e, layout)
    frames = [sites] + ([guide_sites(e2, layout)[0]] if e2 is not None else [])
    chart = {**chart, "patches": [
        {**p, "low": {**p["low"], "roi_xyxy": _scale_roi(p["low"]["roi_xyxy"], b, b)},
         "high": {**p["high"], "roi_xyxy": _scale_roi(p["high"]["roi_xyxy"], b, b)}}
        for p in chart["patches"]
    ]}  # fmt: skip

    def big_enough(roi: list[int]) -> bool:
        x0, y0, x1, y1 = _scale_roi(roi, th, tw)
        return x1 - x0 >= 2 and y1 - y0 >= 2

    kept = [p for p in chart["patches"] if big_enough(p["low"]["roi_xyxy"]) and big_enough(p["high"]["roi_xyxy"])]
    if not kept:
        return {"hdr_patches_used": 0}
    chart = {**chart, "patches": kept}
    low = [[hdr._roi(f, _scale_roi(p["low"]["roi_xyxy"], th, tw)) for f in frames] for p in kept]
    high = [[hdr._roi(f, _scale_roi(p["high"]["roi_xyxy"], th, tw)) for f in frames] for p in kept]
    res = hdr.analyse(chart, low, high, saturation_e=0.98 * saturation_e)
    out = {"cdp_min_level_db_at_0p9": res.get("cdp_min_level_db_at_0p9"), "hdr_patches_used": len(kept)}
    for thr in (1, 10):
        m = res.get(f"min_signal_snr{thr}_e")
        out[f"dr_snr{thr}_db"] = 20 * math.log10(saturation_e / m) if m and m > 0 else float("nan")
    rows = [r for r in res["patches"] if r.get("saturated_fraction", 0) < 0.01 and r["mean_e"] > 0]
    if rows:
        mu = np.array([r["mean_e"] for r in rows])
        s = np.array([r["snr_db"] for r in rows])
        o = np.argsort(mu)
        out["snr_at_18pct_db"] = float(np.interp(math.log(0.18 * saturation_e), np.log(mu[o]), s[o]))
    return out


def despeckle(crop: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """Replace pbrt fireflies (pixels above ``factor`` x the white plateau, the 75th percentile) with
    their 3x3 median; sensor noise on the plateau never reaches that level, so the edge is untouched."""
    hot = crop > factor * np.percentile(crop, 75)
    return np.where(hot, ndimage.median_filter(crop, size=3), crop) if hot.any() else crop


def score_diorama(rgb: np.ndarray, rois: dict, seed: int = 0, noise: np.ndarray | None = None) -> dict:
    """``noise``: difference of two noise realisations / sqrt(2); its PSD is removed from the texture PSD."""
    y = _luma(rgb)
    view = cpiq.viewing_condition(VIEW, y.shape[0])
    out: dict = {}
    if rois.get("slanted_edge"):
        x0, y0, x1, y1 = rois["slanted_edge"]
        r = sfr.slanted_edge_sfr(despeckle(y[y0:y1, x0:x1]))
        out["edge_mtf50_cy_px"] = float(r.mtf50_cy_per_px)
        out["edge_acutance"] = float(cpiq.acutance(r.frequency_cy_per_px, r.mtf, view))
    if rois.get("dead_leaves"):
        x0, y0, x1, y1 = rois["dead_leaves"]
        crop = y[y0:y1, x0:x1]
        ideal = dead_leaves(1024, seed=seed)
        ideal = ndimage.zoom(ideal, (crop.shape[0] / ideal.shape[0], crop.shape[1] / ideal.shape[1]), order=0)
        ideal = ideal[: crop.shape[0], : crop.shape[1]] * crop.mean() / ideal.mean()
        noise_patch = _luma(noise)[y0:y1, x0:x1] if noise is not None else None
        f, m = texture_mtf(crop, ideal, noise_patch)
        low = (f > 0) & (f <= 0.05)
        m = m / (m[low].mean() if low.any() else 1.0)
        out["texture_acutance"] = float(cpiq.acutance(f, m, view))
    return out


def score_skin(rgb: np.ndarray, chart: dict, rois: list) -> dict:
    ch = {**chart, "patches": [{**p, "roi_xyxy": r} for p, r in zip(chart["patches"], rois, strict=True)]}
    if any(r is None for r in rois):
        return {}
    s = skin.score(ch, rgb, "D65")
    return {"skin_de00_mean": s["mean_delta_e00"], "skin_de00_max": s["max_delta_e00"]}


def score_flare(rgb: np.ndarray) -> dict:
    holes = flare.black_hole_glare(_luma(rgb))
    g = [h.glare_percent for h in holes if np.isfinite(h.glare_percent)]
    return {"veiling_glare_pct": float(np.median(g)) if g else float("nan"), "n_holes": len(g)}


def score_recipe(recipe: str, renders: dict, tmp: Path, seed: int, images: dict | None = None) -> dict:
    model = load_camera_model(RECIPE_DIR / f"{recipe}.yaml")
    cfg = noise_config_from_camera_model(model, "", "")
    layout = recipe_layout(cfg)
    ircf = cfg["sensor"]["quantum_efficiency"].get("ircf_csv")
    ccm, _ = cm.colorchecker_spectral_ccm(REPO, layout, ircf_csv=ircf, illuminant="D65")
    arch, _ = hdr.recipe_architecture(recipe)
    row: dict = {"recipe": recipe, "cfa": layout.name, "hdr": arch.name}
    for scene, render in renders.items():
        if scene == "hdr":
            _, fw = apply_emva_noise.iso_scaled_conversion(
                float(cfg["emva"]["overall_system_gain_K_e_per_DN"]),
                float(cfg["adc"]["full_well_e"]),
                float(cfg["emva"].get("iso_gain_factor", 1.0)),
            )
            sens = run_sensor(
                recipe, render, tmp / scene, percentile=100.0, fraction=0.9 * arch.max_reference_e / fw, seed=seed
            )
            k = float(sens["stats"]["exposure_scale_e_per_unit"])
            twin = run_sensor(recipe, render, tmp / "hdr_twin", seed=seed + 1, exposure_scale=k)
            lay_s, b = binned_layout(layout, sens["e"].shape, render_shape(render))
            row.update(score_hdr(sens["e"], lay_s, render["meta"], arch.max_reference_e, b=b, e2=twin["e"]))
            if images is not None:
                images[scene] = guide_sites(sens["e"], lay_s)[0] / arch.max_reference_e
            continue
        target = SCENES[scene][4]
        white = render["rois"].get("white")
        shape = render_shape(render)

        cards = [render["rois"]["slanted_edge"]] if render["rois"].get("slanted_edge") else []

        def white_e(e: np.ndarray, white=white, shape=shape) -> float:
            lay_s, b = binned_layout(layout, e.shape, shape)
            return white_signal_e(e, lay_s, _scale_roi(white, b, b))

        def meter_e(e: np.ndarray, white=white, shape=shape, cards=cards) -> float:
            lay_s, b = binned_layout(layout, e.shape, shape)
            return meter_signal_e(e, lay_s, _scale_roi(white, b, b), [_scale_roi(r, b, b) for r in cards])

        if isinstance(target, tuple) or white is None:
            pct, frac = target if isinstance(target, tuple) else (99.0, 0.8)
            sens = run_sensor(recipe, render, tmp / scene, percentile=pct, fraction=frac, seed=seed)
        else:
            sens = run_sensor(recipe, render, tmp / scene, seed=seed)
            # Spot-meter until converged: a clipped probe under-reads, so one correction is not enough.
            for _ in range(6):
                k = float(sens["stats"]["exposure_scale_e_per_unit"])
                goal = target * saturation_e(sens["stats"])
                ratio = goal / max(meter_e(sens["e"]), 1e-9)
                if abs(ratio - 1.0) < 0.02:
                    break
                sens = run_sensor(recipe, render, tmp / scene, seed=seed, exposure_scale=k * ratio)
        row[f"{scene}_white_e"] = white_e(sens["e"]) if white else None

        def rgb_of(e: np.ndarray, white=white, shape=shape) -> np.ndarray:
            lay_s, b = binned_layout(layout, e.shape, shape)
            rgb = to_srgb_linear(e, lay_s, ccm, _scale_roi(white, b, b) if white else None)
            if rgb.shape[:2] != shape:
                rgb = ndimage.zoom(rgb, (shape[0] / rgb.shape[0], shape[1] / rgb.shape[1], 1), order=1)
            return rgb

        rgb = rgb_of(sens["e"])
        if images is not None:
            images[scene] = rgb
        if scene == "diorama":
            twin_render = {**render, "exr": render.get("exr_twin", render["exr"])}
            twin = run_sensor(recipe, twin_render, tmp / "diorama_twin", seed=seed + 1, exposure_scale=k)
            noise = (rgb - rgb_of(twin["e"])) / math.sqrt(2.0)
            row.update(score_diorama(rgb, render["rois"], noise=noise))
        elif scene == "skin":
            row.update(score_skin(rgb, render["meta"], render["rois"]["patches"]))
        else:
            row.update(score_flare(rgb))
    return row


# ------------------------------------------------------------------------------------------ report
def _fmt(v, f: str) -> str:
    return f.format(v) if isinstance(v, (int, float)) and np.isfinite(v) else "–"


def write_report(rows: list[dict], groups: dict, args: argparse.Namespace, timing: dict) -> None:
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    keys = ["recipe", "optics", "cfa", "hdr"] + [k for k, _, _ in METRICS]
    with (out / "scorecard.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (out / "scorecard.json").write_text(
        json.dumps(
            {
                "settings": {
                    "xres": args.xres,
                    "yres": args.yres,
                    "pixelsamples": args.pixelsamples,
                    "seed": args.seed,
                    "view": VIEW,
                },
                "groups": {k: {"optics": g["optics"], "recipes": g["recipes"]} for k, g in groups.items()},
                "timing_s": timing,
                "rows": rows,
            },
            indent=2,
            default=float,
        )
    )
    md = [
        "# Recipe scorecard",
        "",
        f"{len(rows)} recipes in {len(groups)} optics groups; {args.xres}x{args.yres}, {args.pixelsamples} spp, seed {args.seed}; "
        f"acutance for `{VIEW}`. Generated by `tools/run_recipe_scorecard.py`.",
        "",
        "| Recipe | Optics | CFA | HDR | " + " | ".join(h for _, h, _ in METRICS) + " |",
        "|" + "---|" * (4 + len(METRICS)),
    ]
    for r in rows:
        md.append(
            f"| {r['recipe']} | {r.get('optics', '–')} | {r.get('cfa', '–')} | {r.get('hdr', '–')} | "
            + " | ".join(_fmt(r.get(k), f) for k, _, f in METRICS)
            + " |"
        )
    md += ["", "## Best per metric", ""]
    lower_better = {"skin_de00_mean", "skin_de00_max", "veiling_glare_pct", "cdp_min_level_db_at_0p9"}
    for k, h, f in METRICS:
        vals = [(r.get(k), r["recipe"]) for r in rows if isinstance(r.get(k), (int, float)) and np.isfinite(r.get(k))]
        if vals:
            v, name = (min if k in lower_better else max)(vals)
            md.append(f"- {h}: **{name}** ({_fmt(v, f)})")
    errors = [r for r in rows if r.get("error")]
    if errors:
        md += ["", "## Failed recipes", ""] + [f"- {r['recipe']}: `{r['error']}`" for r in errors]
    (out / "scorecard.md").write_text("\n".join(md) + "\n")


def plot(rows: list[dict], path: Path) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    pairs = [("dr_snr1_db", "edge_acutance"), ("skin_de00_mean", "veiling_glare_pct")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = {k: h for k, h, _ in METRICS}
    for ax, (xk, yk) in zip(axes, pairs, strict=True):
        for r in rows:
            x, y = r.get(xk), r.get(yk)
            if x is None or y is None or not (np.isfinite(x) and np.isfinite(y)):
                continue
            ax.scatter(x, y, s=14)
            ax.annotate(r["recipe"], (x, y), fontsize=5, alpha=0.7)
        ax.set_xlabel(labels[xk])
        ax.set_ylabel(labels[yk])
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipes", nargs="+", default=None, help="default: every config/camera_recipes/*.yaml")
    ap.add_argument("--scenes", nargs="+", choices=list(SCENES), default=list(SCENES))
    ap.add_argument("--xres", type=int, default=720)
    ap.add_argument("--yres", type=int, default=480)
    ap.add_argument("--pixelsamples", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=REPO / "out" / "iq_lab" / "scorecard")
    ap.add_argument("--figure", action="store_true")
    args = ap.parse_args(argv)

    recipes = discover_recipes(args.recipes)
    groups = group_recipes(recipes)
    work = args.out_dir / "work"
    rows, timing = [], {"render": 0.0, "sensor": 0.0}
    settings = {"xres": args.xres, "yres": args.yres, "pixelsamples": args.pixelsamples, "seed": args.seed}
    settings["scenes"] = sorted(args.scenes)
    cache_dir = work / "rows"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def cached(r: str) -> dict | None:
        f = cache_dir / f"{r}.json"
        if not f.is_file():
            return None
        d = json.loads(f.read_text())
        return d["row"] if d.get("settings") == settings else None

    hdr_render = None
    for gi, (key, g) in enumerate(groups.items()):
        print(f"[{gi + 1}/{len(groups)}] {key}: {len(g['recipes'])} recipe(s)", file=sys.stderr)
        if all(cached(r) is not None for r in g["recipes"]):
            rows += [cached(r) for r in g["recipes"]]
            continue
        t0 = time.time()
        renders = {}
        for scene in args.scenes:
            if scene == "hdr":
                if hdr_render is None:
                    hdr_render = render_scene("hdr", g["optics"], work / "hdr", args.xres, args.yres, args.pixelsamples)
                renders["hdr"] = hdr_render
            else:
                d = work / re.sub(r"[^A-Za-z0-9._-]", "_", key) / scene
                renders[scene] = render_scene(scene, g["optics"], d, args.xres, args.yres, args.pixelsamples)
        timing["render"] += time.time() - t0
        t0 = time.time()
        for r in g["recipes"]:
            row = cached(r)
            if row is None:
                try:
                    row = {"optics": key, **score_recipe(r, renders, work / "sensor" / r, args.seed)}
                    (cache_dir / f"{r}.json").write_text(json.dumps({"settings": settings, "row": row}, default=float))
                except Exception as exc:  # one broken recipe must not lose an 85-recipe sweep
                    traceback.print_exc()
                    row = {"recipe": r, "optics": key, "error": f"{type(exc).__name__}: {exc}"}
            rows.append(row)
        timing["sensor"] += time.time() - t0
    rows.sort(key=lambda r: r["recipe"])
    write_report(rows, groups, args, timing)
    if args.figure:
        plot(rows, args.out_dir / "scorecard.png")
    print(f"{len(rows)} recipes, {len(groups)} optics groups -> {args.out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
