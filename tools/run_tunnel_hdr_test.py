#!/usr/bin/env python3
"""Tunnel HDR test: render both tunnel views and score HDR / non-HDR recipes per ROI.

Builds the two views of tools/build_tunnel_scene.py (``outside_in``: black hole,
``inside_out``: white hole), renders them with pbrt and pushes each spectral EXR through
``apply_emva_noise`` for every recipe, metered as tools/run_recipe_scorecard.py does for its
HDR chart (linear recipes: 99th percentile at 80 % of full well; HDR recipes: frame maximum
at 90 % of the top readout). A second capture with another noise seed at the same exposure
gives temporal noise, so the SNR excludes Monte Carlo noise and scene texture.

Per view, recipe and ROI it reports the mean signal [e-], temporal SNR [dB] and the clipped
fraction, plus the target contrast against its background and the measured bright / dark
ratio, next to the scene values from the EXR. Writes ``tunnel_hdr.json``, ``tunnel_hdr.md``
and preview PNGs (tone-mapped scene, each recipe's output) to ``--out-dir``.

    venv/bin/python tools/run_tunnel_hdr_test.py --out-dir out/iq_lab/tunnel_hdr
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_tunnel_scene as bts  # noqa: E402
import run_recipe_scorecard as sc  # noqa: E402
from exr_multispectral import linear_rgb_from_exr  # noqa: E402

REPO = sc.REPO
DEFAULT_RECIPES = ["default", "default_hdr_dcg", "default_hdr_split_pixel", "default_hdr_lofic"]
BACKGROUND = {"outside_in": "threshold_road", "inside_out": "exit_opening"}
WHITE_ROI = "headwall"  # sunlit concrete in outside_in: daylight white balance for both views
LUMA = np.array([0.2126, 0.7152, 0.0722])
DARK_TARGET = 0.18  # "dark" metering: dark ROI mean at 18 % of saturation (mid grey, a choice)


def render_views(out: Path, xres: int, yres: int, spp: int, builder_args: list[str], reuse: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    argv = ["--out-dir", str(out), "--xres", str(xres), "--yres", str(yres), "--pixelsamples", str(spp)]
    built = bts.build(bts.parse_args(argv + builder_args))
    (out / "out").mkdir(exist_ok=True)
    renders = {}
    for view, b in built.items():
        meta = b["meta"]
        if not (reuse and (out / meta["film_output"]).is_file()):
            subprocess.run([str(sc.PBRT), "--quiet", b["scene"].name], cwd=out, check=True)
        rois = {k: r["roi_xyxy"] for k, r in meta["rois"].items()}
        if meta["camera"]["type"] == "realistic":
            P = sc.gbuffer_world_positions(b["scene"], xres, yres)
            rois = {k: world_box_roi(P, r["world_box"]) for k, r in meta["rois"].items() if r["world_box"]}
        renders[view] = {"exr": out / meta["film_output"], "manifest": b["manifest"], "shape": (yres, xres),
                         "meta": meta, "rois": rois}  # fmt: skip
    return renders


def world_box_roi(P: np.ndarray, box: list[float], tol: float = 0.05, shrink: float = 0.15) -> list[int] | None:
    """Bounding box (shrunk) of the pixels whose gbuffer hit point lies in the world box."""
    lo = np.array(box[0::2]) - tol
    hi = np.array(box[1::2]) + tol
    ys, xs = np.nonzero(np.all((lo <= P) & (hi >= P), axis=-1))
    if len(xs) < 4:
        return None
    x0, x1, y0, y1 = np.percentile(xs, 5), np.percentile(xs, 95), np.percentile(ys, 5), np.percentile(ys, 95)
    mx, my = shrink * (x1 - x0), shrink * (y1 - y0)
    r = [math.ceil(x0 + mx), math.ceil(y0 + my), math.floor(x1 - mx) + 1, math.floor(y1 - my) + 1]
    return r if r[2] - r[0] >= 2 and r[3] - r[1] >= 2 else None


def scene_luminance(render: dict) -> dict:
    """Mean luminance per ROI from the EXR (radiance cameras: EXR luminance is cd/m^2)."""
    Y = linear_rgb_from_exr(render["exr"]) @ LUMA
    out = {}
    for k, r in render["rois"].items():
        if r:
            x0, y0, x1, y1 = r
            out[k] = float(Y[y0:y1, x0:x1].mean())
    return out


def roi_stats(e: np.ndarray, e2: np.ndarray, layout, shape: tuple, rois: dict, sat: float) -> dict:
    """Guide-channel mean, temporal SNR and clipped fraction per ROI."""
    lay, b = sc.binned_layout(layout, e.shape, shape)
    s1, th, tw = sc.guide_sites(e, lay)
    s2 = sc.guide_sites(e2, lay)[0]
    out = {}
    for k, r in rois.items():
        if not r:
            continue
        x0, y0, x1, y1 = sc._scale_roi(sc._scale_roi(r, b, b), th, tw)
        a, c = s1[y0:y1, x0:x1], s2[y0:y1, x0:x1]
        if a.size < 4:
            continue
        mu = float(a.mean())
        sd = float(np.std(a - c) / math.sqrt(2.0))
        out[k] = {
            "mean_e": mu,
            "snr_db": 20.0 * math.log10(mu / sd) if mu > 0 and sd > 0 else float("nan"),
            "clipped_fraction": float(np.mean(a >= 0.98 * sat)),
        }
    return out


def michelson(a: float, b: float) -> float:
    return abs(a - b) / (a + b) if a + b > 0 else float("nan")


def tonemap(rgb: np.ndarray, white: float, k: float = 1000.0) -> np.ndarray:
    """Global log tone map for display only: ``white`` maps to 1, k sets the shadow lift."""
    x = np.clip(rgb / white, 0.0, None)
    return np.clip(np.log1p(k * x) / math.log1p(k), 0.0, 1.0)


def save_png(rgb01: np.ndarray, path: Path) -> None:
    from PIL import Image

    v = np.where(rgb01 <= 0.0031308, 12.92 * rgb01, 1.055 * np.power(np.clip(rgb01, 0, 1), 1 / 2.4) - 0.055)
    Image.fromarray((np.clip(v, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)


def wb_gains(e: np.ndarray, layout, shape: tuple, roi: list[int] | None) -> np.ndarray:
    lay, b = sc.binned_layout(layout, e.shape, shape)
    chans = sc.cm.demosaic(np.clip(e, 0, None), lay, "gradient")
    if not roi:
        return np.ones(chans.shape[2])
    w = chans[roi[1] // b : roi[3] // b, roi[0] // b : roi[2] // b].reshape(-1, chans.shape[2]).mean(0)
    return w.mean() / np.maximum(w, 1e-12)


def run_recipe(recipe: str, renders: dict, tmp: Path, seed: int, previews: Path | None, meter: str = "ae") -> dict:
    """``meter="ae"``: scorecard metering; ``"dark"`` (linear recipes): dark ROI at DARK_TARGET of saturation."""
    model = sc.load_camera_model(sc.RECIPE_DIR / f"{recipe}.yaml")
    cfg = sc.noise_config_from_camera_model(model, "", "")
    layout = sc.recipe_layout(cfg)
    ircf = cfg["sensor"]["quantum_efficiency"].get("ircf_csv")
    ccm, _ = sc.cm.colorchecker_spectral_ccm(REPO, layout, ircf_csv=ircf, illuminant="D65")
    arch, _ = sc.hdr.recipe_architecture(recipe)
    is_hdr = arch.name != "linear"
    _, fw = sc.apply_emva_noise.iso_scaled_conversion(
        float(cfg["emva"]["overall_system_gain_K_e_per_DN"]),
        float(cfg["adc"]["full_well_e"]),
        float(cfg["emva"].get("iso_gain_factor", 1.0)),
    )
    pct, frac = (100.0, 0.9 * arch.max_reference_e / fw) if is_hdr else (99.0, 0.8)
    label = recipe if meter == "ae" else f"{recipe}@{meter}"
    row = {"recipe": label, "hdr": arch.name, "metering": meter, "views": {}}
    gains = None
    for view, render in renders.items():
        sens = sc.run_sensor(recipe, render, tmp / view, seed=seed, percentile=pct, fraction=frac)
        k = float(sens["stats"]["exposure_scale_e_per_unit"])
        sat = arch.max_reference_e if is_hdr else sc.saturation_e(sens["stats"])
        if gains is None:
            gains = wb_gains(sens["e"], layout, render["shape"], render["rois"].get(WHITE_ROI))
        meta = render["meta"]
        dr = meta["dynamic_range"]
        if meter == "dark":
            d = dr["dark_roi"]
            mu = roi_stats(sens["e"], sens["e"], layout, render["shape"], {d: render["rois"][d]}, sat)[d]["mean_e"]
            k *= DARK_TARGET * sat / mu
            sens = sc.run_sensor(recipe, render, tmp / f"{view}_dark", seed=seed, exposure_scale=k)
            sat = arch.max_reference_e if is_hdr else sc.saturation_e(sens["stats"])
        twin = sc.run_sensor(recipe, render, tmp / f"{view}_twin", seed=seed + 1, exposure_scale=k)
        stats = roi_stats(sens["e"], twin["e"], layout, render["shape"], render["rois"], sat)
        tgt, bg = stats.get("target"), stats.get(BACKGROUND[view])
        bright, dark = stats.get(dr["bright_roi"]), stats.get(dr["dark_roi"])
        row["views"][view] = {
            "exposure_scale_e_per_unit": k,
            "saturation_e": sat,
            "rois": stats,
            "target_contrast": michelson(tgt["mean_e"], bg["mean_e"]) if tgt and bg else None,
            "bright_dark_db": 20 * math.log10(bright["mean_e"] / dark["mean_e"])
            if bright and dark and dark["mean_e"] > 0
            else None,
        }
        if previews is not None:
            lay, _ = sc.binned_layout(layout, sens["e"].shape, render["shape"])
            rgb = np.clip(sc.to_srgb_linear(sens["e"], lay, ccm, None, sat, gains=gains), 0.0, None)
            white = sat * float(np.min(gains))
            img = tonemap(rgb, white) if is_hdr else np.clip(rgb / white, 0, 1)
            save_png(img, previews / f"{view}_{label.replace('@', '_')}.png")
    return row


def scene_preview(render: dict, path: Path) -> None:
    rgb = linear_rgb_from_exr(render["exr"])
    save_png(tonemap(rgb, float(np.percentile(rgb @ LUMA, 99.9)), k=100.0), path)


def write_report(res: dict, path: Path) -> None:
    lines = ["# Tunnel HDR test", ""]
    for view, sv in res["scene"].items():
        dr = sv["dynamic_range"]
        lines += [
            f"## {view}",
            "",
            f"Scene: {dr['bright_roi']} / {dr['dark_roi']} = {dr['db']:.1f} dB predicted, "
            f"{sv['measured_db']:.1f} dB in the render; target contrast {sv['target_contrast']:.3f}.",
            "",
            "| recipe | HDR | ROI | scene cd/m^2 | mean e- | SNR dB | clipped |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in res["recipes"]:
            v = row["views"][view]
            for roi, s in v["rois"].items():
                lum = sv["luminance_cd_m2"].get(roi, float("nan"))
                lines.append(
                    f"| {row['recipe']} | {row['hdr']} | {roi} | {lum:.3g} | {s['mean_e']:.4g} | "
                    f"{s['snr_db']:.1f} | {100 * s['clipped_fraction']:.0f} % |"
                )
        lines += ["", "| recipe | target contrast | bright/dark dB |", "|---|---|---|"]
        for row in res["recipes"]:
            v = row["views"][view]
            tc, bd = v["target_contrast"], v["bright_dark_db"]
            lines.append(
                f"| {row['recipe']} | {'n/a' if tc is None else f'{tc:.3f}'} | {'n/a' if bd is None else f'{bd:.1f}'} |"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=REPO / "out" / "iq_lab" / "tunnel_hdr")
    ap.add_argument("--recipes", nargs="+", default=DEFAULT_RECIPES)
    ap.add_argument("--xres", type=int, default=960)
    ap.add_argument("--yres", type=int, default=640)
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-previews", action="store_true")
    ap.add_argument("--reuse-render", action="store_true", help="Keep existing EXRs in --out-dir/scene.")
    ap.add_argument("--builder-arg", action="append", default=[], help="Extra build_tunnel_scene.py argument.")
    args = ap.parse_args(argv)
    out = args.out_dir.resolve()
    renders = render_views(out / "scene", args.xres, args.yres, args.spp, args.builder_arg, args.reuse_render)
    previews = None if args.no_previews else out / "previews"
    if previews:
        previews.mkdir(parents=True, exist_ok=True)
    res = {"scene": {}, "recipes": []}
    for view, r in renders.items():
        lum = scene_luminance(r)
        dr = r["meta"]["dynamic_range"]
        tgt, bg = lum.get("target"), lum.get(BACKGROUND[view])
        res["scene"][view] = {
            "dynamic_range": dr,
            "luminance_cd_m2": lum,
            "rois": r["rois"],
            "measured_db": 20 * math.log10(lum[dr["bright_roi"]] / lum[dr["dark_roi"]]),
            "target_contrast": michelson(tgt, bg) if tgt is not None and bg is not None else float("nan"),
        }
        if previews:
            scene_preview(r, previews / f"{view}_scene_tonemapped.png")
    for recipe in args.recipes:
        res["recipes"].append(run_recipe(recipe, renders, out / "tmp" / recipe, args.seed, previews))
        if sc.hdr.recipe_architecture(recipe)[0].name == "linear":
            res["recipes"].append(
                run_recipe(recipe, renders, out / "tmp" / f"{recipe}_dark", args.seed, previews, meter="dark")
            )
    (out / "tunnel_hdr.json").write_text(json.dumps(res, indent=2))
    write_report(res, out / "tunnel_hdr.md")
    print((out / "tunnel_hdr.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
