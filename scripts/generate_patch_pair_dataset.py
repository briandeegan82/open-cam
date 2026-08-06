#!/usr/bin/env python3
"""Patch-pair brightness (strength) sweep through the iPhone-8 EMVA sensor.

Each ColorChecker patch pair is rendered under the locked angled light (north point
light, 45 deg, distance 1.8 -> ~59% mirror-symmetric gradient) at several photometric
brightness levels, defined as the target illuminance (lux) at the bright top edge, then
run through the sensor model (shot/read/row noise, PRNU/DSNU, demosaic).

Radiometry: realistic iphone_8_rggb camera (valid photometric anchor; a perspective
camera mis-scales the electron conversion). The peak-illuminance reference is measured
once from a surround probe of the pair geometry (illuminant/geometry-only, universal).

Layout (out/patch_pairs_emva/):
  pair_LL_RR/lux<NNNN>/noisy_mono_16.png              # RAW mosaic (analysis domain)
                      /noisy_demosaic_linear_rgb16.png # linear demosaiced
                      /clean_demosaic_linear_rgb16.png # noise-free reference
                      /preview.png                     # fixed-exposure sRGB (per pair)
                      /run_stats.json
                      /params.json
  pair_LL_RR/contact.png     # the 4 levels side by side (shared exposure)
  manifest.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
PY = str(REPO / "venv" / "bin" / "python")
sys.path.insert(0, str(REPO / "tools"))
from camera_model import load_camera_model  # noqa: E402
from apply_emva_noise import bilinear_demosaic  # noqa: E402
from scene_illuminance_probe import (  # noqa: E402
    make_probe_scene, render as probe_render, spectral_buckets_from_exr,
    illuminance_lux_from_irradiance,
)

RECIPE = "config/camera_recipes/iphone_8_rggb.yaml"
D65_CSV = "spectra/illuminant/interpolated/D65.csv"
CAM_DIST = 3.75
LENSFILE = "config/lenses/wide_22mm.dat"
APERTURE_MM = 8.756
SPATIAL_SEED = 1234

# Locked light geometry (see build_patch_pair_scene.py); only brightness (target lux) varies.
LIGHT = dict(type="point", direction="north", angle_deg=45.0, distance=1.8, scale=40.0,
             ambient_scale=4.0)
LUX_LEVELS = [1000, 200, 50, 10]
# Fixed capture: brightness differences between levels come from the light, not exposure.
EXPOSURE_S = 0.050
ISO_GAIN = 1.0

PAIRS = [(3, 8), (8, 13), (3, 5), (2, 7), (7, 12), (12, 16), (12, 15), (9, 15),
         (5, 10), (11, 14), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24)]

PNG_DIR = REPO / "out" / "colorchecker_noisy_png"
ELECTRONS_NPZ = REPO / "out" / "sensor_forward_electrons.npz"
EXR_OUT = REPO / "out" / "colorchecker_spectral.exr"


def run(cmd: list[str], quiet: bool = False) -> None:
    if not quiet:
        print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=str(REPO), check=True,
                   stdout=(subprocess.DEVNULL if quiet else None),
                   stderr=(subprocess.DEVNULL if quiet else None))


def builder_extra_args(left: int, right: int) -> list[str]:
    return ["--left-patch", str(left), "--right-patch", str(right),
            "--light-type", LIGHT["type"], "--light-direction", LIGHT["direction"],
            "--light-angle-deg", str(LIGHT["angle_deg"]),
            "--light-distance", str(LIGHT["distance"]),
            "--ambient-scale", str(LIGHT["ambient_scale"])]


def measure_peak_reference(gpu: bool, spp: int = 128) -> float:
    """Peak (top-edge) illuminance of the pair geometry, in raw-EXR lux.

    Builds pair 3-8, strips patches -> surround-only probe, and reads the top-centre
    surround window (brightest, most symmetric). Geometry/illuminant-only, so one value
    anchors every pair and brightness level.
    """
    left, right = 3, 8
    scene = REPO / "scenes" / "generated" / f"patch_pair_{left:02d}_{right:02d}.pbrt"
    run([PY, "tools/build_patch_pair_scene.py", "--repo-root", str(REPO),
         "--left-patch", str(left), "--right-patch", str(right),
         "--light-type", LIGHT["type"], "--light-direction", LIGHT["direction"],
         "--light-angle-deg", str(LIGHT["angle_deg"]), "--light-distance", str(LIGHT["distance"]),
         "--light-scale", str(LIGHT["scale"]), "--ambient-scale", str(LIGHT["ambient_scale"]),
         "--camera", "realistic", "--lensfile", LENSFILE,
         "--aperture-diameter-mm", str(APERTURE_MM), "--focus-distance", str(CAM_DIST),
         "--cam-dist", str(CAM_DIST), "--film", "spectral", "--spectral-nbuckets", "16",
         "--pixelsamples", "64", "--film-output", "out/patch_pair_probe.exr"], quiet=True)
    workdir = REPO / "out" / "patch_pair_probe"
    workdir.mkdir(parents=True, exist_ok=True)
    ppb, pex = workdir / "probe.pbrt", workdir / "probe.exr"
    rho = make_probe_scene(scene, ppb, pex, spp=spp)
    probe_render(ppb, gpu=gpu)
    cube, lam = spectral_buckets_from_exr(pex)
    H, W, _ = cube.shape
    win = cube[int(0.26 * H):int(0.31 * H), int(0.42 * W):int(0.58 * W)].reshape(-1, cube.shape[2])
    E = np.pi * win.mean(axis=0) / rho
    peak = float(illuminance_lux_from_irradiance(lam, E))
    print(f"peak-edge reference: {peak:.1f} raw-EXR lux (rho={rho})", flush=True)
    return peak


def make_configs(cfg_dir: Path, left: int, right: int, lux: int, spp: int,
                 reference_lux_exr: float, gpu: bool) -> tuple[Path, Path]:
    cfg_dir.mkdir(parents=True, exist_ok=True)
    scene_rel = f"scenes/generated/patch_pair_{left:02d}_{right:02d}.pbrt"

    cfg = yaml.safe_load((REPO / "config/pipeline.yaml").read_text())
    r = cfg.setdefault("render", {})
    r["cam_dist"] = CAM_DIST
    r["pixelsamples"] = spp
    r["illuminant"] = D65_CSV
    r["gpu_enabled"] = bool(gpu)
    r["film"] = "spectral"
    r["film_output"] = "out/colorchecker_spectral.exr"
    r["light_scale"] = LIGHT["scale"]
    r["builder_extra_args"] = builder_extra_args(left, right)
    cfg.setdefault("paths", {})["scene_builder"] = "tools/build_patch_pair_scene.py"
    cfg["paths"]["scene_file"] = scene_rel
    lo = cfg.get("lens_overrides") or {}
    lo["realistic_focus_distance"] = CAM_DIST
    cfg["lens_overrides"] = lo
    cfg["exposure_time_override_s"] = EXPOSURE_S
    cfg.setdefault("sensor_forward", {})["target_illuminance_lux"] = float(lux)
    cfg.setdefault("noise", {})["preview_white_balance_enabled"] = False
    cfg["noise"]["seed"] = 0
    ppath = cfg_dir / f"pipeline_{left:02d}_{right:02d}_lux{lux}.yaml"
    ppath.write_text(yaml.safe_dump(cfg, sort_keys=False))

    cm = load_camera_model(REPO / RECIPE)
    cm["lens"]["realistic_focus_distance"] = float(CAM_DIST)
    emva = cm["noise"]["emva"]
    emva["iso_gain_factor"] = ISO_GAIN
    emva["spatial_noise_seed"] = SPATIAL_SEED
    # Anchor: peak edge -> target lux (mirrors apply_chart_radiometry).
    sf = cm["sensor_forward"]["model"]
    cal = sf["calibration"]
    cal["illuminant_override_csv"] = None
    sf.setdefault("pbrt_spectral_exr", {})["radiometric_autocalibration"] = "off"
    cal["irradiance_scale_W_m2nm_per_unit"] = 1.0
    cal["scene_illuminance_reference_exr"] = float(reference_lux_exr)
    cpath = cfg_dir / f"camera_{left:02d}_{right:02d}_lux{lux}.yaml"
    cpath.write_text(yaml.safe_dump(cm, sort_keys=False))
    return ppath, cpath


def run_pipeline(ppath: Path, cpath: Path, name: str, skip_render: bool) -> None:
    for n in ("noisy_mono_16.png", "noisy_demosaic_linear_rgb16.png",
              "clean_demosaic_linear_rgb16.png", "run_stats.json"):
        (PNG_DIR / n).unlink(missing_ok=True)
    ELECTRONS_NPZ.unlink(missing_ok=True)  # recomputed each level (target lux changes)
    cmd = [PY, "tools/run_pipeline.py", "--config", str(ppath),
           "--camera-model-config", str(cpath), "--emit-demosaic-linear16", "--name", name]
    if skip_render:  # the spectral EXR is identical across lux levels of a pair
        cmd.append("--skip-render")
    run(cmd, quiet=True)


# Fixed D65 daylight white-balance gains (R,G,B; green=1) for the iPhone-8 RGGB sensor,
# fit once from a full ColorChecker render under D65 (white_patch reference). The pairs
# lack a neutral reference for per-scene auto-WB, so a fixed WB is required for correct,
# cross-scene-consistent colour. iPhone-8's QE needs only a near-diagonal CCM under D65,
# so this WB + sRGB gamma is the colour-corrected display image.
WB_GAINS_D65 = np.array([2.202, 1.0, 1.836], dtype=np.float64)
_BLACK_DN = 64.0
_MAX_DN = 1023.0  # 10-bit sensor


def _mono16_to_signal_rgb(mono16_path: Path) -> np.ndarray:
    """Demosaic the RAW mono (RGGB) into black-subtracted, white-balanced signal DN.

    The mono16 PNG (DN scaled to 16-bit) is the correctly-exposed sensor output; the
    demosaic_linear16 preview collapses the signal to a handful of codes, so the
    processed sRGB is built from the raw mono here to avoid quantisation banding."""
    dn = np.asarray(Image.open(mono16_path)).astype(np.float64) / 65535.0 * _MAX_DN
    rgb = bilinear_demosaic(dn, "RGGB").astype(np.float64)
    return np.clip(rgb - _BLACK_DN, 0.0, None) * WB_GAINS_D65


def _to_srgb8(sig_rgb: np.ndarray, white: float) -> Image.Image:
    x = np.clip(sig_rgb / max(white, 1e-9), 0.0, 1.0)
    srgb = np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
    return Image.fromarray((srgb * 255).astype(np.uint8))


def process_pair_srgb(pair_dir: Path, lux_levels: list[int]) -> None:
    """Write WB+gamma processed sRGB (noisy_srgb8) per level from the RAW mono, plus a
    4-level contact strip. Shared exposure across levels (brightest level's 99.5 pct
    after demosaic+WB) so the brightness sweep is visible in the colour images."""
    top_sig = _mono16_to_signal_rgb(pair_dir / f"lux{max(lux_levels):04d}" / "noisy_mono_16.png")
    white = float(np.percentile(top_sig, 99.7))
    thumbs = []
    for lux in lux_levels:
        lvl = pair_dir / f"lux{lux:04d}"
        img = _to_srgb8(_mono16_to_signal_rgb(lvl / "noisy_mono_16.png"), white)
        img.save(lvl / "noisy_srgb8.png")
        # A stale clean_srgb8 from the broken linear16 may exist — remove it.
        (lvl / "clean_srgb8.png").unlink(missing_ok=True)
        thumbs.append(img.convert("RGB"))
    tw = 300
    rs = [im.resize((tw, int(im.size[1] * tw / im.size[0]))) for im in thumbs]
    strip = Image.new("RGB", (len(rs) * (tw + 8) + 8, rs[0].size[1] + 8), (20, 20, 20))
    for i, im in enumerate(rs):
        strip.paste(im, (8 + i * (tw + 8), 4))
    strip.save(pair_dir / "contact.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=REPO / "out" / "patch_pairs_emva")
    ap.add_argument("--spp", type=int, default=1024)
    ap.add_argument("--gpu", choices=("true", "false"), default="true")
    ap.add_argument("--pairs", type=str, default=None, help='e.g. "3-8,8-13"')
    ap.add_argument("--lux", type=str, default=None, help='e.g. "1000,200,50,10"')
    args = ap.parse_args()
    gpu = args.gpu == "true"
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = PAIRS if not args.pairs else [tuple(int(x) for x in p.split("-")) for p in args.pairs.split(",")]
    lux_levels = LUX_LEVELS if not args.lux else [int(x) for x in args.lux.split(",")]

    reference = measure_peak_reference(gpu)
    manifest = {"light": LIGHT, "camera": "iphone_8_rggb", "lens": "realistic",
                "exposure_s": EXPOSURE_S, "iso_gain": ISO_GAIN,
                "reference_peak_lux_exr": reference, "lux_levels": lux_levels, "pairs": []}

    for left, right in pairs:
        tag = f"{left:02d}_{right:02d}"
        pair_dir = out_dir / f"pair_{tag}"
        print(f"\n==== pair {left}-{right} ====", flush=True)
        entry = {"pair": [left, right], "dir": f"pair_{tag}", "levels": []}
        for i, lux in enumerate(lux_levels):
            lvl_dir = pair_dir / f"lux{lux:04d}"
            lvl_dir.mkdir(parents=True, exist_ok=True)
            print(f"-- {tag} @ {lux} lux --", flush=True)
            ppath, cpath = make_configs(lvl_dir / "_configs", left, right, lux, args.spp, reference, gpu)
            # Render the pair once (first level); later levels reuse the EXR (lux only
            # rescales electrons, not the render).
            run_pipeline(ppath, cpath, f"pair_{tag}_lux{lux}", skip_render=(i > 0))
            for n in ("noisy_mono_16.png", "noisy_demosaic_linear_rgb16.png",
                      "clean_demosaic_linear_rgb16.png", "run_stats.json"):
                src = PNG_DIR / n
                if not src.exists():
                    raise FileNotFoundError(f"missing pipeline output: {src}")
                shutil.copy2(src, lvl_dir / n)
            stats = json.loads((lvl_dir / "run_stats.json").read_text())
            (lvl_dir / "params.json").write_text(json.dumps(
                {"pair": [left, right], "target_lux_top_edge": lux, "light": LIGHT,
                 "exposure_s": EXPOSURE_S, "iso_gain": ISO_GAIN,
                 "signal_e_mean_mono": stats.get("signal_e_mean_mono")}, indent=2) + "\n")
            entry["levels"].append({"lux": lux, "dir": f"pair_{tag}/lux{lux:04d}",
                                    "signal_e_mean_mono": stats.get("signal_e_mean_mono")})

        # Processed sRGB (D65 white balance + gamma) for clean & noisy, all levels.
        process_pair_srgb(pair_dir, lux_levels)
        manifest["pairs"].append(entry)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote manifest: {out_dir / 'manifest.json'}\nDataset root: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
