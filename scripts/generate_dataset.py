#!/usr/bin/env python3
"""Generate a CCM/Munsell dataset for three quality-tier cameras.

Part 1: ColorChecker under D65 for each camera (for CCM tuning).
Part 2: Munsell hue-family patch images for each camera, under four
        illuminants (D65, 2800K, 4000K, CWF) and three illumination levels
        (low/med/high lux).

The Munsell render is per-camera (each camera's realistic lens). Illumination
level is applied at the electron-conversion stage, so the three lux levels reuse
each rendered EXR (no extra rendering).

Outputs under out/dataset/. Run with --smoke for a fast single-condition check.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
PY = str(REPO / "venv" / "bin" / "python")
PBRT = str(REPO / "third_party" / "pbrt-v4" / "build" / "pbrt")
DATASET = REPO / "out" / "dataset"

sys.path.insert(0, str(REPO / "tools"))
from camera_model import load_camera_model  # noqa: E402

# tier label -> recipe name
CAMERAS = {
    "high_nikon_z7": "nikon_z7",
    "medium_fujifilm_x_t3": "fujifilm_x_t3",
    "low_iphone_8": "iphone_8",
}
# illuminant label -> spectrum CSV (repo-relative)
ILLUMINANTS = {
    "d65": "spectra/illuminant/interpolated/D65.csv",
    "2800k": "spectra/illuminant/interpolated/A.csv",       # CIE A ~2856K
    "4000k": "spectra/illuminant/interpolated/BB4000K.csv",  # generated Planckian
    "cwf": "spectra/illuminant/interpolated/F2_CWF.csv",
}
# level label -> scene illuminance (lux)
LUX = {"low": 10.0, "med": 200.0, "high": 2000.0}

# Realistic lenses have different fields of view, so each needs its own camera
# distance for the full target to fit in frame (determined empirically). Keyed by
# the camera's resolved lensfile -> (colorchecker cam_dist, munsell cam_dist).
LENS_FRAMING = {
    "config/lenses/wide_22mm.dat": {"cc": 4.0, "munsell": 6.0},
    "config/lenses/dgauss.50mm.dat": {"cc": 12.0, "munsell": 16.0},
}
DEFAULT_FRAMING = {"cc": 12.0, "munsell": 16.0}


def run(cmd, quiet=False):
    if not quiet:
        print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=str(REPO), check=True)


def cam_lens(recipe: str):
    m = load_camera_model(REPO / f"config/camera_recipes/{recipe}.yaml")
    lens = m["lens"]
    return lens["realistic_lensfile"], float(lens["realistic_aperture_diameter_mm"])


def framing_for(lensfile: str):
    return LENS_FRAMING.get(lensfile, DEFAULT_FRAMING)


def make_cc_config(recipe: str, cam_dist: float) -> Path:
    """Per-camera pipeline config with camera distance/focus set to fit the target."""
    cfg = yaml.safe_load((REPO / "config/pipeline.yaml").read_text())
    cfg.setdefault("render", {})["cam_dist"] = cam_dist
    lo = cfg.get("lens_overrides") or {}
    lo["realistic_focus_distance"] = cam_dist
    cfg["lens_overrides"] = lo
    cdir = DATASET / "_configs"
    cdir.mkdir(parents=True, exist_ok=True)
    cpath = cdir / f"pipeline_{recipe}.yaml"
    cpath.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cpath


def copy_if(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def part1_colorchecker(cameras):
    print("\n==== PART 1: ColorChecker (D65) for CCM tuning ====", flush=True)
    for tier, recipe in cameras.items():
        outdir = DATASET / "colorchecker" / tier
        outdir.mkdir(parents=True, exist_ok=True)
        lensfile, _ = cam_lens(recipe)
        cc_dist = framing_for(lensfile)["cc"]
        cfg_path = make_cc_config(recipe, cc_dist)
        run([PY, "tools/run_pipeline.py", "--config", str(cfg_path),
             "--camera-model-config", f"config/camera_recipes/{recipe}.yaml",
             "--name", f"cc_{tier}"])
        base = REPO / "out"
        copy_if(base / "colorchecker_noisy_png/clean_demosaic_rgb8.png", outdir / "clean_demosaic_rgb8.png")
        copy_if(base / "colorchecker_noisy_png/noisy_demosaic_rgb8.png", outdir / "noisy_demosaic_rgb8.png")
        copy_if(base / "colorchecker_noisy_png/run_stats.json", outdir / "run_stats.json")
        copy_if(base / "sensor_forward_electrons.npz", outdir / "electrons.npz")
        print(f"  -> {outdir}", flush=True)


def part2_munsell(cameras, illuminants, lux_levels, hues, spp, xres, yres):
    print("\n==== PART 2: Munsell patches ====", flush=True)
    scene_root = REPO / "scenes/generated/munsell"
    for tier, recipe in cameras.items():
        lensfile, aperture = cam_lens(recipe)
        cam_dist = framing_for(lensfile)["munsell"]
        cam_cfg = f"config/camera_recipes/{recipe}.yaml"
        for ilabel, icsv in illuminants.items():
            print(f"\n-- {tier} / {ilabel} (cam_dist={cam_dist}): build + render --", flush=True)
            if scene_root.exists():
                shutil.rmtree(scene_root)
            run([PY, "tools/build_munsell_scenes.py", "--repo-root", str(REPO),
                 "--out-dir", "scenes/generated/munsell", "--illuminant", icsv,
                 "--hues", hues, "--camera", "realistic", "--lensfile", lensfile,
                 "--aperture-diameter-mm", str(aperture),
                 "--focus-distance", str(cam_dist), "--cam-dist", str(cam_dist),
                 "--xres", str(xres), "--yres", str(yres), "--pixelsamples", str(spp),
                 "--film", "spectral", "--spectral-nbuckets", "64",
                 "--spectral-lambda-min", "360", "--spectral-lambda-max", "830"])
            scenes = sorted(scene_root.glob("*/munsell_*.pbrt"))
            if not scenes:
                print(f"  !! no scenes generated for {ilabel}; skipping", flush=True)
                continue
            for sc in scenes:
                run([PBRT, "--gpu", str(sc)])
            # apply sensor model per lux level (reuses each EXR)
            for llabel, lux in lux_levels.items():
                for sc in scenes:
                    hue = sc.stem[len("munsell_"):]
                    exr = REPO / f"out/munsell_{hue}_spectral.exr"
                    if not exr.exists():
                        print(f"  !! missing EXR {exr.name}; skip", flush=True)
                        continue
                    manifest = sc.parent / f"munsell_{hue}_manifest.json"
                    npz = REPO / f"out/munsell_sensor_forward/munsell_{hue}_electrons.npz"
                    npz.parent.mkdir(parents=True, exist_ok=True)
                    run([PY, "tools/pbrt_spectral_exr_to_electrons.py", "--repo-root", str(REPO),
                         "--exr", str(exr), "--camera-model-config", cam_cfg, "--out", str(npz),
                         "--target-illuminance-lux", str(lux),
                         "--scene-manifest-json", str(manifest)], quiet=True)
                    run([PY, "tools/apply_emva_noise.py", "--repo-root", str(REPO),
                         "--camera-model-config", cam_cfg, "--seed", "0",
                         "--linear-exr", str(exr), "--electrons-npz", str(npz),
                         "--preview-percentile", "99.5", "--preview-no-normalize",
                         "--preview-white-balance-enabled", "true",
                         "--preview-color-correction-enabled", "false"], quiet=True)
                    dest = DATASET / "munsell" / tier / ilabel / llabel / hue
                    dest.mkdir(parents=True, exist_ok=True)
                    prev = REPO / "out/colorchecker_noisy_png"
                    copy_if(prev / "noisy_demosaic_rgb8.png", dest / "noisy_demosaic_rgb8.png")
                    copy_if(prev / "clean_demosaic_rgb8.png", dest / "clean_demosaic_rgb8.png")
                print(f"  processed lux={llabel} ({lux}) for {tier}/{ilabel}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="Fast single-condition check")
    ap.add_argument("--skip-colorchecker", action="store_true")
    ap.add_argument("--skip-munsell", action="store_true")
    ap.add_argument("--spp", type=int, default=1024)
    ap.add_argument("--xres", type=int, default=960)
    ap.add_argument("--yres", type=int, default=640)
    ap.add_argument("--hues", default="all")
    args = ap.parse_args()

    cameras, illuminants, lux_levels = CAMERAS, ILLUMINANTS, LUX
    spp, hues = args.spp, args.hues
    if args.smoke:
        cameras = {"low_iphone_8": "iphone_8"}
        illuminants = {"d65": ILLUMINANTS["d65"]}
        lux_levels = {"med": LUX["med"]}
        spp, hues = 128, "R"

    t0 = time.perf_counter()
    if not args.skip_colorchecker:
        part1_colorchecker(cameras)
    if not args.skip_munsell:
        part2_munsell(cameras, illuminants, lux_levels, hues, spp, args.xres, args.yres)
    print(f"\nDone in {time.perf_counter() - t0:.1f}s. Dataset at {DATASET}", flush=True)


if __name__ == "__main__":
    main()
