#!/usr/bin/env python3
"""Generate iPhone 8 dataset with dual exposure/gain protocols.

Protocol 1 (Standard): 500 lux, 0.12s exposure, 1x ISO
Protocol 2 (Low-light): 10 lux, 0.033s exposure, 8x ISO

Outputs under out/dataset_dual/
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
DATASET = REPO / "out" / "dataset_dual"

sys.path.insert(0, str(REPO / "tools"))
from camera_model import load_camera_model  # noqa: E402

PROTOCOLS = {
    "standard": {
        "label": "1000lux_1x",
        "illuminance_lux": 1000.0,
        "exposure_s": 0.020,
        "iso_gain": 1.0,
        "spp": 4096,
        "note": "Baseline: 1000 lux, 20ms, 1x ISO",
    },
    "midlight": {
        "label": "100lux_2x",
        "illuminance_lux": 100.0,
        "exposure_s": 0.020,
        "iso_gain": 2.0,
        "spp": 4096,
        "note": "Mid-light: 100 lux, 20ms, 2x ISO",
    },
    "lowlight": {
        "label": "20lux_8x",
        "illuminance_lux": 20.0,
        "exposure_s": 0.020,
        "iso_gain": 8.0,
        "spp": 4096,
        "note": "Low-light: 20 lux, 20ms, 8x ISO",
    },
}

ILLUMINANTS = {
    "d65": "spectra/illuminant/interpolated/D65.csv",
}

LUX = {"low": 10.0, "med": 200.0, "high": 2000.0}

DEFAULT_FRAMING = {"cc": 6.0, "munsell": 16.0}


def run(cmd, quiet=False):
    if not quiet:
        print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=str(REPO), check=True)


def cam_lens(recipe: str):
    m = load_camera_model(REPO / f"config/camera_recipes/{recipe}.yaml")
    lens = m["lens"]
    return lens["realistic_lensfile"], float(lens["realistic_aperture_diameter_mm"])


def make_cc_config(recipe: str, cam_dist: float, protocol: dict) -> tuple[Path, Path]:
    """Create pipeline and camera model configs with protocol-specific exposure/gain.

    Returns (pipeline_config_path, camera_model_config_path)
    """
    cfg = yaml.safe_load((REPO / "config/pipeline.yaml").read_text())
    cfg.setdefault("render", {})["cam_dist"] = cam_dist
    lo = cfg.get("lens_overrides") or {}
    lo["realistic_focus_distance"] = cam_dist
    cfg["lens_overrides"] = lo

    # Apply protocol settings to pipeline
    cfg["exposure_time_override_s"] = protocol["exposure_s"]
    cfg["sensor_forward"]["target_illuminance_lux"] = protocol["illuminance_lux"]
    cfg["render"]["pixelsamples"] = protocol["spp"]

    # Disable white balance
    cfg["noise"]["preview_white_balance_enabled"] = False

    cdir = DATASET / "_configs"
    cdir.mkdir(parents=True, exist_ok=True)

    # Pipeline config
    ppath = cdir / f"pipeline_iphone8_{protocol['label']}.yaml"
    ppath.write_text(yaml.safe_dump(cfg, sort_keys=False))

    # Camera model config with ISO gain
    if protocol["iso_gain"] != 1.0:
        full_model = yaml.safe_load((REPO / f"config/camera_models/iphone_8.yaml").read_text())
        full_model["noise"]["emva"]["iso_gain_factor"] = protocol["iso_gain"]

        cpath = cdir / f"camera_model_iphone8_{protocol['label']}.yaml"
        cpath.write_text(yaml.safe_dump(full_model, sort_keys=False))
    else:
        cpath = REPO / f"config/camera_recipes/{recipe}.yaml"

    return ppath, cpath


def copy_if(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def generate_colorchecker(protocols, skip_render=False, emit_linear16=False):
    """Generate ColorChecker for each protocol."""
    print("\n==== ColorChecker: D65, iPhone 8 ====", flush=True)
    recipe = "iphone_8"
    lensfile, _ = cam_lens(recipe)
    cc_dist = DEFAULT_FRAMING["cc"]

    for pname, protocol in protocols.items():
        print(f"\n-- Protocol: {protocol['label']} --", flush=True)
        pipeline_cfg, camera_cfg = make_cc_config(recipe, cc_dist, protocol)

        cmd = [PY, "tools/run_pipeline.py",
               "--config", str(pipeline_cfg),
               "--camera-model-config", str(camera_cfg),
               "--name", f"cc_{protocol['label']}"]
        if skip_render:
            cmd.append("--skip-render")
        if emit_linear16:
            cmd.append("--emit-demosaic-linear16")
        run(cmd)

        # Copy outputs (demosaiced and mosaic images)
        base = REPO / "out"
        outdir = DATASET / "colorchecker" / protocol["label"]
        copy_if(base / "colorchecker_noisy_png/clean_demosaic_rgb8.png",
                outdir / "clean_demosaic_rgb8.png")
        copy_if(base / "colorchecker_noisy_png/noisy_demosaic_rgb8.png",
                outdir / "noisy_demosaic_rgb8.png")
        copy_if(base / "colorchecker_noisy_png/clean_demosaic_linear_rgb16.png",
                outdir / "clean_demosaic_linear_rgb16.png")
        copy_if(base / "colorchecker_noisy_png/noisy_demosaic_linear_rgb16.png",
                outdir / "noisy_demosaic_linear_rgb16.png")
        copy_if(base / "colorchecker_noisy_png/clean_rgb8.png",
                outdir / "clean_mosaic_rgb8.png")
        copy_if(base / "colorchecker_noisy_png/noisy_rgb8.png",
                outdir / "noisy_mosaic_rgb8.png")
        copy_if(base / "colorchecker_noisy_png/noisy_mono_16.png",
                outdir / "noisy_mosaic_mono_16.png")
        copy_if(base / "colorchecker_noisy_png/run_stats.json",
                outdir / "run_stats.json")
        copy_if(base / "sensor_forward_electrons.npz",
                outdir / "electrons.npz")
        print(f"  -> {outdir}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-colorchecker", action="store_true")
    ap.add_argument("--protocol", choices=["standard", "midlight", "lowlight", "all"],
                    default="all", help="Which protocol(s) to generate")
    ap.add_argument("--skip-render", action="store_true",
                    help="Reuse existing rendered EXRs (no GPU re-render).")
    ap.add_argument("--emit-demosaic-linear16", action="store_true",
                    help="Also write linear 16-bit demosaiced images.")
    args = ap.parse_args()

    # Filter protocols
    protocols = PROTOCOLS.copy()
    if args.protocol == "standard":
        protocols = {k: v for k, v in protocols.items() if k == "standard"}
    elif args.protocol == "midlight":
        protocols = {k: v for k, v in protocols.items() if k == "midlight"}
    elif args.protocol == "lowlight":
        protocols = {k: v for k, v in protocols.items() if k == "lowlight"}

    t0 = time.perf_counter()
    if not args.skip_colorchecker:
        generate_colorchecker(protocols, skip_render=args.skip_render,
                              emit_linear16=args.emit_demosaic_linear16)

    elapsed = time.perf_counter() - t0
    print(f"\n✅ Done in {elapsed:.1f}s", flush=True)
    print(f"Dataset at {DATASET}", flush=True)


if __name__ == "__main__":
    main()
