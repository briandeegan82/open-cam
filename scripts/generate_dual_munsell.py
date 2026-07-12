#!/usr/bin/env python3
"""Generate Munsell patches for three-protocol iPhone 8 dataset.

Protocols: 1000lux_1x, 100lux_2x, 20lux_8x
Illuminants: CWF (CIE F2), A (2856K)
Lux levels: med (200 lux) for each protocol's base illuminance

Outputs under out/dataset_dual/munsell/ (illuminant subfolders alongside any
existing D65 run).
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

# Protocol definitions
PROTOCOLS = {
    "1000lux_1x": {
        "illuminance_lux": 1000.0,
        "exposure_s": 0.020,
        "iso_gain": 1.0,
        "camera_cfg": REPO / "config/camera_recipes/iphone_8.yaml",
    },
    "100lux_2x": {
        "illuminance_lux": 100.0,
        "exposure_s": 0.020,
        "iso_gain": 2.0,
        "camera_cfg": None,  # Will create custom config
    },
    "20lux_8x": {
        "illuminance_lux": 20.0,
        "exposure_s": 0.020,
        "iso_gain": 8.0,
        "camera_cfg": None,  # Will create custom config
    },
}

ILLUMINANTS = {
    "CWF": "spectra/illuminant/interpolated/F2_CWF.csv",  # CIE F2 cool white fluorescent
    "A": "spectra/illuminant/interpolated/A.csv",          # CIE A ~2856K incandescent
}

# Illuminance for sensor forward (use protocol's base illuminance)

DEFAULT_FRAMING = {"munsell": 6.0}


def run(cmd, quiet=False):
    if not quiet:
        print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=str(REPO), check=True)


def cam_lens(recipe: str):
    m = load_camera_model(recipe)
    lens = m["lens"]
    return lens["realistic_lensfile"], float(lens["realistic_aperture_diameter_mm"])


def make_camera_model_config(protocol_name: str, protocol: dict) -> Path:
    """Create camera model config with ISO gain for non-standard protocols."""
    if protocol["iso_gain"] == 1.0:
        return REPO / "config/camera_recipes/iphone_8.yaml"

    # Create custom camera model with ISO gain
    base_model = yaml.safe_load((REPO / "config/camera_models/iphone_8.yaml").read_text())
    base_model["noise"]["emva"]["iso_gain_factor"] = protocol["iso_gain"]

    cdir = DATASET / "_configs"
    cdir.mkdir(parents=True, exist_ok=True)
    cpath = cdir / f"camera_model_iphone8_{protocol_name}.yaml"
    cpath.write_text(yaml.safe_dump(base_model, sort_keys=False))
    return cpath


def copy_if(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def generate_munsell(protocols, illuminants, hues, spp, xres, yres,
                     skip_render=False, emit_linear16=False):
    """Generate Munsell patches for each protocol and illuminant."""
    print("\n==== MUNSELL PATCHES ====", flush=True)
    scene_root = REPO / "scenes/generated/munsell"
    cam_dist = DEFAULT_FRAMING["munsell"]
    lensfile, aperture = cam_lens(REPO / "config/camera_recipes/iphone_8.yaml")

    for proto_name, protocol in protocols.items():
        camera_cfg = make_camera_model_config(proto_name, protocol)

        for ilabel, icsv in illuminants.items():
            print(f"\n-- {proto_name} / {ilabel} --", flush=True)
            if not skip_render:
                if scene_root.exists():
                    shutil.rmtree(scene_root)

                # Build Munsell scene
                run([PY, "tools/build_munsell_scenes.py",
                     "--repo-root", str(REPO),
                     "--out-dir", "scenes/generated/munsell",
                     "--illuminant", icsv,
                     "--hues", hues,
                     "--camera", "realistic",
                     "--lensfile", lensfile,
                     "--aperture-diameter-mm", str(aperture),
                     "--focus-distance", str(cam_dist),
                     "--cam-dist", str(cam_dist),
                     "--xres", str(xres),
                     "--yres", str(yres),
                     "--pixelsamples", str(spp),
                     "--film", "spectral",
                     "--spectral-nbuckets", "64",
                     "--spectral-lambda-min", "360",
                     "--spectral-lambda-max", "830"])

            scenes = sorted(scene_root.glob("*/munsell_*.pbrt"))
            if not scenes:
                msg = ("no existing scenes found under scenes/generated/munsell"
                       if skip_render else f"no scenes generated for {ilabel}")
                print(f"  !! {msg}; skipping", flush=True)
                continue

            # Render with pbrt (reuse existing EXRs when --skip-render)
            if not skip_render:
                for sc in scenes:
                    run([PBRT, "--gpu", str(sc)])
            else:
                print(f"  skip-render: reusing {len(scenes)} existing EXRs", flush=True)

            # Apply sensor model
            for sc in scenes:
                hue = sc.stem[len("munsell_"):]
                exr = REPO / f"out/munsell_{hue}_spectral.exr"
                if not exr.exists():
                    print(f"  !! missing EXR {exr.name}; skip", flush=True)
                    continue

                manifest = sc.parent / f"munsell_{hue}_manifest.json"
                npz = REPO / f"out/munsell_sensor_forward/munsell_{hue}_electrons.npz"
                npz.parent.mkdir(parents=True, exist_ok=True)

                # Sensor forward with protocol's settings
                run([PY, "tools/pbrt_spectral_exr_to_electrons.py",
                     "--repo-root", str(REPO),
                     "--exr", str(exr),
                     "--camera-model-config", str(camera_cfg),
                     "--out", str(npz),
                     "--target-illuminance-lux", str(protocol["illuminance_lux"]),
                     "--integration-time-s", str(protocol["exposure_s"]),
                     "--scene-manifest-json", str(manifest)], quiet=True)

                # Apply noise
                noise_cmd = [PY, "tools/apply_emva_noise.py",
                             "--repo-root", str(REPO),
                             "--camera-model-config", str(camera_cfg),
                             "--seed", "0",
                             "--linear-exr", str(exr),
                             "--electrons-npz", str(npz),
                             "--preview-percentile", "99.5",
                             "--preview-no-normalize",
                             "--preview-white-balance-enabled", "false",
                             "--preview-color-correction-enabled", "false",
                             "--integration-time-s", str(protocol["exposure_s"])]
                if emit_linear16:
                    noise_cmd.append("--emit-demosaic-linear16")
                run(noise_cmd, quiet=True)

                # Copy outputs
                dest = DATASET / "munsell" / proto_name / ilabel / hue
                dest.mkdir(parents=True, exist_ok=True)
                prev = REPO / "out/colorchecker_noisy_png"
                copy_if(prev / "noisy_demosaic_rgb8.png", dest / "noisy_demosaic_rgb8.png")
                copy_if(prev / "clean_demosaic_rgb8.png", dest / "clean_demosaic_rgb8.png")
                copy_if(prev / "clean_demosaic_linear_rgb16.png",
                        dest / "clean_demosaic_linear_rgb16.png")
                copy_if(prev / "noisy_demosaic_linear_rgb16.png",
                        dest / "noisy_demosaic_linear_rgb16.png")
                copy_if(prev / "noisy_rgb8.png", dest / "noisy_mosaic_rgb8.png")
                copy_if(prev / "clean_rgb8.png", dest / "clean_mosaic_rgb8.png")
                copy_if(prev / "noisy_mono_16.png", dest / "noisy_mosaic_mono_16.png")

            hue_count = len(list((DATASET / "munsell" / proto_name / ilabel).glob("*")))
            print(f"  {proto_name:12s} {ilabel:6s}: {hue_count} hues", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hues", default="all", help="Hue families to generate (e.g. 'R,Y,G' or 'all')")
    ap.add_argument("--spp", type=int, default=4096, help="Samples per pixel for rendering")
    ap.add_argument("--xres", type=int, default=960)
    ap.add_argument("--yres", type=int, default=640)
    ap.add_argument("--skip-render", action="store_true",
                    help="Reuse existing rendered EXRs and scenes (no GPU re-render).")
    ap.add_argument("--emit-demosaic-linear16", action="store_true",
                    help="Also write linear 16-bit demosaiced images.")
    args = ap.parse_args()

    t0 = time.perf_counter()
    generate_munsell(PROTOCOLS, ILLUMINANTS, args.hues, args.spp, args.xres, args.yres,
                     skip_render=args.skip_render,
                     emit_linear16=args.emit_demosaic_linear16)

    elapsed = time.perf_counter() - t0
    print(f"\n✅ Done in {elapsed:.1f}s. Munsell dataset at {DATASET / 'munsell'}", flush=True)


if __name__ == "__main__":
    main()
