#!/usr/bin/env python3
"""Patch-pair defect-pixel sweep under FLAT illumination.

Same 15 ColorChecker patch pairs as the gradient dataset, but lit by a flat (distant,
head-on) light and swept over increasing levels of defect pixels — a mix of stuck-high
(white) and stuck-low (dead/black) photosites — through the iPhone-8 EMVA sensor.

Defects are a RAW-domain artifact applied in the noise stage, so each pair is rendered
ONCE and the defect levels are cheap re-runs (--skip-render). A shared noise seed makes
the defect sets nested (each level adds to the previous). The persistent defect map is
saved per level as ground truth.

Layout (out/patch_pairs_defects/):
  pair_LL_RR/defect<N>/noisy_mono_16.png      # RAW mosaic — defects sharpest here
                      /noisy_srgb8.png         # processed sRGB (from mono; demosaic-spread)
                      /defect_map.npz          # stuck_high/stuck_low/hot masks (ground truth)
                      /run_stats.json, params.json
  pair_LL_RR/contact.png    # the defect levels side by side
  manifest.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tools"))
import importlib.util
_spec = importlib.util.spec_from_file_location("gpp", REPO / "scripts" / "generate_patch_pair_dataset.py")
gpp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gpp)
from camera_model import load_camera_model  # noqa: E402
from scene_illuminance_probe import derive_reference  # noqa: E402

# Flat illumination (distant, head-on) — the light is uniform; only defect density varies.
LIGHT = dict(type="distant", direction="north", angle_deg=0.0, distance=3.0, scale=2.0,
             ambient_scale=0.0)
TARGET_LUX = 1000          # flat illuminance on the patches (well-exposed mid-tone)
EXPOSURE_S = 0.050
ISO_GAIN = 1.0

# Defect levels: equal stuck-low (dead/black) and stuck-high (white) rates. On a 960x640
# mosaic (~614k px) these are ~30 / 123 / 614 / 3070 of EACH type. Shared seed => nested.
DEFECT_LEVELS = [5e-5, 2e-4, 1e-3, 5e-3]
DEFECT_SEED = 7

PAIRS = gpp.PAIRS
PNG_DIR = gpp.PNG_DIR


def builder_extra_args(left: int, right: int) -> list[str]:
    return ["--left-patch", str(left), "--right-patch", str(right),
            "--light-type", LIGHT["type"], "--light-direction", LIGHT["direction"],
            "--light-angle-deg", str(LIGHT["angle_deg"]),
            "--light-distance", str(LIGHT["distance"]),
            "--ambient-scale", str(LIGHT["ambient_scale"])]


def measure_flat_reference(gpu: bool) -> float:
    left, right = 3, 8
    gpp.run([str(REPO / "venv/bin/python"), "tools/build_patch_pair_scene.py",
             "--repo-root", str(REPO), "--left-patch", str(left), "--right-patch", str(right),
             "--light-type", LIGHT["type"], "--light-direction", LIGHT["direction"],
             "--light-angle-deg", str(LIGHT["angle_deg"]), "--light-distance", str(LIGHT["distance"]),
             "--light-scale", str(LIGHT["scale"]), "--ambient-scale", str(LIGHT["ambient_scale"]),
             "--camera", "realistic", "--lensfile", gpp.LENSFILE,
             "--aperture-diameter-mm", str(gpp.APERTURE_MM), "--focus-distance", str(gpp.CAM_DIST),
             "--cam-dist", str(gpp.CAM_DIST), "--film", "spectral", "--spectral-nbuckets", "16",
             "--pixelsamples", "64", "--film-output", "out/patch_pair_probe.exr"], quiet=True)
    scene = REPO / "scenes" / "generated" / f"patch_pair_{left:02d}_{right:02d}.pbrt"
    probe = derive_reference(scene, REPO / "out/patch_pair_defect_probe", gpu=gpu)
    print(f"flat illuminance reference: {probe.e0_lux_exr:.1f} raw-EXR lux", flush=True)
    return probe.e0_lux_exr


def make_configs(cfg_dir: Path, left: int, right: int, rate: float, defect_map_rel: str,
                 spp: int, reference_lux_exr: float, gpu: bool) -> tuple[Path, Path]:
    cfg_dir.mkdir(parents=True, exist_ok=True)
    scene_rel = f"scenes/generated/patch_pair_{left:02d}_{right:02d}.pbrt"
    cfg = yaml.safe_load((REPO / "config/pipeline.yaml").read_text())
    r = cfg.setdefault("render", {})
    r["cam_dist"] = gpp.CAM_DIST
    r["pixelsamples"] = spp
    r["illuminant"] = gpp.D65_CSV
    r["gpu_enabled"] = bool(gpu)
    r["film"] = "spectral"
    r["film_output"] = "out/colorchecker_spectral.exr"
    r["light_scale"] = LIGHT["scale"]
    r["builder_extra_args"] = builder_extra_args(left, right)
    cfg.setdefault("paths", {})["scene_builder"] = "tools/build_patch_pair_scene.py"
    cfg["paths"]["scene_file"] = scene_rel
    lo = cfg.get("lens_overrides") or {}
    lo["realistic_focus_distance"] = gpp.CAM_DIST
    cfg["lens_overrides"] = lo
    cfg["exposure_time_override_s"] = EXPOSURE_S
    cfg.setdefault("sensor_forward", {})["target_illuminance_lux"] = float(TARGET_LUX)
    cfg.setdefault("noise", {})["preview_white_balance_enabled"] = False
    cfg["noise"]["seed"] = DEFECT_SEED  # shared across levels -> nested defect sets
    ppath = cfg_dir / f"pipeline_d.yaml"
    ppath.write_text(yaml.safe_dump(cfg, sort_keys=False))

    cm = load_camera_model(REPO / gpp.RECIPE)
    cm["lens"]["realistic_focus_distance"] = float(gpp.CAM_DIST)
    emva = cm["noise"]["emva"]
    emva["iso_gain_factor"] = ISO_GAIN
    emva["spatial_noise_seed"] = gpp.SPATIAL_SEED
    emva["defect_pixels"] = {
        "enabled": True,
        "stuck_low_rate": float(rate),    # dead / black
        "stuck_high_rate": float(rate),   # stuck / white (defaults to full-well value)
        "persistent_map_npz": defect_map_rel,
    }
    sf = cm["sensor_forward"]["model"]
    cal = sf["calibration"]
    cal["illuminant_override_csv"] = None
    sf.setdefault("pbrt_spectral_exr", {})["radiometric_autocalibration"] = "off"
    cal["irradiance_scale_W_m2nm_per_unit"] = 1.0
    cal["scene_illuminance_reference_exr"] = float(reference_lux_exr)
    cpath = cfg_dir / f"camera_d.yaml"
    cpath.write_text(yaml.safe_dump(cm, sort_keys=False))
    return ppath, cpath


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=REPO / "out" / "patch_pairs_defects")
    ap.add_argument("--spp", type=int, default=1024)
    ap.add_argument("--gpu", choices=("true", "false"), default="true")
    ap.add_argument("--pairs", type=str, default=None, help='e.g. "3-8,8-13"')
    args = ap.parse_args()
    gpu = args.gpu == "true"
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = PAIRS if not args.pairs else [tuple(int(x) for x in p.split("-")) for p in args.pairs.split(",")]

    reference = measure_flat_reference(gpu)
    manifest = {"illumination": "flat", "light": LIGHT, "target_lux": TARGET_LUX,
                "camera": "iphone_8_rggb", "lens": "realistic", "exposure_s": EXPOSURE_S,
                "iso_gain": ISO_GAIN, "reference_lux_exr": reference,
                "defect_levels": DEFECT_LEVELS, "defect_seed": DEFECT_SEED, "pairs": []}

    for left, right in pairs:
        tag = f"{left:02d}_{right:02d}"
        pair_dir = out_dir / f"pair_{tag}"
        print(f"\n==== pair {left}-{right} ====", flush=True)
        entry = {"pair": [left, right], "dir": f"pair_{tag}", "levels": []}
        thumbs = []
        for i, rate in enumerate(DEFECT_LEVELS, start=1):
            lvl_dir = pair_dir / f"defect{i}"
            lvl_dir.mkdir(parents=True, exist_ok=True)
            # apply_emva_noise resolves persistent_map_npz under the repo root, so use a
            # repo-relative temp path and move the saved mask into the variant dir after.
            defect_map_rel = "out/_defect_map_tmp.npz"
            (REPO / defect_map_rel).unlink(missing_ok=True)  # force fresh generation
            print(f"-- {tag} defect{i}: rate {rate} (each type) --", flush=True)
            ppath, cpath = make_configs(lvl_dir / "_configs", left, right, rate,
                                        defect_map_rel, args.spp, reference, gpu)
            gpp.run_pipeline(ppath, cpath, f"pair_{tag}_defect{i}", skip_render=(i > 1))
            # Copy raw mono + the pipeline's demosaiced-linear RGB (clean reference +
            # noisy). These are true 16-bit PNGs — read with a 16-bit-capable reader
            # (imageio 'PNG-FI' / cv2 IMREAD_UNCHANGED); PIL downconverts them to 8-bit.
            for name in ("noisy_mono_16.png", "clean_demosaic_linear_rgb16.png",
                         "noisy_demosaic_linear_rgb16.png", "run_stats.json"):
                shutil.copy2(PNG_DIR / name, lvl_dir / name)
            # Ground-truth defect masks.
            if (REPO / defect_map_rel).exists():
                shutil.move(str(REPO / defect_map_rel), str(lvl_dir / "defect_map.npz"))
            stats = json.loads((lvl_dir / "run_stats.json").read_text())
            dp = stats.get("defect_pixels", {})
            params = {"pair": [left, right], "defect_level": i, "rate_each_type": rate,
                      "illumination": "flat", "target_lux": TARGET_LUX,
                      "stuck_high_count": dp.get("stuck_high_count"),
                      "stuck_low_count": dp.get("stuck_low_count")}
            (lvl_dir / "params.json").write_text(json.dumps(params, indent=2) + "\n")
            entry["levels"].append({"defect_level": i, "rate": rate,
                                    "stuck_high_count": dp.get("stuck_high_count"),
                                    "stuck_low_count": dp.get("stuck_low_count"),
                                    "dir": f"pair_{tag}/defect{i}"})

        # Processed sRGB (from mono; shared exposure = defect1's 99.7 pct) + contact.
        white = float(np.percentile(gpp._mono16_to_signal_rgb(pair_dir / "defect1" / "noisy_mono_16.png"), 99.7))
        for i in range(1, len(DEFECT_LEVELS) + 1):
            lvl_dir = pair_dir / f"defect{i}"
            img = gpp._to_srgb8(gpp._mono16_to_signal_rgb(lvl_dir / "noisy_mono_16.png"), white)
            img.save(lvl_dir / "noisy_srgb8.png")
            thumbs.append(img.convert("RGB"))
        tw = 300
        rs = [im.resize((tw, int(im.size[1] * tw / im.size[0]))) for im in thumbs]
        strip = Image.new("RGB", (len(rs) * (tw + 8) + 8, rs[0].size[1] + 8), (20, 20, 20))
        for i, im in enumerate(rs):
            strip.paste(im, (8 + i * (tw + 8), 4))
        strip.save(pair_dir / "contact.png")
        manifest["pairs"].append(entry)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote manifest: {out_dir / 'manifest.json'}\nDataset root: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
