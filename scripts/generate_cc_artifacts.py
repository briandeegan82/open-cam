#!/usr/bin/env python3
"""Generate a ColorChecker + D65 artifact-injection test dataset.

The ColorChecker is rendered under CIE D65 **once** (iphone_8_rggb, iPhone-8 RGGB
sensor); every artifact variant is then a cheap post-render pass through the EMVA
noise model (``run_pipeline.py --skip-render``), which reuses the rendered spectral
EXR.  Three artifacts are swept as *independent ladders* — one artifact at a time at
four increasing strengths, the other two held off — plus a clean (artifact-free)
reference:

  dead_pixels    stuck-low (dead) photosites          emva.defect_pixels.stuck_low_rate
  uneven_illum   cos^4 lens vignette (corner falloff)  emva.illum_vignette_strength
  row_col_noise  fixed row+column FPN (electrons)      emva.{row,column}_fpn_fixed_std_e

Each variant ships full ground-truth labels: the shared clean reference images, a
per-image ``params.json``, the dead-pixel mask (``defect_map.npz``), and the
illumination gain map (``illum_gain_map.npy``).

Both a demosaiced RGB view and the un-demosaiced RAW mono (Bayer) frame are saved.
Row/column FPN and dead pixels are RAW-domain artifacts that bilinear demosaic
smears and attenuates, so ``noisy_mono_16.png`` is the full-fidelity view of those;
the demosaiced RGB shows how each artifact propagates through the ISP.

Layout (under --out-dir, default out/cc_artifacts):
  clean/clean_demosaic_linear_rgb16.png          # noise-free demosaiced reference
       /reference_mono_16.png                     # baseline RAW mono (no artifact)
  <artifact>/L{1..4}/noisy_demosaic_linear_rgb16.png   # corrupted, demosaiced
                    /noisy_mono_16.png                  # corrupted, RAW mono
                    /defect_map.npz                     # dead-pixel masks
                    /illum_gain_map.npy                 # illumination field
                    /run_stats.json
                    /params.json
  manifest.json
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
PY = str(REPO / "venv" / "bin" / "python")

sys.path.insert(0, str(REPO / "tools"))
from apply_emva_noise import build_illumination_field  # noqa: E402
from camera_model import load_camera_model  # noqa: E402
from scene_illuminance_probe import derive_reference, verify_chart_exr  # noqa: E402

RECIPE = "config/camera_recipes/iphone_8_rggb.yaml"
D65_CSV = "spectra/illuminant/interpolated/D65.csv"
CC_DIST = 3.75  # ColorChecker framing (see dataset-cam-dist-reframe memory)
SPATIAL_SEED = 1234  # fixed PRNU/DSNU/FPN-base pattern shared across the whole dataset

# D65 exposure protocol: well-exposed baseline (lux, integration time, ISO).
PROTOCOL = {"illuminance_lux": 1000.0, "exposure_s": 0.050, "iso_gain": 1.0}
# Low-light protocol for the per-patch mode: at ~tens of electrons of signal, fixed
# row/column FPN and read noise become a visible fraction, so row noise is a factor.
PROTOCOL_LOWLIGHT = {"illuminance_lux": 20.0, "exposure_s": 0.050, "iso_gain": 8.0}

# Per-patch mode: each patch is rendered as a full-frame plane with the same realistic
# camera as the chart, so the illuminance anchor stays valid. Natural lens falloff is
# present in both the corrupted and reference frames and cancels in their ratio.
N_PATCHES = 24
# Fixed row/column FPN (electrons) baked into every per-patch frame so row noise shows
# at low light; the per-patch reference has it disabled for clean isolation.
PATCH_FPN_E = 2.0
# Random per-patch illumination-uniformity field bounds (drawn independently per patch).
PATCH_VIGNETTE_MAX = 0.25   # up to 25% corner falloff within a square
PATCH_GRADIENT_MAX = 0.15   # up to 15% peak-to-peak tilt across a square
PATCH_UNIFORMITY_SEED = 20260806  # independent of SPATIAL_SEED; fixes the per-patch draws

# Independent artifact ladders. Four increasing strengths each; L0 (clean) is the
# separate artifact-free reference render.  Values are chosen against the iphone_8_rggb
# sensor (read noise sigma_d = 1.6 e-, full well 9500 e-, 960x640 mosaic ~= 614k px):
#   dead:  ~6, ~31, ~123, ~614 dead photosites
#   illum: 5..50% corner falloff
#   fpn:   sub-read-noise (0.5 e-) up to ~6x read noise (10 e-)
LADDERS = {
    "dead_pixels": [1e-5, 5e-5, 2e-4, 1e-3],
    "uneven_illum": [0.05, 0.15, 0.30, 0.50],
    "row_col_noise": [0.5, 1.5, 4.0, 10.0],
}
BASE_ARTIFACTS = list(LADDERS)
# "combined" stacks all three base artifacts at each ladder level onto one frame.
ALL_ARTIFACTS = BASE_ARTIFACTS + ["combined"]
N_LEVELS = 4

# Intermediate files run_pipeline (apply_emva_noise) writes; cleared before each run so
# a failed regenerate leaves the destination missing rather than reusing a stale file.
INTERMEDIATE = [
    "clean_demosaic_linear_rgb16.png",
    "noisy_demosaic_linear_rgb16.png",
    "clean_demosaic_rgb8.png",
    "noisy_demosaic_rgb8.png",
    "noisy_mono_16.png",
    "run_stats.json",
]
PNG_DIR = REPO / "out" / "colorchecker_noisy_png"
ELECTRONS_NPZ = REPO / "out" / "sensor_forward_electrons.npz"
EXR_OUT = REPO / "out" / "colorchecker_spectral.exr"


def run(cmd: list[str]) -> None:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=str(REPO), check=True)


def apply_chart_radiometry(camera_model: dict, reference_lux_exr: float) -> None:
    """Anchor chart-scene radiometry to a surround-probe illuminance reference.

    Mirrors scripts/generate_dual_protocol.py: both legacy lux normalisations are
    disabled (or the target lux would be applied twice — the original lux**2 bug) and
    the electrons stage uses E_e = L * rad_to_e * target_lux / reference.
    """
    sf_model = camera_model["sensor_forward"]["model"]
    cal = sf_model["calibration"]
    cal["illuminant_override_csv"] = None
    sf_model.setdefault("pbrt_spectral_exr", {})["radiometric_autocalibration"] = "off"
    cal["irradiance_scale_W_m2nm_per_unit"] = 1.0
    cal["scene_illuminance_reference_exr"] = float(reference_lux_exr)


def make_configs(out_dir: Path, spp: int, reference_lux_exr: float,
                 emva_overrides: dict, gpu: bool = True) -> tuple[Path, Path]:
    """Write pipeline + camera-model YAMLs for one variant.

    ``emva_overrides`` is a nested dict merged into camera_model['noise']['emva']
    (e.g. {'illum_vignette_strength': 0.3} or {'defect_pixels': {...}}).
    """
    cdir = out_dir / "_configs"
    cdir.mkdir(parents=True, exist_ok=True)
    tag = "_".join(f"{k}" for k in sorted(emva_overrides)) or "clean"

    cfg = yaml.safe_load((REPO / "config/pipeline.yaml").read_text())
    cfg.setdefault("render", {})["cam_dist"] = CC_DIST
    cfg["render"]["pixelsamples"] = spp
    cfg["render"]["illuminant"] = D65_CSV
    cfg["render"]["gpu_enabled"] = bool(gpu)
    lo = cfg.get("lens_overrides") or {}
    lo["realistic_focus_distance"] = CC_DIST
    cfg["lens_overrides"] = lo
    cfg["exposure_time_override_s"] = PROTOCOL["exposure_s"]
    cfg.setdefault("sensor_forward", {})["target_illuminance_lux"] = PROTOCOL["illuminance_lux"]
    cfg.setdefault("noise", {})["preview_white_balance_enabled"] = False
    ppath = cdir / f"pipeline_{tag}.yaml"
    ppath.write_text(yaml.safe_dump(cfg, sort_keys=False))

    camera_model = load_camera_model(REPO / RECIPE)
    emva = camera_model["noise"]["emva"]
    emva["iso_gain_factor"] = PROTOCOL["iso_gain"]
    # Pin the spatial-noise seed so every variant (and the reference) is the SAME camera
    # unit: identical PRNU/DSNU and fixed-FPN base pattern.  Without this the seed
    # defaults to a hash of the (per-variant) config path, so the fixed sensor pattern
    # would differ between reference and corrupted frames and contaminate differencing.
    emva["spatial_noise_seed"] = SPATIAL_SEED
    camera_model["lens"]["realistic_focus_distance"] = float(CC_DIST)
    apply_chart_radiometry(camera_model, reference_lux_exr)
    # Merge artifact knobs (one level of nesting for defect_pixels).
    for key, val in emva_overrides.items():
        if isinstance(val, dict):
            emva.setdefault(key, {}).update(val)
        else:
            emva[key] = val
    cpath = cdir / f"camera_model_{tag}.yaml"
    cpath.write_text(yaml.safe_dump(camera_model, sort_keys=False))
    return ppath, cpath


def make_patch_configs(out_dir: Path, spp: int, reference_lux_exr: float, patch_idx: int,
                       emva_overrides: dict, gpu: bool = True,
                       tag: str = "corrupted") -> tuple[Path, Path, str]:
    """Write pipeline + camera-model YAMLs for one low-light single-patch render.

    Drives run_pipeline to build+render patch ``patch_idx`` as a flat full-frame plane
    with a perspective camera (via builder_extra_args + scene_file), under the low-light
    protocol.  ``reference_lux_exr`` MUST be the perspective probe anchor (the realistic
    chart probe is not valid for a perspective camera).  Returns
    (pipeline_path, camera_path, scene_file_rel).
    """
    cdir = out_dir / "_configs"
    cdir.mkdir(parents=True, exist_ok=True)
    scene_rel = f"scenes/generated/colorchecker_patch_{patch_idx:02d}.pbrt"

    cfg = yaml.safe_load((REPO / "config/pipeline.yaml").read_text())
    cfg.setdefault("render", {})["cam_dist"] = CC_DIST
    cfg["render"]["pixelsamples"] = spp
    cfg["render"]["illuminant"] = D65_CSV
    cfg["render"]["gpu_enabled"] = bool(gpu)
    cfg["render"]["film"] = "spectral"
    cfg["render"]["film_output"] = "out/colorchecker_spectral.exr"  # == paths.exr_out
    cfg["render"]["builder_extra_args"] = ["--single-patch-index", str(patch_idx)]
    cfg.setdefault("paths", {})["scene_file"] = scene_rel
    lo = cfg.get("lens_overrides") or {}
    lo["realistic_focus_distance"] = CC_DIST
    cfg["lens_overrides"] = lo
    cfg["exposure_time_override_s"] = PROTOCOL_LOWLIGHT["exposure_s"]
    cfg.setdefault("sensor_forward", {})["target_illuminance_lux"] = PROTOCOL_LOWLIGHT["illuminance_lux"]
    cfg.setdefault("noise", {})["preview_white_balance_enabled"] = False
    ppath = cdir / f"pipeline_patch{patch_idx:02d}_{tag}.yaml"
    ppath.write_text(yaml.safe_dump(cfg, sort_keys=False))

    # Realistic camera (same as the chart mode): keeps the proven, camera-matched
    # radiometry anchor valid.  The lens illumination falloff is physically real and is
    # present in BOTH the corrupted and the per-patch reference frame, so it cancels in
    # their ratio while the injected per-square uniformity remains the labelled residual.
    camera_model = load_camera_model(REPO / RECIPE)
    camera_model["lens"]["realistic_focus_distance"] = float(CC_DIST)
    emva = camera_model["noise"]["emva"]
    emva["iso_gain_factor"] = PROTOCOL_LOWLIGHT["iso_gain"]
    emva["spatial_noise_seed"] = SPATIAL_SEED
    apply_chart_radiometry(camera_model, reference_lux_exr)
    for key, val in emva_overrides.items():
        if isinstance(val, dict):
            emva.setdefault(key, {}).update(val)
        else:
            emva[key] = val
    cpath = cdir / f"camera_model_patch{patch_idx:02d}_{tag}.yaml"
    cpath.write_text(yaml.safe_dump(camera_model, sort_keys=False))
    return ppath, cpath, scene_rel


def draw_patch_uniformity(patch_idx: int) -> dict:
    """Independent random illumination-uniformity field for one square (seeded)."""
    rng = np.random.default_rng(PATCH_UNIFORMITY_SEED + patch_idx)
    return {
        "illum_vignette_strength": float(rng.uniform(0.0, PATCH_VIGNETTE_MAX)),
        "illum_gradient_strength": float(rng.uniform(0.0, PATCH_GRADIENT_MAX)),
        "illum_gradient_angle_deg": float(rng.uniform(0.0, 360.0)),
    }


def clean_intermediate() -> None:
    for name in INTERMEDIATE:
        (PNG_DIR / name).unlink(missing_ok=True)


def run_variant(ppath: Path, cpath: Path, skip_render: bool, seed: int) -> None:
    clean_intermediate()
    if not skip_render:
        ELECTRONS_NPZ.unlink(missing_ok=True)
    cmd = [PY, "tools/run_pipeline.py",
           "--config", str(ppath),
           "--camera-model-config", str(cpath),
           "--emit-demosaic-linear16",
           "--name", f"cc_artifact_{cpath.stem}"]
    if skip_render:
        cmd.append("--skip-render")
    # run_pipeline reads noise.seed from the pipeline config; patch it in.
    cfg = yaml.safe_load(ppath.read_text())
    cfg.setdefault("noise", {})["seed"] = seed
    ppath.write_text(yaml.safe_dump(cfg, sort_keys=False))
    run(cmd)


def overrides_for(artifact: str, level_value: float, defect_map_rel: str) -> dict:
    if artifact == "dead_pixels":
        return {"defect_pixels": {"enabled": True,
                                  "stuck_low_rate": float(level_value),
                                  "persistent_map_npz": defect_map_rel}}
    if artifact == "uneven_illum":
        return {"illum_vignette_strength": float(level_value)}
    if artifact == "row_col_noise":
        return {"row_fpn_fixed_std_e": float(level_value),
                "column_fpn_fixed_std_e": float(level_value)}
    raise ValueError(f"unknown artifact {artifact!r}")


def build_variant(artifact: str, idx: int, defect_map_rel: str) -> tuple[dict, object]:
    """Return (emva_overrides, level_value) for ``artifact`` at ladder index ``idx``.

    ``combined`` stacks all three base artifacts at the same ladder index, so a single
    frame carries a dead-pixel field, a vignette and row/column FPN together.  The base
    ladders use disjoint override keys, so merging them never collides.
    """
    if artifact == "combined":
        overrides: dict = {}
        level_value: dict = {}
        for base in BASE_ARTIFACTS:
            overrides.update(overrides_for(base, LADDERS[base][idx], defect_map_rel))
            level_value[base] = LADDERS[base][idx]
        return overrides, level_value
    val = LADDERS[artifact][idx]
    return overrides_for(artifact, val, defect_map_rel), val


def save_labels(variant_dir: Path, artifact: str, level_value: float,
                overrides: dict) -> dict:
    """Copy the corrupted image + labels into variant_dir; return the params record."""
    variant_dir.mkdir(parents=True, exist_ok=True)
    # Corrupted images (demosaiced RGB + RAW mono) + run stats.
    for name in ("noisy_demosaic_linear_rgb16.png", "noisy_mono_16.png", "run_stats.json"):
        src = PNG_DIR / name
        if not src.exists():
            raise FileNotFoundError(f"expected pipeline output missing: {src}")
        shutil.copy2(src, variant_dir / name)

    # Illumination gain-map label (regenerated from params; ones map when not an
    # illumination variant). Read the rendered resolution from run_stats/electrons.
    stats = json.loads((variant_dir / "run_stats.json").read_text())
    h, w = _frame_shape(stats)
    vig = float(overrides.get("illum_vignette_strength", 0.0))
    gain = build_illumination_field((h, w), vignette_strength=vig)
    np.save(variant_dir / "illum_gain_map.npy", gain)

    # Dead-pixel mask label: apply_emva_noise persists it to the configured path.
    dp = overrides.get("defect_pixels", {})
    if dp.get("persistent_map_npz"):
        src = REPO / dp["persistent_map_npz"]
        if src.exists():
            shutil.copy2(src, variant_dir / "defect_map.npz")

    params = {
        "artifact": artifact,
        "level_value": level_value,
        "emva_overrides": overrides,
        "frame_shape_hw": [h, w],
        "illuminant": "D65",
        "protocol": PROTOCOL,
        "camera": "iphone_8_rggb",
    }
    (variant_dir / "params.json").write_text(json.dumps(params, indent=2) + "\n")
    return params


def _frame_shape(stats: dict) -> tuple[int, int]:
    """Recover the mosaic H×W from the cached electrons npz."""
    data = np.load(ELECTRONS_NPZ)
    arr = data[data.files[0]]
    return int(arr.shape[0]), int(arr.shape[1])


def _copy_patch_outputs(patch_dir: Path, prefix: str) -> Path:
    """Copy the pipeline's mono+demosaiced outputs with a name prefix; return run_stats."""
    pairs = {
        "noisy_mono_16.png": f"{prefix}_mono_16.png",
        "noisy_demosaic_linear_rgb16.png": f"{prefix}_demosaic_linear_rgb16.png",
    }
    for src_name, dst_name in pairs.items():
        src = PNG_DIR / src_name
        if not src.exists():
            raise FileNotFoundError(f"expected pipeline output missing: {src}")
        shutil.copy2(src, patch_dir / dst_name)
    stats_dst = patch_dir / f"{prefix}_run_stats.json"
    shutil.copy2(PNG_DIR / "run_stats.json", stats_dst)
    return stats_dst


def build_chart_probe(gpu: bool, pipe_render: dict):
    """Build the D65 ColorChecker scene (realistic camera) and derive the illuminance
    anchor from its surround-only probe.  Shared by the chart and per-patch modes: the
    anchor is illuminant/geometry/camera-only, so it is valid for any realistic-camera
    render (full chart or single patch) under the same lens and framing."""
    m = load_camera_model(REPO / RECIPE)
    lensfile = m["lens"]["realistic_lensfile"]
    aperture = float(m["lens"]["realistic_aperture_diameter_mm"])
    run([PY, "tools/build_colorchecker_scene.py", "--repo-root", str(REPO),
         "--light-scale", str(pipe_render.get("light_scale", 2.0)),
         "--xres", str(pipe_render.get("xres", 960)),
         "--yres", str(pipe_render.get("yres", 640)),
         "--pixelsamples", "64",
         "--film", "spectral",
         "--spectral-nbuckets", str(int(pipe_render.get("spectral_nbuckets", 16))),
         "--spectral-lambda-min", str(float(pipe_render.get("spectral_lambda_min", 360.0))),
         "--spectral-lambda-max", str(float(pipe_render.get("spectral_lambda_max", 830.0))),
         "--illuminant", D65_CSV,
         "--cam-dist", str(CC_DIST),
         "--camera", "realistic",
         "--lensfile", lensfile,
         "--aperture-diameter-mm", str(aperture),
         "--focus-distance", str(CC_DIST)])
    probe = derive_reference(REPO / "scenes/generated/colorchecker.pbrt",
                             REPO / "out/illuminance_probe", gpu=gpu)
    print(f"illuminance probe: E_scene_exr = {probe.e0_lux_exr:.1f} raw-EXR lux", flush=True)
    return probe


def generate_patches(out_dir: Path, spp: int, gpu: bool, seed: int,
                     reference_lux_exr: float, max_patches: int | None = None) -> dict:
    """Low-light per-patch mode: each ColorChecker patch rendered as its own full-frame
    plane (realistic camera) with an independent random illumination-uniformity field, at
    20 lux / 8x gain so fixed row/column FPN is a visible factor.

    Each patch gets a 'corrupted' frame (uniformity field + FPN) and a 'reference' frame
    (uniform illumination, no FPN — base sensor noise only) for clean isolation, plus the
    injected gain map and a params record.  ``reference_lux_exr`` is the realistic chart
    probe anchor (see build_chart_probe).
    """
    print("\n==== per-patch low-light uniformity mode (20 lux / 8x) ====", flush=True)
    patches_root = out_dir / "patches"
    patches_root.mkdir(parents=True, exist_ok=True)

    section = {"mode": "patches", "protocol": PROTOCOL_LOWLIGHT, "camera": "iphone_8_rggb",
               "lens": "realistic", "fpn_e": PATCH_FPN_E,
               "reference_lux_exr": reference_lux_exr, "patches": []}

    n_patches = N_PATCHES if max_patches is None else min(N_PATCHES, max_patches)
    for idx in range(1, n_patches + 1):
        patch_dir = patches_root / f"patch_{idx:02d}"
        patch_dir.mkdir(parents=True, exist_ok=True)
        unif = draw_patch_uniformity(idx)
        overrides = {**unif,
                     "row_fpn_fixed_std_e": PATCH_FPN_E,
                     "column_fpn_fixed_std_e": PATCH_FPN_E}
        print(f"\n-- patch {idx:02d}: vig={unif['illum_vignette_strength']:.3f} "
              f"grad={unif['illum_gradient_strength']:.3f}@{unif['illum_gradient_angle_deg']:.0f}deg --",
              flush=True)

        # Corrupted frame: full render of this patch, then noise with uniformity + FPN.
        pp, cp, _scene = make_patch_configs(patch_dir, spp, reference_lux_exr, idx,
                                            overrides, gpu=gpu, tag="corrupted")
        run_variant(pp, cp, skip_render=False, seed=seed)
        _copy_patch_outputs(patch_dir, "noisy")

        h, w = _frame_shape({})
        gain = build_illumination_field(
            (h, w),
            vignette_strength=unif["illum_vignette_strength"],
            gradient_strength=unif["illum_gradient_strength"],
            gradient_angle_deg=unif["illum_gradient_angle_deg"],
        )
        np.save(patch_dir / "illum_gain_map.npy", gain)

        # Reference frame: uniform illumination, no FPN — reuse this patch's electrons.
        pp2, cp2, _ = make_patch_configs(patch_dir, spp, reference_lux_exr, idx,
                                         {}, gpu=gpu, tag="reference")
        run_variant(pp2, cp2, skip_render=True, seed=seed)
        _copy_patch_outputs(patch_dir, "reference")

        params = {
            "patch_index": idx,
            "uniformity": unif,
            "fpn_e": PATCH_FPN_E,
            "emva_overrides": overrides,
            "frame_shape_hw": [h, w],
            "illuminant": "D65", "protocol": PROTOCOL_LOWLIGHT, "camera": "iphone_8_rggb",
        }
        (patch_dir / "params.json").write_text(json.dumps(params, indent=2) + "\n")
        section["patches"].append({
            "patch_index": idx, "dir": str(patch_dir.relative_to(out_dir)),
            "corrupted_mono": f"patches/patch_{idx:02d}/noisy_mono_16.png",
            "corrupted_rgb": f"patches/patch_{idx:02d}/noisy_demosaic_linear_rgb16.png",
            "reference_mono": f"patches/patch_{idx:02d}/reference_mono_16.png",
            "reference_rgb": f"patches/patch_{idx:02d}/reference_demosaic_linear_rgb16.png",
            "illum_gain_map": f"patches/patch_{idx:02d}/illum_gain_map.npy",
            "params": params,
        })

    (patches_root / "manifest.json").write_text(json.dumps(section, indent=2) + "\n")
    print(f"\nWrote per-patch manifest: {patches_root / 'manifest.json'}", flush=True)
    return section


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=REPO / "out" / "cc_artifacts")
    ap.add_argument("--spp", type=int, default=1024,
                    help="Pixel samples for the single D65 render (default 1024).")
    ap.add_argument("--seed", type=int, default=0, help="Base noise seed.")
    ap.add_argument("--artifacts", nargs="*", default=ALL_ARTIFACTS,
                    choices=ALL_ARTIFACTS,
                    help="Subset of artifacts to sweep. 'combined' stacks all three base "
                    "artifacts at each level onto one frame.")
    ap.add_argument("--gpu", choices=("true", "false"), default="true",
                    help="Render on GPU (OptiX). Use false to render on CPU, e.g. when "
                    "the GPU is busy with other work.")
    ap.add_argument("--mode", choices=("chart", "patches", "both"), default="both",
                    help="'chart': D65 full-chart artifact ladders. 'patches': low-light "
                    "per-patch uniformity mode. 'both' (default): run each.")
    ap.add_argument("--max-patches", type=int, default=None,
                    help="Limit the per-patch mode to the first N patches (default: all 24).")
    args = ap.parse_args()
    gpu = args.gpu == "true"

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe_render = yaml.safe_load((REPO / "config/pipeline.yaml").read_text())["render"]

    # --- Radiometric anchor: build the CC scene once and derive the illuminance
    #     reference from a surround-only probe (mirrors generate_dual_protocol.py).
    #     Shared by both modes (realistic-camera, illuminant/geometry-only). ---
    probe = build_chart_probe(gpu, pipe_render)

    # --- Per-patch low-light mode. ---
    if args.mode == "patches":
        generate_patches(out_dir, args.spp, gpu, args.seed, probe.e0_lux_exr, args.max_patches)
        print(f"Dataset root: {out_dir}", flush=True)
        return

    manifest: dict = {"illuminant": "D65", "camera": "iphone_8_rggb",
                      "protocol": PROTOCOL, "spp": args.spp,
                      "reference_lux_exr": probe.e0_lux_exr, "variants": []}

    # --- Clean reference: full render (no artifacts). Also produces the shared EXR
    #     and electrons cache that every variant reuses via --skip-render. ---
    print("\n==== clean reference (full render) ====", flush=True)
    ppath, cpath = make_configs(out_dir, args.spp, probe.e0_lux_exr, {}, gpu=gpu)
    run_variant(ppath, cpath, skip_render=False, seed=args.seed)
    worst = verify_chart_exr(EXR_OUT, probe)
    print(f"  verify chart EXR: surround vs probe within {100*worst:.2f}%", flush=True)
    clean_dir = out_dir / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PNG_DIR / "clean_demosaic_linear_rgb16.png",
                 clean_dir / "clean_demosaic_linear_rgb16.png")
    # Baseline RAW mono (base sensor noise, no injected artifact): the RAW-domain
    # reference for row/col-noise and dead-pixel variants.
    shutil.copy2(PNG_DIR / "noisy_mono_16.png", clean_dir / "reference_mono_16.png")
    ref_rgb_rel = "clean/clean_demosaic_linear_rgb16.png"
    ref_mono_rel = "clean/reference_mono_16.png"

    # --- Artifact ladders: each variant is a cheap --skip-render noise pass. ---
    for artifact in args.artifacts:
        print(f"\n==== {artifact} ladder ====", flush=True)
        for idx in range(N_LEVELS):
            level = f"L{idx + 1}"
            variant_dir = out_dir / artifact / level
            variant_dir.mkdir(parents=True, exist_ok=True)
            defect_map_rel = str((variant_dir / "_defect_map.npz").relative_to(REPO))
            overrides, level_value = build_variant(artifact, idx, defect_map_rel)
            print(f"\n-- {artifact}/{level}: {level_value} --", flush=True)
            _pp, cpath = make_configs(variant_dir, args.spp, probe.e0_lux_exr, overrides, gpu=gpu)
            run_variant(_pp, cpath, skip_render=True, seed=args.seed)
            params = save_labels(variant_dir, artifact, level_value, overrides)
            (REPO / defect_map_rel).unlink(missing_ok=True)  # tidy tmp; label copied
            manifest["variants"].append({
                "artifact": artifact, "level": level, "level_value": level_value,
                "dir": str(variant_dir.relative_to(out_dir)),
                "corrupted_rgb": f"{artifact}/{level}/noisy_demosaic_linear_rgb16.png",
                "corrupted_mono": f"{artifact}/{level}/noisy_mono_16.png",
                "reference_rgb": ref_rgb_rel, "reference_mono": ref_mono_rel,
                "params": params,
            })

    if args.mode == "both":
        manifest["patches"] = generate_patches(out_dir, args.spp, gpu, args.seed,
                                               probe.e0_lux_exr, args.max_patches)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote dataset manifest: {out_dir / 'manifest.json'}", flush=True)
    print(f"Dataset root: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
