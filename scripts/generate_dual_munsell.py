#!/usr/bin/env python3
"""Generate Munsell patches for three-protocol iPhone 8 dataset.

Protocols: 1000lux_1x, 100lux_2x, 20lux_8x
Illuminants: CWF (CIE F2), A (2856K)
Cameras: iphone_8 (Bayer GBRG), iphone_8_ryycy (RYYCy CFA)
Lux levels: med (200 lux) for each protocol's base illuminance

Outputs under <dataset>/munsell/ (illuminant subfolders alongside any existing D65
run), where <dataset> is out/dataset_dual for the Bayer camera and
out/dataset_dual_ryycy for the RYYCy camera.
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

sys.path.insert(0, str(REPO / "tools"))
from camera_model import load_camera_model  # noqa: E402
from scene_illuminance_probe import derive_reference, verify_chart_exr  # noqa: E402

# Camera variants. Both share the phone_wide_f18 lens and identical sensor electronics;
# they differ only in the CFA (QE curves), so a render is reusable across them.
CAMERAS = {
    "iphone_8": {
        "recipe": "config/camera_recipes/iphone_8.yaml",
        "dataset": "out/dataset_dual",
        "note": "Bayer GBRG, measured iPhone 8 QE",
    },
    "iphone_8_ryycy": {
        "recipe": "config/camera_recipes/iphone_8_ryycy.yaml",
        "dataset": "out/dataset_dual_ryycy",
        "note": "RYYCy CFA (R, Y->G, Cy->B), generic complementary QE",
    },
}

# Protocol definitions
PROTOCOLS = {
    "1000lux_1x": {
        "illuminance_lux": 1000.0,
        "exposure_s": 0.020,
        "iso_gain": 1.0,
    },
    "100lux_2x": {
        "illuminance_lux": 100.0,
        "exposure_s": 0.020,
        "iso_gain": 2.0,
    },
    "20lux_8x": {
        "illuminance_lux": 20.0,
        "exposure_s": 0.020,
        "iso_gain": 8.0,
    },
}

ILLUMINANTS = {
    "CWF": "spectra/illuminant/interpolated/F2_CWF.csv",  # CIE F2 cool white fluorescent
    "A": "spectra/illuminant/interpolated/A.csv",          # CIE A ~2856K incandescent
    "D65": "spectra/illuminant/interpolated/D65.csv",      # CIE D65 ~6504K daylight
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


def apply_chart_radiometry(camera_model: dict, reference_lux_exr: float) -> None:
    """Configure chart-scene radiometry: scene-illuminance-reference anchoring.

    ``reference_lux_exr`` is the scene illuminance at the chart plane in raw-EXR
    photopic lux, measured per run from a surround-only probe render
    (tools/scene_illuminance_probe.py).  The electrons stage then computes
    ``E_e = L * rad_to_e * target_lux / reference``: linear in target lux,
    f-number-responsive (rad_to_e enters exactly once), and illuminant-independent
    (pbrt normalises light SPDs photometrically, and so is the measured reference —
    A vs F2_CWF agree to 0.09%).  target_illuminance_lux is passed per protocol on
    the tool command line; its semantics are the illuminance ON THE CHART, i.e. a
    lux meter at the chart centre.  Full derivation, measurements and rejected
    alternatives (lux**2 double-application, autocal f-number cancellation, chart-mode
    7.37x cross-illuminant error, frame-mean autocal's 25-30% vignetting bias):
    docs/notes/chart_radiometry_calibration.md.
    """
    sf_model = camera_model["sensor_forward"]["model"]
    cal = sf_model["calibration"]
    # The two legacy lux normalisations are replaced by the reference; both must be
    # off or the target would be applied twice (the original lux**2 bug).
    cal["illuminant_override_csv"] = None
    sf_model.setdefault("pbrt_spectral_exr", {})["radiometric_autocalibration"] = "off"
    # The reference is expressed in raw-EXR lux, so the EXR-unit conversion is unity.
    cal["irradiance_scale_W_m2nm_per_unit"] = 1.0
    cal["scene_illuminance_reference_exr"] = float(reference_lux_exr)


def make_camera_model_config(camera_key: str, protocol_name: str, protocol: dict,
                             ilabel: str, reference_lux_exr: float) -> Path:
    """Create a camera model config carrying the protocol's ISO gain.

    Always composes the full model from the camera *recipe* (config/camera_recipes/ ->
    lens_models/ + sensor_models/), never from the legacy monolith in
    config/camera_models/.  The two disagree materially for the iPhone 8 — the legacy
    file carries K = 0.22 e/DN (a web-scraped value measured at amplified ISO, since
    corrected to the base-ISO 9.906 e/DN), f/2.0 instead of f/1.8, no spatial crosstalk,
    and an IRCF applied on top of QE curves that already include it.  An earlier version
    of this script took the recipe for the 1x protocol but the legacy monolith for
    2x/8x, which put the protocols of a single dataset on two different sensor
    calibrations.
    """
    camera_model = load_camera_model(REPO / CAMERAS[camera_key]["recipe"])
    camera_model["noise"]["emva"]["iso_gain_factor"] = protocol["iso_gain"]
    # The scene is framed and focused at the munsell cam_dist, and the electrons stage
    # derives its close-focus (1+m)^2 correction from lens.realistic_focus_distance;
    # the lens model's default (1.0 m) would overstate it by ~3.8% at this framing.
    camera_model["lens"]["realistic_focus_distance"] = float(DEFAULT_FRAMING["munsell"])
    apply_chart_radiometry(camera_model, reference_lux_exr)

    cdir = REPO / CAMERAS[camera_key]["dataset"] / "_configs"
    cdir.mkdir(parents=True, exist_ok=True)
    cpath = cdir / f"camera_model_{camera_key}_{protocol_name}_{ilabel}.yaml"
    cpath.write_text(yaml.safe_dump(camera_model, sort_keys=False))
    return cpath


def copy_if(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


# Filenames apply_emva_noise.py writes into the shared out/colorchecker_noisy_png/
# intermediate dir. Must be removed before each hue's run so that a tool invocation
# which fails to regenerate one of these (e.g. --emit-demosaic-linear16 omitted)
# leaves the destination copy missing instead of silently reusing a stale file
# left over from a previous hue/protocol/illuminant.
INTERMEDIATE_OUTPUTS = [
    "clean_demosaic_rgb8.png",
    "noisy_demosaic_rgb8.png",
    "clean_demosaic_linear_rgb16.png",
    "noisy_demosaic_linear_rgb16.png",
    "noisy_rgb8.png",
    "clean_rgb8.png",
    "noisy_mono_16.png",
    "run_stats.json",
]


def clean_intermediate(png_dir: Path) -> None:
    for name in INTERMEDIATE_OUTPUTS:
        (png_dir / name).unlink(missing_ok=True)


def generate_munsell(cameras, protocols, illuminants, hues, spp, xres, yres,
                     skip_render=False, emit_linear16=False):
    """Generate Munsell patches for each illuminant x camera x protocol.

    The render (pbrt scene + EXR) only depends on the illuminant and lens, not on the
    protocol's exposure/gain nor on the camera's QE/CFA (both applied in the later,
    CPU-only sensor-forward + noise passes).  So illuminant is the outer loop: each hue
    is rendered once per illuminant and its EXR is reused by every camera x protocol
    below it.  Both cameras share the phone_wide_f18 lens, so adding the RYYCy camera
    costs no extra GPU time.
    """
    print("\n==== MUNSELL PATCHES ====", flush=True)
    scene_root = REPO / "scenes/generated/munsell"
    cam_dist = DEFAULT_FRAMING["munsell"]
    # Lens is shared by every camera in CAMERAS; assert rather than silently render one
    # camera's scenes and reuse them for another with a different prescription.
    lenses = {cam_lens(REPO / CAMERAS[c]["recipe"]) for c in cameras}
    if len(lenses) != 1:
        raise ValueError(
            f"cameras {list(cameras)} do not share one lens ({lenses}); EXR reuse "
            "across cameras would be invalid"
        )
    lensfile, aperture = lenses.pop()

    # Filled on the first illuminant; the probe is illuminant-independent (pbrt
    # normalises light SPDs photometrically; measured A vs F2_CWF: 0.09%).
    probe = {"result": None}

    for ilabel, icsv in illuminants.items():
        print(f"\n== Illuminant: {ilabel} ==", flush=True)
        if not skip_render:
            if scene_root.exists():
                shutil.rmtree(scene_root)

            # Build Munsell scene
            run([PY, "tools/build_munsell_scenes.py",
                 "--repo-root", str(REPO),
                 "--out-dir", "scenes/generated/munsell",
                 "--illuminant", icsv,
                 "--hues", hues,
                 # Same light scale as the ColorChecker rig (config/pipeline.yaml) so one
                 # illuminance probe serves both chart types.
                 "--light-scale", "1.0",
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

        # Radiometric anchor: derive the scene-illuminance reference once per run from
        # a surround-only probe render (illuminant-independent, so the first scene of
        # the first illuminant suffices), then verify every chart EXR's surround
        # against the probe profile so any rig drift (light scale, lens, framing)
        # fails loudly instead of silently rescaling the dataset.
        if probe["result"] is None:
            probe["result"] = derive_reference(scenes[0], REPO / "out/illuminance_probe")
            print(f"  illuminance probe: E_scene_exr = "
                  f"{probe['result'].e0_lux_exr:.1f} raw-EXR lux "
                  f"(surround rho = {probe['result'].rho_surround})", flush=True)
        for sc in scenes:
            hue = sc.stem[len("munsell_"):]
            exr = REPO / f"out/munsell_{hue}_spectral.exr"
            if exr.exists():
                worst = verify_chart_exr(exr, probe["result"])
                print(f"  verify {exr.name}: surround vs probe within "
                      f"{100*worst:.2f}%", flush=True)

        for ckey in cameras:
            for proto_name, protocol in protocols.items():
                sensor_pass(ckey, proto_name, protocol, ilabel, scenes,
                            probe["result"].e0_lux_exr, emit_linear16=emit_linear16)


def sensor_pass(ckey, proto_name, protocol, ilabel, scenes, reference_lux_exr,
                emit_linear16=False):
    """Run the CPU-only sensor-forward + noise passes over already-rendered EXRs."""
    print(f"\n-- {ckey} / {proto_name} / {ilabel} --", flush=True)
    camera_cfg = make_camera_model_config(ckey, proto_name, protocol, ilabel,
                                          reference_lux_exr)
    dataset = REPO / CAMERAS[ckey]["dataset"]

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
        clean_intermediate(REPO / "out/colorchecker_noisy_png")
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
        dest = dataset / "munsell" / proto_name / ilabel / hue
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

    hue_count = len(list((dataset / "munsell" / proto_name / ilabel).glob("*")))
    print(f"  {ckey} {proto_name:12s} {ilabel:6s}: {hue_count} hues", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hues", default="all", help="Hue families to generate (e.g. 'R,Y,G' or 'all')")
    ap.add_argument("--camera", choices=[*CAMERAS, "all"], default="all",
                    help="Which camera(s) to generate")
    ap.add_argument("--illuminant", choices=[*ILLUMINANTS, "all"],
                    default="all", help="Which illuminant(s) to generate")
    ap.add_argument("--spp", type=int, default=4096, help="Samples per pixel for rendering")
    ap.add_argument("--xres", type=int, default=960)
    ap.add_argument("--yres", type=int, default=640)
    ap.add_argument("--skip-render", action="store_true",
                    help="Reuse existing rendered EXRs and scenes (no GPU re-render).")
    ap.add_argument("--emit-demosaic-linear16", action="store_true",
                    help="Also write linear 16-bit demosaiced images.")
    args = ap.parse_args()

    cameras = list(CAMERAS) if args.camera == "all" else [args.camera]

    illuminants = ILLUMINANTS.copy()
    if args.illuminant != "all":
        illuminants = {k: v for k, v in illuminants.items() if k == args.illuminant}

    t0 = time.perf_counter()
    generate_munsell(cameras, PROTOCOLS, illuminants, args.hues, args.spp, args.xres,
                     args.yres,
                     skip_render=args.skip_render,
                     emit_linear16=args.emit_demosaic_linear16)

    elapsed = time.perf_counter() - t0
    print(f"\n✅ Done in {elapsed:.1f}s", flush=True)
    for ckey in cameras:
        print(f"Munsell dataset at {REPO / CAMERAS[ckey]['dataset'] / 'munsell'}", flush=True)


if __name__ == "__main__":
    main()
