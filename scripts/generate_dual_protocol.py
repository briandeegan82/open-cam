#!/usr/bin/env python3
"""Generate iPhone 8 ColorChecker dataset with dual exposure/gain protocols.

Protocols: 1000lux_1x, 100lux_2x, 20lux_8x
Illuminants: CWF (CIE F2), A (2856K)
Cameras: iphone_8 (Bayer GBRG), iphone_8_ryycy (RYYCy CFA)

Outputs under <dataset>/colorchecker/<protocol_label>/<illuminant>/, where <dataset> is
out/dataset_dual for the Bayer camera and out/dataset_dual_ryycy for the RYYCy camera
(alongside any existing flat D65 run at <dataset>/colorchecker/<protocol_label>/).
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
    "CWF": "spectra/illuminant/interpolated/F2_CWF.csv",  # CIE F2 cool white fluorescent
    "A": "spectra/illuminant/interpolated/A.csv",          # CIE A ~2856K incandescent
    "D65": "spectra/illuminant/interpolated/D65.csv",      # CIE D65 ~6504K daylight
}

LUX = {"low": 10.0, "med": 200.0, "high": 2000.0}

DEFAULT_FRAMING = {"cc": 6.0, "munsell": 16.0}


def run(cmd, quiet=False):
    if not quiet:
        print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=str(REPO), check=True)


def cam_lens(recipe_path: str):
    m = load_camera_model(REPO / recipe_path)
    lens = m["lens"]
    return lens["realistic_lensfile"], float(lens["realistic_aperture_diameter_mm"])


def make_cc_config(camera_key: str, cam_dist: float, protocol: dict, illuminant_csv: str,
                    ilabel: str, reference_lux_exr: float) -> tuple[Path, Path]:
    """Create pipeline and camera model configs with protocol-specific exposure/gain.

    Returns (pipeline_config_path, camera_model_config_path)
    """
    dataset = REPO / CAMERAS[camera_key]["dataset"]
    cfg = yaml.safe_load((REPO / "config/pipeline.yaml").read_text())
    cfg.setdefault("render", {})["cam_dist"] = cam_dist
    lo = cfg.get("lens_overrides") or {}
    lo["realistic_focus_distance"] = cam_dist
    cfg["lens_overrides"] = lo

    # Apply protocol settings to pipeline
    cfg["exposure_time_override_s"] = protocol["exposure_s"]
    cfg["sensor_forward"]["target_illuminance_lux"] = protocol["illuminance_lux"]
    cfg["render"]["pixelsamples"] = protocol["spp"]
    cfg["render"]["illuminant"] = illuminant_csv

    # Disable white balance
    cfg["noise"]["preview_white_balance_enabled"] = False

    cdir = dataset / "_configs"
    cdir.mkdir(parents=True, exist_ok=True)

    # Pipeline config
    ppath = cdir / f"pipeline_{camera_key}_{protocol['label']}_{ilabel}.yaml"
    ppath.write_text(yaml.safe_dump(cfg, sort_keys=False))

    # Camera model config with ISO gain.
    #
    # Always compose the full model from the camera *recipe* (config/camera_recipes/ ->
    # lens_models/ + sensor_models/), never from the legacy monolith in
    # config/camera_models/.  The two disagree materially for the iPhone 8 — the legacy
    # file carries K = 0.22 e/DN (a web-scraped value measured at amplified ISO, since
    # corrected to the base-ISO 9.906 e/DN), f/2.0 instead of f/1.8, no spatial
    # crosstalk, and an IRCF applied on top of QE curves that already include it.
    # An earlier version of this script took the recipe for the 1x protocol but the
    # legacy monolith for 2x/8x, which put the protocols of a single dataset on two
    # different sensor calibrations.
    camera_model = load_camera_model(REPO / CAMERAS[camera_key]["recipe"])
    camera_model["noise"]["emva"]["iso_gain_factor"] = protocol["iso_gain"]
    # The scene is framed and focused at cam_dist, and the electrons stage derives its
    # close-focus (1+m)^2 correction from lens.realistic_focus_distance; the lens
    # model's default (1.0 m) would overstate it by ~3.8% at this framing.
    camera_model["lens"]["realistic_focus_distance"] = float(cam_dist)
    apply_chart_radiometry(camera_model, reference_lux_exr)
    cpath = cdir / f"camera_model_{camera_key}_{protocol['label']}_{ilabel}.yaml"
    cpath.write_text(yaml.safe_dump(camera_model, sort_keys=False))

    return ppath, cpath


def apply_chart_radiometry(camera_model: dict, reference_lux_exr: float) -> None:
    """Configure chart-scene radiometry: scene-illuminance-reference anchoring.

    ``reference_lux_exr`` is the scene illuminance at the chart plane in raw-EXR
    photopic lux, measured per run from a surround-only probe render
    (tools/scene_illuminance_probe.py).  The electrons stage then computes
    ``E_e = L * rad_to_e * target_lux / reference``: linear in target lux,
    f-number-responsive (rad_to_e enters exactly once — any aperture weight or SPD
    normalisation pbrt bakes into the EXR is part of the measured reference and
    cancels), and illuminant-independent (pbrt normalises light SPDs photometrically,
    and so is the reference: A vs F2_CWF measured 0.09% apart).
    target_illuminance_lux means the illuminance ON THE CHART — a lux meter at the
    chart centre.  Full derivation, measurements and the rejected alternatives
    (lux**2 double-application, autocal f-number cancellation, chart-mode 7.37x
    cross-illuminant error, frame-mean autocal's 25-30% vignetting bias):
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


def copy_if(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


# Filenames tools/run_pipeline.py (via apply_emva_noise.py) writes into the shared
# out/colorchecker_noisy_png/ intermediate dir. Must be removed before each protocol x
# illuminant run so that a tool invocation which fails to regenerate one of these (e.g.
# --emit-demosaic-linear16 omitted) leaves the destination copy missing instead of
# silently reusing a stale file left over from a previous run.
INTERMEDIATE_OUTPUTS = [
    "clean_demosaic_rgb8.png",
    "noisy_demosaic_rgb8.png",
    "clean_demosaic_linear_rgb16.png",
    "noisy_demosaic_linear_rgb16.png",
    "clean_rgb8.png",
    "noisy_rgb8.png",
    "noisy_mono_16.png",
    "run_stats.json",
]


def clean_intermediate(png_dir: Path) -> None:
    for name in INTERMEDIATE_OUTPUTS:
        (png_dir / name).unlink(missing_ok=True)


def generate_colorchecker(cameras, protocols, illuminants, skip_render=False,
                          emit_linear16=False):
    """Generate ColorChecker for each illuminant x camera x protocol.

    Outputs land in <dataset>/colorchecker/<protocol_label>/<illuminant>/, alongside any
    existing flat (D65) run at <dataset>/colorchecker/<protocol_label>/.

    Illuminant is the outer loop because the pbrt render depends only on the scene,
    illuminant, spp and lens — not on the camera's QE/CFA (applied in the sensor-forward
    pass) nor on the protocol's exposure/gain (applied after the render, see
    run_pipeline.py's exposure_time_override_s).  Both cameras share the phone_wide_f18
    lens, so one render per (illuminant, spp) is reused by every camera x protocol below
    it via --skip-render, which reads back the same paths.exr_out.
    """
    print("\n==== ColorChecker ====", flush=True)
    cc_dist = DEFAULT_FRAMING["cc"]

    # Lens is shared by every camera in CAMERAS (asserted) so one scene/EXR per
    # (illuminant, spp) serves every camera, and one illuminance probe serves the run.
    lenses = {cam_lens(CAMERAS[c]["recipe"]) for c in cameras}
    if len(lenses) != 1:
        raise ValueError(
            f"cameras {list(cameras)} do not share one lens ({lenses}); EXR reuse "
            "across cameras would be invalid"
        )
    lensfile, aperture = lenses.pop()

    # Radiometric anchor: build the chart scene once (same knobs run_pipeline uses,
    # from config/pipeline.yaml) and derive the scene-illuminance reference from a
    # surround-only probe render.  The probe is illuminant- and spp-independent
    # (pbrt normalises light SPDs photometrically; measured A vs F2_CWF: 0.09%), and
    # every chart EXR is verified against the probe profile below, so drift between
    # this build and run_pipeline's rebuild fails loudly.
    pipe_render = yaml.safe_load((REPO / "config/pipeline.yaml").read_text())["render"]
    first_icsv = next(iter(illuminants.values()))
    run([PY, "tools/build_colorchecker_scene.py",
         "--repo-root", str(REPO),
         "--light-scale", str(pipe_render.get("light_scale", 2.0)),
         "--xres", str(pipe_render.get("xres", 960)),
         "--yres", str(pipe_render.get("yres", 640)),
         "--pixelsamples", "64",  # irrelevant: the probe sets its own spp
         "--film", "spectral",
         "--spectral-nbuckets", str(int(pipe_render.get("spectral_nbuckets", 16))),
         "--spectral-lambda-min", str(float(pipe_render.get("spectral_lambda_min", 360.0))),
         "--spectral-lambda-max", str(float(pipe_render.get("spectral_lambda_max", 830.0))),
         "--illuminant", first_icsv,
         "--cam-dist", str(cc_dist),
         "--camera", "realistic",
         "--lensfile", lensfile,
         "--aperture-diameter-mm", str(aperture),
         "--focus-distance", str(cc_dist)])
    probe = derive_reference(REPO / "scenes/generated/colorchecker.pbrt",
                             REPO / "out/illuminance_probe")
    print(f"illuminance probe: E_scene_exr = {probe.e0_lux_exr:.1f} raw-EXR lux "
          f"(surround rho = {probe.rho_surround})", flush=True)

    for ilabel, icsv in illuminants.items():
        print(f"\n== Illuminant: {ilabel} ==", flush=True)
        # Keyed by spp: a cached EXR is only reusable by runs asking for the same
        # sample count. Reset per illuminant since the EXR is illuminant-specific.
        rendered_spp: set[int] = set()

        for ckey in cameras:
            for pname, protocol in protocols.items():
                print(f"\n-- {ckey} / {protocol['label']} / {ilabel} --", flush=True)
                pipeline_cfg, camera_cfg = make_cc_config(ckey, cc_dist, protocol, icsv,
                                                          ilabel, probe.e0_lux_exr)

                clean_intermediate(REPO / "out/colorchecker_noisy_png")
                (REPO / "out/sensor_forward_electrons.npz").unlink(missing_ok=True)
                cmd = [PY, "tools/run_pipeline.py",
                       "--config", str(pipeline_cfg),
                       "--camera-model-config", str(camera_cfg),
                       "--name", f"cc_{ckey}_{protocol['label']}_{ilabel}"]
                reuse = skip_render or protocol["spp"] in rendered_spp
                if reuse:
                    cmd.append("--skip-render")
                if emit_linear16:
                    cmd.append("--emit-demosaic-linear16")
                run(cmd)
                if not reuse:
                    # Chart EXR must reproduce the probe's surround profile; a
                    # mismatch means run_pipeline's scene differs from the probe's
                    # (light scale / lens / framing drift) and the anchor is invalid.
                    worst = verify_chart_exr(REPO / "out/colorchecker_spectral.exr",
                                             probe)
                    print(f"  verify chart EXR: surround vs probe within "
                          f"{100*worst:.2f}%", flush=True)
                rendered_spp.add(protocol["spp"])

                # Copy outputs (demosaiced and mosaic images)
                base = REPO / "out"
                dataset = REPO / CAMERAS[ckey]["dataset"]
                outdir = dataset / "colorchecker" / protocol["label"] / ilabel
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
    ap.add_argument("--camera", choices=[*CAMERAS, "all"], default="all",
                    help="Which camera(s) to generate")
    ap.add_argument("--protocol", choices=["standard", "midlight", "lowlight", "all"],
                    default="all", help="Which protocol(s) to generate")
    ap.add_argument("--illuminant", choices=[*ILLUMINANTS, "all"],
                    default="all", help="Which illuminant(s) to generate")
    ap.add_argument("--skip-render", action="store_true",
                    help="Reuse existing rendered EXRs (no GPU re-render).")
    ap.add_argument("--emit-demosaic-linear16", action="store_true",
                    help="Also write linear 16-bit demosaiced images.")
    args = ap.parse_args()

    # Filter cameras
    cameras = list(CAMERAS) if args.camera == "all" else [args.camera]

    # Filter protocols
    protocols = PROTOCOLS.copy()
    if args.protocol == "standard":
        protocols = {k: v for k, v in protocols.items() if k == "standard"}
    elif args.protocol == "midlight":
        protocols = {k: v for k, v in protocols.items() if k == "midlight"}
    elif args.protocol == "lowlight":
        protocols = {k: v for k, v in protocols.items() if k == "lowlight"}

    # Filter illuminants
    illuminants = ILLUMINANTS.copy()
    if args.illuminant != "all":
        illuminants = {k: v for k, v in illuminants.items() if k == args.illuminant}

    t0 = time.perf_counter()
    if not args.skip_colorchecker:
        generate_colorchecker(cameras, protocols, illuminants, skip_render=args.skip_render,
                              emit_linear16=args.emit_demosaic_linear16)

    elapsed = time.perf_counter() - t0
    print(f"\n✅ Done in {elapsed:.1f}s", flush=True)
    for ckey in cameras:
        print(f"Dataset at {REPO / CAMERAS[ckey]['dataset']}", flush=True)


if __name__ == "__main__":
    main()
