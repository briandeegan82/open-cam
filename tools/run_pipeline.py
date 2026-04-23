#!/usr/bin/env python3
"""Run the full ColorChecker -> PBRT -> EMVA pipeline with one command."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from camera_model import load_camera_model


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd: list[str], cwd: Path, dry_run: bool) -> dict:
    print("$", " ".join(cmd))
    entry = {"cmd": cmd, "cwd": str(cwd)}
    if dry_run:
        entry["returncode"] = None
        entry["dry_run"] = True
        return entry
    p = subprocess.run(cmd, cwd=str(cwd), check=False)
    entry["returncode"] = p.returncode
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}")
    return entry


def p(repo: Path, v: str) -> Path:
    return (repo / v).resolve()


def resolve_camera_model_path(repo: Path, paths: dict, cli_path: Path | None) -> Path:
    if cli_path is not None:
        return cli_path.resolve()
    model_name = paths.get("camera_model_name")
    model_cfg = paths.get("camera_model_config", None)
    if model_name and model_cfg:
        raise ValueError("set only one of paths.camera_model_name or paths.camera_model_config")
    if model_name:
        return (repo / "config" / "camera_models" / f"{model_name}.yaml").resolve()
    return p(repo, model_cfg or "config/camera_models/default.yaml")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    repo_default = Path(__file__).resolve().parent.parent
    ap.add_argument("--repo-root", type=Path, default=repo_default)
    ap.add_argument("--config", type=Path, default=None, help="Pipeline YAML (default: config/pipeline.yaml)")
    ap.add_argument(
        "--camera-model-config",
        type=Path,
        default=None,
        help="Camera model YAML (default: pipeline paths.camera_model_config)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print commands only; do not execute.")
    ap.add_argument("--name", type=str, default=None, help="Optional run name for manifest filename.")
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    cfg_path = (args.config or (repo / "config" / "pipeline.yaml")).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing pipeline config: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text())

    paths = cfg.get("paths", {})
    render = cfg.get("render", {})
    noise = cfg.get("noise", {})
    validate = cfg.get("validate", {})
    validate_emva = cfg.get("validate_emva", {})
    validate_demosaic = cfg.get("validate_demosaic", {})
    realistic_focus_distance_override = cfg.get("realistic_focus_distance_override", None)
    if realistic_focus_distance_override is not None:
        realistic_focus_distance_override = float(realistic_focus_distance_override)

    venv_python = repo / "venv" / "bin" / "python"
    py = str(venv_python if venv_python.is_file() else Path(sys.executable))

    scene_builder = p(repo, paths.get("scene_builder", "tools/build_colorchecker_scene.py"))
    pbrt_bin = p(repo, paths.get("pbrt", "third_party/pbrt-v4/build/pbrt"))
    noise_tool = p(repo, paths.get("noise_tool", "tools/apply_emva_noise.py"))
    sensor_forward_tool = p(repo, paths.get("sensor_forward_tool", "tools/spectral_sensor_forward.py"))
    pbrt_exr_electrons_tool = p(repo, paths.get("pbrt_exr_to_electrons_tool", "tools/pbrt_spectral_exr_to_electrons.py"))
    validate_tool = p(repo, paths.get("validate_tool", "tools/validate_colorchecker.py"))
    validate_demosaic_tool = p(repo, paths.get("validate_demosaic_tool", "tools/validate_demosaic_linear.py"))
    validate_emva_tool = p(repo, paths.get("validate_emva_tool", "tools/validate_emva_model.py"))
    emva_validation_report = p(repo, paths.get("emva_validation_report", "out/emva_validation_report.json"))
    scene_file = p(repo, paths.get("scene_file", "scenes/generated/colorchecker.pbrt"))
    exr_out = p(repo, paths.get("exr_out", "out/colorchecker.exr"))
    camera_model_cfg = resolve_camera_model_path(repo, paths, args.camera_model_config)
    if not camera_model_cfg.is_file():
        raise FileNotFoundError(f"missing camera model config: {camera_model_cfg}")
    camera_model = load_camera_model(camera_model_cfg)
    lens_cfg = camera_model.get("lens", {})
    psf_tool = p(repo, paths.get("psf_tool", "tools/apply_spectral_psf.py"))
    sensor_forward_npz = p(repo, paths.get("sensor_forward_electrons_npz", "out/sensor_forward_electrons.npz"))
    demosaic_metrics_json = p(repo, paths.get("demosaic_metrics_json", "out/demosaic_linear_metrics.json"))
    out_dir = p(repo, paths.get("out_dir", "out"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run and not pbrt_bin.is_file():
        raise FileNotFoundError(f"pbrt binary not found: {pbrt_bin}")

    log: list[dict] = []

    # 1) Build scene
    film = str(render.get("film", "rgb")).lower()
    if film not in ("rgb", "spectral"):
        raise ValueError(f'render.film must be "rgb" or "spectral", got {film!r}')
    build_cmd = [
        py,
        str(scene_builder),
        "--repo-root",
        str(repo),
        "--light-scale",
        str(render.get("light_scale", 2.0)),
        "--xres",
        str(render.get("xres", 960)),
        "--yres",
        str(render.get("yres", 640)),
        "--pixelsamples",
        str(render.get("pixelsamples", 64)),
        "--film",
        film,
    ]
    builder_extra_args = render.get("builder_extra_args", [])
    if builder_extra_args is None:
        builder_extra_args = []
    if not isinstance(builder_extra_args, list):
        raise TypeError("render.builder_extra_args must be a YAML list of CLI tokens")
    build_cmd.extend([str(tok) for tok in builder_extra_args])
    fo = render.get("film_output", None)
    if fo:
        build_cmd.extend(["--film-output", str(fo)])
    if film == "spectral":
        build_cmd.extend(
            [
                "--spectral-nbuckets",
                str(int(render.get("spectral_nbuckets", 16))),
                "--spectral-lambda-min",
                str(float(render.get("spectral_lambda_min", 360.0))),
                "--spectral-lambda-max",
                str(float(render.get("spectral_lambda_max", 830.0))),
            ]
        )
    build_cmd.extend(["--cam-dist", str(float(render.get("cam_dist", 4.25)))])
    cam = str(lens_cfg.get("camera", "perspective")).lower()
    if cam not in ("perspective", "realistic"):
        raise ValueError(f'lens.camera must be "perspective" or "realistic", got {cam!r}')
    build_cmd.extend(["--camera", cam])
    if cam == "realistic":
        build_cmd.extend(
            [
                "--lensfile",
                str(lens_cfg.get("realistic_lensfile", "scenes/lenses/wide_22mm.dat")),
                "--aperture-diameter-mm",
                str(float(lens_cfg.get("realistic_aperture_diameter_mm", 4.0))),
            ]
        )
        focus_dist = realistic_focus_distance_override
        if focus_dist is None:
            focus_dist = lens_cfg.get("realistic_focus_distance", None)
        if focus_dist is not None:
            build_cmd.extend(["--focus-distance", str(float(focus_dist))])
    log.append(run_cmd(build_cmd, repo, args.dry_run))

    # 2) Render with pbrt
    pbrt_cmd = [str(pbrt_bin), str(scene_file)]
    log.append(run_cmd(pbrt_cmd, repo, args.dry_run))

    # 2b) Optional post-render PSF / MTF blur
    post = lens_cfg.get("post_psf") or {}
    if bool(post.get("enabled", False)):
        psf_cmd = [
            py,
            str(psf_tool),
            "--repo-root",
            str(repo),
            "--camera-model-config",
            str(camera_model_cfg),
            "--exr-in",
            str(exr_out),
        ]
        log.append(run_cmd(psf_cmd, repo, args.dry_run))

    # 3) Validate scene/render
    if bool(validate.get("enabled", True)):
        validate_cmd = [py, str(validate_tool), "--repo-root", str(repo), "--exr", str(exr_out)]
        log.append(run_cmd(validate_cmd, repo, args.dry_run))

    if bool(validate_emva.get("enabled", False)):
        emva_cmd = [
            py,
            str(validate_emva_tool),
            "--repo-root",
            str(repo),
            "--camera-model-config",
            str(camera_model_cfg),
            "--json-out",
            str(emva_validation_report),
        ]
        log.append(run_cmd(emva_cmd, repo, args.dry_run))

    sensor_forward = cfg.get("sensor_forward", {})
    integration_time_override_s = cfg.get("exposure_time_override_s", None)
    if integration_time_override_s is not None:
        integration_time_override_s = float(integration_time_override_s)
    if bool(sensor_forward.get("enabled", False)):
        sf_mode = str(sensor_forward.get("mode", "analytic")).lower()
        sf_target_lux = sensor_forward.get("target_illuminance_lux", None)
        if sf_mode == "pbrt_exr":
            sf_cmd = [
                py,
                str(pbrt_exr_electrons_tool),
                "--repo-root",
                str(repo),
                "--exr",
                str(exr_out),
                "--camera-model-config",
                str(camera_model_cfg),
                "--out",
                str(sensor_forward_npz),
            ]
            if sf_target_lux is not None:
                sf_cmd.extend(["--target-illuminance-lux", str(float(sf_target_lux))])
            if integration_time_override_s is not None:
                sf_cmd.extend(["--integration-time-s", str(integration_time_override_s)])
        elif sf_mode == "analytic":
            sf_cmd = [py, str(sensor_forward_tool), "--repo-root", str(repo), "--camera-model-config", str(camera_model_cfg)]
            if sf_target_lux is not None:
                sf_cmd.extend(["--target-illuminance-lux", str(float(sf_target_lux))])
            if integration_time_override_s is not None:
                sf_cmd.extend(["--integration-time-s", str(integration_time_override_s)])
        else:
            raise ValueError(f'sensor_forward.mode must be "analytic" or "pbrt_exr", got {sf_mode!r}')
        log.append(run_cmd(sf_cmd, repo, args.dry_run))

    # 4) Noise post-step
    if bool(noise.get("enabled", True)):
        noise_cmd = [
            py,
            str(noise_tool),
            "--repo-root",
            str(repo),
            "--camera-model-config",
            str(camera_model_cfg),
            "--seed",
            str(noise.get("seed", 0)),
        ]
        noise_cmd.extend(["--linear-exr", str(exr_out)])
        if bool(sensor_forward.get("enabled", False)):
            noise_cmd.extend(["--electrons-npz", str(sensor_forward_npz)])
        if noise.get("exposure_scale", None) is not None:
            noise_cmd.extend(["--exposure-scale", str(noise["exposure_scale"])])
        if noise.get("preview_percentile", None) is not None:
            noise_cmd.extend(["--preview-percentile", str(noise["preview_percentile"])])
        if bool(noise.get("preview_no_normalize", False)):
            noise_cmd.append("--preview-no-normalize")
        if integration_time_override_s is not None:
            noise_cmd.extend(["--integration-time-s", str(integration_time_override_s)])
        log.append(run_cmd(noise_cmd, repo, args.dry_run))

    # 5) Validate Bayer+demosaic linear fidelity (optional)
    if bool(validate_demosaic.get("enabled", False)):
        vd_cmd = [
            py,
            str(validate_demosaic_tool),
            "--repo-root",
            str(repo),
            "--camera-model-config",
            str(camera_model_cfg),
            "--json-out",
            str(demosaic_metrics_json),
            "--crop",
            str(validate_demosaic.get("crop", 2)),
        ]
        if bool(sensor_forward.get("enabled", False)):
            vd_cmd.extend(["--electrons-npz", str(sensor_forward_npz)])
        log.append(run_cmd(vd_cmd, repo, args.dry_run))

    # 6) Manifest
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.name or "pipeline"
    manifest_path = out_dir / f"run_{run_name}_{ts}.json"
    manifest = {
        "timestamp_utc": ts,
        "repo_root": str(repo),
        "pipeline_config": str(cfg_path),
        "pipeline_config_sha256": sha256_file(cfg_path) if cfg_path.is_file() else None,
        "camera_model_config": str(camera_model_cfg),
        "camera_model_config_sha256": sha256_file(camera_model_cfg) if camera_model_cfg.is_file() else None,
        "dry_run": args.dry_run,
        "commands": log,
        "outputs": {
            "scene_file": str(scene_file),
            "scene_exists": scene_file.is_file(),
            "scene_sha256": sha256_file(scene_file) if (scene_file.is_file() and not args.dry_run) else None,
            "exr_out": str(exr_out),
            "exr_exists": exr_out.is_file(),
            "exr_sha256": sha256_file(exr_out) if (exr_out.is_file() and not args.dry_run) else None,
            "noise_stats_json": str((repo / "out" / "colorchecker_noisy_png" / "run_stats.json").resolve()),
            "sensor_forward_electrons_npz": str(sensor_forward_npz),
            "demosaic_metrics_json": str(demosaic_metrics_json),
            "demosaic_metrics_exists": demosaic_metrics_json.is_file(),
            "demosaic_metrics_sha256": sha256_file(demosaic_metrics_json)
            if (demosaic_metrics_json.is_file() and not args.dry_run)
            else None,
            "emva_validation_report": str(emva_validation_report),
            "emva_validation_exists": emva_validation_report.is_file(),
            "emva_validation_sha256": sha256_file(emva_validation_report)
            if (emva_validation_report.is_file() and not args.dry_run)
            else None,
        },
        "params": {
            "render": render,
            "sensor_forward": sensor_forward,
            "noise": noise,
            "validate": validate,
            "validate_emva": validate_emva,
            "validate_demosaic": validate_demosaic,
            "lens": lens_cfg,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote run manifest: {manifest_path}")


if __name__ == "__main__":
    main()
