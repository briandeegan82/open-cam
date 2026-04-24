#!/usr/bin/env python3
"""Emit shell environment variables from pipeline YAML (for shell scripts)."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import yaml
try:
    from camera_model import load_camera_model
except ModuleNotFoundError:  # pragma: no cover - import path variant for tests
    from tools.camera_model import load_camera_model

LEGACY_REALISTIC_LENSFILE = "scenes/lenses/wide_22mm.dat"
DEFAULT_REALISTIC_LENSFILE = "config/lenses/wide_22mm.dat"
DEFAULT_ILLUMINANT_CSV = "spectra/illuminant/interpolated/D55.csv"


def resolve_camera_model_path(repo: Path, paths: dict) -> tuple[str, Path]:
    model_name = paths.get("camera_model_name")
    model_cfg = paths.get("camera_model_config", None)
    if model_name and model_cfg:
        raise ValueError("set only one of paths.camera_model_name or paths.camera_model_config")
    if model_name:
        rel = f"config/camera_recipes/{model_name}.yaml"
        return rel, (repo / rel).resolve()
    rel = str(model_cfg or "config/camera_recipes/default.yaml")
    abs_path = Path(rel).resolve() if Path(rel).is_absolute() else (repo / rel).resolve()
    return rel, abs_path


def _emit_bash(name: str, val: object) -> str:
    if val is None:
        return f"export {name}="
    return f"export {name}={shlex.quote(str(val))}"


def _emit_env0(name: str, val: object) -> str:
    if val is None:
        return f"{name}=\0"
    return f"{name}={val}\0"


def canonicalize_lensfile_rel(path_value: object) -> str:
    lensfile = str(path_value)
    if lensfile == LEGACY_REALISTIC_LENSFILE:
        return DEFAULT_REALISTIC_LENSFILE
    return lensfile


def main() -> None:
    args = sys.argv[1:]
    if len(args) not in (2, 4):
        print("usage: pipeline_shell_env.py REPO_DIR PIPELINE_YAML [--format bash|env0]", file=sys.stderr)
        sys.exit(2)
    fmt = "bash"
    if len(args) == 4:
        if args[2] != "--format" or args[3] not in ("bash", "env0"):
            print("usage: pipeline_shell_env.py REPO_DIR PIPELINE_YAML [--format bash|env0]", file=sys.stderr)
            sys.exit(2)
        fmt = args[3]
    repo = Path(args[0]).resolve()
    raw = args[1]
    cfg_path = Path(raw)
    if not cfg_path.is_absolute():
        cfg_path = (repo / raw).resolve()
    if not cfg_path.is_file():
        print(f"error: missing pipeline config: {cfg_path}", file=sys.stderr)
        sys.exit(2)

    cfg = yaml.safe_load(cfg_path.read_text())
    paths = cfg.get("paths", {})
    render = cfg.get("render", {})
    validate = cfg.get("validate", {})
    sensor_forward = cfg.get("sensor_forward", {})
    noise = cfg.get("noise", {})
    exposure_time_override_s = cfg.get("exposure_time_override_s", None)
    realistic_focus_distance_override = cfg.get("realistic_focus_distance_override", None)
    vd = cfg.get("validate_demosaic", {})
    camera_model_rel, camera_model_abs = resolve_camera_model_path(repo, paths)
    camera_model = load_camera_model(camera_model_abs)
    lens = camera_model.get("lens", {})

    emit = _emit_bash if fmt == "bash" else _emit_env0
    out_lines: list[str] = []

    def export(name: str, val: object) -> None:
        out_lines.append(emit(name, val))

    export("PIPELINE_CONFIG_RESOLVED", str(cfg_path))

    export("SCENE_BUILDER_REL", paths.get("scene_builder", "tools/build_colorchecker_scene.py"))
    export("PBRT_REL", paths.get("pbrt", "third_party/pbrt-v4/build/pbrt"))
    export("NOISE_TOOL_REL", paths.get("noise_tool", "tools/apply_emva_noise.py"))
    export("SENSOR_FORWARD_TOOL_REL", paths.get("sensor_forward_tool", "tools/spectral_sensor_forward.py"))
    export("VALIDATE_TOOL_REL", paths.get("validate_tool", "tools/validate_colorchecker.py"))
    export("VALIDATE_DEMOSAIC_TOOL_REL", paths.get("validate_demosaic_tool", "tools/validate_demosaic_linear.py"))
    export("VALIDATE_EMVA_TOOL_REL", paths.get("validate_emva_tool", "tools/validate_emva_model.py"))
    export("EMVA_VALIDATION_REPORT_REL", paths.get("emva_validation_report", "out/emva_validation_report.json"))
    export("SCENE_FILE_REL", paths.get("scene_file", "scenes/generated/colorchecker.pbrt"))
    export("EXR_OUT_REL", paths.get("exr_out", "out/colorchecker.exr"))
    export("CAMERA_MODEL_CONFIG_REL", camera_model_rel)
    export("SENSOR_FORWARD_NPZ_REL", paths.get("sensor_forward_electrons_npz", "out/sensor_forward_electrons.npz"))
    export("PBRT_EXR_TO_ELECTRONS_TOOL_REL", paths.get("pbrt_exr_to_electrons_tool", "tools/pbrt_spectral_exr_to_electrons.py"))
    export("DEMOSAIC_METRICS_JSON_REL", paths.get("demosaic_metrics_json", "out/demosaic_linear_metrics.json"))

    export("LIGHT_SCALE", render.get("light_scale", 2.0))
    export("XRES", int(render.get("xres", 960)))
    export("YRES", int(render.get("yres", 640)))
    export("PIXELSAMPLES", int(render.get("pixelsamples", 64)))
    export("FILM", str(render.get("film", "rgb")).lower())
    fo = render.get("film_output", None)
    export("FILM_OUTPUT_REL", fo if fo else "")
    export("SPECTRAL_NBUCKETS", int(render.get("spectral_nbuckets", 16)))
    export("SPECTRAL_LAMBDA_MIN", float(render.get("spectral_lambda_min", 360.0)))
    export("SPECTRAL_LAMBDA_MAX", float(render.get("spectral_lambda_max", 830.0)))
    illum = render.get("illuminant", None)
    if illum is None or not str(illum).strip() or str(illum).strip().lower() in ("null", "none"):
        illum = DEFAULT_ILLUMINANT_CSV
    export("RENDER_ILLUMINANT_REL", str(illum).strip())
    bea = render.get("builder_extra_args", [])
    if bea is None:
        bea = []
    if not isinstance(bea, list):
        raise TypeError("render.builder_extra_args must be a YAML list of CLI tokens")
    export("BUILDER_EXTRA_ARGS", json.dumps([str(tok) for tok in bea]))

    export("CAM_DIST", float(render.get("cam_dist", 4.25)))
    export("CAMERA", str(lens.get("camera", "perspective")).lower())
    export(
        "REALISTIC_LENSFILE_REL",
        canonicalize_lensfile_rel(lens.get("realistic_lensfile", DEFAULT_REALISTIC_LENSFILE)),
    )
    export("REALISTIC_APERTURE_MM", float(lens.get("realistic_aperture_diameter_mm", 4.0)))
    rfd = lens.get("realistic_focus_distance", None)
    if realistic_focus_distance_override is not None:
        rfd = realistic_focus_distance_override
    export("REALISTIC_FOCUS_DISTANCE", rfd if rfd is not None else "")

    post_psf_on = "0"
    if bool((lens.get("post_psf") or {}).get("enabled", False)):
        post_psf_on = "1"
    export("POST_PSF_ENABLED", post_psf_on)
    export("PSF_TOOL_REL", paths.get("psf_tool", "tools/apply_spectral_psf.py"))

    export("VALIDATE_RENDER", "1" if bool(validate.get("enabled", True)) else "0")
    export("VALIDATE_EMVA", "1" if bool((cfg.get("validate_emva", {}) or {}).get("enabled", False)) else "0")
    export("SENSOR_FORWARD_ENABLED", "1" if bool(sensor_forward.get("enabled", False)) else "0")
    export("SENSOR_FORWARD_MODE", str(sensor_forward.get("mode", "analytic")).lower())
    sfl = sensor_forward.get("target_illuminance_lux", None)
    export("SENSOR_FORWARD_TARGET_LUX", sfl if sfl is not None else "")
    export("EXPOSURE_TIME_OVERRIDE_S", exposure_time_override_s if exposure_time_override_s is not None else "")
    export("NOISE_ENABLED", "1" if bool(noise.get("enabled", True)) else "0")
    export("DEFAULT_NOISE_SEED", int(noise.get("seed", 0)))
    pe = noise.get("preview_percentile", None)
    export("NOISE_PREVIEW_PERCENTILE", pe if pe is not None else "99.5")
    export("NOISE_PREVIEW_NO_NORMALIZE", "1" if bool(noise.get("preview_no_normalize", False)) else "0")
    wb_override = noise.get("preview_white_balance_enabled", None)
    cc_override = noise.get("preview_color_correction_enabled", None)
    if wb_override is None:
        export("NOISE_PREVIEW_WB_ENABLED", "")
    else:
        export("NOISE_PREVIEW_WB_ENABLED", "true" if bool(wb_override) else "false")
    if cc_override is None:
        export("NOISE_PREVIEW_CC_ENABLED", "")
    else:
        export("NOISE_PREVIEW_CC_ENABLED", "true" if bool(cc_override) else "false")
    es = noise.get("exposure_scale", None)
    export("NOISE_EXPOSURE_SCALE", es if es is not None else "")

    export("VALIDATE_DEMOSAIC", "1" if bool(vd.get("enabled", False)) else "0")
    export("DEMOSAIC_CROP", int(vd.get("crop", 2)))

    if fmt == "bash":
        print("\n".join(out_lines))
    else:
        sys.stdout.write("".join(out_lines))


if __name__ == "__main__":
    main()
