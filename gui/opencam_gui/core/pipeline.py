"""Build and run the real Open Cam pipeline commands for the Image Generation GUI.

Two execution modes, both calling the existing ``tools/*.py`` scripts as
subprocesses — nothing here reimplements scene building, rendering, or noise:

``fast_analytic``
    ColorChecker only. Skips PBRT entirely: ``build_colorchecker_scene.py``
    (pinhole camera) -> ``spectral_sensor_forward.py`` (analytic forward
    model, works straight off the scene's reflectance spectra) ->
    ``apply_emva_noise.py``. Useful when PBRT hasn't been built, or for quick
    iteration.

``pbrt_accurate``
    The full, physically-accurate path used by ``tools/run_pipeline.py``
    (ColorChecker) or by ``scripts/generate_iq_target_image.sh`` (IQ targets):
    build scene -> PBRT spectral render -> optional post-PSF -> sensor
    forward -> EMVA noise. Requires a built PBRT binary.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

from opencam_gui.core.repo import import_tool, pbrt_available, pbrt_binary, python_executable, repo_root

_LEGACY_REALISTIC_LENSFILE = "scenes/lenses/wide_22mm.dat"
_DEFAULT_REALISTIC_LENSFILE = "config/lenses/wide_22mm.dat"


def _canonicalize_lensfile_rel(path_value: object) -> str:
    """Same alias tools/run_pipeline.py and tools/pipeline_shell_env.py apply."""
    lensfile = str(path_value)
    if lensfile == _LEGACY_REALISTIC_LENSFILE:
        return _DEFAULT_REALISTIC_LENSFILE
    return lensfile


def _camera_cli_args(camera_model_config: Path, cam_dist: float) -> list[str]:
    """Resolve --camera/--fov/--lensfile/... from the recipe's lens model, like run_pipeline.py does."""
    camera_model = import_tool("camera_model").load_camera_model(Path(camera_model_config))
    lens = camera_model.get("lens", {}) or {}
    cam = str(lens.get("camera", "pinhole")).lower()
    if cam == "realistic":
        lensfile = _canonicalize_lensfile_rel(lens.get("realistic_lensfile", _DEFAULT_REALISTIC_LENSFILE))
        aperture_mm = float(lens.get("realistic_aperture_diameter_mm", 4.0))
        args = ["--camera", "realistic", "--lensfile", lensfile, "--aperture-diameter-mm", str(aperture_mm)]
        focus = lens.get("realistic_focus_distance", None)
        if focus is not None:
            args.extend(["--focus-distance", str(float(focus))])
        return args
    if cam == "thinlens":
        fov = float(lens.get("thinlens_fov_deg", 35.0))
        radius = float(lens.get("thinlens_lens_radius", 0.03))
        focal = float(lens.get("thinlens_focal_distance", cam_dist))
        return [
            "--camera",
            "thinlens",
            "--fov",
            str(fov),
            "--thinlens-lens-radius",
            str(radius),
            "--thinlens-focal-distance",
            str(focal),
        ]
    fov = float(lens.get("pinhole_fov_deg", 35.0))
    return ["--camera", "pinhole", "--fov", str(fov)]


@dataclass(frozen=True)
class Scene:
    id: str
    label: str
    kind: str  # "colorchecker" | "iq_target"
    iq_target: str | None = None
    supports_fast_analytic: bool = False
    supports_illuminant_spectrum: bool = True
    description: str = ""


SCENES: dict[str, Scene] = {
    "colorchecker": Scene(
        id="colorchecker",
        label="ColorChecker (24-patch chart)",
        kind="colorchecker",
        supports_fast_analytic=True,
        description=(
            "The main documented pipeline target: a 24-patch Macbeth-style chart under a "
            "chosen illuminant. Supports both the fast analytic path and the full PBRT render."
        ),
    ),
    "slanted_edge": Scene(
        id="slanted_edge",
        label="Slanted edge (MTF / sharpness target)",
        kind="iq_target",
        iq_target="slanted_edge",
        supports_fast_analytic=False,
        supports_illuminant_spectrum=False,
        description="ISO 12233-style slanted edge for MTF/sharpness measurement. PBRT render only.",
    ),
    "iso_noise": Scene(
        id="iso_noise",
        label="ISO noise chart (grayscale patches)",
        kind="iq_target",
        iq_target="iso_noise",
        supports_fast_analytic=False,
        supports_illuminant_spectrum=False,
        description="Grid of neutral reflectance patches for photon-transfer / SNR curves. PBRT render only.",
    ),
    "siemens_star": Scene(
        id="siemens_star",
        label="Siemens star (resolution target)",
        kind="iq_target",
        iq_target="siemens_star",
        supports_fast_analytic=False,
        supports_illuminant_spectrum=False,
        description="Radial spoke target for resolution / aliasing demonstrations. PBRT render only.",
    ),
}


def list_scenes() -> list[Scene]:
    return list(SCENES.values())


@dataclass(frozen=True)
class GenerationRequest:
    scene_id: str
    camera_model_config: Path
    mode: str  # "fast_analytic" | "pbrt_accurate"
    illuminant_csv: str | None  # repo-relative, e.g. "spectra/illuminant/interpolated/D65.csv"
    target_illuminance_lux: float
    exposure_time_s: float | None
    seed: int = 0
    xres: int = 480
    yres: int = 320
    pixelsamples: int = 64
    run_label: str = "gui_run"


@dataclass(frozen=True)
class Step:
    name: str
    argv: list[str]


@dataclass
class StepResult:
    step: Step
    returncode: int | None
    dry_run: bool


@dataclass
class RunResult:
    steps: list[StepResult] = field(default_factory=list)
    ok: bool = True
    error: str | None = None


OutputCallback = Callable[[str], None]


def _run_dir(run_label: str) -> Path:
    d = repo_root() / "out" / "gui_runs" / run_label
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run_steps(steps: list[Step], *, dry_run: bool, on_output: OutputCallback | None = None) -> RunResult:
    result = RunResult()
    repo = repo_root()
    for step in steps:
        if on_output:
            on_output(f"$ {' '.join(step.argv)}")
        if dry_run:
            result.steps.append(StepResult(step=step, returncode=None, dry_run=True))
            continue
        try:
            proc = subprocess.run(
                step.argv,
                cwd=str(repo),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            result.ok = False
            result.error = f"{step.name}: {exc}"
            if on_output:
                on_output(str(exc))
            return result
        if on_output and proc.stdout:
            for line in proc.stdout.splitlines():
                on_output(line)
        result.steps.append(StepResult(step=step, returncode=proc.returncode, dry_run=False))
        if proc.returncode != 0:
            result.ok = False
            result.error = f"{step.name} failed (exit {proc.returncode})"
            return result
    return result


# ---------------------------------------------------------------------------
# fast_analytic: ColorChecker only, no PBRT
# ---------------------------------------------------------------------------


def _fast_analytic_steps(req: GenerationRequest) -> list[Step]:
    repo = repo_root()
    py = python_executable()
    tools = repo / "tools"
    illuminant = req.illuminant_csv or "spectra/illuminant/interpolated/D65.csv"

    build_cmd = [
        py,
        str(tools / "build_colorchecker_scene.py"),
        "--repo-root",
        str(repo),
        "--illuminant",
        str(repo / illuminant),
        "--light-scale",
        "2.0",
        "--cam-dist",
        "4.25",
        "--xres",
        str(req.xres),
        "--yres",
        str(req.yres),
        "--pixelsamples",
        str(req.pixelsamples),
        "--film",
        "spectral",
        "--camera",
        "pinhole",
        "--fov",
        "35",
        "--spectral-nbuckets",
        "32",
    ]

    forward_cmd = [
        py,
        str(tools / "spectral_sensor_forward.py"),
        "--repo-root",
        str(repo),
        "--camera-model-config",
        str(req.camera_model_config),
        "--target-illuminance-lux",
        str(req.target_illuminance_lux),
    ]
    if req.exposure_time_s is not None:
        forward_cmd.extend(["--integration-time-s", str(req.exposure_time_s)])

    noise_cmd = [
        py,
        str(tools / "apply_emva_noise.py"),
        "--repo-root",
        str(repo),
        "--camera-model-config",
        str(req.camera_model_config),
        "--seed",
        str(req.seed),
        "--electrons-npz",
        str(repo / "out" / "sensor_forward_electrons.npz"),
        "--preview-percentile",
        "99.5",
    ]
    if req.exposure_time_s is not None:
        noise_cmd.extend(["--integration-time-s", str(req.exposure_time_s)])

    return [
        Step("Build ColorChecker scene (pinhole, no PBRT)", build_cmd),
        Step("Sensor forward (analytic, from chart reflectance spectra)", forward_cmd),
        Step("EMVA noise + Bayer + demosaic preview", noise_cmd),
    ]


# ---------------------------------------------------------------------------
# pbrt_accurate: ColorChecker via tools/run_pipeline.py
# ---------------------------------------------------------------------------


def _write_colorchecker_pipeline_yaml(req: GenerationRequest) -> Path:
    repo = repo_root()
    base_path = repo / "config" / "pipeline.yaml"
    cfg = yaml.safe_load(base_path.read_text())

    cfg.setdefault("paths", {})
    cfg["paths"]["camera_model_config"] = str(req.camera_model_config)
    cfg["paths"].pop("camera_model_name", None)

    cfg.setdefault("render", {})
    cfg["render"]["xres"] = req.xres
    cfg["render"]["yres"] = req.yres
    cfg["render"]["pixelsamples"] = req.pixelsamples
    if req.illuminant_csv:
        cfg["render"]["illuminant"] = req.illuminant_csv

    cfg.setdefault("sensor_forward", {})
    cfg["sensor_forward"]["enabled"] = True
    cfg["sensor_forward"]["target_illuminance_lux"] = req.target_illuminance_lux

    cfg.setdefault("noise", {})
    cfg["noise"]["seed"] = req.seed

    if req.exposure_time_s is not None:
        cfg["exposure_time_override_s"] = req.exposure_time_s

    out_dir = _run_dir(req.run_label)
    cfg_path = out_dir / "pipeline.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg_path


def _pbrt_accurate_colorchecker_steps(req: GenerationRequest) -> list[Step]:
    cfg_path = _write_colorchecker_pipeline_yaml(req)
    repo = repo_root()
    py = python_executable()
    argv = [
        py,
        str(repo / "tools" / "run_pipeline.py"),
        "--repo-root",
        str(repo),
        "--config",
        str(cfg_path),
        "--name",
        req.run_label,
    ]
    return [Step("Full pipeline (scene -> PBRT render -> sensor forward -> EMVA noise)", argv)]


# ---------------------------------------------------------------------------
# pbrt_accurate: IQ targets, mirroring scripts/generate_iq_target_image.sh
# ---------------------------------------------------------------------------


def _pbrt_accurate_iq_target_steps(req: GenerationRequest, scene: Scene) -> list[Step]:
    repo = repo_root()
    py = python_executable()
    tools = repo / "tools"
    target = scene.iq_target
    scene_dir = repo / "scenes" / "generated" / "iq_targets"
    scene_path = scene_dir / f"{target}.pbrt"
    stage_dir = repo / "out" / "iq_targets"
    # tools/build_image_quality_targets.py fixes this path itself; not CLI-overridable.
    exr_path = repo / "out" / f"{target}_spectral.exr"
    npz_path = stage_dir / f"{target}_electrons.npz"

    build_cmd = [
        py,
        str(tools / "build_image_quality_targets.py"),
        "--repo-root",
        str(repo),
        "--out-dir",
        str(scene_dir),
        "--target",
        target,
        "--film",
        "spectral",
        "--xres",
        str(req.xres),
        "--yres",
        str(req.yres),
        "--pixelsamples",
        str(req.pixelsamples),
        "--cam-dist",
        "3.2",
        *_camera_cli_args(req.camera_model_config, cam_dist=3.2),
    ]

    render_cmd = [str(pbrt_binary()), str(scene_path)]

    forward_cmd = [
        py,
        str(tools / "pbrt_spectral_exr_to_electrons.py"),
        "--repo-root",
        str(repo),
        "--exr",
        str(exr_path),
        "--camera-model-config",
        str(req.camera_model_config),
        "--out",
        str(npz_path),
    ]
    if req.target_illuminance_lux:
        forward_cmd.extend(["--target-illuminance-lux", str(req.target_illuminance_lux)])
    if req.exposure_time_s is not None:
        forward_cmd.extend(["--integration-time-s", str(req.exposure_time_s)])

    noise_cmd = [
        py,
        str(tools / "apply_emva_noise.py"),
        "--repo-root",
        str(repo),
        "--camera-model-config",
        str(req.camera_model_config),
        "--seed",
        str(req.seed),
        "--electrons-npz",
        str(npz_path),
        "--preview-percentile",
        "99.5",
    ]
    if req.exposure_time_s is not None:
        noise_cmd.extend(["--integration-time-s", str(req.exposure_time_s)])

    stage_dir.mkdir(parents=True, exist_ok=True)
    return [
        Step(f"Build {scene.label} scene", build_cmd),
        Step("PBRT spectral render", render_cmd),
        Step("Electrons from rendered spectral EXR", forward_cmd),
        Step("EMVA noise + Bayer + demosaic preview", noise_cmd),
    ]


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def build_plan(req: GenerationRequest) -> list[Step]:
    scene = SCENES[req.scene_id]
    if req.mode == "fast_analytic":
        if not scene.supports_fast_analytic:
            raise ValueError(f"{scene.label} does not support the fast/analytic (no-PBRT) path")
        return _fast_analytic_steps(req)
    if req.mode == "pbrt_accurate":
        if scene.kind == "colorchecker":
            return _pbrt_accurate_colorchecker_steps(req)
        return _pbrt_accurate_iq_target_steps(req, scene)
    raise ValueError(f"unknown mode: {req.mode!r}")


def run_plan(req: GenerationRequest, *, dry_run: bool, on_output: OutputCallback | None = None) -> RunResult:
    if req.mode == "pbrt_accurate" and not dry_run and not pbrt_available():
        return RunResult(
            ok=False,
            error=(
                "PBRT binary not found at third_party/pbrt-v4/build/pbrt. "
                "Build it (see docs/BUILD_PBRT.txt) or switch to 'Fast preview (analytic, no PBRT)' "
                "for the ColorChecker scene."
            ),
        )
    steps = build_plan(req)
    return _run_steps(steps, dry_run=dry_run, on_output=on_output)


def preview_output_dir() -> Path:
    """Fixed location apply_emva_noise.py always writes previews to."""
    return repo_root() / "out" / "colorchecker_noisy_png"


def run_stats_json() -> Path:
    return preview_output_dir() / "run_stats.json"


def load_run_stats() -> dict | None:
    p = run_stats_json()
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None
