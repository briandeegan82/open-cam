import pytest
from opencam_gui.core import pipeline as pl
from opencam_gui.core.catalog import find_recipe
from opencam_gui.core.repo import pbrt_available


def _request(**overrides) -> pl.GenerationRequest:
    base = dict(
        scene_id="colorchecker",
        camera_model_config=find_recipe("nikon_z6").path,
        mode="fast_analytic",
        illuminant_csv="spectra/illuminant/interpolated/D65.csv",
        target_illuminance_lux=500.0,
        exposure_time_s=0.01,
        seed=0,
        xres=160,
        yres=120,
        pixelsamples=8,
        run_label="pytest_run",
    )
    base.update(overrides)
    return pl.GenerationRequest(**base)


def test_list_scenes_includes_colorchecker_and_iq_targets():
    ids = {s.id for s in pl.list_scenes()}
    assert {"colorchecker", "slanted_edge", "iso_noise", "siemens_star"} <= ids


def test_fast_analytic_plan_has_three_steps_and_no_pbrt_binary():
    steps = pl.build_plan(_request(mode="fast_analytic"))
    assert len(steps) == 3
    joined = " ".join(" ".join(s.argv) for s in steps)
    assert "build_colorchecker_scene.py" in joined
    assert "spectral_sensor_forward.py" in joined
    assert "apply_emva_noise.py" in joined
    assert "pbrt" not in joined.lower().replace("apply_spectral_psf", "")


def test_iq_target_scene_rejects_fast_analytic():
    req = _request(scene_id="slanted_edge", mode="fast_analytic")
    try:
        pl.build_plan(req)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_pbrt_accurate_colorchecker_plan_calls_run_pipeline():
    steps = pl.build_plan(_request(mode="pbrt_accurate"))
    assert len(steps) == 1
    assert "run_pipeline.py" in " ".join(steps[0].argv)


def test_pbrt_accurate_iq_target_plan_has_four_steps():
    steps = pl.build_plan(_request(scene_id="slanted_edge", mode="pbrt_accurate"))
    names = [s.name for s in steps]
    assert len(steps) == 4
    assert any("PBRT" in n for n in names)


def test_run_plan_dry_run_never_touches_disk_and_reports_ok():
    result = pl.run_plan(_request(mode="fast_analytic"), dry_run=True)
    assert result.ok
    assert all(s.dry_run for s in result.steps)


def test_run_plan_pbrt_accurate_without_binary_fails_gracefully():
    if pbrt_available():
        return  # this repo happens to have PBRT built; nothing to assert here
    result = pl.run_plan(_request(mode="pbrt_accurate"), dry_run=False)
    assert not result.ok
    assert "PBRT" in result.error


def test_fast_analytic_end_to_end_writes_preview_pngs(tmp_path):
    lines: list[str] = []
    result = pl.run_plan(_request(mode="fast_analytic"), dry_run=False, on_output=lines.append)
    assert result.ok, f"pipeline failed: {result.error}\n" + "\n".join(lines)
    out_dir = pl.preview_output_dir()
    assert (out_dir / "clean_demosaic_rgb8.png").is_file()
    assert (out_dir / "noisy_demosaic_rgb8.png").is_file()
    stats = pl.load_run_stats()
    assert stats is not None
    assert stats["full_well_effective_e"] > 0


def test_run_steps_streams_output_before_step_exits(tmp_path):
    import sys
    import time

    flag = tmp_path / "release"
    child = (
        "import pathlib, time\n"
        "print('first line')\n"
        f"while not pathlib.Path({str(flag)!r}).exists(): time.sleep(0.01)\n"
        "print('second line')\n"
    )
    seen = []

    def on_output(line):
        seen.append(line)
        if line == "first line":
            flag.touch()

    t0 = time.monotonic()
    result = pl._run_steps([pl.Step("stream", [sys.executable, "-c", child])], dry_run=False, on_output=on_output)
    assert time.monotonic() - t0 < 30
    assert result.ok
    assert seen[1:] == ["first line", "second line"]


def test_run_steps_reports_nonzero_exit_and_stops(tmp_path):
    import sys

    seen = []
    steps = [
        pl.Step("boom", [sys.executable, "-c", "import sys; print('oops'); sys.exit(3)"]),
        pl.Step("never", [sys.executable, "-c", "print('should not run')"]),
    ]
    result = pl._run_steps(steps, dry_run=False, on_output=seen.append)
    assert not result.ok
    assert result.error == "boom failed (exit 3)"
    assert [s.returncode for s in result.steps] == [3]
    assert "oops" in seen and "should not run" not in seen


def test_accurate_colorchecker_frames_chart_like_fast_mode():
    import math

    repo = pl.repo_root()
    req = pl.GenerationRequest(
        scene_id="colorchecker",
        camera_model_config=repo / "config" / "camera_recipes" / "nikon_z6.yaml",
        mode="pbrt_accurate",
        illuminant_csv=None,
        target_illuminance_lux=1000.0,
        exposure_time_s=0.01,
        xres=320,
        yres=240,
    )
    d = pl._realistic_cam_dist_matching_fast_framing(req, {"lens_type_override": "realistic"})
    # 50 mm lens on pbrt's 35 mm-diagonal film (21 mm short side at 4:3) vs 35 deg pinhole at 4.25.
    expected = 4.25 * math.tan(math.radians(17.5)) / (10.5 / 50.0)
    assert d == pytest.approx(expected, rel=1e-3)

    pinhole_req = pl.GenerationRequest(
        **{**req.__dict__, "camera_model_config": repo / "config" / "camera_recipes" / "default.yaml"}
    )
    assert pl._realistic_cam_dist_matching_fast_framing(pinhole_req, {"lens_type_override": None}) is None
