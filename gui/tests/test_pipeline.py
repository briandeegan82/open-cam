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
