from __future__ import annotations

from _repo import REPO, ensure_pbrt_output_dir, out_dir, pbrt_path


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENCAM_OUT", raising=False)
    monkeypatch.delenv("OPENCAM_PBRT", raising=False)
    assert out_dir("iq_lab", "x") == REPO / "out" / "iq_lab" / "x"
    assert pbrt_path() == REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"
    monkeypatch.setenv("OPENCAM_OUT", str(tmp_path))
    monkeypatch.setenv("OPENCAM_PBRT", "/opt/pbrt")
    assert out_dir("iq_lab") == tmp_path / "iq_lab"
    assert str(pbrt_path()) == "/opt/pbrt"


def test_pbrt_output_dir_from_film_or_outfile(tmp_path):
    scene = tmp_path / "s" / "a.pbrt"
    scene.parent.mkdir()
    scene.write_text('Film "spectral"\n    "string filename" ["out/a.exr"]\nWorldBegin\n')
    assert ensure_pbrt_output_dir(scene) == scene.parent / "out"
    assert (scene.parent / "out").is_dir()
    assert ensure_pbrt_output_dir(scene, tmp_path) == tmp_path / "out"  # relative to pbrt's cwd
    assert ensure_pbrt_output_dir(scene, outfile="r/b.exr") == scene.parent / "r"
    scene.write_text("WorldBegin\n")
    assert ensure_pbrt_output_dir(scene) is None
