from __future__ import annotations

from pathlib import Path

import pbrt_patchstamp as ps


def _patches(d: Path, files: dict[str, str]) -> Path:
    d.mkdir()
    for name, body in files.items():
        (d / name).write_text(body)
    return d


def test_stamp_depends_on_names_contents_and_order_only(tmp_path):
    a = _patches(tmp_path / "a", {"0002-b.patch": "y", "0001-a.patch": "x", "notes.txt": "ignored"})
    b = _patches(tmp_path / "b", {"0001-a.patch": "x", "0002-b.patch": "y"})
    c = _patches(tmp_path / "c", {"0001-a.patch": "x", "0002-b.patch": "z"})
    assert ps.patch_stamp(a) == ps.patch_stamp(b) != ps.patch_stamp(c)


def test_stamp_problem_missing_stale_and_ok(tmp_path):
    patches = _patches(tmp_path / "p", {"0001-a.patch": "x"})
    build = tmp_path / "build"
    build.mkdir()
    pbrt = build / "pbrt"
    pbrt.write_text("")
    assert "missing" in ps.stamp_problem(pbrt, patches)
    (build / ps.STAMP_NAME).write_text("0" * 64 + "\n")
    assert "older patch stack" in ps.stamp_problem(pbrt, patches)
    (build / ps.STAMP_NAME).write_text(ps.patch_stamp(patches) + "\n")
    assert ps.stamp_problem(pbrt, patches) is None


def test_build_script_writes_the_same_stamp():
    script = (ps.REPO / "tools" / "build_pbrt.sh").read_text()
    assert "export LC_ALL=C" in script
    assert 'pbrt_patchstamp.py" >"$build/pbrt.patchstamp"' in script
