"""Patch 0030: --wavefront reports CPU-only patch features instead of dropping them silently."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from synthetic_data import REPO

PBRT = Path(os.environ.get("OPENCAM_PBRT", REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"))

HEAD = 'Camera "perspective" "float fov" [40]\nSampler "independent" "integer pixelsamples" [1]\n'
WORLD = (
    "WorldBegin\n"
    'LightSource "distant" "point3 from" [0 0 0] "point3 to" [0 0 1] "blackbody L" [5500]\n'
    'AttributeBegin\n{mat}\nTranslate 0 0 5\nShape "sphere" "float radius" [1]\nAttributeEnd\n'
)
FILM = 'Film "spectral" "integer xresolution" [4] "integer yresolution" [4] "string filename" ["out/w.exr"]{extra}\n'


def _run(tmp: Path, scene: str) -> subprocess.CompletedProcess:
    (tmp / "out").mkdir(exist_ok=True)
    (tmp / "w.pbrt").write_text(scene)
    return subprocess.run([str(PBRT), "--wavefront", "w.pbrt"], cwd=tmp, capture_output=True, text=True, timeout=300)


def _has_0030() -> bool:
    if not PBRT.is_file():
        return False
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        r = _run(
            Path(td),
            FILM.format(extra="") + HEAD + 'Integrator "volpath" "bool causticsonly" true\n' + WORLD.format(mat=""),
        )
        return r.returncode != 0 and "patch 0021" in r.stderr + r.stdout


pytestmark = pytest.mark.skipif(not _has_0030(), reason="patched pbrt not built (patch 0030); see docs/BUILD_PBRT.txt")


def test_cpu_only_material_and_film_features_warn(tmp_path):
    mat = (
        'Material "fluorescent" "spectrum excitation" [360 0.5 830 0] "spectrum emission" [500 0 600 1 700 0]\n'
        'Shape "sphere" "float radius" [0.2]\n'
        'Material "dielectric" "bool passthrough" true'
    )
    r = _run(tmp_path, FILM.format(extra="") + HEAD + WORLD.format(mat=mat))
    log = r.stderr + r.stdout
    assert r.returncode == 0, log
    for patch in ("patch 0002", "patches 0017/0020", "patch 0023"):
        assert patch in log, (patch, log)


def test_sample_jitter_silences_the_film_warning(tmp_path):
    r = _run(tmp_path, FILM.format(extra=' "string wavelengthjitter" "sample"') + HEAD + WORLD.format(mat=""))
    assert r.returncode == 0 and "patch 0023" not in r.stderr + r.stdout


def test_caustic_split_is_an_error(tmp_path):
    for flag in ("skipcaustics", "causticsonly"):
        r = _run(
            tmp_path, FILM.format(extra="") + HEAD + f'Integrator "volpath" "bool {flag}" true\n' + WORLD.format(mat="")
        )
        assert r.returncode != 0 and "patch 0021" in r.stderr + r.stdout
