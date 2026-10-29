"""Patch 0031: the "guidedpath" integrator (Open PGL path guiding) converges to the path tracer's image."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from exr_multispectral import spectral_buckets_from_exr
from synthetic_data import REPO

PBRT = Path(os.environ.get("OPENCAM_PBRT", REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"))

# A glossy floor lit by a small bright panel that it sees only via a matte wall: the path
# tracer reaches that light through BSDF sampling of the floor, guiding learns where it is.
SCENE = """Film "spectral" "integer xresolution" [24] "integer yresolution" [16] "string filename" ["out/g.exr"]
    "integer nbuckets" [4]
LookAt 0 1 -3  0 0.3 1  0 1 0
Camera "perspective" "float fov" [50]
Sampler "independent" "integer pixelsamples" [{spp}]
Integrator "{integrator}" "integer maxdepth" [4]{extra}
WorldBegin
AttributeBegin
  Material "conductor" "float roughness" [0.05]
  Shape "bilinearmesh" "point3 P" [-3 0 -4  3 0 -4  -3 0 4  3 0 4]
AttributeEnd
AttributeBegin
  Material "diffuse" "rgb reflectance" [0.8 0.8 0.8]
  Shape "bilinearmesh" "point3 P" [-3 0 3  3 0 3  -3 3 3  3 3 3]
AttributeEnd
AttributeBegin
  AreaLightSource "diffuse" "blackbody L" [5000] "float scale" [40]
  Material "diffuse" "rgb reflectance" [0 0 0]
  Shape "bilinearmesh" "point3 P" [0.2 2.5 2.9  -0.2 2.5 2.9  0.2 2.9 2.9  -0.2 2.9 2.9]
AttributeEnd
AttributeBegin
  Material "diffuse" "rgb reflectance" [0.5 0.5 0.5]
  Shape "bilinearmesh" "point3 P" [-0.5 0.01 0.5  0.5 0.01 0.5  -0.5 0.01 1.5  0.5 0.01 1.5]
AttributeEnd
"""


def _render(tmp: Path, integrator: str, spp: int, seed: int, extra: str = "") -> subprocess.CompletedProcess:
    (tmp / "out").mkdir(exist_ok=True)
    (tmp / "g.pbrt").write_text(SCENE.format(integrator=integrator, spp=spp, extra=extra))
    return subprocess.run(
        [str(PBRT), "--quiet", "--seed", str(seed), "g.pbrt"], cwd=tmp, capture_output=True, text=True, timeout=600
    )


def _has_guidedpath() -> bool:
    if not PBRT.is_file():
        return False
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        return _render(Path(td), "guidedpath", 1, 0).returncode == 0


pytestmark = pytest.mark.skipif(
    not _has_guidedpath(), reason="pbrt not built with Open PGL (PBRT_OPENPGL=1 tools/build_pbrt.sh; patch 0031)"
)


def _means(tmp: Path, integrator: str, seeds: range) -> np.ndarray:
    out = []
    for s in seeds:
        r = _render(tmp, integrator, 256, s)
        assert r.returncode == 0, r.stderr
        cube, _ = spectral_buckets_from_exr(tmp / "out" / "g.exr")
        out.append(cube.mean(axis=(0, 1)))
    return np.array(out)


def test_guidedpath_matches_path_tracer(tmp_path):
    p, g = _means(tmp_path, "path", range(4)), _means(tmp_path, "guidedpath", range(4))
    se = np.hypot(p.std(0, ddof=1), g.std(0, ddof=1)) / 2
    z = (g.mean(0) - p.mean(0)) / se
    assert np.all(np.abs(z) < 4.5), z
    assert abs(g.mean() / p.mean() - 1) < 0.02


def test_guidedpath_parameters_are_checked(tmp_path):
    r = _render(tmp_path, "guidedpath", 1, 0, ' "float guidingprobability" [1]')
    assert r.returncode != 0 and "guidingprobability" in r.stderr + r.stdout
