"""pbrt patch 0026: SpectralFilm splats (light tracing, BDPT t=1 paths) use the same units as samples.

``AddSample`` stores the bucket mean of ``L * CIE_Y_integral``; ``AddSplat`` stored ``L / pdf / N``
(the integral over the bucket), so BDPT's light-traced share came out several times too dim, and
pixels only reached by splats (all of ``lightpath``) were written as 0. A converged lightpath or
BDPT render of a diffuse scene must match the path tracer bucket by bucket.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from exr_multispectral import spectral_buckets_from_exr
from synthetic_data import REPO

PBRT = Path(os.environ.get("OPENCAM_PBRT", REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"))
PBRT_TIMEOUT_S = int(os.environ.get("OPENCAM_PBRT_TIMEOUT", "600"))

if not PBRT.is_file() and os.environ.get("OPENCAM_REQUIRE_PBRT") == "1":
    raise RuntimeError(f"OPENCAM_REQUIRE_PBRT=1 but pbrt binary not found at {PBRT}")

pytestmark = pytest.mark.skipif(not PBRT.is_file(), reason=f"pbrt binary not built ({PBRT})")


def _render(tmp: Path, integrator: str) -> np.ndarray:
    exr = tmp / f"{integrator}.exr"
    scene = tmp / f"{integrator}.pbrt"
    scene.write_text(
        f"""LookAt 0 1.2 -3  0 0.3 0  0 1 0
Camera "perspective" "float fov" 40
Sampler "zsobol" "integer pixelsamples" 128
Integrator "{integrator}" "integer maxdepth" 5
Film "spectral" "integer xresolution" 48 "integer yresolution" 32 "integer nbuckets" 8
    "float lambdamin" 400 "float lambdamax" 800 "string filename" "{exr}"
PixelFilter "box" "float xradius" 0.5 "float yradius" 0.5
WorldBegin
AttributeBegin
  AreaLightSource "diffuse" "blackbody L" [4000] "float scale" 4
  Translate 0 2 0
  Rotate 90 1 0 0
  Shape "bilinearmesh" "point3 P" [-0.5 -0.5 0  0.5 -0.5 0  -0.5 0.5 0  0.5 0.5 0]
AttributeEnd
Material "diffuse" "rgb reflectance" [0.6 0.5 0.3]
Shape "bilinearmesh" "point3 P" [-3 0 -3  3 0 -3  -3 0 3  3 0 3]
Material "diffuse" "rgb reflectance" [0.2 0.4 0.7]
AttributeBegin
  Translate 0 0.4 0
  Shape "sphere" "float radius" 0.4
AttributeEnd
"""
    )
    subprocess.run([str(PBRT), "--quiet", str(scene)], check=True, cwd=tmp, timeout=PBRT_TIMEOUT_S)
    return spectral_buckets_from_exr(exr)[0]


@pytest.mark.parametrize("integrator", ["lightpath", "bdpt"])
def test_splatting_integrators_match_the_path_tracer_per_bucket(tmp_path: Path, integrator: str) -> None:
    ref = _render(tmp_path, "path").mean((0, 1))
    np.testing.assert_allclose(_render(tmp_path, integrator).mean((0, 1)), ref, rtol=0.05)
