"""pbrt patch 0025: ``diffusetransmission`` keeps R + T <= 1 at every wavelength.

White furnace: a thin double-sided leaf sheet inside a uniform environment of radiance 1. An
energy-conserving material can never look brighter than the environment, and with R + T = 1
(after the joint rescale) it is invisible, i.e. exactly 1. Clamping R and T to 1 on their own let
R = T = 0.9 return 1.8x the incident energy, so the sheet rendered well above 1.
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


def _furnace(tmp: Path, r: float, t: float, scale: float = 1.0) -> np.ndarray:
    exr = tmp / f"furnace_{r}_{t}_{scale}.exr"
    scene = tmp / f"furnace_{r}_{t}_{scale}.pbrt"
    scene.write_text(
        f"""LookAt 0 0 3  0 0 0  0 1 0
Camera "perspective" "float fov" 20
Sampler "zsobol" "integer pixelsamples" 64
Integrator "volpath" "integer maxdepth" 64
Film "spectral" "integer xresolution" 16 "integer yresolution" 16 "integer nbuckets" 8
    "float lambdamin" 400 "float lambdamax" 800 "string filename" "{exr}"
PixelFilter "box" "float xradius" 0.5 "float yradius" 0.5
WorldBegin
LightSource "infinite" "blackbody L" [5500] "float scale" 1
Texture "r" "spectrum" "constant" "spectrum value" [400 {r} 800 {r}]
Texture "t" "spectrum" "constant" "spectrum value" [400 {t} 800 {t}]
Material "diffusetransmission" "texture reflectance" "r" "texture transmittance" "t" "float scale" {scale}
Shape "bilinearmesh" "point3 P" [-2 -2 0  2 -2 0  -2 2 0  2 2 0]
"""
    )
    subprocess.run([str(PBRT), "--quiet", str(scene)], check=True, cwd=tmp, timeout=PBRT_TIMEOUT_S)
    return spectral_buckets_from_exr(exr)[0]


def test_leaf_sheet_never_brighter_than_its_furnace(tmp_path: Path) -> None:
    empty = _furnace(tmp_path, 0.0, 0.0, scale=0.0)  # black sheet: background = environment ...
    env = _furnace(tmp_path, 0.5, 0.5)  # ... R + T = 1: the sheet vanishes into the furnace
    over = _furnace(tmp_path, 0.9, 0.9)  # R + T = 1.8 must be rescaled to 0.5 / 0.5
    scaled = _furnace(tmp_path, 0.45, 0.45, scale=2.0)  # "scale" folds into R and T before the bound
    ref = env.mean((0, 1))
    assert np.all(empty.mean((0, 1)) < 0.05 * ref)  # the sheet fills the frame
    for img in (over, scaled):
        np.testing.assert_allclose(img.mean((0, 1)), ref, rtol=0.03)
