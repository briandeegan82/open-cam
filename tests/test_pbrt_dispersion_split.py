"""pbrt patch 0005: ``"bool dispersionsplit"`` traces every sampled wavelength through the dispersive lens.

Hero-only tracing leaves secondary wavelengths unrecorded (pdf 0) on every camera sample; split mode
must record all of them, keep the same mean spectrum and cut the per-pixel colour noise.
"""

from __future__ import annotations

import os
import subprocess
import sys
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


def _render(tmp: Path, name: str, *extra: str) -> np.ndarray:
    scene = tmp / name
    exr = scene / "cc.exr"
    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_colorchecker_scene.py"),
            "--out-dir",
            str(scene),
            "--camera",
            "realistic",
            "--lensfile",
            str(REPO / "config" / "lenses" / "wide_22mm.dat"),
            "--xres",
            "96",
            "--yres",
            "64",
            "--pixelsamples",
            "16",
            "--film",
            "spectral",
            "--film-output",
            str(exr),
            *extra,
        ],
        check=True,
        capture_output=True,
    )
    pbrt_file = scene / "colorchecker.pbrt"
    assert ('"bool dispersionsplit" true' in pbrt_file.read_text()) == ("--lens-dispersion-split" in extra)
    subprocess.run([str(PBRT), "--quiet", str(pbrt_file)], check=True, cwd=scene, timeout=PBRT_TIMEOUT_S)
    return spectral_buckets_from_exr(exr)[0]


def test_dispersion_split_records_all_wavelengths(tmp_path: Path) -> None:
    hero = _render(tmp_path, "hero")
    split = _render(tmp_path, "split", "--lens-dispersion-split")
    lit = hero.sum(2) > 0.2 * np.percentile(hero.sum(2), 99)
    # Per-pixel colour noise: spread of the bucket-to-mean ratio over lit pixels.
    noise = {k: np.std((s / s.mean(2, keepdims=True))[lit]) for k, s in (("hero", hero), ("split", split))}
    assert noise["split"] < 0.7 * noise["hero"]
    assert np.mean(split[lit] == 0) < 0.5 * np.mean(hero[lit] == 0)
    np.testing.assert_allclose(split[lit].mean(0), hero[lit].mean(0), rtol=0.1)
