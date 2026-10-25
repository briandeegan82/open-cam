"""pbrt patch 0028: goniometric lamps that use the same map share one image and distribution.

Every lamp used to load its own copy of the map and build its own sampling distribution, so the
night highway's ~55 headlamps held ~700 MiB of identical data. Sharing must not change what a
lamp emits: N lamps on one map light the floor exactly N times as much as one lamp does.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from synthetic_data import REPO

sys.path.insert(0, str(REPO / "tools"))
from exr_multispectral import read_separate_exr_channels  # noqa: E402
from highway_night import write_y_exr  # noqa: E402

PBRT = Path(os.environ.get("OPENCAM_PBRT", REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"))
PBRT_TIMEOUT_S = int(os.environ.get("OPENCAM_PBRT_TIMEOUT", "600"))

if not PBRT.is_file() and os.environ.get("OPENCAM_REQUIRE_PBRT") == "1":
    raise RuntimeError(f"OPENCAM_REQUIRE_PBRT=1 but pbrt binary not found at {PBRT}")

pytestmark = pytest.mark.skipif(not PBRT.is_file(), reason=f"pbrt binary not built ({PBRT})")

_MIB = {"B": 1 / 2**20, "kB": 1 / 2**10, "KiB": 1 / 2**10, "MiB": 1.0, "GiB": 2.0**10}


def _render(tmp: Path, maps: list[str]) -> tuple[np.ndarray, float]:
    """Floor lit by one goniometric lamp per entry of ``maps`` (all at the same spot); returns
    the image and pbrt's "Light image and distributions" memory in MiB."""
    n = 128
    c = (np.arange(n) + 0.5) / n
    u, v = np.meshgrid(c, c)
    beam = np.where(np.abs(2 * u - 1) + np.abs(2 * v - 1) < 0.3, 100.0, 1.0).astype(np.float32)
    for m in set(maps):
        write_y_exr(tmp / m, beam)
    lamps = [
        'AttributeBegin Translate 0 2 0 Rotate 180 1 0 0 LightSource "goniometric" '
        f'"spectrum I" "stdillum-A" "string filename" "{m}" "float scale" [1] AttributeEnd'
        for m in maps
    ]
    tag = f"{len(maps)}_{len(set(maps))}"
    (tmp / f"{tag}.pbrt").write_text(
        "\n".join(
            [
                "LookAt 0 6 -0.01  0 0 0  0 1 0",
                'Camera "perspective" "float fov" [40]',
                'Sampler "independent" "integer pixelsamples" [4]',
                'Integrator "volpath" "integer maxdepth" [1]',
                f'Film "rgb" "integer xresolution" [32] "integer yresolution" [32] "string filename" "{tag}.exr"',
                "WorldBegin",
                *lamps,
                'Material "diffuse" "rgb reflectance" [0.5 0.5 0.5]',
                'Shape "bilinearmesh" "point3 P" [-20 0 -20  20 0 -20  -20 0 20  20 0 20]',
            ]
        )
    )
    r = subprocess.run(
        [str(PBRT), "--stats", "--outfile", f"{tag}.exr", f"{tag}.pbrt"],
        check=True,
        cwd=tmp,
        capture_output=True,
        text=True,
        timeout=PBRT_TIMEOUT_S,
    )
    m = re.search(r"Light image and distributions\s+([\d.]+) (\w+)", r.stdout + r.stderr)
    assert m, "pbrt --stats printed no light image memory"
    d = read_separate_exr_channels(tmp / f"{tag}.exr")
    img = np.mean([d[k] for k in ("R", "G", "B")], axis=0).astype(np.float64)
    return img, float(m.group(1)) * _MIB[m.group(2)]


def test_lamps_on_one_map_share_its_memory_and_add_up(tmp_path: Path) -> None:
    one, mem_one = _render(tmp_path, ["a.exr"])
    six, mem_six = _render(tmp_path, ["a.exr"] * 6)
    two_maps, mem_two_maps = _render(tmp_path, ["a.exr", "b.exr"] * 3)
    assert mem_six == pytest.approx(mem_one, rel=1e-3)  # one copy, not six (stats print 4-5 digits)
    assert mem_two_maps == pytest.approx(2 * mem_one, rel=1e-3)  # one copy per distinct map
    np.testing.assert_allclose(six, 6 * one, rtol=5e-3)  # only wavelength + light-pick noise
    np.testing.assert_allclose(two_maps, 6 * one, rtol=5e-3)
