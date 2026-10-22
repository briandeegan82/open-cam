"""--road-detile: offset-only stochastic re-tiling (concrete) and the tonal mask (concrete, paving)."""

import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import highway_road_wear as rw  # noqa: E402


def _stripes(n: int = 96) -> list[np.ndarray]:
    rng = np.random.default_rng(1)
    lum = np.tile(0.5 + 0.2 * np.sin(np.arange(n) * 2 * np.pi / 8), (n, 1)) + 0.01 * rng.standard_normal((n, n))
    return [lum, np.full((n, n), 0.5), np.zeros((n, n)), np.zeros((n, n))]


def _grad(a: np.ndarray, axis: int) -> float:
    return float(np.abs(np.diff(a, axis=axis)).mean())


def test_offset_only_keeps_orientation_and_mean() -> None:
    lum = rw.detile(_stripes(), 192, 4, np.random.default_rng(3), rotate=False)[0]
    # vertical stripes stay vertical: steep gradients across x, gentle ones (cell blends) down y
    assert _grad(lum, 0) < 0.3 * _grad(lum, 1)
    assert abs(lum.mean() - _stripes()[0].mean()) < 0.01


def test_rotation_mixes_orientation_and_offsets_unchanged() -> None:
    rot = rw.detile(_stripes(), 192, 4, np.random.default_rng(3))[0]
    assert _grad(rot, 0) > 0.3 * _grad(rot, 1)
    # same rng draws either way, so rotate=True output for asphalt is unchanged by the new option
    a = rw.detile(_stripes(), 192, 4, np.random.default_rng(5))
    b = rw.detile(_stripes(), 192, 4, np.random.default_rng(5), rotate=True)
    np.testing.assert_array_equal(a[0], b[0])


def test_tonal_mask_statistics() -> None:
    import OpenEXR

    with tempfile.TemporaryDirectory() as td:
        p = rw.tonal_mask(Path(td), 4)
        with OpenEXR.File(str(p)) as f:
            a = np.asarray(f.channels()["Y"].pixels, dtype=np.float64)
    assert a.shape == (round(rw.TONE_TILE_M / rw.TONE_TEXEL_M),) * 2
    assert abs(a.mean() - 1.0) < 2e-3
    assert 0.6 * rw.TONE_SD < a.std() < 1.2 * rw.TONE_SD
    # periodic: wrap-around steps are no larger than interior ones
    assert np.abs(a[0] - a[-1]).mean() < 2 * np.abs(np.diff(a, axis=0)).mean()


@pytest.mark.parametrize(("road", "retiled"), [("concrete046", True), ("pavingstones070", False)])
def test_builder_with_assets(road: str, retiled: bool) -> None:
    import build_highway_scene as hw

    prep = ROOT / "scenes" / "assets" / "highway" / "raw" / f"tex_{road}"
    if not prep.is_dir():
        pytest.skip(f"tex_{road} not fetched")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        hw.main(["--out-dir", str(out), "--road", road, "--traffic", "none", "--xres", "64", "--yres", "36"])
        txt = (out / "highway.pbrt").read_text()

        for f in re.findall(r'"string (?:filename|normalmap)" "([^"]+\.(?:exr|png))"', txt):
            assert (out / f).is_file(), f
        man = json.loads((out / "highway_manifest.json").read_text())
    assert man["road"]["detile"] == "auto"
    assert 'Texture "asphalt:tone"' in txt
    assert ("_detile6m_norot_" in txt) == retiled
