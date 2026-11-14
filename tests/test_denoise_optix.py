"""Opt-in OptiX denoiser (tools/denoise_optix.py): runs only where tools/optix_denoise is built and a GPU answers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import denoise_optix as dx  # noqa: E402

pytestmark = pytest.mark.skipif(not dx.BINARY.is_file(), reason="tools/optix_denoise not built (denoise_optix.py --build)")


def _scene(rng):
    h, w, k = 64, 64, 5
    cube = np.full((h, w, k), 1.0, np.float32) + rng.normal(0, 0.1, (h, w, k)).astype(np.float32)
    cube[32, 32] = 5000.0  # a 1 px lamp
    rgb = cube[..., :3].copy()
    albedo = np.full((h, w, 3), 0.5, np.float32)
    normal = np.zeros((h, w, 3), np.float32)
    normal[..., 2] = 1
    return np.maximum(cube, 0), np.maximum(rgb, 0), albedo, normal


@pytest.mark.parametrize("mode", dx.MODES)
def test_denoises_and_keeps_the_lamp_with_bypass(mode):
    cube, rgb, alb, nrm = _scene(np.random.default_rng(0))
    try:
        den, rgb_d, keep = dx.denoise_cube(cube, rgb, alb, nrm, mode=mode, firefly_ratio=0)
    except RuntimeError as e:
        pytest.skip(f"no OptiX device: {e}")
    assert keep[32, 32, 0] < 0.01
    assert den.sum() == pytest.approx(cube.sum(), rel=0.01)
    bg = np.ones(cube.shape[:2], bool)
    bg[28:37, 28:37] = False  # around the lamp
    bg[:8], bg[-8:], bg[:, :8], bg[:, -8:] = False, False, False, False
    assert (den - 1)[bg].std() < 0.5 * (cube - 1)[bg].std()
    assert den[32, 32].mean() == pytest.approx(5000.0, rel=0.01)


def test_without_bypass_optix_removes_a_1px_lamp():
    cube, rgb, alb, nrm = _scene(np.random.default_rng(1))
    try:
        den, _, _ = dx.denoise_cube(cube, rgb, alb, nrm, mode="hdr", firefly_ratio=0, bypass=0)
    except RuntimeError as e:
        pytest.skip(f"no OptiX device: {e}")
    assert den[32, 32].mean() < 0.2 * 5000.0
