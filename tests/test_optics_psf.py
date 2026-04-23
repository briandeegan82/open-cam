"""Tests for multispectral EXR write and separable Gaussian PSF."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.apply_spectral_psf import separable_gaussian_blur_2d
from tools.exr_multispectral import read_separate_exr_channels, write_separate_channels_exr


class TestOpticsPsf(unittest.TestCase):
    def test_separable_gaussian_sigma_zero_identity(self) -> None:
        rng = np.random.default_rng(0)
        img = rng.random((32, 24), dtype=np.float64).astype(np.float32)
        out = separable_gaussian_blur_2d(img, 0.0)
        np.testing.assert_allclose(out, img, rtol=1e-6)

    def test_exr_roundtrip_channels(self) -> None:
        try:
            import OpenEXR  # noqa: F401
        except ImportError:
            self.skipTest("OpenEXR not installed")
        h, w = 16, 12
        chans = {
            "R": np.ones((h, w), dtype=np.float32) * 0.1,
            "G": np.ones((h, w), dtype=np.float32) * 0.2,
            "B": np.ones((h, w), dtype=np.float32) * 0.3,
            "S0.550,0nm": np.linspace(0, 1, h * w, dtype=np.float32).reshape(h, w),
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.exr"
            write_separate_channels_exr(p, chans)
            back = read_separate_exr_channels(p)
        self.assertEqual(set(back.keys()), set(chans.keys()))
        for k in chans:
            np.testing.assert_allclose(back[k], chans[k], rtol=1e-5, atol=1e-6)
