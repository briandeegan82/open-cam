"""Optimised hot paths must match the straightforward reference implementations they replaced."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from apply_emva_noise import _BAYER_PHASE, bayer_sample_rgb  # noqa: E402
from apply_spectral_psf import _gaussian_kernel_1d, separable_gaussian_blur_2d  # noqa: E402
from sensor_radiometry import (  # noqa: E402
    integrate_spectral_planes,
    photon_flux_density_from_irradiance,
    spectral_electron_weights,
)


def _reference_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    k = _gaussian_kernel_1d(sigma)
    pad = k.size // 2
    acc = np.asarray(img, dtype=np.float64)
    x = np.pad(acc, ((0, 0), (pad, pad)), mode="reflect")
    tmp = np.empty_like(acc)
    for i in range(acc.shape[0]):
        tmp[i, :] = np.convolve(x[i, :], k, mode="valid")
    y = np.pad(tmp, ((pad, pad), (0, 0)), mode="reflect")
    out = np.empty_like(acc)
    for j in range(acc.shape[1]):
        out[:, j] = np.convolve(y[:, j], k, mode="valid")
    return out.astype(np.float32)


def _reference_bayer(rgb: np.ndarray, pattern: str) -> np.ndarray:
    lut = np.array(_BAYER_PHASE[pattern], dtype=np.intp)
    h, w = rgb.shape[:2]
    jj, ii = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return rgb[jj, ii, lut[(jj & 1) * 2 + (ii & 1)]]


def _reference_spectral_electrons(planes, lam, qe_kx3, bin_w, scale, geom):
    phi = photon_flux_density_from_irradiance(planes.astype(np.float64) * scale, lam)
    return np.stack([np.sum(phi * (qe_kx3[:, c] * bin_w), axis=2) * geom for c in range(qe_kx3.shape[1])], axis=2)


class TestSeparableGaussianBlur(unittest.TestCase):
    def test_matches_reflect_padded_reference(self) -> None:
        rng = np.random.default_rng(0)
        for shape in ((40, 57), (9, 13), (64, 3)):
            img = rng.random(shape).astype(np.float32)
            for sigma in (0.4, 1.3, 3.7):
                with self.subTest(shape=shape, sigma=sigma):
                    np.testing.assert_allclose(
                        separable_gaussian_blur_2d(img, sigma), _reference_blur(img, sigma), rtol=1e-5, atol=1e-6
                    )

    def test_interior_impulse_keeps_its_energy(self) -> None:
        img = np.zeros((51, 51), dtype=np.float32)
        img[25, 25] = 1.0
        out = separable_gaussian_blur_2d(img, 2.0)
        self.assertAlmostEqual(float(out.sum()), 1.0, places=5)
        self.assertEqual(np.unravel_index(np.argmax(out), out.shape), (25, 25))
        np.testing.assert_allclose(out, out.T, atol=1e-7)

    def test_non_positive_sigma_is_identity(self) -> None:
        img = np.arange(12, dtype=np.float64).reshape(3, 4)
        for sigma in (0.0, -1.0):
            out = separable_gaussian_blur_2d(img, sigma)
            self.assertEqual(out.dtype, np.float32)
            np.testing.assert_array_equal(out, img.astype(np.float32))


class TestBayerSampling(unittest.TestCase):
    def test_matches_meshgrid_reference_for_every_pattern_and_odd_sizes(self) -> None:
        rng = np.random.default_rng(1)
        for shape in ((8, 8), (7, 5), (1, 3)):
            rgb = rng.random((*shape, 3)).astype(np.float32)
            for pattern in _BAYER_PHASE:
                with self.subTest(shape=shape, pattern=pattern):
                    out = bayer_sample_rgb(rgb, pattern)
                    self.assertEqual(out.dtype, rgb.dtype)
                    np.testing.assert_array_equal(out, _reference_bayer(rgb, pattern))

    def test_pattern_is_case_insensitive_and_validated(self) -> None:
        rgb = np.zeros((4, 4, 3), dtype=np.float32)
        np.testing.assert_array_equal(bayer_sample_rgb(rgb, "rggb"), bayer_sample_rgb(rgb, "RGGB"))
        with self.assertRaises(ValueError):
            bayer_sample_rgb(rgb, "RGBW")
        with self.assertRaises(ValueError):
            bayer_sample_rgb(rgb[:, :, 0], "RGGB")


class TestSpectralIntegration(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(2)
        self.lam = np.linspace(400.0, 700.0, 31)
        self.planes = rng.random((11, 13, self.lam.size)).astype(np.float32)
        self.qe = rng.random((self.lam.size, 3))
        self.bin_w = np.full(self.lam.size, 10.0)
        self.geom = 0.01 * (3e-6) ** 2

    def test_scalar_scale_matches_per_channel_sum(self) -> None:
        wts = spectral_electron_weights(self.lam, self.qe, self.bin_w, 2.5e-4, self.geom)
        got = integrate_spectral_planes(self.planes, wts)
        ref = _reference_spectral_electrons(self.planes, self.lam, self.qe, self.bin_w, 2.5e-4, self.geom)
        np.testing.assert_allclose(got, ref, rtol=1e-12)

    def test_per_wavelength_scale_and_small_blocks(self) -> None:
        scale = np.linspace(0.5, 1.0, self.lam.size) * 1e-3
        wts = spectral_electron_weights(self.lam, self.qe, self.bin_w, scale, self.geom)
        got = integrate_spectral_planes(self.planes, wts, block_pixels=7)
        ref = _reference_spectral_electrons(self.planes, self.lam, self.qe, self.bin_w, scale, self.geom)
        np.testing.assert_allclose(got, ref, rtol=1e-12)

    def test_shape_mismatches_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            spectral_electron_weights(self.lam, self.qe[:-1], self.bin_w, 1.0, 1.0)
        with self.assertRaises(ValueError):
            integrate_spectral_planes(self.planes, np.ones((self.lam.size + 1, 3)))


if __name__ == "__main__":
    unittest.main()
