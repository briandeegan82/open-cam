from __future__ import annotations

import unittest

import numpy as np

from tools.apply_emva_noise import (
    bayer_sample_rgb,
    bilinear_demosaic,
    demosaic_requested,
    linear_to_srgb,
    to_png8_preview,
)


class TestDemosaic(unittest.TestCase):
    def test_flat_field_round_trip_all_patterns(self) -> None:
        mono = np.full((32, 32), 123.0, dtype=np.float64)
        for pattern in ("RGGB", "BGGR", "GRBG", "GBRG"):
            rgb = bilinear_demosaic(mono, pattern)
            self.assertEqual(rgb.shape, (32, 32, 3))
            self.assertTrue(np.allclose(rgb, 123.0))

    def test_constant_planes_recover_interior(self) -> None:
        h, w = 64, 64
        full = np.zeros((h, w, 3), dtype=np.float32)
        full[:, :, 0] = 1.0
        full[:, :, 1] = 0.5
        full[:, :, 2] = 0.25
        mono = bayer_sample_rgb(full, "RGGB").astype(np.float64)
        rec = bilinear_demosaic(mono, "RGGB")
        sl = np.s_[4:-4, 4:-4, :]
        self.assertTrue(np.allclose(rec[sl], full[sl], atol=1e-6))

    def test_odd_image_sizes_supported(self) -> None:
        for h, w in ((5, 7), (3, 3), (641, 959)):
            mono = np.random.RandomState(0).rand(h, w).astype(np.float64)
            rgb = bilinear_demosaic(mono, "RGGB")
            self.assertEqual(rgb.shape, (h, w, 3))
            self.assertTrue(np.isfinite(rgb).all())

    def test_bayer_sampler_pattern_mapping(self) -> None:
        rgb = np.zeros((2, 2, 3), dtype=np.float32)
        rgb[:, :, 0] = 100.0
        rgb[:, :, 1] = 200.0
        rgb[:, :, 2] = 300.0
        expected = {
            "RGGB": np.array([[100.0, 200.0], [200.0, 300.0]], dtype=np.float32),
            "BGGR": np.array([[300.0, 200.0], [200.0, 100.0]], dtype=np.float32),
            "GRBG": np.array([[200.0, 100.0], [300.0, 200.0]], dtype=np.float32),
            "GBRG": np.array([[200.0, 300.0], [100.0, 200.0]], dtype=np.float32),
        }
        for pattern, exp in expected.items():
            got = bayer_sample_rgb(rgb, pattern)
            self.assertTrue(np.array_equal(got, exp), msg=pattern)

    def test_linear_to_srgb_endpoints(self) -> None:
        self.assertAlmostEqual(float(linear_to_srgb(np.array(0.0))), 0.0, places=5)
        self.assertAlmostEqual(float(linear_to_srgb(np.array(1.0))), 1.0, places=5)

    def test_png8_preview_srgb_changes_midtones(self) -> None:
        dn = np.full((4, 4, 3), 500.0, dtype=np.float32)
        lin = to_png8_preview(dn, 10, black_dn=0.0, white_dn=1000.0, srgb=False)
        enc = to_png8_preview(dn, 10, black_dn=0.0, white_dn=1000.0, srgb=True)
        self.assertFalse(np.array_equal(lin, enc))
        self.assertGreater(int(enc[0, 0, 0]), int(lin[0, 0, 0]))

    def test_demosaic_requested_rejects_unknown_method(self) -> None:
        with self.assertRaises(ValueError):
            demosaic_requested({"demosaic": "malvar"})


if __name__ == "__main__":
    unittest.main()
