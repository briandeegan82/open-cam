"""write_png uses a fast zlib level; PNG stays lossless, so pixels must round-trip exactly."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from apply_emva_noise import write_png


class TestWritePng(unittest.TestCase):
    def test_lossless_roundtrip(self) -> None:
        rng = np.random.default_rng(0)
        cases = {
            "rgb8.png": rng.integers(0, 256, (17, 23, 3), dtype=np.uint8),
            "mono16.png": rng.integers(0, 1024, (17, 23), dtype=np.uint16),
        }
        with tempfile.TemporaryDirectory() as d:
            for name, arr in cases.items():
                path = Path(d) / "sub" / name
                write_png(path, arr)
                np.testing.assert_array_equal(iio.imread(path), arr)


if __name__ == "__main__":
    unittest.main()
