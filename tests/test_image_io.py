from __future__ import annotations

import numpy as np
from exr_multispectral import read_separate_exr_channels
from image_io import write_exr, write_png
from PIL import Image


def test_exr_round_trip(tmp_path):
    y = np.arange(12, dtype=np.float64).reshape(3, 4) / 7
    write_exr(tmp_path / "a" / "y.exr", {"Y": y})
    ch = read_separate_exr_channels(tmp_path / "a" / "y.exr")
    assert np.array_equal(ch["Y"], y.astype(np.float32))
    write_exr(tmp_path / "h.exr", {"Y": y}, np.float16)
    assert np.allclose(read_separate_exr_channels(tmp_path / "h.exr")["Y"], y, atol=1e-3)


def test_png_matches_pillow(tmp_path):
    a = np.random.default_rng(0).integers(0, 256, (5, 6, 3), dtype=np.uint8)
    write_png(tmp_path / "d" / "a.png", a)
    Image.fromarray(a).save(tmp_path / "b.png")
    assert (tmp_path / "d" / "a.png").read_bytes() == (tmp_path / "b.png").read_bytes()
    write_png(tmp_path / "c.png", a, compress_level=1)
    assert np.array_equal(np.asarray(Image.open(tmp_path / "c.png")), a)
