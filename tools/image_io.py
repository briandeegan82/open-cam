"""Image writers shared by the tools: ZIP scanline EXR via the OpenEXR >= 3 API, and PNG via imageio."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_exr(path: Path, channels: dict[str, np.ndarray], dtype: type = np.float32) -> None:
    """ZIP-compressed scanline EXR; each value is HxW (one channel) or HxWxC under a "RGB"/"RGBA" key."""
    import OpenEXR

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    with OpenEXR.File(header, {k: np.ascontiguousarray(v, dtype=dtype) for k, v in channels.items()}) as f:
        f.write(str(path))


def write_png(path: Path, arr: np.ndarray, compress_level: int | None = None) -> None:
    """uint8/uint16 PNG (lossless; ``compress_level`` 1 is ~5x faster than the default 6 for a larger file)."""
    import imageio.v3 as iio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, arr, **({} if compress_level is None else {"compress_level": compress_level}))
