"""Read pbrt-v4 SpectralFilm (and RGB) OpenEXR with all channels preserved.

imageio's FreeImage backend collapses multispectral EXR to RGB and discards S0.* planes;
use the OpenEXR Python bindings (``pip install OpenEXR``) for spectral buckets.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_S0_RE = re.compile(r"^S0\.([\d,]+)nm$")


def _require_openexr():
    try:
        import OpenEXR  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Multispectral / separate-channel EXR requires the OpenEXR Python package. "
            "Install with: pip install OpenEXR"
        ) from exc


def parse_s0_wavelength_nm(channel_name: str) -> float | None:
    m = _S0_RE.match(channel_name)
    if not m:
        return None
    return float(m.group(1).replace(",", "."))


def read_separate_exr_channels(path: Path) -> dict[str, np.ndarray]:
    """Return channel name -> HxW float32 array (OpenEXR, separate_channels=True)."""
    _require_openexr()
    import OpenEXR

    p = Path(path)
    out: dict[str, np.ndarray] = {}
    with OpenEXR.File(str(p), separate_channels=True) as f:
        for name, ch in f.channels().items():
            out[name] = np.asarray(ch.pixels, dtype=np.float32)
    return out


def write_separate_channels_exr(path: Path, channels: dict[str, np.ndarray]) -> None:
    """Write float32 separate-channel EXR (SpectralFilm-compatible: R,G,B + S0.<lambda>nm).

    Supports both the new OpenEXR >= 3.x Python API (``pip install OpenEXR``) and the
    legacy Imath-based API used by older distributions.  The new ``OpenEXR.Channel`` /
    ``OpenEXR.File`` path is tried first; the Imath path is used only as a fallback when
    the new API raises ``TypeError`` or ``AttributeError`` (i.e. old-style bindings).
    """
    _require_openexr()
    import OpenEXR

    if not channels:
        raise ValueError("channels dict is empty")
    first = next(iter(channels.values()))
    if first.ndim != 2:
        raise ValueError(f"expected HxW per channel, got shape {first.shape}")
    h0, w0 = first.shape
    for name, arr in channels.items():
        if arr.shape != (h0, w0):
            raise ValueError(f'channel "{name}" shape {arr.shape} != {(h0, w0)}')

    # New OpenEXR >= 3.x API: Channel objects accept pixel arrays directly.
    try:
        ch = {
            name: OpenEXR.Channel(pixels=np.ascontiguousarray(arr, dtype=np.float32))
            for name, arr in channels.items()
        }
        with OpenEXR.File(channels=ch) as f:
            f.write(str(Path(path)))
        return
    except (TypeError, AttributeError):
        pass

    # Legacy Imath-based fallback (OpenEXR < 3.x distributions that ship Imath).
    try:
        import Imath
    except ImportError as exc:
        raise RuntimeError(
            "write_separate_channels_exr requires either OpenEXR >= 3.x "
            "(new Channel/File API) or the legacy Imath package. "
            "Install with: pip install OpenEXR"
        ) from exc
    width, height = int(w0), int(h0)
    header = OpenEXR.Header(width, height)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    header["channels"] = {name: Imath.Channel(pt) for name in channels}
    out = OpenEXR.OutputFile(str(path), header)
    payload = {
        name: np.ascontiguousarray(arr, dtype=np.float32).tobytes() for name, arr in channels.items()
    }
    out.writePixels(payload)
    out.close()


def linear_rgb_from_exr(path: Path) -> np.ndarray:
    """HxWx3 float32: R,G,B from EXR (correct for SpectralFilm + RGB film)."""
    try:
        ch = read_separate_exr_channels(path)
    except RuntimeError:
        return _linear_rgb_imageio(path)
    if "R" in ch and "G" in ch and "B" in ch:
        return np.stack([ch["R"], ch["G"], ch["B"]], axis=2)
    if "RGB" in ch:
        x = ch["RGB"]
        if x.ndim == 3 and x.shape[2] >= 3:
            return x[:, :, :3].astype(np.float32, copy=False)
    raise ValueError(f"EXR has no R,G,B or RGB channel group: {sorted(ch.keys())}")


def spectral_buckets_from_exr(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Spectral planes HxWxK and bucket center wavelengths (nm), sorted by λ."""
    ch = read_separate_exr_channels(path)
    pairs: list[tuple[float, np.ndarray]] = []
    for name, arr in ch.items():
        lam = parse_s0_wavelength_nm(name)
        if lam is None:
            continue
        pairs.append((lam, arr.astype(np.float32, copy=False)))
    if not pairs:
        raise ValueError(f"no S0.*nm spectral channels in {path}")
    pairs.sort(key=lambda t: t[0])
    lambdas = np.array([t[0] for t in pairs], dtype=np.float64)
    stack = np.stack([t[1] for t in pairs], axis=2)
    return stack, lambdas


def trapezoid_weights_nm(lambdas_nm: np.ndarray) -> np.ndarray:
    """Per-bin weights (nm) for ∫·dλ on bucket samples (interior trapezoid)."""
    if lambdas_nm.size < 2:
        return np.ones_like(lambdas_nm)
    w = np.zeros_like(lambdas_nm)
    w[0] = (lambdas_nm[1] - lambdas_nm[0]) * 0.5
    w[-1] = (lambdas_nm[-1] - lambdas_nm[-2]) * 0.5
    if lambdas_nm.size > 2:
        w[1:-1] = (lambdas_nm[2:] - lambdas_nm[:-2]) * 0.5
    return w


def _linear_rgb_imageio(path: Path) -> np.ndarray:
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise RuntimeError("imageio is required to read EXR.") from exc
    img = iio.imread(path)
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"expected HxWx>=3 image, got shape {arr.shape}")
    return arr[:, :, :3]
