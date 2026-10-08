"""Synthetic inputs (QE curves, EXRs, configs) and runners for tool module tests."""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from exr_multispectral import write_separate_channels_exr  # noqa: E402

QE_WAVELENGTHS_NM = np.arange(380.0, 781.0, 5.0)
SPECTRAL_LAMBDAS_NM = np.arange(400.0, 701.0, 10.0)


def write_curve(path: Path, wavelength_nm: np.ndarray, values: np.ndarray) -> Path:
    path.write_text("".join(f"{w:.6f},{v:.8f}\n" for w, v in zip(wavelength_nm, values, strict=True)))
    return path


def write_flat_qe(directory: Path, value: float = 0.5) -> dict:
    """Wavelength-independent QE for all three channels (analytic electron counts)."""
    wl = QE_WAVELENGTHS_NM
    return {
        f"{ch}_csv": str(write_curve(directory / f"qe_flat_{ch}.csv", wl, np.full(wl.shape, value)))
        for ch in ("red", "green", "blue")
    }


def write_gaussian_qe(directory: Path) -> dict:
    """Plausible R/G/B QE bands peaking at 600 / 540 / 460 nm."""
    wl = QE_WAVELENGTHS_NM
    peaks = {"red": 600.0, "green": 540.0, "blue": 460.0}
    return {
        f"{ch}_csv": str(write_curve(directory / f"qe_{ch}.csv", wl, 0.6 * np.exp(-0.5 * ((wl - peak) / 40.0) ** 2)))
        for ch, peak in peaks.items()
    }


def spectral_channel_name(lambda_nm: float) -> str:
    return f"S0.{lambda_nm:g}nm".replace(".", ",").replace("S0,", "S0.")


def write_spectral_exr(path: Path, planes: np.ndarray, lambdas_nm: np.ndarray, rgb: np.ndarray | None = None) -> Path:
    """Write an HxWxK spectral cube as a SpectralFilm-style EXR (R,G,B + S0.<lambda>nm)."""
    planes = np.asarray(planes, dtype=np.float32)
    if rgb is None:
        mean = planes.mean(axis=2)
        rgb = np.stack([mean, mean, mean], axis=2)
    channels = {"R": rgb[:, :, 0], "G": rgb[:, :, 1], "B": rgb[:, :, 2]}
    for k, lam in enumerate(lambdas_nm):
        channels[spectral_channel_name(float(lam))] = planes[:, :, k]
    write_separate_channels_exr(path, channels)
    return path


def write_rgb_exr(path: Path, rgb: np.ndarray) -> Path:
    rgb = np.asarray(rgb, dtype=np.float32)
    write_separate_channels_exr(path, {"R": rgb[:, :, 0], "G": rgb[:, :, 1], "B": rgb[:, :, 2]})
    return path


def colour_bars(h: int = 24, w: int = 32) -> np.ndarray:
    """HxWx3 linear RGB test chart: vertical colour bars plus a horizontal ramp."""
    bars = np.array(
        [[1.0, 1.0, 1.0], [1.0, 0.2, 0.2], [0.2, 1.0, 0.2], [0.2, 0.2, 1.0]],
        dtype=np.float32,
    )
    idx = (np.arange(w) * len(bars)) // w
    img = np.broadcast_to(bars[idx][None, :, :], (h, w, 3)).copy()
    ramp = np.linspace(0.1, 1.0, h, dtype=np.float32)[:, None, None]
    return img * ramp


def write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def run_tool_main(main, argv: list[str]) -> str:
    """Run a tool's ``main()`` with ``argv``; return captured stdout (stderr is discarded)."""
    out, err = io.StringIO(), io.StringIO()
    with (
        mock.patch.object(sys, "argv", ["tool", *argv]),
        contextlib.redirect_stdout(out),
        contextlib.redirect_stderr(err),
    ):
        main()
    return out.getvalue()
