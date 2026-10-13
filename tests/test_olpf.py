"""Optical low-pass filter (tools/olpf.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import olpf  # noqa: E402
import sfr_analysis as sfr  # noqa: E402
from pbrt_spectral_exr_to_electrons import spectral_radiance_to_electrons  # noqa: E402


def test_quartz_indices_match_handbook():
    # Sodium D: n_o = 1.5442, n_e = 1.5533 (e.g. Hecht, Optics, table 8.1).
    n_o, n_e = olpf.quartz_indices(589.3)
    assert n_o == pytest.approx(1.5442, abs=2e-4) and n_e == pytest.approx(1.5533, abs=2e-4)
    # 45 deg cut: tan(rho) ~ (n_e^2 - n_o^2) / (n_e^2 + n_o^2), ~5.9 um walk-off per mm.
    assert olpf.walkoff_tan(589.3) == pytest.approx((n_e**2 - n_o**2) / (n_e**2 + n_o**2), rel=1e-12)
    assert 5.8e-3 < float(olpf.walkoff_tan(589.3)) < 5.95e-3
    t = olpf.walkoff_tan([450.0, 550.0, 650.0])
    assert t[0] > t[1] > t[2]  # birefringence falls with wavelength
    assert olpf.walkoff_tan(550.0, 0.0) == 0.0


def test_integer_split_is_two_shifted_copies():
    rng = np.random.default_rng(1)
    img = rng.random((40, 50)).astype(np.float64)
    out = olpf.apply_plane(img, 2.0, 2.0)
    ref = 0.25 * sum(np.roll(np.roll(img, sy, 0), sx, 1) for sx in (-1, 1) for sy in (-1, 1))
    np.testing.assert_allclose(out[4:-4, 4:-4], ref[4:-4, 4:-4], atol=1e-10)


@pytest.mark.parametrize("d", [0.6, 1.0, 1.5])
def test_slanted_edge_mtf_is_cosine(d):
    img = sfr.synthetic_slanted_edge(128, 128, 5.0)
    base = sfr.slanted_edge_sfr(img)
    res = sfr.slanted_edge_sfr(olpf.apply_plane(img, d, 0.0))
    f = base.frequency_cy_per_px
    k = (f > 0.02) & (f < 0.45)
    ratio = np.interp(f[k], res.frequency_cy_per_px, res.mtf) / base.mtf[k]
    np.testing.assert_allclose(ratio, np.abs(np.cos(np.pi * f[k] * d)), atol=0.03)
    assert olpf.apply_plane(img, d, 0.0).sum() == pytest.approx(img.sum(), rel=1e-9)


def test_split_resolution_and_validation():
    sensor = {"pixel_pitch_um": 4.0, "olpf": {"enabled": True, "split_x_um": 4.0, "thickness_y_um": 680.0}}
    cfg = olpf.resolve_config(sensor)
    assert olpf.split_px(cfg, "x", [450, 650]) == pytest.approx([1.0, 1.0])
    dy = olpf.split_px(cfg, "y", [450.0, 650.0])
    assert dy == pytest.approx(680.0 * olpf.walkoff_tan([450.0, 650.0]) / 4.0)
    assert 0.98 < dy[1] < dy[0] < 1.03
    with pytest.raises(ValueError):
        olpf.split_px(olpf.resolve_config({"olpf": {"split_pixels": 1, "split_x_um": 3}}), "x", 550)
    with pytest.raises(KeyError):
        olpf.split_px(olpf.resolve_config({"olpf": {"split_x_um": 3}}), "x", 550)


def _cfa_false_colour(e: np.ndarray) -> float:
    """Std of R - B over the image after a Bayer RGGB sample-and-hold demosaic (crude false-colour proxy)."""
    r = np.repeat(np.repeat(e[0::2, 0::2, 0], 2, 0), 2, 1)
    b = np.repeat(np.repeat(e[1::2, 1::2, 2], 2, 0), 2, 1)
    return float(np.std(r - b) / np.mean(e))


def test_electron_conversion_applies_olpf_once_and_cuts_colour_aliasing():
    sensor_yaml = yaml.safe_load((REPO / "config/sensor_models/default.yaml").read_text())
    sensor, model = sensor_yaml["sensor"], dict(sensor_yaml["sensor_forward"]["model"])
    assert sensor["olpf"]["enabled"] is False
    lam = np.arange(400.0, 701.0, 50.0)
    yy, xx = np.mgrid[0:64, 0:64]
    grey = 1.0 + 0.9 * np.cos(np.pi * 0.9 * xx)  # 0.45 cy/px vertical bars: neutral scene
    L = (grey[:, :, None] * np.full(lam.size, 1e-3)[None, None, :]).astype(np.float32)
    kw = {"repo": REPO, "model": model, "lens_cfg": {"camera": "pinhole"}}
    e_off, m_off = spectral_radiance_to_electrons(L, lam, sensor=sensor, **kw)
    on = {**sensor, "olpf": {"enabled": True, "split_pixels": 1.0}}
    e_on, m_on = spectral_radiance_to_electrons(L, lam, sensor=on, **kw)
    assert m_off["olpf"] == {"enabled": False} and m_on["olpf"]["enabled"]
    # Spots that land outside the frame are lost (<= split / 2 / width at the edge columns).
    np.testing.assert_allclose(e_on.mean((0, 1)), e_off.mean((0, 1)), rtol=5e-3)
    assert _cfa_false_colour(e_on) < 0.3 * _cfa_false_colour(e_off)
