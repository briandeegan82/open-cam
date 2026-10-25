"""Hybrid IR-cut filter (tools/ircf_model.py) and which camera models apply an IR-cut filter."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import ircf_model as im  # noqa: E402


def _csv(rel):
    a = np.loadtxt(REPO / rel, delimiter=",", comments="#")
    return a[:, 0], a[:, 1]


def test_hybrid_half_power_at_t50_and_nir_leak():
    lam = np.arange(300.0, 1101.0)
    t, coat, lc = im.hybrid(lam)
    peak = t[(lam >= 450) & (lam <= 600)].max()
    assert abs(np.interp(650.0, lam, t) / peak - 0.5) < 1e-3
    assert lc > 650.0
    nir = t[(lam >= 700) & (lam <= 1100)]
    assert nir.min() > 1e-4 and nir.max() < 0.01
    assert np.allclose(t, coat * im.glass_transmittance(lam, thickness_mm=0.21))
    assert (np.diff(t[(lam >= 560) & (lam <= 680)]) < 0).all()


def test_bundled_csvs_match_the_model():
    lam, t = _csv("spectra/QE/ircf/QE_IRCF_hybrid.csv")
    _, c = _csv("spectra/QE/ircf/QE_IRCF_hybrid_coating.csv")
    tm, cm, _ = im.hybrid(lam)
    assert np.allclose(t, tm, rtol=1e-4, atol=1e-7) and np.allclose(c, cm, rtol=1e-4, atol=1e-7)
    assert lam[0] <= 360 and lam[-1] >= 1100


def test_glass_scales_with_thickness():
    lam = np.array([650.0, 700.0])
    assert np.allclose(
        im.glass_transmittance(lam, thickness_mm=0.42), im.glass_transmittance(lam, thickness_mm=0.21) ** 2
    )


def test_measured_cameras_skip_generic_ircf_research_cameras_use_hybrid():
    for p in sorted((REPO / "config/sensor_models").glob("*.yaml")):
        qe = (yaml.safe_load(p.read_text()).get("sensor") or {}).get("quantum_efficiency") or {}
        if "spectra/QE/cameras/" in str(qe.get("red_csv", "")):
            assert qe.get("ircf_csv") is None, p.name
        elif qe.get("ircf_csv"):
            assert qe["ircf_csv"] == "spectra/QE/ircf/QE_IRCF_hybrid.csv", p.name
            assert qe["ircf_coating_csv"] == "spectra/QE/ircf/QE_IRCF_hybrid_coating.csv", p.name
            assert str(qe["red_csv"]).startswith("spectra/QE/nir/"), p.name


def test_legacy_camera_models_need_opt_in(monkeypatch):
    import camera_model
    import pytest

    legacy = REPO / "config/camera_models/sony_alpha_6100.yaml"
    monkeypatch.delenv(camera_model.ALLOW_LEGACY_ENV, raising=False)
    with pytest.raises(ValueError, match="camera_recipes/sony_alpha_6100.yaml"):
        camera_model.load_camera_model(legacy)
    monkeypatch.setenv(camera_model.ALLOW_LEGACY_ENV, "1")
    assert "sensor" in camera_model.load_camera_model(legacy)
