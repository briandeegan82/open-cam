"""Generic periodic CFA (tools/cfa_mosaic.py) and its opt-in hook in apply_emva_noise."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import apply_emva_noise  # noqa: E402  (synthetic_data puts tools/ on sys.path)
import cfa_mosaic as cm
import numpy as np
import pytest
from spectral_sensor_forward import build_spatial_transmission_map
from synthetic_data import (
    QE_WAVELENGTHS_NM,
    SPECTRAL_LAMBDAS_NM,
    run_tool_main,
    write_curve,
    write_spectral_exr,
    write_yaml,
)

REPO = Path(__file__).resolve().parents[1]
H, W = 24, 36
LAYOUTS = [
    "RGGB",
    "RGBW",
    "RGBW_4X4",
    "RCCB",
    "RYYCY",
    "RCCG",
    "QUAD_BAYER",
    [["R", "G", "B"], ["G", "B", "R"], ["B", "R", "G"]],
]
PEAKS = {"R": 610.0, "G": 540.0, "B": 460.0, "W": None, "C": None, "Ye": 575.0, "Cy": 495.0}


def _scene() -> np.ndarray:
    lam = SPECTRAL_LAMBDAS_NM
    yy, xx = np.mgrid[0:H, 0:W]
    a = (0.2 + xx / W)[:, :, None]
    b = (0.3 + yy / H)[:, :, None]
    return (0.05 + a * np.exp(-0.5 * ((lam - 480) / 40) ** 2) + b * np.exp(-0.5 * ((lam - 620) / 50) ** 2)).astype(
        np.float32
    )


def _channel_qe(tmp: Path) -> dict:
    out = {}
    for ch, peak in PEAKS.items():
        v = (
            np.full(QE_WAVELENGTHS_NM.shape, 0.7)
            if peak is None
            else 0.6 * np.exp(-0.5 * ((QE_WAVELENGTHS_NM - peak) / 35) ** 2)
        )
        out[ch] = {"qe_csv": str(write_curve(tmp / f"qe_{ch}.csv", QE_WAVELENGTHS_NM, v))}
    return out


def _cfg(tmp: Path, exr: Path, cfa: dict, emva: dict | None = None, **proc) -> Path:
    cfg = {
        "sensor": {
            "pixel_pitch_um": 3.0,
            "f_number": 2.8,
            "integration_time_s": 0.01,
            "fill_factor": 1.0,
            "quantum_efficiency": {
                "red_csv": cfa["channels"]["R"]["qe_csv"],
                "green_csv": cfa["channels"]["G"]["qe_csv"],
                "blue_csv": cfa["channels"]["B"]["qe_csv"],
            },
        },
        "emva": {
            "overall_system_gain_K_e_per_DN": 0.25,
            "sigma_d_e": 0.0,
            "dsnu_std_e": 0.0,
            "prnu_std_fraction": 0.0,
            "dark_current_e_per_s": 0.0,
            "use_poisson_shot_noise": False,
            "black_level_DN": 64.0,
            **(emva or {}),
        },
        "adc": {"full_well_e": 1e6, "bit_depth": 16},
        "processing": {
            "linear_exr_mode": "integrate_qe",
            "preview_white_balance": {"enabled": False},
            "preview_color_correction": {"enabled": False},
            **proc,
        },
        "bayer": {"enabled": True, **cfa},
        "output": {"linear_rgb_in": str(exr), "raw_out": str(tmp / "noisy.raw16")},
    }
    return write_yaml(tmp / "noise.yaml", cfg)


def _run(tmp: Path, cfg: Path, seed: int = 0) -> tuple[np.ndarray, dict]:
    run_tool_main(apply_emva_noise.main, ["--repo-root", str(tmp), "--config", str(cfg), "--seed", str(seed)])
    stats = json.loads((tmp / "noisy_png" / "run_stats.json").read_text())
    return np.fromfile(tmp / "noisy.raw16", dtype=np.uint16), stats


@pytest.mark.parametrize("layout", LAYOUTS, ids=str)
def test_noise_free_pipeline_samples_each_site_exactly(layout) -> None:
    """Noise off: raw DN at every site == own-channel electrons / K + black (exact, <= ADC rounding)."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        exr = write_spectral_exr(tmp / "s.exr", _scene(), SPECTRAL_LAMBDAS_NM)
        cfa = {"layout": layout, "channels": _channel_qe(tmp), "demosaic": "bilinear"}
        raw, stats = _run(tmp, _cfg(tmp, exr, cfa))
        lay = cm.resolve_layout(cfa)
        sensor = {"pixel_pitch_um": 3.0, "f_number": 2.8, "integration_time_s": 0.01, "fill_factor": 1.0}
        e = apply_emva_noise.integrate_exr_spectral_qe(
            exr, tmp, {}, sensor, {}, channel_qe=lambda lam: cm.qe_stack_for_layout(tmp, lay, lam)
        )
        # Independent analytic check of one channel: E = L*pi/(4N^2), e = sum E*lam/(hc)*QE*dlam*A*t.
        from exr_multispectral import trapezoid_weights_nm  # noqa: PLC0415
        from sensor_radiometry import C_LIGHT, H_PLANCK  # noqa: PLC0415

        lam = SPECTRAL_LAMBDAS_NM
        qe0 = cm.qe_stack_for_layout(tmp, lay, lam)[0].astype(np.float64)
        # 1e-3 = default calibration.irradiance_scale_W_m2nm_per_unit
        k = (
            1e-3
            * np.pi
            / (4 * 2.8**2)
            * lam
            * 1e-9
            / (H_PLANCK * C_LIGHT)
            * qe0
            * trapezoid_weights_nm(lam)
            * 9e-12
            * 0.01
        )
        np.testing.assert_allclose(e[:, :, 0], _scene().astype(np.float64) @ k, rtol=1e-5)
        expected = cm.mosaic(e, lay).astype(np.float64) / 0.25 + 64.0
        assert np.max(np.abs(raw.reshape(H, W) - expected)) <= 0.5 + 1e-3
        idx = lay.site_index(H, W)
        for i in range(len(lay.channels)):  # every site holds its own channel, never another's
            others = [j for j in range(len(lay.channels)) if j != i]
            if others:
                wrong = np.min([np.abs(e[:, :, j] - e[:, :, i]) for j in others], axis=0)[idx == i]
                assert np.all(wrong > 0)
        assert stats["cfa"]["channels"] == list(lay.channels)
        assert (tmp / "noisy_png" / "noisy_demosaic_rgb8.png").is_file()


def test_layout_is_opt_in() -> None:
    assert cm.resolve_layout({"enabled": True, "pattern": "RGGB"}) is None
    assert cm.resolve_layout(None) is None
    lay = cm.resolve_layout({"layout": ["R C", "C B"]})
    assert lay.tile == (("R", "C"), ("C", "B")) and lay.qe_csv["C"].endswith("QE_mono.csv")
    with pytest.raises(ValueError, match="no QE CSV"):
        cm.resolve_layout({"layout": [["R", "X"]]})


def test_rggb_mosaic_and_bilinear_match_legacy_bayer() -> None:
    img = np.random.default_rng(1).random((16, 20, 3)).astype(np.float32)
    lay = cm.resolve_layout({"layout": "RGGB"})
    raw = cm.mosaic(img, lay)
    assert np.array_equal(raw, apply_emva_noise.bayer_sample_rgb(img, "RGGB"))
    legacy = apply_emva_noise.bilinear_demosaic(raw, "RGGB")
    np.testing.assert_allclose(cm.demosaic_bilinear(raw, lay)[2:-2, 2:-2], legacy[2:-2, 2:-2], atol=1e-6)


@pytest.mark.parametrize("layout", LAYOUTS, ids=str)
@pytest.mark.parametrize("method", ["bilinear", "gradient", "edge"])
def test_demosaic_preserves_samples_and_flat_fields(layout, method) -> None:
    lay = cm.resolve_layout({"layout": layout})
    rng = np.random.default_rng(2)
    raw = rng.random((18, 24))
    out = cm.demosaic(raw, lay, method)
    assert np.allclose(cm.mosaic(out, lay), raw, atol=1e-6)
    flat = cm.mosaic(
        np.broadcast_to(np.arange(1, len(lay.channels) + 1, dtype=float), (18, 24, len(lay.channels))), lay
    )
    np.testing.assert_allclose(
        cm.demosaic(flat, lay, method),
        np.broadcast_to(np.arange(1, len(lay.channels) + 1), (18, 24, len(lay.channels))),
        atol=1e-6,
    )


def test_gradient_demosaic_beats_bilinear_on_edges() -> None:
    """Colour-difference + gradient interpolation reduces error on a sharp grey edge (RGGB)."""
    lay = cm.resolve_layout({"layout": "RGGB"})
    img = np.zeros((32, 32, 3))
    img[:, 15:] = 1.0
    img[:, :, 0] *= 0.8
    raw = cm.mosaic(img, lay)
    e_bil = np.abs(cm.demosaic_bilinear(raw, lay) - img)[2:-2, 2:-2].mean()
    e_grad = np.abs(cm.demosaic_gradient(raw, lay) - img)[2:-2, 2:-2].mean()
    assert e_grad < 0.5 * e_bil


@pytest.mark.parametrize("layout", LAYOUTS, ids=str)
@pytest.mark.parametrize("method", ["gradient", "edge"])
def test_gradient_demosaic_does_not_overshoot_hard_edges(layout, method) -> None:
    """A 2-px lamp line at 100x its background must not drive a channel below its neighbours
    (the colour difference from the bright side used to go negative -> saturated green after a CCM)."""
    lay = cm.resolve_layout({"layout": layout})
    img = np.full((32, 32, len(lay.channels)), 100.0)
    img[:, 15:17] = 1.0e4
    img[20:, :] = 1.0e4
    raw = cm.mosaic(img, lay)
    out = cm.demosaic(raw, lay, method)
    assert np.allclose(cm.mosaic(out, lay), raw)
    assert out.min() >= 100.0 - 1e-3
    assert out.max() <= 1.0e4 + 1e-3


def test_site_crosstalk_conserves_charge_and_routes_by_source() -> None:
    lay = cm.resolve_layout({"layout": "RCCB"})
    raw = np.zeros((20, 20), dtype=np.float32)
    raw[10, 10] = 1000.0  # an R site (10%2==0, 10%2==0)
    cfg = {"enabled": True, "sigma_pixels": 0.3, "sigma_pixels_R": 0.6}
    out = cm.apply_site_crosstalk(raw, lay, cfg)
    assert out.sum() == pytest.approx(1000.0, rel=1e-5)
    narrow = cm.apply_site_crosstalk(raw, lay, {"enabled": True, "sigma_pixels": 0.3})
    assert out[10, 11] > narrow[10, 11] > 0  # wider red diffusion leaks more into the C neighbour


def test_quad_bayer_binning_and_remosaic() -> None:
    q = cm.resolve_layout({"layout": "QUAD_BAYER"})
    img = np.broadcast_to(np.array([3.0, 5.0, 7.0]), (16, 16, 3))
    b, bl = cm.bin_mosaic(cm.mosaic(img, q), q)
    assert bl.tile == (("R", "G"), ("G", "B")) and b.shape == (8, 8)
    np.testing.assert_allclose(b, 4.0 * cm.mosaic(img[:8, :8], bl))
    r, rl = cm.remosaic_to_bayer(cm.mosaic(img, q), q)
    np.testing.assert_allclose(r, cm.mosaic(img, rl), atol=1e-9)
    with pytest.raises(ValueError):
        cm.binned_layout(cm.resolve_layout({"layout": "RGGB"}))


def test_charge_binning_snr_gain_over_digital() -> None:
    """Read-noise-limited flat field: charge binning (one read) has ~2x the SNR of digital (four reads)."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        exr = write_spectral_exr(
            tmp / "s.exr", np.full((64, 64, SPECTRAL_LAMBDAS_NM.size), 0.05, np.float32), SPECTRAL_LAMBDAS_NM
        )
        res = {}
        for mode in ("charge", "digital"):
            cfa = {"layout": "QUAD_BAYER", "channels": _channel_qe(tmp), "binning": mode, "demosaic": "bilinear"}
            raw, st = _run(tmp, _cfg(tmp, exr, cfa, emva={"sigma_d_e": 8.0, "overall_system_gain_K_e_per_DN": 0.05}))
            g = (raw.reshape(32, 32).astype(float) - 64.0)[0::2, 1::2]  # G sites of the binned RGGB
            res[mode] = g.mean() / g.std()
            assert st["cfa"]["output_tile"] == [["R", "G"], ["G", "B"]]
        assert res["charge"] / res["digital"] == pytest.approx(2.0, rel=0.25)


def test_spectral_ccm_white_preserving_and_accurate_for_rggb() -> None:
    lay = cm.resolve_layout({"layout": "RGGB"})
    ccm, meta = cm.colorchecker_spectral_ccm(REPO, lay, ircf_csv="spectra/QE/interpolated/QE_IRCF.csv")
    assert ccm.shape == (3, 3)
    a = np.random.default_rng(0).random((10, 4))
    t = np.random.default_rng(1).random((10, 3))
    m = cm.fit_channel_ccm(a, t, white_index=3)
    np.testing.assert_allclose(a[3] @ m, t[3], atol=1e-10)


def test_generic_mode_requires_spectral_input() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        exr = write_spectral_exr(tmp / "s.exr", _scene(), SPECTRAL_LAMBDAS_NM)
        cfg = _cfg(tmp, exr, {"layout": "RCCB", "channels": _channel_qe(tmp)}, linear_exr_mode="rgb")
        with pytest.raises(ValueError, match="integrate_qe"):
            _run(tmp, cfg)


def test_spatial_transmission_map_supports_c_channels() -> None:
    m, _ = build_spatial_transmission_map(
        4, 6, {"enabled": False}, repo=REPO, wavelength_nm=np.arange(3.0), qe_rgb=np.ones((5, 3))
    )
    assert m.shape == (4, 6, 5)


def test_clip_highlights_renders_clipped_pixels_neutral() -> None:
    gains = np.array([1.3, 0.8, 1.2])
    clipped = np.full((2, 2, 3), 1000.0) * gains  # raw-clipped in every channel, then white-balanced
    out = cm.clip_highlights(clipped, gains, 1000.0)
    np.testing.assert_allclose(out, 800.0)
    mid = np.array([[[100.0, 200.0, 300.0]]])
    np.testing.assert_array_equal(cm.clip_highlights(mid, gains, 1000.0), mid)
    np.testing.assert_array_equal(cm.clip_highlights(clipped, gains, None), clipped)


def test_render_cfa_rgb_keeps_clipped_highlights_white() -> None:
    lay = cm.resolve_layout({"layout": "RGGB"})
    black, clip = 64.0, 959.0
    raw = np.full((16, 16), black + 300.0)
    raw[::2, ::2] = black + 200.0  # tinted scene: R and B below G
    raw[1::2, 1::2] = black + 250.0
    raw[4:12, 4:12] = black + clip  # clipped highlight: every site at the raw ceiling
    for clip_dn, neutral in ((None, False), (clip, True)):
        (rgb,), _ = cm.render_cfa_rgb(
            [raw], black, lay, repo=REPO, cfa_cfg={"demosaic": "bilinear"}, wb_method="gray_world", clip_dn=clip_dn
        )
        hi = rgb[7, 7] - black
        assert bool(np.ptp(hi) < 1e-6 * hi.max()) is neutral


def _zone_plate(n: int = 256) -> np.ndarray:
    """Colour zone plate, 0 -> 0.175 cycles/px at the edge centres (the docs/CFA_NIR.md PSNR table)."""
    yy, xx = np.mgrid[0:n, 0:n]
    zp = 0.5 + 0.5 * np.cos(np.pi * 0.35 * ((xx - n / 2) ** 2 + (yy - n / 2) ** 2) / n)
    return np.stack([zp, 0.8 * zp + 0.1, 0.6 * zp + 0.2], -1)


def _channels(rgb: np.ndarray, names) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    d = {"R": r, "G": g, "B": b, "W": r + g + b, "C": r + g + b, "Ye": r + g, "Cy": g + b, "Mg": r + b}
    return np.stack([d[c] for c in names], -1)


@pytest.mark.parametrize("layout", ["RGGB", "RCCB", "RYYCY", "RGBW_4X4"])
def test_edge_demosaic_beats_gradient_on_a_zone_plate(layout) -> None:
    lay = cm.resolve_layout({"layout": layout})
    full = _channels(_zone_plate(), lay.channels)
    raw = cm.mosaic(full, lay)

    def mse(m):
        return float(np.mean((cm.demosaic(raw, lay, m) - full)[8:-8, 8:-8] ** 2))

    assert cm.edge_applies(lay)
    assert mse("edge") < 0.8 * mse("gradient")


def test_edge_falls_back_without_a_dense_guide() -> None:
    lay = cm.resolve_layout({"layout": "RGBW"})
    raw = np.random.default_rng(3).random((16, 16))
    assert not cm.edge_applies(lay)
    np.testing.assert_array_equal(cm.demosaic(raw, lay, "edge"), cm.demosaic_gradient(raw, lay))


def test_quad_bayer_remosaic_method() -> None:
    q = cm.resolve_layout({"layout": "QUAD_BAYER"})
    full = _channels(_zone_plate(64), q.channels)
    raw = cm.mosaic(full, q)
    for m in ("gradient", "edge"):
        r, rl = cm.remosaic_to_bayer(raw, q, method=m)
        same = cm.mosaic(np.ones((64, 64, 3)), rl) * 0 + (cm.mosaic(full, q) == cm.mosaic(full, rl))
        # sites whose colour is unchanged keep the measured value
        np.testing.assert_allclose(r[same.astype(bool)], raw[same.astype(bool)], rtol=1e-6)
