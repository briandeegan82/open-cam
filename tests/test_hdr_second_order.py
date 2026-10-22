"""Second-order HDR pixel effects (``noise.hdr.second_order``) against analytic expectations."""

from __future__ import annotations

import copy
from pathlib import Path

import hdr_pixel as hp
import numpy as np
import pytest
from camera_model import load_camera_model, noise_config_from_camera_model
from cfa_mosaic import CfaLayout

REPO = Path(__file__).resolve().parent.parent


def _arch(recipe: str, **so) -> hp.HdrArchitecture:
    cm = load_camera_model(REPO / "config" / "camera_recipes" / f"{recipe}.yaml")
    cfg = copy.deepcopy(noise_config_from_camera_model(cm, "", ""))
    if so:
        cfg["hdr"]["second_order"] = {"enabled": True, **so}
    return hp.build_architecture(cfg["hdr"], hp.base_from_noise_config(cfg))


def _sim(arch, sig, seed=3):
    return hp.simulate_captures(
        arch, sig, np.random.default_rng(seed), spatial_rng=np.random.default_rng(7), use_poisson=False
    )


def test_disabled_block_is_bit_identical():
    sig = np.full((16, 20), 500.0)
    a0, a1 = _arch("default_hdr_dcg"), _arch("default_hdr_dcg")
    a1 = hp.HdrArchitecture(**{**a1.__dict__, "second_order": hp._second_order_cfg({"enabled": False}, "dcg")})
    d0, d1 = _sim(a0, sig), _sim(a1, sig)
    assert all(np.array_equal(d0[k], d1[k]) for k in d0)
    with pytest.raises(ValueError):
        _arch("default_hdr_dcg", spd_offset_e=5.0)


def test_row_fpn_sets_row_mean_spread():
    a = _arch("default_hdr_dcg", row_fpn_e=20.0)
    r = a.readouts[0]
    dn = _sim(a, np.full((64, 4000), 200.0))[r.name]
    row_e = (dn.mean(axis=1) - r.black_DN) * r.K_e_per_DN
    assert row_e.std() == pytest.approx(20.0, rel=0.25)


def test_adc_inl_bow_peaks_at_mid_scale():
    a0, a1 = _arch("default_hdr_dcg"), _arch("default_hdr_dcg", adc_inl_dn=2.0)
    r = a0.readouts[0]
    sig = np.full((64, 512), 0.5 * (r.max_dn - r.black_DN) * r.K_e_per_DN)
    diff = _sim(a1, sig)[r.name].mean() - _sim(a0, sig)[r.name].mean()
    assert diff == pytest.approx(2.0, abs=0.15)


def test_pink_row_noise_has_unit_slope_psd():
    rng = np.random.default_rng(0)
    psd = np.mean([np.abs(np.fft.rfft(hp._pink_rows(rng, 1024, 3.0))) ** 2 for _ in range(200)], axis=0)
    f = np.fft.rfftfreq(1024)
    slope = np.polyfit(np.log(f[2:]), np.log(psd[2:]), 1)[0]
    assert slope == pytest.approx(-1.0, abs=0.1)
    assert hp._pink_rows(rng, 1024, 3.0).std() == pytest.approx(3.0)


def test_blooming_spills_into_column_neighbours_only():
    a0, a1 = _arch("default_hdr_dcg"), _arch("default_hdr_dcg", blooming_fraction=0.1)
    fw = a0.collector("pd").full_well_e
    r = a0.readouts[-1]
    sig = np.zeros((9, 9))
    sig[4, 4] = 3.0 * fw
    d = (_sim(a1, sig)[r.name] - _sim(a0, sig)[r.name]) * r.K_e_per_DN
    assert d[3, 4] == pytest.approx(0.2 * fw, rel=0.05) and d[5, 4] == pytest.approx(0.2 * fw, rel=0.05)
    assert abs(d[4, 3]) < 2 * r.K_e_per_DN and abs(d[4, 5]) < 2 * r.K_e_per_DN


def test_spd_offset_and_crosstalk_bias_the_merged_signal():
    base = _arch("default_hdr_split_pixel")
    tr = dict(base.transitions_e())
    lpd_top = max(t for n, t in tr.items() if n.startswith("lpd"))
    sig = np.full((32, 64), 0.5 * (lpd_top + base.max_reference_e))
    r_spd = base.collector("spd").response

    def merged(a):
        return hp.merge_captures(a, _sim(a, sig))[0].mean()

    m0 = merged(base)
    off = merged(_arch("default_hdr_split_pixel", spd_offset_e=40.0)) - m0
    assert off == pytest.approx(40.0 / r_spd, rel=0.1)
    xt = merged(_arch("default_hdr_split_pixel", spd_crosstalk_fraction=0.002)) - m0
    r_lpd = base.collector("lpd").response
    assert xt == pytest.approx(0.002 * r_lpd / r_spd * sig.mean(), rel=0.1)


def test_stochastic_switching_spreads_the_transition():
    def invalid_fraction(a):
        sig = np.full((64, 256), 0.97 * a.transitions_e()[0][1])
        hist = hp.merge_captures(a, _sim(a, sig), spatial_rng=np.random.default_rng(1))[1]
        h = np.asarray(hist["n_valid_readouts_histogram"], float)
        return 1.0 - h[-1] / h.sum()

    assert invalid_fraction(_arch("default_hdr_dcg")) < 0.01
    frac = invalid_fraction(_arch("default_hdr_dcg", switch_threshold_std=0.03))
    assert 0.08 < frac < 0.25  # Phi(-1) = 0.16


def test_adc_dnl_code_widths_and_missing_codes():
    t0 = hp.dnl_transitions(4095, 0.0, np.random.default_rng(0))
    ramp = np.linspace(-3.0, 4100.0, 200001) + 1e-7  # no exact halves
    assert np.array_equal(np.searchsorted(t0, ramp, side="right"), np.clip(np.rint(ramp), 0, 4095))
    t = hp.dnl_transitions(4095, 0.3, np.random.default_rng(1))
    assert t[0] == pytest.approx(0.5) and t[-1] == pytest.approx(4094.5)
    counts = np.bincount(np.searchsorted(t, np.linspace(0.0, 4095.0, 4_095_001), side="right"), minlength=4096)
    widths = counts[1:-1] / 1000.0
    assert widths.std() == pytest.approx(0.3, rel=0.1)
    assert np.allclose(widths, np.diff(t), atol=2e-3)
    inl = t - (np.arange(4095) + 0.5)
    assert np.abs(inl).max() < 5 * 0.3 / np.sqrt(2)  # bounded, not a random walk
    missing = np.mean(np.diff(hp.dnl_transitions(4095, 1.0, np.random.default_rng(2))) == 0)
    assert 0.12 < missing < 0.2  # about Phi(-1) = 0.16


def test_adc_dnl_is_a_fixed_code_table_in_the_readouts():
    a = _arch("default_hdr_dcg", adc_dnl_lsb=0.5)
    r = a.readouts[0]
    sig = np.linspace(0.0, 0.5 * (r.max_dn - r.black_DN) * r.K_e_per_DN, 64 * 512).reshape(64, 512)
    d1, d2 = _sim(a, sig, seed=3)[r.name], _sim(a, sig, seed=4)[r.name]
    d0 = _sim(_arch("default_hdr_dcg"), sig, seed=3)[r.name]
    assert not np.array_equal(d1, d0)
    assert np.abs(d1 - d0).max() <= 2  # transitions move by < 2 LSB
    codes = np.unique(np.concatenate([d1.ravel(), d2.ravel()]))
    t = hp.dnl_transitions(r.max_dn, 0.5, np.random.default_rng(7))
    assert codes.size > 100 and t.size == int(r.max_dn)


def test_lofic_soft_knee_compresses_hcg_and_conserves_charge():
    a0, a1 = _arch("default_hdr_lofic"), _arch("default_hdr_lofic", lofic_knee_e=400.0)
    fw = a0.collector("pd").full_well_e
    q = np.array([0.5 * fw, 0.9 * fw, fw, 2.0 * fw])
    qpd = hp.lofic_soft_split(q, fw, 400.0)
    assert qpd[2] == pytest.approx(fw - 400.0 * np.log(2.0), rel=1e-9)
    assert qpd[0] == pytest.approx(0.5 * fw - 400.0 * np.exp(-0.5 * fw / 400.0), abs=1e-6)
    assert qpd[3] == pytest.approx(fw, abs=1e-3)
    assert np.array_equal(hp.lofic_soft_split(q, fw, 0.0), np.minimum(q, fw))
    sig = np.full((32, 64), 0.85 * fw)
    h, lc = a0.readouts[0], a0.readouts[1]
    s0, s1 = _sim(a0, sig), _sim(a1, sig)
    deficit = 400.0 * np.log1p(np.exp((0.85 * fw - fw) / 400.0))
    assert (s0[h.name] - s1[h.name]).mean() * h.K_e_per_DN == pytest.approx(deficit, rel=0.05)
    assert np.array_equal(s0[lc.name], s1[lc.name])
    m0, m1 = hp.merge_captures(a0, s0)[0].mean(), hp.merge_captures(a1, s1)[0].mean()
    assert m1 - m0 < -0.5 * deficit  # the HCG-weighted merge inherits the compression
    with pytest.raises(ValueError):
        _arch("default_hdr_dcg", lofic_knee_e=100.0)


def test_spd_centroid_offset_shifts_only_the_spd_image():
    yy, xx = np.mgrid[0:64, 0:64].astype(float)
    blob = np.exp(-((xx - 30.0) ** 2 + (yy - 33.0) ** 2) / (2 * 3.0**2))
    sh = hp.spd_centroid_shift(blob, 0.4, -0.25)
    cx, cy = (sh * xx).sum() / sh.sum(), (sh * yy).sum() / sh.sum()
    assert cx == pytest.approx(30.0 - 0.4, abs=0.01) and cy == pytest.approx(33.0 + 0.25, abs=0.01)
    mosaic = hp.spd_centroid_shift(np.repeat(np.repeat(blob[::2, ::2], 2, 0), 2, 1), 0.4, 0.0, "RGGB")
    assert np.allclose(mosaic[0::2, 0::2], mosaic[1::2, 1::2])  # planes shifted alike, colours not mixed
    assert hp.spd_centroid_shift(blob, 0.0, 0.0) is not None and np.array_equal(hp.spd_centroid_shift(blob, 0, 0), blob)

    rccb = CfaLayout(tile=(("R", "C"), ("C", "B")), qe_csv={"R": "r", "C": "c", "B": "b"})
    level = rccb.site_index(64, 64).astype(float) * 4.0 + 1.0  # a different level per channel
    g = hp.spd_centroid_shift(level * blob, 0.4, 0.0, cfa_layout=rccb)
    for i in (0, 1):
        for j in (0, 1):  # each tile site is shifted on its own sub-lattice, so channels don't mix
            want = np.maximum(level[i, j] * hp._fourier_shift(blob[i::2, j::2], 0.2, 0.0), 0.0)
            assert np.allclose(g[i::2, j::2], want, atol=1e-12)
    assert np.array_equal(g, hp.spd_centroid_shift(level * blob, 0.4, 0.0, "RGGB"))
    rgbw = CfaLayout(tile=(("R", "G", "B", "W"),) * 4, qe_csv={c: c for c in "RGBW"})
    g4 = hp.spd_centroid_shift(blob, 0.4, 0.0, cfa_layout=rgbw)
    assert np.allclose(g4[:, 1::4], np.maximum(hp._fourier_shift(blob[:, 1::4], 0.1, 0.0), 0.0), atol=1e-12)
    arch = _arch("default_hdr_split_pixel", spd_centroid_offset_x_px=0.4)
    caps = hp.simulate_captures(
        arch,
        500.0 * level,
        np.random.default_rng(3),
        spatial_rng=np.random.default_rng(7),
        cfa_layout=rccb,
        use_poisson=False,
    )
    assert set(caps) == {r.name for r in arch.readouts}

    base = _arch("default_hdr_split_pixel")
    lpd_top = max(t for n, t in base.transitions_e() if n.startswith("lpd"))
    edge = np.clip((xx - 32.0) / 6.0 + 0.5, 0.0, 1.0)  # a 6 px ramp, all of it above the LPD transition
    sig = lpd_top * (2.0 + 20.0 * edge)

    def half_level_x(a):
        m = hp.merge_captures(a, _sim(a, sig))[0].mean(axis=0)
        return np.interp(0.5 * (m[2] + m[-3]), m[2:-2], xx[0, 2:-2])

    a1 = _arch("default_hdr_split_pixel", spd_centroid_offset_x_px=0.5)
    assert half_level_x(a1) - half_level_x(base) == pytest.approx(-0.5, abs=0.05)
    with pytest.raises(ValueError):
        _arch("default_hdr_dcg", spd_centroid_offset_x_px=0.5)
