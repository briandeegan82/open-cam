"""Second-order HDR pixel effects (``noise.hdr.second_order``) against analytic expectations."""

from __future__ import annotations

import copy
from pathlib import Path

import hdr_pixel as hp
import numpy as np
import pytest
from camera_model import load_camera_model, noise_config_from_camera_model

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
