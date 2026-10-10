"""Analytic ground-truth validation of the IQ-lab metrics (tools/validate_iqlab_metrics.py)."""

import json

import numpy as np
import pytest
import sfr_analysis as sfr
import validate_iqlab_metrics as v


@pytest.mark.parametrize("name", sorted(v.SUITES))
def test_suite_within_tolerance(name):
    rows = v.SUITES[name](seed=0)
    assert rows
    assert v.check(name, rows) == []


def test_check_flags_out_of_tolerance_row():
    row = {"read_noise_e": 3.0, "snr_threshold": 1.0, "db_abs": 2.0}
    fails = v.check("dr", [row])
    assert len(fails) == 1 and "db_abs" in fails[0]


def test_true_edge_mtf50_is_half_power_point():
    for sigma in (0.5, 1.0, 2.0):
        f50 = v.true_edge_mtf50(sigma)
        assert v.true_edge_mtf(np.array([f50]), sigma)[0] == pytest.approx(0.5, abs=1e-9)


def test_sharp_edge_bias_is_from_4x_binning():
    """At MTF50 ~ 0.32 cy/px the ISO-default 4x ESF binning reads ~2.5 % low; 8x removes it."""
    sigma = 0.5
    truth = v.true_edge_mtf50(sigma)
    img = v.blurred_slanted_edge(sigma, 5.0)
    bias4 = sfr.slanted_edge_sfr(img, oversampling=4).mtf50_cy_per_px / truth - 1
    bias8 = sfr.slanted_edge_sfr(img, oversampling=8).mtf50_cy_per_px / truth - 1
    assert -0.035 < bias4 < -0.015
    assert abs(bias8) < 0.01


def test_cli_writes_reports(tmp_path):
    rc = v.main(["--suites", "dr", "cdp", "--out-dir", str(tmp_path)])
    assert rc == 0
    data = json.loads((tmp_path / "iqlab_validation.json").read_text())
    assert set(data["results"]) == {"dr", "cdp"}
    assert "## cdp" in (tmp_path / "iqlab_validation.md").read_text()
