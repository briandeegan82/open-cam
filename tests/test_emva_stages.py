"""Stage functions split out of apply_emva_noise.main (R-29)."""

from __future__ import annotations

import math

import apply_emva_noise as aen
import numpy as np
import pytest


@pytest.mark.parametrize(
    "raw, want",
    [(True, True), (False, False), ("hard", True), ("Soft", False), (" off ", False), (1, True), (0, False)],
)
def test_parse_adc_clipping(raw, want):
    assert aen.parse_adc_clipping(raw) is want


def test_parse_adc_clipping_rejects_unknown():
    with pytest.raises(ValueError, match="adc.clipping"):
        aen.parse_adc_clipping("sometimes")


def test_dark_temperature_scale_doubling_and_arrhenius():
    assert aen.dark_temperature_scale(32.0, 20.0, 6.0, 0.0) == pytest.approx(4.0)
    assert aen.dark_temperature_scale(20.0, 20.0, 6.0, 1.1) == pytest.approx(1.0)
    t, t0 = 333.15, 293.15
    want = (t / t0) ** 1.5 * math.exp(1.1 / 8.617333262e-5 * (1 / t0 - 1 / t))
    assert aen.dark_temperature_scale(60.0, 20.0, 6.0, 1.1) == pytest.approx(want)


def test_relative_qe_rgb_normalises_to_max():
    wl = np.array([400.0, 500.0, 600.0])
    curves = [(wl, [0.2, 0.2, 0.2]), (wl, [0.4, 0.4, 0.4]), (wl, [0.1, 0.1, 0.1])]
    np.testing.assert_allclose(aen.relative_qe_rgb(curves, None, None), [0.5, 1.0, 0.25], rtol=1e-6)
    with pytest.raises(ValueError, match="non-positive"):
        aen.relative_qe_rgb([(wl, [0, 0, 0])] * 3, None, None)


def test_crosstalk_matrix_rows_normalised():
    m = aen.crosstalk_matrix_from_cfg({"matrix_3x3": [[2, 0, 0], [1, 1, 0], [0, 0, 4]]})
    np.testing.assert_allclose(m.sum(axis=1), 1.0)
    assert m.dtype == np.float32


def test_build_arg_parser_defaults():
    args = aen.build_arg_parser().parse_args([])
    assert args.seed == 0 and args.preview_percentile == 99.5 and args.config is None
