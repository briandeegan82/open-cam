from __future__ import annotations

import defect_correction as dc  # noqa: E402  (conftest puts tools/ on sys.path)
import numpy as np
import pytest

K, SIGMA_D, BLACK = 0.5, 2.0, 64.0
RGGB = dc.bayer_index_tile("RGGB")


def _noisy_flat(level_e: float, shape=(128, 128), seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = level_e + rng.standard_normal(shape) * np.sqrt(level_e + SIGMA_D**2)
    return (e / K + BLACK).astype(np.float32)


def _run(raw, tile=RGGB, **cfg):
    return dc.correct_defects(raw, tile, {"enabled": True, **cfg}, black_dn=BLACK, K_e_per_DN=K, sigma_d_e=SIGMA_D)


def test_bayer_offsets_stay_on_same_colour():
    offs = dc.same_colour_offsets(RGGB, 8)
    assert all(dy % 2 == 0 and dx % 2 == 0 for dy, dx in offs[(0, 0)])
    assert sorted(offs[(0, 1)][:4]) == [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def test_disabled_is_passthrough():
    raw = _noisy_flat(500.0)
    out, stats = dc.correct_defects(raw, RGGB, None, black_dn=BLACK, K_e_per_DN=K, sigma_d_e=SIGMA_D)
    assert out is raw and stats == {"enabled": False}


def test_noise_only_has_few_false_positives():
    _, stats = _run(_noisy_flat(500.0))
    assert stats["corrected_fraction"] < 1e-3


@pytest.mark.parametrize("polarity", ["both", "hot", "cold"])
def test_singlets_corrected_by_polarity(polarity):
    raw = _noisy_flat(500.0, seed=1)
    hot = [(10, 10), (40, 41), (77, 90)]
    cold = [(20, 30), (61, 61)]
    for y, x in hot:
        raw[y, x] += 2000.0
    for y, x in cold:
        raw[y, x] = BLACK
    out, stats = _run(raw, polarity=polarity)
    if polarity != "cold":
        assert all(abs(out[y, x] - (500.0 / K + BLACK)) < 150 for y, x in hot)
        assert stats["hot_count"] >= len(hot)
    else:
        assert all(out[y, x] == raw[y, x] for y, x in hot)
    if polarity != "hot":
        assert all(out[y, x] > BLACK + 500 for y, x in cold)
    else:
        assert all(out[y, x] == BLACK for y, x in cold)


def test_noiseless_edge_untouched():
    raw = np.full((64, 64), BLACK + 100.0, np.float32)
    raw[:, 32:] = BLACK + 3000.0
    out, stats = _run(raw)
    assert stats["corrected_count"] == 0
    np.testing.assert_array_equal(out, raw)


def test_generic_tile_layout():
    tile = np.array([[0, 1], [2, 0]])  # W R / B W
    raw = _noisy_flat(800.0, seed=2)
    raw[33, 17] += 3000.0
    out, stats = _run(raw, tile=tile)
    assert out[33, 17] < raw[33, 17] - 2000
    assert stats["corrected_fraction"] < 1e-3


def test_bad_inputs():
    with pytest.raises(ValueError):
        dc.bayer_index_tile("RGBX")
    with pytest.raises(ValueError):
        _run(_noisy_flat(10.0), polarity="up")
