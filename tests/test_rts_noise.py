"""Random telegraph signal noise (apply_emva_noise.random_telegraph_offsets)."""

from __future__ import annotations

import apply_emva_noise as emva
import numpy as np
import pytest

SHAPE = (400, 500)
CFG = {"enabled": True, "pixel_fraction": 0.01, "amplitude_mean_e": 4.0, "high_state_probability": 0.3}


def test_disabled_returns_none_and_draws_nothing():
    rng = np.random.default_rng(5)
    off, stats = emva.random_telegraph_offsets(SHAPE, {}, 1, rng)
    assert off is None and stats == {"enabled": False}
    assert rng.random() == np.random.default_rng(5).random()


def test_fixed_pixels_two_levels_and_statistics():
    frames = [emva.random_telegraph_offsets(SHAPE, CFG, 7, np.random.default_rng(k)) for k in range(40)]
    offs = np.stack([f[0] for f in frames])
    stats = frames[0][1]
    assert stats["pixels"] == pytest.approx(0.01 * offs[0].size, rel=0.05)
    assert stats["amplitude_mean_e"] == pytest.approx(4.0, rel=0.05)
    active = offs.max(0) > 0
    # Same RTS pixels in every frame (fixed by the spatial seed); others never move.
    assert all(f[1]["pixels"] == stats["pixels"] for f in frames)
    assert np.all(offs[:, ~active] == 0)
    # Each RTS pixel takes exactly two values: 0 and its own amplitude.
    amp = offs.max(0)
    assert np.all((offs[:, active] == 0) | np.isclose(offs[:, active], amp[active]))
    assert (offs[:, active] > 0).mean() == pytest.approx(0.3, abs=0.01)


def test_dwell_times_set_high_probability_and_new_seed_moves_pixels():
    cfg = {**CFG, "tau_high_s": 0.01, "tau_low_s": 0.03}
    _, s = emva.random_telegraph_offsets(SHAPE, cfg, 7, np.random.default_rng(0))
    assert s["high_state_probability"] == pytest.approx(0.25)
    a, _ = emva.random_telegraph_offsets(SHAPE, CFG, 7, np.random.default_rng(0))
    b, _ = emva.random_telegraph_offsets(SHAPE, CFG, 8, np.random.default_rng(0))
    assert not np.array_equal(a > 0, b > 0)
    with pytest.raises(ValueError):
        emva.random_telegraph_offsets(SHAPE, {**CFG, "pixel_fraction": 2.0}, 7, np.random.default_rng(0))
