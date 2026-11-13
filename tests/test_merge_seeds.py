"""tools/merge_seeds.py: robust merge of independent renders (speckle becomes a local glow, energy is kept)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import merge_seeds as ms  # noqa: E402
from exr_multispectral import read_separate_exr_channels, write_separate_channels_exr  # noqa: E402

H, W, C, K = 40, 48, 4, 4


def _renders(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """K noisy renders of a smooth scene with one small bright object, plus speckles in single renders."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    truth = np.repeat((1.0 + 0.02 * xx + 0.01 * yy)[..., None], C, axis=2)
    truth[10:12, 20:22] = 200.0  # a lamp: in every render
    stack = truth[None] * rng.normal(1.0, 0.03, size=(K, H, W, C))
    for k, y, x, c in ((0, 30, 10, 1), (1, 5, 40, 2), (2, 25, 30, 0), (3, 33, 44, 3), (0, 18, 6, 0)):
        stack[k, y, x, c] += 400.0
    return stack, truth


def test_speckle_is_spread_and_energy_kept() -> None:
    stack, truth = _renders()
    merged, moved = ms.robust_merge(stack)
    plain = stack.mean(axis=0)
    np.testing.assert_allclose(merged.sum(axis=(0, 1)), plain.sum(axis=(0, 1)), rtol=1e-9)
    assert plain[30, 10, 1] > 90 and merged[30, 10, 1] < 12  # the dot is gone ...
    assert merged[28:33, 8:13, 1].sum() > plain[28:33, 8:13, 1].sum() * 0.7  # ... its light is still around it
    np.testing.assert_allclose(merged[10:12, 20:22], plain[10:12, 20:22], rtol=1e-12)  # the lamp is untouched
    assert moved[10:12, 20:22].max() == 0
    assert np.mean((merged - truth) ** 2) < 0.2 * np.mean((plain - truth) ** 2)


def test_clean_renders_are_the_plain_mean() -> None:
    rng = np.random.default_rng(1)
    stack = 5.0 * rng.normal(1.0, 0.05, size=(K, H, W, C))
    merged, moved = ms.robust_merge(stack)
    assert moved.max() == 0
    np.testing.assert_allclose(merged, stack.mean(axis=0), rtol=1e-12)


def test_negative_levels_are_not_pulled_down() -> None:
    rng = np.random.default_rng(2)
    stack = -5.0 + 0.2 * rng.normal(size=(K, H, W, C))  # e.g. sensor G under a red lamp
    stack[1, 7, 9, 0] += 300.0
    merged, moved = ms.robust_merge(stack)
    plain = stack.mean(axis=0)
    assert np.count_nonzero(moved) == 1 and moved[7, 9, 0] > 70
    assert merged[20, 20, 0] == pytest.approx(plain[20, 20, 0], rel=1e-12)
    np.testing.assert_allclose(merged.sum(), plain.sum(), rtol=1e-9)


def test_needs_three_renders_and_valid_parameters() -> None:
    stack, _ = _renders()
    with pytest.raises(ValueError, match="K >= 3"):
        ms.robust_merge(stack[:2])
    with pytest.raises(ValueError, match="ratio"):
        ms.robust_merge(stack, ratio=0.5)


def test_exr_merge_keeps_channels_and_writes_sidecar(tmp_path: Path) -> None:
    stack, _ = _renders()
    names = ["R", "G", "B", "S0.450nm", "S0.550nm", "S0.650nm", "Albedo.R"]
    paths = []
    for k in range(K):
        ch = {n: stack[k, ..., i % C].astype(np.float32) for i, n in enumerate(names)}
        paths.append(tmp_path / f"r{k}.exr")
        write_separate_channels_exr(paths[-1], ch)
    out = tmp_path / "merged.exr"
    assert ms.main(["--exr-in", *map(str, paths), "--exr-out", str(out)]) == 0
    merged = read_separate_exr_channels(out)
    assert sorted(merged) == sorted(names)
    plain = {n: np.mean([read_separate_exr_channels(p)[n] for p in paths], axis=0) for n in names}
    for n in names:
        assert float(merged[n].sum()) == pytest.approx(float(plain[n].sum()), rel=1e-5)
    np.testing.assert_allclose(merged["Albedo.R"], plain["Albedo.R"], rtol=1e-6)  # not a radiance channel: averaged
    assert merged["S0.650nm"][30, 10] < 0.2 * plain["S0.650nm"][30, 10]  # stack channel 1
    report = json.loads((tmp_path / "merged.exr.merge.json").read_text())
    assert report["n_renders"] == K and report["merged_seeds"] is True
    assert report["spectral_energy_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert 0 < report["spectral_moved_fraction"] < 0.2
    with pytest.raises(ValueError, match="at least 3"):
        ms.merge_exrs(paths[:2], out)
