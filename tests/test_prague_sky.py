"""Prague sky model port: format reader and interpolation on a synthetic dataset; reference values
from the authors' C++ CLI when the real dataset is available ($OPEN_CAM_PRAGUE_DATASET)."""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from prague_sky import DATASET_ENV, PragueSky, dataset_path  # noqa: E402

VIS, ALB, ALT, ELE = [20.0, 60.0], [0.0, 1.0], [0.0], [0.0, 45.0, 90.0]
CH_START, CH_WIDTH, N_CH = 380.0, 40.0, 2
BREAKS = [0.0, np.pi]
ZEN_SCALE = 4.0


def _a(vi: int, ai: int, ei: int, ch: int) -> float:
    return 1.0 + vi + 2 * ai + 4 * ei + 8 * ch


def _write_synthetic(path: Path) -> None:
    out = bytearray()
    for vals in (VIS, ALB, ALT, ELE):
        out += struct.pack("<i", len(vals)) + np.asarray(vals, "<f8").tobytes()
    out += struct.pack("<i", N_CH) + struct.pack("<d", CH_START) + struct.pack("<d", CH_WIDTH)
    out += struct.pack("<i", 1)  # rank
    for _ in range(3):  # sun, zenith, emph breaks
        out += struct.pack("<i", 2) + np.asarray(BREAKS, "<f8").tobytes()
    for vi in range(len(VIS)):
        for ai in range(len(ALB)):
            for _alt in ALT:
                for ei in range(len(ELE)):
                    for ch in range(N_CH):
                        a = _a(vi, ai, ei, ch)
                        out += np.asarray([a, 3 * a], "<f2").tobytes()  # sun: a at gamma 0, 3a at pi
                        out += struct.pack("<d", ZEN_SCALE)
                        out += np.asarray([0.5 * ZEN_SCALE] * 2, "<f2").tobytes()  # zenith: 0.5
                        out += np.asarray([1.0, 1.0], "<f2").tobytes()  # emph
    path.write_bytes(bytes(out))


@pytest.fixture
def synthetic(tmp_path: Path) -> PragueSky:
    p = tmp_path / "synthetic.dat"
    _write_synthetic(p)
    return PragueSky(p)


def test_reader_metadata(synthetic: PragueSky) -> None:
    assert synthetic.channels == N_CH and synthetic.rank == 1
    np.testing.assert_allclose(synthetic.channel_centres_nm, [400.0, 440.0])
    np.testing.assert_allclose(synthetic.elevations, ELE)


def test_multilinear_interpolation(synthetic: PragueSky) -> None:
    ele = 22.5
    sun = np.array([np.cos(np.radians(ele)), 0.0, np.sin(np.radians(ele))])
    r = synthetic.radiance(sun[None], ele, visibility_km=40.0, albedo=0.25)  # view at the sun: gamma 0
    expect = [0.5 * (1.0 + 0.5 + 2 * 0.25 + 4 * 0.5 + 8 * ch) for ch in range(N_CH)]
    np.testing.assert_allclose(r[0], expect, rtol=1e-12)


def test_sun_angle_dependence(synthetic: PragueSky) -> None:
    d = np.array([[0.0, 0.0, 1.0]])  # zenith, sun at 45 deg elevation: gamma = 45 deg
    r = synthetic.radiance(d, 45.0, visibility_km=20.0, albedo=0.0, channels=[0])
    a = _a(0, 0, 1, 0)
    np.testing.assert_allclose(r[0, 0], 0.5 * a * (1.0 + 2.0 * 0.25), rtol=1e-9)


def test_dataset_path_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(DATASET_ENV, raising=False)
    with pytest.raises(FileNotFoundError, match=DATASET_ENV):
        dataset_path(None)
    with pytest.raises(FileNotFoundError, match="not found"):
        dataset_path(tmp_path / "missing.dat")


def _fisheye(row: int, col: int, res: int, side: bool) -> np.ndarray:
    r = res / 2
    sx, sy = (row + 0.5 - r) / r, (col + 0.5 - r) / r
    den = sx * sx + sy * sy + 1
    if side:
        return np.array([-2 * sy / den, -(den - 2) / den, -2 * sx / den])
    return np.array([2 * sx / den, 2 * sy / den, -(den - 2) / den])


# PragueSkyModelCli (PetrVevoda/pragueskymodel) on PragueSkyModelDatasetSWIR.dat, -res 32, pixel [row, col]:
# (channel index, side view, elevation, visibility, albedo, azimuth, [(row, col, radiance), ...])
CLI_REFERENCE = [
    (
        4,
        False,
        5.3,
        30.0,
        0.2,
        40.0,
        [
            (16, 16, 0.014257491566240788),
            (4, 16, 0.029483402147889137),
            (16, 4, 0.028018685057759285),
            (10, 25, 0.026882214471697807),
            (28, 12, 0.034983329474925995),
        ],
    ),
    (11, True, -2.0, 59.4, 0.5, 0.0, [(4, 16, 0.0004549055593088269), (10, 25, 0.0011633590329438448)]),
]


@pytest.mark.skipif(not os.environ.get(DATASET_ENV), reason=f"{DATASET_ENV} not set (dataset not redistributable)")
@pytest.mark.parametrize("case", CLI_REFERENCE)
def test_matches_reference_cli(case) -> None:
    ch, side, ele, vis, alb, azi, pts = case
    sky = PragueSky()
    if sky.channels != 55:
        pytest.skip("reference values are for the 55-channel SWIR dataset")
    d = np.stack([_fisheye(r, c, 32, side) for r, c, _ in pts])
    got = sky.radiance(d, ele, vis, alb, azi, channels=[ch])[:, 0]
    np.testing.assert_allclose(got, [v for *_, v in pts], rtol=1e-6)


@pytest.mark.skipif(not os.environ.get(DATASET_ENV), reason=f"{DATASET_ENV} not set (dataset not redistributable)")
def test_highway_builder_prague_sky(tmp_path: Path) -> None:
    import highway_spectral_sky as ss
    from test_highway_spectral_sky import build_offline

    m, scene = build_offline(tmp_path, "--sky", "prague", "--sun-elevation", "40", "--prague-visibility-km", "40")
    sky = m["lighting"]["sky"]["spectral"]
    assert '"spectrum L" ["spd/sky_basis_0.spd"' in scene
    exr = tmp_path / "scene" / "textures" / "sky_prague_e40.0_v40.0_daylight.exr"
    assert ss.read_weight_exr(exr).shape == (ss.HOSEK_MAP_SIZE, ss.HOSEK_MAP_SIZE, 3)
    assert sky["visibility_km"] == 40.0 and sky["dataset"] == Path(os.environ[DATASET_ENV]).name
    hosek, _ = build_offline(tmp_path / "hosek", "--sky", "hosek", "--sun-elevation", "40")
    ratio = (
        sky["model_illuminance_horizontal_lux"]
        / hosek["lighting"]["sky"]["spectral"]["model_illuminance_horizontal_lux"]
    )
    assert 0.5 < ratio < 2.0


def test_builder_prague_needs_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from test_highway_spectral_sky import build_offline

    monkeypatch.delenv(DATASET_ENV, raising=False)
    with pytest.raises(SystemExit, match=DATASET_ENV):
        build_offline(tmp_path, "--sky", "prague")


def test_highway_swir_prague_tails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import test_prague_sky as tp
    from test_highway_spectral_sky import build_offline

    monkeypatch.setattr(tp, "N_CH", 31)  # 380-1620 nm
    ds = tmp_path / "swir.dat"
    tp._write_synthetic(ds)
    monkeypatch.setenv(DATASET_ENV, str(ds))
    m, _ = build_offline(
        tmp_path,
        "--sky", "prague", "--sun-elevation", "40",
        "--spectral-lambda-max", "1600", "--film-lambda-min", "900", "--spectral-nbuckets", "36",
    )  # fmt: skip
    tails = m["lighting"]["sky"]["spectral"]["swir_tails"]
    assert tails["channels_nm"][0] == 880.0 and tails["fit_relative_rms"] < 0.05
    assert "Prague" in m["swir"]["sky"]
    b = np.loadtxt(tmp_path / "scene" / "spd" / "sky_basis_0.spd")
    assert b[-1, 0] == 1600.0 and np.all(b[:, 1] >= 0.0)
