"""pbrt patches 0022/0024: light sampling of goniometric lamps.

Stock pbrt bounds a goniometric lamp by its mean intensity, so a narrow beam is picked rarely where
it dominates and its light lands as fireflies. "bvhbeam" (patch 0022) mixes in a peak-intensity tree;
the default "bvh" (patch 0024) scores each goniometric lamp by its intensity toward the shading point.
Both must keep the image mean ("bvhmean" is the stock sampler) and cut the per-pixel noise under the
beam, the exact scoring more than the mixture.

Patch 0035: with goniometric lamps and other lights in a scene, "bvh" draws one light from each of the
two groups at every vertex (light strata), gives a dim sky its share of the irradiance instead of half
the samples, and bounds area lights that have an "angularfilename" map by the map. "bvhexact" is the
one-light mixture of patch 0024. Patch 0036: "lampregularize" leaves directly seen surfaces alone.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from synthetic_data import REPO

sys.path.insert(0, str(REPO / "tools"))
from exr_multispectral import read_separate_exr_channels  # noqa: E402
from highway_night import write_y_exr  # noqa: E402

PBRT = Path(os.environ.get("OPENCAM_PBRT", REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"))
PBRT_TIMEOUT_S = int(os.environ.get("OPENCAM_PBRT_TIMEOUT", "600"))

if not PBRT.is_file() and os.environ.get("OPENCAM_REQUIRE_PBRT") == "1":
    raise RuntimeError(f"OPENCAM_REQUIRE_PBRT=1 but pbrt binary not found at {PBRT}")

pytestmark = pytest.mark.skipif(not PBRT.is_file(), reason=f"pbrt binary not built ({PBRT})")


def _beam_map(n: int = 64) -> np.ndarray:
    """Equal-area map: a 1000 cd hot spot around image +z (the square's centre), 1 cd elsewhere."""
    c = (np.arange(n) + 0.5) / n
    u, v = np.meshgrid(c, c)
    r = np.abs(2 * u - 1) + np.abs(2 * v - 1)  # 0 at +z
    return np.where(r < 0.1, 1000.0, 1.0).astype(np.float32)


# glossy wall, smooth-coated sphere and an area light: bounce paths and MIS hits on the area light
_BOUNCE_LINES = [
    'AttributeBegin AreaLightSource "diffuse" "blackbody L" [3000] "float scale" [5] '
    'Shape "bilinearmesh" "point3 P" [-3 2.5 3.9  -2 2.5 3.9  -3 3.5 3.9  -2 3.5 3.9] AttributeEnd',
    'Material "conductor" "float roughness" [0.05]',
    'Shape "bilinearmesh" "point3 P" [-6 0 4  6 0 4  -6 6 4  6 6 4]',
    'Material "coateddiffuse" "rgb reflectance" [0.6 0.1 0.1] "float roughness" [0.01]',
    'AttributeBegin Translate 0.5 0.5 1 Shape "sphere" "float radius" [0.5] AttributeEnd',
]


def _scene(tmp: Path, sampler: str, seed: int, bounces: bool = False) -> np.ndarray:
    write_y_exr(tmp / "beam.exr", _beam_map())
    lamps = [
        f'AttributeBegin Translate {x} 3 {z} LightSource "point" "spectrum I" "stdillum-A" "float scale" [20] AttributeEnd'
        for x in np.linspace(-10, 10, 6)
        for z in np.linspace(-10, 10, 5)
    ]
    name = f"s_{sampler}_{int(bounces)}.pbrt"
    (tmp / name).write_text(
        "\n".join(
            [
                "LookAt 0 3 -4  0 0.5 2  0 1 0" if bounces else "LookAt 0 6 -0.01  0 0 0  0 1 0",
                'Camera "perspective" "float fov" [30]',
                f'Sampler "independent" "integer pixelsamples" [{256 if bounces else 64}]',
                f'Integrator "volpath" "integer maxdepth" [{5 if bounces else 1}] "string lightsampler" "{sampler}"',
                f'Film "rgb" "integer xresolution" [48] "integer yresolution" [48] "string filename" "{sampler}.exr"',
                "WorldBegin",
                # beam points down: image +z is the light frame's +y, flipped by the rotation
                'AttributeBegin Translate 0.5 2 0 Rotate 180 1 0 0 LightSource "goniometric" '
                '"spectrum I" "stdillum-A" "string filename" "beam.exr" "float scale" [1] AttributeEnd',
                *lamps,
                'Material "diffuse" "rgb reflectance" [0.5 0.5 0.5]',
                'Shape "bilinearmesh" "point3 P" [-20 0 -20  20 0 -20  -20 0 20  20 0 20]',
                *(_BOUNCE_LINES if bounces else []),
            ]
        )
    )
    out = tmp / f"{sampler}_{seed}_{int(bounces)}.exr"
    subprocess.run(
        [str(PBRT), "--quiet", "--seed", str(seed), "--outfile", str(out), str(tmp / name)],
        check=True,
        cwd=tmp,
        timeout=PBRT_TIMEOUT_S,
    )
    d = read_separate_exr_channels(out)
    return np.mean([d[k] for k in ("R", "G", "B")], axis=0).astype(np.float64)


def test_beam_mixture_is_unbiased_and_quieter(tmp_path: Path) -> None:
    img = {s: [_scene(tmp_path, s, seed) for seed in (1, 2)] for s in ("bvh", "bvhbeam", "bvhmean")}
    mean = {s: np.mean(v) for s, v in img.items()}
    assert mean["bvh"] == pytest.approx(mean["bvhmean"], rel=0.03)
    assert mean["bvhbeam"] == pytest.approx(mean["bvhmean"], rel=0.03)
    avg = np.mean(img["bvh"][0] + img["bvh"][1]) / 2
    beam = (img["bvh"][0] + img["bvh"][1]) / 2 > 3 * avg
    assert beam.mean() > 0.01
    noise = {
        s: np.sqrt(np.mean((v[0] - v[1])[beam] ** 2) / 2) / np.mean((v[0] + v[1])[beam] / 2) for s, v in img.items()
    }
    assert noise["bvhbeam"] < 0.7 * noise["bvhmean"], noise
    assert noise["bvh"] < 0.9 * noise["bvhbeam"], noise


def test_beam_mixture_is_unbiased_with_bounces(tmp_path: Path) -> None:
    mean = {
        s: np.mean([_scene(tmp_path, s, seed, bounces=True).mean() for seed in (1, 2)])
        for s in ("bvh", "bvhbeam", "bvhmean")
    }
    assert mean["bvh"] == pytest.approx(mean["bvhmean"], rel=0.02)
    assert mean["bvhbeam"] == pytest.approx(mean["bvhmean"], rel=0.02)


def _night_scene(tmp: Path, sampler: str, seed: int, extra: str = "", maxdepth: int = 3) -> np.ndarray:
    """A lamp with a beam, a dim sky, a signal lamp with an angular map, on a glossy coated floor."""
    write_y_exr(tmp / "beam.exr", _beam_map())
    c = (np.arange(64) + 0.5) / 64
    u, v = np.meshgrid(c, c)
    write_y_exr(tmp / "signal.exr", (np.abs(2 * u - 1) + np.abs(2 * v - 1) < 0.6).astype(np.float32))  # around +z
    name = f"n_{sampler}_{maxdepth}_{len(extra)}.pbrt"
    (tmp / name).write_text(
        "\n".join(
            [
                "LookAt 0 1.5 -6  0 0 2  0 1 0",
                'Camera "perspective" "float fov" [40]',
                'Sampler "independent" "integer pixelsamples" [256]',
                f'Integrator "path" "integer maxdepth" [{maxdepth}] "string lightsampler" "{sampler}"{extra}',
                'Film "rgb" "integer xresolution" [48] "integer yresolution" [32] "string filename" "n.exr"',
                "WorldBegin",
                'LightSource "infinite" "blackbody L" [6500] "float scale" [0.002]',
                'AttributeBegin Translate 0 3 0 Rotate 180 1 0 0 LightSource "goniometric" '
                '"spectrum I" "stdillum-A" "string filename" "beam.exr" "float scale" [0.02] AttributeEnd',
                # signal lamp: faces -z (towards the camera), its map only emits around its +z axis
                'AttributeBegin Translate 1.5 0.6 4 Rotate 180 0 1 0 AreaLightSource "diffuse" "blackbody L" [2000] '
                '"float scale" [40] "string angularfilename" "signal.exr" '
                'Shape "bilinearmesh" "point3 P" [-0.2 -0.1 0  0.2 -0.1 0  -0.2 0.1 0  0.2 0.1 0] AttributeEnd',
                'AttributeBegin AreaLightSource "diffuse" "blackbody L" [3000] "float scale" [2] '
                'Shape "bilinearmesh" "point3 P" [-3 2.5 5.9  -2 2.5 5.9  -3 3.5 5.9  -2 3.5 5.9] AttributeEnd',
                'Material "coateddiffuse" "rgb reflectance" [0.3 0.3 0.3] "float roughness" [0.04]',
                'Shape "bilinearmesh" "point3 P" [-20 0 -20  20 0 -20  -20 0 20  20 0 20]',
                'Material "coateddiffuse" "rgb reflectance" [0.6 0.1 0.1] "float roughness" [0.03] "bool remaproughness" false',
                'AttributeBegin Translate -1 0.7 3 Shape "sphere" "float radius" [0.7] AttributeEnd',
            ]
        )
    )
    out = tmp / f"n_{sampler}_{seed}_{maxdepth}_{len(extra)}.exr"
    subprocess.run(
        [str(PBRT), "--quiet", "--seed", str(seed), "--outfile", str(out), str(tmp / name)],
        check=True,
        cwd=tmp,
        timeout=PBRT_TIMEOUT_S,
    )
    d = read_separate_exr_channels(out)
    return np.mean([d[k] for k in ("R", "G", "B")], axis=0).astype(np.float64)


def test_light_strata_are_unbiased_and_quieter(tmp_path: Path) -> None:
    seeds = (1, 2, 3, 4)
    img = {s: np.array([_night_scene(tmp_path, s, seed) for seed in seeds]) for s in ("bvh", "bvhexact", "bvhmean")}
    floor = np.s_[:, 20:, :]  # the lit floor in front of the camera
    mean = {s: v[floor].mean() for s, v in img.items()}
    sem = {s: v[floor].mean(axis=(1, 2)).std(ddof=1) / np.sqrt(len(seeds)) for s, v in img.items()}
    for ref in ("bvhexact", "bvhmean"):
        assert abs(mean["bvh"] - mean[ref]) < 4 * np.hypot(sem["bvh"], sem[ref]) + 0.01 * mean[ref], (mean, sem)
    var = {s: np.median(v[floor].var(axis=0, ddof=1)) for s, v in img.items()}
    assert var["bvh"] < 0.6 * var["bvhexact"], var


def test_lampregularize_leaves_direct_light_alone(tmp_path: Path) -> None:
    reg = ' "float lampregularize" [0.2]'
    direct = [np.mean([_night_scene(tmp_path, "bvh", s, e, maxdepth=1).mean() for s in (1, 2)]) for e in ("", reg)]
    assert direct[1] == pytest.approx(direct[0], rel=0.02)
    bounced = [np.mean([_night_scene(tmp_path, "bvh", s, e).mean() for s in (1, 2)]) for e in ("", reg)]
    assert bounced[1] == pytest.approx(bounced[0], rel=0.1)  # blurred, not removed
    with pytest.raises(subprocess.CalledProcessError):
        _night_scene(tmp_path, "bvh", 1, ' "float lampregularize" [2]')
