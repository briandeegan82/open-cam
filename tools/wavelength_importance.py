#!/usr/bin/env python3
"""Sensor-response-weighted wavelength sampling for pbrt's spectral film (opt-in).

The spectral film samples wavelengths uniformly over [lambdamin, lambdamax]. With a visible
RGB sensor behind an IR-cut filter about half of those samples land where the sensor barely
responds. pbrt patch 0012 adds ``"float bucketweights" [w_1 ... w_n]`` to ``Film "spectral"``:
wavelengths are drawn per bucket in proportion to ``w_b``, uniformly inside each bucket. The pdf
is constant within a bucket, so each bucket's average radiance stays unbiased; only the number of
samples per bucket changes (more where the sensor is sensitive, fewer elsewhere).

Weights here are the photon-weighted response ``sum_c QE_c(l) IRCF(l) l`` averaged per bucket,
mixed with a uniform share ``--uniform-fraction`` (default 0.25) so every bucket keeps samples.
A uniform share u multiplies the variance of the unweighted (e.g. NIR) buckets by up to 1/u, so
use this only when the spectral EXR is consumed by the sensor the weights were built for.

    venv/bin/python tools/wavelength_importance.py scene.pbrt --camera-model config/camera_models/sony_alpha_6100.yaml
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pbrt_spectral_exr_to_electrons import qe_stack_on_lambdas  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FILM_RE = re.compile(r'Film\s+"spectral"')


def sensor_qe_config(camera_model: dict) -> dict:
    qe = (camera_model.get("sensor") or {}).get("quantum_efficiency")
    if not qe:
        raise ValueError("camera model has no sensor.quantum_efficiency block")
    return qe


def bucket_weights(
    qe_cfg: dict,
    lambda_min: float,
    lambda_max: float,
    nbuckets: int,
    uniform_fraction: float = 0.25,
    repo: Path = REPO,
    oversample: int = 16,
) -> np.ndarray:
    """Normalised per-bucket sampling weights (mean 1): uniform share + photon-weighted response."""
    if not 0.0 < uniform_fraction <= 1.0:
        raise ValueError("uniform_fraction must be in (0, 1] (0 would leave buckets unsampled)")
    edges = np.linspace(lambda_min, lambda_max, nbuckets + 1)
    fine = np.concatenate(
        [
            np.linspace(a, b, oversample, endpoint=False) + (b - a) / (2 * oversample)
            for a, b in zip(edges[:-1], edges[1:])
        ]
    )
    qe = qe_stack_on_lambdas(repo, qe_cfg, fine).astype(np.float64)  # [3, nb * oversample]
    resp = (qe.sum(0) * fine).reshape(nbuckets, oversample).mean(1)
    if resp.sum() <= 0:
        raise ValueError("sensor response is zero over the film range")
    w = uniform_fraction / nbuckets + (1.0 - uniform_fraction) * resp / resp.sum()
    return w * nbuckets


def film_params(scene: str) -> tuple[float, float, int]:
    m = FILM_RE.search(scene)
    if not m:
        raise ValueError('scene has no Film "spectral" block')
    block = scene[m.end() : m.end() + 4000].split("\n\n")[0]

    def num(name: str, default: float) -> float:
        mm = re.search(rf'"(?:float|integer) {name}"\s*\[\s*([-0-9.eE+]+)', block)
        return float(mm.group(1)) if mm else default

    return num("lambdamin", 360.0), num("lambdamax", 830.0), int(num("nbuckets", 16))


def apply_to_scene(scene: str, weights: np.ndarray) -> str:
    if '"float bucketweights"' in scene:
        raise ValueError("scene already has bucketweights")
    m = FILM_RE.search(scene)
    if not m:
        raise ValueError('scene has no Film "spectral" block')
    vals = " ".join(f"{w:.6g}" for w in weights)
    return scene[: m.end()] + f' "float bucketweights" [ {vals} ]' + scene[m.end() :]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", type=Path, help="pbrt scene with a spectral film (edited in place unless --out)")
    ap.add_argument("--camera-model", type=Path, required=True, help="camera model YAML (sensor.quantum_efficiency)")
    ap.add_argument("--uniform-fraction", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    text = args.scene.read_text()
    lo, hi, nb = film_params(text)
    model = yaml.safe_load(args.camera_model.read_text())
    w = bucket_weights(sensor_qe_config(model), lo, hi, nb, args.uniform_fraction)
    (args.out or args.scene).write_text(apply_to_scene(text, w))
    print(f"bucketweights ({nb} buckets, {lo:g}-{hi:g} nm): min {w.min():.3f} max {w.max():.3f}")


if __name__ == "__main__":
    main()
