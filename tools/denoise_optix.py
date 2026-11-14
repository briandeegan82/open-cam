"""Opt-in NVIDIA OptiX denoiser for pbrt spectral EXRs (display/demo renders only; docs/DENOISING.md).

The same stage as the OIDN denoiser (tools/denoise_spectral_exr.py: albedo + normal guides from a pbrt ``gbuffer``
render, the single-pixel spike clamp, the lamp bypass), on an NVIDIA GPU through ``tools/optix_denoise`` (built
against the OptiX 7.7 SDK by ``--build``; ``$OPTIX_SDK``, default ``~/NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64``).

* ``hdr`` (default): ``R,G,B`` and the ``S0.<lambda>nm`` buckets three at a time, each denoised on its own, as the
  OIDN stage does.
* ``aov``: one ``OPTIX_DENOISER_MODEL_KIND_AOV`` call; the filter is predicted from ``R,G,B`` and the guides and
  applied to every bucket. On the night wet highway it did not keep the spectrum better than ``hdr``.

Like OIDN it is a learned filter that removes fine texture: never compute IQ metrics on its output.

    venv/bin/python tools/denoise_optix.py --build
    venv/bin/python tools/denoise_optix.py --exr-in out/x.exr --scene x.pbrt
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from denoise_spectral_exr import bypass_keep, clamp_fireflies, render_guides  # noqa: E402

SRC = REPO / "tools" / "optix_denoise" / "optix_denoise.cpp"
BINARY = REPO / "tools" / "optix_denoise" / "build" / "optix_denoise"
OPTIX_SDK = Path(os.environ.get("OPTIX_SDK", Path.home() / "NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64"))
MODES = ("hdr", "aov")
METRICS_WARNING = "Denoised render (NVIDIA OptiX): fine texture is altered; do not use for IQ metrics."


def build() -> Path:
    """Compile ``tools/optix_denoise`` with nvcc against ``$OPTIX_SDK``."""
    BINARY.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["nvcc", "-O2", "-std=c++17", f"-I{OPTIX_SDK / 'include'}", str(SRC), "-o", str(BINARY)]
    subprocess.run(cmd, check=True)
    return BINARY


def run_optix(mode: str, beauty: np.ndarray, layers: np.ndarray, albedo: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Denoise ``beauty`` (H, W, 3) and ``layers`` (N, H, W, 3) with the ``optix_denoise`` binary; returns both."""
    h, w, _ = beauty.shape
    n, c = layers.shape[0], layers.shape[3]
    with tempfile.TemporaryDirectory(prefix="opencam_optix_") as td:
        d = Path(td)
        np.concatenate([beauty.ravel(), layers.ravel()]).astype("<f4").tofile(d / "in.f32")
        albedo.astype("<f4").tofile(d / "alb.f32")
        normal.astype("<f4").tofile(d / "nrm.f32")
        cmd = [str(BINARY), mode, str(w), str(h), str(n), str(c), *(str(d / f) for f in ("in.f32", "alb.f32", "nrm.f32", "out.f32"))]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"optix_denoise failed (exit {r.returncode}): {(r.stderr or r.stdout)[-500:]}")
        out = np.fromfile(d / "out.f32", dtype="<f4")
    return out[: h * w * 3].reshape(h, w, 3), out[h * w * 3 :].reshape(n, h, w, c)


def denoise_cube(cube: np.ndarray, rgb: np.ndarray, albedo: np.ndarray, normal: np.ndarray, *, mode: str = "hdr",
                 firefly_ratio: float = 20.0, bypass: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Denoise the (H, W, K) buckets and (H, W, 3) ``rgb``; returns ``(cube, rgb, keep)`` (``keep``: ``bypass_keep``)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    h, w, k = cube.shape
    keep = bypass_keep(np.maximum(cube, 0).sum(-1), bypass)
    cube_c, _ = clamp_fireflies(np.maximum(cube, 0).astype(np.float32), firefly_ratio)
    rgb_c, _ = clamp_fireflies(np.maximum(rgb, 0).astype(np.float32), firefly_ratio)
    # buckets as 3-channel layers (OptiX 7.7 rejects FLOAT1 AOV layers); the last one zero-padded
    trip = np.concatenate([cube_c * keep, np.zeros((h, w, -k % 3), np.float32)], axis=2)
    layers = np.moveaxis(trip.reshape(h, w, -1, 3), 2, 0)
    rgb_d, lay = run_optix(mode, rgb_c * keep, layers, albedo, normal)
    den = np.moveaxis(lay, 0, 2).reshape(h, w, -1)[..., :k]
    return np.maximum(den, 0) + cube_c * (1 - keep), np.maximum(rgb_d, 0) + rgb_c * (1 - keep), keep


def denoise_exr(exr_in: Path, exr_out: Path, *, scene: Path, pbrt: Path, mode: str = "hdr", guide_spp: int = 16,
                firefly_ratio: float = 20.0, bypass: float = 2.0) -> dict:
    """Denoise the ``S0.*nm`` buckets and ``R,G,B`` of ``exr_in`` into ``exr_out``; writes ``<exr_out>.denoise.json``."""
    from exr_multispectral import parse_s0_wavelength_nm, read_separate_exr_channels, write_separate_channels_exr  # noqa: PLC0415

    ch = read_separate_exr_channels(exr_in)
    h, w = next(iter(ch.values())).shape
    names = sorted((n for n in ch if parse_s0_wavelength_nm(n) is not None), key=parse_s0_wavelength_nm)
    if not names or not all(c in ch for c in "RGB"):
        raise ValueError(f"{exr_in}: needs S0.*nm buckets and R,G,B")
    albedo, normal = render_guides(scene, pbrt, w, h, guide_spp)
    cube = np.stack([ch[n] for n in names], axis=2).astype(np.float32)
    rgb = np.stack([ch[c] for c in "RGB"], axis=2).astype(np.float32)
    den, rgb_d, keep = denoise_cube(cube, rgb, albedo, normal, mode=mode, firefly_ratio=firefly_ratio, bypass=bypass)
    out = dict(ch)
    out |= {n: den[..., j] for j, n in enumerate(names)}
    out |= {c: rgb_d[..., j] for j, c in enumerate("RGB")}
    write_separate_channels_exr(exr_out, out)
    lum = np.maximum(cube, 0).sum(-1)
    report = {
        "denoised": True,
        "warning": METRICS_WARNING,
        "denoiser": f"NVIDIA OptiX ({mode} model)",
        "n_spectral_buckets": len(names),
        "spectral_energy_ratio": float(den.sum() / max(float(lum.sum()), 1e-30)),
        "rgb_energy_ratio": float(rgb_d.sum() / max(float(np.maximum(rgb, 0).sum()), 1e-30)),
        "guide_spp": guide_spp,
        "firefly_ratio": firefly_ratio,
        "bypass": bypass,
        "bypass_pixels": int((keep < 1).sum()),
        "bypass_energy_fraction": float((lum * (1 - keep[..., 0])).sum() / max(float(lum.sum()), 1e-30)),
        "exr_in": str(exr_in),
        "exr_out": str(exr_out),
    }
    Path(f"{exr_out}.denoise.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", action="store_true", help="compile tools/optix_denoise first")
    ap.add_argument("--exr-in", type=Path)
    ap.add_argument("--exr-out", type=Path, default=None, help="default: <stem>_optix.exr")
    ap.add_argument("--scene", type=Path, help="pbrt scene of the render (for the albedo/normal guides)")
    ap.add_argument("--pbrt", type=Path, default=REPO / "third_party" / "pbrt-v4" / "build" / "pbrt")
    ap.add_argument("--mode", choices=MODES, default="hdr")
    ap.add_argument("--guide-spp", type=int, default=16)
    ap.add_argument("--firefly-ratio", type=float, default=20.0, help="single-pixel spike clamp first (0 = off)")
    ap.add_argument("--bypass", type=float, default=2.0, help="light above this x the 99th percentile skips OptiX (lamps; 0 = off)")
    a = ap.parse_args(argv)
    if a.build or not BINARY.is_file():
        build()
    if a.exr_in is None:
        return 0
    if a.scene is None:
        ap.error("--scene is needed for the guides")
    out = a.exr_out or a.exr_in.with_name(f"{a.exr_in.stem}_optix{a.exr_in.suffix}")
    report = denoise_exr(a.exr_in, out, scene=a.scene, pbrt=a.pbrt, mode=a.mode, guide_spp=a.guide_spp,
                         firefly_ratio=a.firefly_ratio, bypass=a.bypass)
    print(json.dumps(report, indent=2))
    print(f"warning: {METRICS_WARNING}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
