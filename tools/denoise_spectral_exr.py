"""Opt-in Monte Carlo denoiser for pbrt spectral EXRs (Intel Open Image Denoise, display/demo renders only).

pbrt's path-traced output carries Monte Carlo noise and fireflies (rare high-energy paths, e.g. specular bounces
that reach a small lamp) that converge only as 1/sqrt(spp). This stage runs Intel Open Image Denoise
(``oidnDenoise``, Apache-2.0, fetched by ``tools/fetch_oidn.py``) on the spectral EXR *before* the sensor model,
so the sensor noise that follows is still physical and only the renderer noise is removed.

Every ``S0.<lambda>nm`` bucket is denoised (three buckets per OIDN call, guided by albedo + normal from a cheap
pbrt ``gbuffer`` render of the same scene); ``R,G,B`` are denoised as one image; other channels are copied.

OIDN is a learned filter. It keeps the mean and hard edges, but it also removes fine stochastic texture that
looks like noise (on the IQ-lab dead-leaves card it removed about half of the 1-3 px detail). Never compute IQ
metrics (texture MTF, SNR, colour noise) on a denoised render. The stage is off by default, writes a
``*.denoise.json`` sidecar next to the output and keeps the noisy original. See docs/DENOISING.md.

    venv/bin/python tools/denoise_spectral_exr.py --exr-in out/x_spectral.exr --scene scenes/generated/x.pbrt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OIDN = REPO / "third_party" / "oidn" / "bin" / "oidnDenoise"
GUIDE_CHOICES = ("albedo_normal", "albedo", "none")
QUALITY_CHOICES = ("high", "balanced", "fast")
METRICS_WARNING = "Denoised render (Intel Open Image Denoise): fine texture is altered; do not use for IQ metrics."


def find_oidn(explicit: str | Path | None = None) -> Path:
    """``oidnDenoise`` from ``explicit``, ``$OPENCAM_OIDN``, ``third_party/oidn`` or ``PATH``."""
    candidates = [explicit, os.environ.get("OPENCAM_OIDN"), DEFAULT_OIDN, shutil.which("oidnDenoise")]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    raise FileNotFoundError(
        "oidnDenoise not found (checked --oidn, $OPENCAM_OIDN, third_party/oidn/bin, PATH). "
        "Install it with: venv/bin/python tools/fetch_oidn.py"
    )


def write_pfm(path: Path, img: np.ndarray) -> None:
    """Little-endian 3-channel PFM (rows stored bottom-to-top)."""
    img = np.asarray(img, dtype=np.float32)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"PFM writer expects HxWx3, got {img.shape}")
    h, w, _ = img.shape
    with open(path, "wb") as f:
        f.write(f"PF\n{w} {h}\n-1.0\n".encode("ascii"))
        f.write(np.ascontiguousarray(img[::-1], dtype="<f4").tobytes())


def read_pfm(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        kind = f.readline().strip()
        if kind not in (b"PF", b"Pf"):
            raise ValueError(f"{path}: not a PFM file")
        w, h = (int(v) for v in f.readline().split())
        scale = float(f.readline())
        n = 3 if kind == b"PF" else 1
        data = np.frombuffer(f.read(), dtype="<f4" if scale < 0 else ">f4", count=w * h * n)
    img = data.reshape(h, w, n)[::-1].astype(np.float32)
    return np.repeat(img, 3, axis=2) if n == 1 else img


def guide_scene_text(scene_text: str, xres: int, yres: int, spp: int, filename: str) -> str:
    """``scene_text`` with the film swapped for a camera-space ``gbuffer`` film and ``spp`` samples."""
    film = (
        f'Film "gbuffer"\n    "string filename" ["{filename}"]\n    "integer xresolution" [{xres}]\n'
        f'    "integer yresolution" [{yres}]\n    "bool savefp16" false'
    )
    s, n = re.subn(r'Film "[a-z]+".*?(?=\n\S)', film, scene_text, count=1, flags=re.S)
    if n != 1:
        raise ValueError("scene has no Film block")
    s = re.sub(r'"integer pixelsamples" \[\s*\d+\s*\]', f'"integer pixelsamples" [{spp}]', s, count=1)
    # The guides only need geometry and albedo; the 4-wavelength dispersion split would just cost time.
    return s.replace('"bool dispersionsplit" true', '"bool dispersionsplit" false')


def render_guides(scene: Path, pbrt: Path, xres: int, yres: int, spp: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """``(albedo, normal)`` HxWx3 from a ``gbuffer`` render of ``scene`` (written next to it, then removed)."""
    from exr_multispectral import read_separate_exr_channels  # noqa: PLC0415

    scene, pbrt = Path(scene).resolve(), Path(pbrt).resolve()
    tag = f"_denoise_guides_{os.getpid()}"
    exr = scene.parent / f"{scene.stem}{tag}.exr"
    guide = scene.parent / f"{scene.stem}{tag}.pbrt"
    guide.write_text(guide_scene_text(scene.read_text(), xres, yres, spp, exr.name))
    try:
        subprocess.run([str(pbrt), "--quiet", guide.name], cwd=scene.parent, check=True)
        ch = read_separate_exr_channels(exr)
    finally:
        guide.unlink(missing_ok=True)
        exr.unlink(missing_ok=True)
    albedo = np.stack([ch["Albedo.R"], ch["Albedo.G"], ch["Albedo.B"]], axis=2)
    normal = np.stack([ch["N.X"], ch["N.Y"], ch["N.Z"]], axis=2)
    if albedo.shape[:2] != (yres, xres):
        raise ValueError(f"guide render is {albedo.shape[:2]}, expected {(yres, xres)}")
    return np.nan_to_num(albedo), np.nan_to_num(normal)


class OidnRunner:
    """Calls ``oidnDenoise`` on HxWx3 HDR images through PFM files in a temp directory."""

    def __init__(
        self,
        oidn: Path,
        albedo: np.ndarray | None = None,
        normal: np.ndarray | None = None,
        quality: str = "high",
        device: str = "cpu",
    ):
        if quality not in QUALITY_CHOICES:
            raise ValueError(f"quality must be one of {QUALITY_CHOICES}, got {quality!r}")
        if normal is not None and albedo is None:
            raise ValueError("OIDN needs albedo whenever normal is given")
        self.oidn, self.quality, self.device = Path(oidn), quality, device
        self._tmp = tempfile.TemporaryDirectory(prefix="opencam_oidn_")
        self.dir = Path(self._tmp.name)
        self.aux: list[str] = []
        for flag, img in (("--alb", albedo), ("--nrm", normal)):
            if img is not None:
                p = self.dir / f"{flag[2:]}.pfm"
                write_pfm(p, img)
                self.aux += [flag, str(p)]

    def __call__(self, img: np.ndarray) -> np.ndarray:
        src, dst = self.dir / "color.pfm", self.dir / "out.pfm"
        write_pfm(src, img)
        # --affinity 0: OIDN's thread pinning raised SIGFPE at device init on some cloud Xeon VMs.
        cmd = [str(self.oidn), "-d", self.device, "--affinity", "0", "--hdr", str(src), *self.aux]
        cmd += ["-q", self.quality, "-o", str(dst)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not dst.is_file():
            raise RuntimeError(f"oidnDenoise failed (exit {r.returncode}): {(r.stderr or r.stdout)[-500:]}")
        out = read_pfm(dst)
        dst.unlink()
        if out.shape != img.shape:
            raise RuntimeError(f"oidnDenoise returned {out.shape}, expected {img.shape}")
        return out

    def close(self) -> None:
        self._tmp.cleanup()


def clamp_fireflies(stack: np.ndarray, max_ratio: float = 20.0) -> tuple[np.ndarray, int]:
    """Clamp extreme isolated spatial outliers before OIDN so 1000x fireflies do not ring or amplify.

    A pixel is clamped only when its plane value exceeds ``max_ratio`` times the second-largest
    value in its 3x3 neighbourhood (so 2-pixel features and lines are never touched, only single-
    pixel Monte Carlo spikes). Returns ``(clamped_stack, n_pixels_clamped)``.
    """
    if max_ratio <= 0 or stack.size == 0:
        return stack, 0
    pad = np.pad(stack, ((1, 1), (1, 1), (0, 0)), mode="edge")
    h, w, _ = stack.shape
    m1 = np.full_like(stack, -np.inf)
    m2 = np.full_like(stack, -np.inf)
    for dy in range(3):
        for dx in range(3):
            v = pad[dy : dy + h, dx : dx + w]
            swap = v > m1
            m2 = np.where(swap, m1, np.maximum(m2, v))
            m1 = np.where(swap, v, m1)
    floor = float(np.percentile(stack, 99)) * 0.1 if stack.size >= 16 else 0.0
    cap = np.maximum(max_ratio * m2, floor)
    over = (m2 > 0) & (stack > cap)
    return np.where(over, cap, stack), int(np.any(over, axis=2).sum())


def denoise_stack(stack: np.ndarray, denoise, max_firefly_ratio: float = 20.0) -> tuple[np.ndarray, float]:
    """Denoise HxWxK planes three at a time; returns ``(denoised, fraction of negative samples clipped)``.

    OIDN treats a triplet as one colour image, so adjacent buckets share edge decisions without being mixed.
    The last triplet is zero-padded.
    """
    stack = np.asarray(stack, dtype=np.float32)
    neg = float((stack < 0).mean()) if stack.size else 0.0
    src, _ = clamp_fireflies(np.maximum(stack, 0.0), max_firefly_ratio)
    out = np.empty_like(src)
    k = src.shape[2]
    for i in range(0, k, 3):
        idx = list(range(i, min(i + 3, k)))
        trip = np.zeros(src.shape[:2] + (3,), dtype=np.float32)
        trip[..., : len(idx)] = src[..., idx]
        out[..., idx] = np.maximum(denoise(trip)[..., : len(idx)], 0.0)
    return out, neg


def denoise_channels(channels: dict[str, np.ndarray], denoise) -> tuple[dict[str, np.ndarray], dict]:
    """Denoise the ``S0.*nm`` buckets (sorted by wavelength) and ``R,G,B`` of a pbrt EXR channel dict."""
    from exr_multispectral import parse_s0_wavelength_nm  # noqa: PLC0415

    out = dict(channels)
    spectral = sorted((lam, n) for n in channels if (lam := parse_s0_wavelength_nm(n)) is not None)
    report: dict = {"n_spectral_buckets": len(spectral)}
    if spectral:
        names = [n for _, n in spectral]
        cube = np.stack([channels[n] for n in names], axis=2)
        den, neg = denoise_stack(cube, denoise)
        for j, n in enumerate(names):
            out[n] = den[..., j]
        before, after = float(np.maximum(cube, 0).sum()), float(den.sum())
        report["spectral_energy_ratio"] = after / before if before > 0 else 1.0
        report["spectral_negative_fraction_clipped"] = neg
    if all(c in channels for c in "RGB"):
        rgb = np.stack([channels[c] for c in "RGB"], axis=2)
        den, neg = denoise_stack(rgb, denoise)
        for j, c in enumerate("RGB"):
            out[c] = den[..., j]
        before = float(np.maximum(rgb, 0).sum())
        report["rgb_energy_ratio"] = float(den.sum()) / before if before > 0 else 1.0
        report["rgb_negative_fraction_clipped"] = neg
    if not spectral and "R" not in channels:
        raise ValueError(f"EXR has neither S0.*nm nor R,G,B channels: {sorted(channels)}")
    return out, report


def noisy_copy_path(exr: Path) -> Path:
    return exr.with_name(f"{exr.stem}_noisy{exr.suffix}")


def denoise_exr(
    exr_in: Path,
    exr_out: Path | None = None,
    *,
    scene: Path | None = None,
    pbrt: Path | None = None,
    guides: str = "albedo_normal",
    guide_spp: int = 16,
    quality: str = "high",
    device: str = "cpu",
    oidn: Path | None = None,
) -> dict:
    """Denoise ``exr_in`` into ``exr_out`` (default ``<stem>_denoised.exr``) and write the sidecar report.

    Denoising in place (``exr_out == exr_in``) first copies the noisy render to ``<stem>_noisy.exr``.
    """
    from exr_multispectral import read_separate_exr_channels, write_separate_channels_exr  # noqa: PLC0415

    if guides not in GUIDE_CHOICES:
        raise ValueError(f"guides must be one of {GUIDE_CHOICES}, got {guides!r}")
    exr_in = Path(exr_in)
    exr_out = Path(exr_out) if exr_out else exr_in.with_name(f"{exr_in.stem}_denoised{exr_in.suffix}")
    binary = find_oidn(oidn)
    channels = read_separate_exr_channels(exr_in)
    h, w = next(iter(channels.values())).shape
    albedo = normal = None
    if guides != "none":
        if scene is None or pbrt is None:
            raise ValueError(f"guides={guides!r} needs the pbrt scene and binary (or use guides='none')")
        albedo, normal = render_guides(Path(scene), Path(pbrt), w, h, guide_spp)
        if guides == "albedo":
            normal = None
    runner = OidnRunner(binary, albedo, normal, quality=quality, device=device)
    try:
        out, report = denoise_channels(channels, runner)
    finally:
        runner.close()
    noisy = None
    if exr_out.resolve() == exr_in.resolve():
        noisy = noisy_copy_path(exr_in)
        shutil.copyfile(exr_in, noisy)
    write_separate_channels_exr(exr_out, out)
    report.update(
        {
            "denoised": True,
            "warning": METRICS_WARNING,
            "denoiser": "Intel Open Image Denoise (oidnDenoise, RT filter)",
            "oidn_binary": str(binary),
            "quality": quality,
            "device": device,
            "guides": guides,
            "guide_spp": guide_spp if guides != "none" else None,
            "exr_in": str(exr_in),
            "exr_out": str(exr_out),
            "noisy_copy": str(noisy) if noisy else None,
        }
    )
    Path(f"{exr_out}.denoise.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exr-in", type=Path, required=True)
    ap.add_argument("--exr-out", type=Path, default=None, help="default: <stem>_denoised.exr; = --exr-in for in place")
    ap.add_argument("--scene", type=Path, default=None, help="pbrt scene of the render (for albedo/normal guides)")
    ap.add_argument("--pbrt", type=Path, default=REPO / "third_party" / "pbrt-v4" / "build" / "pbrt")
    ap.add_argument("--guides", choices=GUIDE_CHOICES, default="albedo_normal")
    ap.add_argument("--guide-spp", type=int, default=16)
    ap.add_argument("--quality", choices=QUALITY_CHOICES, default="high")
    ap.add_argument("--device", default="cpu", help="oidnDenoise -d value (cpu, cuda, sycl, hip, metal, default)")
    ap.add_argument("--oidn", type=Path, default=None, help="oidnDenoise binary (default: auto-detect)")
    a = ap.parse_args(argv)
    report = denoise_exr(
        a.exr_in,
        a.exr_out,
        scene=a.scene,
        pbrt=a.pbrt,
        guides=a.guides,
        guide_spp=a.guide_spp,
        quality=a.quality,
        device=a.device,
        oidn=a.oidn,
    )
    print(json.dumps(report, indent=2))
    print(f"warning: {METRICS_WARNING}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
