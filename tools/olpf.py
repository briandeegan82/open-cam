#!/usr/bin/env python3
"""Optical low-pass (anti-aliasing) filter: birefringent beam splitting in front of the sensor.

A plate of uniaxial crystal cut with its optic axis at ``axis_angle_deg`` to the plate normal splits
every ray into an ordinary and an extraordinary ray of equal power. The extraordinary ray walks off
sideways by ``d = t tan(rho)``, with (Yariv & Yeh, *Optical Waves in Crystals*, sec. 4.3)

    tan(rho) = (1 - n_o^2/n_e^2) tan(theta) / (1 + (n_o^2/n_e^2) tan^2(theta)).

Two crossed plates (with a depolarising quarter-wave plate between them) give the usual four-spot
filter: point spread 1/4 at (+-dx/2, +-dy/2), transfer function cos(pi fx dx) cos(pi fy dy), with its
first zero at fx = 1 / (2 dx). A split of one pixel pitch puts the zero at the pixel Nyquist frequency.

The spot separation is given either directly (``split_x_um`` / ``split_y_um`` or ``split_pixels``,
wavelength independent) or as a quartz plate thickness (``thickness_x_um`` / ``thickness_y_um``),
for which d(lambda) follows the quartz dispersion of Ghosh, Opt. Commun. 163, 95 (1999).

The filter acts on the spectral EXR before the radiance -> electrons conversion, one EXR pixel per
sensor pixel (as in that conversion). The EXR is already integrated over each pixel, so the filter is
applied as its transfer function on the sampled image (exact for integer even splits, band-limited
interpolation otherwise): it suppresses detail below the pixel Nyquist frequency and the CFA colour
aliasing that comes from it, but cannot remove aliasing that pbrt already folded into the pixel
grid above Nyquist. Not modelled: the IR-cut coating and cover-glass of the filter stack (see
``sensor.quantum_efficiency.ircf_csv``), angle-of-incidence dependence of the walk-off, and the
focus shift of the plate (pbrt's film position already includes whatever the lens file assumes).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

# Ghosh (1999) quartz, n^2 - 1 = B0 + B1 l^2 / (l^2 - C1) + B2 l^2 / (l^2 - C2), l in um
# (refractiveindex.info "formula 2", SiO2 Ghosh-o / Ghosh-e, 0.198-2.05 um).
QUARTZ_GHOSH_O = (0.28604141, 1.07044083, 1.00585997e-2, 1.10202242, 100.0)
QUARTZ_GHOSH_E = (0.28851804, 1.09509924, 1.02101864e-2, 1.15662475, 100.0)

DEFAULTS = {
    "enabled": False,
    "split_pixels": None,
    "split_x_um": None,
    "split_y_um": None,
    "thickness_x_um": None,
    "thickness_y_um": None,
    "axis_angle_deg": 45.0,
    "exr_pixel_pitch_um": None,
}


def _ghosh(c: tuple[float, ...], lam_nm) -> np.ndarray:
    l2 = (np.asarray(lam_nm, dtype=np.float64) * 1e-3) ** 2
    return np.sqrt(1.0 + c[0] + c[1] * l2 / (l2 - c[2]) + c[3] * l2 / (l2 - c[4]))


def quartz_indices(lam_nm) -> tuple[np.ndarray, np.ndarray]:
    """(n_o, n_e) of crystalline quartz at ``lam_nm``."""
    return _ghosh(QUARTZ_GHOSH_O, lam_nm), _ghosh(QUARTZ_GHOSH_E, lam_nm)


def walkoff_tan(lam_nm, axis_angle_deg: float = 45.0) -> np.ndarray:
    """tan(rho) of the extraordinary ray in quartz with the optic axis at ``axis_angle_deg`` to the normal."""
    n_o, n_e = quartz_indices(lam_nm)
    r = (n_o / n_e) ** 2
    t = math.tan(math.radians(axis_angle_deg))
    return (1.0 - r) * t / (1.0 + r * t * t)


def resolve_config(sensor_cfg: dict) -> dict:
    cfg = {**DEFAULTS, **((sensor_cfg or {}).get("olpf") or {})}
    pitch = cfg["exr_pixel_pitch_um"] or (sensor_cfg or {}).get("pixel_pitch_um")
    cfg["pixel_pitch_um"] = None if pitch is None else float(pitch)
    return cfg


def split_px(cfg: dict, axis: str, lam_nm) -> np.ndarray:
    """Spot separation along ``axis`` ("x" or "y") in EXR pixels at each wavelength (0: no split)."""
    lam = np.atleast_1d(np.asarray(lam_nm, dtype=np.float64))
    keys = [k for k in (f"split_{axis}_um", f"thickness_{axis}_um") if cfg.get(k) is not None]
    if cfg.get("split_pixels") is not None:
        if keys:
            raise ValueError(f"olpf: give split_pixels or {' / '.join(keys)}, not both")
        return np.full(lam.shape, float(cfg["split_pixels"]))
    if len(keys) > 1:
        raise ValueError(f"olpf: give split_{axis}_um or thickness_{axis}_um, not both")
    if not keys:
        return np.zeros(lam.shape)
    if cfg.get("pixel_pitch_um") is None:
        raise KeyError("olpf: split in um needs sensor.pixel_pitch_um (or olpf.exr_pixel_pitch_um)")
    if keys[0].startswith("split"):
        d_um = np.full(lam.shape, float(cfg[keys[0]]))
    else:
        d_um = float(cfg[keys[0]]) * walkoff_tan(lam, float(cfg["axis_angle_deg"]))
    return d_um / float(cfg["pixel_pitch_um"])


def transfer(fx, fy, dx_px: float, dy_px: float) -> np.ndarray:
    """OTF of the four-spot filter at frequencies in cycles / pixel (real, may be negative)."""
    return np.cos(np.pi * np.asarray(fx) * dx_px) * np.cos(np.pi * np.asarray(fy) * dy_px)


def apply_plane(img: np.ndarray, dx_px: float, dy_px: float) -> np.ndarray:
    """Filter one image plane: multiply its spectrum by ``transfer`` (reflect-padded, energy conserving)."""
    if dx_px == 0 and dy_px == 0:
        return img
    a = np.asarray(img, dtype=np.float64)
    pad = int(math.ceil(max(abs(dx_px), abs(dy_px)))) + 4
    p = np.pad(a, pad, mode="reflect")
    fy = np.fft.fftfreq(p.shape[0])[:, None]
    fx = np.fft.rfftfreq(p.shape[1])[None, :]
    out = np.fft.irfft2(np.fft.rfft2(p) * transfer(fx, fy, dx_px, dy_px), s=p.shape)
    return out[pad:-pad, pad:-pad].astype(img.dtype, copy=False)


def apply_channels(chans: dict[str, np.ndarray], sensor_cfg: dict) -> tuple[dict[str, np.ndarray], dict]:
    """Apply ``sensor.olpf`` to every image channel (S0.<lambda>nm at its wavelength, others at 550 nm).

    Negative undershoot of the band-limited interpolation at hard edges (a few % for non-integer splits)
    is clipped to zero; the clipped energy is reported.
    """
    from exr_multispectral import parse_s0_wavelength_nm

    cfg = resolve_config(sensor_cfg)
    if not cfg["enabled"]:
        return chans, {"enabled": False}
    out, splits, clipped, total = {}, {}, 0.0, 0.0
    for name, arr in chans.items():
        if arr.ndim != 2 or name.upper() in ("A", "ALPHA") or name.startswith(("P.", "N.", "Albedo")):
            out[name] = arr
            continue
        lam = parse_s0_wavelength_nm(name) or 550.0
        dx, dy = float(split_px(cfg, "x", lam)[0]), float(split_px(cfg, "y", lam)[0])
        f = apply_plane(arr, dx, dy)
        neg = f < 0
        clipped += float(-f[neg].sum())
        total += float(np.abs(arr).sum())
        out[name] = np.where(neg, 0, f).astype(arr.dtype, copy=False)
        splits[name] = (round(dx, 4), round(dy, 4))
    vals = list(splits.values())
    info = {
        "enabled": True,
        "split_px_range": [
            min(v[0] for v in vals),
            max(v[0] for v in vals),
            min(v[1] for v in vals),
            max(v[1] for v in vals),
        ]
        if vals
        else None,
        "clipped_energy_fraction": clipped / total if total else 0.0,
    }
    return out, info


def apply_cube(cube: np.ndarray, lam_nm, sensor_cfg: dict) -> tuple[np.ndarray, dict]:
    """``apply_channels`` for an H x W x K spectral cube with bucket wavelengths ``lam_nm``."""
    lam = np.asarray(lam_nm, dtype=np.float64).ravel()
    names = [f"S0.{x:g}nm" for x in lam]
    if not resolve_config(sensor_cfg)["enabled"]:
        return cube, {"enabled": False}
    out, info = apply_channels({n: cube[:, :, k] for k, n in enumerate(names)}, sensor_cfg)
    return np.stack([out[n] for n in names], axis=2), info


def main() -> None:
    from camera_model import load_camera_model
    from exr_multispectral import read_separate_exr_channels, write_separate_channels_exr

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo-root", type=Path, default=TOOLS.parent)
    ap.add_argument("--camera-model-config", type=Path, required=True)
    ap.add_argument("--exr-in", type=Path, required=True)
    ap.add_argument("--exr-out", type=Path, default=None, help="default: overwrite --exr-in")
    args = ap.parse_args()
    sensor = load_camera_model(args.camera_model_config.resolve()).get("sensor") or {}
    if not resolve_config(sensor)["enabled"]:
        print("sensor.olpf.enabled is false; nothing to do.", file=sys.stderr)
        return
    exr_in = args.exr_in if args.exr_in.is_absolute() else (args.repo_root / args.exr_in).resolve()
    out, info = apply_channels(read_separate_exr_channels(exr_in), sensor)
    exr_out = (args.exr_out or exr_in).resolve()
    exr_out.parent.mkdir(parents=True, exist_ok=True)
    write_separate_channels_exr(exr_out, out)
    print(f"wrote {exr_out} (olpf split px x/y range {info['split_px_range']})")


if __name__ == "__main__":
    main()
