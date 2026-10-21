"""SWIR (> 830 nm) spectra for the highway scene.

``--swir-spectra`` (default ``auto``: on when ``--spectral-lambda-max`` > 830 nm) replaces the parts
of the day scene that the visible models do not define above 830 nm. Each tail is spliced onto the
visible spectrum at the join wavelength, so everything at or below it is unchanged:

* sun: ASTM G173-03 direct-normal shape (SMARTS, AM1.5; includes the 0.94/1.13/1.38/1.87 um
  water-vapour bands) instead of the analytic Rayleigh/aerosol model, joined at 830 nm;
* sky: the CIE daylight basis (defined to 830 nm) is extended with the G173 diffuse
  (global - direct) shape, so the sky stays dark-but-nonzero in the SWIR;
* vegetation: grass and leaf reflectance/transmittance follow the measured USGS splib07a fresh
  oak leaf above 800 nm (1.45/1.94 um liquid-water absorption).

Other analytic surfaces (car and road paint, sign sheeting, ...) are held at their 830 nm value.
``--spectral-library usgs`` gives measured asphalt/concrete/galvanised/plastic to 2500 nm.
``--film-lambda-min/max`` (see build_highway_scene.add_film_args) set the rendered band, e.g. an
ideal 900-1600 nm passband, independently of the spectra grid, which must keep the visible range
for the photometric light normalisation.
"""

from __future__ import annotations

import argparse
from functools import cache

import numpy as np
from swir_spectra import REPO, read_g173

JOIN_NM = 830.0
LEAF_JOIN_NM = 800.0
LEAF_CSV = "spectra/usgs_nir/01_oak_leaf_fresh.csv"
VEGETATION = ("grass", "leaf_reflectance", "leaf_transmittance")
G173_SMOOTH_NM = 10.0


def add_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("SWIR spectra (tools/highway_swir.py)")
    g.add_argument(
        "--swir-spectra",
        choices=("auto", "on", "off"),
        default="auto",
        help="G173 sun/sky and measured-leaf tails above 830 nm (auto: when --spectral-lambda-max > 830).",
    )


def enabled(args: argparse.Namespace) -> bool:
    mode = getattr(args, "swir_spectra", "off")
    return mode == "on" or (mode == "auto" and float(args.spectral_lambda_max) > JOIN_NM)


@cache
def _g173_smoothed(kind: str) -> tuple[np.ndarray, np.ndarray]:
    """G173 ``direct`` or ``diffuse`` (global - direct) on a 1 nm grid, box-smoothed to 10 nm."""
    g = read_g173(REPO)
    w = np.arange(300.0, 4000.0 + 1e-9, 1.0)
    direct = np.interp(w, g["wavelength_nm"], g["direct"])
    v = direct if kind == "direct" else np.interp(w, g["wavelength_nm"], g["global"]) - direct
    n = int(G173_SMOOTH_NM)
    v = np.convolve(v, np.ones(n) / n, mode="same")
    return w, np.clip(v, 0.0, None)


@cache
def _leaf() -> tuple[np.ndarray, np.ndarray]:
    d = np.loadtxt(REPO / LEAF_CSV, delimiter=",", comments="#")
    return d[:, 0], d[:, 1]


def splice(wl: np.ndarray, v: np.ndarray, shape_wl: np.ndarray, shape_v: np.ndarray, join_nm: float) -> np.ndarray:
    """``v`` at and below ``join_nm``; above it, ``shape`` scaled to meet ``v`` at the join."""
    wl, out = np.asarray(wl, float), np.array(v, float)
    hi = wl > join_nm
    if hi.any():
        k = np.interp(join_nm, wl, out) / max(float(np.interp(join_nm, shape_wl, shape_v)), 1e-30)
        out[hi] = k * np.interp(wl[hi], shape_wl, shape_v)
    return out


def sun_spectrum(wl: np.ndarray, visible: np.ndarray) -> np.ndarray:
    return splice(wl, visible, *_g173_smoothed("direct"), JOIN_NM)


def vegetation(name: str, wl: np.ndarray, visible: np.ndarray) -> np.ndarray:
    return splice(wl, visible, *_leaf(), LEAF_JOIN_NM)


def extend_sky_basis(wl: np.ndarray, basis: np.ndarray, lambda_max: float, step_nm: float = 5.0):
    """Extend daylight basis spectra (K, n) past ``wl[-1]`` with the G173 diffuse shape."""
    extra = np.arange(wl[-1] + step_nm, lambda_max + 1e-9, step_nm)
    if extra.size == 0:
        return wl, basis
    sw, sv = _g173_smoothed("diffuse")
    tail = np.interp(extra, sw, sv) / np.interp(wl[-1], sw, sv)
    return np.concatenate([wl, extra]), np.concatenate([basis, basis[:, -1:] * tail[None, :]], axis=1)


def manifest(args: argparse.Namespace) -> dict:
    return {
        "lambda_max_nm": float(args.spectral_lambda_max),
        "sun": f"ASTM G173-03 direct shape above {JOIN_NM:g} nm (pvlib copy, BSD-3)",
        "sky": (
            f"CIE daylight basis extended with the G173 diffuse (global - direct) shape above {JOIN_NM:g} nm"
            if getattr(args, "sky", None) != "prague"
            else "Prague sky model SWIR channels: one least-squares tail per daylight basis spectrum "
            "(lighting.sky.spectral.swir_tails; G173 diffuse if the dataset stops at 830 nm)"
        ),
        "vegetation": f"USGS splib07a fresh oak leaf ({LEAF_CSV}) above {LEAF_JOIN_NM:g} nm",
        "other_surfaces": "analytic spectra held at their 830 nm value (not SWIR measurements)",
    }


def approximation() -> str:
    return (
        "SWIR (> 830 nm): G173 sun/sky shapes and a measured-leaf tail spliced onto the visible models; "
        "car/road paint and sign sheeting are analytic and flat above 830 nm; no path water-vapour "
        "absorption between scene and camera; night lamps are not SWIR-calibrated."
    )
