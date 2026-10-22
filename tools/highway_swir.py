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

* other analytic surfaces (car and road paint, sign sheeting, rubber, bark, asphalt, concrete,
  galvanised steel): the shape of a USGS splib07a record above 830 nm (``SURFACE_PROXIES``), scaled
  to meet the analytic value at 830 nm. Paint and sheeting have no SWIR record in splib07a, so
  pigmented polymers and painted aluminium of the same colour stand in for them (proxies, not
  measurements of the real materials); soil is still held at its 830 nm value;
* gas absorption along scene paths (``--swir-h2o-gm3``, with ``--haze``): the band optical depth
  of the G173 direct beam above 830 nm (mostly H2O; the O2 1.27 um and CO2 1.6/2.0 um bands come
  with it) per g/m^2 of its 14.2 kg/m^2 water column at air mass 1.5, added to the haze sigma_a.
  Beer scaling of a band average from the long solar path: a lower bound for short paths in
  saturated band centres.
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


#: analytic surface -> highway_materials.USGS_SPECTRA / USGS_SWIR_PROXIES key (shape above JOIN_NM).
SURFACE_PROXIES = {
    **dict.fromkeys(("paint_road_white", "sheeting_white", "carpaint_white"), "white_vinyl"),
    **dict.fromkeys(("paint_road_yellow", "sheeting_yellow"), "yellow_hdpe"),
    **dict.fromkeys(("sheeting_red", "carpaint_red"), "red_vinyl"),
    **dict.fromkeys(("sheeting_blue", "carpaint_blue"), "blue_pipe"),
    **dict.fromkeys(("sheeting_green", "carpaint_darkgreen"), "green_painted_aluminium"),
    **dict.fromkeys(("sheeting_black", "carpaint_black"), "black_hdpe"),
    "rubber": "black_ldpe",
    "carpaint_silver": "brushed_aluminium",
    "carpaint_gray": "grey_hdpe",
    "bark": "weathered_cedar",
    **dict.fromkeys(("asphalt_aged", "asphalt_new"), "asphalt"),
    "concrete": "concrete",
    "galvanized": "galvanized",
}
G173_AIRMASS = 1.5  # ASTM G173: sun at 48.19 deg zenith
G173_WATER_GM2 = 14_200.0  # 1.42 cm precipitable water
#: band-free windows used for the Rayleigh + aerosol continuum under the G173 gas bands [nm]
G173_WINDOWS_NM = (870.0, 1040.0, 1245.0, 1560.0, 1660.0, 2130.0, 2240.0)
H2O_DEFAULT_GM3 = 7.5  # ~10 C at 80 % RH (nominal)


def add_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("SWIR spectra (tools/highway_swir.py)")
    g.add_argument(
        "--swir-spectra",
        choices=("auto", "on", "off"),
        default="auto",
        help="G173 sun/sky and measured-leaf tails above 830 nm (auto: when --spectral-lambda-max > 830).",
    )
    g.add_argument(
        "--swir-h2o-gm3",
        type=float,
        default=H2O_DEFAULT_GM3,
        help="SWIR with --haze: water-vapour density [g/m^3] for the gas-band absorption along scene paths (0: off).",
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


def surface(name: str, wl: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """``visible`` with the USGS proxy's shape above 830 nm (unchanged if ``name`` has no proxy)."""
    if name not in SURFACE_PROXIES:
        return np.asarray(visible, float)
    from highway_materials import read_usgs

    grid = np.arange(350.0, 2500.0 + 1e-9, 1.0)
    shape = np.convolve(read_usgs(SURFACE_PROXIES[name], grid), np.ones(11) / 11, mode="same")
    shape[:5], shape[-5:] = shape[5], shape[-6]
    return np.clip(splice(wl, visible, grid, shape, JOIN_NM), 0.0, 1.0)


@cache
def _gas_tau_slant() -> tuple[np.ndarray, np.ndarray]:
    """G173 gas-band optical depth (slant, air mass 1.5) above 830 nm, 10 nm smoothed."""
    g = read_g173(REPO)
    w = np.arange(300.0, 4000.0 + 1e-9, 1.0)
    n = int(G173_SMOOTH_NM)
    t = [
        np.convolve(np.interp(w, g["wavelength_nm"], g[k]), np.ones(n) / n, mode="same")
        for k in ("direct", "extraterrestrial")
    ]
    tau = -np.log(np.clip(t[0] / np.maximum(t[1], 1e-30), 1e-12, 1.0))
    win = np.asarray(G173_WINDOWS_NM)
    b, a = np.polyfit(np.log(win), np.log(np.interp(win, w, tau)), 1)
    gas = np.clip(tau - np.exp(a) * w**b, 0.0, None)
    gas[w <= JOIN_NM] = 0.0
    return w, gas


def gas_sigma_a(wl: np.ndarray, h2o_gm3: float) -> np.ndarray:
    """Gas-band absorption coefficient [1/m] at water-vapour density ``h2o_gm3`` (0 at <= 830 nm)."""
    w, gas = _gas_tau_slant()
    return np.interp(np.asarray(wl, float), w, gas) * float(h2o_gm3) / (G173_AIRMASS * G173_WATER_GM2)


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
        "sky": f"CIE daylight basis extended with the G173 diffuse (global - direct) shape above {JOIN_NM:g} nm",
        "vegetation": f"USGS splib07a fresh oak leaf ({LEAF_CSV}) above {LEAF_JOIN_NM:g} nm",
        "surfaces": {k: f"USGS splib07a {v} shape above {JOIN_NM:g} nm" for k, v in SURFACE_PROXIES.items()},
        "surface_note": "paint/sheeting proxies are pigmented polymers or painted aluminium, not the real materials",
        "lamps": f"lamp spectra written to {float(args.spectral_lambda_max):g} nm (HPS 1138/1140 nm Na doublet, nominal)",
        **(
            {"path_gas_absorption": {"h2o_gm3": args.swir_h2o_gm3, "model": "G173 band optical depth, Beer scaling"}}
            if getattr(args, "haze", "none") != "none" and args.swir_h2o_gm3 > 0
            else {}
        ),
    }


def approximation() -> str:
    return (
        "SWIR (> 830 nm): G173 sun/sky shapes, a measured-leaf tail and USGS proxy shapes (pigmented "
        "polymers for paint and sheeting) spliced onto the visible models; path gas absorption only with "
        "--haze (G173 band depth, Beer-scaled); lamp SWIR tails are nominal (blackbody halogen, HPS Na lines)."
    )
