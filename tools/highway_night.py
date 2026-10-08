#!/usr/bin/env python3
"""Dusk / night lighting for the highway scene: twilight and night sky, moon, vehicle lamps,
street lights and retroreflective markings/signs (all in absolute photometric units).

Used by tools/build_highway_scene.py through ``add_night_args`` and the ``*_lines`` helpers;
``--time-of-day day`` (the default) leaves the daytime scene untouched.

Units follow the builder's convention: every pbrt light is normalised photometrically
(pbrt divides a light spectrum by its luminance), so ``float illuminance`` is in lux,
area-light ``scale`` is luminance in cd/m^2 and a goniometric map with ``scale 1`` holds
luminous intensity in cd. The manifest's EXR conversion (683 * CIE_Y_integral) therefore
stays valid for every new source.

Sources (see docs/HIGHWAY_SCENES.md, "Night and dusk"):

* Natural illuminance: 400 lx at sunset, 3.4 lx at the end of civil twilight (sun -6 deg),
  ~0.008 lx at the end of nautical twilight (-12 deg), 0.002 lx moonless clear night sky
  with airglow (commonly cited values, e.g. Bond & Henderson 1963 "The Conquest of
  Darkness"; US Naval Observatory twilight definitions; Schlyter "Radiometry and photometry
  in astronomy"). Moon: Krisciunas & Schaefer 1991, PASP 103:1033, eq. 8-9.
* Night-sky spectrum: Leinert et al. 1998, A&AS 127:1 (airglow lines OI 557.7/630.0 nm,
  Na 589 nm, OH Meinel bands in the red/NIR on a solar-like continuum).
* Lamp spectra: halogen = 3200 K blackbody (H7/H11 class); white LED = InGaN blue pump
  (~450 nm, ~20 nm FWHM) + Ce:YAG (and for warm white a nitride red) phosphor, tuned to the
  target CCT; high-pressure sodium = self-reversed Na D emission 550-650 nm with the
  568.8/615.4/498.3/466.5/819.5 nm lines (de Groot & van Vliet 1986 "The High-Pressure
  Sodium Lamp"); red signal LED = AlInGaP, peak ~626 nm, ~17 nm FWHM.
* Low beam: UN ECE R112 / R149 class-B (passing beam, right-hand traffic) photometric
  test points; tail/stop lamp intensities from UN ECE R7 (R148).
* Street lights: full-cutoff IES Type III medium distribution (ANSI/IES RP-8), 12 m
  twin-arm median poles; 4000 K LED or high-pressure sodium; levels checked against
  EN 13201-2 motorway classes.
* Retroreflection: coefficient of retroreflection R_A (ASTM E808/E810; sheeting minima
  ASTM D4956 Type III / EN 12899-1 class RA2) and coefficient of retroreflected luminance
  R_L (EN 1436, 30 m geometry); see ``retro_ra`` for the model implemented by the pbrt
  patch third_party/patches/0001-retroreflective-material.patch.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
from colour_science import correlated_colour_temperature, tristimulus, xy_chromaticity
from highway_sky import (
    LUMA,
    analytic_clear_sky,
    clear_sky_illuminance_lux,
    equal_area_directions,
    kasten_young_airmass,
    solar_direct_spectrum,
    write_rgb_exr,
)
from highway_spectra import reflectance

TIMES = ("day", "dusk", "night")
LIGHT_WL = np.arange(360.0, 831.0, 1.0)  # emission spectra are written at 1 nm (sharp lines)

# --------------------------------------------------------------------------------------
# natural light
# --------------------------------------------------------------------------------------
# (sun elevation [deg], clear-sky horizontal illuminance without moon [lx]).
TWILIGHT_ANCHORS = ((-0.833, 400.0), (-6.0, 3.4), (-12.0, 0.008), (-18.0, 0.002))
NIGHT_SKY_LUX = 0.002
LUX_PER_FOOTCANDLE = 10.7639
MOON_EXTINCTION_MAG = 0.172  # V-band extinction per air mass used by Krisciunas & Schaefer


def twilight_illuminance_lux(sun_elevation_deg: float) -> float:
    """Horizontal illuminance from the clear twilight/night sky (no moon), log-interpolated."""
    h = float(sun_elevation_deg)
    el = np.array([a[0] for a in TWILIGHT_ANCHORS])[::-1]
    lg = np.log(np.array([a[1] for a in TWILIGHT_ANCHORS]))[::-1]
    if h <= el[0]:
        return NIGHT_SKY_LUX
    if h >= el[-1]:  # sun still just above -0.833 deg: extrapolate the first segment
        slope = (lg[-1] - lg[-2]) / (el[-1] - el[-2])
        return float(math.exp(lg[-1] + slope * (min(h, 0.0) - el[-1])))
    return float(math.exp(np.interp(h, el, lg)))


def moon_illuminance_lux(phase_angle_deg: float, elevation_deg: float) -> float:
    """Direct moonlight on a surface normal to the moon [lx] (Krisciunas & Schaefer 1991 eq. 8)."""
    if elevation_deg <= 0.0:
        return 0.0
    a = abs(float(phase_angle_deg))
    i_star_fc = 10.0 ** (-0.4 * (3.84 + 0.026 * a + 4e-9 * a**4))
    transmission = 10.0 ** (-0.4 * MOON_EXTINCTION_MAG * kasten_young_airmass(elevation_deg))
    return float(i_star_fc * LUX_PER_FOOTCANDLE * transmission)


def planck(wl_nm: np.ndarray, t_k: float) -> np.ndarray:
    wl = np.asarray(wl_nm, dtype=np.float64) * 1e-9
    v = 1.0 / wl**5 / np.expm1(1.4387769e-2 / (wl * t_k))
    return v / np.interp(560e-9, wl, v)


def _g(wl, c, s):
    return np.exp(-0.5 * ((wl - c) / s) ** 2)


def night_sky_spectrum(wl: np.ndarray) -> np.ndarray:
    """Moonless night sky radiance: scattered starlight/zodiacal continuum + airglow (Leinert 1998)."""
    wl = np.asarray(wl, dtype=np.float64)
    cont = planck(wl, 5000.0)
    lines = 1.6 * _g(wl, 557.7, 1.2) + 0.6 * _g(wl, 589.3, 1.2) + 0.9 * _g(wl, 630.0, 1.2) + 0.3 * _g(wl, 636.4, 1.2)
    oh = 1.8 * np.clip((wl - 690.0) / 140.0, 0.0, None) ** 1.5 * (1.0 + 0.4 * np.sin(wl / 9.0))
    return cont + lines + oh


def moon_spectrum(wl: np.ndarray, elevation_deg: float) -> np.ndarray:
    """Moonlight at the ground: solar SPD x lunar reflectance (redder than the sun, Lane & Irvine 1973)."""
    wl = np.asarray(wl, dtype=np.float64)
    return solar_direct_spectrum(wl, max(elevation_deg, 1.0)) * (wl / 560.0) ** 1.2


def twilight_sky_map(n: int, sun_elevation_deg: float) -> np.ndarray:
    """Analytic clear twilight sky on an equal-area map (sun below the horizon at azimuth 0).

    Relative luminance: zenith 1, a horizon glow concentrated towards the sun's azimuth
    that fades as the sun sinks; colour goes from orange near the solar horizon through the
    pinkish anti-twilight arch to the Chappuis-ozone blue zenith. The absolute level is set by
    the light's ``illuminance`` (``twilight_illuminance_lux``).
    """
    d = equal_area_directions(n)
    el = np.arcsin(np.clip(d[..., 2], -1.0, 1.0))
    psi = np.abs(np.arctan2(d[..., 1], d[..., 0]))  # azimuth from the sun
    depth = float(np.clip(-sun_elevation_deg / 12.0, 0.0, 1.0))
    glow = (6.0 * (1.0 - depth) + 0.5) * np.exp(-psi / 0.9) * np.exp(-np.maximum(el, 0.0) / 0.22)
    belt = 0.6 * np.exp(-np.maximum(el, 0.0) / 0.12)
    lum = 1.0 + glow + belt
    orange, pink, blue = np.array([1.0, 0.5, 0.18]), np.array([0.8, 0.62, 0.72]), np.array([0.22, 0.36, 0.9])
    wg = glow / lum
    wb = belt / lum
    rgb = wg[..., None] * orange + wb[..., None] * pink + (1.0 - wg - wb)[..., None] * blue
    rgb = rgb / (rgb @ LUMA)[..., None]
    out = rgb * lum[..., None]
    out[d[..., 2] <= 0.0] = 0.0
    return out


# --------------------------------------------------------------------------------------
# lamp spectra
# --------------------------------------------------------------------------------------
def cct(wl: np.ndarray, spd: np.ndarray) -> float:
    return correlated_colour_temperature(xy_chromaticity(np.array(tristimulus(wl, spd)))[0])


def pc_led(wl: np.ndarray, cct_k: float, red_phosphor: float = 0.0) -> np.ndarray:
    """Phosphor-converted white LED with its blue/phosphor ratio solved for ``cct_k``."""
    wl = np.asarray(wl, dtype=np.float64)
    blue = np.exp(-0.5 * ((wl - 450.0) / 9.5) ** 2)
    phos = np.exp(-0.5 * ((wl - 558.0) / 48.0) ** 2) + red_phosphor * np.exp(-0.5 * ((wl - 625.0) / 38.0) ** 2)
    lo, hi = 0.01, 10.0
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if cct(wl, mid * blue + phos) < cct_k:
            lo = mid
        else:
            hi = mid
    v = math.sqrt(lo * hi) * blue + phos
    return v / v.max()


def hps_spectrum(wl: np.ndarray) -> np.ndarray:
    """High-pressure sodium: pressure-broadened, self-reversed Na D band plus Na lines."""
    wl = np.asarray(wl, dtype=np.float64)
    band = (1.4 * _g(wl, 584.0, 11.0) + 1.1 * _g(wl, 598.0, 9.0) + 0.35 * _g(wl, 620.0, 22.0)) * (
        1.0 - 0.93 * _g(wl, 589.3, 2.6)
    )
    lines = (
        0.45 * _g(wl, 568.8, 1.6)
        + 0.30 * _g(wl, 615.8, 1.8)
        + 2.0 * _g(wl, 498.3, 1.4)
        + 1.2 * _g(wl, 466.5, 1.4)
        + 0.8 * _g(wl, 515.0, 1.4)
        + 0.70 * _g(wl, 819.5, 2.5)
    )
    tail = 0.03 * np.clip((wl - 400.0) / 300.0, 0.0, 1.0)
    v = band + lines + tail
    return v / v.max()


def red_led_spectrum(wl: np.ndarray, peak_nm: float = 626.0, fwhm_nm: float = 17.0) -> np.ndarray:
    wl = np.asarray(wl, dtype=np.float64)
    s = fwhm_nm / 2.3548
    s_side = np.where(wl < peak_nm, 0.85 * s, 1.15 * s)  # slight red-side tail of AlInGaP
    return np.exp(-0.5 * ((wl - peak_nm) / s_side) ** 2)


LAMP_SPECTRA = {
    "halogen": lambda wl: planck(wl, 3200.0),
    "led_headlamp": lambda wl: pc_led(wl, 5700.0),
    "led_4000k": lambda wl: pc_led(wl, 4000.0, red_phosphor=0.45),
    "hps": hps_spectrum,
    "red_led": red_led_spectrum,
    "night_sky": night_sky_spectrum,
}


# --------------------------------------------------------------------------------------
# luminous intensity distributions
# --------------------------------------------------------------------------------------
# ECE R112 class B passing beam (RHT) test points: (name, h [deg, + right], v [deg, + up],
# min lux, max lux) at 25 m; intensity = E * 625 cd.
R112_POINTS = (
    ("B50L", -3.43, 0.57, None, 0.4),
    ("75R", 1.15, -0.57, 12.0, None),
    ("75L", -3.43, -0.57, None, 12.0),
    ("50L", -3.43, -0.86, None, 15.0),
    ("50R", 1.72, -0.86, 12.0, None),
    ("50V", 0.0, -0.86, 6.0, None),
    ("25L", -9.0, -1.72, 2.0, None),
    ("25R", 9.0, -1.72, 2.0, None),
)
R112_ZONE_III = ((-8.0, 1.0), (-4.0, 1.0), (0.0, 1.0), (4.0, 1.0), (8.0, 1.0), (-8.0, 4.0), (0.0, 4.0), (8.0, 4.0))
R112_ZONE_III_MAX_LUX = 0.7
HEADLAMP_PEAK_CD = {"halogen": 20_000.0, "led": 28_000.0}


def low_beam_cd(h_deg: np.ndarray, v_deg: np.ndarray, peak_cd: float = 20_000.0) -> np.ndarray:
    """Analytic RHT low beam: flat cut-off at 0.57D left of V-V, 15-deg rising shoulder right.

    A hot spot just right of and below the elbow, a wide foreground spread, and a small
    stray-light floor above the cut-off. Shaped to satisfy the R112 class-B points above.
    """
    h = np.asarray(h_deg, dtype=np.float64)
    v = np.asarray(v_deg, dtype=np.float64)
    vc = np.where(h <= 0.0, -0.57, np.minimum(-0.57 + math.tan(math.radians(15.0)) * h, 0.5))
    below = 0.5 * (1.0 + np.tanh((vc - v) / 0.24))  # cut-off gradient ~0.1-0.2 deg
    hot = np.exp(-(((h - 1.5) / 4.0) ** 2) - ((v + 1.2) / 1.3) ** 2)
    spread = 0.15 * np.exp(-((h / 20.0) ** 2)) * np.exp(-(((np.minimum(v, -2.0) + 2.0) / 5.0) ** 2))
    beam = peak_cd * (hot + spread) * below
    floor = 60.0 * np.exp(-((h / 30.0) ** 2)) * np.exp(-((v / 12.0) ** 2))
    housing = np.clip((v + 70.0) / 30.0, 0.0, 1.0) * np.clip((95.0 - np.abs(h)) / 30.0, 0.0, 1.0)
    return (beam + floor) * housing


def beam_angles(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(h, v) [deg] of goniometric image-frame directions (x right, y forward, z up)."""
    h = np.degrees(np.arctan2(d[..., 0], d[..., 1]))
    v = np.degrees(np.arcsin(np.clip(d[..., 2], -1.0, 1.0)))
    return h, v


def flux_lm(intensity_map: np.ndarray) -> float:
    """Luminous flux of an equal-area intensity map [cd -> lm] (pixels subtend 4 pi / N)."""
    return float(4.0 * math.pi * intensity_map.mean())


def headlamp_map(n: int, kind: str) -> np.ndarray:
    h, v = beam_angles(equal_area_directions(n))
    return low_beam_cd(h, v, HEADLAMP_PEAK_CD[kind])


# Street luminaire: full-cutoff IES Type III medium. Image frame: x = street side (towards the
# carriageway), y = along the road, z = up. Peak candela ~68 deg from nadir, 25 deg off the
# road axis towards the street side; no light at or above the horizontal.
STREETLIGHT = {
    "led_4000k": {"flux_lm": 16_000.0, "spectrum": "led_4000k"},
    "hps": {"flux_lm": 22_000.0, "spectrum": "hps"},  # 250 W HPS (27.5 klm lamp) x ~0.8 LOR
}
POLE_HEIGHT_M, ARM_OUTREACH_M, POLE_SPACING_M = 12.0, 2.5, 48.0
LENS_FLUX_FRACTION = 0.08  # flux carried by the visible Lambertian lens emitter


def type3_shape(d: np.ndarray) -> np.ndarray:
    gamma = np.degrees(np.arccos(np.clip(-d[..., 2], -1.0, 1.0)))  # angle from nadir
    psi = np.degrees(np.arctan2(d[..., 1], d[..., 0]))  # 0 = street side, +-90 = along road
    lateral = np.exp(-(((np.abs(psi) - 65.0) / 32.0) ** 2))
    beam = np.exp(-(((gamma - 68.0) / 13.0) ** 2)) * lateral
    fill = 0.22 * np.cos(np.radians(np.minimum(gamma, 90.0))) ** 1.5 * (0.6 + 0.4 * np.cos(np.radians(psi)))
    cutoff = np.clip((89.0 - gamma) / 9.0, 0.0, 1.0) ** 2
    return (beam + fill) * cutoff


def streetlight_map(n: int, flux: float) -> np.ndarray:
    m = type3_shape(equal_area_directions(n))
    return m * (flux / flux_lm(m))


def write_y_exr(path: Path, y: np.ndarray) -> None:
    import OpenEXR

    path.parent.mkdir(parents=True, exist_ok=True)
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    with OpenEXR.File(header, {"Y": np.ascontiguousarray(y, dtype=np.float32)}) as f:
        f.write(str(path))


def equal_area_sphere_to_square(d: np.ndarray) -> np.ndarray:
    """Vectorised port of pbrt-v4 ``EqualAreaSphereToSquare``: directions (..., 3) -> uv in [0, 1]."""
    x, y, z = np.abs(d[..., 0]), np.abs(d[..., 1]), np.abs(d[..., 2])
    r = np.sqrt(np.clip(1.0 - z, 0.0, None))
    a, b = np.maximum(x, y), np.minimum(x, y)
    phi = np.arctan(np.where(a == 0, 0.0, b / np.where(a == 0, 1.0, a))) * 2.0 / math.pi
    phi = np.where(x < y, 1.0 - phi, phi)
    v = phi * r
    u = r - v
    neg = d[..., 2] < 0
    u, v = np.where(neg, 1.0 - v, u), np.where(neg, 1.0 - u, v)
    u = np.copysign(u, d[..., 0])
    v = np.copysign(v, d[..., 1])
    return np.stack([0.5 * (u + 1.0), 0.5 * (v + 1.0)], -1)


def map_intensity(m: np.ndarray, d_img: np.ndarray) -> np.ndarray:
    """Nearest-pixel lookup of an equal-area map along image-frame directions (as pbrt)."""
    n = m.shape[0]
    uv = equal_area_sphere_to_square(d_img)
    col = np.clip((uv[..., 0] * n).astype(int), 0, n - 1)
    row = np.clip((uv[..., 1] * n).astype(int), 0, n - 1)
    return m[row, col]


def streetlight_positions(median_x: float, z0: float, z1: float) -> list[tuple[float, float, int]]:
    """(x, z, street-side sign) of every luminaire: twin arms on median poles."""
    out = []
    for z in np.arange(z0, z1, POLE_SPACING_M):
        out += [(median_x + ARM_OUTREACH_M, float(z), 1), (median_x - ARM_OUTREACH_M, float(z), -1)]
    return out


def road_illuminance(
    flux: float, median_x: float, x_range: tuple[float, float], z_range: tuple[float, float], n: int = 256
) -> dict:
    """Horizontal illuminance from the street lights on a carriageway grid (EN 13201-3 style)."""
    m = streetlight_map(n, flux * (1.0 - LENS_FLUX_FRACTION))
    xs = np.linspace(*x_range, 12)
    zs = np.linspace(*z_range, 20)
    X, Z = np.meshgrid(xs, zs)
    E = np.zeros_like(X)
    for lx, lz, side in streetlight_positions(median_x, z_range[0] - 240.0, z_range[1] + 240.0):
        dx, dz, dy = X - lx, Z - lz, -POLE_HEIGHT_M
        r2 = dx**2 + dz**2 + dy**2
        r = np.sqrt(r2)
        d_img = np.stack([side * dx / r, side * dz / r, dy / r], -1)
        E += map_intensity(m, d_img) * (POLE_HEIGHT_M / r) / r2
        # Lens emitter: Lambertian downward disc carrying LENS_FLUX_FRACTION of the flux.
        E += (LENS_FLUX_FRACTION * flux / math.pi) * (POLE_HEIGHT_M / r) ** 2 / r2
    return {"e_avg_lux": float(E.mean()), "e_min_lux": float(E.min()), "uniformity_u0": float(E.min() / E.mean())}


# --------------------------------------------------------------------------------------
# retroreflection (Python mirror of RetroreflectiveBxDF in the pbrt patch)
# --------------------------------------------------------------------------------------
# R_A(alpha, beta) = ra exp(-(alpha - alpha0)/w) (cos_i cos_o)^(q/2); BRDF f = R_A/(cos_i cos_o).
# ASTM D4956 Type III minima [cd/lx/m^2] at (alpha, beta): (0.2,-4), (0.2,+30), (0.5,-4), (0.5,+30).
D4956_TYPE_III = {
    "white": (250.0, 150.0, 95.0, 65.0),
    "yellow": (170.0, 100.0, 62.0, 45.0),
    "green": (45.0, 25.0, 21.0, 15.0),
    "red": (45.0, 25.0, 15.0, 10.0),
}
D4956_GEOMETRIES = ((0.2, -4.0), (0.2, 30.0), (0.5, -4.0), (0.5, 30.0))
SHEETING_MARGIN = 1.2  # new sheeting exceeds the minima; scale applied at (0.2, -4)
SHEETING = {"alpha0_deg": 0.2, "width_deg": 0.34, "q": 2.7}
# EN 1436 30 m geometry: viewing elevation 2.29 deg, illumination elevation 1.24 deg
# (observation angle 1.05 deg). R_L [mcd/m^2/lx]: white class R5 (>= 300), yellow R4 (>= 200).
EN1436 = {"view_elev_deg": 2.29, "illum_elev_deg": 1.24}
MARKING_RL = {"white": 300.0, "yellow": 200.0}
MARKING = {"alpha0_deg": 1.05, "width_deg": 1.6, "q": 1.0}


def retro_ra(alpha_rad, ci, co, ra: float, alpha0_deg: float, width_deg: float, q: float) -> np.ndarray:
    """Coefficient of retroreflection of the model [cd/lx/m^2] (no energy clamp)."""
    a0, w = math.radians(alpha0_deg), math.radians(width_deg)
    return ra * np.exp(-(np.asarray(alpha_rad) - a0) / w) * (np.asarray(ci) * np.asarray(co)) ** (q / 2.0)


def sheeting_geometry(alpha_deg: float, beta_deg: float) -> tuple[float, float, float]:
    """(alpha [rad], cos_i, cos_o) for an ASTM E810 geometry, observer displaced in the entrance plane."""
    b, a = math.radians(abs(beta_deg)), math.radians(alpha_deg)
    return a, math.cos(b), math.cos(b + a)


def sheeting_ra_param(colour: str) -> float:
    """Model ``ra`` (normal entrance, alpha0) giving SHEETING_MARGIN x the (0.2, -4) minimum."""
    a, ci, co = sheeting_geometry(0.2, -4.0)
    target = SHEETING_MARGIN * D4956_TYPE_III[colour][0]
    return target / float(retro_ra(a, ci, co, 1.0, **_kw(SHEETING)))


def marking_geometry() -> tuple[float, float, float]:
    ev, ei = math.radians(EN1436["view_elev_deg"]), math.radians(EN1436["illum_elev_deg"])
    return ev - ei, math.sin(ei), math.sin(ev)


def marking_ra_param(rl_mcd: float) -> float:
    """Model ``ra`` reproducing R_L = L / E_perp = R_A / cos_o at the EN 1436 geometry."""
    a, ci, co = marking_geometry()
    return 1e-3 * rl_mcd * co / float(retro_ra(a, ci, co, 1.0, **_kw(MARKING)))


def _kw(p: dict) -> dict:
    return {"alpha0_deg": p["alpha0_deg"], "width_deg": p["width_deg"], "q": p["q"]}


def _photopic_mean_under_a(wl: np.ndarray, s: np.ndarray) -> float:
    """Luminous-weighted mean of a spectral factor under CIE illuminant A (ASTM E810 source)."""
    a = planck(wl, 2856.0)
    return tristimulus(wl, s * a)[1] / tristimulus(wl, a)[1]


def retro_ra_spectrum(wl: np.ndarray, surface: str, value: float) -> np.ndarray:
    """Spectral ``ra`` with the colour of ``surface`` and photometric value ``value`` (illuminant A)."""
    s = reflectance(surface, wl)
    return value * s / _photopic_mean_under_a(wl, s)


RETRO_MATERIALS = {
    # name in the builder -> (diffuse reflectance surface, retro colour, kind)
    "paint_white": ("paint_road_white", "white", "marking"),
    "paint_yellow": ("paint_road_yellow", "yellow", "marking"),
    "sheet_white": ("sheeting_white", "white", "sheeting"),
    "sheet_green": ("sheeting_green", "green", "sheeting"),
    "sheet_black": ("sheeting_black", None, "sheeting"),
}


def retro_material_lines(spd_dir: Path, out_dir: Path, wl: np.ndarray) -> dict[str, list[str]]:
    """``MakeNamedMaterial`` blocks replacing the builder's marking/sheeting materials."""
    out = {}
    for name, (surface, colour, kind) in RETRO_MATERIALS.items():
        p = MARKING if kind == "marking" else SHEETING
        rel_r = os.path.relpath(spd_dir / f"{surface}.spd", out_dir)
        lines = [
            f'MakeNamedMaterial "{name}" "string type" "retroreflective" "spectrum reflectance" "{rel_r}"',
            f'    "float observationangle" [{p["alpha0_deg"]}] "float lobewidth" [{p["width_deg"]}]'
            f' "float entranceexponent" [{p["q"]}]',
        ]
        if colour is not None:
            val = marking_ra_param(MARKING_RL[colour]) if kind == "marking" else sheeting_ra_param(colour)
            f = spd_dir / f"ra_{name}.spd"
            _write_spd(f, wl, retro_ra_spectrum(wl, surface, val))
            lines.append(f'    "spectrum ra" "{os.path.relpath(f, out_dir)}"')
        out[name] = lines
    return out


def strip_normal_map(lines: list[str], name: str = "asphalt") -> list[str]:
    """Drop the normal map of material ``name``.

    Normal mapping has no masking/shadowing, so under headlamps at a few degrees of incidence
    the tilted texel normals produce per-pixel speckle of the order of the mean (and colour
    speckle where lamps of different spectra light the same point); used for dusk/night.
    """
    out, inside = [], False
    for ln in lines:
        if ln.startswith(f'MakeNamedMaterial "{name}"'):
            inside = True
        elif not ln.startswith("    "):
            inside = False
        if inside and '"string normalmap"' in ln:
            continue
        out.append(ln)
    return out


def replace_named_materials(lines: list[str], repl: dict[str, list[str]]) -> list[str]:
    """Swap ``MakeNamedMaterial "<name>"`` statements (plus indented continuations) in place."""
    out: list[str] = []
    skipping = False
    for ln in lines:
        if skipping and ln.startswith("    "):
            continue
        skipping = False
        for name, new in repl.items():
            if ln.startswith(f'MakeNamedMaterial "{name}" '):
                out += new
                skipping = True
                break
        else:
            out.append(ln)
    return out


# --------------------------------------------------------------------------------------
# scene lines
# --------------------------------------------------------------------------------------
def _f(x: float) -> str:
    return f"{float(x):.6g}"


def _write_spd(path: Path, wl: np.ndarray, v: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    v = np.where(np.abs(v) < 1e-20, 0.0, v)  # pbrt's float parser rejects denormals
    path.write_text("".join(f"{a:g} {b:.6g}\n" for a, b in zip(wl, v, strict=True)))


def add_night_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("night / dusk (tools/highway_night.py)")
    g.add_argument("--time-of-day", choices=TIMES, default="day")
    g.add_argument("--twilight-sun-elevation", type=float, default=-4.0, help="Sun elevation for dusk [deg].")
    g.add_argument("--night-sun-elevation", type=float, default=-30.0, help="Sun elevation for night [deg].")
    g.add_argument("--moon-phase-deg", type=float, default=None, help="Moon phase angle (0 = full); default none.")
    g.add_argument("--moon-elevation", type=float, default=35.0)
    g.add_argument("--moon-azimuth", type=float, default=-40.0, help="From the road direction, + = right.")
    g.add_argument("--headlamps", choices=("auto", "mixed", "led", "halogen", "off"), default="auto")
    g.add_argument("--brake-lights", choices=("none", "lead", "all"), default="lead")
    g.add_argument("--streetlights", choices=("auto", "none", "led_4000k", "hps"), default="auto")
    g.add_argument("--retroreflective", choices=("auto", "on", "off"), default="auto")


def resolve(args: argparse.Namespace) -> dict:
    """Effective night options (``auto`` -> on for dusk/night, off for day)."""
    dark = args.time_of_day != "day"
    return {
        "dark": dark,
        "headlamps": ("mixed" if dark else "off") if args.headlamps == "auto" else args.headlamps,
        "streetlights": ("led_4000k" if dark else "none") if args.streetlights == "auto" else args.streetlights,
        "retroreflective": dark if args.retroreflective == "auto" else args.retroreflective == "on",
    }


def natural_light(args: argparse.Namespace, out_dir: Path, spd_dir: Path) -> tuple[list[str], dict]:
    """Sky (+ moon) light lines and the manifest ``lighting`` block for dusk/night."""
    tod = args.time_of_day
    sun_el = args.twilight_sun_elevation if tod == "dusk" else args.night_sun_elevation
    e_sky = twilight_illuminance_lux(sun_el)
    lines = []
    meta: dict = {"time_of_day": tod, "sun_elevation_deg": sun_el}
    if tod == "dusk" and sun_el > -18.0:
        sky_file = out_dir / "textures" / f"sky_twilight_{sun_el:.1f}.exr"
        if not sky_file.is_file():
            write_rgb_exr(sky_file, twilight_sky_map(512, sun_el))
        rot = args.sun_azimuth - 90.0  # map's sun azimuth is light +x -> world azimuth 90 deg
        lines += [
            f"# Twilight sky: sun {sun_el:.1f} deg, {e_sky:.3g} lux horizontal",
            "AttributeBegin",
            f"    Rotate {_f(rot)} 0 1 0",
            "    Rotate -90 1 0 0",
            f'    LightSource "infinite" "string filename" "{os.path.relpath(sky_file, out_dir)}"',
            f'        "float illuminance" [{_f(e_sky)}]',
            "AttributeEnd",
        ]
        meta["sky"] = {"model": "analytic twilight (RGB)", "illuminance_horizontal_lux": e_sky}
    else:
        _write_spd(spd_dir / "night_sky.spd", LIGHT_WL, night_sky_spectrum(LIGHT_WL))
        lines += [
            f"# Night sky (airglow + starlight): {e_sky:.3g} lux horizontal",
            'LightSource "infinite" "spectrum L" "spd/night_sky.spd"',
            f'    "float illuminance" [{_f(e_sky)}]',
        ]
        meta["sky"] = {"model": "uniform night sky, spectral (Leinert 1998)", "illuminance_horizontal_lux": e_sky}
    e_ref = e_sky
    if args.moon_phase_deg is not None and args.moon_elevation > 0:
        el, az = float(args.moon_elevation), float(args.moon_azimuth)
        e_n = moon_illuminance_lux(args.moon_phase_deg, el)
        s = math.sin(math.radians(el))
        e_dn_day, e_d_day = clear_sky_illuminance_lux(el)
        e_moon_sky = e_n * s * e_d_day / (e_dn_day * s)
        c = math.cos(math.radians(el))
        moon_w = [c * math.sin(math.radians(az)), s, c * math.cos(math.radians(az))]
        _write_spd(spd_dir / "moon.spd", LIGHT_WL, moon_spectrum(LIGHT_WL, el))
        sky_file = out_dir / "textures" / f"sky_moonlit_{el:.1f}.exr"
        if not sky_file.is_file():
            write_rgb_exr(sky_file, analytic_clear_sky(256, el))
        lines += [
            f"# Moon: phase angle {args.moon_phase_deg:g} deg, elevation {el:g} deg, {e_n:.3g} lux normal",
            'LightSource "distant" "spectrum L" "spd/moon.spd"',
            f'    "float illuminance" [{_f(e_n)}]',
            f'    "point3 from" [{_f(moon_w[0])} {_f(moon_w[1])} {_f(moon_w[2])}] "point3 to" [0 0 0]',
            "AttributeBegin",
            f"    Rotate {_f(az - 90.0)} 0 1 0",
            "    Rotate -90 1 0 0",
            f'    LightSource "infinite" "string filename" "{os.path.relpath(sky_file, out_dir)}"',
            f'        "float illuminance" [{_f(e_moon_sky)}]',
            "AttributeEnd",
        ]
        meta["moon"] = {
            "phase_angle_deg": args.moon_phase_deg,
            "elevation_deg": el,
            "azimuth_deg": az,
            "illuminance_normal_lux": e_n,
            "illuminance_horizontal_lux": e_n * s,
            "moonlit_sky_horizontal_lux": e_moon_sky,
            "model": "Krisciunas & Schaefer 1991; moonlit sky = clear-sky diffuse/direct ratio at the moon elevation",
        }
        e_ref += e_n * s + e_moon_sky
    meta["natural_illuminance_lux"] = e_ref
    return lines + [""], meta


def _box_mesh(cx: float, y0: float, cz: float, sx: float, sy: float, sz: float) -> list[str]:
    x0, x1, y1, z0, z1 = cx - sx / 2, cx + sx / 2, y0 + sy, cz - sz / 2, cz + sz / 2
    p = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1)]
    p.append((x0, y1, z1))
    t = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5]
    pts = " ".join(f"{_f(a)} {_f(b)} {_f(c)}" for a, b, c in p)
    return [f'Shape "trianglemesh" "integer indices" [{" ".join(map(str, t))}] "point3 P" [{pts}]']


def _quad_light(center, right, up, spd_rel: str, luminance: float) -> list[str]:
    """Rectangular one-sided Lambertian emitter; emits along cross(right, up)."""
    c, r, u = (np.asarray(v, dtype=np.float64) for v in (center, right, up))
    p00, p10, p01, p11 = c - r - u, c + r - u, c - r + u, c + r + u
    pts = " ".join(_f(x) for p in (p00, p10, p01, p11) for x in p)
    return [
        "AttributeBegin",
        f'    AreaLightSource "diffuse" "spectrum L" "{spd_rel}" "float scale" [{_f(luminance)}]',
        f'    Shape "bilinearmesh" "point3 P" [{pts}]',
        "AttributeEnd",
    ]


def write_light_assets(out_dir: Path, spd_dir: Path, opts: dict, n: int = 1024) -> dict:
    """Write lamp spectra and goniometric maps; returns their relative paths and fluxes."""
    for name, fn in LAMP_SPECTRA.items():
        _write_spd(spd_dir / f"lamp_{name}.spd", LIGHT_WL, fn(LIGHT_WL))
    info = {}
    for kind in ("halogen", "led"):
        m = headlamp_map(n, kind)
        f = out_dir / "textures" / f"lowbeam_{kind}.exr"
        write_y_exr(f, m * (1.0 - HEADLAMP_LENS_FRACTION))
        info[kind] = {"map": os.path.relpath(f, out_dir), "flux_lm": flux_lm(m), "peak_cd": float(m.max())}
    if opts["streetlights"] != "none":
        sl = STREETLIGHT[opts["streetlights"]]
        m = streetlight_map(n, sl["flux_lm"] * (1.0 - LENS_FLUX_FRACTION))
        f = out_dir / "textures" / f"streetlight_{opts['streetlights']}.exr"
        write_y_exr(f, m)
        info["streetlight"] = {"map": os.path.relpath(f, out_dir), "flux_lm": sl["flux_lm"], "peak_cd": float(m.max())}
    return info


# Vehicle lamp geometry / photometry.
HEADLAMP_LENS_FRACTION = 0.05  # flux moved into the visible lens emitter (see docs)
HEADLAMP_LENS = (0.20, 0.09)  # visible lens size [m]
TAIL_LAMP = (0.30, 0.10)
TAIL_CD, STOP_CD = 6.0, 80.0  # UN ECE R7: tail 4-12 cd (single), stop 60-185 cd at H-V
EGO_LAMP_OFFSET = (0.75, 0.65, 1.9)  # (half track, height, ahead of the camera) [m]


def _gonio(pos, heading_deg: float, map_rel: str, spd_rel: str, rot_extra: list[str] | None = None) -> list[str]:
    return [
        "AttributeBegin",
        f"    Translate {_f(pos[0])} {_f(pos[1])} {_f(pos[2])}",
        f"    Rotate {_f(heading_deg)} 0 1 0",
        *(rot_extra or []),
        f'    LightSource "goniometric" "spectrum I" "{spd_rel}" "string filename" "{map_rel}" "float scale" [1]',
        "AttributeEnd",
    ]


def vehicle_light_lines(cars: list[dict], ego: tuple[float, float], opts: dict, assets: dict) -> tuple[list, dict]:
    """Headlamps (goniometric) + lens emitters and tail/stop lamps (area lights) for every car.

    ``cars``: dicts with x, z, heading_deg (0 = along +z), length_m, width_m, height_m, lane.
    """
    lines = ["# ---- vehicle lamps (tools/highway_night.py)"]
    meta = {"headlamps": [], "tail_lamps": []}
    mode = opts["headlamps"]
    lens_area = 4.0 * (HEADLAMP_LENS[0] / 2) * (HEADLAMP_LENS[1] / 2)

    def lamp_kind(k: int) -> str:
        return mode if mode in ("led", "halogen") else ("led" if k % 2 == 0 else "halogen")

    if mode != "off":
        kind = lamp_kind(0)
        for sx in (-1, 1):
            p = (ego[0] + sx * EGO_LAMP_OFFSET[0], EGO_LAMP_OFFSET[1], ego[1] + EGO_LAMP_OFFSET[2])
            spd = "spd/lamp_led_headlamp.spd" if kind == "led" else "spd/lamp_halogen.spd"
            lines += _gonio(p, 0.0, assets[kind]["map"], spd)
        meta["headlamps"].append({"car": "ego", "kind": kind, "flux_lm_per_lamp": assets[kind]["flux_lm"]})
    lead = None
    if opts["brake"] == "lead":
        ahead = [c for c in cars if c["lane"] == opts["ego_lane"] and c["z"] > ego[1]]
        lead = min(ahead, key=lambda c: c["z"])["id"] if ahead else None
    for k, c in enumerate(cars):
        hd, half_l, half_w = c["heading_deg"], c["length_m"] / 2, c["width_m"] / 2
        lamp_y = float(np.clip(0.47 * c["height_m"], 0.55, 0.8))
        tail_y = float(np.clip(0.6 * c["height_m"], 0.7, 0.95))
        lines += ["AttributeBegin", f"    Translate {_f(c['x'])} 0 {_f(c['z'])}", f"    Rotate {_f(hd)} 0 1 0"]
        if mode != "off":
            kind = lamp_kind(k + 1)
            spd = "spd/lamp_led_headlamp.spd" if kind == "led" else "spd/lamp_halogen.spd"
            lens_l = HEADLAMP_LENS_FRACTION * assets[kind]["flux_lm"] / (math.pi * lens_area)
            for sx in (-1, 1):
                x = sx * (half_w - 0.28)
                lines += _gonio((x, lamp_y, half_l + 0.06), 0.0, assets[kind]["map"], spd)
                lines += _quad_light(
                    (x, lamp_y, half_l + 0.02), (HEADLAMP_LENS[0] / 2, 0, 0), (0, HEADLAMP_LENS[1] / 2, 0), spd, lens_l
                )
            meta["headlamps"].append({"car": c["id"], "kind": kind, "lens_luminance_cd_m2": lens_l})
        braking = opts["brake"] == "all" or c["id"] == lead
        tail_area = TAIL_LAMP[0] * TAIL_LAMP[1]
        lum = (TAIL_CD + (STOP_CD if braking else 0.0)) / tail_area  # Lambertian: I(0) = L A
        for sx in (-1, 1):
            x = sx * (half_w - 0.22)
            # right x up = -z for a quad whose "right" is -x: emits backwards.
            lines += _quad_light(
                (x, tail_y, -half_l - 0.015),
                (-TAIL_LAMP[0] / 2, 0, 0),
                (0, TAIL_LAMP[1] / 2, 0),
                "spd/lamp_red_led.spd",
                lum,
            )
        meta["tail_lamps"].append({"car": c["id"], "braking": braking, "intensity_cd": lum * tail_area})
        lines += ["AttributeEnd"]
    return lines + [""], meta


def streetlight_lines(
    opts: dict, assets: dict, median_x: float, z0: float, z1: float, carriageways: tuple[tuple[float, float], ...]
) -> tuple[list[str], dict]:
    """Twin-arm 12 m median poles with full-cutoff Type III luminaires (goniometric + lens)."""
    kind = opts["streetlights"]
    if kind == "none":
        return [], {}
    sl = STREETLIGHT[kind]
    spd = f"spd/lamp_{sl['spectrum']}.spd"
    head = (0.7, 0.14, 0.32)
    lens = (0.6, 0.26)
    lens_l = LENS_FLUX_FRACTION * sl["flux_lm"] / (math.pi * lens[0] * lens[1])
    lines = [
        "# ---- street lights (tools/highway_night.py)",
        'MakeNamedMaterial "pole" "string type" "coateddiffuse" "spectrum reflectance" "spd/galvanized.spd"'
        ' "float roughness" [0.3]',
        'MakeNamedMaterial "luminaire" "string type" "diffuse" "rgb reflectance" [0.08 0.08 0.08]',
    ]
    poles = np.arange(z0, z1, POLE_SPACING_M)
    lines.append('NamedMaterial "pole"')
    for z in poles:
        lines += [
            "AttributeBegin",
            f"    Translate {_f(median_x)} 0 {_f(z)}",
            "    Rotate -90 1 0 0",
            f'    Shape "cylinder" "float radius" [0.09] "float zmin" [0] "float zmax" [{_f(POLE_HEIGHT_M)}]',
            "AttributeEnd",
            *_box_mesh(median_x, POLE_HEIGHT_M - 0.25, z, 2 * ARM_OUTREACH_M, 0.08, 0.08),
        ]
    y_lens = POLE_HEIGHT_M - 0.25 - head[1]
    lines.append('NamedMaterial "luminaire"')
    for x, z, side in streetlight_positions(median_x, z0, z1):
        lines += _box_mesh(x + side * head[0] / 2 - side * 0.1, y_lens + 0.002, z, head[0], head[1], head[2])
        lines += _quad_light(
            (x + side * 0.25, y_lens, z), (0, 0, lens[1] / 2), (lens[0] / 2, 0, 0), spd, lens_l
        )  # right(z) x up(x) = -y: emits downwards
        lines += _gonio(
            (x + side * 0.25, y_lens - 0.03, z), 0.0 if side > 0 else 180.0, assets["streetlight"]["map"], spd
        )
    e = road_illuminance(sl["flux_lm"], median_x, carriageways[0], (100.0, 100.0 + POLE_SPACING_M))
    meta = {
        "kind": kind,
        "spectrum": spd,
        "luminaire_flux_lm": sl["flux_lm"],
        "lens_flux_fraction": LENS_FLUX_FRACTION,
        "pole_height_m": POLE_HEIGHT_M,
        "spacing_m": POLE_SPACING_M,
        "arm_outreach_m": ARM_OUTREACH_M,
        "poles": len(poles),
        "distribution": "analytic full-cutoff IES Type III medium",
        "carriageway_illuminance": e,
    }
    return lines + [""], meta


def car_geometry(info: dict | None, length_m: float, x: float, heading_deg: float) -> dict:
    """World placement and bounding size of a car (asset bbox scaled to length, or the proxy)."""
    if info is None:
        size = (4.6, 1.8, 1.5)
    else:
        lo, hi = np.array(info["bbox_min"]), np.array(info["bbox_max"])
        s = length_m / float(hi[0] - lo[0])
        size = (length_m, float(hi[2] - lo[2]) * s, float(hi[1] - lo[1]) * s)
    return {
        "x": float(x),
        "heading_deg": float(heading_deg),
        "length_m": size[0],
        "width_m": size[1],
        "height_m": size[2],
    }


def scene_lights(
    args: argparse.Namespace,
    opts: dict,
    out_dir: Path,
    spd_dir: Path,
    cars_meta: list[dict],
    eye: np.ndarray,
    median_x: float,
    carriageway: tuple[float, float],
    pole_z: tuple[float, float],
) -> tuple[list[str], dict]:
    """All artificial lights (vehicle lamps, street lights); empty when everything is off."""
    if opts["headlamps"] == "off" and opts["streetlights"] == "none" and not opts["dark"]:
        return [], {}
    assets = write_light_assets(out_dir, spd_dir, opts)
    cars = [dict(c, z=c["distance_m"]) for c in cars_meta]
    vopts = dict(opts, brake=args.brake_lights, ego_lane=args.cam_lane)
    lines, vmeta = vehicle_light_lines(cars, (float(eye[0]), float(eye[2])), vopts, assets)
    slines, smeta = streetlight_lines(opts, assets, median_x, pole_z[0], pole_z[1], (carriageway,))
    meta = {
        "headlamp_beam": {
            "pattern": "analytic ECE R112/R149 class-B passing beam (RHT), goniometric map in cd",
            **{k: {kk: v for kk, v in assets[k].items() if kk != "map"} for k in ("halogen", "led")},
            "lens_flux_fraction": HEADLAMP_LENS_FRACTION,
        },
        **vmeta,
        **({"streetlights": smeta} if smeta else {}),
    }
    return lines + slines, meta


def update_manifest(manifest: dict, opts: dict, natural: dict | None, artificial: dict) -> None:
    """Night/dusk: natural reference illuminance + artificial-light metadata; retro note."""
    from pbrt_spectral_exr_to_electrons import PBRT_CIE_Y_INTEGRAL

    if natural is not None:
        e = natural["natural_illuminance_lux"]
        manifest["lighting"] = {
            **natural,
            "reference_illuminance_lux": e,
            "reference_illuminance_exr_lux": 683.0 * PBRT_CIE_Y_INTEGRAL * e,
            "reference": "horizontal illuminance from the natural sources (twilight/night sky + moon) at road "
            "level, unoccluded; lamps are absolute photometric sources (cd, cd/m^2) in the same units",
        }
        manifest["approximations"].append(
            "Dusk sky radiance is an analytic RGB twilight map; the night sky is uniform (no stars, no horizon "
            "brightening); vehicle-lamp and luminaire lenses are Lambertian emitters carrying a small fixed "
            "fraction of the flux, so their apparent luminance is lower than real optics."
        )
    if artificial:
        manifest.setdefault("lighting", {})["artificial"] = artificial
    if opts["retroreflective"]:
        manifest["approximations"] = [
            a for a in manifest["approximations"] if not a.startswith("Road-marking glass-bead retroreflection")
        ]
        manifest["retroreflection"] = {
            "material": "retroreflective (third_party/patches/0001-retroreflective-material.patch)",
            "sheeting": {
                "standard": "ASTM D4956 Type III minima x " + str(SHEETING_MARGIN),
                **SHEETING,
                "ra_at_0.2deg_-4deg": {c: SHEETING_MARGIN * v[0] for c, v in D4956_TYPE_III.items()},
            },
            "markings": {"standard": "EN 1436 30 m geometry", **MARKING, "rl_mcd_m2_lx": MARKING_RL},
        }
