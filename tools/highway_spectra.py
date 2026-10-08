#!/usr/bin/env python3
"""Spectral reflectances for the highway scene (smooth analytic fits to published spectra).

Curves are written as pbrt ``.spd`` files by tools/build_highway_scene.py. Sources the
shapes are fitted to: Herold et al. 2004 (urban spectral library: asphalt, concrete),
TiO2 / organic-pigment traffic paints, ASTM D4956 sheeting colours (daytime diffuse part),
automotive basecoat colours, USGS speclib vegetation and soil.
"""

from __future__ import annotations

import numpy as np


def _sig(wl, c, w):
    return 1.0 / (1.0 + np.exp(-(wl - c) / w))


def _g(wl, c, s):
    return np.exp(-0.5 * ((wl - c) / s) ** 2)


def _ramp(wl):
    return np.clip((wl - 380.0) / 400.0, 0.0, None)


SURFACES = {
    # Aged highway asphalt (oxidised binder, exposed aggregate) ~0.10-0.16; fresh ~0.05.
    "asphalt_aged": lambda wl: 0.095 + 0.06 * _ramp(wl),
    "asphalt_new": lambda wl: 0.045 + 0.015 * _ramp(wl),
    # Weathered concrete (Herold 2004 ~0.2-0.3).
    "concrete": lambda wl: 0.20 + 0.09 * _ramp(wl) - 0.02 * _g(wl, 400.0, 25.0),
    # Waterborne TiO2 road paint (UV edge ~410 nm) and lead-free organic yellow.
    "paint_road_white": lambda wl: 0.12 + 0.62 * _sig(wl, 412.0, 10.0),
    "paint_road_yellow": lambda wl: 0.07 + 0.58 * _sig(wl, 512.0, 14.0),
    "sheeting_white": lambda wl: 0.10 + 0.72 * _sig(wl, 410.0, 12.0),
    "sheeting_green": lambda wl: 0.035 + 0.17 * _g(wl, 520.0, 32.0) + 0.04 * _sig(wl, 690.0, 20.0),
    "sheeting_black": lambda wl: np.full_like(wl, 0.04),
    "sheeting_red": lambda wl: 0.05 + 0.62 * _sig(wl, 600.0, 12.0),
    # Automotive basecoats (under the clearcoat interface added by coateddiffuse).
    "carpaint_white": lambda wl: 0.10 + 0.72 * _sig(wl, 418.0, 12.0),
    "carpaint_black": lambda wl: np.full_like(wl, 0.025),
    "carpaint_silver": lambda wl: 0.42 - 0.03 * _ramp(wl),
    "carpaint_gray": lambda wl: 0.16 + 0.01 * _ramp(wl),
    "carpaint_red": lambda wl: 0.04 + 0.62 * _sig(wl, 598.0, 11.0),
    "carpaint_blue": lambda wl: 0.04 + 0.30 * _g(wl, 462.0, 34.0) + 0.05 * _sig(wl, 720.0, 25.0),
    "carpaint_darkgreen": lambda wl: 0.03 + 0.10 * _g(wl, 530.0, 35.0) + 0.06 * _sig(wl, 700.0, 20.0),
    "galvanized": lambda wl: 0.42 + 0.04 * _ramp(wl),
    "rubber": lambda wl: 0.03 + 0.012 * _ramp(wl),
    # Green vegetation: chlorophyll wells at 430/670 nm, green bump, red edge at ~715 nm.
    "grass": lambda wl: np.clip(
        0.04 + 0.075 * _g(wl, 552.0, 30.0) - 0.02 * _g(wl, 672.0, 15.0) + 0.42 * _sig(wl, 718.0, 11.0), 0.02, 1.0
    ),
    "leaf_reflectance": lambda wl: np.clip(
        0.045 + 0.09 * _g(wl, 552.0, 28.0) - 0.025 * _g(wl, 672.0, 14.0) + 0.42 * _sig(wl, 715.0, 10.0), 0.02, 1.0
    ),
    "leaf_transmittance": lambda wl: np.clip(
        0.01 + 0.08 * _g(wl, 555.0, 30.0) + 0.42 * _sig(wl, 715.0, 10.0), 0.0, 1.0
    ),
    "bark": lambda wl: 0.06 + 0.16 * _ramp(wl) ** 1.3,
    "soil": lambda wl: 0.07 + 0.22 * _ramp(wl),
}

CAR_PAINTS = ("white", "black", "silver", "gray", "red", "blue", "darkgreen")


def reflectance(name: str, wl: np.ndarray) -> np.ndarray:
    if name not in SURFACES:
        raise KeyError(f"unknown surface {name!r}; known: {sorted(SURFACES)}")
    return np.clip(SURFACES[name](np.asarray(wl, dtype=np.float64)), 0.0, 1.0)
