"""Skin reflectance 350-1100 nm from a two-layer tissue optics model (for NIR / 940 nm DMS).

Measured skin spectra in the repo (ColorChecker skin patches) stop at 730 nm. This model follows
S. L. Jacques, "Skin optics", Oregon Medical Laser Center News (1998),
https://omlc.org/news/jan98/skinoptics.html:

* epidermis: mu_a = f_mel * 6.6e11 nm^-3.33 + (1 - f_mel) * baseline, baseline = 7.84e8 nm^-3.255;
  two passes through ``d_epi`` (60 um).
* dermis: mu_a = f_blood * mu_a,blood + f_water * mu_a,water + baseline; blood from Prahl's
  haemoglobin extinction (150 g/L, saturation ``so2``), water from Hale & Querry (1973).
  mu_s' = 2e12 nm^-4 + 2e5 nm^-1.5 (Rayleigh + Mie on collagen).
* diffuse reflectance of the semi-infinite dermis: Rd = exp(-A / sqrt(3 (1 + mu_s'/mu_a))),
  A = 7.5 (OMLC "Diffuse reflectance from a semi-infinite medium": A ~ 7-8 for soft tissue).
* plus a 2.8 % specular surface term (n = 1.4).

Engineering approximation, not measured data: f_water, f_blood and A are nominal values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
HB_MOLAR_MASS = 64500.0
A_RD = 7.5
SPECULAR = 0.028
#: Fitzpatrick-style presets: volume fraction of melanosomes in the epidermis.
MELANIN = {"I": 0.02, "II": 0.04, "III": 0.08, "IV": 0.15, "V": 0.25, "VI": 0.40}


def _csv(rel: str) -> np.ndarray:
    lines = [ln for ln in (REPO / rel).read_text().splitlines() if ln and not ln.startswith("#")]
    return np.genfromtxt(lines, delimiter=",", names=True)


def water_mua(wl_nm: np.ndarray) -> np.ndarray:
    d = _csv("spectra/water/hale_querry_1973.csv")
    return np.interp(wl_nm, d["wavelength_nm"], d["absorption_per_cm"])


def blood_mua(wl_nm: np.ndarray, so2: float = 0.75, hb_g_l: float = 150.0) -> np.ndarray:
    d = _csv("spectra/blood/prahl_hemoglobin.csv")
    e = so2 * d["hbo2"] + (1 - so2) * d["hb"]
    # Prahl's table stops at 1000 nm; hold the last value beyond (blood is a minor term there).
    return 2.303 * np.interp(wl_nm, d["wavelength_nm"], e) * hb_g_l / HB_MOLAR_MASS


def skin_reflectance(
    wl_nm: np.ndarray,
    f_mel: float = MELANIN["III"],
    f_blood: float = 0.002,
    f_water: float = 0.65,
    so2: float = 0.75,
    d_epi_cm: float = 60e-4,
) -> np.ndarray:
    wl = np.asarray(wl_nm, float)
    base = 7.84e8 * wl**-3.255
    mua_epi = f_mel * 6.6e11 * wl**-3.33 + (1 - f_mel) * base
    mua_derm = f_blood * blood_mua(wl, so2) + f_water * water_mua(wl) + base
    musp = 2e12 * wl**-4 + 2e5 * wl**-1.5
    rd = np.exp(-A_RD / np.sqrt(3.0 * (1.0 + musp / mua_derm)))
    return SPECULAR + (1 - SPECULAR) ** 2 * rd * np.exp(-2.0 * mua_epi * d_epi_cm)


def extend_measured(
    wl_nm: np.ndarray, meas_wl: np.ndarray, meas_r: np.ndarray, join_nm: float = 730.0, **model
) -> np.ndarray:
    """Measured reflectance up to ``join_nm``; beyond it the measurement times the model's
    relative shape R_model(lambda) / R_model(join_nm). The absolute model level is not trusted
    (it under-predicts the ColorChecker skin patches in the red), only its NIR shape."""
    wl = np.asarray(wl_nm, float)
    r_join = np.interp(join_nm, meas_wl, meas_r)
    shape = skin_reflectance(wl, **model) / skin_reflectance(np.array([join_nm]), **model)[0]
    return np.where(wl <= join_nm, np.interp(wl, meas_wl, meas_r), np.clip(r_join * shape, 0.0, 1.0))
