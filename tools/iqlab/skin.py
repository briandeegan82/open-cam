"""Skin-tone test spectra and metrics.

Spectra: the measured X-Rite ColorChecker "light skin" (patch 02) reflectance is the base; a range of
darker/lighter tones is synthesised by changing the epidermal melanin optical depth with the melanin
absorption power law mu_a ~ lambda^-3.33 (S. L. Jacques, "Optical properties of biological tissues: a
review", Phys. Med. Biol. 58 R37 (2013); melanosome fit lambda^-3.33). Two passes through the
epidermis: R(lambda) = R_base(lambda) * exp(-tau * (lambda/500 nm)^-3.33). The haemoglobin "W" and
dermal scattering come from the measured base, so these are physically plausible but *synthetic* --
not measured population data. The measured ColorChecker dark/light skin patches are included as-is.

Metrics: CIEDE2000, lightness / chroma / hue-angle shifts, and the individual typology angle
ITA = atan2(L* - 50, b*) (Chardon et al. 1991), the usual skin-tone classification axis
(> 55 very light ... < -30 dark).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from colour_science import _read_csv_curve, cmf_on_grid, delta_e_2000, xyz_to_lab

MELANIN_EXPONENT = -3.33
DEFAULT_TAU = (-0.25, 0.0, 0.35, 0.8, 1.3, 1.9, 2.6, 3.4)


def melanin_reflectance(wl_nm: np.ndarray, base: np.ndarray, tau: float) -> np.ndarray:
    return np.clip(base * np.exp(-float(tau) * (np.asarray(wl_nm) / 500.0) ** MELANIN_EXPONENT), 0.0, 1.0)


def skin_tone_set(repo: Path, wl_nm: np.ndarray, taus=DEFAULT_TAU) -> list[dict]:
    """Synthetic melanin series on the ColorChecker light-skin base + the two measured skin patches."""
    xr = Path(repo) / "spectra" / "xrite"
    curves = {k: _read_csv_curve(xr / f) for k, f in (("dark", "01_dark_skin.csv"), ("light", "02_light_skin.csv"))}
    base = np.interp(wl_nm, *curves["light"])
    out = [
        {
            "name": f"skin_tau{t:+.2f}",
            "source": "synthetic melanin (Jacques 2013) on ColorChecker light skin",
            "tau": t,
            "reflectance": melanin_reflectance(wl_nm, base, t),
        }
        for t in taus
    ]
    for k in ("light", "dark"):
        out.append(
            {
                "name": f"colorchecker_{k}_skin",
                "source": "X-Rite ColorChecker (measured)",
                "tau": None,
                "reflectance": np.interp(wl_nm, *curves[k]),
            }
        )
    return out


def reflectance_lab(wl_nm: np.ndarray, refl: np.ndarray, illum: np.ndarray) -> np.ndarray:
    """CIELAB of reflectances (n, L) under ``illum`` (L,), white = perfect diffuser under the same illuminant."""
    cmf = cmf_on_grid(wl_nm)
    w = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    norm = w(illum * cmf[1], wl_nm)
    xyz = np.stack([w(np.atleast_2d(refl) * illum * cmf[i], wl_nm, axis=-1) for i in range(3)], -1) / norm
    white = np.array([w(illum * cmf[i], wl_nm) for i in range(3)]) / norm
    return xyz_to_lab(xyz, white)


def lab_to_xyz(lab: np.ndarray, white: np.ndarray) -> np.ndarray:
    """Inverse CIELAB (``white`` = reference white XYZ)."""
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16.0) / 116.0
    f = np.stack([fy + lab[..., 1] / 500.0, fy, fy - lab[..., 2] / 200.0], -1)
    d = 6.0 / 29.0
    return np.where(f > d, f**3, 3 * d * d * (f - 4.0 / 29.0)) * np.asarray(white, dtype=np.float64)


def ita_deg(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    return np.degrees(np.arctan2(lab[..., 0] - 50.0, lab[..., 2]))


def skin_metrics(lab_meas: np.ndarray, lab_ref: np.ndarray) -> dict:
    """Per-tone errors of measured vs reference CIELAB (arrays (n, 3))."""
    m, r = np.atleast_2d(lab_meas).astype(float), np.atleast_2d(lab_ref).astype(float)
    cm, cr = np.hypot(m[:, 1], m[:, 2]), np.hypot(r[:, 1], r[:, 2])
    dh = (np.degrees(np.arctan2(m[:, 2], m[:, 1]) - np.arctan2(r[:, 2], r[:, 1])) + 180.0) % 360.0 - 180.0
    return {
        "delta_e00": delta_e_2000(m, r),
        "delta_L": m[:, 0] - r[:, 0],
        "delta_C": cm - cr,
        "delta_hue_deg": dh,
        "ita_measured": ita_deg(m),
        "ita_reference": ita_deg(r),
    }
