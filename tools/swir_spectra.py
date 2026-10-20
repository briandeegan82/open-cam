#!/usr/bin/env python3
"""SWIR (to ~2.5 um) spectral data: InGaAs photodiode QE models, ASTM G173 sunlight, Planck lamps.

Silicon stops responding near 1100 nm, so short-wave infrared cameras use InGaAs.  This module
builds QE curves from a small physical model rather than digitised datasheet plots:

* **Bandgap** -- In(1-x)Ga(x)As at temperature ``T``: Varshni parameters of the binaries and the
  bowing parameter from I. Vurgaftman, J. R. Meyer, L. R. Ram-Mohan, "Band parameters for III-V
  compound semiconductors and their alloys", J. Appl. Phys. 89, 5815 (2001).  x = 0.47 (lattice
  matched to InP) gives 0.738 eV / 1.68 um at 25 C; the extended-wavelength detector uses
  In0.81Ga0.19As (cutoff ~2.6 um).
* **Absorption** -- direct-gap ``alpha = A sqrt(E - Eg)`` with an exponential Urbach tail
  (energy ``E_u``) below the gap.  ``A`` (3e4 /cm/eV^0.5) reproduces the ~0.8e4 /cm of
  In0.53Ga0.47As at 1.55 um; it underestimates the visible absorption, which does not matter
  because micron-thick absorbers are opaque there anyway.
* **Window** -- light reaches the absorber through InP (Eg = 1.353 eV / 916 nm at 300 K, same
  model), which sets the ~0.9 um cut-on of standard InGaAs: an InP cap/substrate of thickness
  ``window_um``.  "VisSWIR" sensors remove the substrate (IMX990-class) and keep only a thin cap.
* **QE** -- ``(1 - R) T_window (1 - exp(-alpha d)) eta_c``: AR-coated front reflectance ``R``,
  absorber thickness ``d`` and a wavelength-independent collection efficiency ``eta_c``.

The default parameters reproduce Hamamatsu's G12180 (standard) and G12183 (extended) InGaAs
PIN photodiode datasheets: peak photosensitivity 1.1 A/W at 1.55 um and 1.3 A/W at 2.3 um, and
response ranges 0.9-1.7 / 0.9-2.6 um at 25 C that shrink to 1.65 / 2.55 um at -20 C (checked in
``tests/test_swir.py``).  Focal-plane arrays add fill factor and readout losses (set them via
the sensor model's ``fill_factor``); the VisSWIR visible response is an assumption (thin cap),
not a fit to a product.

Illuminants:

* **Sunlight** -- ASTM G173-03 reference spectra (SMARTS 2.9.2, AM1.5, 280-4000 nm, with the
  1.4 / 1.9 um water-vapour bands), from the BSD-3-licensed copy in pvlib
  (``pvlib/data/ASTMG173.csv``; ``spectra/illuminant/swir/ASTMG173.csv``).
* **Tungsten / halogen** -- Planck blackbody (CIE illuminant A is 2856 K).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
GRID_NM = np.arange(360.0, 2601.0, 1.0)
HC_EV_NM = 1239.841984
G173_CSV = "spectra/illuminant/swir/ASTMG173.csv"

# Vurgaftman et al. (2001): Eg(T) = Eg0 - a T^2 / (T + b), a in eV/K, b in K.
_VARSHNI = {"InAs": (0.417, 0.276e-3, 93.0), "GaAs": (1.519, 0.5405e-3, 204.0), "InP": (1.4236, 0.363e-3, 162.0)}
INGAAS_BOWING_EV = 0.477


def varshni_ev(material: str, temp_k: float) -> float:
    eg0, a, b = _VARSHNI[material]
    return eg0 - a * temp_k**2 / (temp_k + b)


def ingaas_bandgap_ev(x_ga: float, temp_k: float = 298.15) -> float:
    """Bandgap of In(1-x)Ga(x)As (eV)."""
    return (
        x_ga * varshni_ev("GaAs", temp_k)
        + (1.0 - x_ga) * varshni_ev("InAs", temp_k)
        - x_ga * (1.0 - x_ga) * INGAAS_BOWING_EV
    )


def direct_gap_alpha_per_um(
    lambdas_nm: np.ndarray, eg_ev: float, *, a_per_cm: float = 3.0e4, urbach_ev: float = 0.008
) -> np.ndarray:
    """Absorption coefficient (1/um): ``A sqrt(E - Eg)`` above the gap, Urbach tail below.

    The tail joins the square root at ``Eg + E_u / 2`` with matching value and slope.
    """
    e = HC_EV_NM / np.asarray(lambdas_nm, dtype=np.float64)
    e_j = eg_ev + 0.5 * urbach_ev
    a_j = a_per_cm * np.sqrt(e_j - eg_ev)
    above = a_per_cm * np.sqrt(np.clip(e - eg_ev, 0.0, None))
    tail = a_j * np.exp(np.clip((e - e_j) / urbach_ev, -700.0, 0.0))
    return np.where(e >= e_j, above, tail) * 1e-4


@dataclass(frozen=True)
class InGaAsDetector:
    name: str
    x_ga: float
    absorber_um: float
    window_um: float
    reflectance: float = 0.03
    collection_efficiency: float = 1.0
    note: str = ""

    def qe(self, lambdas_nm: np.ndarray, temp_c: float = 25.0) -> np.ndarray:
        t = temp_c + 273.15
        lam = np.asarray(lambdas_nm, dtype=np.float64)
        a_abs = direct_gap_alpha_per_um(lam, ingaas_bandgap_ev(self.x_ga, t))
        a_win = direct_gap_alpha_per_um(lam, varshni_ev("InP", t), a_per_cm=4.0e4)
        t_win = np.exp(-a_win * self.window_um)
        return (1.0 - self.reflectance) * t_win * (1.0 - np.exp(-a_abs * self.absorber_um)) * self.collection_efficiency


DETECTORS: dict[str, InGaAsDetector] = {
    "ingaas_standard": InGaAsDetector(
        "ingaas_standard",
        0.47,
        4.0,
        3.0,
        collection_efficiency=0.96,
        note="In0.53Ga0.47As on InP, 0.9-1.7 um (fit to Hamamatsu G12180)",
    ),
    "ingaas_visswir": InGaAsDetector(
        "ingaas_visswir",
        0.47,
        4.0,
        0.02,
        collection_efficiency=0.96,
        note="substrate-removed In0.53Ga0.47As, 20 nm InP cap, 0.4-1.7 um (assumed cap; IMX990-class)",
    ),
    "ingaas_extended": InGaAsDetector(
        "ingaas_extended",
        0.19,
        4.0,
        0.5,
        collection_efficiency=0.77,
        note="In0.81Ga0.19As, 0.9-2.6 um (fit to Hamamatsu G12183)",
    ),
}


def responsivity_a_per_w(lambdas_nm: np.ndarray, qe: np.ndarray) -> np.ndarray:
    return np.asarray(qe) * np.asarray(lambdas_nm) / HC_EV_NM


def response_range_nm(lambdas_nm: np.ndarray, qe: np.ndarray, frac: float = 0.1) -> tuple[float, float]:
    """Shortest / longest wavelength where QE >= ``frac`` x peak (linear interpolation)."""
    lam = np.asarray(lambdas_nm, dtype=np.float64)
    q = np.asarray(qe, dtype=np.float64) / np.max(qe) - frac
    idx = np.nonzero(q >= 0)[0]
    i0, i1 = idx[0], idx[-1]
    lo = lam[i0] if i0 == 0 else np.interp(0.0, [q[i0 - 1], q[i0]], [lam[i0 - 1], lam[i0]])
    hi = lam[i1] if i1 == len(lam) - 1 else np.interp(0.0, [q[i1 + 1], q[i1]], [lam[i1 + 1], lam[i1]])
    return float(lo), float(hi)


def read_g173(repo: Path = REPO) -> dict[str, np.ndarray]:
    """ASTM G173-03: wavelength (nm) and extraterrestrial / global (37 deg tilt) / direct, W/m^2/nm."""
    d = np.loadtxt(Path(repo) / G173_CSV, delimiter=",", skiprows=2)
    return {"wavelength_nm": d[:, 0], "extraterrestrial": d[:, 1], "global": d[:, 2], "direct": d[:, 3]}


def planck_spd(lambdas_nm: np.ndarray, temp_k: float) -> np.ndarray:
    """Blackbody spectral radiance (W / m^2 / sr / nm)."""
    lam = np.asarray(lambdas_nm, dtype=np.float64) * 1e-9
    h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    return 2 * h * c**2 / lam**5 / np.expm1(h * c / (lam * k * temp_k)) * 1e-9


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=Path, default=REPO)
    args = ap.parse_args()
    repo = args.repo_root
    from nir_spectra import _write  # noqa: PLC0415

    for det in DETECTORS.values():
        _write(
            repo / f"spectra/QE/swir/QE_{det.name}.csv",
            GRID_NM,
            det.qe(GRID_NM),
            f"{det.note}; 25 C model, tools/swir_spectra.py",
        )
    g = read_g173(repo)
    for col in ("global", "direct"):
        _write(
            repo / f"spectra/illuminant/swir/sun_astm_g173_{col}.csv",
            GRID_NM,
            np.interp(GRID_NM, g["wavelength_nm"], g[col]),
            f"ASTM G173-03 AM1.5 {col} irradiance (W/m^2/nm), resampled to 1 nm; tools/swir_spectra.py",
        )
    bb = planck_spd(GRID_NM, 2856.0)
    _write(
        repo / "spectra/illuminant/swir/tungsten_2856K.csv",
        GRID_NM,
        bb / np.interp(560.0, GRID_NM, bb),
        "Planck blackbody 2856 K (CIE A colour temperature), normalised to 1 at 560 nm; tools/swir_spectra.py",
    )


if __name__ == "__main__":
    main()
