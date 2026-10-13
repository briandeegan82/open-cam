"""Glass dispersion for traced lens prescriptions (pbrt ``RealisticCamera`` lens files).

A pbrt lens file gives one refractive index per glass (its n_d), so stock pbrt traces every
wavelength through the same indices and renders no chromatic aberration.  A sidecar
``<lens>.glass.yaml`` next to the ``.dat`` assigns each glass row a catalogue glass whose
Sellmeier curve n_g(lambda) supplies the shape of the dispersion, and optionally the design Abbe
number nu_d (from the patent).  The index used at wavelength lambda is

    n(lambda) = n_file + s * (n_g(lambda) - n_g(lambda_d)),
    s = ((n_file - 1) / nu_d) / (n_g(lambda_F) - n_g(lambda_C)),

so n(lambda_d) = n_file exactly (focus, f-number and EFL at 587.6 nm are unchanged) and
n_F - n_C = (n_file - 1) / nu_d, the design's principal dispersion.  Without ``nu_d`` the
catalogue glass's own Abbe number is used.  The same seven numbers per row (s, B1..B3, C1..C3)
are passed to the patched pbrt camera as ``"float dispersion"`` (third_party/patches/
0003-realistic-camera-dispersion.patch), and :func:`refractive_indices` gives the Python tracers
(``lens_diffraction_psf``) the identical n(lambda).

Sellmeier coefficients: SCHOTT optical glass catalogue (2018-06-01 data sheet, as shipped in the
``opticalglass`` 2.0.2 package); n^2 = 1 + sum_k B_k l^2 / (l^2 - C_k), l in um, C_k in um^2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LAMBDA_D_NM = 587.5618  # He d line
LAMBDA_F_NM = 486.1327  # H F line
LAMBDA_C_NM = 656.2725  # H C line

# name: (n_d, nu_d, (B1, B2, B3, C1, C2, C3))
SCHOTT_SELLMEIER: dict[str, tuple[float, float, tuple[float, ...]]] = {
    "F2": (1.62004, 36.37, (1.34533359, 0.209073176, 0.937357162, 0.00997743871, 0.0470450767, 111.886764)),
    "F5": (1.60342, 38.03, (1.3104463, 0.19603426, 0.96612977, 0.00958633048, 0.0457627627, 115.011883)),
    "N-BAF10": (1.67003, 47.11, (1.5851495, 0.143559385, 1.08521269, 0.00926681282, 0.0424489805, 105.613573)),
    "N-BAK2": (1.53996, 59.71, (1.01662154, 0.319903051, 0.937232995, 0.00592383763, 0.0203828415, 113.118417)),
    "N-LAF34": (1.7725, 49.62, (1.75836958, 0.313537785, 1.18925231, 0.00872810026, 0.0293020832, 85.1780644)),
    "N-LAK10": (1.72003, 50.62, (1.72878017, 0.169257825, 1.19386956, 0.00886014635, 0.0363416509, 82.9009069)),
    "N-LAK22": (1.65113, 55.89, (1.14229781, 0.535138441, 1.04088385, 0.00585778594, 0.0198546147, 100.834017)),
    "N-LAK8": (1.713, 53.83, (1.33183167, 0.546623206, 1.19084015, 0.00620023871, 0.0216465439, 82.5827736)),
    "N-SF15": (1.69892, 30.2, (1.57055634, 0.218987094, 1.50824017, 0.0116507014, 0.0597856897, 132.709339)),
    "N-SF6": (1.80518, 25.36, (1.77931763, 0.338149866, 2.08734474, 0.0133714182, 0.0617533621, 174.01759)),
}

# Largest |n_file - n_d(glass)| accepted when assigning a catalogue glass to a lens-file index.
MAX_ND_MISMATCH = 0.03


# Internal transmittance tau_i (bulk only, no surface reflection) of a 10 mm sample, wavelength nm ->
# tau_i, 300-1060 nm: SCHOTT optical glass overview (Excel, 2025-11-13, TAUI10 columns, as shipped in
# ``opticalglass`` 2.0.2).  Wavelengths below the shortest listed value are opaque in the catalogue.
SCHOTT_TAUI10: dict[str, dict[int, float]] = {
    "F2": {
        310: 0,
        320: 0.08,
        334: 0.537,
        350: 0.905,
        365: 0.968,
        370: 0.975,
        380: 0.985,
        390: 0.991,
        400: 0.994,
        405: 0.995,
        420: 0.996,
        436: 0.997,
        460: 0.998,
        500: 0.999,
        546: 0.999,
        580: 0.999,
        620: 0.999,
        660: 0.999,
        700: 0.999,
        1060: 0.999,
    },
    "F5": {
        310: 0,
        320: 0.08,
        334: 0.618,
        350: 0.896,
        365: 0.963,
        370: 0.971,
        380: 0.984,
        390: 0.989,
        400: 0.993,
        405: 0.994,
        420: 0.995,
        436: 0.996,
        460: 0.996,
        500: 0.998,
        546: 0.998,
        580: 0.998,
        620: 0.998,
        660: 0.998,
        700: 0.999,
        1060: 0.999,
    },
    "N-BAF10": {
        350: 0.176,
        365: 0.626,
        370: 0.72,
        380: 0.847,
        390: 0.915,
        400: 0.95,
        405: 0.959,
        420: 0.976,
        436: 0.981,
        460: 0.987,
        500: 0.992,
        546: 0.996,
        580: 0.996,
        620: 0.996,
        660: 0.996,
        700: 0.998,
        1060: 0.998,
    },
    "N-BAK2": {
        300: 0.398,
        310: 0.693,
        320: 0.867,
        334: 0.963,
        350: 0.9883,
        365: 0.9944,
        370: 0.9956,
        380: 0.9956,
        390: 0.997,
        400: 0.9968,
        405: 0.9968,
        420: 0.9968,
        436: 0.997,
        460: 0.997,
        500: 0.998,
        546: 0.9976,
        580: 0.9976,
        620: 0.998,
        660: 0.9976,
        700: 0.998,
        1060: 0.9988,
    },
    "N-LAF34": {
        300: 0.069,
        310: 0.236,
        320: 0.424,
        334: 0.643,
        350: 0.815,
        365: 0.908,
        370: 0.927,
        380: 0.955,
        390: 0.971,
        400: 0.98,
        405: 0.983,
        420: 0.988,
        436: 0.991,
        460: 0.994,
        500: 0.997,
        546: 0.998,
        580: 0.998,
        620: 0.998,
        660: 0.998,
        700: 0.998,
        1060: 0.999,
    },
    "N-LAK10": {
        334: 0.06,
        350: 0.504,
        365: 0.804,
        370: 0.861,
        380: 0.919,
        390: 0.948,
        400: 0.964,
        405: 0.97,
        420: 0.98,
        436: 0.988,
        460: 0.991,
        500: 0.995,
        546: 0.9976,
        580: 0.9972,
        620: 0.9976,
        660: 0.998,
        700: 0.998,
        1060: 0.998,
    },
    "N-LAK22": {
        300: 0.433,
        310: 0.6292,
        320: 0.771,
        334: 0.8865,
        350: 0.9471,
        365: 0.9733,
        370: 0.978,
        380: 0.9848,
        390: 0.9885,
        400: 0.9907,
        405: 0.9915,
        420: 0.9928,
        436: 0.9932,
        460: 0.9944,
        500: 0.9964,
        546: 0.998,
        580: 0.998,
        620: 0.998,
        660: 0.9978,
        700: 0.9984,
        1060: 0.999,
    },
    "N-LAK8": {
        300: 0.044,
        310: 0.137,
        320: 0.2759,
        334: 0.5092,
        350: 0.7393,
        365: 0.8769,
        370: 0.9054,
        380: 0.9458,
        390: 0.9651,
        400: 0.9768,
        405: 0.9805,
        420: 0.9879,
        436: 0.9915,
        460: 0.9948,
        500: 0.9976,
        546: 0.998,
        580: 0.9976,
        620: 0.9976,
        660: 0.998,
        700: 0.9984,
        1060: 0.9976,
    },
    "N-SF15": {
        365: 0.044,
        370: 0.158,
        380: 0.525,
        390: 0.746,
        400: 0.857,
        405: 0.887,
        420: 0.941,
        436: 0.964,
        460: 0.977,
        500: 0.988,
        546: 0.994,
        580: 0.994,
        620: 0.994,
        660: 0.993,
        700: 0.995,
        1060: 0.998,
    },
    "N-SF6": {
        365: 0.004,
        370: 0.1585,
        380: 0.4804,
        390: 0.7,
        400: 0.8206,
        405: 0.857,
        420: 0.9192,
        436: 0.9458,
        460: 0.9609,
        500: 0.9768,
        546: 0.9887,
        580: 0.992,
        620: 0.9911,
        660: 0.9903,
        700: 0.9932,
        1060: 0.9984,
    },
}


def sellmeier_n(coeffs: tuple[float, ...], lam_nm) -> np.ndarray:
    """Sellmeier index for ``coeffs = (B1, B2, B3, C1, C2, C3)`` at ``lam_nm`` (nm)."""
    l2 = (np.asarray(lam_nm, dtype=np.float64) * 1e-3) ** 2
    n2 = 1.0 + sum(b * l2 / (l2 - c) for b, c in zip(coeffs[:3], coeffs[3:]))
    return np.sqrt(n2)


def internal_transmittance(glass: str, lam_nm, thickness_mm) -> np.ndarray:
    """Bulk transmittance of ``thickness_mm`` of ``glass``: tau_10(lambda) ** (d / 10 mm) (Lambert-Beer).

    tau_10 is linear in wavelength between the catalogue points and held at the end values outside them.
    """
    tab = SCHOTT_TAUI10[glass]
    wl = np.array(sorted(tab), dtype=np.float64)
    t10 = np.interp(np.asarray(lam_nm, dtype=np.float64), wl, np.array([tab[k] for k in sorted(tab)]))
    return t10 ** (np.asarray(thickness_mm, dtype=np.float64) / 10.0)


@dataclass(frozen=True)
class GlassDispersion:
    """Dispersion of one lens-file glass row (see the module docstring)."""

    glass: str
    n_file: float
    nu_d: float
    coeffs: tuple[float, ...]

    @property
    def scale(self) -> float:
        dn_glass = float(sellmeier_n(self.coeffs, LAMBDA_F_NM) - sellmeier_n(self.coeffs, LAMBDA_C_NM))
        return (self.n_file - 1.0) / self.nu_d / dn_glass

    def n(self, lam_nm) -> np.ndarray:
        return self.n_file + self.scale * (sellmeier_n(self.coeffs, lam_nm) - sellmeier_n(self.coeffs, LAMBDA_D_NM))


def glass_map_path(lens_path: str | Path) -> Path:
    """Sidecar path: ``dgauss.50mm.dat`` -> ``dgauss.50mm.glass.yaml``."""
    return Path(lens_path).with_suffix(".glass.yaml")


def _lens_rows(lens_path: Path) -> list[tuple[float, float, float, float]]:
    from lens_prescription import load_lens_file

    return list(load_lens_file(lens_path))


def load_dispersion(lens_path: str | Path) -> list[GlassDispersion | None] | None:
    """Per lens-file row dispersion (None for air / stop rows), or None if the lens has no sidecar."""
    import yaml

    lens_path = Path(lens_path)
    side = glass_map_path(lens_path)
    if not side.is_file():
        return None
    cfg = yaml.safe_load(side.read_text()) or {}
    rows = _lens_rows(lens_path)
    spec = {int(k): v for k, v in (cfg.get("rows") or {}).items()}
    glass_rows = [i for i, r in enumerate(rows) if r[2] not in (0.0, 1.0)]
    if sorted(spec) != glass_rows:
        raise ValueError(
            f"{side}: rows {sorted(spec)} must list exactly the glass rows {glass_rows} of {lens_path.name}"
        )
    out: list[GlassDispersion | None] = [None] * len(rows)
    for i, entry in spec.items():
        name = str(entry["glass"])
        if name not in SCHOTT_SELLMEIER:
            raise ValueError(f"{side}: row {i}: unknown glass {name!r}; add it to lens_glass.SCHOTT_SELLMEIER")
        nd_g, vd_g, coeffs = SCHOTT_SELLMEIER[name]
        n_file = float(rows[i][2])
        if abs(n_file - nd_g) > MAX_ND_MISMATCH:
            raise ValueError(f"{side}: row {i}: n_file {n_file} too far from {name} n_d {nd_g}")
        out[i] = GlassDispersion(name, n_file, float(entry.get("nu_d", vd_g)), coeffs)
    return out


def refractive_indices(lens_path: str | Path, lam_nm: float) -> np.ndarray:
    """Index after each surface (lens-file order, air = 1) at ``lam_nm``; n_d values without a sidecar."""
    rows = _lens_rows(Path(lens_path))
    disp = load_dispersion(lens_path) or [None] * len(rows)
    return np.array(
        [1.0 if r[2] == 0.0 else (float(g.n(lam_nm)) if g is not None else float(r[2])) for r, g in zip(rows, disp)]
    )


def pbrt_dispersion_values(lens_path: str | Path) -> list[float] | None:
    """Seven floats per lens-file row (s, B1..B3, C1..C3; zeros = no dispersion) for pbrt."""
    disp = load_dispersion(lens_path)
    if disp is None:
        return None
    vals: list[float] = []
    for g in disp:
        vals += [0.0] * 7 if g is None else [g.scale, *g.coeffs]
    return vals


def pbrt_camera_lines(lens_path: str | Path, enabled: bool = True, split: bool = False) -> list[str]:
    """Lens sidecar lines for a ``Camera "realistic"`` block.

    ``"float dispersion"`` when ``enabled`` and a glass sidecar exists, plus ``"float asphere"``
    whenever an asphere sidecar exists (surface shape is geometry, not an option).  ``split`` adds
    ``"bool dispersionsplit" true``: pbrt traces every wavelength of a pixel sample as its own camera
    ray instead of the hero wavelength only (about 4x the render time, far less colour noise).
    """
    from lens_asphere import pbrt_camera_lines as asphere_lines

    vals = pbrt_dispersion_values(lens_path) if enabled else None
    lines = [f'    "float dispersion" [{" ".join(f"{v:.9g}" for v in vals)}]'] if vals else []
    if vals and split:
        lines.append('    "bool dispersionsplit" true')
    return lines + asphere_lines(lens_path)


def abbe_number(g: GlassDispersion) -> float:
    """nu_d actually realised by :meth:`GlassDispersion.n` (for tests / reports)."""
    nd, nf, nc = (float(g.n(lam)) for lam in (LAMBDA_D_NM, LAMBDA_F_NM, LAMBDA_C_NM))
    return (nd - 1.0) / (nf - nc) if not math.isclose(nf, nc) else math.inf
