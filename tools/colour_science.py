"""Colour science shared by the ISP pipeline, the validators and the teaching demos.

This module owns the pieces that were previously either missing or locked up
inside a single script: the CIE 1931 observer, XYZ/sRGB/Lab conversions, colour
difference, and -- most importantly -- the spectral integration that turns a
reflectance, an illuminant and a set of QE curves into a colour.

That integration is the whole reason camera colour is hard. The eye and the
sensor integrate the *same* spectrum against *different* curves, so two spectra
the eye cannot tell apart can land on different sensor RGB values, and vice
versa. Having both integrals in one place makes that comparison exact rather
than rhetorical.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# NumPy compatibility (np.trapezoid in newer versions, np.trapz in older).
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# =====================================================================
# CIE 1931 2-degree standard observer, 5 nm steps 380-780 nm
# =====================================================================
CMF_WAVELENGTH_NM = np.arange(380, 781, 5, dtype=np.float64)

CMF_X = np.array(
    [
        0.001368, 0.002236, 0.004243, 0.007650, 0.014310, 0.023120, 0.043510, 0.077630,
        0.134380, 0.214770, 0.283900, 0.328500, 0.348280, 0.348060, 0.336200, 0.318700,
        0.290800, 0.251100, 0.195360, 0.142100, 0.095640, 0.057950, 0.032010, 0.014700,
        0.004900, 0.002400, 0.009300, 0.029100, 0.063270, 0.109600, 0.165500, 0.225750,
        0.290400, 0.359700, 0.433450, 0.512050, 0.594500, 0.678400, 0.762100, 0.842500,
        0.916300, 0.978600, 1.026300, 1.056700, 1.062200, 1.045600, 1.002600, 0.938400,
        0.854450, 0.751400, 0.642400, 0.541900, 0.447900, 0.360800, 0.283500, 0.218700,
        0.164900, 0.121200, 0.087400, 0.063600, 0.046770, 0.032900, 0.022700, 0.015840,
        0.011359, 0.008111, 0.005790, 0.004109, 0.002899, 0.002049, 0.001440, 0.001000,
        0.000690, 0.000476, 0.000332, 0.000235, 0.000166, 0.000117, 0.000083, 0.000059,
        0.000042,
    ],
    dtype=np.float64,
)

CMF_Y = np.array(
    [
        0.000039, 0.000064, 0.000120, 0.000217, 0.000396, 0.000640, 0.001210, 0.002180,
        0.004000, 0.007300, 0.011600, 0.016840, 0.023000, 0.029800, 0.038000, 0.048000,
        0.060000, 0.073900, 0.090980, 0.112600, 0.139020, 0.169300, 0.208020, 0.258600,
        0.323000, 0.407300, 0.503000, 0.608200, 0.710000, 0.793200, 0.862000, 0.914850,
        0.954000, 0.980300, 0.995000, 1.000000, 0.995000, 0.978600, 0.952000, 0.915400,
        0.870000, 0.816300, 0.757000, 0.694900, 0.631000, 0.566800, 0.503000, 0.441200,
        0.381000, 0.321000, 0.265000, 0.217000, 0.175000, 0.138200, 0.107000, 0.081600,
        0.061000, 0.044580, 0.032000, 0.023200, 0.017000, 0.011920, 0.008210, 0.005723,
        0.004102, 0.002929, 0.002091, 0.001484, 0.001047, 0.000740, 0.000520, 0.000361,
        0.000249, 0.000172, 0.000120, 0.000083, 0.000057, 0.000039, 0.000027, 0.000018,
        0.000012,
    ],
    dtype=np.float64,
)

CMF_Z = np.array(
    [
        0.006450, 0.010550, 0.020050, 0.036210, 0.067850, 0.110200, 0.207400, 0.371300,
        0.645600, 1.039050, 1.385600, 1.622960, 1.747060, 1.782600, 1.772110, 1.744100,
        1.669200, 1.528100, 1.287640, 1.041900, 0.812950, 0.616200, 0.465180, 0.353300,
        0.272000, 0.212300, 0.158200, 0.111700, 0.078250, 0.057250, 0.042160, 0.029840,
        0.020300, 0.013400, 0.008750, 0.005750, 0.003900, 0.002750, 0.002100, 0.001800,
        0.001650, 0.001400, 0.001100, 0.001000, 0.001800, 0.002900, 0.004900, 0.007400,
        0.009300, 0.008800, 0.007700, 0.005900, 0.004500, 0.003400, 0.002400, 0.001800,
        0.001400, 0.001100, 0.001000, 0.001000, 0.001000, 0.001000, 0.001000, 0.001000,
        0.001000, 0.001000, 0.001000, 0.001000, 0.001000, 0.001000, 0.001000, 0.001000,
        0.001000, 0.001000, 0.001000, 0.001000, 0.001000, 0.001000, 0.001000, 0.001000,
        0.001000,
    ],
    dtype=np.float64,
)


def cmf_on_grid(wavelength_nm: np.ndarray) -> np.ndarray:
    """The observer resampled onto *wavelength_nm*, shape ``(3, K)``."""
    wl = np.asarray(wavelength_nm, dtype=np.float64)
    return np.stack([
        np.interp(wl, CMF_WAVELENGTH_NM, CMF_X, left=0.0, right=0.0),
        np.interp(wl, CMF_WAVELENGTH_NM, CMF_Y, left=0.0, right=0.0),
        np.interp(wl, CMF_WAVELENGTH_NM, CMF_Z, left=0.0, right=0.0),
    ])


def tristimulus(wavelength_nm: np.ndarray, spd: np.ndarray) -> tuple[float, float, float]:
    """Integrate a single SPD against the observer: ``(X, Y, Z)``."""
    xyz = integrate_spectra(wavelength_nm, np.atleast_2d(spd), cmf_on_grid(wavelength_nm))
    return float(xyz[0, 0]), float(xyz[0, 1]), float(xyz[0, 2])


# =====================================================================
# Spectral integration
# =====================================================================
def integrate_spectra(wavelength_nm: np.ndarray, spectra: np.ndarray,
                      responses: np.ndarray) -> np.ndarray:
    """Integrate each spectrum against each response curve.

    ``spectra`` is ``(N, K)`` and ``responses`` is ``(C, K)`` on the shared
    wavelength grid; the result is ``(N, C)``.
    """
    wl = np.asarray(wavelength_nm, dtype=np.float64)
    s = np.atleast_2d(np.asarray(spectra, dtype=np.float64))
    r = np.atleast_2d(np.asarray(responses, dtype=np.float64))
    if s.shape[1] != wl.size or r.shape[1] != wl.size:
        raise ValueError(
            f"spectra {s.shape} and responses {r.shape} must both be sampled on the "
            f"{wl.size}-point wavelength grid")
    return _trapz(s[:, None, :] * r[None, :, :], wl, axis=2)


def radiance_spectra(reflectance: np.ndarray, illuminant: np.ndarray) -> np.ndarray:
    """What actually reaches the camera: reflectance times the illuminant SPD.

    Everything downstream -- the eye's XYZ and the sensor's RGB alike --
    integrates *this*, not the reflectance. Which is why changing the light
    changes the colour even though the surface has not moved.
    """
    refl = np.atleast_2d(np.asarray(reflectance, dtype=np.float64))
    return refl * np.asarray(illuminant, dtype=np.float64)[None, :]


def xyz_from_spectra(wavelength_nm: np.ndarray, reflectance: np.ndarray,
                     illuminant: np.ndarray, *, normalise: bool = True) -> np.ndarray:
    """``(N, 3)`` CIE XYZ for each reflectance under *illuminant*.

    With ``normalise`` the scale is set so a perfect white diffuser gives
    ``Y = 1``, which is the convention Lab and the sRGB matrices expect.
    """
    cmf = cmf_on_grid(wavelength_nm)
    xyz = integrate_spectra(wavelength_nm, radiance_spectra(reflectance, illuminant), cmf)
    if normalise:
        white_y = integrate_spectra(wavelength_nm, np.atleast_2d(illuminant), cmf)[0, 1]
        xyz = xyz / max(float(white_y), 1e-12)
    return xyz


def camera_rgb_from_spectra(wavelength_nm: np.ndarray, reflectance: np.ndarray,
                            illuminant: np.ndarray, qe_rgb: np.ndarray,
                            *, normalise: bool = True) -> np.ndarray:
    """``(N, 3)`` raw camera RGB for each reflectance under *illuminant*.

    The same integral as :func:`xyz_from_spectra` with the observer swapped for
    the sensor's QE curves. Comparing the two is the entire colour problem: if
    the QE curves were a linear transform of the CMFs the camera would be
    colorimetric and a 3x3 CCM would be exact. They are not, so it is not.
    """
    rgb = integrate_spectra(wavelength_nm, radiance_spectra(reflectance, illuminant), qe_rgb)
    if normalise:
        white = integrate_spectra(wavelength_nm, np.atleast_2d(illuminant), qe_rgb)[0]
        rgb = rgb / np.maximum(white, 1e-12)[None, :]
    return rgb


def luther_condition_error(wavelength_nm: np.ndarray, qe_rgb: np.ndarray) -> float:
    """How far the QE curves are from being a linear transform of the observer.

    Zero means the Luther-Ives condition holds and the camera is colorimetric:
    some fixed 3x3 maps its RGB to XYZ exactly, for every possible spectrum.
    Real sensors return a few percent, which is the irreducible error a CCM
    cannot fit away and the reason cameras disagree about difficult colours.
    """
    cmf = cmf_on_grid(wavelength_nm)
    qe = np.atleast_2d(np.asarray(qe_rgb, dtype=np.float64))
    # Best linear map from QE curves to CMFs, evaluated per wavelength.
    fit, *_ = np.linalg.lstsq(qe.T, cmf.T, rcond=None)
    residual = cmf.T - qe.T @ fit
    return float(np.linalg.norm(residual) / max(np.linalg.norm(cmf.T), 1e-12))


# =====================================================================
# XYZ / sRGB / Lab
# =====================================================================
#: Linear sRGB primaries, D65 white (IEC 61966-2-1).
XYZ_FROM_SRGB = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)

SRGB_FROM_XYZ = np.linalg.inv(XYZ_FROM_SRGB)

#: D65 white point, normalised to Y = 1.
WHITE_D65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)

#: Bradford cone response, for chromatic adaptation between white points.
_BRADFORD = np.array([
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
], dtype=np.float64)


def xyz_to_srgb_linear(xyz: np.ndarray) -> np.ndarray:
    """XYZ to linear sRGB. Values outside [0, 1] are out of gamut, not errors."""
    return np.asarray(xyz, dtype=np.float64) @ SRGB_FROM_XYZ.T


def srgb_linear_to_xyz(rgb: np.ndarray) -> np.ndarray:
    return np.asarray(rgb, dtype=np.float64) @ XYZ_FROM_SRGB.T


def chromatic_adaptation_matrix(source_white: np.ndarray,
                                target_white: np.ndarray) -> np.ndarray:
    """Bradford adaptation from one white point to another.

    This is what "white balance done properly" means: not a per-channel gain in
    camera space, but a move between illuminants in cone space.
    """
    src = _BRADFORD @ np.asarray(source_white, dtype=np.float64)
    dst = _BRADFORD @ np.asarray(target_white, dtype=np.float64)
    return np.linalg.inv(_BRADFORD) @ np.diag(dst / np.maximum(src, 1e-12)) @ _BRADFORD


def white_point_xyz(wavelength_nm: np.ndarray, illuminant: np.ndarray) -> np.ndarray:
    """The illuminant's own XYZ, normalised to ``Y = 1``."""
    x, y, z = tristimulus(wavelength_nm, illuminant)
    return np.array([x, y, z], dtype=np.float64) / max(y, 1e-12)


def xy_chromaticity(xyz: np.ndarray) -> np.ndarray:
    """CIE xy, dropping luminance so colours can be compared on one diagram."""
    a = np.atleast_2d(np.asarray(xyz, dtype=np.float64))
    total = np.maximum(a.sum(axis=1, keepdims=True), 1e-12)
    return (a[:, :2] / total)


def correlated_colour_temperature(xy: np.ndarray) -> float:
    """McCamy's cubic approximation to CCT, in kelvin."""
    x, y = float(np.asarray(xy).ravel()[0]), float(np.asarray(xy).ravel()[1])
    denominator = 0.1858 - y
    if abs(denominator) < 1e-12:
        return float("nan")
    n = (x - 0.3320) / denominator
    return 449.0 * n ** 3 + 3525.0 * n ** 2 + 6823.3 * n + 5520.33


def xyz_to_lab(xyz: np.ndarray, white: np.ndarray = WHITE_D65) -> np.ndarray:
    """CIE L*a*b*, the space in which equal distances look equally different."""
    ratio = np.atleast_2d(np.asarray(xyz, dtype=np.float64)) / np.asarray(white, dtype=np.float64)
    eps, kappa = 216.0 / 24389.0, 24389.0 / 27.0
    f = np.where(ratio > eps, np.cbrt(np.maximum(ratio, 0.0)), (kappa * ratio + 16.0) / 116.0)
    return np.stack([
        116.0 * f[:, 1] - 16.0,
        500.0 * (f[:, 0] - f[:, 1]),
        200.0 * (f[:, 1] - f[:, 2]),
    ], axis=1)


def delta_e_76(lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
    """Plain Euclidean distance in Lab. Simple, and overstates hue errors."""
    a = np.atleast_2d(np.asarray(lab_a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(lab_b, dtype=np.float64))
    return np.sqrt(((a - b) ** 2).sum(axis=1))


def delta_e_2000(lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
    """CIEDE2000 colour difference -- the metric camera reviews quote.

    Roughly: 1 is a just-noticeable difference on a hard edge, under 2 is good
    camera colour, and over 5 is obvious side by side.
    """
    a = np.atleast_2d(np.asarray(lab_a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(lab_b, dtype=np.float64))
    L1, a1, b1 = a[:, 0], a[:, 1], a[:, 2]
    L2, a2, b2 = b[:, 0], b[:, 1], b[:, 2]

    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    C_bar = 0.5 * (C1 + C2)
    G = 0.5 * (1.0 - np.sqrt(C_bar ** 7 / (C_bar ** 7 + 25.0 ** 7)))
    a1p, a2p = (1.0 + G) * a1, (1.0 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180.0, dhp - 360.0, np.where(dhp < -180.0, dhp + 360.0, dhp))
    dhp = np.where(C1p * C2p == 0.0, 0.0, dhp)
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))

    Lp_bar = 0.5 * (L1 + L2)
    Cp_bar = 0.5 * (C1p + C2p)
    h_sum, h_diff = h1p + h2p, np.abs(h1p - h2p)
    hp_bar = np.where(
        C1p * C2p == 0.0, h_sum,
        np.where(h_diff <= 180.0, 0.5 * h_sum,
                 np.where(h_sum < 360.0, 0.5 * (h_sum + 360.0), 0.5 * (h_sum - 360.0))))

    T = (1.0
         - 0.17 * np.cos(np.radians(hp_bar - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * hp_bar))
         + 0.32 * np.cos(np.radians(3.0 * hp_bar + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * hp_bar - 63.0)))

    S_L = 1.0 + (0.015 * (Lp_bar - 50.0) ** 2) / np.sqrt(20.0 + (Lp_bar - 50.0) ** 2)
    S_C = 1.0 + 0.045 * Cp_bar
    S_H = 1.0 + 0.015 * Cp_bar * T

    d_theta = 30.0 * np.exp(-(((hp_bar - 275.0) / 25.0) ** 2))
    R_C = 2.0 * np.sqrt(Cp_bar ** 7 / (Cp_bar ** 7 + 25.0 ** 7))
    R_T = -R_C * np.sin(np.radians(2.0 * d_theta))

    return np.sqrt(
        (dLp / S_L) ** 2
        + (dCp / S_C) ** 2
        + (dHp / S_H) ** 2
        + R_T * (dCp / S_C) * (dHp / S_H)
    )


# =====================================================================
# Data loading
# =====================================================================
COLORCHECKER_PATCHES = 24
#: The neutral ladder, white through black -- patches 19-24 on the bottom row.
COLORCHECKER_NEUTRAL_SLICE = slice(18, 24)


@dataclass(frozen=True)
class SpectralChart:
    """Reflectance spectra for a chart, on a shared wavelength grid."""

    wavelength_nm: np.ndarray
    reflectance: np.ndarray       # (N, K)
    names: tuple[str, ...]


def _read_csv_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", dtype=np.float64)
    return data[:, 0], data[:, 1]


def load_colorchecker(repo_root: Path, wavelength_nm: np.ndarray | None = None) -> SpectralChart:
    """The 24 X-Rite ColorChecker patches from ``spectra/xrite/``."""
    files = sorted((Path(repo_root) / "spectra" / "xrite").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no ColorChecker reflectance CSVs under {repo_root}/spectra/xrite")

    curves = [_read_csv_curve(p) for p in files]
    grid = np.asarray(wavelength_nm, dtype=np.float64) if wavelength_nm is not None else curves[0][0]
    refl = np.stack([np.interp(grid, wl, val) for wl, val in curves])
    names = tuple(p.stem for p in files)
    return SpectralChart(wavelength_nm=grid, reflectance=refl, names=names)


def load_illuminant(repo_root: Path, illuminant_id: str,
                    wavelength_nm: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """An illuminant SPD from ``spectra/illuminant/interpolated/``."""
    path = Path(repo_root) / "spectra" / "illuminant" / "interpolated" / f"{illuminant_id}.csv"
    if not path.is_file():
        available = sorted(p.stem for p in path.parent.glob("*.csv"))
        raise FileNotFoundError(
            f"unknown illuminant {illuminant_id!r}; available: {', '.join(available)}")
    wl, spd = _read_csv_curve(path)
    if wavelength_nm is None:
        return wl, spd
    grid = np.asarray(wavelength_nm, dtype=np.float64)
    return grid, np.interp(grid, wl, spd)


def list_illuminant_ids(repo_root: Path) -> list[str]:
    d = Path(repo_root) / "spectra" / "illuminant" / "interpolated"
    return sorted(p.stem for p in d.glob("*.csv"))


def load_qe_rgb(repo_root: Path, paths: dict[str, str],
                wavelength_nm: np.ndarray) -> np.ndarray:
    """QE curves resampled onto a shared grid, shape ``(3, K)``.

    ``paths`` maps ``"red"``/``"green"``/``"blue"`` (and optionally ``"ircf"``)
    to repo-relative CSVs, matching the ``sensor.quantum_efficiency`` block of a
    camera YAML. The IR cut filter multiplies every channel.
    """
    grid = np.asarray(wavelength_nm, dtype=np.float64)
    root = Path(repo_root)

    def curve(key: str) -> np.ndarray:
        wl, val = _read_csv_curve(root / paths[key])
        return np.interp(grid, wl, val, left=0.0, right=0.0)

    qe = np.stack([curve("red"), curve("green"), curve("blue")])
    if paths.get("ircf"):
        qe = qe * curve("ircf")[None, :]
    return qe
