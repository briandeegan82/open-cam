"""Shared radiometry helpers for spectral sensor forward models."""

from __future__ import annotations

import numpy as np

# SI constants
H_PLANCK = 6.62607015e-34  # J·s
C_LIGHT = 299792458.0  # m/s


def photon_flux_density_from_irradiance(
    spectral_irradiance_W_m2nm: np.ndarray,
    wavelength_nm: np.ndarray,
) -> np.ndarray:
    """Convert spectral irradiance E_e(λ) [W/(m²·nm)] to photon flux density [photons/(s·m²·nm)].

    Φ_p(λ) = E_e(λ) · λ / (h c) with λ in meters.
    """
    lam_m = np.asarray(wavelength_nm, dtype=np.float64) * 1e-9
    return spectral_irradiance_W_m2nm * lam_m / (H_PLANCK * C_LIGHT)


def spectral_electron_weights(
    wavelength_nm: np.ndarray,
    qe: np.ndarray,
    bin_width_nm: np.ndarray,
    irradiance_scale: np.ndarray | float,
    geometry_factor: float,
) -> np.ndarray:
    """K x C weights so that ``electrons = planes @ weights`` for HxWxK irradiance-unit planes.

    Folds the energy-to-photon conversion ``lambda / (h c)``, the per-bin QE, the
    integration bin widths, any per-wavelength irradiance scale and the
    pixel-area x integration-time x fill-factor geometry into one matrix, so the
    per-pixel integral is a single matrix product instead of materialising a
    float64 HxWxK photon-flux cube per channel.

    ``qe`` is K x C (one column per colour channel).
    """
    lam = np.asarray(wavelength_nm, dtype=np.float64)
    q = np.asarray(qe, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != lam.size:
        raise ValueError(f"qe must be K x C with K={lam.size}, got {q.shape}")
    per_bin = (
        np.asarray(bin_width_nm, dtype=np.float64)
        * np.asarray(irradiance_scale, dtype=np.float64)
        * photon_flux_density_from_irradiance(1.0, lam)
        * float(geometry_factor)
    )
    return q * per_bin[:, np.newaxis]


def integrate_spectral_planes(
    planes: np.ndarray,
    weights: np.ndarray,
    *,
    block_pixels: int = 1 << 18,
) -> np.ndarray:
    """``planes`` (HxWxK) @ ``weights`` (KxC) -> HxWxC float64, in pixel blocks.

    Blocking keeps the float64 working copy bounded regardless of image size.
    """
    h, w, k = planes.shape
    wts = np.asarray(weights, dtype=np.float64)
    if wts.shape[0] != k:
        raise ValueError(f"weights rows ({wts.shape[0]}) must match spectral planes ({k})")
    flat = planes.reshape(-1, k)
    out = np.empty((flat.shape[0], wts.shape[1]), dtype=np.float64)
    for start in range(0, flat.shape[0], block_pixels):
        stop = start + block_pixels
        out[start:stop] = flat[start:stop].astype(np.float64) @ wts
    return out.reshape(h, w, wts.shape[1])


def cosine_illuminance_factor(
    surface_normal: np.ndarray,
    direction_to_light_world: np.ndarray,
) -> float:
    """Lambert receiver: factor = max(0, n·omega) with omega unit vector toward the source."""
    n = np.asarray(surface_normal, dtype=np.float64)
    w = np.asarray(direction_to_light_world, dtype=np.float64)
    n = n / max(1e-15, np.linalg.norm(n))
    w = w / max(1e-15, np.linalg.norm(w))
    return float(max(0.0, np.dot(n, w)))


def cos4_vignetting_from_pinhole(
    xw: np.ndarray,
    yw: np.ndarray,
    cam_dist: float,
) -> np.ndarray:
    """Cos^4 vignetting vs chief ray angle for pinhole at (0,0,cam_dist) viewing z=0 plane."""
    r = np.sqrt(xw * xw + yw * yw + cam_dist * cam_dist)
    cos_t = np.abs(cam_dist) / np.maximum(1e-9, r)
    return cos_t**4


# =====================================================================
# Photometric exposure chain: scene luminance -> illuminance -> electrons
# =====================================================================
# Constants from the standard reflected-light metering model.
METER_CALIBRATION_K = 12.5  # reflected-light meter constant used by Canon/Nikon/Sekonic
DAYLIGHT_LUMINOUS_EFFICACY_LM_PER_W = 250.0  # luminous efficacy of daylight-ish radiation


def image_plane_illuminance_lux(
    scene_luminance_cd_m2: float | np.ndarray,
    f_number: float | np.ndarray,
    *,
    transmission: float = 0.9,
    relative_illumination: float = 1.0,
) -> float | np.ndarray:
    """Camera equation: ``E = pi/4 * T * L * RI / N^2`` [lux].

    Illuminance falls as the square of the f-number, which is the whole reason
    the f-number scale steps by sqrt(2): each stop halves the light.
    """
    N = np.asarray(f_number, dtype=np.float64)
    if np.any(N <= 0):
        raise ValueError("f_number must be positive")
    return (
        (np.pi / 4.0)
        * transmission
        * np.asarray(scene_luminance_cd_m2, dtype=np.float64)
        * relative_illumination
        / (N**2)
    )


def photons_per_second_per_pixel(
    illuminance_lux: float | np.ndarray,
    pixel_area_m2: float,
    *,
    luminous_efficacy_lm_per_W: float = DAYLIGHT_LUMINOUS_EFFICACY_LM_PER_W,
    wavelength_nm: float = 550.0,
) -> float | np.ndarray:
    """Photon arrival rate at one pixel [photons/s].

    Lux is a photometric unit -- it already has the eye's response folded in --
    so getting back to photons needs the luminous efficacy of the particular
    spectrum, not a universal constant.
    """
    if luminous_efficacy_lm_per_W <= 0:
        raise ValueError("luminous_efficacy_lm_per_W must be positive")
    irradiance_W_m2 = np.asarray(illuminance_lux, dtype=np.float64) / luminous_efficacy_lm_per_W
    photon_energy_J = H_PLANCK * C_LIGHT / (wavelength_nm * 1e-9)
    return irradiance_W_m2 * pixel_area_m2 / photon_energy_J


def electrons_from_exposure(
    scene_luminance_cd_m2: float | np.ndarray,
    f_number: float | np.ndarray,
    integration_time_s: float | np.ndarray,
    *,
    pixel_pitch_um: float,
    quantum_efficiency: float,
    fill_factor: float = 1.0,
    transmission: float = 0.9,
    relative_illumination: float = 1.0,
    luminous_efficacy_lm_per_W: float = DAYLIGHT_LUMINOUS_EFFICACY_LM_PER_W,
    wavelength_nm: float = 550.0,
) -> float | np.ndarray:
    """Mean signal electrons collected in one pixel over one integration.

    The full chain the exposure triangle rests on: scene luminance through the
    aperture to image-plane illuminance, into photons via the luminous efficacy,
    onto the pixel's collecting area, through QE, for a length of time.
    """
    lux = image_plane_illuminance_lux(
        scene_luminance_cd_m2, f_number, transmission=transmission, relative_illumination=relative_illumination
    )
    area_m2 = (pixel_pitch_um * 1e-6) ** 2 * fill_factor
    rate = photons_per_second_per_pixel(
        lux, area_m2, luminous_efficacy_lm_per_W=luminous_efficacy_lm_per_W, wavelength_nm=wavelength_nm
    )
    return rate * quantum_efficiency * np.asarray(integration_time_s, dtype=np.float64)


def exposure_value(f_number: float | np.ndarray, integration_time_s: float | np.ndarray) -> float | np.ndarray:
    """``EV = log2(N^2 / t)``: the camera-side half of the exposure triangle.

    Every (N, t) pair on one EV line delivers the same number of electrons, which
    is exactly what leaves photographers free to trade depth of field for motion
    blur at constant brightness.
    """
    N = np.asarray(f_number, dtype=np.float64)
    t = np.asarray(integration_time_s, dtype=np.float64)
    if np.any(N <= 0) or np.any(t <= 0):
        raise ValueError("f_number and integration_time_s must be positive")
    return np.log2(N**2 / t)


def ev100_from_luminance(
    scene_luminance_cd_m2: float | np.ndarray, *, calibration_K: float = METER_CALIBRATION_K
) -> float | np.ndarray:
    """Scene-side EV at ISO 100: ``EV100 = log2(L * 100 / K)``."""
    L = np.asarray(scene_luminance_cd_m2, dtype=np.float64)
    if np.any(L <= 0):
        raise ValueError("scene_luminance_cd_m2 must be positive")
    return np.log2(L * 100.0 / calibration_K)


def luminance_from_ev100(
    ev100: float | np.ndarray, *, calibration_K: float = METER_CALIBRATION_K
) -> float | np.ndarray:
    """Inverse of :func:`ev100_from_luminance`."""
    return calibration_K * np.power(2.0, np.asarray(ev100, dtype=np.float64)) / 100.0


def shutter_for_exposure_value(f_number: float | np.ndarray, ev: float | np.ndarray) -> float | np.ndarray:
    """Integration time that puts ``f_number`` on the given EV line."""
    N = np.asarray(f_number, dtype=np.float64)
    return N**2 / np.power(2.0, np.asarray(ev, dtype=np.float64))


def f_number_for_exposure_value(integration_time_s: float | np.ndarray, ev: float | np.ndarray) -> float | np.ndarray:
    """F-number that puts ``integration_time_s`` on the given EV line."""
    t = np.asarray(integration_time_s, dtype=np.float64)
    return np.sqrt(t * np.power(2.0, np.asarray(ev, dtype=np.float64)))
