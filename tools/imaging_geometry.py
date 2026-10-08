"""First-order (Gaussian) imaging geometry: what the lens does before it blurs.

``apply_spectral_psf.py`` models how a point spreads. This module models where
that point lands in the first place -- the thin-lens conjugate relation, field
of view, magnification, depth of field, natural cos^4 falloff and Brown-Conrady
distortion. It is deliberately free of PBRT and file I/O so it can be called
from a slider callback.

Sign and unit conventions
-------------------------
* All lengths are millimetres unless a name says otherwise.
* Object distance ``s_o`` is measured from the front principal plane and is
  positive for real objects; ``math.inf`` means focused at infinity.
* Magnification is signed: ``m < 0`` is the inverted real image a camera forms.
  Where only the scale matters (exposure, effective f-number) ``abs(m)`` is used.
* Distortion coefficients follow the same convention as the ``lens.distortion_*``
  keys in ``config/lens_models/*.yaml``: ``k1 < 0`` is barrel, ``k1 > 0`` is
  pincushion, and they act on aspect-corrected normalised image coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sensor_radiometry import cos4_vignetting_from_pinhole

# 36 x 24 mm reference format that "full frame" and crop factor are defined against.
FULL_FRAME_WIDTH_MM = 36.0
FULL_FRAME_HEIGHT_MM = 24.0
FULL_FRAME_DIAGONAL_MM = math.hypot(FULL_FRAME_WIDTH_MM, FULL_FRAME_HEIGHT_MM)

#: Zeiss/"d/1440" convention: the largest blur a print viewer will not resolve.
DEFAULT_COC_DIVISOR = 1440.0


# =====================================================================
# Sensor formats
# =====================================================================
@dataclass(frozen=True)
class SensorFormat:
    """A sensor's physical size. Pixel pitch comes from the camera recipe."""

    name: str
    width_mm: float
    height_mm: float

    @property
    def diagonal_mm(self) -> float:
        return math.hypot(self.width_mm, self.height_mm)

    @property
    def aspect_ratio(self) -> float:
        return self.width_mm / self.height_mm

    @property
    def crop_factor(self) -> float:
        return FULL_FRAME_DIAGONAL_MM / self.diagonal_mm

    def megapixels(self, pixel_pitch_um: float) -> float:
        pitch_mm = pixel_pitch_um / 1000.0
        if pitch_mm <= 0:
            return 0.0
        return (self.width_mm / pitch_mm) * (self.height_mm / pitch_mm) / 1e6


#: Formats that span the range the tutorials compare, smallest sensor first.
SENSOR_FORMATS: dict[str, SensorFormat] = {
    "phone_1_2_55": SensorFormat('Phone 1/2.55"', 5.76, 4.29),
    "type_1_inch": SensorFormat("1-inch (RX100 class)", 13.2, 8.8),
    "micro_four_thirds": SensorFormat("Micro Four Thirds", 17.3, 13.0),
    "aps_c": SensorFormat("APS-C", 23.6, 15.7),
    "full_frame": SensorFormat("Full frame (35 mm)", FULL_FRAME_WIDTH_MM, FULL_FRAME_HEIGHT_MM),
    "medium_format": SensorFormat("Medium format (44x33)", 44.0, 33.0),
}


def sensor_format(name: str) -> SensorFormat:
    try:
        return SENSOR_FORMATS[name]
    except KeyError:
        raise ValueError(f"unknown sensor format {name!r}; use one of {sorted(SENSOR_FORMATS)}") from None


def crop_factor(sensor_diagonal_mm: float) -> float:
    """Ratio of the full-frame diagonal to this sensor's diagonal."""
    if sensor_diagonal_mm <= 0:
        raise ValueError("sensor_diagonal_mm must be positive")
    return FULL_FRAME_DIAGONAL_MM / sensor_diagonal_mm


def equivalent_focal_length_mm(focal_length_mm: float, sensor_diagonal_mm: float) -> float:
    """The full-frame focal length giving the same field of view ("35 mm equivalent")."""
    return focal_length_mm * crop_factor(sensor_diagonal_mm)


# =====================================================================
# Thin lens
# =====================================================================
def thin_lens_image_distance_mm(focal_length_mm: float, object_distance_mm: float) -> float:
    """Solve ``1/f = 1/s_o + 1/s_i`` for the image distance.

    Returns ``f`` for an object at infinity, and ``math.inf`` for an object at
    the front focal point (the collimated case -- no image is formed).
    """
    if focal_length_mm <= 0:
        raise ValueError("focal_length_mm must be positive")
    if not math.isfinite(object_distance_mm):
        return focal_length_mm
    if object_distance_mm <= focal_length_mm:
        return math.inf
    return focal_length_mm * object_distance_mm / (object_distance_mm - focal_length_mm)


def magnification(focal_length_mm: float, object_distance_mm: float) -> float:
    """Signed transverse magnification ``m = -s_i / s_o`` (negative = inverted)."""
    if not math.isfinite(object_distance_mm):
        return 0.0
    if object_distance_mm <= focal_length_mm:
        return -math.inf
    return -focal_length_mm / (object_distance_mm - focal_length_mm)


def object_distance_for_magnification_mm(focal_length_mm: float, target_magnification: float) -> float:
    """Focus distance giving ``|m| = target_magnification`` (1.0 = life size)."""
    m = abs(target_magnification)
    if m <= 0:
        return math.inf
    return focal_length_mm * (1.0 + 1.0 / m)


def bellows_exposure_factor(magnification_value: float) -> float:
    """Light lost to close focus: ``(1 + |m|)^2``.

    At 1:1 macro this is 4x -- two stops -- which is why the image-generation
    pipeline applies the same ``(1+m)^2`` term when converting radiance to
    irradiance at close focus distances.
    """
    return (1.0 + abs(magnification_value)) ** 2


def effective_f_number(f_number: float, magnification_value: float) -> float:
    """Working f-number at finite focus: ``N_eff = N (1 + |m|)``."""
    return f_number * (1.0 + abs(magnification_value))


# =====================================================================
# Field of view
# =====================================================================
@dataclass(frozen=True)
class FieldOfView:
    horizontal_deg: float
    vertical_deg: float
    diagonal_deg: float
    image_distance_mm: float


def field_of_view(
    focal_length_mm: float,
    sensor_width_mm: float,
    sensor_height_mm: float,
    object_distance_mm: float = math.inf,
) -> FieldOfView:
    """Angular coverage of the sensor.

    Uses the image distance rather than the focal length, so the (real, slightly
    narrower) field at close focus falls out of the same expression that gives
    ``2 atan(d / 2f)`` at infinity.
    """
    s_i = thin_lens_image_distance_mm(focal_length_mm, object_distance_mm)
    if not math.isfinite(s_i):
        return FieldOfView(0.0, 0.0, 0.0, s_i)

    def half_angle(extent_mm: float) -> float:
        return math.degrees(2.0 * math.atan2(extent_mm / 2.0, s_i))

    return FieldOfView(
        horizontal_deg=half_angle(sensor_width_mm),
        vertical_deg=half_angle(sensor_height_mm),
        diagonal_deg=half_angle(math.hypot(sensor_width_mm, sensor_height_mm)),
        image_distance_mm=s_i,
    )


def subject_extent_mm(fov_deg: float, distance_mm: float) -> float:
    """How much of the world (in mm) the given angle covers at *distance_mm*."""
    if not math.isfinite(distance_mm):
        return math.inf
    return 2.0 * distance_mm * math.tan(math.radians(fov_deg) / 2.0)


# =====================================================================
# Depth of field
# =====================================================================
def circle_of_confusion_mm(sensor_diagonal_mm: float, divisor: float = DEFAULT_COC_DIVISOR) -> float:
    """Print-based CoC criterion: sensor diagonal / *divisor*."""
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    return sensor_diagonal_mm / divisor


def pixel_circle_of_confusion_mm(pixel_pitch_um: float, n_pixels: float = 2.0) -> float:
    """Pixel-based CoC criterion: *n_pixels* pitches.

    On a modern high-resolution sensor this is far stricter than the print
    criterion, which is why "sharp" depth of field shrinks when you pixel-peep.
    """
    return n_pixels * pixel_pitch_um / 1000.0


def hyperfocal_distance_mm(focal_length_mm: float, f_number: float, coc_mm: float) -> float:
    """``H = f^2 / (N c) + f`` -- focus here and everything past H/2 is acceptably sharp."""
    if f_number <= 0 or coc_mm <= 0:
        raise ValueError("f_number and coc_mm must be positive")
    return focal_length_mm**2 / (f_number * coc_mm) + focal_length_mm


@dataclass(frozen=True)
class DepthOfField:
    near_mm: float
    far_mm: float
    hyperfocal_mm: float
    focus_distance_mm: float

    @property
    def total_mm(self) -> float:
        return self.far_mm - self.near_mm

    @property
    def in_front_mm(self) -> float:
        return self.focus_distance_mm - self.near_mm

    @property
    def behind_mm(self) -> float:
        return self.far_mm - self.focus_distance_mm

    @property
    def is_infinite(self) -> bool:
        return not math.isfinite(self.far_mm)


def depth_of_field(
    focal_length_mm: float,
    f_number: float,
    focus_distance_mm: float,
    coc_mm: float,
) -> DepthOfField:
    """Near and far limits of acceptable sharpness around the focus distance.

    Once the focus distance reaches the hyperfocal distance the far limit runs
    to infinity, which is the whole point of hyperfocal focusing.
    """
    H = hyperfocal_distance_mm(focal_length_mm, f_number, coc_mm)
    s = focus_distance_mm
    f = focal_length_mm

    if not math.isfinite(s):
        return DepthOfField(near_mm=H, far_mm=math.inf, hyperfocal_mm=H, focus_distance_mm=s)

    near = s * (H - f) / (H + s - 2.0 * f)
    far = math.inf if s >= H else s * (H - f) / (H - s)
    return DepthOfField(near_mm=near, far_mm=far, hyperfocal_mm=H, focus_distance_mm=s)


def defocus_blur_diameter_mm(
    focal_length_mm: float,
    f_number: float,
    focus_distance_mm: float,
    object_distance_mm: float,
) -> float:
    """Diameter of the blur disc an out-of-focus point projects onto the sensor.

    This is what the depth-of-field limits are a threshold on: the DoF near/far
    distances are exactly where this equals the circle of confusion.
    """
    if not math.isfinite(object_distance_mm):
        object_distance_mm = 1e12
    if not math.isfinite(focus_distance_mm):
        focus_distance_mm = 1e12
    f = focal_length_mm
    aperture = f / f_number
    denom = object_distance_mm * (focus_distance_mm - f)
    if denom == 0:
        return math.inf
    return abs(aperture * f * (focus_distance_mm - object_distance_mm) / denom)


def airy_disk_diameter_mm(f_number: float, wavelength_nm: float = 550.0) -> float:
    """Diffraction spot diameter ``2.44 N lambda``.

    Stopping down shrinks the defocus blur but grows this, so the two curves
    cross at an optimum aperture -- the reason "just stop down more" stops
    working past roughly f/11 on a full-frame sensor.
    """
    return 2.44 * f_number * wavelength_nm / 1e6


def diffraction_limited_f_number(coc_mm: float, wavelength_nm: float = 550.0) -> float:
    """The f-number at which the Airy disk alone fills the circle of confusion."""
    return coc_mm * 1e6 / (2.44 * wavelength_nm)


# =====================================================================
# Natural (cos^4) vignetting
# =====================================================================
def relative_illumination_cos4(image_height_mm: np.ndarray | float, image_distance_mm: float) -> np.ndarray:
    """cos^4 falloff vs distance from the optical axis on the sensor.

    Delegates to :func:`sensor_radiometry.cos4_vignetting_from_pinhole`, the
    same function the analytic sensor-forward path uses, by treating the image
    height as an off-axis offset seen from the exit pupil.
    """
    r = np.abs(np.asarray(image_height_mm, dtype=np.float64))
    return cos4_vignetting_from_pinhole(r, np.zeros_like(r), float(image_distance_mm))


def relative_illumination_profile(
    focal_length_mm: float,
    sensor_diagonal_mm: float,
    object_distance_mm: float = math.inf,
    n_samples: int = 129,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample cos^4 relative illumination from the centre to the frame corner.

    Returns ``(image_height_mm, relative_illumination)`` where 1.0 is the
    on-axis value. Wide lenses on large sensors reach a large chief-ray angle
    at the corner and lose over a stop before any mechanical vignetting.
    """
    s_i = thin_lens_image_distance_mm(focal_length_mm, object_distance_mm)
    r = np.linspace(0.0, sensor_diagonal_mm / 2.0, int(n_samples))
    return r, relative_illumination_cos4(r, s_i)


def corner_falloff_stops(focal_length_mm: float, sensor_diagonal_mm: float) -> float:
    """Corner light loss in photographic stops (``-log2`` of relative illumination)."""
    _, ri = relative_illumination_profile(focal_length_mm, sensor_diagonal_mm, n_samples=2)
    corner = float(ri[-1])
    if corner <= 0:
        return math.inf
    return -math.log2(corner)


# =====================================================================
# Brown-Conrady distortion
# =====================================================================
def brown_conrady_distort(
    xn: np.ndarray, yn: np.ndarray, k1: float, k2: float, p1: float = 0.0, p2: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Forward model: ideal normalised coordinates -> where the lens actually puts them."""
    xu = np.asarray(xn, dtype=np.float64)
    yu = np.asarray(yn, dtype=np.float64)
    r2 = xu * xu + yu * yu
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    xd = xu * radial + (2.0 * p1 * xu * yu + p2 * (r2 + 2.0 * xu * xu))
    yd = yu * radial + (p1 * (r2 + 2.0 * yu * yu) + 2.0 * p2 * xu * yu)
    return xd, yd


def brown_conrady_undistort(
    xn: np.ndarray,
    yn: np.ndarray,
    k1: float,
    k2: float,
    p1: float = 0.0,
    p2: float = 0.0,
    iterations: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse model by fixed-point iteration.

    This is the identical scheme ``spectral_sensor_forward.py`` runs to map
    distorted pixel coordinates back to undistorted ray directions, so the demo
    and the pipeline cannot disagree about what a coefficient means.
    """
    xd = np.asarray(xn, dtype=np.float64)
    yd = np.asarray(yn, dtype=np.float64)
    xu, yu = xd.copy(), yd.copy()
    for _ in range(int(iterations)):
        r2 = xu**2 + yu**2
        radial = 1.0 + k1 * r2 + k2 * r2**2
        xu = (xd - (2.0 * p1 * xu * yu + p2 * (r2 + 2.0 * xu**2))) / radial
        yu = (yd - (p1 * (r2 + 2.0 * yu**2) + 2.0 * p2 * xu * yu)) / radial
    return xu, yu


def distortion_grid(
    k1: float,
    k2: float,
    p1: float = 0.0,
    p2: float = 0.0,
    n_lines: int = 9,
    n_samples: int = 41,
    aspect_ratio: float = 1.5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """A straight-line grid pushed through the forward distortion model.

    Returns one ``(x, y)`` polyline per grid line in normalised image
    coordinates. Straight lines bowing outward is barrel, inward is pincushion --
    the classic way distortion is shown because only straight lines make it
    obvious.
    """
    half_x = aspect_ratio / math.hypot(aspect_ratio, 1.0)
    half_y = 1.0 / math.hypot(aspect_ratio, 1.0)
    lines: list[tuple[np.ndarray, np.ndarray]] = []

    for y0 in np.linspace(-half_y, half_y, n_lines):
        xs = np.linspace(-half_x, half_x, n_samples)
        lines.append(brown_conrady_distort(xs, np.full_like(xs, y0), k1, k2, p1, p2))
    for x0 in np.linspace(-half_x, half_x, n_lines):
        ys = np.linspace(-half_y, half_y, n_samples)
        lines.append(brown_conrady_distort(np.full_like(ys, x0), ys, k1, k2, p1, p2))
    return lines


def radial_distortion_percent(k1: float, k2: float, radius: float = 1.0) -> float:
    """Signed distortion at *radius*, in percent of the undistorted height.

    Negative is barrel, positive is pincushion. The standard SMIA/TV figure is
    quoted at the corner, i.e. ``radius = 1``.
    """
    r2 = radius * radius
    return 100.0 * (k1 * r2 + k2 * r2 * r2)


def distortion_coefficients(lens_cfg: dict) -> tuple[float, float, float, float]:
    """Read ``distortion_k1/k2/p1/p2`` out of a camera model's ``lens`` block."""
    cfg = lens_cfg or {}
    return (
        float(cfg.get("distortion_k1", 0.0) or 0.0),
        float(cfg.get("distortion_k2", 0.0) or 0.0),
        float(cfg.get("distortion_p1", 0.0) or 0.0),
        float(cfg.get("distortion_p2", 0.0) or 0.0),
    )
