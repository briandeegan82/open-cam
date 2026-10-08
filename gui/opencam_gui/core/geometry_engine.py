"""Imaging-geometry adapter over ``tools/imaging_geometry.py``.

All the optics lives in that module. This file only assembles its outputs into
the arrays the demo plots: construction-ray polylines, sweeps of depth of field
against focus distance, blur against aperture, and so on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from opencam_gui.core.repo import import_tool


def _geom():
    return import_tool("imaging_geometry")


def sensor_formats() -> dict:
    """``{format_id: SensorFormat}`` in ascending size order."""
    return dict(_geom().SENSOR_FORMATS)


def sensor_format(name: str):
    return _geom().sensor_format(name)


# =====================================================================
# Numeric summary
# =====================================================================
@dataclass(frozen=True)
class GeometrySummary:
    focal_length_mm: float
    f_number: float
    focus_distance_mm: float
    format_name: str
    sensor_width_mm: float
    sensor_height_mm: float
    sensor_diagonal_mm: float
    crop_factor: float
    equivalent_focal_length_mm: float
    image_distance_mm: float
    magnification: float
    bellows_factor: float
    effective_f_number: float
    fov_horizontal_deg: float
    fov_vertical_deg: float
    fov_diagonal_deg: float
    subject_width_mm: float
    subject_height_mm: float
    coc_mm: float
    pixel_coc_mm: float
    hyperfocal_mm: float
    dof_near_mm: float
    dof_far_mm: float
    dof_total_mm: float
    corner_falloff_stops: float
    distortion_percent: float
    megapixels: float
    aperture_diameter_mm: float


def summarize(
    *,
    focal_length_mm: float,
    f_number: float,
    focus_distance_mm: float,
    format_id: str,
    pixel_pitch_um: float,
    use_pixel_coc: bool = False,
    k1: float = 0.0,
    k2: float = 0.0,
) -> GeometrySummary:
    g = _geom()
    fmt = g.sensor_format(format_id)
    fov = g.field_of_view(focal_length_mm, fmt.width_mm, fmt.height_mm, focus_distance_mm)
    m = g.magnification(focal_length_mm, focus_distance_mm)

    coc_print = g.circle_of_confusion_mm(fmt.diagonal_mm)
    coc_pixel = g.pixel_circle_of_confusion_mm(pixel_pitch_um)
    coc = coc_pixel if use_pixel_coc else coc_print
    dof = g.depth_of_field(focal_length_mm, f_number, focus_distance_mm, coc)

    return GeometrySummary(
        focal_length_mm=focal_length_mm,
        f_number=f_number,
        focus_distance_mm=focus_distance_mm,
        format_name=fmt.name,
        sensor_width_mm=fmt.width_mm,
        sensor_height_mm=fmt.height_mm,
        sensor_diagonal_mm=fmt.diagonal_mm,
        crop_factor=fmt.crop_factor,
        equivalent_focal_length_mm=g.equivalent_focal_length_mm(focal_length_mm, fmt.diagonal_mm),
        image_distance_mm=fov.image_distance_mm,
        magnification=m,
        bellows_factor=g.bellows_exposure_factor(m),
        effective_f_number=g.effective_f_number(f_number, m),
        fov_horizontal_deg=fov.horizontal_deg,
        fov_vertical_deg=fov.vertical_deg,
        fov_diagonal_deg=fov.diagonal_deg,
        subject_width_mm=g.subject_extent_mm(fov.horizontal_deg, focus_distance_mm),
        subject_height_mm=g.subject_extent_mm(fov.vertical_deg, focus_distance_mm),
        coc_mm=coc,
        pixel_coc_mm=coc_pixel,
        hyperfocal_mm=dof.hyperfocal_mm,
        dof_near_mm=dof.near_mm,
        dof_far_mm=dof.far_mm,
        dof_total_mm=dof.total_mm,
        corner_falloff_stops=g.corner_falloff_stops(focal_length_mm, fmt.diagonal_mm),
        distortion_percent=g.radial_distortion_percent(k1, k2),
        megapixels=fmt.megapixels(pixel_pitch_um),
        aperture_diameter_mm=focal_length_mm / f_number,
    )


# =====================================================================
# Thin-lens ray diagram
# =====================================================================
@dataclass(frozen=True)
class RayDiagram:
    """Polylines for a schematic thin-lens construction.

    The object and image sides are scaled independently (a portrait subject is
    60x further from the lens than the sensor is, so a true-to-scale drawing
    would put the image plane inside the lens glyph). Heights are normalised the
    same way. The construction stays geometrically honest because every ray has
    vertices only at the object plane, the lens plane, the focal points and the
    image plane, and a piecewise-linear axis keeps straight segments straight --
    but read the magnification off the numeric summary, not the picture.
    """

    rays: list[tuple[list[float], list[float]]]
    object_arrow: tuple[list[float], list[float]]
    image_arrow: tuple[list[float], list[float]]
    lens_outline: tuple[list[float], list[float]]
    focal_points: tuple[list[float], list[float]]
    x_limits: tuple[float, float]
    y_limits: tuple[float, float]
    image_is_virtual: bool


_OBJECT_SPAN = 1.0
_IMAGE_SPAN = 0.75


def ray_diagram(
    *, focal_length_mm: float, object_distance_mm: float, object_height_mm: float | None = None
) -> RayDiagram:
    g = _geom()
    f = float(focal_length_mm)
    s_o = float(object_distance_mm)
    s_i = g.thin_lens_image_distance_mm(f, s_o)
    m = g.magnification(f, s_o)

    virtual = not math.isfinite(s_i)
    if virtual:
        # No real image; show the object side only rather than drawing nonsense.
        s_i = 4.0 * f
        m = -1.0

    h_o = float(object_height_mm) if object_height_mm else s_o / 8.0
    h_i = m * h_o
    h_norm = max(abs(h_o), abs(h_i), 1e-9)
    y_o = h_o / h_norm
    y_i = h_i / h_norm

    def ox(x_mm: float) -> float:
        """Object-side mm (measured toward -x) -> display x."""
        return -_OBJECT_SPAN * (x_mm / s_o)

    def ix(x_mm: float) -> float:
        """Image-side mm -> display x."""
        return _IMAGE_SPAN * (x_mm / s_i)

    x_obj, x_img = ox(s_o), ix(s_i)
    lens_semi = 1.25

    rays: list[tuple[list[float], list[float]]] = []
    # 1. Parallel to the axis, refracted through the back focal point.
    rays.append(([x_obj, 0.0, x_img], [y_o, y_o, y_i]))
    # 2. Chief ray straight through the optical centre.
    rays.append(([x_obj, 0.0, x_img], [y_o, 0.0, y_i]))
    # 3. Through the front focal point, emerging parallel to the axis.
    rays.append(([x_obj, 0.0, x_img], [y_o, y_i, y_i]))

    return RayDiagram(
        rays=rays,
        object_arrow=([x_obj, x_obj], [0.0, y_o]),
        image_arrow=([x_img, x_img], [0.0, y_i]),
        lens_outline=([0.0, 0.0], [-lens_semi, lens_semi]),
        focal_points=([ox(f) if s_o > f else -_OBJECT_SPAN, ix(f)], [0.0, 0.0]),
        x_limits=(-_OBJECT_SPAN * 1.15, _IMAGE_SPAN * 1.3),
        y_limits=(-lens_semi * 1.2, lens_semi * 1.2),
        image_is_virtual=virtual,
    )


# =====================================================================
# Field of view
# =====================================================================
def fov_vs_focal_length(
    *,
    sensor_width_mm: float,
    sensor_height_mm: float,
    f_min_mm: float = 8.0,
    f_max_mm: float = 300.0,
    n: int = 160,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonal field of view across a focal-length sweep, for the context plot."""
    g = _geom()
    focal = np.geomspace(f_min_mm, f_max_mm, int(n))
    fov = np.array([g.field_of_view(float(f), sensor_width_mm, sensor_height_mm).diagonal_deg for f in focal])
    return focal, fov


def subject_footprint(
    *, focal_length_mm: float, sensor_width_mm: float, sensor_height_mm: float, distance_mm: float
) -> tuple[list[float], list[float]]:
    """Closed rectangle (in metres) of what the frame covers at *distance_mm*."""
    g = _geom()
    fov = g.field_of_view(focal_length_mm, sensor_width_mm, sensor_height_mm, distance_mm)
    w = g.subject_extent_mm(fov.horizontal_deg, distance_mm) / 1000.0
    h = g.subject_extent_mm(fov.vertical_deg, distance_mm) / 1000.0
    xs = [-w / 2, w / 2, w / 2, -w / 2, -w / 2]
    ys = [-h / 2, -h / 2, h / 2, h / 2, -h / 2]
    return xs, ys


# =====================================================================
# Depth of field
# =====================================================================
@dataclass(frozen=True)
class DofSweep:
    focus_distance_m: np.ndarray
    near_m: np.ndarray
    far_m: np.ndarray
    hyperfocal_m: float


def dof_vs_focus_distance(
    *,
    focal_length_mm: float,
    f_number: float,
    coc_mm: float,
    min_distance_mm: float,
    max_distance_mm: float,
    n: int = 200,
) -> DofSweep:
    """Near and far sharpness limits as the focus distance sweeps.

    Plotted log-log this is the picture of hyperfocal focusing: the far limit
    runs away to infinity exactly when the focus distance reaches H.
    """
    g = _geom()
    distances = np.geomspace(max(min_distance_mm, 1.0), max_distance_mm, int(n))
    near, far = [], []
    H = g.hyperfocal_distance_mm(focal_length_mm, f_number, coc_mm)
    for s in distances:
        dof = g.depth_of_field(focal_length_mm, f_number, float(s), coc_mm)
        near.append(dof.near_mm / 1000.0)
        # Clamp the runaway branch so a log axis stays plottable.
        far.append(min(dof.far_mm, 1e6) / 1000.0)
    return DofSweep(
        focus_distance_m=distances / 1000.0,
        near_m=np.array(near),
        far_m=np.array(far),
        hyperfocal_m=H / 1000.0,
    )


def blur_vs_object_distance(
    *,
    focal_length_mm: float,
    f_number: float,
    focus_distance_mm: float,
    min_distance_mm: float,
    max_distance_mm: float,
    n: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Defocus blur-disc diameter (um) vs where the object actually is (m).

    The focus distance is forced into the sample grid so the curve's zero
    actually appears; a plain geomspace would straddle it and clip the dip.
    """
    g = _geom()
    d = np.geomspace(max(min_distance_mm, 1.0), max_distance_mm, int(n))
    if min_distance_mm <= focus_distance_mm <= max_distance_mm:
        d = np.unique(np.append(d, focus_distance_mm))
    blur = np.array(
        [g.defocus_blur_diameter_mm(focal_length_mm, f_number, focus_distance_mm, float(x)) * 1000.0 for x in d]
    )
    return d / 1000.0, blur


@dataclass(frozen=True)
class ApertureTradeoff:
    f_numbers: np.ndarray
    defocus_blur_um: np.ndarray
    diffraction_blur_um: np.ndarray
    total_blur_um: np.ndarray
    optimum_f_number: float


def aperture_tradeoff(
    *,
    focal_length_mm: float,
    focus_distance_mm: float,
    object_distance_mm: float,
    wavelength_nm: float = 550.0,
    f_min: float = 1.0,
    f_max: float = 32.0,
    n: int = 160,
) -> ApertureTradeoff:
    """Defocus blur vs diffraction blur across the aperture range.

    Defocus shrinks as you stop down, diffraction grows, so the quadrature sum
    has a minimum: the sharpest aperture for that particular depth error. This
    is the quantitative version of "stopping down stops helping past f/11".
    """
    g = _geom()
    N = np.geomspace(f_min, f_max, int(n))
    defocus = np.array(
        [
            g.defocus_blur_diameter_mm(focal_length_mm, float(n_), focus_distance_mm, object_distance_mm) * 1000.0
            for n_ in N
        ]
    )
    diffraction = np.array([g.airy_disk_diameter_mm(float(n_), wavelength_nm) * 1000.0 for n_ in N])
    total = np.sqrt(defocus**2 + diffraction**2)
    return ApertureTradeoff(
        f_numbers=N,
        defocus_blur_um=defocus,
        diffraction_blur_um=diffraction,
        total_blur_um=total,
        optimum_f_number=float(N[int(np.argmin(total))]),
    )


# =====================================================================
# Relative illumination and distortion
# =====================================================================
def relative_illumination(
    *, focal_length_mm: float, sensor_diagonal_mm: float, n: int = 129
) -> tuple[np.ndarray, np.ndarray]:
    """``(normalised image height 0..1, relative illumination)``."""
    r, ri = _geom().relative_illumination_profile(focal_length_mm, sensor_diagonal_mm, n_samples=n)
    half_diag = max(sensor_diagonal_mm / 2.0, 1e-9)
    return r / half_diag, ri


def vignetting_preview(
    *, focal_length_mm: float, sensor_width_mm: float, sensor_height_mm: float, size: int = 96
) -> np.ndarray:
    """A frame-shaped cos^4 brightness map, so corner falloff is visible not just plotted."""
    g = _geom()
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    x_mm = (xs / (size - 1) - 0.5) * sensor_width_mm
    y_mm = (ys / (size - 1) - 0.5) * sensor_height_mm
    r_mm = np.hypot(x_mm, y_mm)
    return g.relative_illumination_cos4(r_mm, focal_length_mm)


def distortion_grid(
    *, k1: float, k2: float, p1: float = 0.0, p2: float = 0.0, aspect_ratio: float = 1.5
) -> list[tuple[list[float], list[float]]]:
    lines = _geom().distortion_grid(k1, k2, p1, p2, aspect_ratio=aspect_ratio)
    return [(xs.tolist(), ys.tolist()) for xs, ys in lines]


def distortion_coefficients(model: dict) -> tuple[float, float, float, float]:
    return _geom().distortion_coefficients((model or {}).get("lens", {}))
