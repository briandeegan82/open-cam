"""Exposure and sensor-defect adapter over ``tools/sensor_radiometry.py`` and
``tools/apply_emva_noise.py``.

Two halves. The exposure half runs the photometric chain -- scene luminance,
aperture, integration time and ISO into electrons -- and locates where that
lands on the same photon-transfer curve the sensor demo plots, so students can
see a chosen exposure fall into the read-noise-limited, shot-limited or clipped
region rather than being told which one it is.

The defect half drives the pipeline's own defect models one at a time against a
shared base frame, so each artefact's signature is isolated. Nothing here
reimplements a defect: blooming, hot/stuck pixels, kTC, ADC DNL/INL, row and
column FPN and 1/f flicker all come from ``apply_emva_noise``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opencam_gui.core.repo import import_tool

PREVIEW_SIZE = 96
BASE_SEED = 20240517


def _rad():
    return import_tool("sensor_radiometry")


def _noise():
    return import_tool("apply_emva_noise")


def _emva():
    return import_tool("emva_theory")


# =====================================================================
# Exposure
# =====================================================================
@dataclass(frozen=True)
class ExposurePoint:
    """Where one (luminance, N, t, ISO) choice lands on the sensor."""

    scene_luminance_cd_m2: float
    f_number: float
    integration_time_s: float
    iso_gain: float
    ev: float
    ev100_scene: float
    illuminance_lux: float
    signal_e: float
    full_well_e: float
    K_e_per_DN: float
    mean_dn: float
    max_dn: float
    shot_noise_e: float
    read_noise_e: float
    total_noise_e: float
    snr_db: float
    saturation_fraction: float
    regime: str


def exposure_point(
    *,
    scene_luminance_cd_m2: float,
    f_number: float,
    integration_time_s: float,
    iso_gain: float = 1.0,
    pixel_pitch_um: float,
    quantum_efficiency: float,
    K_e_per_DN: float,
    full_well_e: float,
    sigma_d_e: float,
    sigma_amp_e: float = 0.0,
    black_level_DN: float = 0.0,
    bit_depth: int = 12,
    fill_factor: float = 1.0,
    transmission: float = 0.9,
) -> ExposurePoint:
    """Run one exposure through the photometric chain and classify the result."""
    rad, noise = _rad(), _noise()

    signal_e = float(
        rad.electrons_from_exposure(
            scene_luminance_cd_m2,
            f_number,
            integration_time_s,
            pixel_pitch_um=pixel_pitch_um,
            quantum_efficiency=quantum_efficiency,
            fill_factor=fill_factor,
            transmission=transmission,
        )
    )

    # ISO is amplifier gain: it rescales the ADC path, so the same electrons land
    # on a different code and the well clips sooner.
    k_eff, well_eff = noise.iso_scaled_conversion(K_e_per_DN, full_well_e, iso_gain)
    read_e = noise.amplifier_read_noise_e(sigma_d_e, sigma_amp_e, iso_gain)

    captured_e = min(signal_e, well_eff)
    shot_e = float(np.sqrt(max(captured_e, 0.0)))
    total_e = float(np.sqrt(shot_e**2 + read_e**2))
    max_dn = float((1 << bit_depth) - 1)
    mean_dn = min(captured_e / k_eff + black_level_DN, max_dn)

    if signal_e >= well_eff:
        regime = "clipped"
    elif shot_e <= read_e:
        regime = "read-noise limited"
    else:
        regime = "shot-noise limited"

    return ExposurePoint(
        scene_luminance_cd_m2=float(scene_luminance_cd_m2),
        f_number=float(f_number),
        integration_time_s=float(integration_time_s),
        iso_gain=float(iso_gain),
        ev=float(rad.exposure_value(f_number, integration_time_s)),
        ev100_scene=float(rad.ev100_from_luminance(scene_luminance_cd_m2)),
        illuminance_lux=float(
            rad.image_plane_illuminance_lux(scene_luminance_cd_m2, f_number, transmission=transmission)
        ),
        signal_e=signal_e,
        full_well_e=float(well_eff),
        K_e_per_DN=float(k_eff),
        mean_dn=float(mean_dn),
        max_dn=max_dn,
        shot_noise_e=shot_e,
        read_noise_e=float(read_e),
        total_noise_e=total_e,
        snr_db=float(20.0 * np.log10(max(captured_e, 1e-9) / max(total_e, 1e-9))),
        saturation_fraction=float(signal_e / well_eff) if well_eff > 0 else 0.0,
        regime=regime,
    )


@dataclass(frozen=True)
class ExposureTriangle:
    """Iso-exposure lines in the (f-number, shutter) plane."""

    f_number: np.ndarray
    shutter_s: np.ndarray  # the line through the current setting
    ev_lines: list[tuple[float, np.ndarray]]
    current_f_number: float
    current_shutter_s: float
    current_ev: float


def exposure_triangle(
    *,
    f_number: float,
    integration_time_s: float,
    n_points: int = 96,
    ev_offsets: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0),
    f_min: float = 1.0,
    f_max: float = 32.0,
) -> ExposureTriangle:
    """Constant-exposure contours: every point on a line gives the same electrons.

    This is the nomogram form of the triangle. Reading along one line is the
    aperture-for-shutter trade, and stepping between lines is one stop of
    exposure -- which is also one stop of ISO if the light cannot be changed.
    """
    rad = _rad()
    f_axis = np.geomspace(f_min, f_max, n_points)
    base_ev = float(rad.exposure_value(f_number, integration_time_s))

    lines = [(base_ev + off, np.asarray(rad.shutter_for_exposure_value(f_axis, base_ev + off))) for off in ev_offsets]
    return ExposureTriangle(
        f_number=f_axis,
        shutter_s=np.asarray(rad.shutter_for_exposure_value(f_axis, base_ev)),
        ev_lines=lines,
        current_f_number=float(f_number),
        current_shutter_s=float(integration_time_s),
        current_ev=base_ev,
    )


@dataclass(frozen=True)
class IsoSweep:
    """How read noise, well depth and SNR move as ISO is pushed."""

    iso_gain: np.ndarray
    read_noise_e: np.ndarray
    full_well_e: np.ndarray
    dynamic_range_db: np.ndarray
    snr_db: np.ndarray
    signal_e: float


def iso_sweep(
    *,
    signal_e: float,
    K_e_per_DN: float,
    full_well_e: float,
    sigma_d_e: float,
    sigma_amp_e: float = 0.0,
    gains: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0),
) -> IsoSweep:
    """ISO is gain, not sensitivity.

    The electron count is fixed by the light and the exposure; raising ISO only
    changes how it is amplified. What moves is the headroom (the well clips
    proportionally sooner) and, once amplifier noise matters, the noise floor.
    """
    noise = _noise()
    g = np.asarray(gains, dtype=np.float64)

    read = np.array([noise.amplifier_read_noise_e(sigma_d_e, sigma_amp_e, float(x)) for x in g])
    well = np.array([noise.iso_scaled_conversion(K_e_per_DN, full_well_e, float(x))[1] for x in g])

    captured = np.minimum(signal_e, well)
    total = np.sqrt(captured + read**2)
    return IsoSweep(
        iso_gain=g,
        read_noise_e=read,
        full_well_e=well,
        dynamic_range_db=20.0 * np.log10(np.maximum(well, 1e-9) / np.maximum(read, 1e-9)),
        snr_db=20.0 * np.log10(np.maximum(captured, 1e-9) / np.maximum(total, 1e-9)),
        signal_e=float(signal_e),
    )


# =====================================================================
# Defects
# =====================================================================
DEFECTS = (
    "blooming",
    "hot_pixels",
    "ktc",
    "adc_inl",
    "adc_dnl",
    "row_fpn",
    "column_fpn",
    "flicker",
)

DEFECT_LABELS = {
    "blooming": "Blooming",
    "hot_pixels": "Hot / stuck pixels",
    "ktc": "kTC reset noise",
    "adc_inl": "ADC INL",
    "adc_dnl": "ADC DNL",
    "row_fpn": "Row FPN",
    "column_fpn": "Column FPN",
    "flicker": "1/f flicker",
}


def base_frame_e(
    *,
    size: int = PREVIEW_SIZE,
    full_well_e: float,
    background_fraction: float = 0.18,
    highlight_fraction: float = 6.0,
    highlight_radius_px: float = 6.0,
) -> np.ndarray:
    """A flat mid-grey field with one deliberately over-full highlight.

    The flat field makes fixed-pattern structure visible and the blown highlight
    gives blooming something to spill out of.
    """
    y, x = np.mgrid[0:size, 0:size]
    cy = cx = (size - 1) / 2.0
    frame = np.full((size, size), full_well_e * background_fraction, dtype=np.float64)
    disc = ((x - cx) ** 2 + (y - cy) ** 2) <= highlight_radius_px**2
    frame[disc] = full_well_e * highlight_fraction
    return frame


@dataclass(frozen=True)
class DefectFrame:
    """One rendering of the base frame with a chosen set of defects enabled."""

    dn: np.ndarray
    electrons: np.ndarray
    max_dn: float
    black_dn: float
    enabled: tuple[str, ...]
    sigma_ktc_e: float
    hot_pixel_count: int


def render_defects(
    *,
    enabled: tuple[str, ...] | set[str],
    full_well_e: float,
    K_e_per_DN: float,
    sigma_d_e: float,
    bit_depth: int = 12,
    black_level_DN: float = 0.0,
    temperature_c: float = 20.0,
    size: int = PREVIEW_SIZE,
    bloom_spread: float = 0.5,
    hot_pixel_fraction: float = 2e-3,
    row_fpn_std_e: float = 12.0,
    col_fpn_std_e: float = 12.0,
    flicker_std_e: float = 10.0,
    adc_inl_fraction: float = 0.02,
    adc_dnl_std_lsb: float = 0.6,
    seed: int = BASE_SEED,
) -> DefectFrame:
    """Apply the selected defects to a shared base frame.

    Every defect is off unless named, and the seeds are fixed, so toggling one
    changes only that one's contribution -- which is what makes the signatures
    comparable.
    """
    noise = _noise()
    wanted = set(enabled)
    unknown = wanted - set(DEFECTS)
    if unknown:
        raise ValueError(f"unknown defects: {sorted(unknown)}")

    rng = np.random.default_rng(seed)
    fixed_rng = np.random.default_rng(seed + 1)  # per-unit patterns: DNL table, hot pixels

    frame = base_frame_e(size=size, full_well_e=full_well_e)

    hot_count = 0
    if "hot_pixels" in wanted:
        frame, hot_info = noise.apply_hot_stuck_pixel_model(
            frame.astype(np.float32),
            fixed_rng,
            {
                "enabled": True,
                "hot_pixel_rate": hot_pixel_fraction,
                "stuck_high_rate": hot_pixel_fraction / 4.0,
                "stuck_low_rate": hot_pixel_fraction / 4.0,
                "hot_dark_e_min": full_well_e * 0.05,
                "hot_dark_e_max": full_well_e * 0.8,
            },
            full_well_e,
        )
        frame = np.asarray(frame, dtype=np.float64)
        hot_count = int(
            hot_info.get("hot_pixel_count", 0)
            + hot_info.get("stuck_high_count", 0)
            + hot_info.get("stuck_low_count", 0)
        )

    if "blooming" in wanted:
        frame = np.asarray(
            noise.apply_blooming(frame.astype(np.float32), full_well_e, spread_fraction=bloom_spread), dtype=np.float64
        )
    else:
        frame = np.minimum(frame, full_well_e)

    # Readout-path offsets, all in electrons, added downstream of the sense node.
    offsets = np.zeros_like(frame)
    if "row_fpn" in wanted or "column_fpn" in wanted:
        row, col = noise.row_column_fpn_offsets(
            frame.shape,
            row_fpn_std_e if "row_fpn" in wanted else 0.0,
            col_fpn_std_e if "column_fpn" in wanted else 0.0,
            fixed_rng,
        )
        offsets = offsets + row + col
    if "flicker" in wanted:
        offsets = offsets + noise.flicker_row_offsets(frame.shape[0], flicker_std_e, rng)[:, None]

    sigma_ktc = 0.0
    if "ktc" in wanted:
        sigma_ktc = noise.ktc_sigma_e(temperature_c=temperature_c, K_e_per_DN=K_e_per_DN, bit_depth=bit_depth)

    read = rng.normal(0.0, sigma_d_e, size=frame.shape)
    if sigma_ktc > 0.0:
        read = read + rng.normal(0.0, sigma_ktc, size=frame.shape)

    electrons = frame + offsets + read

    max_dn = float((1 << bit_depth) - 1)
    dn = np.clip(electrons / K_e_per_DN + black_level_DN, 0.0, max_dn)
    if "adc_inl" in wanted:
        dn = noise.apply_adc_inl(dn, black_dn=black_level_DN, max_dn=max_dn, quadratic_fraction=adc_inl_fraction)
    if "adc_dnl" in wanted:
        dn = noise.apply_adc_dnl(dn, noise.adc_dnl_table(max_dn, adc_dnl_std_lsb, fixed_rng), max_dn)

    return DefectFrame(
        dn=dn,
        electrons=electrons,
        max_dn=max_dn,
        black_dn=float(black_level_DN),
        enabled=tuple(sorted(wanted)),
        sigma_ktc_e=float(sigma_ktc),
        hot_pixel_count=hot_count,
    )


@dataclass(frozen=True)
class AdcTransfer:
    """ADC transfer curve and its deviation from the ideal ramp."""

    code_in: np.ndarray
    code_out: np.ndarray
    deviation_lsb: np.ndarray
    inl_peak_lsb: float
    dnl_peak_lsb: float


def adc_transfer(
    *,
    bit_depth: int = 12,
    black_level_DN: float = 0.0,
    inl_fraction: float = 0.02,
    dnl_std_lsb: float = 0.6,
    n_points: int = 512,
    seed: int = BASE_SEED + 1,
) -> AdcTransfer:
    """Sweep an ideal ramp through the converter and plot what comes back.

    INL is the smooth bow -- a tone-curve error. DNL is the per-code jitter
    around it, and because the table is fixed rather than redrawn, it is the one
    kind of "noise" that frame averaging will not remove.
    """
    noise = _noise()
    max_dn = float((1 << bit_depth) - 1)
    ramp = np.linspace(0.0, max_dn, n_points)

    out = noise.apply_adc_inl(ramp.copy(), black_dn=black_level_DN, max_dn=max_dn, quadratic_fraction=inl_fraction)
    inl_peak = float(np.max(np.abs(out - ramp)))
    if dnl_std_lsb > 0.0:
        table = noise.adc_dnl_table(max_dn, dnl_std_lsb, np.random.default_rng(seed))
        after = noise.apply_adc_dnl(out.copy(), table, max_dn)
        dnl_peak = float(np.max(np.abs(after - out)))
        out = after
    else:
        dnl_peak = 0.0

    return AdcTransfer(
        code_in=ramp,
        code_out=out,
        deviation_lsb=out - ramp,
        inl_peak_lsb=inl_peak,
        dnl_peak_lsb=dnl_peak,
    )


def row_column_profiles(frame: DefectFrame) -> tuple[np.ndarray, np.ndarray]:
    """Median DN per row and per column.

    Collapsing a whole row averages its read noise down by sqrt(width) while
    leaving a row offset untouched, so banding far below the per-pixel noise
    floor stands out here -- which is how row/column FPN and flicker are actually
    diagnosed. The median rather than the mean so a blown highlight crossing a
    few rows does not masquerade as banding.
    """
    return np.median(frame.dn, axis=1), np.median(frame.dn, axis=0)


def preview_rgb(frame: DefectFrame, *, gamma: float = 2.2) -> np.ndarray:
    """Normalised HxWx3 preview of a defect frame for display."""
    norm = np.clip(frame.dn / max(frame.max_dn, 1e-9), 0.0, 1.0) ** (1.0 / gamma)
    return np.repeat(norm.astype(np.float32)[:, :, None], 3, axis=2)
