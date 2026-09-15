"""Thin convenience wrapper around ``tools/camera_model.load_camera_model``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from opencam_gui.core.repo import import_tool


def load_camera_model(path: Path) -> dict:
    camera_model = import_tool("camera_model")
    return camera_model.load_camera_model(Path(path))


@dataclass(frozen=True)
class OpticsSummary:
    f_number: float
    pixel_pitch_um: float
    camera_type: str
    post_psf_enabled: bool
    post_psf_mode: str
    sigma_geometric_pixels: float
    lateral_ca_coefficient: float
    lambda_reference_nm: float


def optics_summary(model: dict) -> OpticsSummary:
    sensor = model.get("sensor", {}) or {}
    lens = model.get("lens", {}) or {}
    post_psf = lens.get("post_psf", {}) or {}
    return OpticsSummary(
        f_number=float(sensor.get("f_number", 2.0)),
        pixel_pitch_um=float(sensor.get("pixel_pitch_um", 1.4)),
        camera_type=str(lens.get("camera", "pinhole")),
        post_psf_enabled=bool(post_psf.get("enabled", False)),
        post_psf_mode=str(post_psf.get("mode", "chromatic_gaussian")),
        sigma_geometric_pixels=float(post_psf.get("sigma_geometric_pixels", 0.5)),
        lateral_ca_coefficient=float(post_psf.get("lateral_ca_coefficient", 0.0)),
        lambda_reference_nm=float(post_psf.get("lambda_reference_nm", 550.0)),
    )


@dataclass(frozen=True)
class EmvaSummary:
    K_e_per_DN: float
    sigma_d_e: float
    black_level_DN: float
    full_well_e: float
    use_poisson: bool
    dark_current_e_per_s: float
    dark_current_reference_temp_c: float
    temperature_c: float
    dark_current_doubling_per_c: float
    dark_activation_energy_eV: float
    integration_time_s: float
    prnu_std_fraction: float
    dsnu_std_e: float
    bit_depth: int


def emva_summary(model: dict) -> EmvaSummary:
    sensor = model.get("sensor", {}) or {}
    emva = (model.get("noise", {}) or {}).get("emva", {}) or {}
    adc = (model.get("noise", {}) or {}).get("adc", {}) or {}
    return EmvaSummary(
        K_e_per_DN=float(emva.get("overall_system_gain_K_e_per_DN", 4.77)),
        sigma_d_e=float(emva.get("sigma_d_e", 2.6)),
        black_level_DN=float(emva.get("black_level_DN", 16)),
        full_well_e=float(adc.get("full_well_e", 4800)),
        use_poisson=bool(emva.get("use_poisson_shot_noise", True)),
        dark_current_e_per_s=float(emva.get("dark_current_e_per_s", 0.15)),
        dark_current_reference_temp_c=float(emva.get("dark_current_reference_temp_c", 20.0)),
        temperature_c=float(emva.get("temperature_c", 20.0)),
        dark_current_doubling_per_c=float(emva.get("dark_current_doubling_per_c", 6.0)),
        dark_activation_energy_eV=float(emva.get("dark_activation_energy_eV", 0.0)),
        integration_time_s=float(sensor.get("integration_time_s", 0.01)),
        prnu_std_fraction=float(emva.get("prnu_std_fraction", 0.01)),
        dsnu_std_e=float(emva.get("dsnu_std_e", 0.3)),
        bit_depth=int(adc.get("bit_depth", 12)),
    )
