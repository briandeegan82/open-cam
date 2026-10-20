"""JSON-ready listings for the Studio API: scenes, cameras, illuminants, quality presets."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from opencam_gui.core import pipeline as pl
from opencam_gui.core.camera import load_camera_model
from opencam_gui.core.catalog import illuminant_label, list_camera_recipes, list_illuminants, load_spectrum_csv
from opencam_gui.core.repo import pbrt_available, repo_root


@dataclass(frozen=True)
class QualityPreset:
    id: str
    label: str
    description: str
    xres: int
    yres: int
    pixelsamples: int


QUALITY_PRESETS: tuple[QualityPreset, ...] = (
    QualityPreset("preview", "Preview", "Quick look; visible render noise.", 320, 240, 32),
    QualityPreset("standard", "Standard", "Good for checking framing, colour and exposure.", 480, 320, 128),
    QualityPreset("final", "Final", "Low Monte Carlo noise; use for metrics and figures.", 960, 640, 1024),
)

# Rough CPU pbrt cost measured on the ColorChecker scene (960x640, 1024 spp ~ 4.5 min on 8 cores).
_PBRT_CORE_SECONDS_PER_SAMPLE = 270.0 * 8 / (960 * 640 * 1024)
_PBRT_FIXED_OVERHEAD_S = 8.0
_FAST_ANALYTIC_S = 4.0


def estimate_seconds(mode: str, xres: int, yres: int, pixelsamples: int) -> float:
    """Order-of-magnitude wall time for a run, for display only."""
    if mode == "fast_analytic":
        return _FAST_ANALYTIC_S
    cores = max(1, os.cpu_count() or 1)
    return _PBRT_FIXED_OVERHEAD_S + _PBRT_CORE_SECONDS_PER_SAMPLE * xres * yres * pixelsamples / cores


def scenes() -> list[dict]:
    out = []
    for s in pl.list_scenes():
        modes = (["fast_analytic"] if s.supports_fast_analytic else []) + ["pbrt_accurate"]
        out.append(
            {
                "id": s.id,
                "label": s.label,
                "kind": s.kind,
                "description": s.description,
                "modes": modes,
                "supports_illuminant": s.supports_illuminant_spectrum,
            }
        )
    return out


def _family(recipe_id: str) -> str:
    if recipe_id.startswith("default"):
        return "Reference sensors"
    head = recipe_id.split("_", 1)[0]
    names = {"iphone": "Apple", "google": "Google", "konica": "Konica Minolta"}
    return names.get(head, head.capitalize())


def _camera_summary(recipe_id: str, display_name: str, path: Path) -> dict:
    m = load_camera_model(path)
    sensor = m.get("sensor", {}) or {}
    lens = m.get("lens", {}) or {}
    cfa = m.get("cfa", {}) or {}
    noise = m.get("noise", {}) or {}
    emva = noise.get("emva", {}) or {}
    adc = noise.get("adc", {}) or {}
    source = m.get("source", {}) or {}
    hdr = m.get("hdr") or (sensor.get("hdr") if isinstance(sensor.get("hdr"), dict) else None)
    return {
        "id": recipe_id,
        "display_name": display_name,
        "family": _family(recipe_id),
        "sensor_class": source.get("sensor_class"),
        "calibration_tier": source.get("calibration_tier"),
        "cfa": (cfa.get("layout") or cfa.get("pattern")) if cfa.get("enabled", True) else "mono",
        "demosaic": cfa.get("demosaic"),
        "pixel_pitch_um": sensor.get("pixel_pitch_um"),
        "f_number": sensor.get("f_number"),
        "integration_time_s": sensor.get("integration_time_s"),
        "lens_camera": lens.get("camera"),
        "focal_length_mm": lens.get("focal_length_mm"),
        "bit_depth": adc.get("bit_depth"),
        "full_well_e": adc.get("full_well_e"),
        "gain_e_per_dn": emva.get("overall_system_gain_K_e_per_DN"),
        "read_noise_e": emva.get("sigma_d_e"),
        "hdr": bool(hdr) and bool(hdr.get("enabled", True)) if isinstance(hdr, dict) else False,
        "ircf": bool((sensor.get("quantum_efficiency", {}) or {}).get("ircf_csv")),
        "path": str(path.relative_to(repo_root())),
    }


@lru_cache(maxsize=1)
def cameras() -> tuple[dict, ...]:
    out = []
    for r in list_camera_recipes():
        try:
            out.append(_camera_summary(r.id, r.display_name, r.path))
        except Exception as exc:  # a broken recipe must not take the whole list down
            out.append({"id": r.id, "display_name": r.display_name, "family": _family(r.id), "error": str(exc)})
    return tuple(out)


def illuminants() -> list[dict]:
    return [{"id": i.id, "label": illuminant_label(i), "path": i.repo_relative} for i in list_illuminants()]


def illuminant_spectrum(illuminant_id: str) -> dict:
    for i in list_illuminants():
        if i.id == illuminant_id:
            wl, val = load_spectrum_csv(i.path)
            return {"id": i.id, "wavelength_nm": wl, "value": val}
    raise KeyError(illuminant_id)


def quality_presets() -> list[dict]:
    return [asdict(p) for p in QUALITY_PRESETS]


def status() -> dict:
    return {
        "pbrt_available": pbrt_available(),
        "cpu_count": os.cpu_count() or 1,
        "repo_root": str(repo_root()),
    }
