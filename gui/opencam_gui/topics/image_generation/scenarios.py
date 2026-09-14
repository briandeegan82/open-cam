"""Scripted lecture scenarios for the Image Generation demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    teaching_point: str
    notes: str
    scene_id: str
    camera_recipe_id: str
    illuminant_id: str | None  # None for scenes that don't support a spectrum override
    target_illuminance_lux: float
    exposure_time_s: float
    mode: str  # "fast_analytic" | "pbrt_accurate"


SCENARIOS: dict[str, Scenario] = {
    "studio_daylight": Scenario(
        id="studio_daylight",
        title="Studio daylight ColorChecker",
        teaching_point="D65 daylight at a comfortable studio illuminance  -  the reference render.",
        notes="Good baseline before changing anything else.",
        scene_id="colorchecker",
        camera_recipe_id="nikon_z6",
        illuminant_id="D65",
        target_illuminance_lux=1000.0,
        exposure_time_s=0.01,
        mode="fast_analytic",
    ),
    "tungsten_indoor": Scenario(
        id="tungsten_indoor",
        title="Tungsten indoor lighting",
        teaching_point=(
            "Illuminant A (2856 K tungsten) at typical room illuminance  -  everything shifts "
            "warm relative to the D65 render unless white balance corrects for it."
        ),
        notes="Compare clean_demosaic_rgb8.png against the D65 studio scenario.",
        scene_id="colorchecker",
        camera_recipe_id="nikon_z6",
        illuminant_id="A",
        target_illuminance_lux=150.0,
        exposure_time_s=0.02,
        mode="fast_analytic",
    ),
    "low_light_phone": Scenario(
        id="low_light_phone",
        title="Low-light phone shot",
        teaching_point=(
            "A small 1.4 um pixel and only 20 lux push signal down near the read-noise floor "
            "explored in the Sensor Modelling tutorial  -  expect visible shot + read noise."
        ),
        notes="Try raising exposure_time_s to claw back SNR, exactly like a longer shutter speed would.",
        scene_id="colorchecker",
        camera_recipe_id="iphone_8",
        illuminant_id="D50",
        target_illuminance_lux=20.0,
        exposure_time_s=0.01,
        mode="fast_analytic",
    ),
    "bright_overcast": Scenario(
        id="bright_overcast",
        title="Bright overcast day",
        teaching_point="High illuminance (15000 lux)  -  comfortably shot-noise-limited, near full well.",
        notes="Push target_illuminance_lux further and watch highlight patches clip toward full well.",
        scene_id="colorchecker",
        camera_recipe_id="nikon_z6",
        illuminant_id="D75",
        target_illuminance_lux=15000.0,
        exposure_time_s=0.004,
        mode="fast_analytic",
    ),
    "fluorescent_office": Scenario(
        id="fluorescent_office",
        title="Fluorescent office lighting",
        teaching_point="A narrow tri-band fluorescent spectrum (F11)  -  a harder colour-rendering case than daylight.",
        notes="Look for patches whose hue shifts more than under D65 or A.",
        scene_id="colorchecker",
        camera_recipe_id="nikon_z6",
        illuminant_id="F11",
        target_illuminance_lux=400.0,
        exposure_time_s=0.01,
        mode="fast_analytic",
    ),
    "resolution_target_pbrt": Scenario(
        id="resolution_target_pbrt",
        title="Siemens star resolution target (full PBRT render)",
        teaching_point=(
            "IQ targets only support the physically-accurate PBRT path  -  this scenario needs "
            "a built PBRT binary (see docs/BUILD_PBRT.txt)."
        ),
        notes="Use Dry run first to see the exact command sequence without needing PBRT built.",
        scene_id="siemens_star",
        camera_recipe_id="nikon_z6",
        illuminant_id=None,
        target_illuminance_lux=1000.0,
        exposure_time_s=0.01,
        mode="pbrt_accurate",
    ),
}


def list_scenarios() -> list[Scenario]:
    return list(SCENARIOS.values())


def get_scenario(scenario_id: str) -> Scenario:
    try:
        return SCENARIOS[scenario_id]
    except KeyError as exc:
        known = ", ".join(SCENARIOS)
        raise KeyError(f"unknown scenario {scenario_id!r}; choose from: {known}") from exc
