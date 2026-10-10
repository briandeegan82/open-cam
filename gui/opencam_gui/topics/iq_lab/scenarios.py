"""Scripted lecture scenarios for the IQ lab demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    teaching_point: str
    notes: str
    scene: str  # "diorama" | "hdr" | "skin" | "flare" (keys of run_recipe_scorecard.SCENES)
    camera_recipe_id: str
    xres: int = 360
    yres: int = 240
    pixelsamples: int = 16


SCENARIOS: dict[str, Scenario] = {
    "sharpness": Scenario(
        id="sharpness",
        title="Edge and texture acutance (diorama)",
        teaching_point=(
            "The slanted edge measures MTF on a high-contrast step; dead leaves measures it on "
            "low-contrast texture, which noise and denoising affect very differently."
        ),
        notes="Raise PBRT samples/px to 64+: Monte-Carlo noise inflates texture acutance at low spp.",
        scene="diorama",
        camera_recipe_id="default",
    ),
    "hdr_linear": Scenario(
        id="hdr_linear",
        title="Dynamic range: linear sensor",
        teaching_point="DR = saturation / lowest signal that still reaches SNR 1 (or 10), measured on the HDR chart.",
        notes="Note DR (SNR=1) and the CDP level, then run the dual-gain scenario on the same chart.",
        scene="hdr",
        camera_recipe_id="default",
    ),
    "hdr_dcg": Scenario(
        id="hdr_dcg",
        title="Dynamic range: dual conversion gain",
        teaching_point=(
            "The same sensor with dual conversion gain: the high-gain read lowers the noise floor, "
            "so DR at SNR = 1 rises by roughly 18 dB."
        ),
        notes="Same optics and chart as the linear scenario, only the pixel architecture changes.",
        scene="hdr",
        camera_recipe_id="default_hdr_dcg",
    ),
    "skin_phone": Scenario(
        id="skin_phone",
        title="Skin-tone accuracy (ΔE00) through a phone lens",
        teaching_point=(
            "ΔE00 on synthetic skin patches after white balance and the sensor's own ColorChecker CCM. "
            "The skin chart is never used to fit the CCM."
        ),
        notes="Compare against the default recipe: QE and IR-cut filter set how well a 3x3 CCM can fit.",
        scene="skin",
        camera_recipe_id="iphone_8",
    ),
    "flare": Scenario(
        id="flare",
        title="Veiling glare (black-hole target)",
        teaching_point="Light scattered into the black holes of a bright field, as a percentage of the field.",
        notes="pbrt has no coating/ghost model here; real-lens recipes scatter more than pinholes.",
        scene="flare",
        camera_recipe_id="iphone_8",
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
