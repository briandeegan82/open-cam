"""Scripted lecture scenarios for the colour / ISP demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    teaching_point: str
    notes: str
    illuminant_id: str
    stages: tuple[str, ...]
    demosaic_method: str = "bilinear"
    wb_method: str = "gray_world"
    bayer_pattern: str = "RGGB"
    patch_index: int = 21
    camera_recipe_id: str | None = None


ALL_STAGES = ("mosaic", "demosaic", "white_balance", "ccm", "srgb")


SCENARIOS: dict[str, Scenario] = {
    "full_pipeline": Scenario(
        id="full_pipeline",
        title="The whole ISP, end to end",
        teaching_point=(
            "Spectra in, picture out. Every stage between them is undoing damage the "
            "previous one did or the sensor never avoided in the first place."
        ),
        notes=(
            "Read the delta-E readout, then start switching stages off from the bottom "
            "and watch which ones you can live without."
        ),
        illuminant_id="D65",
        stages=ALL_STAGES,
        wb_method="white_patch",
        camera_recipe_id="nikon_z6",
    ),
    "raw_is_not_a_colour_space": Scenario(
        id="raw_is_not_a_colour_space",
        title="Raw is not a colour space",
        teaching_point=(
            "Straight sensor RGB is off by a delta-E in the teens. The numbers are real "
            "measurements, but they are measurements in the camera's own basis, not in "
            "anyone's colour space."
        ),
        notes=(
            "Turn white balance and the CCM back on one at a time and watch delta-E fall "
            "from the teens to about one."
        ),
        illuminant_id="D65",
        stages=("mosaic", "demosaic", "srgb"),
        camera_recipe_id="nikon_z6",
    ),
    "gray_world_fails": Scenario(
        id="gray_world_fails",
        title="Gray world versus white patch",
        teaching_point=(
            "Gray world assumes the scene averages to grey. A ColorChecker does not, so it "
            "leaves a blue cast on the neutral ladder that white patch avoids entirely."
        ),
        notes=(
            "Watch the neutral-cast readout, not the image. Switch between the two methods "
            "and compare how far from (1, 1, 1) each one lands."
        ),
        illuminant_id="D65",
        stages=("mosaic", "demosaic", "white_balance", "srgb"),
        wb_method="gray_world",
        camera_recipe_id="nikon_z6",
    ),
    "tungsten": Scenario(
        id="tungsten",
        title="Tungsten: illuminant A at 2856 K",
        teaching_point=(
            "A domestic bulb is overwhelmingly red. The blue channel is starved, so white "
            "balance has to amplify it hard  -  and amplifying a weak channel amplifies its "
            "noise with it."
        ),
        notes=(
            "Compare the blue white-balance gain here against the D65 scenario, then look "
            "at the illuminant SPD to see where that gain comes from."
        ),
        illuminant_id="A",
        stages=ALL_STAGES,
        wb_method="white_patch",
        patch_index=18,
        camera_recipe_id="nikon_z6",
    ),
    "fluorescent_spikes": Scenario(
        id="fluorescent_spikes",
        title="F11: a spiky illuminant and metamerism",
        teaching_point=(
            "F11 is three narrow mercury lines, not a smooth spectrum. Surfaces that match "
            "under daylight can come apart under it, because the camera and the eye are "
            "sampling those spikes with different curves."
        ),
        notes=(
            "Look at the spectral overlay: where the lamp has no power, neither the eye nor "
            "the sensor can see the surface at all."
        ),
        illuminant_id="F11",
        stages=ALL_STAGES,
        wb_method="white_patch",
        patch_index=14,
        camera_recipe_id="nikon_z6",
    ),
    "narrowband_led": Scenario(
        id="narrowband_led",
        title="RGB LED: the hardest illuminant of all",
        teaching_point=(
            "Three narrow LED peaks can look white to the eye while carrying almost no power "
            "between them. Colour rendering collapses, and no white balance gain or 3x3 "
            "recovers information the light never delivered."
        ),
        notes=(
            "This is the worst delta-E in the set even with the full pipeline running. "
            "Compare it against D65 with identical settings."
        ),
        illuminant_id="LED_RGB1",
        stages=ALL_STAGES,
        wb_method="white_patch",
        patch_index=14,
        camera_recipe_id="nikon_z6",
    ),
    "mosaic_only": Scenario(
        id="mosaic_only",
        title="The CFA alone: two thirds of the data thrown away",
        teaching_point=(
            "Before any reconstruction, each pixel holds one number. The colour image is "
            "inferred, not measured  -  which is why demosaic artefacts exist at all."
        ),
        notes=(
            "Turn demosaic on and off to see the mosaic itself, then move to the demosaic "
            "tab to see what the reconstruction gets wrong."
        ),
        illuminant_id="D65",
        stages=("mosaic", "srgb"),
        camera_recipe_id="nikon_z6",
    ),
    "demosaic_quality": Scenario(
        id="demosaic_quality",
        title="Bilinear versus Malvar",
        teaching_point=(
            "Malvar's gradient correction beats bilinear by roughly two to one on edges, "
            "but only because real images carry sharp detail in luminance and slow "
            "variation in chroma. On an image without that structure the advantage vanishes."
        ),
        notes=(
            "Compare the two error maps on the demosaic tab. The differences sit on edges, "
            "which is exactly where the eye looks."
        ),
        illuminant_id="D65",
        stages=ALL_STAGES,
        demosaic_method="malvar",
        wb_method="white_patch",
        camera_recipe_id="nikon_z6",
    ),
    "no_ccm": Scenario(
        id="no_ccm",
        title="White balanced but uncorrected",
        teaching_point=(
            "White balance fixes the neutrals and nothing else. The greys go grey while the "
            "saturated patches stay wrong, because a per-channel gain cannot reshape a "
            "channel's spectral response."
        ),
        notes=(
            "The neutral ladder looks right and delta-E is still around six. Turn the CCM "
            "on to see where that error was hiding."
        ),
        illuminant_id="D65",
        stages=("mosaic", "demosaic", "white_balance", "srgb"),
        wb_method="white_patch",
        camera_recipe_id="nikon_z6",
    ),
    "rccb": Scenario(
        id="rccb",
        title="RCCB: cyan in place of a green",
        teaching_point=(
            "An RCCB array puts cyan where Bayer puts green. Near-IR leaks through more "
            "easily, and the Luther error grows because cyan is a worse match to the eye's "
            "V(lambda) than green is. The same 3x3 cannot save a worse spectral basis."
        ),
        notes=(
            "Compare the QE overlay and the Luther error against the D65 Bayer scenario. "
            "The green curve is now cyan."
        ),
        illuminant_id="D65",
        stages=ALL_STAGES,
        wb_method="white_patch",
        camera_recipe_id="default_rccb",
    ),
    "cmy": Scenario(
        id="cmy",
        title="CMY: complementary filters",
        teaching_point=(
            "Cyan, magenta and yellow filters pass two thirds of the spectrum each, so more "
            "photons arrive -- and the three curves overlap more, which is exactly what "
            "makes the Luther condition harder to meet."
        ),
        notes=(
            "Look at the QE overlay: every channel is broad. Then read delta-E -- more "
            "light did not buy more colour accuracy."
        ),
        illuminant_id="D65",
        stages=ALL_STAGES,
        wb_method="white_patch",
        camera_recipe_id="default_cmy",
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
