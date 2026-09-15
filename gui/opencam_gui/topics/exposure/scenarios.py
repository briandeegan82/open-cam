"""Scripted lecture scenarios for the exposure / sensor-defects demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    teaching_point: str
    notes: str
    scene_luminance_cd_m2: float
    f_number: float
    integration_time_s: float
    iso_gain: float
    defects: tuple[str, ...] = ()
    temperature_c: float = 20.0
    row_fpn_std_e: float = 12.0
    col_fpn_std_e: float = 12.0
    flicker_std_e: float = 10.0
    adc_inl_fraction: float = 0.02
    adc_dnl_std_lsb: float = 0.6
    hot_pixel_fraction: float = 2e-3
    bloom_spread: float = 0.5
    camera_recipe_id: str | None = None


SCENARIOS: dict[str, Scenario] = {
    "sunny_16": Scenario(
        id="sunny_16",
        title="Sunny 16: the reference exposure",
        teaching_point=(
            "At ISO 100 in direct sun, f/16 with a shutter of one over the ISO lands mid-grey "
            "where it belongs. Every other exposure in this demo is a departure from here."
        ),
        notes=(
            "Check the EV readout against the scene EV100. Then walk one stop down the "
            "aperture and one stop up the shutter and watch the electron count stay put."
        ),
        scene_luminance_cd_m2=4000.0,
        f_number=16.0,
        integration_time_s=1.0 / 125.0,
        iso_gain=1.0,
        camera_recipe_id="nikon_z6",
    ),
    "reciprocity": Scenario(
        id="reciprocity",
        title="Reciprocity: the same exposure three ways",
        teaching_point=(
            "Aperture and shutter trade one for one along a line of constant EV. The sensor "
            "cannot tell which one you moved  -  only the motion blur and the depth of field can."
        ),
        notes=(
            "Slide the f-number back and forth. The signal-electron readout should barely "
            "move while the EV line stays fixed."
        ),
        scene_luminance_cd_m2=800.0,
        f_number=4.0,
        integration_time_s=1.0 / 250.0,
        iso_gain=1.0,
        camera_recipe_id="nikon_z6",
    ),
    "read_noise_limited": Scenario(
        id="read_noise_limited",
        title="Underexposed: down in the read noise",
        teaching_point=(
            "Shot noise is sqrt(signal), so it only falls below a read noise of sigma when the "
            "signal drops under sigma squared  -  a handful of electrons, not a few hundred. "
            "Below that the sensor, not the light, sets the noise floor: the left-hand branch "
            "of the photon transfer curve."
        ),
        notes=(
            "Watch the regime label. One stop of aperture is enough to flip it to shot-noise "
            "limited, which is how narrow this branch really is."
        ),
        scene_luminance_cd_m2=0.6,
        f_number=8.0,
        integration_time_s=1.0 / 500.0,
        iso_gain=1.0,
        camera_recipe_id="nikon_z6",
    ),
    "iso_is_gain": Scenario(
        id="iso_is_gain",
        title="ISO is gain, not sensitivity",
        teaching_point=(
            "Raising ISO does not collect one extra photon. It amplifies what was already "
            "there, and clips the well proportionally sooner  -  so every stop of ISO is a stop "
            "of highlight headroom spent."
        ),
        notes=(
            "Push the ISO gain and read the full-well and dynamic-range curves. The signal "
            "electrons never change; only the headroom and the noise floor do."
        ),
        scene_luminance_cd_m2=20.0,
        f_number=2.8,
        integration_time_s=1.0 / 60.0,
        iso_gain=8.0,
        camera_recipe_id="nikon_z6",
    ),
    "blown_highlight": Scenario(
        id="blown_highlight",
        title="Clipped: blooming out of a saturated highlight",
        teaching_point=(
            "Past full well the charge has to go somewhere. It spills into the neighbours, so a "
            "blown highlight grows beyond the pixels that actually saw the light."
        ),
        notes=(
            "Toggle blooming on and off against the same frame and watch the highlight's "
            "footprint change while its centre stays clipped either way."
        ),
        scene_luminance_cd_m2=6000.0,
        f_number=2.0,
        integration_time_s=1.0 / 60.0,
        iso_gain=1.0,
        defects=("blooming",),
        camera_recipe_id="nikon_z6",
    ),
    "hot_pixels_long_exposure": Scenario(
        id="hot_pixels_long_exposure",
        title="Long exposure, hot sensor",
        teaching_point=(
            "Dark current doubles roughly every 6 degrees, and the worst pixels run far above "
            "the average. On a long warm exposure they appear as fixed bright specks that are "
            "in the same place on every frame."
        ),
        notes=(
            "Raise the temperature and re-render. The specks do not move  -  which is exactly "
            "what lets dark-frame subtraction remove them."
        ),
        scene_luminance_cd_m2=1.0,
        f_number=4.0,
        integration_time_s=2.0,
        iso_gain=4.0,
        defects=("hot_pixels",),
        temperature_c=45.0,
        hot_pixel_fraction=4e-3,
        camera_recipe_id="nikon_z6",
    ),
    "banding": Scenario(
        id="banding",
        title="Readout banding: row, column and 1/f",
        teaching_point=(
            "Row and column offsets survive spatial averaging in one direction and vanish in "
            "the other, so the row and column profiles separate them instantly. 1/f flicker "
            "lands on rows too, but smoothly, because a rolling shutter reads rows in time order."
        ),
        notes=(
            "Enable one of the three at a time and watch which profile moves. The image itself "
            "barely changes; the profiles are unambiguous."
        ),
        scene_luminance_cd_m2=60.0,
        f_number=4.0,
        integration_time_s=1.0 / 60.0,
        iso_gain=4.0,
        defects=("row_fpn", "flicker"),
        row_fpn_std_e=25.0,
        flicker_std_e=20.0,
        camera_recipe_id="nikon_z6",
    ),
    "adc_nonlinearity": Scenario(
        id="adc_nonlinearity",
        title="ADC INL and DNL: the converter's own errors",
        teaching_point=(
            "INL is a smooth bow across the range  -  a tone-curve error that peaks mid-scale. "
            "DNL is per-code and fixed, so unlike read noise it does not average away no matter "
            "how many frames you stack."
        ),
        notes=(
            "Read the transfer-curve deviation plot rather than the image. Toggle each and "
            "note that one is smooth and the other is not."
        ),
        scene_luminance_cd_m2=200.0,
        f_number=5.6,
        integration_time_s=1.0 / 125.0,
        iso_gain=1.0,
        defects=("adc_inl", "adc_dnl"),
        adc_inl_fraction=0.04,
        adc_dnl_std_lsb=1.2,
        camera_recipe_id="nikon_z6",
    ),
    "ktc_no_cds": Scenario(
        id="ktc_no_cds",
        title="kTC reset noise: the sensor without CDS",
        teaching_point=(
            "Resetting the sense node leaves a random charge of sqrt(kTC)/q electrons. "
            "Correlated double sampling cancels it by measuring the reset level and subtracting "
            "it, which is why every modern CMOS sensor does CDS and this switch is normally off."
        ),
        notes=(
            "Enable kTC and watch the read-noise floor jump. Raise the temperature to confirm "
            "the sqrt(T) dependence is real but weak next to the capacitance term."
        ),
        scene_luminance_cd_m2=100.0,
        f_number=4.0,
        integration_time_s=1.0 / 125.0,
        iso_gain=1.0,
        defects=("ktc",),
        camera_recipe_id="nikon_z6",
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
