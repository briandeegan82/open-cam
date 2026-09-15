"""Scripted lecture scenarios for the resolution / MTF demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    teaching_point: str
    notes: str
    mode: str  # "chromatic_gaussian" | "airy_disk"
    f_number: float
    pixel_pitch_um: float
    sigma_geometric_px: float
    edge_angle_deg: float = 5.0
    spokes: int = 72
    downsample: int = 4
    prefilter_sigma_px: float = 0.0
    camera_recipe_id: str | None = None


SCENARIOS: dict[str, Scenario] = {
    "sharp_reference": Scenario(
        id="sharp_reference",
        title="A sharp system: MTF50 near Nyquist",
        teaching_point=(
            "MTF50 is the frequency where contrast has fallen to half. Quoted in cycles per "
            "pixel it is directly comparable against the 0.5 Nyquist limit, so you can see at "
            "a glance whether the lens or the sensor is the bottleneck."
        ),
        notes="Note the MTF50 in both cycles/px and cycles/mm; only the second depends on pixel size.",
        mode="chromatic_gaussian",
        f_number=4.0,
        pixel_pitch_um=4.3,
        sigma_geometric_px=0.3,
    ),
    "soft_lens": Scenario(
        id="soft_lens",
        title="A soft lens: aberration-limited",
        teaching_point=(
            "Raising the geometric aberration term widens the PSF, and MTF50 drops roughly in "
            "proportion. The picture of the blur and the number describing it are the same fact."
        ),
        notes="Compare the LSF width against the sharp reference, then compare the MTF50 values.",
        mode="chromatic_gaussian",
        f_number=4.0,
        pixel_pitch_um=4.3,
        sigma_geometric_px=1.6,
    ),
    "diffraction_limited_f16": Scenario(
        id="diffraction_limited_f16",
        title="Stopped down to f/16: diffraction sets the cutoff",
        teaching_point=(
            "A diffraction-limited lens transmits exactly nothing above 1/(lambda N). At f/16 and "
            "550 nm that cutoff is around 114 cycles/mm, and the measured MTF should hug the "
            "theoretical diffraction curve all the way down to it."
        ),
        notes="Check that the measured curve reaches zero at the marked cutoff, not before or after.",
        mode="airy_disk",
        f_number=16.0,
        pixel_pitch_um=4.3,
        sigma_geometric_px=0.1,
    ),
    "phone_pixel_diffraction": Scenario(
        id="phone_pixel_diffraction",
        title="Tiny pixels: why phones never stop down",
        teaching_point=(
            "On a 1.4 um pixel the diffraction cutoff at f/11 falls below the sensor's own "
            "Nyquist limit. Stopping down cannot buy depth of field without throwing away "
            "resolution the sensor was built to capture."
        ),
        notes="Watch where the diffraction cutoff marker sits relative to the Nyquist line.",
        mode="airy_disk",
        f_number=11.0,
        pixel_pitch_um=1.4,
        sigma_geometric_px=0.2,
        camera_recipe_id="iphone_8",
    ),
    "aliasing_no_filter": Scenario(
        id="aliasing_no_filter",
        title="Past Nyquist: aliasing invents detail",
        teaching_point=(
            "Point-sampling a Siemens star with no optical low-pass filter folds every frequency "
            "above Nyquist back into the image as moire. The fine spokes do not merely disappear "
            "  -  they reappear as coarse patterns running the wrong way."
        ),
        notes="Find the Nyquist radius marker: outside it the spokes are honest, inside they are not.",
        mode="chromatic_gaussian",
        f_number=4.0,
        pixel_pitch_um=4.3,
        sigma_geometric_px=0.2,
        prefilter_sigma_px=0.0,
    ),
    "aliasing_with_olpf": Scenario(
        id="aliasing_with_olpf",
        title="The optical low-pass filter trade",
        teaching_point=(
            "Blurring before sampling removes the frequencies that would alias. You lose real "
            "detail near Nyquist and gain an image that does not lie  -  which is why an OLPF "
            "is a deliberate choice, not a defect."
        ),
        notes="Toggle the prefilter sigma between 0 and 1.5 and compare the moire in the centre.",
        mode="chromatic_gaussian",
        f_number=4.0,
        pixel_pitch_um=4.3,
        sigma_geometric_px=0.2,
        prefilter_sigma_px=1.5,
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
