"""Scripted lecture scenarios for the optics / PSF demo."""

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
    lateral_ca_coefficient: float = 0.0
    camera_recipe_id: str | None = None


SCENARIOS: dict[str, Scenario] = {
    "diffraction_limited_compact": Scenario(
        id="diffraction_limited_compact",
        title="Diffraction-limited compact sensor",
        teaching_point=(
            "A 1 um pixel is small enough that the diffraction spot alone spans several "
            "pixels  -  the lens is diffraction-limited before geometric aberration even matters."
        ),
        notes="Compare the red-channel Airy radius against the pixel pitch in the status readout.",
        mode="airy_disk",
        f_number=2.8,
        pixel_pitch_um=1.0,
        sigma_geometric_px=0.2,
    ),
    "fast_phone_lens": Scenario(
        id="fast_phone_lens",
        title="Fast wide-aperture phone lens",
        teaching_point=(
            "A fast f/1.8 aperture keeps the diffraction spot small, so a larger geometric "
            "(aberration) blur term now dominates the total PSF width instead."
        ),
        notes="Raise sigma_geometric_pixels and watch which term wins the quadrature sum.",
        mode="chromatic_gaussian",
        f_number=1.8,
        pixel_pitch_um=1.4,
        sigma_geometric_px=0.6,
        camera_recipe_id="iphone_8",
    ),
    "stopped_down_dslr": Scenario(
        id="stopped_down_dslr",
        title="Stopped-down DSLR (landscape diffraction limit)",
        teaching_point=(
            "Even on a large full-frame pixel, stopping down to f/16 for depth of field "
            "eventually makes diffraction  -  not the lens's aberrations  -  the sharpness limit."
        ),
        notes="Sweep f_number from f/4 to f/16 and watch the Airy rings in the 2-D kernel grow.",
        mode="airy_disk",
        f_number=16.0,
        pixel_pitch_um=4.3,
        sigma_geometric_px=0.1,
    ),
    "uncorrected_lateral_ca": Scenario(
        id="uncorrected_lateral_ca",
        title="Uncorrected achromat: visible lateral CA",
        teaching_point=(
            "Wavelength-dependent magnification M(lambda) shifts red and blue outward/inward by "
            "different amounts, producing colour fringing that grows toward the image edges."
        ),
        notes="Watch the ring/spoke test chart: fringes appear where the per-channel shift is largest.",
        mode="chromatic_gaussian",
        f_number=4.0,
        pixel_pitch_um=2.0,
        sigma_geometric_px=0.4,
        lateral_ca_coefficient=0.04,
    ),
    "airy_vs_gaussian": Scenario(
        id="airy_vs_gaussian",
        title="Airy rings vs Gaussian approximation",
        teaching_point=(
            "chromatic_gaussian collapses diffraction to a single sigma; airy_disk keeps the "
            "central lobe and first two rings  -  same aperture, visibly different PSF shape."
        ),
        notes="Toggle the PSF mode control with these same parameters to compare the two directly.",
        mode="airy_disk",
        f_number=5.6,
        pixel_pitch_um=3.0,
        sigma_geometric_px=0.3,
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
