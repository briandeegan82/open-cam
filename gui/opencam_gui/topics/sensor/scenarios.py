"""Scripted lecture scenarios for the EMVA1288 sensor-noise demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    teaching_point: str
    notes: str
    sigma_d_e: float
    K_e_per_DN: float
    black_level_DN: float
    full_well_e: float
    use_poisson: bool = True
    dark_current_e_per_s: float = 0.15
    temperature_c: float = 20.0
    dark_current_reference_temp_c: float = 20.0
    dark_current_doubling_per_c: float = 6.0
    dark_activation_energy_eV: float = 0.0
    integration_time_s: float = 0.01
    camera_recipe_id: str | None = None


SCENARIOS: dict[str, Scenario] = {
    "ideal_photon_limited": Scenario(
        id="ideal_photon_limited",
        title="Ideal photon-limited sensor",
        teaching_point=(
            "Read noise is tiny relative to shot noise almost everywhere: the PTC is a "
            "straight line of slope 1 on log-log axes, Var(DN) ~ mean_e / K^2."
        ),
        notes="Compare the shot/read crossover electron count against the sensor's full well.",
        sigma_d_e=0.5,
        K_e_per_DN=1.0,
        black_level_DN=0.0,
        full_well_e=10000.0,
    ),
    "read_noise_limited": Scenario(
        id="read_noise_limited",
        title="Read-noise-limited low-light sensor",
        teaching_point=(
            "A large read noise flattens the PTC at low signal: below the shot/read "
            "crossover (u_e ~ sigma_d^2) the sensor is read-noise-limited, not photon-limited."
        ),
        notes="Raise sigma_d_e further and watch the crossover point move right.",
        sigma_d_e=8.0,
        K_e_per_DN=3.0,
        black_level_DN=32.0,
        full_well_e=4000.0,
    ),
    "hot_sensor_dark_current": Scenario(
        id="hot_sensor_dark_current",
        title="Hot sensor: dark current takes over",
        teaching_point=(
            "Long exposure at high temperature (Arrhenius model, Ea=0.63 eV) raises the "
            "dark floor's variance  -  visible as the leftmost PTC point lifting off the axis."
        ),
        notes="Drag temperature up from 20degC to 60degC and watch the dark-floor marker rise.",
        sigma_d_e=2.6,
        K_e_per_DN=4.77,
        black_level_DN=16.0,
        full_well_e=4800.0,
        dark_current_e_per_s=1.5,
        temperature_c=55.0,
        dark_activation_energy_eV=0.63,
        integration_time_s=1.0,
    ),
    "real_nikon_z6": Scenario(
        id="real_nikon_z6",
        title="Real camera: Nikon Z6",
        teaching_point="A full-frame ILC: large full well, moderate read noise, base-ISO gain.",
        notes="Loaded straight from config/camera_recipes/nikon_z6.yaml  -  no simulation shortcuts.",
        sigma_d_e=2.3,
        K_e_per_DN=4.0955,
        black_level_DN=512.0,
        full_well_e=65000.0,
        camera_recipe_id="nikon_z6",
    ),
    "real_iphone_8": Scenario(
        id="real_iphone_8",
        title="Real camera: iPhone 8",
        teaching_point=(
            "A 1.4 um phone pixel: small full well and higher relative noise mean the whole "
            "usable range sits much further left on the PTC than the Nikon Z6's."
        ),
        notes="Load both recipes back to back and compare the two PTC curves.",
        sigma_d_e=2.6,
        K_e_per_DN=4.77,
        black_level_DN=16.0,
        full_well_e=4800.0,
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
