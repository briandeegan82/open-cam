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
    prnu_std_fraction: float = 0.01
    dsnu_std_e: float = 0.3
    dsnu_model: str = "gaussian"
    n_measure_frames: int = 50


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
        prnu_std_fraction=0.007,
        dsnu_std_e=0.3,
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
        prnu_std_fraction=0.018,
        dsnu_std_e=0.16,
    ),
    "no_fpn": Scenario(
        id="no_fpn",
        title="No FPN: averaging kills spatial noise",
        teaching_point=(
            "With DSNU = PRNU = 0 the spatial std of a dark average is just leftover "
            "read noise, sigma_d / sqrt(L). EMVA1288 subtracts that residual so DSNU1288 ~ 0."
        ),
        notes="Open the DSNU/PRNU tab, run the measurement at L=10 then L=50, and compare uncorrected vs corrected.",
        sigma_d_e=8.0,
        K_e_per_DN=1.0,
        black_level_DN=0.0,
        full_well_e=4000.0,
        prnu_std_fraction=0.0,
        dsnu_std_e=0.0,
        n_measure_frames=50,
    ),
    "dsnu_only": Scenario(
        id="dsnu_only",
        title="DSNU only: additive dark pattern",
        teaching_point=(
            "DSNU is a frozen additive offset. Averaging more frames does not wash it out, "
            "and the spatial-std curve stays flat versus signal (PRNU = 0)."
        ),
        notes="The DSNU map is the same pattern you would recover from a stack of dark frames.",
        sigma_d_e=1.5,
        K_e_per_DN=1.0,
        black_level_DN=0.0,
        full_well_e=4000.0,
        prnu_std_fraction=0.0,
        dsnu_std_e=3.0,
        n_measure_frames=50,
    ),
    "prnu_only": Scenario(
        id="prnu_only",
        title="PRNU only: multiplicative gain map",
        teaching_point=(
            "PRNU is invisible in the dark (the DSNU map is flat) and grows linearly with "
            "signal: sigma_spatial = PRNU x mu_e. EMVA1288 measures it from a 50% flat field "
            "after subtracting the dark spatial variance."
        ),
        notes="Run the measurement and check PRNU1288 against the 3% slider value.",
        sigma_d_e=1.5,
        K_e_per_DN=1.0,
        black_level_DN=0.0,
        full_well_e=4000.0,
        prnu_std_fraction=0.03,
        dsnu_std_e=0.0,
        n_measure_frames=50,
    ),
    "hot_pixels_lognormal": Scenario(
        id="hot_pixels_lognormal",
        title="Hot pixels: log-normal DSNU",
        teaching_point=(
            "The pipeline's DSNU is a log-normal dark-current map (long positive tail), "
            "not a Gaussian. A few hot pixels dominate the spatial std  -  the histogram "
            "is the giveaway."
        ),
        notes="Needs enough dark signal for the log-normal to be well-conditioned (1 s, 40 C).",
        sigma_d_e=2.0,
        K_e_per_DN=1.0,
        black_level_DN=0.0,
        full_well_e=8000.0,
        dark_current_e_per_s=2.0,
        temperature_c=40.0,
        dark_activation_energy_eV=0.63,
        integration_time_s=1.0,
        prnu_std_fraction=0.0,
        dsnu_std_e=4.0,
        dsnu_model="lognormal",
        n_measure_frames=50,
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
