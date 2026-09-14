"""Dark-current temperature scaling, mirrored (read-only) from ``tools/apply_emva_noise.py``.

The sensor-noise pipeline computes this multiplier inline inside
``apply_emva_noise.main()`` (see the doubling-rule branch around line 1383 and
the model selection documented in ``config/sensor_models/default.yaml``'s
``noise.emva.dark_activation_energy_eV`` comment) rather than exposing it as an
importable function, so a temperature slider has nothing to call into for a
live plot. This module reproduces those two documented formulas verbatim,
purely for interactive display — it does not participate in image generation;
the real noise pipeline always computes dark current itself when a scene is
actually rendered through ``tools/apply_emva_noise.py``.

Two models, matching the config file exactly:

    doubling rule (``dark_activation_energy_eV == 0``):
        I(T) = I_ref * 2 ** ((T - T_ref) / doubling_per_C)

    Arrhenius with T^1.5 pre-exponential (``dark_activation_energy_eV > 0``):
        I(T) = I_ref * (T_K / T_ref_K) ** 1.5 * exp(Ea/kB * (1/T_ref_K - 1/T_K))
"""

from __future__ import annotations

_KB_EV_PER_K = 8.617333262e-5  # Boltzmann constant, eV/K


def dark_current_multiplier(
    temperature_c: float,
    reference_temp_c: float,
    doubling_per_c: float,
    activation_energy_eV: float,
) -> float:
    """I(T) / I_ref for the selected dark-current model."""
    if activation_energy_eV > 0.0:
        t_k = temperature_c + 273.15
        t_ref_k = reference_temp_c + 273.15
        return (t_k / t_ref_k) ** 1.5 * pow(
            2.718281828459045,
            activation_energy_eV / _KB_EV_PER_K * (1.0 / t_ref_k - 1.0 / t_k),
        )
    return 2.0 ** ((temperature_c - reference_temp_c) / max(1e-6, doubling_per_c))


def dark_current_electrons_per_s(
    dark_current_e_per_s_ref: float,
    temperature_c: float,
    reference_temp_c: float,
    doubling_per_c: float,
    activation_energy_eV: float,
) -> float:
    return dark_current_e_per_s_ref * dark_current_multiplier(
        temperature_c, reference_temp_c, doubling_per_c, activation_energy_eV
    )
