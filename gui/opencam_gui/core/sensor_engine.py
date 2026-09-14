"""EMVA1288 photon-transfer adapter over ``tools/emva_theory.py``.

All noise statistics (shot, read, dark, quantization mapping) come from the
real ``emva_theory`` functions that ``tools/validate_emva_model.py`` also
uses to check a camera config against its datasheet — this module only
sweeps them across a signal range for a live plot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opencam_gui.core.repo import import_tool


def _emva_module():
    return import_tool("emva_theory")


def mean_dn_linear(mu_e: float, K_e_per_DN: float, black_level_DN: float) -> float:
    return _emva_module().mean_dn_linear(mu_e, K_e_per_DN, black_level_DN)


@dataclass(frozen=True)
class PtcCurve:
    mu_e: np.ndarray
    mean_dn: np.ndarray
    var_dn: np.ndarray
    dark_mean_dn: float
    dark_var_dn: float
    full_well_e: float
    shot_read_crossover_e: float


def photon_transfer_curve(
    *,
    sigma_d_e: float,
    K_e_per_DN: float,
    black_level_DN: float,
    full_well_e: float,
    use_poisson: bool,
    mu_dark_e: float = 0.0,
    n_points: int = 200,
) -> PtcCurve:
    m = _emva_module()
    mu_e = np.linspace(max(1.0, full_well_e * 1e-4), full_well_e, n_points)
    mean_dn = np.array([m.mean_dn_linear(mu, K_e_per_DN, black_level_DN) for mu in mu_e])
    var_dn = np.array(
        [
            m.temporal_variance_dn_squared(
                mu, sigma_d_e, K_e_per_DN, use_poisson=use_poisson, mu_dark_e=mu_dark_e
            )
            for mu in mu_e
        ]
    )
    dark_mean_dn, dark_var_dn = m.dark_floor_clip_mean_var_dn(sigma_d_e, K_e_per_DN, black_level_DN)
    return PtcCurve(
        mu_e=mu_e,
        mean_dn=mean_dn,
        var_dn=var_dn,
        dark_mean_dn=dark_mean_dn,
        dark_var_dn=dark_var_dn,
        full_well_e=full_well_e,
        shot_read_crossover_e=float(sigma_d_e) ** 2,
    )


@dataclass(frozen=True)
class VerifyResult:
    mu_e: float
    theory_mean_dn: float
    theory_var_dn: float
    mc_mean_dn: float
    mc_var_dn: float


def verify_against_monte_carlo(
    *,
    mu_e: float,
    sigma_d_e: float,
    K_e_per_DN: float,
    black_level_DN: float,
    full_well_e: float,
    use_poisson: bool,
    mu_dark_e: float = 0.0,
    n_trials: int = 20000,
    seed: int = 0,
) -> VerifyResult:
    m = _emva_module()
    if mu_e < 1e-9:
        theory_mean, theory_var = m.dark_floor_clip_mean_var_dn(sigma_d_e, K_e_per_DN, black_level_DN)
    else:
        theory_mean = m.mean_dn_linear(mu_e, K_e_per_DN, black_level_DN)
        theory_var = m.temporal_variance_dn_squared(
            mu_e, sigma_d_e, K_e_per_DN, use_poisson=use_poisson, mu_dark_e=mu_dark_e
        )
    mc_mean, mc_var = m.monte_carlo_temporal_dn_stats(
        mu_e,
        sigma_d_e,
        K_e_per_DN,
        black_level_DN,
        use_poisson=use_poisson,
        full_well_e=full_well_e,
        n_trials=n_trials,
        seed=seed,
        mu_dark_e=mu_dark_e,
    )
    return VerifyResult(
        mu_e=mu_e,
        theory_mean_dn=theory_mean,
        theory_var_dn=theory_var,
        mc_mean_dn=mc_mean,
        mc_var_dn=mc_var,
    )
