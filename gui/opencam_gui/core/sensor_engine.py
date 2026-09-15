"""EMVA1288 photon-transfer adapter over ``tools/emva_theory.py``.

All noise statistics (shot, read, dark, quantization mapping) come from the
real ``emva_theory`` functions that ``tools/validate_emva_model.py`` also
uses to check a camera config against its datasheet — this module only
sweeps them across a signal range for a live plot.

DSNU / PRNU maps and the EMVA1288 spatial estimators (temporal averaging,
residual-temporal correction, DSNU1288, PRNU1288) likewise come from
``tools/emva_theory.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opencam_gui.core.repo import import_tool

FPN_MAP_SIZE = 48


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


@dataclass(frozen=True)
class FpnPreview:
    prnu_gain: np.ndarray
    dsnu_e: np.ndarray
    mu_e: np.ndarray
    spatial_std_e: np.ndarray
    prnu_term_e: np.ndarray


def fpn_preview(
    *,
    prnu_std_fraction: float,
    dsnu_std_e: float,
    dark_mean_e: float,
    dsnu_model: str,
    full_well_e: float,
    seed: int = 0,
    size: int = FPN_MAP_SIZE,
    n_points: int = 80,
) -> FpnPreview:
    m = _emva_module()
    rng = np.random.default_rng(seed)
    prnu = m.prnu_gain_map((size, size), prnu_std_fraction, rng)
    dsnu = m.dsnu_offset_map(
        (size, size),
        dsnu_std_e=dsnu_std_e,
        dark_mean_e=dark_mean_e,
        rng=rng,
        model=dsnu_model,
    )
    mu_e = np.linspace(0.0, max(full_well_e, 1.0), n_points)
    spatial = m.spatial_std_electrons(mu_e, dsnu_std_e, prnu_std_fraction)
    return FpnPreview(
        prnu_gain=prnu,
        dsnu_e=dsnu,
        mu_e=mu_e,
        spatial_std_e=spatial,
        prnu_term_e=prnu_std_fraction * mu_e,
    )


@dataclass(frozen=True)
class Emva1288Measurement:
    dsnu1288_e: float
    prnu1288: float
    uncorrected_dark_std_e: float
    residual_temporal_std_e: float
    dsnu_config_e: float
    prnu_config: float
    n_frames: int
    height: int
    width: int
    mu_50_e: float
    spatial_mu_e: np.ndarray
    spatial_std_measured_e: np.ndarray
    spatial_std_theory_e: np.ndarray


def measure_emva1288(
    *,
    prnu_std_fraction: float,
    dsnu_std_e: float,
    dark_mean_e: float,
    dsnu_model: str,
    sigma_d_e: float,
    K_e_per_DN: float,
    black_level_DN: float,
    full_well_e: float,
    use_poisson: bool,
    n_frames: int,
    seed: int = 0,
    size: int = FPN_MAP_SIZE,
) -> Emva1288Measurement:
    m = _emva_module()
    rng = np.random.default_rng(seed)
    prnu = m.prnu_gain_map((size, size), prnu_std_fraction, rng)
    dsnu = m.dsnu_offset_map(
        (size, size),
        dsnu_std_e=dsnu_std_e,
        dark_mean_e=dark_mean_e,
        rng=rng,
        model=dsnu_model,
    )
    k = float(K_e_per_DN)
    dark = m.simulate_uniform_stack(
        mu_e=0.0,
        n_frames=n_frames,
        prnu_map=prnu,
        dsnu_map=dsnu,
        dark_mean_e=dark_mean_e,
        sigma_d_e=sigma_d_e,
        K_e_per_DN=k,
        black_level_DN=black_level_DN,
        full_well_e=full_well_e,
        use_poisson=use_poisson,
        seed=seed + 101,
    )
    mu_50 = 0.5 * float(full_well_e)
    bright = m.simulate_uniform_stack(
        mu_e=mu_50,
        n_frames=n_frames,
        prnu_map=prnu,
        dsnu_map=dsnu,
        dark_mean_e=dark_mean_e,
        sigma_d_e=sigma_d_e,
        K_e_per_DN=k,
        black_level_DN=black_level_DN,
        full_well_e=full_well_e,
        use_poisson=use_poisson,
        seed=seed + 202,
    )
    dsnu_m = m.emva1288_dsnu(dark, k)
    prnu_m = m.emva1288_prnu(dark, bright)

    mu_levels = np.array([0.0, 0.1, 0.25, 0.5, 0.8], dtype=np.float64) * float(full_well_e)
    measured = [dsnu_m.dsnu_e]
    for i, mu in enumerate(mu_levels[1:]):
        stack = m.simulate_uniform_stack(
            mu_e=float(mu),
            n_frames=n_frames,
            prnu_map=prnu,
            dsnu_map=dsnu,
            dark_mean_e=dark_mean_e,
            sigma_d_e=sigma_d_e,
            K_e_per_DN=k,
            black_level_DN=black_level_DN,
            full_well_e=full_well_e,
            use_poisson=use_poisson,
            seed=seed + 303 + 17 * i,
        )
        stats = m.emva1288_spatial_stats(stack)
        measured.append(stats.corrected_spatial_std_dn * k)
    measured_arr = np.asarray(measured, dtype=np.float64)
    theory = m.spatial_std_electrons(mu_levels, dsnu_std_e, prnu_std_fraction)
    return Emva1288Measurement(
        dsnu1288_e=dsnu_m.dsnu_e,
        prnu1288=prnu_m.prnu_fraction,
        uncorrected_dark_std_e=dsnu_m.uncorrected_spatial_std_dn * k,
        residual_temporal_std_e=dsnu_m.residual_temporal_std_dn * k,
        dsnu_config_e=float(dsnu_std_e),
        prnu_config=float(prnu_std_fraction),
        n_frames=int(n_frames),
        height=size,
        width=size,
        mu_50_e=mu_50,
        spatial_mu_e=mu_levels,
        spatial_std_measured_e=measured_arr,
        spatial_std_theory_e=theory,
    )
