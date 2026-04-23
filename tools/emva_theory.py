"""Analytic EMVA1288-style temporal noise predictions for the electron → DN model.

Matches ``apply_emva_noise.py``: DN = e / K + black_level, Poisson(μ_e) shot,
Gaussian(0, σ_d) read noise, optional hard clip at full well before DN conversion.
"""

from __future__ import annotations

import numpy as np


def temporal_variance_electrons_squared(mu_e: float, sigma_d_e: float, *, use_poisson: bool) -> float:
    """Variance of electron count after shot + read (before full-well clip)."""
    v_shot = float(mu_e) if use_poisson else 0.0
    return v_shot + float(sigma_d_e) ** 2


def temporal_variance_dn_squared(
    mu_e: float,
    sigma_d_e: float,
    K_e_per_DN: float,
    *,
    use_poisson: bool,
) -> float:
    """Var(DN) for temporal noise only (no PRNU/DSNU), linear regime, no saturation."""
    return temporal_variance_electrons_squared(mu_e, sigma_d_e, use_poisson=use_poisson) / float(K_e_per_DN) ** 2


def dark_floor_clip_mean_var_dn(
    sigma_d_e: float,
    K_e_per_DN: float,
    black_dn: float,
) -> tuple[float, float]:
    """Mean and Var(DN) for μ_e=0: e = max(0, N(0, σ_d²)), then DN = e/K + black.

    Matches ``apply_emva_noise`` lower clip on electrons before ADC (ignores full-well
    clip, which is irrelevant in the dark).
    """
    s = float(sigma_d_e)
    mean_e = s / np.sqrt(2.0 * np.pi)
    var_e = 0.5 * s * s - mean_e * mean_e
    k = float(K_e_per_DN)
    return mean_e / k + float(black_dn), var_e / (k * k)


def mean_dn_linear(mu_e: float, K_e_per_DN: float, black_dn: float) -> float:
    """Expected mean DN in the linear (unsaturated) regime."""
    return float(mu_e) / float(K_e_per_DN) + float(black_dn)


def monte_carlo_temporal_dn_stats(
    mu_e: float,
    sigma_d_e: float,
    K_e_per_DN: float,
    black_dn: float,
    *,
    use_poisson: bool,
    full_well_e: float | None,
    n_trials: int,
    seed: int,
) -> tuple[float, float]:
    """Return (mean DN, sample variance of DN) from independent temporal draws."""
    rng = np.random.default_rng(seed)
    if use_poisson:
        e = rng.poisson(mu_e, size=n_trials).astype(np.float64)
    else:
        e = np.full(n_trials, mu_e, dtype=np.float64)
    e = e + rng.normal(0.0, sigma_d_e, size=n_trials)
    fw = float(full_well_e) if full_well_e is not None else np.inf
    e = np.clip(e, 0.0, fw)
    dn = e / float(K_e_per_DN) + float(black_dn)
    return float(np.mean(dn)), float(np.var(dn, ddof=1))


def photon_transfer_curve_checks(
    mu_levels_e: np.ndarray,
    sigma_d_e: float,
    K_e_per_DN: float,
    black_dn: float,
    *,
    use_poisson: bool,
    full_well_e: float | None,
    n_trials: int,
    seed: int,
    variance_rtol: float,
    mean_atol: float,
) -> list[dict]:
    """Compare theory vs Monte Carlo at several mean signal levels."""
    rows: list[dict] = []
    base_seed = int(seed)
    for i, mu in enumerate(np.asarray(mu_levels_e, dtype=np.float64)):
        if float(mu) < 1e-9:
            pred_mean, pred_var = dark_floor_clip_mean_var_dn(sigma_d_e, K_e_per_DN, black_dn)
        else:
            pred_mean = mean_dn_linear(mu, K_e_per_DN, black_dn)
            pred_var = temporal_variance_dn_squared(mu, sigma_d_e, K_e_per_DN, use_poisson=use_poisson)
        m_mc, v_mc = monte_carlo_temporal_dn_stats(
            float(mu),
            sigma_d_e,
            K_e_per_DN,
            black_dn,
            use_poisson=use_poisson,
            full_well_e=full_well_e,
            n_trials=n_trials,
            seed=base_seed + 1000 * i + 17,
        )
        var_ok = abs(v_mc - pred_var) <= variance_rtol * max(pred_var, 1e-12)
        mean_ok = abs(m_mc - pred_mean) <= mean_atol
        rows.append(
            {
                "mu_e": float(mu),
                "pred_mean_dn": pred_mean,
                "mc_mean_dn": m_mc,
                "mean_ok": bool(mean_ok),
                "pred_var_dn": pred_var,
                "mc_var_dn": v_mc,
                "var_ok": bool(var_ok),
            }
        )
    return rows


def compare_config_to_datasheet(
    K_cfg: float,
    sigma_cfg: float,
    fw_cfg: float,
    black_cfg: float,
    K_ds: float,
    sigma_ds: float,
    fw_ds: float,
    black_ds: float,
    rtol: float,
    *,
    bit_depth_cfg: int | None = None,
    bit_depth_ds: int | None = None,
    gain_convention_ds: str = "e_per_dn",
) -> dict:
    """Return pass/fail for each parameter vs datasheet targets.

    Optional normalization:
    - ``gain_convention_ds``: ``"e_per_dn"`` (default) or ``"dn_per_e"``
    - ``bit_depth_ds`` and ``bit_depth_cfg`` rescale datasheet black level to
      config ADC depth before comparison.
    """
    gain_mode = str(gain_convention_ds).strip().lower()
    if gain_mode not in ("e_per_dn", "dn_per_e"):
        raise ValueError('gain_convention_ds must be "e_per_dn" or "dn_per_e"')
    K_ds_eff = float(K_ds) if gain_mode == "e_per_dn" else 1.0 / max(1e-12, float(K_ds))
    black_ds_eff = float(black_ds)
    if bit_depth_cfg is not None and bit_depth_ds is not None:
        black_ds_eff *= float(2 ** (int(bit_depth_cfg) - int(bit_depth_ds)))

    checks = []
    for name, a, b in (
        ("K_e_per_DN", K_cfg, K_ds_eff),
        ("sigma_d_e", sigma_cfg, sigma_ds),
        ("full_well_e", fw_cfg, fw_ds),
        ("black_level_DN", black_cfg, black_ds_eff),
    ):
        denom = max(abs(b), 1e-12)
        ok = abs(a - b) <= rtol * denom
        checks.append({"name": name, "config": a, "datasheet": b, "ok": bool(ok)})
    return {
        "parameter_checks": checks,
        "normalization": {
            "gain_convention_ds": gain_mode,
            "bit_depth_cfg": bit_depth_cfg,
            "bit_depth_ds": bit_depth_ds,
        },
        "all_ok": bool(all(c["ok"] for c in checks)),
    }
