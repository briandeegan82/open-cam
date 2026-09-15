"""Analytic EMVA1288-style temporal noise predictions for the electron → DN model.

Matches ``apply_emva_noise.py``: DN = e / K + black_level, Poisson(μ_e) shot,
Gaussian(0, σ_d) read noise, optional hard clip at full well before DN conversion.

Also implements the EMVA1288 *spatial* estimators for DSNU and PRNU (Release 3.1
§7–8): temporal averaging of a uniform-field stack, residual-temporal correction,
then DSNU1288 from dark frames and PRNU1288 from a ~50 % saturation flat field.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def temporal_variance_electrons_squared(
    mu_e: float,
    sigma_d_e: float,
    *,
    use_poisson: bool,
    sigma_ktc_e: float = 0.0,
    mu_dark_e: float = 0.0,
) -> float:
    """Variance of electron count after shot + read + optional kTC (before full-well clip).

    ``mu_dark_e`` is the mean dark-current electrons per pixel.  Dark charge arrives
    via independent Poisson statistics, so its shot variance adds to the signal shot
    variance.  At typical short exposures (< 0.1 s) the effect is negligible, but it
    becomes significant at elevated temperature or long integration time.
    """
    v_shot = float(mu_e + mu_dark_e) if use_poisson else 0.0
    return v_shot + float(sigma_d_e) ** 2 + float(sigma_ktc_e) ** 2


def temporal_variance_dn_squared(
    mu_e: float,
    sigma_d_e: float,
    K_e_per_DN: float,
    *,
    use_poisson: bool,
    sigma_ktc_e: float = 0.0,
    mu_dark_e: float = 0.0,
) -> float:
    """Var(DN) for temporal noise only (no PRNU/DSNU), linear regime, no saturation."""
    return (
        temporal_variance_electrons_squared(
            mu_e, sigma_d_e, use_poisson=use_poisson, sigma_ktc_e=sigma_ktc_e,
            mu_dark_e=mu_dark_e,
        )
        / float(K_e_per_DN) ** 2
    )


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
    sigma_ktc_e: float = 0.0,
    mu_dark_e: float = 0.0,
) -> tuple[float, float]:
    """Return (mean DN, sample variance of DN) from independent temporal draws."""
    rng = np.random.default_rng(seed)
    if use_poisson:
        e = rng.poisson(float(mu_e + mu_dark_e), size=n_trials).astype(np.float64)
    else:
        e = np.full(n_trials, float(mu_e + mu_dark_e), dtype=np.float64)
    e = e + rng.normal(0.0, sigma_d_e, size=n_trials)
    if sigma_ktc_e > 0.0:
        e = e + rng.normal(0.0, sigma_ktc_e, size=n_trials)
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
    mu_dark_e: float = 0.0,
) -> list[dict]:
    """Compare theory vs Monte Carlo at several mean signal levels."""
    rows: list[dict] = []
    base_seed = int(seed)
    for i, mu in enumerate(np.asarray(mu_levels_e, dtype=np.float64)):
        if float(mu) < 1e-9:
            pred_mean, pred_var = dark_floor_clip_mean_var_dn(sigma_d_e, K_e_per_DN, black_dn)
        else:
            pred_mean = mean_dn_linear(mu, K_e_per_DN, black_dn)
            pred_var = temporal_variance_dn_squared(
                mu, sigma_d_e, K_e_per_DN, use_poisson=use_poisson, mu_dark_e=mu_dark_e,
            )
        m_mc, v_mc = monte_carlo_temporal_dn_stats(
            float(mu),
            sigma_d_e,
            K_e_per_DN,
            black_dn,
            use_poisson=use_poisson,
            full_well_e=full_well_e,
            n_trials=n_trials,
            seed=base_seed + 1000 * i + 17,
            mu_dark_e=mu_dark_e,
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


# ---------------------------------------------------------------------------
# Spatial FPN: DSNU / PRNU maps and the EMVA1288 measurement protocol
# ---------------------------------------------------------------------------
#
# EMVA 1288 Release 3.1 §7 (spatial variance) and §8 (DSNU / PRNU):
#
#   ȳ[m,n]     = (1/L) Σ_i y_i[m,n]                         temporal mean image
#   s²_ȳ       = (1/(MN−1)) Σ (ȳ[m,n] − μ_ȳ)²               spatial var of ȳ
#   σ²_y       = mean_{m,n} of the per-pixel temporal variance
#   s²_y       = s²_ȳ − σ²_y / L                            residual-temporal correction
#
#   DSNU1288   = s_y.dark / K                                 (e⁻)
#   PRNU1288   = √(s²_y.50 − s²_y.dark) / (μ_y.50 − μ_y.dark) (fraction)
#
# Map generators match ``apply_emva_noise.py`` (PRNU: 1+N(0,σ²) clipped ≥ 0;
# DSNU log-normal: zero-mean offset of a log-normal dark-current map).  The
# Gaussian DSNU option is the textbook EMVA statistical model: a zero-mean
# additive field whose spatial std *is* DSNU, so the estimator has a known
# ground truth.


def prnu_gain_map(
    shape: tuple[int, ...],
    prnu_std_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Per-pixel multiplicative gain, matching ``apply_emva_noise`` (mono path)."""
    std = float(prnu_std_fraction)
    if std <= 0.0:
        return np.ones(shape, dtype=np.float64)
    gain = 1.0 + rng.normal(0.0, std, size=shape)
    return np.maximum(gain, 0.0)


def dsnu_offset_map(
    shape: tuple[int, ...],
    *,
    dsnu_std_e: float,
    dark_mean_e: float,
    rng: np.random.Generator,
    model: str = "gaussian",
) -> np.ndarray:
    """Per-pixel additive dark offset in electrons.

    ``gaussian``: zero-mean N(0, DSNU²), the EMVA definition of DSNU as a
    spatial standard deviation.
    ``lognormal``: matches ``apply_emva_noise``'s legacy ``dsnu_std_e`` path —
    log-normal absolute dark signal centred on ``dark_mean_e``, returned as a
    zero-mean offset.  Degenerates to zeros when ``dark_mean_e`` is ~0.
    """
    std = float(dsnu_std_e)
    if std <= 0.0:
        return np.zeros(shape, dtype=np.float64)
    mode = str(model).strip().lower()
    if mode == "gaussian":
        return rng.normal(0.0, std, size=shape).astype(np.float64)
    if mode == "lognormal":
        mu_dark = float(dark_mean_e)
        if mu_dark <= 1e-9:
            return np.zeros(shape, dtype=np.float64)
        v_ratio = std / mu_dark
        sigma_ln = float(np.sqrt(np.log1p(v_ratio ** 2)))
        mu_ln = np.log(mu_dark) - 0.5 * sigma_ln ** 2
        abs_map = rng.lognormal(mu_ln, sigma_ln, size=shape)
        return (abs_map - mu_dark).astype(np.float64)
    raise ValueError('dsnu model must be "gaussian" or "lognormal"')


def spatial_std_electrons(
    mu_e: np.ndarray | float,
    dsnu_std_e: float,
    prnu_std_fraction: float,
) -> np.ndarray:
    """Closed-form spatial RMS (e⁻) after infinite temporal averaging.

    σ_spatial(μ) = √( DSNU² + (PRNU · μ)² ).  DSNU is additive and independent
    of signal; PRNU is multiplicative and grows linearly with the mean.
    """
    mu = np.asarray(mu_e, dtype=np.float64)
    return np.sqrt(float(dsnu_std_e) ** 2 + (float(prnu_std_fraction) * mu) ** 2)


@dataclass(frozen=True)
class SpatialStackStats:
    """EMVA1288 spatial statistics of one uniform-field stack, in stack units."""

    n_frames: int
    mean_dn: float
    spatial_std_mean_image_dn: float
    temporal_std_dn: float
    corrected_spatial_std_dn: float


def emva1288_spatial_stats(stack_dn: np.ndarray) -> SpatialStackStats:
    """Temporal-mean image, residual-temporal correction, spatial std.

    ``stack_dn`` has shape (L, H, W) with L ≥ 2.  ``spatial_std_mean_image_dn``
    is the uncorrected s_ȳ (still contains σ_temporal / √L).
    ``corrected_spatial_std_dn`` is s_y = √(s²_ȳ − σ²_y / L).
    """
    stack = np.asarray(stack_dn, dtype=np.float64)
    if stack.ndim != 3 or stack.shape[0] < 2:
        raise ValueError("stack_dn must have shape (L, H, W) with L >= 2")
    n_frames = int(stack.shape[0])
    mean_img = stack.mean(axis=0)
    s_mean = float(mean_img.std(ddof=1))
    temporal_var = float(stack.var(axis=0, ddof=1).mean())
    corrected_var = s_mean * s_mean - temporal_var / n_frames
    return SpatialStackStats(
        n_frames=n_frames,
        mean_dn=float(mean_img.mean()),
        spatial_std_mean_image_dn=s_mean,
        temporal_std_dn=float(np.sqrt(max(temporal_var, 0.0))),
        corrected_spatial_std_dn=float(np.sqrt(max(corrected_var, 0.0))),
    )


@dataclass(frozen=True)
class Emva1288DsnuResult:
    dsnu_dn: float
    dsnu_e: float
    uncorrected_spatial_std_dn: float
    residual_temporal_std_dn: float
    mean_dark_dn: float
    n_frames: int


def emva1288_dsnu(dark_stack_dn: np.ndarray, K_e_per_DN: float) -> Emva1288DsnuResult:
    """DSNU1288 from a stack of dark frames (EMVA 1288 §8.1).

    DSNU1288 = s_y.dark / K  (electrons).  Also returns the uncorrected spatial
    std of the mean image, which still includes residual temporal noise σ/√L.
    """
    stats = emva1288_spatial_stats(dark_stack_dn)
    k = float(K_e_per_DN)
    return Emva1288DsnuResult(
        dsnu_dn=stats.corrected_spatial_std_dn,
        dsnu_e=stats.corrected_spatial_std_dn * k,
        uncorrected_spatial_std_dn=stats.spatial_std_mean_image_dn,
        residual_temporal_std_dn=stats.temporal_std_dn / np.sqrt(stats.n_frames),
        mean_dark_dn=stats.mean_dn,
        n_frames=stats.n_frames,
    )


@dataclass(frozen=True)
class Emva1288PrnuResult:
    prnu_fraction: float
    s_y50_dn: float
    s_ydark_dn: float
    mean_photo_dn: float
    n_frames: int


def emva1288_prnu(
    dark_stack_dn: np.ndarray,
    bright_stack_dn: np.ndarray,
) -> Emva1288PrnuResult:
    """PRNU1288 from dark + ~50 % saturation stacks (EMVA 1288 §8.2).

    PRNU1288 = √(s²_y.50 − s²_y.dark) / (μ_y.50 − μ_y.dark).
    """
    dark = emva1288_spatial_stats(dark_stack_dn)
    bright = emva1288_spatial_stats(bright_stack_dn)
    s2_photo = bright.corrected_spatial_std_dn ** 2 - dark.corrected_spatial_std_dn ** 2
    s_photo = float(np.sqrt(max(s2_photo, 0.0)))
    mean_photo = bright.mean_dn - dark.mean_dn
    prnu = s_photo / max(abs(mean_photo), 1e-12)
    return Emva1288PrnuResult(
        prnu_fraction=float(prnu),
        s_y50_dn=bright.corrected_spatial_std_dn,
        s_ydark_dn=dark.corrected_spatial_std_dn,
        mean_photo_dn=float(mean_photo),
        n_frames=bright.n_frames,
    )


def simulate_uniform_stack(
    *,
    mu_e: float,
    n_frames: int,
    prnu_map: np.ndarray,
    dsnu_map: np.ndarray,
    dark_mean_e: float,
    sigma_d_e: float,
    K_e_per_DN: float,
    black_level_DN: float,
    full_well_e: float,
    use_poisson: bool,
    seed: int,
) -> np.ndarray:
    """Stack of uniform-field frames (L, H, W) in DN.

    Mean electrons match the EMVA picture used by ``apply_emva_noise``:

    ``photo_mean = μ_e · g_PRNU + dark_mean`` (Poisson shot on this mean),
    then a spatially fixed additive DSNU offset, then Gaussian read noise,
    full-well clip, DN = e/K + black.  Applying DSNU *after* the Poisson draw
    keeps a zero-mean Gaussian DSNU field linear at the dark floor, which is
    what the EMVA1288 spatial protocol assumes.
    """
    if n_frames < 2:
        raise ValueError("n_frames must be >= 2")
    gain = np.asarray(prnu_map, dtype=np.float64)
    dsnu = np.asarray(dsnu_map, dtype=np.float64)
    if gain.shape != dsnu.shape:
        raise ValueError("prnu_map and dsnu_map must have the same shape")
    rng = np.random.default_rng(seed)
    # Photo + mean dark current, then Poisson.  DSNU is applied afterwards as a
    # spatially fixed additive offset -- the EMVA1288 definition -- so a
    # zero-mean Gaussian DSNU field is not rectified at the dark floor.
    photo_mean = np.maximum(float(mu_e) * gain + float(dark_mean_e), 0.0)
    height, width = photo_mean.shape
    if use_poisson:
        electrons = rng.poisson(photo_mean, size=(n_frames, height, width)).astype(np.float64)
    else:
        electrons = np.broadcast_to(photo_mean, (n_frames, height, width)).copy()
    electrons = electrons + dsnu
    electrons = electrons + rng.normal(0.0, float(sigma_d_e), size=electrons.shape)
    # Upper-clip at full well.  No lower clip: the EMVA1288 spatial protocol
    # assumes an analog offset / optical black so the dark histogram is linear.
    # (The PTC dark-floor fold is modelled separately in ``dark_floor_clip_mean_var_dn``.)
    electrons = np.minimum(electrons, float(full_well_e))
    return electrons / float(K_e_per_DN) + float(black_level_DN)
