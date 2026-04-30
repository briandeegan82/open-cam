#!/usr/bin/env python3
"""Check EMVA-style temporal noise: analytic vs Monte Carlo; model vs datasheet targets.

Run from repository root::

    venv/bin/python tools/validate_emva_model.py

Uses ``config/camera_recipes/default.yaml`` (``validation`` + noise sections).
Replace the ``validation.datasheet`` block with values from an EMVA1288 report or sensor
datasheet when calibrating a real device.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from camera_model import load_camera_model, noise_config_from_camera_model

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from emva_theory import (
    compare_config_to_datasheet,
    dark_floor_clip_mean_var_dn,
    monte_carlo_temporal_dn_stats,
    photon_transfer_curve_checks,
)


def _effective_mean_tol_dn(fixed_atol: float, pred_var_dn: float, n_trials: int) -> float:
    # Monte-Carlo mean uncertainty is sqrt(var / n). Use 3σ to avoid false
    # negatives from statistical fluctuation when fixed tolerances are tight.
    se = float(np.sqrt(max(pred_var_dn, 0.0) / max(1, n_trials)))
    return max(float(fixed_atol), 3.0 * se)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    repo_default = Path(__file__).resolve().parent.parent
    ap.add_argument("--repo-root", type=Path, default=repo_default)
    ap.add_argument(
        "--camera-model-config",
        type=Path,
        default=None,
        help="Default: config/camera_recipes/default.yaml",
    )
    ap.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write full report JSON (default: out/emva_validation_report.json).",
    )
    ap.add_argument(
        "--strict-calibration",
        action="store_true",
        help=(
            "Fail validation when EMVA parameters are heuristic or when "
            "validation.datasheet.source is missing."
        ),
    )
    ap.add_argument(
        "--calibration-tier-policy",
        type=str,
        default=None,
        choices=("strict", "semi_strict", "research"),
        help=(
            "Calibration-tier gate policy. "
            "strict: require measured tier + datasheet source; "
            "semi_strict: allow measured or inferred + datasheet source; "
            "research: informational only."
        ),
    )
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    val_path = (args.camera_model_config or (repo / "config" / "camera_recipes" / "default.yaml")).resolve()
    if not val_path.is_file():
        print(f"error: missing {val_path}", file=sys.stderr)
        sys.exit(2)

    camera_model = load_camera_model(val_path)
    val = camera_model.get("validation", {})
    noise = noise_config_from_camera_model(
        camera_model,
        linear_rgb_in="out/colorchecker_spectral.exr",
        raw_out="out/colorchecker_noisy.raw16",
    )
    emva = noise.get("emva", {})
    adc = noise.get("adc", {})

    K = float(emva.get("overall_system_gain_K_e_per_DN", 0.08))
    sigma_d = float(emva.get("sigma_d_e", 2.0))
    black = float(emva.get("black_level_DN", 64.0))
    use_poisson = bool(emva.get("use_poisson_shot_noise", True))
    if "full_well_e" not in adc:
        raise KeyError(
            "adc.full_well_e is required but not set in the sensor config. "
            "This value determines the ADC clipping point; there is no safe generic default."
        )
    full_well = float(adc["full_well_e"])
    bit_depth_cfg = int(adc.get("bit_depth", 12))

    mc_trials = int(val.get("monte_carlo_trials", 20_000))
    seed = int(val.get("random_seed", 0))
    var_rtol = float(val.get("variance_rtol", 0.06))
    mean_atol = float(val.get("mean_abs_dn_atol", 0.12))

    mu_levels = val.get("ptc_mu_e_levels")
    if mu_levels is None:
        # Skip very low μ where lower electron clip dominates (handled separately at μ=0).
        mu_levels = [0.0, 200.0, 1000.0, 4000.0, 12000.0]

    ptc_rows = photon_transfer_curve_checks(
        np.asarray(mu_levels, dtype=np.float64),
        sigma_d,
        K,
        black,
        use_poisson=use_poisson,
        full_well_e=None,
        n_trials=mc_trials,
        seed=seed,
        variance_rtol=var_rtol,
        mean_atol=mean_atol,
    )
    for row in ptc_rows:
        tol_eff = _effective_mean_tol_dn(mean_atol, float(row["pred_var_dn"]), mc_trials)
        row["mean_tol_dn"] = tol_eff
        row["mean_ok"] = bool(abs(float(row["mc_mean_dn"]) - float(row["pred_mean_dn"])) <= tol_eff)
    ptc_ok = all(bool(r["mean_ok"]) and bool(r["var_ok"]) for r in ptc_rows)

    dark_mean_pred, dark_var_pred = dark_floor_clip_mean_var_dn(sigma_d, K, black)
    dark_mean_mc, dark_var_mc = monte_carlo_temporal_dn_stats(
        0.0,
        sigma_d,
        K,
        black,
        use_poisson=use_poisson,
        full_well_e=None,
        n_trials=mc_trials,
        seed=seed + 99,
    )
    dark_var_ok = abs(dark_var_mc - dark_var_pred) <= var_rtol * max(dark_var_pred, 1e-12)
    dark_mean_tol = _effective_mean_tol_dn(mean_atol, dark_var_pred, mc_trials)
    dark_mean_ok = abs(dark_mean_mc - dark_mean_pred) <= dark_mean_tol

    ds = val.get("datasheet") or {}
    ds_enabled = bool(ds.get("enabled", True))
    ds_block = {k: v for k, v in ds.items() if k not in ("enabled", "parameter_rtol")}
    param_cmp: dict | None = None
    skip_ds_reason = None
    source = camera_model.get("source", {}) or {}
    emva_method = str(source.get("emva_param_method", "")).strip().lower()
    ds_source = str(ds.get("source", "")).strip()
    strict_calibration = bool(args.strict_calibration or val.get("strict_calibration", False))
    tier_policy = str(
        args.calibration_tier_policy
        or val.get("calibration_tier_policy", "research")
    ).strip().lower()
    if tier_policy not in ("strict", "semi_strict", "research"):
        raise ValueError(
            "validation.calibration_tier_policy must be one of "
            '"strict", "semi_strict", "research"'
        )
    calibration_tier = str(source.get("calibration_tier", "")).strip().lower()
    if not calibration_tier:
        calibration_tier = "unspecified"
    calibration_quality = {
        "emva_param_method": emva_method or None,
        "calibration_tier": calibration_tier,
        "calibration_tier_policy": tier_policy,
        "datasheet_source_present": bool(ds_source),
        "strict_calibration_enabled": strict_calibration,
        "issues": [],
    }
    if emva_method.startswith("heuristic"):
        calibration_quality["issues"].append("heuristic_emva_parameters")
    if ds_enabled and not ds_source:
        calibration_quality["issues"].append("missing_datasheet_source")
    if ds_enabled and emva_method.startswith("heuristic") and not ds_source:
        skip_ds_reason = (
            "skipped datasheet comparison: heuristic EMVA parameters detected and "
            "validation.datasheet.source is missing"
        )
    if ds_enabled and skip_ds_reason is None and "overall_system_gain_K_e_per_DN" in ds_block:
        rtol = float(ds.get("parameter_rtol", 0.02))
        ds_gain_conv = str(ds.get("gain_convention", "e_per_dn")).strip().lower()
        ds_bit_depth = ds.get("bit_depth", None)
        ds_bit_depth_int = int(ds_bit_depth) if ds_bit_depth is not None else None
        param_cmp = compare_config_to_datasheet(
            K,
            sigma_d,
            full_well,
            black,
            float(ds_block["overall_system_gain_K_e_per_DN"]),
            float(ds_block.get("temporal_dark_noise_sigma_d_e", sigma_d)),
            float(ds_block.get("full_well_e", full_well)),
            float(ds_block.get("black_level_DN", black)),
            rtol,
            bit_depth_cfg=bit_depth_cfg,
            bit_depth_ds=ds_bit_depth_int,
            gain_convention_ds=ds_gain_conv,
        )

    report = {
        "validation_config": str(val_path),
        "noise_config": "embedded_in_camera_model",
        "monte_carlo_trials": mc_trials,
        "model": {
            "K_e_per_DN": K,
            "sigma_d_e": sigma_d,
            "black_level_DN": black,
            "full_well_e": full_well,
            "use_poisson_shot_noise": use_poisson,
        },
        "dark_frame": {
            "pred_mean_dn": dark_mean_pred,
            "mc_mean_dn": dark_mean_mc,
            "mean_ok": bool(dark_mean_ok),
            "mean_tol_dn": dark_mean_tol,
            "pred_var_dn": dark_var_pred,
            "mc_var_dn": dark_var_mc,
            "var_ok": bool(dark_var_ok),
        },
        "photon_transfer_curve": ptc_rows,
        "ptc_all_ok": bool(ptc_ok),
        "datasheet_comparison": param_cmp,
        "datasheet_comparison_skipped_reason": skip_ds_reason,
        "calibration_quality": calibration_quality,
    }
    calibration_ok = True
    calibration_failure_reasons: list[str] = []
    policy_failure_reasons: list[str] = []
    if tier_policy == "strict":
        if calibration_tier != "measured":
            policy_failure_reasons.append("strict_policy_requires_measured_tier")
        if ds_enabled and not ds_source:
            policy_failure_reasons.append("strict_policy_requires_datasheet_source")
    elif tier_policy == "semi_strict":
        if calibration_tier not in ("measured", "inferred"):
            policy_failure_reasons.append("semi_strict_policy_disallows_heuristic_or_unspecified_tier")
        if ds_enabled and not ds_source:
            policy_failure_reasons.append("semi_strict_policy_requires_datasheet_source")
    calibration_quality["policy_issues"] = policy_failure_reasons
    if strict_calibration and calibration_quality["issues"]:
        calibration_ok = False
        calibration_failure_reasons = list(calibration_quality["issues"])
    if policy_failure_reasons:
        calibration_ok = False
        calibration_failure_reasons = calibration_failure_reasons + policy_failure_reasons
    report["calibration_ok"] = calibration_ok
    report["calibration_failure_reasons"] = calibration_failure_reasons
    report["all_ok"] = bool(
        ptc_ok
        and dark_var_ok
        and dark_mean_ok
        and (param_cmp is None or param_cmp["all_ok"])
        and calibration_ok
    )

    json_out = args.json_out
    if json_out is None:
        json_out = repo / "out" / "emva_validation_report.json"
    json_out = Path(json_out).resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2) + "\n")

    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "all_ok",
                    "ptc_all_ok",
                    "dark_frame",
                    "datasheet_comparison",
                    "datasheet_comparison_skipped_reason",
                )
            },
            indent=2,
        )
    )
    print(f"Wrote {json_out}")
    sys.exit(0 if report["all_ok"] else 1)


if __name__ == "__main__":
    main()
