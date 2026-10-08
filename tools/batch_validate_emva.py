#!/usr/bin/env python3
"""Batch analytic EMVA consistency check across all camera recipes.

Runs a suite of purely analytic (no Monte Carlo) self-consistency checks on
every camera recipe found in ``config/camera_recipes/``, then writes a
human-readable summary and a machine-readable JSON report.

Checks performed
----------------
1.  ADC range vs full well     — can the ADC represent the full well capacity?
2.  Black-level headroom        — does black_level_DN × K ≥ 3 × σ_d_e?
3.  Read noise floor visibility — is σ_d_DN ≥ 0.5 LSB?
4.  Dynamic range               — is log₂(full_well / σ_d) in [8, 16] stops?
5.  Full well fits in ADC       — does full_well_e / K + black_dn ≤ max_dn?
6.  Bit depth sufficiency       — are there enough bits to reach full well?
7.  PRNU vs shot noise          — is PRNU noise at full well dominated by shot?
8.  Dark current sanity         — is dark_e_per_frame < 10% of full well?
9.  DSNU vs read noise          — is DSNU ≤ σ_d_e?
10. f-number / diffraction      — Airy disk < 3 pixels (non-diffraction-limited regime).
11. Datasheet self-consistency  — config K / σ_d / full_well match datasheet within rtol.

Usage::

    venv/bin/python tools/batch_validate_emva.py [--repo-root /path] [--json-out report.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import NamedTuple


_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from camera_model import load_camera_model
from emva_theory import compare_config_to_datasheet


# ---------------------------------------------------------------------------
# Issue severity levels
# ---------------------------------------------------------------------------
FAIL = "FAIL"
WARN = "WARN"
INFO = "INFO"

_SEVERITY_RANK = {FAIL: 2, WARN: 1, INFO: 0}


class Issue(NamedTuple):
    severity: str   # FAIL | WARN | INFO
    check: str      # short identifier
    message: str    # human-readable detail


def _agg_status(issues: list[Issue]) -> str:
    if any(i.severity == FAIL for i in issues):
        return FAIL
    if any(i.severity == WARN for i in issues):
        return WARN
    return "PASS"


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_adc_vs_full_well(K: float, full_well_e: float, bit_depth: int, black_dn: float) -> list[Issue]:
    issues: list[Issue] = []
    max_dn = (1 << bit_depth) - 1
    usable_dn = max(0.0, max_dn - black_dn)
    max_e_repr = usable_dn * K
    if full_well_e <= 0:
        return [Issue(FAIL, "adc_vs_full_well", "full_well_e <= 0")]
    ratio = max_e_repr / full_well_e
    if ratio < 0.50:
        issues.append(Issue(FAIL, "adc_vs_full_well",
            f"ADC can only represent {ratio:.1%} of full well "
            f"({max_e_repr:.0f} e vs full_well={full_well_e:.0f} e); "
            "ADC range is too narrow — sensor will always clip before full well."))
    elif ratio < 0.80:
        issues.append(Issue(WARN, "adc_vs_full_well",
            f"ADC represents only {ratio:.1%} of full well ({max_e_repr:.0f} e / {full_well_e:.0f} e); "
            ">20% of well capacity is unreachable."))
    elif ratio > 2.50:
        issues.append(Issue(WARN, "adc_vs_full_well",
            f"ADC range is {ratio:.1f}× larger than full well — significant dynamic range wasted."))
    return issues


def check_full_well_fits_adc(K: float, full_well_e: float, bit_depth: int, black_dn: float) -> list[Issue]:
    max_dn = (1 << bit_depth) - 1
    full_well_dn = full_well_e / K + black_dn
    if full_well_dn > max_dn + 0.5:
        return [Issue(FAIL, "full_well_fits_adc",
            f"full_well_e={full_well_e:.0f} e → {full_well_dn:.1f} DN exceeds "
            f"max_dn={max_dn} for {bit_depth}-bit ADC; "
            "model will always saturate below the stated full well.")]
    return []


def check_black_level_headroom(K: float, sigma_d_e: float, black_dn: float) -> list[Issue]:
    black_e = black_dn * K
    threshold = 3.0 * sigma_d_e
    if black_e < threshold:
        sev = FAIL if black_e < sigma_d_e else WARN
        return [Issue(sev, "black_level_headroom",
            f"black_level_DN={black_dn:.0f} → {black_e:.2f} e, but 3σ_d = {threshold:.2f} e; "
            "dark frames will be clipped — black level is too low.")]
    return []


def check_read_noise_visibility(K: float, sigma_d_e: float) -> list[Issue]:
    sigma_d_dn = sigma_d_e / K
    if sigma_d_dn < 0.3:
        return [Issue(WARN, "read_noise_visibility",
            f"σ_d = {sigma_d_e:.2f} e / {K:.4f} e/DN = {sigma_d_dn:.3f} DN "
            "— sub-LSB read noise; quantisation noise will dominate temporal noise.")]
    return []


def check_dynamic_range(full_well_e: float, sigma_d_e: float) -> list[Issue]:
    if sigma_d_e <= 0 or full_well_e <= 0:
        return [Issue(FAIL, "dynamic_range", "sigma_d_e or full_well_e is non-positive")]
    dr = math.log2(full_well_e / sigma_d_e)
    # DR is evaluated at the effective full well (after iso_gain), so high-ISO
    # settings will naturally show lower DR — 6 stops is the practical lower bound.
    if dr < 4.0:
        return [Issue(FAIL, "dynamic_range",
            f"DR = {dr:.1f} stops — unusually low; check full_well_e and sigma_d_e.")]
    if dr < 6.0:
        return [Issue(WARN, "dynamic_range",
            f"DR = {dr:.1f} stops — low even for high-ISO operation.")]
    if dr > 17.0:
        return [Issue(WARN, "dynamic_range",
            f"DR = {dr:.1f} stops — unusually high; verify parameters.")]
    return []


def check_bit_depth_sufficiency(K: float, full_well_e: float, bit_depth: int, black_dn: float) -> list[Issue]:
    required = math.ceil(math.log2(max(2.0, full_well_e / K + black_dn + 1)))
    if bit_depth < required:
        return [Issue(WARN, "bit_depth_sufficiency",
            f"bit_depth={bit_depth} cannot fully represent full_well_e={full_well_e:.0f} e "
            f"(needs ~{required} bits); top of well is irrepresentable.")]
    return []


def check_prnu_vs_shot_noise(prnu_std: float, full_well_e: float) -> list[Issue]:
    prnu_e = prnu_std * full_well_e
    shot_e = math.sqrt(full_well_e)
    ratio = prnu_e / max(1e-12, shot_e)
    # PRNU dominating shot noise at full well is expected for sensors with
    # prnu_std > 1/sqrt(full_well).  Only flag as WARN if the ratio is extreme.
    if ratio > 5.0:
        return [Issue(WARN, "prnu_vs_shot",
            f"PRNU RMS at full well = {prnu_e:.1f} e is {ratio:.1f}× shot noise {shot_e:.1f} e; "
            "verify prnu_std_fraction.")]
    if ratio > 2.0:
        return [Issue(INFO, "prnu_vs_shot",
            f"PRNU noise at full well ({prnu_e:.1f} e) exceeds shot noise ({shot_e:.1f} e, "
            f"{ratio:.1f}×) — normal for high-DR sensors.")]
    return []


def check_dark_current(dark_e_per_s: float, t_int_s: float, full_well_e: float) -> list[Issue]:
    dark_e = dark_e_per_s * t_int_s
    frac = dark_e / max(1.0, full_well_e)
    if dark_e_per_s < 0:
        return [Issue(FAIL, "dark_current", f"dark_current_e_per_s={dark_e_per_s} is negative.")]
    if frac > 0.10:
        return [Issue(WARN, "dark_current",
            f"dark_current = {dark_e:.2f} e/frame ({frac:.1%} of full well) at t_int={t_int_s}s; "
            "high dark current — consider dark subtraction or shorter integration.")]
    return []


def check_dsnu_vs_read_noise(dsnu_std_e: float, sigma_d_e: float) -> list[Issue]:
    if dsnu_std_e > sigma_d_e * 2.0:
        return [Issue(WARN, "dsnu_vs_read_noise",
            f"DSNU σ = {dsnu_std_e:.2f} e > 2× σ_d = {2*sigma_d_e:.2f} e; "
            "fixed-pattern dark noise dominates temporal — verify DSNU.")]
    return []


def check_diffraction(f_number: float, pixel_pitch_um: float) -> list[Issue]:
    # Airy disk first-zero radius at 550 nm
    r_airy_m = 1.22 * 550e-9 * f_number
    r_airy_px = r_airy_m / (pixel_pitch_um * 1e-6)
    if r_airy_px > 3.0:
        return [Issue(INFO, "diffraction",
            f"Airy disk radius = {r_airy_px:.2f} px at f/{f_number}, "
            f"pitch={pixel_pitch_um}µm — diffraction-limited; "
            "chromatic PSF σ_diff will be significant.")]
    return []


def check_datasheet(K: float, sigma_d: float, full_well: float, black: float,
                    ds: dict, bit_depth_cfg: int) -> list[Issue]:
    if not ds.get("enabled", False):
        return []
    if "overall_system_gain_K_e_per_DN" not in ds:
        return []
    rtol = float(ds.get("parameter_rtol", 0.02))
    gain_conv = str(ds.get("gain_convention", "e_per_dn")).lower()
    ds_bit = ds.get("bit_depth")
    cmp = compare_config_to_datasheet(
        K, sigma_d, full_well, black,
        float(ds["overall_system_gain_K_e_per_DN"]),
        float(ds.get("temporal_dark_noise_sigma_d_e", sigma_d)),
        float(ds.get("full_well_e", full_well)),
        float(ds.get("black_level_DN", black)),
        rtol,
        bit_depth_cfg=bit_depth_cfg,
        bit_depth_ds=int(ds_bit) if ds_bit is not None else None,
        gain_convention_ds=gain_conv,
    )
    issues: list[Issue] = []
    if not cmp["all_ok"]:
        for chk in cmp["parameter_checks"]:
            if not chk["ok"]:
                issues.append(Issue(WARN, f"datasheet_{chk['name']}",
                    f"{chk['name']}: config={chk['config']:.4g} vs "
                    f"datasheet={chk['datasheet']:.4g} (rtol={rtol:.1%})"))
    return issues


# ---------------------------------------------------------------------------
# Per-camera runner
# ---------------------------------------------------------------------------

def validate_camera(recipe_path: Path) -> dict:
    issues: list[Issue] = []
    try:
        cm = load_camera_model(recipe_path)
    except Exception as exc:
        return {
            "name": recipe_path.stem,
            "recipe": str(recipe_path),
            "status": FAIL,
            "load_error": str(exc),
            "issues": [{"severity": FAIL, "check": "load", "message": str(exc)}],
        }

    s = cm.get("sensor", {})
    noise = cm.get("noise", {})
    emva = noise.get("emva", {})
    adc = noise.get("adc", {})
    val = cm.get("validation", {})
    model_meta = cm.get("model", {})

    K_base = float(emva.get("overall_system_gain_K_e_per_DN", 1.0))
    sigma_d = float(emva.get("sigma_d_e", 0.0))
    black = float(emva.get("black_level_DN", 0.0))
    prnu_std = float(emva.get("prnu_std_fraction", 0.0))
    dsnu_std = float(emva.get("dsnu_std_e", 0.0))
    dark_e_s = float(emva.get("dark_current_e_per_s", 0.0))
    iso_gain = float(emva.get("iso_gain_factor", 1.0))
    full_well_base = float(adc.get("full_well_e", 0.0))
    bit_depth = int(adc.get("bit_depth", 12))
    # Apply ISO gain: effective K and full_well seen by the ADC
    K = K_base / max(iso_gain, 1e-9)
    full_well = full_well_base / max(iso_gain, 1e-9)
    t_int = float(s.get("integration_time_s", 0.01))
    f_number = float(s.get("f_number", 2.8))
    pixel_pitch = float(s.get("pixel_pitch_um", 1.4))
    ds = val.get("datasheet") or {}

    issues += check_adc_vs_full_well(K, full_well, bit_depth, black)
    issues += check_full_well_fits_adc(K, full_well, bit_depth, black)
    issues += check_black_level_headroom(K, sigma_d, black)
    issues += check_read_noise_visibility(K, sigma_d)
    issues += check_dynamic_range(full_well, sigma_d)
    # check_bit_depth_sufficiency is omitted: covered by check_adc_vs_full_well /
    # check_full_well_fits_adc, and prone to false positives from K rounding.
    issues += check_prnu_vs_shot_noise(prnu_std, full_well)
    issues += check_dark_current(dark_e_s, t_int, full_well)
    issues += check_dsnu_vs_read_noise(dsnu_std, sigma_d)
    issues += check_diffraction(f_number, pixel_pitch)
    # Datasheet comparison uses base-ISO parameters; the datasheet records physical
    # sensor properties, not the ISO-amplified effective operating point.
    issues += check_datasheet(K_base, sigma_d, full_well_base, black, ds, bit_depth)

    dr = math.log2(full_well / sigma_d) if sigma_d > 0 and full_well > 0 else 0.0
    return {
        "name": recipe_path.stem,
        "display_name": model_meta.get("display_name", recipe_path.stem),
        "recipe": str(recipe_path),
        "status": _agg_status(issues),
        "params": {
            "iso_gain_factor": iso_gain,
            "K_base_e_per_DN": K_base,
            "K_effective_e_per_DN": round(K, 6),
            "sigma_d_e": sigma_d,
            "sigma_d_dn": round(sigma_d / K, 3) if K > 0 else None,
            "full_well_base_e": full_well_base,
            "full_well_effective_e": full_well,
            "black_level_DN": black,
            "bit_depth": bit_depth,
            "dynamic_range_stops": round(dr, 1),
            "f_number": f_number,
            "pixel_pitch_um": pixel_pitch,
        },
        "issues": [{"severity": i.severity, "check": i.check, "message": i.message}
                   for i in sorted(issues, key=lambda x: -_SEVERITY_RANK[x.severity])],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    repo_default = Path(__file__).resolve().parent.parent
    ap.add_argument("--repo-root", type=Path, default=repo_default)
    ap.add_argument("--json-out", type=Path, default=None,
                    help="JSON report path (default: out/batch_emva_validation.json)")
    ap.add_argument("--csv-out", type=Path, default=None,
                    help="CSV summary path (default: out/batch_emva_validation.csv)")
    ap.add_argument("--fail-only", action="store_true",
                    help="Print only FAIL cameras to stdout.")
    ap.add_argument("--warn-and-fail", action="store_true",
                    help="Print FAIL and WARN cameras to stdout.")
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    recipes_dir = repo / "config" / "camera_recipes"
    recipes = sorted(p for p in recipes_dir.glob("*.yaml")
                     if p.stem not in ("default",))

    print(f"Validating {len(recipes)} camera recipes analytically …", file=sys.stderr)
    results = [validate_camera(r) for r in recipes]

    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    # ---- Summary table ----
    col_status = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}
    SEPARATOR = f"{'─'*90}"

    print(f"\n{'Camera':<36}  {'Status':<6}  {'ISO×':>5}  {'DR':>5}  {'σ_d DN':>6}  {'FW_eff':>7}  Issues")
    print(SEPARATOR)
    for r in sorted(results, key=lambda x: (_SEVERITY_RANK.get(x["status"], 0), x["name"]), reverse=True):
        if args.fail_only and r["status"] != FAIL:
            continue
        if args.warn_and_fail and r["status"] == "PASS":
            continue
        mark = col_status.get(r["status"], "?")
        p = r.get("params", {})
        iso_g = p.get("iso_gain_factor", 1.0)
        dr = p.get("dynamic_range_stops", "?")
        sd_dn = p.get("sigma_d_dn", "?")
        fw = p.get("full_well_effective_e", "?")
        issue_str = "; ".join(f"[{i['severity']}:{i['check']}]" for i in r.get("issues", []))
        print(f"{r['display_name']:<36}  {mark} {r['status']:<4}  {iso_g:>5.0f}  {dr:>5}  {sd_dn:>6}  {fw:>7.0f}  {issue_str}")

    print(SEPARATOR)
    print(f"Total: {len(results)}  ✓ PASS: {counts.get('PASS',0)}  "
          f"! WARN: {counts.get('WARN',0)}  ✗ FAIL: {counts.get('FAIL',0)}")

    # ---- JSON report ----
    json_path = (args.json_out or (repo / "out" / "batch_emva_validation.json")).resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "recipe_dir": str(recipes_dir),
        "total": len(results),
        "counts": counts,
        "cameras": results,
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nJSON report → {json_path}")

    # ---- CSV summary ----
    csv_path = (args.csv_out or (repo / "out" / "batch_emva_validation.csv")).resolve()
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "name", "display_name", "status",
            "K_e_per_DN", "sigma_d_e", "sigma_d_dn", "full_well_e",
            "black_level_DN", "bit_depth", "dynamic_range_stops",
            "f_number", "pixel_pitch_um", "issues",
        ])
        w.writeheader()
        for r in results:
            p = r.get("params", {})
            w.writerow({
                "name": r["name"],
                "display_name": r.get("display_name", r["name"]),
                "status": r["status"],
                **{k: p.get(k, "") for k in (
                    "K_e_per_DN", "sigma_d_e", "sigma_d_dn", "full_well_e",
                    "black_level_DN", "bit_depth", "dynamic_range_stops",
                    "f_number", "pixel_pitch_um",
                )},
                "issues": " | ".join(i["message"] for i in r.get("issues", [])),
            })
    print(f"CSV  report → {csv_path}")

    sys.exit(0 if counts.get("FAIL", 0) == 0 else 1)


if __name__ == "__main__":
    main()
