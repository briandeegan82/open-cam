#!/usr/bin/env python3
"""Integrate pbrt-v4 SpectralFilm EXR (spectral radiance buckets) into per-pixel electrons.

PBRT stores ``S0.<lambda>`` channels with metadata ``emissiveUnits`` = W·m⁻²·sr⁻¹
(spectral radiance L [W/(m²·sr·nm)] per bucket). This tool converts L → sensor-plane
spectral irradiance E [W/(m²·nm)] with a thin-lens factor E = (π τ)/(4 N²) L by default,
then applies photon-counting QE integration (same spirit as ``spectral_sensor_forward.py``).

Requires a multispectral OpenEXR (``pip install OpenEXR``); RGB-only renders are unsupported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from camera_model import (
    effective_f_number,
    load_camera_model,
    sensor_forward_config_from_camera_model,
)
from exr_multispectral import spectral_buckets_from_exr, trapezoid_weights_nm
from qe_curves import load_qe_curves_rgb, read_csv_curve
from sensor_radiometry import integrate_spectral_planes, spectral_electron_weights
from spectral_sensor_forward import (
    build_spatial_transmission_map,
    illuminance_lux_from_irradiance,
    spatial_transmission_cfg,
)

_AUTOCAL_OFF = ("off", "none", "disabled", "false", "0")


def photometry_calibration_scale(
    repo: Path,
    cal: dict,
    *,
    auto_cal_mode: str = "off",
) -> float:
    """Align EXR irradiance with analytic ``spectral_sensor_forward`` photometry.

    Analytic mode scales chart spectral irradiance by ``irradiance_scale_W_m2nm_per_unit``
    and optionally ``target_illuminance_lux`` (via the illuminant CSV). The renderer uses
    the same relative SPD but arbitrary absolute units unless we apply the same scale here.

    When ``auto_cal_mode`` is ``"mean_photopic_lux"`` the lux normalisation is handled
    downstream from the rendered EXR; ``illuminant_override_csv`` is not required in
    that case.  In all other modes, omitting ``illuminant_override_csv`` while
    ``target_illuminance_lux`` is set is an error because the lux normalisation would be
    silently skipped.
    """
    irr_scale = float(cal.get("irradiance_scale_W_m2nm_per_unit", 1.0e-3))
    target_lux = cal.get("target_illuminance_lux")
    illum_csv = cal.get("illuminant_override_csv")
    illuminance_scale = 1.0
    if target_lux is not None and illum_csv:
        e_wl, e_v = read_csv_curve((repo / illum_csv).resolve())
        ill_in = illuminance_lux_from_irradiance(e_wl, e_v * irr_scale)
        if ill_in > 0:
            illuminance_scale = float(target_lux) / ill_in
        else:
            print("warning: photopic illuminance <= 0; illuminance_scale left at 1", file=sys.stderr)
    elif target_lux is not None and not illum_csv:
        _autocal_active = auto_cal_mode.lower() not in ("off", "none", "disabled", "false", "0")
        if not _autocal_active:
            raise RuntimeError(
                "calibration.target_illuminance_lux is set but "
                "calibration.illuminant_override_csv is missing. "
                "Without it the lux normalisation is skipped and electron counts will be wrong. "
                "Either add calibration.illuminant_override_csv, remove target_illuminance_lux, "
                "or enable model.pbrt_spectral_exr.radiometric_autocalibration: mean_photopic_lux "
                "for automatic lux calibration from the rendered EXR."
            )
        # autocal active: lux normalisation handled downstream from the EXR; nothing to do here.
    return irr_scale * illuminance_scale


def qe_stack_on_lambdas(
    repo: Path,
    qe_cfg: dict,
    lambdas_nm: np.ndarray,
    *,
    strict_qe_validation: bool = False,
) -> np.ndarray:
    """Shape [3, K] QE for R,G,B interpolated at bucket centers."""
    ircf_csv = qe_cfg.get("ircf_csv")
    ircf = np.ones_like(lambdas_nm, dtype=np.float64)
    if ircf_csv:
        i_wl, i_v = read_csv_curve((repo / ircf_csv).resolve())
        ircf = np.clip(np.interp(lambdas_nm, i_wl, i_v, left=0.0, right=0.0), 0.0, 1.0)
    out = []
    q_r, q_g, q_b = load_qe_curves_rgb(
        repo,
        qe_cfg,
        strict_qe_validation=strict_qe_validation,
    )
    for q_wl, q_v in (q_r, q_g, q_b):
        q = np.interp(lambdas_nm, q_wl, q_v, left=0.0, right=0.0)
        out.append(np.clip(q * ircf, 0.0, 1.0))
    return np.stack(out, axis=0).astype(np.float32)


def magnification_factor(lens_cfg: dict | None, cal: dict) -> float:
    """(1+|m|)² close-focus factor, with m = f/(u−f) at focus distance u.

    Negligible at photographic distances (u >> f); ≈4 for 1:1 macro. Returns 1 when the
    lens has no ``focal_length_mm``.
    """
    lens = lens_cfg or {}
    fl_mm = lens.get("focal_length_mm", None)
    focus_m = float(lens.get("realistic_focus_distance", None) or cal.get("focus_distance_m", 1e6))
    if fl_mm is None or float(fl_mm) <= 0.0 or focus_m <= 0.0:
        return 1.0
    m = float(fl_mm) / (focus_m * 1000.0 - float(fl_mm))
    return (1.0 + abs(m)) ** 2


def spectral_radiance_to_electrons(
    L: np.ndarray,
    lambdas_nm: np.ndarray,
    *,
    repo: Path,
    sensor: dict,
    model: dict,
    lens_cfg: dict | None = None,
    integration_time_s: float | None = None,
    strict_qe_validation: bool = False,
    tag: str = "pbrt_exr",
) -> tuple[np.ndarray, dict]:
    """HxWxK spectral radiance [W/(m²·sr·nm) per EXR unit] → HxWx3 electrons.

    Single radiometric chain shared by this tool and ``apply_emva_noise`` (integrate_qe)::

        E_λ = L_λ · π/(4N²(1+m)²) · τ(λ) · extra · photometry · autocal   [W/m²/nm]
        e_c = Σ_λ E_λ · λ/(hc) · QE_c(λ) · IRCF(λ) · Δλ · A_px · t · FF · V_c(x, y)

    ``model`` is the ``sensor_forward.model`` block (``calibration``,
    ``pbrt_spectral_exr``, ``optics_transmittance_spatial``). Photometric normalisation:
    an ``illuminant_override_csv`` + ``target_illuminance_lux`` pair takes precedence;
    ``radiometric_autocalibration: mean_photopic_lux`` is used only without the CSV, and
    the two are never stacked.
    """
    cal = model.get("calibration", {}) or {}
    pbrt_cfg = model.get("pbrt_spectral_exr", {}) or {}
    extra_scale = float(pbrt_cfg.get("extra_irradiance_scale", 1.0))
    rad_scale_user = pbrt_cfg.get("radiance_to_irradiance_scale", None)
    rad_mode = str(pbrt_cfg.get("radiance_to_irradiance", "thin_lens")).lower()
    auto_cal_mode = str(
        pbrt_cfg.get("radiometric_autocalibration", cal.get("radiometric_autocalibration", "off"))
    ).lower()
    autocal_active = auto_cal_mode not in _AUTOCAL_OFF
    if autocal_active and auto_cal_mode != "mean_photopic_lux":
        raise ValueError(
            "model.pbrt_spectral_exr.radiometric_autocalibration must be "
            '"off" or "mean_photopic_lux"'
        )

    qe_cfg = sensor.get("quantum_efficiency", {}) or {}
    fill_factor = float(sensor.get("fill_factor", 1.0))
    t_int = float(integration_time_s if integration_time_s is not None else sensor.get("integration_time_s", 0.01))
    pixel_pitch_um = float(sensor.get("pixel_pitch_um", 3.45))

    f_number = effective_f_number(sensor, lens_cfg, tag=tag)
    mag = magnification_factor(lens_cfg, cal)
    if rad_scale_user is not None:
        rad_to_e = float(rad_scale_user)
    elif rad_mode in ("thin_lens", "pinhole"):
        rad_to_e = np.pi / (4.0 * max(1e-12, f_number**2) * mag)
    else:
        raise ValueError(
            'model.pbrt_spectral_exr.radiance_to_irradiance must be '
            '"thin_lens" or "pinhole" when radiance_to_irradiance_scale is unset'
        )

    lam = np.asarray(lambdas_nm, dtype=np.float64)
    w = trapezoid_weights_nm(lam).astype(np.float64)
    optics_t = float(cal.get("optics_transmittance", 1.0))
    optics_t_csv = cal.get("optics_transmittance_csv", None)
    if optics_t_csv:
        t_wl, t_v = read_csv_curve((repo / str(optics_t_csv)).resolve())
        tau_lambda = np.clip(np.interp(lam, t_wl, t_v, left=0.0, right=0.0), 0.0, 1.0)
        optics_mode = "spectral_csv"
    else:
        tau_lambda = np.full_like(lam, float(np.clip(optics_t, 0.0, 1.0)))
        optics_mode = "scalar"
    irr_per_lambda = (rad_to_e * extra_scale) * tau_lambda

    target_lux = cal.get("target_illuminance_lux", None)
    exr_autocal_scale = 1.0
    if autocal_active and target_lux is not None and not cal.get("illuminant_override_csv"):
        # Absolute level comes from the render itself; irradiance_scale is not applied.
        photometry_scale = 1.0
        e_mean = np.mean(L, axis=(0, 1), dtype=np.float64) * irr_per_lambda
        scene_lux = illuminance_lux_from_irradiance(lam, e_mean)
        if scene_lux > 0:
            exr_autocal_scale = float(target_lux) / float(scene_lux)
        else:
            print(f"warning [{tag}]: EXR-derived scene illuminance <= 0; skipping autocalibration", file=sys.stderr)
    else:
        if autocal_active and target_lux is None:
            print(
                f"warning [{tag}]: radiometric_autocalibration requested but "
                "calibration.target_illuminance_lux is unset; skipping autocalibration",
                file=sys.stderr,
            )
        photometry_scale = photometry_calibration_scale(repo, cal, auto_cal_mode=auto_cal_mode)
    irr_per_lambda = irr_per_lambda * (photometry_scale * exr_autocal_scale)

    qe = qe_stack_on_lambdas(
        repo,
        qe_cfg,
        lam,
        strict_qe_validation=bool(strict_qe_validation or qe_cfg.get("strict_validation", False)),
    )
    geom = t_int * fill_factor * (pixel_pitch_um * 1e-6) ** 2
    weights = spectral_electron_weights(lam, qe.T, w, irr_per_lambda, geom)
    electrons = np.clip(integrate_spectral_planes(L, weights), 0.0, None)
    spatial_map, spatial_meta = build_spatial_transmission_map(
        L.shape[0],
        L.shape[1],
        spatial_transmission_cfg(model),
        repo=repo,
        wavelength_nm=lam,
        qe_rgb=qe,
    )
    electrons = np.clip(electrons * spatial_map, 0.0, None).astype(np.float32)
    meta = {
        "f_number": f_number,
        "magnification_factor": mag,
        "radiance_to_irradiance_mode": rad_mode,
        "radiance_to_irradiance": rad_to_e,
        "extra_irradiance_scale": extra_scale,
        "photometry_calibration_scale": photometry_scale,
        "exr_radiometric_autocalibration": auto_cal_mode,
        "exr_radiometric_autocalibration_scale": exr_autocal_scale,
        "geometry_factor": geom,
        "optics_transmittance_mode": optics_mode,
        "optics_transmittance_scalar": optics_t,
        "optics_transmittance_mean": float(np.mean(tau_lambda)),
        "optics_transmittance_spatial": spatial_meta,
    }
    return electrons, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    repo_default = Path(__file__).resolve().parent.parent
    ap.add_argument("--repo-root", type=Path, default=repo_default)
    ap.add_argument("--exr", type=Path, required=True, help="Multispectral PBRT EXR (SpectralFilm).")
    ap.add_argument("--camera-model-config", type=Path, default=None, help="Camera model YAML path (preferred).")
    ap.add_argument("--sensor-config", type=Path, default=None, help="sensor_forward.yaml")
    ap.add_argument("--noise-config", type=Path, default=None, help="noise_emva.yaml (sensor + QE paths).")
    ap.add_argument(
        "--scene-manifest-json",
        type=Path,
        default=None,
        help="Optional scene manifest override for EXR resolution check.",
    )
    ap.add_argument("--out", type=Path, default=None, help="Output NPZ (default: sensor_forward output path).")
    ap.add_argument(
        "--target-illuminance-lux",
        type=float,
        default=None,
        help="Optional lux override for calibration.target_illuminance_lux.",
    )
    ap.add_argument(
        "--integration-time-s",
        type=float,
        default=None,
        help="Optional integration time override in seconds (replaces sensor.integration_time_s).",
    )
    ap.add_argument(
        "--strict-qe-validation",
        action="store_true",
        help="Fail on QE wavelength-axis auto-remap or likely R/B swap detection.",
    )
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    if args.camera_model_config is not None:
        cfg_path = args.camera_model_config.resolve()
        camera_model = load_camera_model(cfg_path)
        cfg = sensor_forward_config_from_camera_model(
            camera_model,
            spectral_reference_npz="scenes/generated/spectral_reference_1nm.npz",
            scene_manifest_json="scenes/generated/colorchecker_manifest.json",
            electrons_npz="out/sensor_forward_electrons.npz",
        )
        noise_path = None
    else:
        cfg_path = (args.sensor_config or (repo / "config" / "sensor_forward.yaml")).resolve()
        noise_path = (args.noise_config or (repo / "config" / "noise_emva.yaml")).resolve()
        if not cfg_path.is_file():
            raise FileNotFoundError(cfg_path)
        if not noise_path.is_file():
            raise FileNotFoundError(noise_path)
        cfg = yaml.safe_load(cfg_path.read_text())
    model = cfg.get("model", {})
    cal = model.get("calibration", {}) or {}
    if args.target_illuminance_lux is not None:
        cal["target_illuminance_lux"] = float(args.target_illuminance_lux)
    cal_mode = str(cal.get("mode", "photon_counting")).lower()
    if cal_mode != "photon_counting":
        print(
            "error: pbrt EXR integration supports calibration.mode: photon_counting only "
            f"(got {cal_mode!r}; use spectral_sensor_forward.py for legacy).",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.camera_model_config is not None:
        sensor = camera_model.get("sensor", {})
        if not sensor:
            raise RuntimeError("camera model must provide sensor settings")
        lens_cfg = camera_model.get("lens", {})
    else:
        ncfg = yaml.safe_load(noise_path.read_text())
        sensor = ncfg.get("sensor", {})
        lens_cfg = None

    out_npz = (
        args.out or (repo / cfg.get("output", {}).get("electrons_npz", "out/sensor_forward_electrons.npz"))
    ).resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    exr_path = args.exr if args.exr.is_absolute() else (repo / args.exr).resolve()
    if not exr_path.is_file():
        raise FileNotFoundError(exr_path)

    manifest_raw = args.scene_manifest_json or cfg.get("inputs", {}).get(
        "scene_manifest_json", "scenes/generated/colorchecker_manifest.json"
    )
    manifest_path = (
        manifest_raw.resolve()
        if isinstance(manifest_raw, Path) and manifest_raw.is_absolute()
        else (repo / manifest_raw).resolve()
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"need {manifest_path} for resolution check (run build_colorchecker_scene.py)")
    manifest = json.loads(manifest_path.read_text())
    xres = int(manifest["film"]["xresolution"])
    yres = int(manifest["film"]["yresolution"])

    L, lambdas = spectral_buckets_from_exr(exr_path)
    if L.shape[:2] != (yres, xres):
        raise ValueError(f"EXR size {L.shape[:2]} does not match manifest {yres}x{xres}")

    electrons, meta = spectral_radiance_to_electrons(
        L,
        lambdas,
        repo=repo,
        sensor=sensor,
        model=model,
        lens_cfg=lens_cfg,
        integration_time_s=args.integration_time_s,
        strict_qe_validation=args.strict_qe_validation,
    )
    if meta["f_number"] != float(sensor.get("f_number", 2.8)):
        print(
            f"info: realistic camera — effective f/{meta['f_number']:.3f} from the lens prescription "
            "replaces sensor.f_number for radiance→irradiance conversion.",
            file=sys.stderr,
        )
    lam = np.asarray(lambdas, dtype=np.float64)

    preview = np.clip(electrons, 0.0, None)
    p99 = float(np.percentile(preview, 99.5))
    if p99 > 0:
        preview = np.clip(preview / p99, 0.0, 1.0)
    preview_u8 = np.rint(preview * 255.0).astype(np.uint8)

    np.savez_compressed(
        out_npz,
        electrons_rgb=electrons,
        wavelength_nm=lam.astype(np.float32),
        source=np.array("pbrt_spectral_exr"),
        exr_path=np.array(str(exr_path)),
        calibration_mode=np.array(cal_mode),
        **{
            k: (np.array(json.dumps(v)) if isinstance(v, dict) else np.array(v) if isinstance(v, str) else np.float64(v))
            for k, v in meta.items()
        },
    )
    try:
        import imageio.v3 as iio

        iio.imwrite(out_npz.with_suffix(".png"), preview_u8)
    except Exception:
        pass

    print(f"Wrote electrons npz (from PBRT spectral EXR): {out_npz}")
    print(f"Wrote preview png: {out_npz.with_suffix('.png')}")


if __name__ == "__main__":
    main()
