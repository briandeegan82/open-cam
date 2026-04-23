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

from camera_model import load_camera_model, sensor_forward_config_from_camera_model
from exr_multispectral import spectral_buckets_from_exr, trapezoid_weights_nm
from sensor_radiometry import photon_flux_density_from_irradiance
from spectral_sensor_forward import illuminance_lux_from_irradiance, load_qe_curves_rgb, read_csv_curve


def photometry_calibration_scale(repo: Path, cal: dict) -> float:
    """Align EXR irradiance with analytic ``spectral_sensor_forward`` photometry.

    Analytic mode scales chart spectral irradiance by ``irradiance_scale_W_m2nm_per_unit``
    and optionally ``target_illuminance_lux`` (via the illuminant CSV). The renderer uses
    the same relative SPD but arbitrary absolute units unless we apply the same scale here.
    """
    irr_scale = float(cal.get("irradiance_scale_W_m2nm_per_unit", 1.0e-3))
    target_lux = cal.get("target_illuminance_lux", None)
    illum_csv = cal.get("illuminant_override_csv", None)
    illuminance_scale = 1.0
    if target_lux is not None and illum_csv:
        e_wl, e_v = read_csv_curve((repo / illum_csv).resolve())
        ill_in = illuminance_lux_from_irradiance(e_wl, e_v * irr_scale)
        if ill_in > 0:
            illuminance_scale = float(target_lux) / ill_in
        else:
            print("warning: photopic illuminance <= 0; illuminance_scale left at 1", file=sys.stderr)
    elif target_lux is not None and not illum_csv:
        print(
            "warning: target_illuminance_lux set but illuminant_override_csv missing; "
            "using irradiance_scale only",
            file=sys.stderr,
        )
    return irr_scale * illuminance_scale


def qe_stack_on_lambdas(
    repo: Path,
    qe_cfg: dict,
    lambdas_nm: np.ndarray,
) -> np.ndarray:
    """Shape [3, K] QE for R,G,B interpolated at bucket centers."""
    ircf_csv = qe_cfg.get("ircf_csv")
    ircf = np.ones_like(lambdas_nm, dtype=np.float64)
    if ircf_csv:
        i_wl, i_v = read_csv_curve((repo / ircf_csv).resolve())
        ircf = np.interp(lambdas_nm, i_wl, i_v, left=0.0, right=0.0)
    out = []
    q_r, q_g, q_b = load_qe_curves_rgb(repo, qe_cfg)
    for q_wl, q_v in (q_r, q_g, q_b):
        q = np.interp(lambdas_nm, q_wl, q_v, left=0.0, right=0.0)
        out.append(np.clip(q * ircf, 0.0, 1.0))
    return np.stack(out, axis=0).astype(np.float32)


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

    pbrt_cfg = model.get("pbrt_spectral_exr", {}) or {}
    extra_scale = float(pbrt_cfg.get("extra_irradiance_scale", 1.0))
    rad_scale_user = pbrt_cfg.get("radiance_to_irradiance_scale", None)
    rad_mode = str(pbrt_cfg.get("radiance_to_irradiance", "thin_lens")).lower()
    auto_cal_mode = str(pbrt_cfg.get("radiometric_autocalibration", "off")).lower()

    if args.camera_model_config is not None:
        sensor = camera_model.get("sensor", {})
        if not sensor:
            raise RuntimeError("camera model must provide sensor settings")
    else:
        ncfg = yaml.safe_load(noise_path.read_text())
        sensor = ncfg.get("sensor", {})
    qe_cfg = sensor.get("quantum_efficiency", {})
    fill_factor = float(sensor.get("fill_factor", 1.0))
    t_int = float(sensor.get("integration_time_s", 0.01))
    if args.integration_time_s is not None:
        t_int = float(args.integration_time_s)
    f_number = float(sensor.get("f_number", 2.8))
    pixel_pitch_um = float(sensor.get("pixel_pitch_um", 3.45))
    optics_t = float(cal.get("optics_transmittance", 1.0))

    out_npz = (args.out or (repo / cfg.get("output", {}).get("electrons_npz", "out/sensor_forward_electrons.npz"))).resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    exr_path = args.exr if args.exr.is_absolute() else (repo / args.exr).resolve()
    if not exr_path.is_file():
        raise FileNotFoundError(exr_path)

    manifest_raw = args.scene_manifest_json or cfg.get("inputs", {}).get("scene_manifest_json", "scenes/generated/colorchecker_manifest.json")
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

    if rad_scale_user is not None:
        rad_to_e = float(rad_scale_user)
    elif rad_mode in ("thin_lens", "pinhole"):
        rad_to_e = (np.pi * optics_t) / (4.0 * max(1e-12, f_number**2))
    else:
        raise ValueError(
            'model.pbrt_spectral_exr.radiance_to_irradiance must be '
            '"thin_lens" or "pinhole" when radiance_to_irradiance_scale is unset'
        )

    photometry_scale = photometry_calibration_scale(repo, cal)
    E_raw = L.astype(np.float64) * (rad_to_e * extra_scale)
    lam = lambdas.astype(np.float64)
    w = trapezoid_weights_nm(lam).astype(np.float64)

    exr_autocal_scale = 1.0
    if auto_cal_mode not in ("off", "none", "disabled", "false", "0"):
        if auto_cal_mode != "mean_photopic_lux":
            raise ValueError(
                "model.pbrt_spectral_exr.radiometric_autocalibration must be "
                '"off" or "mean_photopic_lux"'
            )
        target_lux = cal.get("target_illuminance_lux", None)
        if target_lux is None:
            print(
                "warning: radiometric_autocalibration requested but calibration.target_illuminance_lux is unset; "
                "skipping EXR autocalibration",
                file=sys.stderr,
            )
        else:
            E_scene_mean = np.mean(E_raw, axis=(0, 1))
            scene_lux = illuminance_lux_from_irradiance(lam, E_scene_mean)
            if scene_lux > 0:
                exr_autocal_scale = float(target_lux) / float(scene_lux)
            else:
                print(
                    "warning: EXR-derived scene illuminance <= 0; skipping EXR autocalibration",
                    file=sys.stderr,
                )

    E_e = E_raw * (photometry_scale * exr_autocal_scale)

    qe = qe_stack_on_lambdas(repo, qe_cfg, lam)

    phi = photon_flux_density_from_irradiance(E_e.astype(np.float64), lam)

    pixel_area = (pixel_pitch_um * 1e-6) ** 2
    geom = t_int * fill_factor * pixel_area

    contrib = np.zeros((yres, xres, 3), dtype=np.float64)
    for c in range(3):
        contrib[:, :, c] = np.sum(phi * (qe[c][np.newaxis, np.newaxis, :] * w[np.newaxis, np.newaxis, :]), axis=2)

    electrons = np.clip(contrib * float(geom), 0.0, None).astype(np.float32)

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
        radiance_to_irradiance_mode=np.array(rad_mode),
        radiance_to_irradiance=np.float64(rad_to_e),
        extra_irradiance_scale=np.float64(extra_scale),
        photometry_calibration_scale=np.float64(photometry_scale),
        exr_radiometric_autocalibration=np.array(auto_cal_mode),
        exr_radiometric_autocalibration_scale=np.float64(exr_autocal_scale),
        calibration_mode=np.array(cal_mode),
        geometry_factor=np.float64(geom),
        optics_transmittance=np.float64(optics_t),
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
