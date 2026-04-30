#!/usr/bin/env python3
"""Compute a spectral forward model from chart spectra to per-pixel electrons.

Pixel→world X uses the negated horizontal NDC term so the chart plane matches pbrt-v4
LookAt (camera +X along world −X); see README.md (“Patch layout and camera axes”).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

from camera_model import load_camera_model, sensor_forward_config_from_camera_model
from sensor_radiometry import (
    cos4_vignetting_from_pinhole,
    cosine_illuminance_factor,
    photon_flux_density_from_irradiance,
)

# NumPy compatibility helper (np.trapezoid in newer versions, np.trapz in older).
if hasattr(np, "trapezoid"):
    _trapz = np.trapezoid
else:
    _trapz = np.trapz

# CIE 1924 photopic luminous efficiency V(lambda), tabulated at 5nm (380..780).
_V_WL_5NM = np.arange(380.0, 781.0, 5.0, dtype=np.float64)
_V_5NM = np.array(
    [
        0.000039,
        0.000064,
        0.000120,
        0.000217,
        0.000396,
        0.000640,
        0.001210,
        0.002180,
        0.004000,
        0.007300,
        0.011600,
        0.016840,
        0.023000,
        0.029800,
        0.038000,
        0.048000,
        0.060000,
        0.073900,
        0.090980,
        0.112600,
        0.139020,
        0.169300,
        0.208020,
        0.258600,
        0.323000,
        0.407300,
        0.503000,
        0.608200,
        0.710000,
        0.793200,
        0.862000,
        0.914850,
        0.954000,
        0.980300,
        0.995000,
        1.000000,
        0.995000,
        0.978600,
        0.952000,
        0.915400,
        0.870000,
        0.816300,
        0.757000,
        0.694900,
        0.631000,
        0.566800,
        0.503000,
        0.441200,
        0.381000,
        0.321000,
        0.265000,
        0.217000,
        0.175000,
        0.138200,
        0.107000,
        0.081600,
        0.061000,
        0.044580,
        0.032000,
        0.023200,
        0.017000,
        0.011920,
        0.008210,
        0.005723,
        0.004102,
        0.002929,
        0.002091,
        0.001484,
        0.001047,
        0.000740,
        0.000520,
        0.000361,
        0.000249,
        0.000172,
        0.000120,
        0.000083,
        0.000057,
        0.000039,
        0.000027,
        0.000018,
        0.000012,
    ],
    dtype=np.float64,
)


def read_csv_curve(path: Path, *, strict_wavelength_axis: bool = False) -> tuple[np.ndarray, np.ndarray]:
    wl: list[float] = []
    val: list[float] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r",\s*", line, maxsplit=1)
        if len(parts) != 2:
            continue
        wl.append(float(parts[0]))
        val.append(float(parts[1]))
    if not wl:
        raise ValueError(f"no data in {path}")
    w = np.asarray(wl, dtype=np.float64)
    v = np.asarray(val, dtype=np.float64)

    ok = np.isfinite(w) & np.isfinite(v)
    w = w[ok]
    v = v[ok]
    if w.size == 0:
        raise ValueError(f"no finite samples in {path}")

    # Some imported camera QE CSVs are stored in a normalized x-domain (roughly 0..1)
    # instead of nanometers. Detect and map to visible wavelengths for stable integration.
    if float(np.max(w)) <= 10.0:
        wmin = float(np.min(w))
        wmax = float(np.max(w))
        if wmax - wmin <= 1e-12:
            raise ValueError(f"invalid wavelength axis in {path}: near-constant normalized domain")
        if strict_wavelength_axis:
            raise ValueError(
                "strict QE validation: normalized wavelength axis detected "
                f"in {path}; provide explicit wavelength-in-nm CSV"
            )
        w = 380.0 + (w - wmin) * (400.0 / (wmax - wmin))
        print(f"warning: mapped normalized wavelength axis to 380..780 nm for {path}", file=sys.stderr)

    idx = np.argsort(w)
    w = w[idx]
    v = v[idx]
    return w, v


def load_qe_curves_rgb(
    repo: Path,
    qe_cfg: dict,
    *,
    strict_qe_validation: bool = False,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Load QE curves and auto-correct known import artifact (R/B swapped)."""
    r = read_csv_curve((repo / qe_cfg["red_csv"]).resolve(), strict_wavelength_axis=strict_qe_validation)
    g = read_csv_curve((repo / qe_cfg["green_csv"]).resolve(), strict_wavelength_axis=strict_qe_validation)
    b = read_csv_curve((repo / qe_cfg["blue_csv"]).resolve(), strict_wavelength_axis=strict_qe_validation)
    r_peak = float(r[0][int(np.argmax(r[1]))])
    g_peak = float(g[0][int(np.argmax(g[1]))])
    b_peak = float(b[0][int(np.argmax(b[1]))])
    # Expected Bayer-like ordering: blue peak < green peak < red peak.
    # Some imported model CSVs are inverted between R/B.
    if r_peak < g_peak and b_peak > g_peak:
        if strict_qe_validation:
            raise ValueError(
                "strict QE validation: detected likely QE red/blue inversion; "
                "fix channel assignments in QE CSVs"
            )
        print(
            "warning: detected likely QE red/blue inversion; swapping channels "
            f"(red_peak={r_peak:.1f}nm, green_peak={g_peak:.1f}nm, blue_peak={b_peak:.1f}nm)",
            file=sys.stderr,
        )
        r, b = b, r
    return r, g, b


def illuminance_lux_from_irradiance(
    wavelength_nm: np.ndarray,
    spectral_irradiance_W_m2nm: np.ndarray,
) -> float:
    """Compute photopic illuminance [lux] from spectral irradiance [W/(m^2*nm)]."""
    v = np.interp(wavelength_nm, _V_WL_5NM, _V_5NM, left=0.0, right=0.0)
    return float(683.0 * _trapz(spectral_irradiance_W_m2nm * v, wavelength_nm))


def project_world_to_pixel(
    x: np.ndarray,
    y: np.ndarray,
    cam_dist: float,
    fov_deg: float,
    xres: int,
    yres: int,
) -> tuple[np.ndarray, np.ndarray]:
    aspect = xres / yres
    tan_half = np.tan(np.deg2rad(fov_deg) * 0.5)
    z = cam_dist
    # PBRT LookAt: right = cross(up, view); with up=+Y and view=-Z, camera +X is world -X.
    x_ndc = -x / (z * tan_half * aspect)
    y_ndc = y / (z * tan_half)
    u = (x_ndc + 1.0) * 0.5 * xres
    v = (1.0 - y_ndc) * 0.5 * yres
    return u, v


def _radial_map(yres: int, xres: int, edge_factor: float, exponent: float) -> np.ndarray:
    """Build one radial center-to-edge map in [0,1]."""
    edge_factor = float(np.clip(edge_factor, 0.0, 1.0))
    exponent = max(1e-6, float(exponent))
    yy, xx = np.meshgrid(
        np.arange(yres, dtype=np.float64),
        np.arange(xres, dtype=np.float64),
        indexing="ij",
    )
    cx = 0.5 * (xres - 1)
    cy = 0.5 * (yres - 1)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rmax = float(np.max(rr)) if rr.size else 1.0
    rn = rr / max(1e-12, rmax)
    m = edge_factor + (1.0 - edge_factor) * (1.0 - np.power(np.clip(rn, 0.0, 1.0), exponent))
    return np.clip(m, 0.0, 1.0).astype(np.float32)


def build_spatial_transmission_map(
    yres: int,
    xres: int,
    cfg: dict,
    *,
    repo: Path,
    wavelength_nm: np.ndarray,
    qe_rgb: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Build per-channel center-to-edge radial transmission maps in [0,1]."""
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return np.ones((yres, xres, 3), dtype=np.float32), {
            "enabled": False,
            "mode": "off",
            "edge_factor": 1.0,
            "exponent": 2.0,
        }
    mode = str(cfg.get("mode", "radial_power")).lower().strip()
    if mode != "radial_power":
        raise ValueError('optics_transmittance_spatial.mode must be "radial_power"')
    edge_factor = float(cfg.get("edge_factor", 0.9))
    exponent = float(cfg.get("exponent", 2.0))
    edge_rgb = np.full(3, float(np.clip(edge_factor, 0.0, 1.0)), dtype=np.float64)
    spectral_csv = cfg.get("spectral_edge_factors_csv", None)
    if spectral_csv:
        s_wl, s_v = read_csv_curve((repo / str(spectral_csv)).resolve())
        edge_lambda = np.clip(np.interp(wavelength_nm, s_wl, s_v, left=0.0, right=0.0), 0.0, 1.0)
        qe = np.asarray(qe_rgb, dtype=np.float64)
        if qe.shape[0] != 3:
            raise ValueError(f"expected QE stack [3,K], got {qe.shape}")
        for c in range(3):
            w = np.clip(qe[c], 0.0, None)
            sw = float(np.sum(w))
            if sw > 0.0:
                edge_rgb[c] = float(np.sum(w * edge_lambda) / sw)
        edge_source = "spectral_edge_factors_csv"
    else:
        edge_source = "scalar_edge_factor"
    maps = np.stack([_radial_map(yres, xres, float(edge_rgb[c]), exponent) for c in range(3)], axis=2)
    return maps, {
        "enabled": True,
        "mode": mode,
        "edge_factor": edge_factor,
        "edge_factor_rgb": edge_rgb.tolist(),
        "edge_source": edge_source,
        "spectral_edge_factors_csv": str(spectral_csv) if spectral_csv else None,
        "exponent": exponent,
        "min": float(np.min(maps)),
        "max": float(np.max(maps)),
        "mean": float(np.mean(maps)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    repo_default = Path(__file__).resolve().parent.parent
    ap.add_argument("--repo-root", type=Path, default=repo_default)
    ap.add_argument("--config", type=Path, default=None, help="Sensor-forward YAML config")
    ap.add_argument("--camera-model-config", type=Path, default=None, help="Camera model YAML path (preferred).")
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
    else:
        cfg_path = (args.config or (repo / "config" / "sensor_forward.yaml")).resolve()
        cfg = yaml.safe_load(cfg_path.read_text())

    inputs = cfg.get("inputs", {})
    model = cfg.get("model", {})
    output = cfg.get("output", {})

    spectral_npz = (repo / inputs.get("spectral_reference_npz", "scenes/generated/spectral_reference_1nm.npz")).resolve()
    scene_manifest = (repo / inputs.get("scene_manifest_json", "scenes/generated/colorchecker_manifest.json")).resolve()
    noise_cfg_path = (repo / inputs.get("noise_config_yaml", "config/noise_emva.yaml")).resolve()
    out_npz = (repo / output.get("electrons_npz", "out/sensor_forward_electrons.npz")).resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    spec = np.load(spectral_npz)
    wl = np.asarray(spec["wavelength_nm"], dtype=np.float64)
    E = np.asarray(spec["illuminant"], dtype=np.float64)
    R = np.asarray(spec["reflectance"], dtype=np.float64)  # [24, n_lambda]
    if R.shape[0] != 24:
        raise RuntimeError(f"expected 24 patch spectra, got {R.shape[0]}")

    manifest = json.loads(scene_manifest.read_text())
    xres = int(manifest["film"]["xresolution"])
    yres = int(manifest["film"]["yresolution"])
    cam_dist = float(manifest["camera"]["cam_dist"])
    cam_type = str(manifest["camera"].get("type", "pinhole")).lower()
    _fov_raw = manifest["camera"].get("fov_deg")
    if cam_type == "realistic" or _fov_raw is None:
        raise RuntimeError(
            "spectral_sensor_forward (analytic mode) cannot be used with a realistic camera: "
            "the effective field of view is determined by the lens prescription and is not "
            "recorded in the scene manifest. Use sensor_forward.mode: pbrt_exr instead, "
            "which derives electron counts directly from the rendered spectral EXR."
        )
    fov_deg = float(_fov_raw)
    pw = float(manifest["geometry"]["patch_width"])
    ph = float(manifest["geometry"]["patch_height"])
    gap = float(manifest["geometry"]["gap"])
    board_w = float(manifest["geometry"]["board_size"][0])
    board_h = float(manifest["geometry"]["board_size"][1])

    if args.camera_model_config is not None:
        sensor = camera_model.get("sensor", {})
        if not sensor:
            raise RuntimeError("camera model must provide sensor settings for spectral forward model")
    else:
        ncfg = yaml.safe_load(noise_cfg_path.read_text())
        sensor = ncfg.get("sensor", {})
    qe_cfg = sensor.get("quantum_efficiency", {})
    fill_factor = float(sensor.get("fill_factor", 1.0))
    t_int = float(sensor.get("integration_time_s", 0.01))
    if args.integration_time_s is not None:
        t_int = float(args.integration_time_s)
    f_number = float(sensor.get("f_number", 2.8))
    pixel_pitch_um = float(sensor.get("pixel_pitch_um", 3.45))

    ircf_csv = qe_cfg.get("ircf_csv")
    ircf = np.ones_like(wl)
    if ircf_csv:
        i_wl, i_v = read_csv_curve((repo / ircf_csv).resolve())
        ircf = np.interp(wl, i_wl, i_v, left=0.0, right=0.0)

    q_r, q_g, q_b = load_qe_curves_rgb(
        repo,
        qe_cfg,
        strict_qe_validation=bool(args.strict_qe_validation or qe_cfg.get("strict_validation", False)),
    )
    qes = []
    for q_wl, q_v in (q_r, q_g, q_b):
        q = np.interp(wl, q_wl, q_v, left=0.0, right=0.0)
        qes.append(np.clip(q * ircf, 0.0, 1.0))
    qe = np.stack(qes, axis=0)  # [3, n_lambda]

    cal = model.get("calibration", {}) or {}
    if args.target_illuminance_lux is not None:
        cal["target_illuminance_lux"] = float(args.target_illuminance_lux)
    cal_mode = str(cal.get("mode", "legacy")).lower()
    use_aperture = bool(cal.get("use_aperture_factor", True))
    aperture_factor = (np.pi / (4.0 * max(1e-9, f_number**2))) if use_aperture else 1.0
    pixel_area = (pixel_pitch_um * 1e-6) ** 2
    optics_t = float(cal.get("optics_transmittance", 1.0))
    optics_t_csv = cal.get("optics_transmittance_csv", None)
    if optics_t_csv:
        t_wl, t_v = read_csv_curve((repo / str(optics_t_csv)).resolve())
        tau_lambda = np.interp(wl, t_wl, t_v, left=0.0, right=0.0)
        tau_lambda = np.clip(tau_lambda, 0.0, 1.0)
        optics_mode = "spectral_csv"
    else:
        tau_lambda = np.full_like(wl, float(np.clip(optics_t, 0.0, 1.0)))
        optics_mode = "scalar"
    spatial_cfg = cal.get("optics_transmittance_spatial", {}) or {}

    lighting = (manifest.get("lighting") or {}).get("distant") or {}
    from_pt = np.asarray(lighting.get("from", [0.12, 0.55, 2.9]), dtype=np.float64)
    to_pt = np.asarray(lighting.get("to", [0.0, 0.0, 0.0]), dtype=np.float64)
    w_to_light = from_pt - to_pt
    w_to_light = w_to_light / max(1e-15, float(np.linalg.norm(w_to_light)))
    chart_n = np.asarray(model.get("chart_normal", [0.0, 0.0, 1.0]), dtype=np.float64)
    use_cos = bool(model.get("cosine_shading", True)) and cal_mode == "photon_counting"
    cos_theta = cosine_illuminance_factor(chart_n, w_to_light) if use_cos else 1.0

    # Build pixel-to-chart mapping by intersecting camera rays with chart plane z=0.
    jj, ii = np.meshgrid(np.arange(yres, dtype=np.float64), np.arange(xres, dtype=np.float64), indexing="ij")
    x_ndc = 2.0 * ((ii + 0.5) / xres) - 1.0
    y_ndc = 1.0 - 2.0 * ((jj + 0.5) / yres)
    aspect = xres / yres
    tan_half = np.tan(np.deg2rad(fov_deg) * 0.5)
    # Match pbrt-v4 LookAt (camera +X along world -X): increasing image column -> decreasing world X.
    xw = -x_ndc * cam_dist * tan_half * aspect
    yw = y_ndc * cam_dist * tan_half

    use_vig = bool(model.get("vignetting_cos4", False))
    vig_map = (
        cos4_vignetting_from_pinhole(xw, yw, cam_dist).astype(np.float32)
        if use_vig
        else np.ones_like(xw, dtype=np.float32)
    )

    # Patch-channel responses from spectra.
    target_lux = cal.get("target_illuminance_lux", None)
    illum_csv = cal.get("illuminant_override_csv", None)
    if illum_csv:
        e_wl, e_v = read_csv_curve((repo / illum_csv).resolve())
        E = np.interp(wl, e_wl, e_v, left=e_v[0], right=e_v[-1])

    illuminance_input_lux = None
    illuminance_scale = 1.0

    if cal_mode == "photon_counting":
        irr_scale = float(cal.get("irradiance_scale_W_m2nm_per_unit", 1.0e-3))
        if target_lux is not None:
            illuminance_input_lux = illuminance_lux_from_irradiance(wl, E * irr_scale)
            if illuminance_input_lux <= 0:
                raise RuntimeError("input illuminance <= 0; cannot normalize to target lux")
            illuminance_scale = float(target_lux) / illuminance_input_lux
        irr_scale_eff = irr_scale * illuminance_scale
        base = E[None, :] * R * (irr_scale_eff * cos_theta) * tau_lambda[None, :]
        patch_resp = np.zeros((24, 3), dtype=np.float64)
        for i in range(24):
            phi = photon_flux_density_from_irradiance(base[i], wl)
            for c in range(3):
                patch_resp[i, c] = _trapz(phi * qe[c], wl)
        geom = t_int * fill_factor * pixel_area * aperture_factor
        scalar = float(geom)
        patch_e = np.clip(patch_resp * geom, 0.0, None)
    else:
        base = E[None, :] * R * tau_lambda[None, :]
        patch_resp = np.zeros((24, 3), dtype=np.float64)
        for i in range(24):
            for c in range(3):
                patch_resp[i, c] = _trapz(base[i] * qe[c], wl)
        electrons_scale = float(model.get("electrons_scale", 25000.0))
        scalar = t_int * fill_factor * pixel_area * aperture_factor * electrons_scale
        patch_e = np.clip(patch_resp * scalar, 0.0, None)

    # Chart bounds in world XY (z=0 plane).
    x_min_board = -board_w * 0.5
    x_max_board = board_w * 0.5
    y_min_board = -board_h * 0.5
    y_max_board = board_h * 0.5
    in_board = (xw >= x_min_board) & (xw <= x_max_board) & (yw >= y_min_board) & (yw <= y_max_board)

    electrons = np.zeros((yres, xres, 3), dtype=np.float32)
    include_surround = bool(model.get("include_surround", True))
    surround_reflectance = float(model.get("surround_reflectance", 0.22))
    if include_surround:
        surround_resp = np.zeros(3, dtype=np.float64)
        if cal_mode == "photon_counting":
            irr_scale = float(cal.get("irradiance_scale_W_m2nm_per_unit", 1.0e-3)) * illuminance_scale
            s_irr = E * surround_reflectance * irr_scale * cos_theta * tau_lambda
            phi_s = photon_flux_density_from_irradiance(s_irr, wl)
            geom = t_int * fill_factor * pixel_area * aperture_factor
            for c in range(3):
                surround_resp[c] = _trapz(phi_s * qe[c], wl)
            surround_vec = np.clip(surround_resp * geom, 0.0, None).astype(np.float32)
        else:
            for c in range(3):
                surround_resp[c] = _trapz(E * surround_reflectance * tau_lambda * qe[c], wl)
            surround_vec = np.clip(surround_resp * scalar, 0.0, None).astype(np.float32)
        electrons[in_board] = surround_vec * vig_map[in_board][:, None]

    # Patch layout in world coordinates (same as scene builder).
    x0 = -board_w / 2.0
    y_top = board_h / 2.0
    for row in range(4):
        for col in range(6):
            idx = row * 6 + col  # 0..23
            xL = x0 + col * (pw + gap)
            xR = xL + pw
            px0, px1 = -xR, -xL
            py1 = y_top - row * (ph + gap)
            py0 = py1 - ph
            m = (xw >= px0) & (xw <= px1) & (yw >= py0) & (yw <= py1)
            electrons[m] = (patch_e[idx] * vig_map[m, None]).astype(np.float32)

    spatial_map, spatial_meta = build_spatial_transmission_map(
        yres,
        xres,
        spatial_cfg,
        repo=repo,
        wavelength_nm=wl,
        qe_rgb=qe,
    )
    electrons = np.clip(electrons * spatial_map, 0.0, None).astype(np.float32)

    # Also export a simple linear preview image from electrons.
    # This is only for quick visual checks and not physically authoritative.
    preview = np.clip(electrons, 0.0, None)
    p99 = float(np.percentile(preview, 99.5))
    if p99 > 0:
        preview = np.clip(preview / p99, 0.0, 1.0)
    preview_u8 = np.rint(preview * 255.0).astype(np.uint8)

    np.savez_compressed(
        out_npz,
        electrons_rgb=electrons,
        wavelength_nm=wl,
        patch_electrons_rgb=patch_e.astype(np.float32),
        scalar=float(scalar),
        calibration_mode=np.array(cal_mode),
        optics_transmittance_mode=np.array(optics_mode),
        optics_transmittance_scalar=np.float64(optics_t),
        optics_transmittance_mean=np.float64(float(np.mean(tau_lambda))),
        optics_transmittance_spatial=np.array(json.dumps(spatial_meta)),
        cos_theta=np.float64(cos_theta),
        vignetting_cos4=np.bool_(use_vig),
        illuminance_target_lux=np.float64(float(target_lux) if target_lux is not None else np.nan),
        illuminance_input_lux=np.float64(float(illuminance_input_lux) if illuminance_input_lux is not None else np.nan),
        illuminance_scale=np.float64(illuminance_scale),
    )
    try:
        import imageio.v3 as iio

        iio.imwrite(out_npz.with_suffix(".png"), preview_u8)
    except Exception:
        pass

    print(f"Wrote electrons npz: {out_npz}")
    print(f"Wrote preview png: {out_npz.with_suffix('.png')}")


if __name__ == "__main__":
    main()
