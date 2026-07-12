#!/usr/bin/env python3
"""Apply a spatial PSF to a pbrt SpectralFilm EXR (all float channels).

Intended as a post-render optics step (measured MTF / defocus approximation) after
PBRT.  Uses numpy + scipy.

Three modes
-----------
gaussian
    A single ``sigma_pixels`` is applied identically to every channel.  Legacy mode,
    kept for backward compatibility.  Uses a separable Gaussian (fast).

chromatic_gaussian
    Wavelength-dependent blur: for every spectral bucket (S0.*nm channels) the PSF
    width is computed as the quadrature sum of a constant geometric-aberration term and
    a diffraction-limited term that scales linearly with wavelength::

        σ_diff(λ) = 0.437 × N × λ_nm / (1000 × pixel_pitch_um)   [pixels]
        σ(λ)      = sqrt( σ_diff(λ)² + σ_geom² )

    where N is the f-number and ``σ_geom = sigma_geometric_pixels`` from the config.
    The coefficient 0.437 = 1.028 / 2.355 converts the Airy-disk FWHM (1.028 λN) to
    an equivalent Gaussian σ (FWHM = 2.355 σ).

    This models longitudinal chromatic aberration and the wavelength dependence of the
    Airy disk size.  Non-spectral (R/G/B) channels each receive a σ computed at a
    representative wavelength (620/540/460 nm respectively).

airy_disk
    Physically correct 2-D Airy disk PSF for each spectral channel, convolved via
    FFT.  First-zero radius::

        ρ₀(λ) = 1.22 × N × λ_nm / (1000 × pixel_pitch_um)   [pixels]

    The total PSF is the quadrature-sum of the diffraction Airy disk and a Gaussian
    geometric-aberration component (``sigma_geometric_pixels``), combined in the
    Fourier domain.  Unlike ``chromatic_gaussian``, the Airy-disk mode preserves the
    correct ring structure and the sharp central lobe.  Slower than the Gaussian modes
    due to per-channel 2-D FFT convolution.

Lateral chromatic aberration (all modes)
    When ``lateral_ca_coefficient > 0``, each spectral channel is resampled with a
    wavelength-dependent magnification::

        M(λ) = 1 + lateral_ca_coefficient × (λ - lambda_reference_nm) / lambda_reference_nm

    Long wavelengths (λ > λ_ref) are magnified; short wavelengths are demagnified.
    Bilinear interpolation is used; edges clamp to nearest valid sample.

``f_number`` and ``pixel_pitch_um`` are read from the camera model
(``sensor.f_number``, ``sensor.pixel_pitch_um``).  They may also be overridden
directly in the ``post_psf`` config block.

See config/sensor_models/default.yaml (lens.post_psf) and README (optics section).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from camera_model import load_camera_model
from exr_multispectral import (
    parse_s0_wavelength_nm,
    read_separate_exr_channels,
    write_separate_channels_exr,
)

# Representative center wavelengths (nm) for broadband R/G/B channels.
_RGB_CENTER_NM: dict[str, float] = {"R": 620.0, "G": 540.0, "B": 460.0}


# ---------------------------------------------------------------------------
# Gaussian helpers (separable, fast)
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.ones(1, dtype=np.float64)
    r = int(max(1, np.ceil(3.0 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= np.sum(k)
    return k


def separable_gaussian_blur_2d(img: np.ndarray, sigma: float) -> np.ndarray:
    """HxW float — reflect-pad separable Gaussian."""
    if sigma <= 0:
        return np.asarray(img, dtype=np.float32, copy=False)
    k = _gaussian_kernel_1d(sigma)
    if k.size == 1:
        return np.asarray(img, dtype=np.float32, copy=False)
    pad = k.size // 2
    acc = np.asarray(img, dtype=np.float64)
    # horizontal
    x = np.pad(acc, ((0, 0), (pad, pad)), mode="reflect")
    tmp = np.empty_like(acc)
    for i in range(acc.shape[0]):
        tmp[i, :] = np.convolve(x[i, :], k, mode="valid")
    # vertical
    y = np.pad(tmp, ((pad, pad), (0, 0)), mode="reflect")
    out = np.empty_like(acc)
    for j in range(acc.shape[1]):
        out[:, j] = np.convolve(y[:, j], k, mode="valid")
    return out.astype(np.float32, copy=False)


def psf_sigma_chromatic(
    wavelength_nm: float,
    f_number: float,
    pixel_pitch_um: float,
    sigma_geometric_px: float,
) -> float:
    """Effective PSF σ (pixels) combining diffraction and geometric aberration.

    Airy disk first-zero radius converted to equivalent Gaussian σ:

        σ_diff(λ) = 0.437 × N × λ_nm / (1000 × pixel_pitch_um)

    Total PSF σ in quadrature (both components modelled as Gaussian):

        σ(λ) = sqrt( σ_diff(λ)² + σ_geom² )

    Parameters
    ----------
    wavelength_nm:
        Wavelength in nanometres.
    f_number:
        Lens f-number (N = f/D).
    pixel_pitch_um:
        Pixel pitch in micrometres.
    sigma_geometric_px:
        Constant geometric-aberration blur radius (defocus, coma …) in pixels.
    """
    # 0.437 = 1.028 / 2.355 converts Airy FWHM (1.028 λN) to Gaussian σ (FWHM = 2.355 σ).
    sigma_diff = 0.437 * f_number * wavelength_nm / (1000.0 * pixel_pitch_um)
    return float(np.sqrt(sigma_diff**2 + sigma_geometric_px**2))


# ---------------------------------------------------------------------------
# Airy disk helpers (2-D FFT, physically accurate)
# ---------------------------------------------------------------------------

def _airy_first_zero_px(
    wavelength_nm: float,
    f_number: float,
    pixel_pitch_um: float,
) -> float:
    """First-zero radius of the Airy disk in pixels: ρ₀ = 1.22 λ N / pitch."""
    return 1.22 * f_number * wavelength_nm / (1000.0 * pixel_pitch_um)


def _airy_kernel_2d(rho0_px: float, n_zeros: float = 4.0) -> np.ndarray:
    """Normalised 2-D Airy disk kernel.

    Parameters
    ----------
    rho0_px:
        First-zero radius of the Airy disk in pixels.
    n_zeros:
        Kernel half-extent as a multiple of the first-zero radius.  4 zeros
        captures the central lobe plus the first two rings.
    """
    from scipy.special import j1  # Bessel function of the first kind, order 1

    half = max(1, int(np.ceil(n_zeros * max(rho0_px, 0.5))))
    gy, gx = np.mgrid[-half : half + 1, -half : half + 1].astype(np.float64)
    r = np.sqrt(gx**2 + gy**2)

    # Airy disk: h(r) = [2 J₁(π r / ρ₀) / (π r / ρ₀)]²
    # The function 2 J₁(x)/x is the jinc function (limit = 1 at x=0).
    u = np.pi * r / rho0_px
    with np.errstate(invalid="ignore", divide="ignore"):
        jinc = np.where(u == 0.0, 1.0, 2.0 * j1(u) / u)
    h = jinc**2
    h /= h.sum()
    return h.astype(np.float64)


def airy_disk_convolve(img: np.ndarray, rho0_px: float, sigma_geom_px: float = 0.0) -> np.ndarray:
    """Convolve *img* (HxW) with the Airy disk PSF, optionally combined with geometric blur.

    If ``sigma_geom_px > 0`` the total PSF is the Fourier product of the Airy disk
    kernel and a Gaussian, which is equivalent to quadrature-summing their widths in
    the spatial domain (valid when both are approximately band-limited).
    """
    from scipy.signal import fftconvolve

    if rho0_px <= 0 and sigma_geom_px <= 0:
        return np.asarray(img, dtype=np.float32, copy=False)

    src = np.asarray(img, dtype=np.float64)

    if rho0_px > 0:
        kernel = _airy_kernel_2d(rho0_px)
        blurred = fftconvolve(src, kernel, mode="same")
    else:
        blurred = src

    if sigma_geom_px > 0:
        blurred = separable_gaussian_blur_2d(blurred, sigma_geom_px).astype(np.float64)

    return blurred.astype(np.float32)


# ---------------------------------------------------------------------------
# Lateral chromatic aberration (wavelength-dependent magnification)
# ---------------------------------------------------------------------------

def apply_lateral_ca(
    arr: np.ndarray,
    wavelength_nm: float,
    lca_coeff: float,
    lambda_ref_nm: float = 550.0,
) -> np.ndarray:
    """Resample *arr* with the LCA magnification for *wavelength_nm*.

    LCA model: the image formed at wavelength λ is magnified by::

        M(λ) = 1 + lca_coeff × (λ − λ_ref) / λ_ref

    Long wavelengths (λ > λ_ref) are magnified; short wavelengths are demagnified,
    matching the typical behaviour of a positive (uncorrected) singlet.  Output pixels
    outside the magnified image domain clamp to the nearest valid input sample.

    Parameters
    ----------
    lca_coeff:
        Dimensionless coefficient.  Typical real-lens values: 0.002–0.010.
        0.0 is a no-op.
    lambda_ref_nm:
        Reference wavelength in nm (zero LCA shift; default 550 nm).
    """
    from scipy.ndimage import map_coordinates

    if abs(lca_coeff) < 1e-9:
        return np.asarray(arr, dtype=np.float32, copy=False)

    M = 1.0 + lca_coeff * (wavelength_nm - lambda_ref_nm) / lambda_ref_nm
    if abs(M - 1.0) < 1e-6:
        return np.asarray(arr, dtype=np.float32, copy=False)

    H, W = arr.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0

    j_out, i_out = np.mgrid[0:H, 0:W].astype(np.float64)
    # Output pixel (i,j) samples the input at (cx + (i-cx)/M, cy + (j-cy)/M)
    i_in = cx + (i_out - cx) / M
    j_in = cy + (j_out - cy) / M

    out = map_coordinates(
        arr.astype(np.float64), [j_in, i_in], order=1, mode="nearest"
    )
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Stray light — veiling glare, halo, ghost reflections, aperture diffraction
# ---------------------------------------------------------------------------

def _n_polygon_mask(n_blades: int, size: int, rotation_deg: float = 0.0) -> np.ndarray:
    """Binary aperture mask for a regular N-gon (N-blade iris), (size × size)."""
    cy, cx = (size - 1) / 2.0, (size - 1) / 2.0
    r_inner = (size / 2.0) - 0.5          # just inside the array boundary
    angle0 = np.deg2rad(rotation_deg)
    angles = angle0 + np.linspace(0, 2 * np.pi, n_blades, endpoint=False)
    vx = cx + r_inner * np.cos(angles)    # vertex x-coords
    vy = cy + r_inner * np.sin(angles)    # vertex y-coords

    # Point-in-polygon test via winding-number / cross-product for each edge.
    gy, gx = np.mgrid[0:size, 0:size].astype(np.float64)
    inside = np.ones((size, size), dtype=bool)
    for i in range(n_blades):
        i1 = (i + 1) % n_blades
        ex = vx[i1] - vx[i]
        ey = vy[i1] - vy[i]
        # cross product: (P - V_i) × edge ≥ 0 means point is on the left of edge
        cross = (gx - vx[i]) * ey - (gy - vy[i]) * ex
        inside &= cross <= 0.0   # clockwise winding in image coords (y-down)
    return inside.astype(np.float64)


def _aperture_diffraction_psf(n_blades: int, size: int, rotation_deg: float = 0.0) -> np.ndarray:
    """Compute the normalised aperture diffraction PSF for an N-blade iris.

    The PSF is the squared magnitude of the Fourier transform of the aperture
    mask (Fraunhofer diffraction).  For a regular N-gon the pattern has a
    bright Airy-like core plus 2 N spikes pointing perpendicular to each blade
    edge (N spikes for even N since opposing pairs coincide).
    """
    mask = _n_polygon_mask(n_blades, size, rotation_deg)
    fft_mask = np.fft.fft2(mask)
    psf = np.abs(np.fft.fftshift(fft_mask)) ** 2
    # Taper to zero at the kernel boundary: the pattern is truncated at the
    # array edge, and without a window the residual energy there stamps a
    # visible square outline around bright sources after convolution.
    yy, xx = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size), indexing="ij")
    r = np.sqrt(xx * xx + yy * yy)
    taper = 0.5 * (1.0 + np.cos(np.pi * np.clip((r - 0.7) / 0.3, 0.0, 1.0)))
    psf *= taper
    psf /= psf.sum()
    return psf.astype(np.float64)


def apply_stray_light(arr: np.ndarray, cfg: dict) -> np.ndarray:
    """Apply stray-light model: veiling glare, halo, ghost reflections, aperture diffraction.

    All sub-models are gated by their own ``enabled`` flag under ``stray_light``.

    veiling_glare_fraction
        Uniform additive fog: ``out = (1−v)·img + v·mean(img)``.
    halo_sigma_pixels / halo_strength
        Wide Gaussian scatter around bright sources.
    ghost_reflections.enabled
        Sensor-surface → rear-element → sensor double reflection.  Modelled as a
        brightness-scaled, geometrically inverted (180° rotated) copy of the image.
        ``ghost_strength`` controls the fraction added (typical 0.01–0.05).
    aperture_diffraction.enabled
        N-blade iris diffraction spikes via |FFT(aperture)|² convolution.
        ``n_blades`` (default 6), ``strength`` (0.02–0.10), ``rotation_deg``,
        ``psf_kernel_size`` (default 128).
    """
    if not bool(cfg.get("enabled", False)):
        return np.asarray(arr, dtype=np.float32, copy=False)

    out = np.asarray(arr, dtype=np.float32, copy=False)

    # ---- Veiling glare ----
    veiling = float(np.clip(cfg.get("veiling_glare_fraction", 0.0), 0.0, 1.0))
    if veiling > 0.0:
        mean_l = float(np.mean(out))
        out = (1.0 - veiling) * out + veiling * mean_l

    # ---- Gaussian halo ----
    halo_sigma = float(cfg.get("halo_sigma_pixels", 0.0))
    halo_strength = float(np.clip(cfg.get("halo_strength", 0.0), 0.0, 1.0))
    if halo_sigma > 0.0 and halo_strength > 0.0:
        halo = separable_gaussian_blur_2d(out, halo_sigma)
        out = out + halo_strength * halo  # additive: scatter adds light, doesn't steal from primary

    # ---- Ghost reflections ----
    ghost_cfg = cfg.get("ghost_reflections", {}) or {}
    if bool(ghost_cfg.get("enabled", False)):
        ghost_strength = float(np.clip(ghost_cfg.get("ghost_strength", 0.02), 0.0, 1.0))
        # The dominant first-order ghost is a 180°-rotated (both axes flipped) copy of
        # the image — the light path is sensor → rear element → sensor, inverting the
        # image through the optical axis.  Ghosts are additive: the reflected energy
        # reaches the sensor in addition to the primary image (the reflective loss is
        # already captured by optics_transmittance, not duplicated here).
        ghost = np.rot90(out, k=2)
        out = out + ghost_strength * ghost

    # ---- Aperture blade diffraction ----
    diff_cfg = cfg.get("aperture_diffraction", {}) or {}
    if bool(diff_cfg.get("enabled", False)):
        from scipy.signal import fftconvolve  # noqa: PLC0415
        n_blades = max(3, int(diff_cfg.get("n_blades", 6)))
        strength = float(np.clip(diff_cfg.get("strength", 0.05), 0.0, 1.0))
        rotation_deg = float(diff_cfg.get("rotation_deg", 0.0))
        psf_size = int(diff_cfg.get("psf_kernel_size", 128))
        psf = _aperture_diffraction_psf(n_blades, psf_size, rotation_deg)
        diffracted = fftconvolve(out.astype(np.float64), psf, mode="same").astype(np.float32)
        out = (1.0 - strength) * out + strength * diffracted

    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    repo_default = Path(__file__).resolve().parent.parent
    ap.add_argument("--repo-root", type=Path, default=repo_default)
    ap.add_argument("--exr-in", type=Path, required=True, help="Input multispectral EXR.")
    ap.add_argument("--exr-out", type=Path, default=None, help="Output EXR (default: overwrite --exr-in).")
    ap.add_argument("--camera-model-config", type=Path, default=None, help="Camera model YAML path (preferred).")
    ap.add_argument("--config", type=Path, default=None, help="config/optics.yaml")
    args = ap.parse_args()

    repo = args.repo_root.resolve()

    camera_model: dict = {}
    if args.camera_model_config is not None:
        cfg_path = args.camera_model_config.resolve()
        if not cfg_path.is_file():
            print(f"error: missing {cfg_path}", file=sys.stderr)
            sys.exit(1)
        camera_model = load_camera_model(cfg_path)
        psf = (camera_model.get("lens", {}) or {}).get("post_psf", {}) or {}
    else:
        cfg_path = (args.config or (repo / "config" / "optics.yaml")).resolve()
        if not cfg_path.is_file():
            print(f"error: missing {cfg_path}", file=sys.stderr)
            sys.exit(1)
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        psf = cfg.get("post_psf", {}) or {}

    if not bool(psf.get("enabled", False)):
        print("post_psf.enabled is false; nothing to do.", file=sys.stderr)
        sys.exit(0)

    mode = str(psf.get("mode", "gaussian")).lower()
    if mode == "none":
        print('post_psf.mode is "none"; nothing to do.', file=sys.stderr)
        sys.exit(0)
    if mode not in ("gaussian", "chromatic_gaussian", "airy_disk"):
        raise ValueError(
            f'post_psf.mode must be "gaussian", "chromatic_gaussian", "airy_disk", or "none", got {mode!r}'
        )

    stray = psf.get("stray_light", {}) or {}
    lca_coeff = float(psf.get("lateral_ca_coefficient", 0.0))
    lambda_ref_nm = float(psf.get("lambda_reference_nm", 550.0))

    exr_in = args.exr_in if args.exr_in.is_absolute() else (repo / args.exr_in).resolve()
    exr_out = (args.exr_out or exr_in).resolve()
    if not exr_in.is_file():
        raise FileNotFoundError(exr_in)

    chans = read_separate_exr_channels(exr_in)
    out: dict[str, np.ndarray] = {}

    if mode == "gaussian":
        sigma = float(psf.get("sigma_pixels", 0.0))
        if sigma < 0:
            raise ValueError("sigma_pixels must be >= 0")
        for name, arr in chans.items():
            blurred = separable_gaussian_blur_2d(arr, sigma)
            if lca_coeff != 0.0:
                lam_nm = parse_s0_wavelength_nm(name) or _RGB_CENTER_NM.get(name.upper(), 550.0)
                blurred = apply_lateral_ca(blurred, lam_nm, lca_coeff, lambda_ref_nm)
            out[name] = apply_stray_light(blurred, stray)
        print(
            f"wrote {exr_out} (gaussian sigma_px={sigma}, "
            f"lca={lca_coeff:.4f}, "
            f"stray_light={'on' if bool(stray.get('enabled', False)) else 'off'})"
        )

    elif mode in ("chromatic_gaussian", "airy_disk"):
        sensor_cfg = camera_model.get("sensor", {}) or {}

        _fn_raw = psf.get("f_number", None)
        if _fn_raw is not None:
            f_number = float(_fn_raw)
        elif "f_number" in sensor_cfg:
            f_number = float(sensor_cfg["f_number"])
        else:
            raise KeyError(
                f"{mode} mode requires f_number. Set sensor.f_number in the "
                "camera model or add f_number to the lens.post_psf block."
            )

        _pp_raw = psf.get("pixel_pitch_um", None)
        if _pp_raw is not None:
            pixel_pitch_um = float(_pp_raw)
        elif "pixel_pitch_um" in sensor_cfg:
            pixel_pitch_um = float(sensor_cfg["pixel_pitch_um"])
        else:
            raise KeyError(
                f"{mode} mode requires pixel_pitch_um. Set sensor.pixel_pitch_um "
                "in the camera model or add pixel_pitch_um to the lens.post_psf block."
            )

        sigma_geom = float(psf.get("sigma_geometric_pixels", 0.0))
        if sigma_geom < 0:
            raise ValueError("sigma_geometric_pixels must be >= 0")

        widths_applied: list[tuple[str, float]] = []

        for name, arr in chans.items():
            lam_nm = parse_s0_wavelength_nm(name)
            if lam_nm is None:
                lam_nm = _RGB_CENTER_NM.get(name.upper(), 550.0)

            if mode == "chromatic_gaussian":
                sigma = psf_sigma_chromatic(lam_nm, f_number, pixel_pitch_um, sigma_geom)
                blurred = separable_gaussian_blur_2d(arr, sigma)
                widths_applied.append((name, sigma))
            else:  # airy_disk
                rho0 = _airy_first_zero_px(lam_nm, f_number, pixel_pitch_um)
                blurred = airy_disk_convolve(arr, rho0, sigma_geom)
                widths_applied.append((name, rho0))

            if lca_coeff != 0.0:
                blurred = apply_lateral_ca(blurred, lam_nm, lca_coeff, lambda_ref_nm)

            out[name] = apply_stray_light(blurred, stray)

        spectral_widths = [w for n, w in widths_applied if parse_s0_wavelength_nm(n) is not None]
        width_label = "σ" if mode == "chromatic_gaussian" else "ρ₀"
        if spectral_widths:
            print(
                f"wrote {exr_out} ({mode}: f/{f_number}, "
                f"pitch={pixel_pitch_um}µm, σ_geom={sigma_geom}px, "
                f"{width_label}(λ) range {min(spectral_widths):.3f}–{max(spectral_widths):.3f}px, "
                f"lca={lca_coeff:.4f}, "
                f"stray_light={'on' if bool(stray.get('enabled', False)) else 'off'})"
            )
        else:
            print(f"wrote {exr_out} ({mode}: no spectral channels found, applied RGB mode)")

    exr_out.parent.mkdir(parents=True, exist_ok=True)
    write_separate_channels_exr(exr_out, out)


if __name__ == "__main__":
    main()
