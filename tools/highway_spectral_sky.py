"""Spectral sky radiance for the highway scene.

pbrt-v4's image ``infinite`` light is RGB-only, so the sky is written as a *spectral basis*
map: an equal-area EXR with channels ``B0..B{K-1}`` that are per-pixel luminance weights of
K non-negative daylight spectra, rendered by the open-cam pbrt patch
``third_party/patches/pbrt-v4-spectral-basis-infinite-light.patch``
(``Le(w, lambda) = scale * sum_k B_k(w) * L_k(lambda) / Y(L_k)``).

The basis spans the CIE daylight model S0 + M1 S1 + M2 S2 (Judd, MacAdam & Wyszecki 1964,
JOSA 54:1031; CIE 015:2018 sec. 4.1.2), derived from measured daylight (sun + sky and
skylight alone). Sources of per-pixel radiance:

* ``hosek``: the Hosek-Wilkie spectral clear-sky model (Hosek & Wilkie 2012, "An Analytic
  Model for Full Spectral Sky-Dome Radiance", ACM TOG 31(4):95; BSD-3 sample code and data,
  shipped in ``third_party/pbrt-v4/src/ext/skymodel`` and copied to
  ``spectra/sky/hosek_wilkie_2012_spectral.npz``), 320-720 nm every 40 nm.
* an RGB HDRI (or the RGB CIE type-12 map), read as linear sRGB.

Either way each pixel becomes the daylight-model spectrum with the same CIE 1931 XYZ (exact
for colours inside the basis gamut; outside it negative weights are clamped and the luminance
preserved).

Daylight components: CIE (2022) "Relative spectral power distributions of CIE daylight
components S0, S1, S2", doi:10.25039/CIE.DS.w7zunnny, CC BY-SA 4.0,
``spectra/illuminant/original/CIE_illum_Dxx_comp.csv``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from colour_science import cmf_on_grid
from highway_sky import equal_area_directions, read_rgb_exr

REPO = Path(__file__).resolve().parents[1]
DAYLIGHT_CSV = REPO / "spectra" / "illuminant" / "original" / "CIE_illum_Dxx_comp.csv"
HOSEK_NPZ = REPO / "spectra" / "sky" / "hosek_wilkie_2012_spectral.npz"
KM = 683.0  # lm/W
SRGB_TO_XYZ = np.array(
    [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]]
)
# Daylight-model coefficients (M1, M2) of the K = 3 basis spectra S0 + M1 S1 + M2 S2: three
# vertices of the region where S0 + M1 S1 + M2 S2 >= 0 over 360-830 nm (so every basis spectrum
# is a physical, non-negative SPD), chosen so the triangle contains the CIE daylight locus from
# ~3300 K to infinity and the Hosek-Wilkie clear skies. Their chromaticities: (0.427, 0.416)
# ~3260 K; (0.188, 0.184) deep blue; (0.269, 0.319) ~9600 K, Duv -0.022.
BASIS_M = np.array([[-2.052, 5.206], [9.443, 5.343], [-0.425, -8.328]])
HOSEK_GROUND_ALBEDO = 0.1  # grass/asphalt surroundings (visible albedo ~0.1-0.2)
HOSEK_MAP_SIZE = 512


def daylight_components() -> tuple[np.ndarray, np.ndarray]:
    """CIE daylight components: wavelengths [nm] (300-830, 5 nm) and S (3, n)."""
    d = np.loadtxt(DAYLIGHT_CSV, delimiter=",")
    return d[:, 0], d[:, 1:].T


def daylight_m1_m2(x: float, y: float) -> tuple[float, float]:
    """CIE 015:2018 eq. 4.10: daylight-model coefficients for chromaticity (x, y)."""
    m = 0.0241 + 0.2562 * x - 0.7341 * y
    return (-1.3515 - 1.7703 * x + 5.9114 * y) / m, (0.0300 - 31.4424 * x + 30.0717 * y) / m


def daylight_chromaticity(cct: float) -> tuple[float, float]:
    """CIE 015:2018 eq. 4.7-4.9: chromaticity of the daylight locus at CCT [K] (4000-25000 K)."""
    t = float(cct)
    if t <= 7000.0:
        x = -4.6070e9 / t**3 + 2.9678e6 / t**2 + 0.09911e3 / t + 0.244063
    else:
        x = -2.0064e9 / t**3 + 1.9018e6 / t**2 + 0.24748e3 / t + 0.237040
    return x, -3.0 * x * x + 2.870 * x - 0.275


def daylight_spd(cct: float, wl: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """CIE daylight illuminant D at CCT (relative, S0 + M1 S1 + M2 S2)."""
    w0, s = daylight_components()
    m1, m2 = daylight_m1_m2(*daylight_chromaticity(cct))
    spd = s[0] + m1 * s[1] + m2 * s[2]
    if wl is None:
        return w0, spd
    return np.asarray(wl, float), np.interp(wl, w0, spd)


def spectrum_xyz(wl: np.ndarray, spd: np.ndarray) -> np.ndarray:
    """CIE 1931 XYZ (unnormalised, = sum xbar*S*dlambda) of spectra (..., n) on grid wl."""
    wl = np.asarray(wl, float)
    dl = np.gradient(wl)
    return np.einsum("...n,cn->...c", np.asarray(spd, float), cmf_on_grid(wl) * dl)


def _uv(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, float)
    den = xyz[..., 0] + 15.0 * xyz[..., 1] + 3.0 * xyz[..., 2]
    return np.stack([4.0 * xyz[..., 0] / den, 6.0 * xyz[..., 1] / den], -1)


_PLANCK_T = np.geomspace(1000.0, 1.0e6, 4000)


def _planck_uv() -> np.ndarray:
    wl = np.arange(360.0, 831.0, 1.0) * 1e-9
    c2 = 1.4388e-2
    spd = 1.0 / (wl**5 * np.expm1(c2 / (wl[None, :] * _PLANCK_T[:, None])))
    return _uv(spectrum_xyz(wl * 1e9, spd))


_PLANCK_UV = _planck_uv()


def cct_duv(xyz: np.ndarray) -> tuple[float, float]:
    """Correlated colour temperature [K] and Duv (CIE 1960 uv distance from the Planckian locus,
    + above) by direct search of the locus (CIE 015:2018 sec. 9.5), valid 1000 K - 1 MK."""
    uv = _uv(xyz)
    d = np.hypot(*(_PLANCK_UV - uv).T)
    i = int(np.clip(np.argmin(d), 1, len(d) - 2))
    # parabolic refinement in log T
    lt = np.log(_PLANCK_T[i - 1 : i + 2])
    a, b, _ = np.polyfit(lt, d[i - 1 : i + 2], 2)
    t = float(np.exp(-b / (2 * a))) if a > 0 else float(_PLANCK_T[i])
    t = float(np.clip(t, _PLANCK_T[i - 1], _PLANCK_T[i + 1]))
    p = np.array([np.interp(np.log(t), np.log(_PLANCK_T), _PLANCK_UV[:, k]) for k in (0, 1)])
    tangent = _PLANCK_UV[i + 1] - _PLANCK_UV[i - 1]
    sign = -np.sign(tangent[0] * (uv[1] - p[1]) - tangent[1] * (uv[0] - p[0]))
    return t, float(sign * np.hypot(*(uv - p)))


# ---------------------------------------------------------------- daylight spectral basis


def basis_spectra(m: np.ndarray = BASIS_M) -> tuple[np.ndarray, np.ndarray]:
    """Basis spectra S0 + M1 S1 + M2 S2 (K, n) on the CIE grid, each scaled to unit luminance."""
    wl, s = daylight_components()
    spd = s[0][None] + m[:, :1] * s[1][None] + m[:, 1:] * s[2][None]
    spd = np.clip(spd, 0.0, None)  # vertices rounded to 3 decimals dip to ~-1e-4 of peak
    return wl, spd / spectrum_xyz(wl, spd)[:, 1:2]


def _raw_weights(c: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wl, s = daylight_components()
    y_s = spectrum_xyz(wl, s)[:, 1]
    raw = np.concatenate([np.ones((len(m), 1)), m], 1)  # (K, 3): basis k = raw[k] @ S
    a = (raw / (raw @ y_s)[:, None]).T  # coefficients of the unit-luminance basis, (3, K)
    c = np.asarray(c, float)
    return c @ np.linalg.pinv(a).T, c @ y_s


def coeffs_to_weights(c: np.ndarray, m: np.ndarray = BASIS_M) -> np.ndarray:
    """Daylight-model coefficients (..., 3) -> luminance weights (..., K) of the unit-luminance
    basis. Out-of-gamut pixels: negative weights are clamped and the rest rescaled so the
    luminance (sum of weights) is preserved."""
    w, lum = _raw_weights(c, m)
    w_pos = np.clip(w, 0.0, None)
    tot = w_pos.sum(-1, keepdims=True)
    scale = np.where(tot > 0, np.clip(lum, 0.0, None)[..., None] / np.where(tot > 0, tot, 1.0), 0.0)
    return w_pos * scale


def out_of_gamut_fraction(c: np.ndarray, m: np.ndarray = BASIS_M) -> float:
    """Fraction of the total luminance in pixels outside the basis gamut (clamped)."""
    w, lum = _raw_weights(c, m)
    lum = np.clip(lum, 0.0, None)
    tot = lum.sum()
    return float(lum[(w < -1e-9 * np.abs(w).max()).any(-1)].sum() / tot) if tot > 0 else 0.0


def xyz_to_coeffs(xyz: np.ndarray) -> np.ndarray:
    """Daylight-model coefficients (c0, c1, c2) whose spectrum has the given CIE XYZ."""
    wl, s = daylight_components()
    return np.asarray(xyz, float) @ np.linalg.inv(spectrum_xyz(wl, s))


def rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(np.asarray(rgb, float)), 0.0, None) @ SRGB_TO_XYZ.T


def rgb_to_weights(rgb: np.ndarray, m: np.ndarray = BASIS_M) -> np.ndarray:
    """Linear-sRGB pixels (..., 3) -> basis luminance weights (..., K): the daylight-model
    spectrum with the same CIE XYZ. The weights sum to the pixel's luminance."""
    return coeffs_to_weights(xyz_to_coeffs(rgb_to_xyz(rgb)), m)


# ---------------------------------------------------------------- Hosek-Wilkie spectral sky


class HosekWilkieSky:
    """NumPy port of ``arhosekskymodelstate_alloc_init`` / ``arhosekskymodel_radiance``
    (ArHosekSkyModel.c, Hosek & Wilkie 2012) evaluated at the 11 dataset wavelengths.

    Radiance units follow the reference implementation (spectral radiance,
    W m^-2 sr^-1 nm^-1). Valid for turbidity 1-10, solar elevation 0-90 deg.
    """

    def __init__(self, sun_elevation_deg: float, turbidity: float = 3.0, albedo: float = 0.1):
        if not 1.0 <= turbidity <= 10.0:
            raise ValueError("Hosek-Wilkie turbidity must be in [1, 10]")
        if not 0.0 <= sun_elevation_deg <= 90.0:
            raise ValueError("Hosek-Wilkie solar elevation must be in [0, 90] deg")
        d = np.load(HOSEK_NPZ)
        self.wavelength_nm = d["wavelength_nm"]
        t = (np.radians(sun_elevation_deg) / (np.pi / 2)) ** (1.0 / 3.0)
        bern = np.array([(1 - t) ** 5, 5 * (1 - t) ** 4 * t, 10 * (1 - t) ** 3 * t**2] + [0.0] * 3)
        bern[3:] = [10 * (1 - t) ** 2 * t**3, 5 * (1 - t) * t**4, t**5]
        it = int(turbidity)
        rem = turbidity - it
        terms = [(0, it - 1, (1 - albedo) * (1 - rem)), (1, it - 1, albedo * (1 - rem))]
        if it < 10:
            terms += [(0, it, (1 - albedo) * rem), (1, it, albedo * rem)]
        ds, dr = d["datasets"], d["datasets_rad"]  # (11, albedo, turbidity, 6, 9), (11, 2, 10, 6)
        self.config = sum(f * np.einsum("j,wji->wi", bern, ds[:, a, ti]) for a, ti, f in terms)
        self.rad = sum(f * dr[:, a, ti] @ bern for a, ti, f in terms)

    def radiance(self, cos_theta: np.ndarray, cos_gamma: np.ndarray) -> np.ndarray:
        """Spectral radiance (..., 11) for view zenith cosine and cosine of the angle to the sun."""
        c = self.config  # (11, 9)
        ct = np.clip(np.asarray(cos_theta, float), 0.0, 1.0)[..., None]
        cg = np.clip(np.asarray(cos_gamma, float), -1.0, 1.0)[..., None]
        gamma = np.arccos(cg)
        exp_m = np.exp(c[:, 4] * gamma)
        ray_m = cg * cg
        mie_m = (1.0 + cg * cg) / (1.0 + c[:, 8] ** 2 - 2.0 * c[:, 8] * cg) ** 1.5
        f = (1.0 + c[:, 0] * np.exp(c[:, 1] / (ct + 0.01))) * (
            c[:, 2] + c[:, 3] * exp_m + c[:, 5] * ray_m + c[:, 6] * mie_m + c[:, 7] * np.sqrt(ct)
        )
        return f * self.rad


def hosek_spectral_map(
    n: int, sun_elevation_deg: float, turbidity: float = 3.0, albedo: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-area (n, n, 11) Hosek-Wilkie sky radiance (sun at light-space azimuth 0, zero below
    the horizon, sun disc excluded) and its wavelengths [nm]."""
    d = equal_area_directions(n)
    zs = np.radians(90.0 - sun_elevation_deg)
    sun = np.array([np.sin(zs), 0.0, np.cos(zs)])
    sky = HosekWilkieSky(sun_elevation_deg, turbidity, albedo)
    out = sky.radiance(d[..., 2], d @ sun)
    out[d[..., 2] <= 0.0] = 0.0
    return sky.wavelength_nm, out


def hosek_xyz_map(n: int, sun_elevation_deg: float, turbidity: float = 3.0, albedo: float = 0.1) -> np.ndarray:
    """Equal-area (n, n, 3) CIE XYZ of the Hosek-Wilkie sky (spectra linearly interpolated to
    5 nm, as in ``arhosekskymodel_radiance``; 683 * Y is luminance in cd/m^2)."""
    wl, spd = hosek_spectral_map(n, sun_elevation_deg, turbidity, albedo)
    grid = np.arange(wl[0], wl[-1] + 1e-9, 5.0)
    interp = np.stack([np.interp(grid, wl, e) for e in np.eye(len(wl))])  # (11, n_grid)
    return spectrum_xyz(grid, spd @ interp)


def hosek_weight_map(n: int, sun_elevation_deg: float, turbidity: float = 3.0, albedo: float = 0.1) -> np.ndarray:
    """Equal-area (n, n, K) basis-weight map of the Hosek-Wilkie sky (weights in Hosek luminance
    units: 683 * sum_k B_k is cd/m^2)."""
    return coeffs_to_weights(xyz_to_coeffs(hosek_xyz_map(n, sun_elevation_deg, turbidity, albedo)))


def write_weight_exr(path: Path, w: np.ndarray) -> None:
    """Basis-weight map as a float EXR with channels B0..B{K-1} (pbrt patch convention)."""
    import OpenEXR

    path.parent.mkdir(parents=True, exist_ok=True)
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    chans = {f"B{k}": np.ascontiguousarray(w[..., k], dtype=np.float32) for k in range(w.shape[-1])}
    with OpenEXR.File(header, chans) as f:
        f.write(str(path))


def read_weight_exr(path: Path) -> np.ndarray:
    import OpenEXR

    with OpenEXR.File(str(path)) as f:
        ch = f.channels()
        k = sum(1 for c in ch if c.startswith("B") and c[1:].isdigit())
        return np.stack([np.asarray(ch[f"B{i}"].pixels, float) for i in range(k)], -1)


def write_basis_spds(spd_dir: Path, prefix: str = "sky_basis") -> list[Path]:
    """Write the unit-luminance basis spectra as pbrt .spd files; returns their paths."""
    wl, b = basis_spectra()
    spd_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for k, v in enumerate(b):
        p = spd_dir / f"{prefix}_{k}.spd"
        p.write_text("\n".join(f"{w:.1f} {x:.6g}" for w, x in zip(wl, v)) + "\n")
        paths.append(p)
    return paths


def weight_map_spectra(w: np.ndarray, wl: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Spectral radiance (..., n) represented by basis weights (..., K) (same luminance units)."""
    w0, b = basis_spectra()
    if wl is not None:
        b = np.stack([np.interp(wl, w0, v) for v in b])
        w0 = np.asarray(wl, float)
    return w0, np.asarray(w, float) @ b


def horizontal_spectrum(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Relative spectral irradiance on a horizontal plane from an equal-area weight map."""
    n = w.shape[0]
    cos = np.clip(equal_area_directions(n)[..., 2], 0.0, None)
    wsum = (w * cos[..., None]).sum((0, 1)) * 4.0 * np.pi / n**2
    return weight_map_spectra(wsum)


def sky_colour_summary(w: np.ndarray, sun_dir_light: np.ndarray | None = None) -> dict:
    """CCT/Duv of the horizontal sky irradiance and of the zenith, for the manifest/validation."""
    n = w.shape[0]
    wl, e = horizontal_spectrum(w)
    d = equal_area_directions(n)
    zen = np.unravel_index(np.argmax(d[..., 2]), d.shape[:2])
    _, lz = weight_map_spectra(w[zen])
    cct_h, duv_h = cct_duv(spectrum_xyz(wl, e))
    cct_z, duv_z = cct_duv(spectrum_xyz(wl, lz))
    return {
        "horizontal_cct_k": round(cct_h, 1),
        "horizontal_duv": round(duv_h, 5),
        "zenith_cct_k": round(cct_z, 1),
        "zenith_duv": round(duv_z, 5),
    }


def map_horizontal_luminance_integral(w: np.ndarray) -> float:
    """sum_k integral over the upper hemisphere of B_k cos(theta) d omega (map luminance units)."""
    n = w.shape[0]
    cos = np.clip(equal_area_directions(n)[..., 2], 0.0, None)
    return float((w.sum(-1) * cos).sum() * 4.0 * np.pi / n**2)


def spectral_sky_light(args, sky_file: Path, elev: float, out_dir: Path, spd_dir: Path) -> dict | None:
    """Builder hook: spectral-basis sky map + basis SPDs, or None for the stock RGB sky.

    The absolute level is still set by the builder's ``"float illuminance"`` (the patched light
    rescales the map to it), so the manifest radiometry is unchanged.
    """
    if args.sky != "hosek" and args.sky_spectrum == "rgb":
        return None
    tex = out_dir / "textures"
    info: dict = {"basis": "CIE daylight S0 + M1 S1 + M2 S2 (doi:10.25039/CIE.DS.w7zunnny)"}
    info["basis_m1_m2"] = BASIS_M.tolist()
    if args.sky == "hosek":
        f = tex / f"sky_hosek_e{elev:.1f}_t{args.turbidity:.2f}_daylight.exr"
        c = xyz_to_coeffs(hosek_xyz_map(HOSEK_MAP_SIZE, elev, args.turbidity, HOSEK_GROUND_ALBEDO))
        w = coeffs_to_weights(c)
        write_weight_exr(f, w)
        desc = f"Hosek-Wilkie spectral clear sky (turbidity {args.turbidity:g}) on a CIE daylight basis"
        info |= {
            "model": "Hosek & Wilkie 2012 spectral sky -> daylight-model spectrum with the same XYZ per pixel",
            "turbidity": args.turbidity,
            "ground_albedo": HOSEK_GROUND_ALBEDO,
            "model_illuminance_horizontal_lux": KM * map_horizontal_luminance_integral(w),
        }
        approx = (
            "Sky radiance: Hosek-Wilkie spectral model (320-720 nm) represented by its CIE daylight-model "
            "metamer per pixel (~4-10 % spectral residual 400-720 nm); absolute level from the builder's "
            "clear-sky fit."
        )
    else:
        f = tex / f"{sky_file.stem}_daylight.exr"
        c = xyz_to_coeffs(rgb_to_xyz(read_rgb_exr(sky_file)))
        w = coeffs_to_weights(c)
        write_weight_exr(f, w)
        desc = f"{args.sky} sky map converted to CIE daylight-basis spectra (same XYZ per pixel)"
        info["model"] = "linear-sRGB pixel -> daylight-model spectrum with the same CIE XYZ"
        approx = (
            "Sky radiance: RGB sky map converted per pixel to the CIE daylight-model spectrum with the same "
            "XYZ (exact inside the basis gamut); the sun is spectral."
        )
    up = equal_area_directions(w.shape[0])[..., 2] > 0.0
    info["out_of_gamut_luminance_fraction"] = round(out_of_gamut_fraction(c[up]), 4)
    info |= sky_colour_summary(w)
    spds = write_basis_spds(spd_dir)
    rel = [Path(os.path.relpath(p, out_dir)).as_posix() for p in spds]
    info["spectra"] = rel
    return {
        "file": f,
        "spectra": " ".join(f'"{r}"' for r in rel),
        "description": desc,
        "manifest": info,
        "approximation": approx,
    }
