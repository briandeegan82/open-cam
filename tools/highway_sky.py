#!/usr/bin/env python3
"""Sky / sun helpers for the highway scenes: equal-area env maps, sun excision, solar spectra.

pbrt-v4 ``infinite`` lights need square equal-area (Clarberg octahedral) images, and
``imgtool`` is not part of the open-cam pbrt build, so the conversion lives here.

Light-space convention (same as pbrt's infinite light): +z is the zenith. The builder
places the map with ``Rotate <az> 0 1 0`` then ``Rotate -90 1 0 0`` so light +z -> world +y.
Equirect inputs use polar angle theta from the top row and phi = 2*pi*col/width.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Spectral luminous efficacy (lm/W) and the extraterrestrial direct-normal illuminance.
SOLAR_ILLUMINANCE_CONSTANT_LUX = 128_000.0
LUMA = np.array([0.2126, 0.7152, 0.0722])


def equal_area_square_to_sphere(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorised port of pbrt-v4 ``EqualAreaSquareToSphere`` (u, v in [0, 1]) -> (..., 3)."""
    uu, vv = 2.0 * u - 1.0, 2.0 * v - 1.0
    up, vp = np.abs(uu), np.abs(vv)
    signed = 1.0 - (up + vp)
    r = 1.0 - np.abs(signed)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = np.where(r == 0, 1.0, (vp - up) / np.where(r == 0, 1.0, r) + 1.0) * np.pi / 4.0
    z = np.copysign(1.0 - r * r, signed)
    s = r * np.sqrt(np.clip(2.0 - r * r, 0.0, None))
    return np.stack([np.copysign(np.cos(phi), uu) * s, np.copysign(np.sin(phi), vv) * s, z], axis=-1)


def equal_area_directions(n: int) -> np.ndarray:
    """Light-space direction of each pixel centre of an n x n equal-area map, shape (n, n, 3) [row, col]."""
    c = (np.arange(n) + 0.5) / n
    u, v = np.meshgrid(c, c)  # u = column (x), v = row (y), as pbrt's Image lookup
    return equal_area_square_to_sphere(u, v)


def equirect_solid_angles(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """(theta per row, solid angle per pixel per row) for an equirect map."""
    theta = (np.arange(h) + 0.5) / h * np.pi
    return theta, (2.0 * np.pi / w) * (np.pi / h) * np.sin(theta)


def equirect_dirs(h: int, w: int) -> np.ndarray:
    theta, _ = equirect_solid_angles(h, w)
    phi = (np.arange(w) + 0.5) / w * 2.0 * np.pi
    st = np.sin(theta)[:, None]
    return np.stack([st * np.cos(phi)[None, :], st * np.sin(phi)[None, :], np.repeat(np.cos(theta)[:, None], w, 1)], -1)


def sample_equirect(img: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Bilinear lookup of an equirect RGB image along light-space directions (..., 3)."""
    from scipy.ndimage import map_coordinates

    h, w, _ = img.shape
    theta = np.arccos(np.clip(dirs[..., 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(dirs[..., 1], dirs[..., 0]), 2.0 * np.pi)
    rows = theta / np.pi * h - 0.5
    cols = phi / (2.0 * np.pi) * w - 0.5
    padded = np.concatenate([img[:, -1:], img, img[:, :1]], axis=1)
    out = [
        map_coordinates(padded[..., c], [rows.ravel(), cols.ravel() + 1.0], order=1, mode="nearest") for c in range(3)
    ]
    return np.stack(out, -1).reshape(*dirs.shape[:-1], 3)


def read_rgb_exr(path: Path) -> np.ndarray:
    import OpenEXR

    with OpenEXR.File(str(path)) as f:
        ch = f.channels()
        if "RGB" in ch or "RGBA" in ch:
            px = ch["RGB" if "RGB" in ch else "RGBA"].pixels[..., :3]
        else:
            px = np.stack([ch[c].pixels for c in ("R", "G", "B")], -1)
    return np.asarray(px, dtype=np.float64)


def write_rgb_exr(path: Path, rgb: np.ndarray) -> None:
    import OpenEXR

    path.parent.mkdir(parents=True, exist_ok=True)
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    with OpenEXR.File(header, {"RGB": np.ascontiguousarray(rgb, dtype=np.float32)}) as f:
        f.write(str(path))


def excise_sun(img: np.ndarray, *, core_deg: float = 0.75, search_deg: float = 4.0) -> tuple[np.ndarray, dict]:
    """Remove the sun disc from an unclipped equirect sky and measure it.

    Returns the sun-free sky and a dict with the sun's light-space direction and the
    relative (HDRI-unit) perpendicular sun illuminance and horizontal sky illuminance,
    both photometric (Rec.709 luma weights).
    """
    h, w, _ = img.shape
    img = np.clip(np.nan_to_num(img), 0.0, None)
    y = img @ LUMA
    theta, dom = equirect_solid_angles(h, w)
    dirs = equirect_dirs(h, w)
    peak = np.unravel_index(np.argmax(y), y.shape)
    d0 = dirs[peak]
    ang = np.degrees(np.arccos(np.clip(dirs @ d0, -1.0, 1.0)))
    ring = (ang > search_deg) & (ang < search_deg + 2.0)
    fill = np.median(img[ring], axis=0)
    y_fill = float(fill @ LUMA)
    mask = (ang < core_deg) | ((ang < search_deg) & (y > 4.0 * y_fill))
    excess = np.clip(y - y_fill, 0.0, None) * mask * dom[:, None]
    sun_dir = (dirs * excess[..., None]).sum((0, 1))
    sun_dir /= np.linalg.norm(sun_dir)
    sky = img.copy()
    sky[mask] = fill
    sky[theta > np.pi / 2] = 0.0  # nothing below the horizon: terrain covers it
    cos_t = np.clip(np.cos(theta), 0.0, None)
    e_sky_h = float(((sky @ LUMA) * (dom * cos_t)[:, None]).sum())
    e_sun_perp = float(excess.sum())
    elev = float(np.degrees(np.arcsin(sun_dir[2])))
    return sky, {
        "sun_dir_light": [float(x) for x in sun_dir],
        "sun_elevation_deg": elev,
        "sun_azimuth_light_deg": float(np.degrees(np.arctan2(sun_dir[1], sun_dir[0]))),
        "sun_pixels": int(mask.sum()),
        "e_sun_perp_rel": e_sun_perp,
        "e_sky_horizontal_rel": e_sky_h,
        "sky_to_sun_horizontal_ratio": e_sky_h / max(1e-12, e_sun_perp * max(np.sin(np.radians(elev)), 1e-3)),
    }


def equirect_to_equal_area(img: np.ndarray, n: int = 1024) -> np.ndarray:
    return sample_equirect(img, equal_area_directions(n))


def analytic_clear_sky(n: int, sun_elevation_deg: float) -> np.ndarray:
    """CIE standard general sky type 12 (clear, low turbidity) on an equal-area map.

    Luminance follows the CIE S 011/E:2003 gradation/indicatrix functions; colour is a
    zenith-blue to horizon-white blend (a stand-in, the absolute level is set by the light's
    ``illuminance``). Sun at light-space azimuth 0.
    """
    d = equal_area_directions(n)
    zs = np.radians(90.0 - sun_elevation_deg)
    sun = np.array([np.sin(zs), 0.0, np.cos(zs)])
    z = np.arccos(np.clip(d[..., 2], -1.0, 1.0))
    chi = np.arccos(np.clip(d @ sun, -1.0, 1.0))
    a, b, c, dd, e = -1.0, -0.32, 10.0, -3.0, 0.45

    def grad(zen):
        return 1.0 + a * np.exp(b / np.maximum(np.cos(zen), 1e-3))

    def ind(x):
        return 1.0 + c * (np.exp(dd * x) - np.exp(dd * np.pi / 2)) + e * np.cos(x) ** 2

    lum = ind(chi) * grad(z) / (ind(zs) * grad(0.0))
    t = np.clip(z / (np.pi / 2), 0.0, 1.0) ** 2
    zen_rgb, hor_rgb = np.array([0.30, 0.48, 1.0]), np.array([0.78, 0.86, 1.0])
    rgb = (1.0 - t)[..., None] * zen_rgb + t[..., None] * hor_rgb
    rgb = rgb / (rgb @ LUMA)[..., None]
    out = rgb * lum[..., None]
    out[d[..., 2] <= 0.0] = 0.0
    return out


def kasten_young_airmass(elevation_deg: float) -> float:
    h = max(float(elevation_deg), 0.0)
    return 1.0 / (np.sin(np.radians(h)) + 0.50572 * (h + 6.07995) ** -1.6364)


def clear_sky_illuminance_lux(elevation_deg: float) -> tuple[float, float]:
    """(direct-normal sun, diffuse horizontal sky) illuminance for a clear sky [lux].

    Direct: Beer-Lambert with a luminous extinction coefficient 0.21 per air mass
    (IESNA clear sky); diffuse: IESNA clear-sky fit 0.8 + 15.5 sqrt(sin h) klux.
    """
    m = kasten_young_airmass(elevation_deg)
    e_dn = SOLAR_ILLUMINANCE_CONSTANT_LUX * np.exp(-0.21 * m)
    e_d = 800.0 + 15_500.0 * np.sqrt(max(np.sin(np.radians(elevation_deg)), 0.0))
    return float(e_dn), float(e_d)


def solar_direct_spectrum(wl_nm: np.ndarray, elevation_deg: float, aerosol_beta: float = 0.06) -> np.ndarray:
    """Relative direct-normal solar SPD at the ground for a given sun elevation.

    5778 K Planck extraterrestrial envelope x Beer-Lambert transmission along the
    Kasten-Young air mass: Rayleigh (0.008735 um^-4.08), Angstrom aerosol
    (beta * um^-1.3) and Chappuis ozone, plus O2-B/H2O/O2-A notches. Normalised to 1 at 560 nm.
    """
    wl = np.asarray(wl_nm, dtype=np.float64)
    um = wl * 1e-3
    hc_k = 1.4387769e-2
    planck = 1.0 / (wl * 1e-9) ** 5 / (np.exp(hc_k / (wl * 1e-9 * 5778.0)) - 1.0)
    m = kasten_young_airmass(elevation_deg)
    tau = 0.008735 * um**-4.08 + aerosol_beta * um**-1.3 + 0.03 * np.exp(-0.5 * ((wl - 600.0) / 80.0) ** 2)
    notches = (
        0.06 * np.exp(-0.5 * ((wl - 687.0) / 4.0) ** 2)
        + 0.12 * np.exp(-0.5 * ((wl - 719.0) / 8.0) ** 2)
        + 0.55 * np.exp(-0.5 * ((wl - 761.0) / 3.5) ** 2)
        + 0.25 * np.exp(-0.5 * ((wl - 820.0) / 10.0) ** 2)
    )
    v = planck * np.exp(-m * tau) * (1.0 - np.clip(notches, 0.0, 0.95)) ** min(m, 6.0)
    return v / np.interp(560.0, wl, v)
