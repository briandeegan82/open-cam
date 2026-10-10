"""CPIQ-style (IEEE 1858) perceptual metrics: acutance, visual noise, chroma level, colour uniformity.

Implemented from the published CPIQ / ISO descriptions (Baxter et al., "Development of the I3A CPIQ
spatial metrics", Proc. SPIE 8293, 2012; ISO 15739:2013 visual noise). The standards are not open, so
constants are exposed as arguments; check them against IEEE 1858 before quoting absolute numbers.
JND / quality-loss mapping is not included.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from colour_science import WHITE_D65, delta_e_2000, srgb_linear_to_xyz, xyz_to_lab
from scipy import ndimage


def _lab(xyz: np.ndarray, white: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    return xyz_to_lab(xyz.reshape(-1, 3), white).reshape(xyz.shape)


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def csf_cpiq(nu_cpd: np.ndarray, a: float = 75.0, b: float = 0.2, c: float = 0.8, k: float = 34.05) -> np.ndarray:
    """Luminance contrast sensitivity used by CPIQ acutance: a * nu^c * exp(-b nu) / K (nu in cycles/degree)."""
    nu = np.maximum(np.asarray(nu_cpd, dtype=np.float64), 0.0)
    return a * nu**c * np.exp(-b * nu) / k


@dataclass(frozen=True)
class ViewingCondition:
    """Image of ``image_height_px`` shown ``display_height_m`` tall, viewed from ``distance_m``."""

    image_height_px: int
    display_height_m: float
    distance_m: float

    @property
    def pixels_per_degree(self) -> float:
        return self.image_height_px / self.display_height_m * 2.0 * self.distance_m * np.tan(np.radians(0.5))

    def to_cpd(self, cy_per_px: np.ndarray) -> np.ndarray:
        return np.asarray(cy_per_px, dtype=np.float64) * self.pixels_per_degree


def viewing_condition(name: str, image_height_px: int) -> ViewingCondition:
    """Named presets: ``monitor_100pct`` (0.254 mm pixels, 0.6 m), ``uhd_24in_0p6m``, ``print_4x6_0p4m``."""
    presets = {
        "monitor_100pct": (image_height_px * 0.254e-3, 0.6),
        "uhd_24in_0p6m": (0.2989, 0.6),
        "print_4x6_0p4m": (0.1016, 0.4),
    }
    if name not in presets:
        raise ValueError(f"unknown viewing condition {name!r}; choose from {sorted(presets)}")
    h, d = presets[name]
    return ViewingCondition(image_height_px, h, d)


def acutance(freq_cy_px: np.ndarray, mtf: np.ndarray, view: ViewingCondition, f_max_cy_px: float = 0.5) -> float:
    """CSF-weighted MTF area: integral(MTF * CSF) / integral(CSF) over 0..f_max (cycles/pixel)."""
    f = np.asarray(freq_cy_px, dtype=np.float64)
    m = np.asarray(mtf, dtype=np.float64)
    sel = f <= f_max_cy_px + 1e-12
    f, m = f[sel], m[sel]
    nu = view.to_cpd(f)
    w = csf_cpiq(nu)
    den = _trapz(w, nu)
    return _trapz(m * w, nu) / den if den > 0 else float("nan")


def srgb_decode(v: np.ndarray) -> np.ndarray:
    v = np.clip(np.asarray(v, dtype=np.float64), 0.0, 1.0)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def xyz_to_luv(xyz: np.ndarray, white: np.ndarray = WHITE_D65) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    lab = _lab(xyz, white)
    den = xyz[..., 0] + 15 * xyz[..., 1] + 3 * xyz[..., 2]
    den_w = white[0] + 15 * white[1] + 3 * white[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        up = np.where(den > 0, 4 * xyz[..., 0] / den, 4 * white[0] / den_w)
        vp = np.where(den > 0, 9 * xyz[..., 1] / den, 9 * white[1] / den_w)
    l_ = lab[..., 0]
    return np.stack([l_, 13 * l_ * (up - 4 * white[0] / den_w), 13 * l_ * (vp - 9 * white[1] / den_w)], axis=-1)


def _csf_filter(channel: np.ndarray, view: ViewingCondition, kind: str, chroma_sigma_cpd: float) -> np.ndarray:
    h, w = channel.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    nu = view.to_cpd(np.hypot(fx, fy))
    if kind == "lum":
        g = csf_cpiq(nu)
        g = g / g.max()
        g[0, 0] = 1.0
    else:
        g = np.exp(-0.5 * (nu / chroma_sigma_cpd) ** 2)
    mean = channel.mean()
    return np.real(np.fft.ifft2(np.fft.fft2(channel - mean) * g)) + mean


def visual_noise(
    patch_srgb: np.ndarray,
    view: ViewingCondition,
    *,
    weights: tuple[float, float, float] = (1.0, 0.852, 0.323),
    chroma_sigma_cpd: float = 4.0,
    filter_csf: bool = True,
) -> float:
    """Visual noise of a uniform sRGB patch (values 0..1): log10(1 + w1 var(L*) + w2 var(u*) + w3 var(v*)).

    Luminance is filtered with the CPIQ CSF, chroma with a Gaussian low-pass (``chroma_sigma_cpd``),
    in the viewing geometry, before conversion to CIELUV (ISO 15739 structure; weights per its
    published formula, exposed for verification).
    """
    rgb = srgb_decode(patch_srgb)
    xyz = srgb_linear_to_xyz(rgb.reshape(-1, 3)).reshape(rgb.shape)
    if filter_csf:
        y = xyz[..., 1]
        c1 = xyz[..., 0] - y
        c2 = y - xyz[..., 2]
        y = _csf_filter(y, view, "lum", chroma_sigma_cpd)
        c1 = _csf_filter(c1, view, "chroma", chroma_sigma_cpd)
        c2 = _csf_filter(c2, view, "chroma", chroma_sigma_cpd)
        xyz = np.stack([c1 + y, y, y - c2], axis=-1)
    luv = xyz_to_luv(np.clip(xyz, 0.0, None))
    var = luv.reshape(-1, 3).var(axis=0, ddof=1)
    return float(np.log10(1.0 + float(np.dot(weights, var))))


def chroma(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    return np.hypot(lab[..., 1], lab[..., 2])


def chroma_level(lab_measured: np.ndarray, lab_reference: np.ndarray, indices: list[int] | None = None) -> float:
    """Mean measured chroma / mean reference chroma x 100 over the chosen patches (CPIQ chroma level)."""
    m = np.asarray(lab_measured)
    r = np.asarray(lab_reference)
    if indices is not None:
        m, r = m[indices], r[indices]
    return float(100.0 * chroma(m).mean() / chroma(r).mean())


@dataclass(frozen=True)
class ColourUniformity:
    max_delta_uv: float
    max_delta_e00: float
    block_lab: np.ndarray


def colour_uniformity(flat_srgb: np.ndarray, grid: tuple[int, int] = (9, 12)) -> ColourUniformity:
    """Colour shading of a flat-field image: block means vs the centre block, as max Δu'v' and max ΔE00."""
    rgb = srgb_decode(flat_srgb)
    h, w, _ = rgb.shape
    gy, gx = grid
    blocks = np.array(
        [
            [
                rgb[i * h // gy : (i + 1) * h // gy, j * w // gx : (j + 1) * w // gx].reshape(-1, 3).mean(0)
                for j in range(gx)
            ]
            for i in range(gy)
        ]
    )
    xyz = srgb_linear_to_xyz(blocks.reshape(-1, 3)).reshape(gy, gx, 3)
    centre = xyz[gy // 2, gx // 2]
    xyz_n = xyz / xyz[..., 1:2] * centre[1]
    lab = _lab(xyz_n, centre / centre[1])
    den = xyz[..., 0] + 15 * xyz[..., 1] + 3 * xyz[..., 2]
    uv = np.stack([4 * xyz[..., 0] / den, 9 * xyz[..., 1] / den], axis=-1)
    duv = np.linalg.norm(uv - uv[gy // 2, gx // 2], axis=-1)
    de = delta_e_2000(lab.reshape(-1, 3), np.broadcast_to(lab[gy // 2, gx // 2], (gy * gx, 3)))
    return ColourUniformity(float(duv.max()), float(np.max(de)), lab)


def lab_from_srgb(srgb: np.ndarray, white: np.ndarray = WHITE_D65) -> np.ndarray:
    rgb = srgb_decode(srgb)
    return _lab(srgb_linear_to_xyz(rgb.reshape(-1, 3)).reshape(rgb.shape), white)


def smooth(img: np.ndarray, sigma: float) -> np.ndarray:
    return ndimage.gaussian_filter(img, sigma)
