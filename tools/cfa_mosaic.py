"""Generic periodic colour-filter-array (CFA) model: arbitrary NxM tiles, per-site spectral QE.

The legacy path in :mod:`apply_emva_noise` integrates three R/G/B electron planes and samples
them on a 2x2 Bayer lattice (Bayer, US 3,971,065, 1976).  Non-Bayer sensors (RGBW, RCCB, RYYCy,
RCCG, Quad Bayer, ...) were previously *QE proxies* on that lattice.  This module implements the
real thing and is used only when ``cfa.layout`` is set (opt-in; the Bayer path is untouched):

* **Layout** -- a periodic tile of channel names (``[[R, C], [C, B]]``) or a preset name.  Pixel
  ``(y, x)`` carries channel ``tile[y % rows][x % cols]``.
* **Spectral integration** -- every *distinct* channel has its own QE CSV and is integrated
  directly from the pbrt spectral planes, ``e_c = sum_k E_k * lambda_k/(hc) * QE_c(lambda_k)
  * IRCF(lambda_k) * dlambda * A t FF`` (same radiometry as
  :func:`pbrt_spectral_exr_to_electrons.spectral_radiance_to_electrons`); the mosaic then keeps,
  at each site, only its own channel.  No RGB proxy is involved.
* **Crosstalk / PRNU** -- routed per *site* by the channel at that site (see
  :func:`apply_site_crosstalk`, :func:`site_parameter_map`).
* **Demosaic** -- normalised-convolution bilinear interpolation on each channel's own sub-lattice
  and a gradient-weighted, colour-difference ("constant hue") method guided by the densest
  channel (Cok, US 4,642,678, 1987; Adams & Hamilton, US 5,629,734, 1997; review: Gunturk et al.,
  "Demosaicking: color filter array interpolation", IEEE Signal Process. Mag. 22(1), 2005).
  Measured samples are always preserved exactly.
* **Quad Bayer** -- 2x2 same-colour binning (charge or digital) and a reference remosaic to a
  standard RGGB mosaic (demosaic + resample; not a vendor remosaic algorithm).
* **Colour** -- channels -> linear sRGB/XYZ via a ``C x 3`` least-squares CCM fitted on
  ColorChecker patches (:func:`fit_channel_ccm`).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

QE_DIR = "spectra/QE/interpolated"
# ColorChecker reflectances / CIE illuminants always come from this checkout's spectra/.
DATA_ROOT = Path(__file__).resolve().parent.parent

# Default spectral QE per channel name (total QE = silicon QE x colour filter, as in the repo's
# existing QE_*.csv).  Clear / white sites have no colour filter: the monochrome QE.
DEFAULT_CHANNEL_QE: dict[str, str] = {
    "R": f"{QE_DIR}/QE_red.csv",
    "G": f"{QE_DIR}/QE_green.csv",
    "B": f"{QE_DIR}/QE_blue.csv",
    "W": f"{QE_DIR}/QE_mono.csv",
    "C": f"{QE_DIR}/QE_mono.csv",
    "Cy": f"{QE_DIR}/QE_cyan.csv",
    "Ye": f"{QE_DIR}/QE_yellow.csv",
    "Mg": f"{QE_DIR}/QE_magenta.csv",
}

PRESET_LAYOUTS: dict[str, tuple[tuple[str, ...], ...]] = {
    "RGGB": (("R", "G"), ("G", "B")),
    "BGGR": (("B", "G"), ("G", "R")),
    "GRBG": (("G", "R"), ("B", "G")),
    "GBRG": (("G", "B"), ("R", "G")),
    # 2x2 RGBW: one G of the Bayer quad replaced by a panchromatic site (as in
    # paper/ei2027/scripts/fig_cfa.py).
    "RGBW": (("R", "G"), ("W", "B")),
    # 4x4 RGBW with a 50 % panchromatic checkerboard; colour sites keep a Bayer-like order.
    "RGBW_4X4": (
        ("W", "G", "W", "R"),
        ("G", "W", "R", "W"),
        ("W", "B", "W", "G"),
        ("B", "W", "G", "W"),
    ),
    "RCCB": (("R", "C"), ("C", "B")),
    "RCCC": (("R", "C"), ("C", "C")),
    "RCCG": (("R", "C"), ("C", "G")),
    "RYYCY": (("R", "Ye"), ("Ye", "Cy")),
    "RGGCY": (("R", "G"), ("G", "Cy")),
    "CMY": (("Cy", "Mg"), ("Mg", "Ye")),
    "CYGM": (("Cy", "Ye"), ("G", "Mg")),
    "QUAD_BAYER": (
        ("R", "R", "G", "G"),
        ("R", "R", "G", "G"),
        ("G", "G", "B", "B"),
        ("G", "G", "B", "B"),
    ),
}


@dataclass(frozen=True)
class CfaLayout:
    """A periodic CFA: ``tile[r][c]`` is the channel at pixel ``(y, x)`` with ``y%rows==r``."""

    tile: tuple[tuple[str, ...], ...]
    qe_csv: Mapping[str, str]
    name: str = "custom"

    def __post_init__(self) -> None:
        if not self.tile or not self.tile[0]:
            raise ValueError("CFA tile must be a non-empty 2-D list of channel names")
        if len({len(r) for r in self.tile}) != 1:
            raise ValueError(f"CFA tile rows must have equal length, got {self.tile}")
        missing = [c for c in self.channels if c not in self.qe_csv]
        if missing:
            raise ValueError(f"no QE CSV for CFA channel(s) {missing}; set cfa.channels.<name>.qe_csv")

    @property
    def period(self) -> tuple[int, int]:
        return len(self.tile), len(self.tile[0])

    @property
    def channels(self) -> tuple[str, ...]:
        """Distinct channel names in raster order of first appearance."""
        seen: list[str] = []
        for row in self.tile:
            for ch in row:
                if ch not in seen:
                    seen.append(ch)
        return tuple(seen)

    @property
    def index_tile(self) -> np.ndarray:
        lut = {c: i for i, c in enumerate(self.channels)}
        return np.array([[lut[c] for c in row] for row in self.tile], dtype=np.intp)

    def site_index(self, h: int, w: int) -> np.ndarray:
        """HxW map of channel indices (into :attr:`channels`)."""
        pr, pc = self.period
        reps = (-(-h // pr), -(-w // pc))
        return np.tile(self.index_tile, reps)[:h, :w]

    def channel_mask(self, h: int, w: int, ch: int) -> np.ndarray:
        return self.site_index(h, w) == ch

    def density(self) -> np.ndarray:
        """Fraction of sites per channel."""
        idx = self.index_tile
        return np.array([(idx == i).mean() for i in range(len(self.channels))])


def _parse_tile(spec) -> tuple[str, tuple[tuple[str, ...], ...]]:
    if isinstance(spec, str):
        key = spec.strip().upper().replace("-", "_").replace(" ", "_")
        if key not in PRESET_LAYOUTS:
            raise ValueError(f"unknown CFA layout preset {spec!r}; presets: {sorted(PRESET_LAYOUTS)}")
        return key, PRESET_LAYOUTS[key]
    rows = tuple(tuple(str(c) for c in (r.split() if isinstance(r, str) else r)) for r in spec)
    return "custom", rows


def resolve_layout(cfa_cfg: Mapping | None) -> CfaLayout | None:
    """Build the generic layout from a ``cfa`` config section; ``None`` if ``layout`` is unset.

    ``cfa.layout`` is a preset name (see :data:`PRESET_LAYOUTS`) or a list of rows, each a list
    of channel names or a whitespace-separated string.  ``cfa.channels.<name>.qe_csv`` overrides
    or adds the QE curve of a channel (custom channel names are allowed).
    """
    if not cfa_cfg or cfa_cfg.get("layout") in (None, "", False):
        return None
    name, tile = _parse_tile(cfa_cfg["layout"])
    qe = dict(DEFAULT_CHANNEL_QE)
    for ch, spec in (cfa_cfg.get("channels") or {}).items():
        csv = spec.get("qe_csv") if isinstance(spec, Mapping) else spec
        if csv:
            qe[str(ch)] = str(csv)
    used = {c for row in tile for c in row}
    return CfaLayout(tile=tile, qe_csv={c: qe[c] for c in used if c in qe}, name=name)


def qe_stack_for_layout(
    repo: Path,
    layout: CfaLayout,
    lambdas_nm: np.ndarray,
    *,
    ircf_csv: str | None = None,
) -> np.ndarray:
    """``[C, K]`` QE per distinct channel on the bucket centres, times the IRCF if given.

    ``ircf_csv=None`` models a sensor without an IR-cut filter (bare colour-filter + silicon QE).
    Curves are zero outside their tabulated range (no extrapolation).
    """
    from qe_curves import read_csv_curve  # noqa: PLC0415

    lam = np.asarray(lambdas_nm, dtype=np.float64)
    ircf = np.ones_like(lam)
    if ircf_csv:
        i_wl, i_v = read_csv_curve((Path(repo) / ircf_csv).resolve())
        ircf = np.clip(np.interp(lam, i_wl, i_v, left=0.0, right=0.0), 0.0, 1.0)
    out = []
    for ch in layout.channels:
        q_wl, q_v = read_csv_curve((Path(repo) / layout.qe_csv[ch]).resolve())
        out.append(np.clip(np.interp(lam, q_wl, q_v, left=0.0, right=0.0) * ircf, 0.0, 1.0))
    return np.stack(out, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# Sampling, per-site parameters, crosstalk
# ---------------------------------------------------------------------------------------------
def mosaic(channel_img: np.ndarray, layout: CfaLayout) -> np.ndarray:
    """Sample an HxWxC channel image (C = len(layout.channels)) into the HxW raw mosaic."""
    img = np.asarray(channel_img)
    if img.ndim != 3 or img.shape[2] != len(layout.channels):
        raise ValueError(f"expected HxWx{len(layout.channels)} channel image, got {img.shape}")
    idx = layout.site_index(*img.shape[:2])
    return np.take_along_axis(img, idx[:, :, None], axis=2)[:, :, 0]


def site_parameter_map(values: Sequence[float], layout: CfaLayout, shape: tuple[int, int]) -> np.ndarray:
    """HxW map whose value at each site is ``values[channel_of_site]``."""
    v = np.asarray(values, dtype=np.float64)
    return v[layout.site_index(*shape)]


def channel_parameter(cfg: Mapping, base_key: str, ch: str, default: float) -> float:
    """Per-channel parameter lookup: ``<base>_<CH>`` then legacy ``<base>_r/g/b`` then ``<base>``."""
    for key in (f"{base_key}_{ch}", f"{base_key}_{ch.lower()}"):
        if key in cfg and cfg[key] is not None:
            return float(cfg[key])
    return float(cfg.get(base_key, default) if cfg.get(base_key) is not None else default)


def apply_site_crosstalk(raw_e: np.ndarray, layout: CfaLayout, cfg: Mapping | None) -> np.ndarray:
    """Lateral carrier-diffusion crosstalk on the raw mosaic, routed by *source* site.

    Charge generated under a site of channel ``c`` spreads to neighbouring wells with an
    isotropic Gaussian of width ``sigma_pixels_<c>`` (default ``sigma_pixels``): longer
    wavelengths are absorbed deeper and diffuse further, so the width is a property of the
    photon's colour, i.e. of the source site.  ``out = sum_c G_{sigma_c} * (raw . 1[site==c])``;
    each kernel sums to one so electrons are conserved (up to the reflect boundary).  This is a
    first-order empirical model of diffusion/optical crosstalk (no microlens/CRA dependence).
    """
    if not cfg or not bool(cfg.get("enabled", False)):
        return raw_e
    from scipy.ndimage import gaussian_filter  # noqa: PLC0415

    base = float(cfg.get("sigma_pixels", 0.3))
    sig = [channel_parameter(cfg, "sigma_pixels", ch, base) for ch in layout.channels]
    x = np.asarray(raw_e, dtype=np.float64)
    if len(set(sig)) == 1:
        out = gaussian_filter(x, sigma=sig[0], mode="reflect") if sig[0] > 0 else x
    else:
        idx = layout.site_index(*x.shape)
        out = np.zeros_like(x)
        for i, s in enumerate(sig):
            part = np.where(idx == i, x, 0.0)
            out += gaussian_filter(part, sigma=s, mode="reflect") if s > 0 else part
    return np.clip(out, 0.0, None).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# Demosaic
# ---------------------------------------------------------------------------------------------
def _coverage_radius(layout: CfaLayout, ch: int) -> tuple[int, int]:
    """Smallest tent half-width (ry, rx) such that every pixel sees >= 1 site of ``ch``."""
    pr, pc = layout.period
    mask = layout.index_tile == ch
    big = np.tile(mask, (3, 3))
    for s in range(1, max(pr, pc) + 1):
        ry, rx = min(s, pr), min(s, pc)
        ok = True
        for y in range(pr):
            for x in range(pc):
                win = big[pr + y - ry + 1 : pr + y + ry, pc + x - rx + 1 : pc + x + rx]
                if not win.any():
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return ry, rx
    return pr, pc


def _tent(ry: int, rx: int) -> np.ndarray:
    wy = 1.0 - np.abs(np.arange(-ry + 1, ry)) / ry
    wx = 1.0 - np.abs(np.arange(-rx + 1, rx)) / rx
    return np.outer(wy, wx)


def _normconv(values: np.ndarray, mask: np.ndarray, kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalised convolution; widens the tent where image borders leave a pixel without samples."""
    from scipy.ndimage import convolve  # noqa: PLC0415

    m = mask.astype(np.float64)
    v = np.where(mask, values, 0.0)
    num = convolve(v, kernel, mode="constant", cval=0.0)
    den = convolve(m, kernel, mode="constant", cval=0.0)
    ry, rx = (kernel.shape[0] + 1) // 2, (kernel.shape[1] + 1) // 2
    for _ in range(8):
        hole = den <= 1e-12
        if not hole.any() or not mask.any():
            break
        ry, rx = ry + 1, rx + 1
        k = _tent(ry, rx)
        num = np.where(hole, convolve(v, k, mode="constant", cval=0.0), num)
        den = np.where(hole, convolve(m, k, mode="constant", cval=0.0), den)
    return num, den


def demosaic_bilinear(raw: np.ndarray, layout: CfaLayout) -> np.ndarray:
    """HxW mosaic -> HxWxC by normalised convolution on each channel's own sites.

    The tent kernel half-width is the smallest that reaches a same-channel site from every pixel
    (2 for Bayer R/B/G, giving exactly the classic bilinear weights).  Zero padding plus
    normalisation handles image borders without breaking the CFA phase.
    """
    x = np.asarray(raw, dtype=np.float64)
    idx = layout.site_index(*x.shape)
    out = np.empty((*x.shape, len(layout.channels)), dtype=np.float64)
    for i in range(len(layout.channels)):
        mask = idx == i
        num, den = _normconv(x, mask, _tent(*_coverage_radius(layout, i)))
        plane = num / np.maximum(den, 1e-12)
        plane[mask] = x[mask]
        out[:, :, i] = plane
    return out.astype(np.float32)


def guide_channel(layout: CfaLayout) -> int:
    """The densest channel (ties: first in raster order) guides the colour-difference demosaic."""
    return int(np.argmax(layout.density()))


def demosaic_gradient(raw: np.ndarray, layout: CfaLayout, guide: int | None = None) -> np.ndarray:
    """Gradient-weighted guide interpolation + colour-difference interpolation of the others.

    1. Guide channel ``g`` (densest): horizontal-only and vertical-only normalised-convolution
       estimates are blended with weights ``1/(eps + |grad|)`` of the local gradient across that
       direction (interpolate *along* edges, Adams & Hamilton 1997); isotropic bilinear is used
       where a directional estimate has no support.
    2. Other channels: interpolate ``c - g`` on the sites of ``c`` (constant colour difference,
       Cok 1987) and add the full-resolution guide back.  At hard edges the colour difference
       taken on the bright side can push ``c`` far outside its neighbours (negative next to a
       bright lamp, which a CCM turns into saturated green), so each interpolated ``c`` is clamped
       to the range of the measured ``c`` samples under the interpolation tent.
    Measured samples are preserved exactly.
    """
    from scipy.ndimage import convolve1d  # noqa: PLC0415

    x = np.asarray(raw, dtype=np.float64)
    idx = layout.site_index(*x.shape)
    gi = guide_channel(layout) if guide is None else int(guide)
    gmask = idx == gi
    ry, rx = _coverage_radius(layout, gi)
    num, den = _normconv(x, gmask, _tent(ry, rx))
    g_iso = num / np.maximum(den, 1e-12)

    def _dir(axis: int, r: int) -> tuple[np.ndarray, np.ndarray]:
        k = 1.0 - np.abs(np.arange(-r + 1, r)) / r
        n = convolve1d(np.where(gmask, x, 0.0), k, axis=axis, mode="constant", cval=0.0)
        d = convolve1d(gmask.astype(np.float64), k, axis=axis, mode="constant", cval=0.0)
        return n / np.maximum(d, 1e-12), d > 1e-9

    gh, okh = _dir(1, max(rx, 2))
    gv, okv = _dir(0, max(ry, 2))
    grad_x = np.abs(np.gradient(g_iso, axis=1))
    grad_y = np.abs(np.gradient(g_iso, axis=0))
    eps = 1e-6 * (float(np.max(np.abs(x))) + 1.0)
    wh = np.where(okh, 1.0 / (eps + grad_x), 0.0)
    wv = np.where(okv, 1.0 / (eps + grad_y), 0.0)
    wsum = wh + wv
    g = np.where(wsum > 0, (wh * gh + wv * gv) / np.maximum(wsum, 1e-30), g_iso)
    g[gmask] = x[gmask]

    out = np.empty((*x.shape, len(layout.channels)), dtype=np.float64)
    for i in range(len(layout.channels)):
        if i == gi:
            out[:, :, i] = g
            continue
        mask = idx == i
        ry_c, rx_c = _coverage_radius(layout, i)
        num, den = _normconv(x - g, mask, _tent(ry_c, rx_c))
        plane = g + num / np.maximum(den, 1e-12)
        _clamp_to_local_range(plane, x, mask, ry_c, rx_c)
        plane[mask] = x[mask]
        out[:, :, i] = plane
    return out.astype(np.float32)


def _clamp_to_local_range(plane: np.ndarray, x: np.ndarray, mask: np.ndarray, ry: int, rx: int) -> None:
    """Clamp ``plane`` in place to the range of the measured samples (``mask``) under the tent."""
    from scipy.ndimage import maximum_filter, minimum_filter  # noqa: PLC0415

    lo = np.full(x.shape, np.inf)
    hi = np.full(x.shape, -np.inf)
    for grow in range(8):  # widen where a pixel has no own sample, as _normconv does
        size = (2 * ry - 1 + 2 * grow, 2 * rx - 1 + 2 * grow)
        todo = ~np.isfinite(lo)
        lo[todo] = minimum_filter(np.where(mask, x, np.inf), size=size, mode="constant", cval=np.inf)[todo]
        hi[todo] = maximum_filter(np.where(mask, x, -np.inf), size=size, mode="constant", cval=-np.inf)[todo]
        if np.isfinite(lo).all():
            break
    has = np.isfinite(lo)
    plane[has] = np.clip(plane[has], lo[has], hi[has])


def _shift(a: np.ndarray, k: int, axis: int) -> np.ndarray:
    """``out[i] = a[i + k]`` along ``axis``, edge-replicated."""
    n = a.shape[axis]
    i = np.clip(np.arange(n) + k, 0, n - 1)
    return np.take(a, i, axis=axis)


def edge_applies(layout: CfaLayout) -> bool:
    """The edge-directed method needs a guide on at least half the sites (Bayer G, RCCB C, the
    4x4 RGBW W, Quad Bayer G). Layouts without one (2x2 RGBW, CYGM) use :func:`demosaic_gradient`,
    which measured better on them."""
    return float(layout.density().max()) >= 0.5


def demosaic_edge(raw: np.ndarray, layout: CfaLayout, guide: int | None = None) -> np.ndarray:
    """Edge-directed guide + residual interpolation of the other channels (any periodic layout).

    1. Guide channel ``g`` (densest): horizontal and vertical 1-D estimates, with the
       Hamilton-Adams second-order term ``(2 c0 - c(-2a) - c(+2a)) / 4`` of the site's own channel
       wherever the guide sits at +-a and the own channel at +-2a along that direction (a = 1 for
       Bayer-like tiles, 2 for Quad Bayer's 2x2 clusters; Hamilton &
       Adams, US 5,629,734, 1997). The two are blended with weights ``1 / (eps + E_d)^2``, where
       ``E_d`` is the same-channel gradient energy of the mosaic summed over a 5x5 window (the
       windowed directional decision of RCD / Menon et al., IEEE TIP 16(1), 2007), and clamped to
       the range of the guide samples under the interpolation tent.
    2. Other channels: residual interpolation (Kiku et al., "Residual interpolation for color
       image demosaicking", ICIP 2013). A guided filter fitted on the channel's own sites gives a
       tentative ``a g + b``; the residual ``c - (a g + b)`` on those sites is interpolated by
       normalised convolution and added back, then clamped to the local range of measured ``c``
       as in :func:`demosaic_gradient`.
    Measured samples are preserved exactly; flat fields are reproduced exactly. Layouts without a
    50 % guide channel fall back to :func:`demosaic_gradient` (:func:`edge_applies`).
    """
    from scipy.ndimage import convolve1d, uniform_filter  # noqa: PLC0415

    if guide is None and not edge_applies(layout):
        return demosaic_gradient(raw, layout)

    x = np.asarray(raw, dtype=np.float64)
    idx = layout.site_index(*x.shape)
    gi = guide_channel(layout) if guide is None else int(guide)
    gmask = idx == gi
    ry, rx = _coverage_radius(layout, gi)
    num, den = _normconv(x, gmask, _tent(ry, rx))
    g_iso = num / np.maximum(den, 1e-12)
    pr, pc = layout.period
    scale = float(np.mean(np.abs(x))) + 1e-30
    eps = 1e-3 * scale

    est, wts = [], []
    for axis, r, per in ((1, max(rx, 2), pc), (0, max(ry, 2), pr)):
        k = 1.0 - np.abs(np.arange(-r + 1, r)) / r
        n = convolve1d(np.where(gmask, x, 0.0), k, axis=axis, mode="constant", cval=0.0)
        d = convolve1d(gmask.astype(np.float64), k, axis=axis, mode="constant", cval=0.0)
        e = n / np.maximum(d, 1e-12)
        ok = d > 1e-9
        done = gmask.copy()
        for a in (1, 2, 3):  # guide at +-a and own channel at +-2a (a = 2: Quad Bayer clusters)
            own = (_shift(idx, -2 * a, axis) == idx) & (_shift(idx, 2 * a, axis) == idx)
            gd = (_shift(idx, -a, axis) == gi) & (_shift(idx, a, axis) == gi)
            ha = ~done & own & gd
            lap = (2 * x - _shift(x, -2 * a, axis) - _shift(x, 2 * a, axis)) / 4.0
            e = np.where(ha, e + lap, e)
            done |= ha
        # same-channel gradient energy: sites one period apart always carry the same channel
        grad = np.abs(_shift(x, per, axis) - _shift(x, -per, axis)) / (2 * per)
        grad = grad + np.abs(np.gradient(g_iso, axis=axis))
        energy = uniform_filter(grad, size=5, mode="nearest") * 25.0
        est.append(e)
        wts.append(np.where(ok, 1.0 / (eps + energy) ** 2, 0.0))
    wsum = wts[0] + wts[1]
    g = np.where(wsum > 0, (wts[0] * est[0] + wts[1] * est[1]) / np.maximum(wsum, 1e-300), g_iso)
    _clamp_to_local_range(g, x, gmask, ry, rx)  # the second-order term can overshoot at hard edges
    g[gmask] = x[gmask]

    out = np.empty((*x.shape, len(layout.channels)), dtype=np.float64)
    size = 2 * max(pr, pc) + 1
    reg = (1e-2 * scale) ** 2
    for i in range(len(layout.channels)):
        if i == gi:
            out[:, :, i] = g
            continue
        mask = idx == i
        m = mask.astype(np.float64)
        cnt = uniform_filter(m, size=size, mode="constant")
        safe = np.maximum(cnt, 1e-12)
        mg = uniform_filter(m * g, size=size, mode="constant") / safe
        mc = uniform_filter(m * x, size=size, mode="constant") / safe
        cov = uniform_filter(m * g * x, size=size, mode="constant") / safe - mg * mc
        var = uniform_filter(m * g * g, size=size, mode="constant") / safe - mg * mg
        a = np.where(cnt > 1e-9, cov / (np.maximum(var, 0.0) + reg), 0.0)
        b = mc - a * mg
        tent = uniform_filter(a, size=size, mode="nearest") * g + uniform_filter(b, size=size, mode="nearest")
        ry_c, rx_c = _coverage_radius(layout, i)
        num, den = _normconv(np.where(mask, x - tent, 0.0), mask, _tent(ry_c, rx_c))
        plane = tent + num / np.maximum(den, 1e-12)
        _clamp_to_local_range(plane, x, mask, ry_c, rx_c)
        plane[mask] = x[mask]
        out[:, :, i] = plane
    return out.astype(np.float32)


def bayer_pattern(layout: CfaLayout) -> str | None:
    """``"RGGB"``/``"BGGR"``/``"GRBG"``/``"GBRG"`` for a 2x2 R/G/B Bayer layout, else ``None``."""
    flat = "".join(layout.tile[0] + layout.tile[1]) if layout.period == (2, 2) else ""
    return flat if flat in ("RGGB", "BGGR", "GRBG", "GBRG") else None


def demosaic(raw: np.ndarray, layout: CfaLayout, method: str = "gradient") -> np.ndarray:
    m = str(method).lower()
    if m == "bilinear":
        return demosaic_bilinear(raw, layout)
    if m == "malvar":
        pattern = bayer_pattern(layout)
        if pattern is not None:
            try:
                from apply_emva_noise import malvar_demosaic  # noqa: PLC0415 (apply_emva_noise imports this module)
            except ModuleNotFoundError:  # pragma: no cover - import path variant for tests
                from tools.apply_emva_noise import malvar_demosaic  # noqa: PLC0415
            rgb = malvar_demosaic(raw, pattern)
            return rgb[..., ["RGB".index(c) for c in layout.channels]]
        warnings.warn(f"Malvar demosaic is Bayer-only; CFA {layout.name!r} uses the gradient method", stacklevel=2)
        return demosaic_gradient(raw, layout)
    if m == "gradient":
        return demosaic_gradient(raw, layout)
    if m == "edge":
        return demosaic_edge(raw, layout)
    raise ValueError(f"unknown generic demosaic method {method!r} (bilinear | gradient | edge)")


# ---------------------------------------------------------------------------------------------
# Quad Bayer: binning and remosaic
# ---------------------------------------------------------------------------------------------
def binned_layout(layout: CfaLayout, block: int = 2) -> CfaLayout:
    """Layout after ``block x block`` same-colour binning (Quad Bayer -> RGGB)."""
    pr, pc = layout.period
    t = np.array(layout.tile, dtype=object)
    if pr % block or pc % block:
        raise ValueError(f"CFA period {layout.period} not divisible by binning block {block}")
    for y in range(0, pr, block):
        for x in range(0, pc, block):
            if len(set(t[y : y + block, x : x + block].ravel())) != 1:
                raise ValueError(f"layout {layout.name} has mixed-colour {block}x{block} cells; cannot bin")
    tile = tuple(tuple(str(c) for c in row) for row in t[::block, ::block])
    return CfaLayout(tile=tile, qe_csv=layout.qe_csv, name=f"{layout.name}_binned{block}")


def bin_mosaic(
    raw: np.ndarray, layout: CfaLayout, block: int = 2, *, mode: str = "sum"
) -> tuple[np.ndarray, CfaLayout]:
    """Sum (charge / digital-sum) or average ``block x block`` same-colour cells.

    Returns the ``(H/block) x (W/block)`` mosaic and its layout.  The image is cropped to a whole
    number of tiles first so cells never straddle the border.
    """
    out_layout = binned_layout(layout, block)
    pr, pc = layout.period
    x = np.asarray(raw, dtype=np.float64)
    h, w = (x.shape[0] // pr) * pr, (x.shape[1] // pc) * pc
    x = x[:h, :w].reshape(h // block, block, w // block, block)
    s = x.sum(axis=(1, 3))
    return (s / (block * block) if mode == "mean" else s), out_layout


def remosaic_to_bayer(
    raw: np.ndarray, layout: CfaLayout, pattern: str = "RGGB", method: str = "gradient"
) -> tuple[np.ndarray, CfaLayout]:
    """Reference remosaic: full-res demosaic (``method``, see :func:`demosaic`; ``edge`` is the
    edge-directed one), then resample on a standard Bayer tile. Sites whose colour is unchanged
    keep their measured value; the others take the interpolated estimate."""
    full = demosaic(raw, layout, method)
    target = CfaLayout(tile=PRESET_LAYOUTS[pattern.upper()], qe_csv=layout.qe_csv, name=pattern.upper())
    order = [layout.channels.index(c) for c in target.channels]
    return mosaic(full[:, :, order], target), target


# ---------------------------------------------------------------------------------------------
# Colour conversion
# ---------------------------------------------------------------------------------------------
def fit_channel_ccm(cam: np.ndarray, target: np.ndarray, *, white_index: int | None = None) -> np.ndarray:
    """``C x 3`` least-squares CCM with ``cam @ M ~= target`` (rows = patches or pixels).

    With ``white_index`` the fit is constrained so that patch maps exactly onto its target
    (white-point preserving), via a Lagrange-multiplier (KKT) solve per output column.
    """
    a = np.asarray(cam, dtype=np.float64).reshape(-1, np.shape(cam)[-1])
    t = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if white_index is None:
        m, *_ = np.linalg.lstsq(a, t, rcond=None)
        return m
    c = a.shape[1]
    w = a[white_index]
    kkt = np.zeros((c + 1, c + 1))
    kkt[:c, :c] = 2.0 * a.T @ a
    kkt[:c, c] = w
    kkt[c, :c] = w
    m = np.empty((c, 3))
    for j in range(3):
        rhs = np.concatenate([2.0 * a.T @ t[:, j], [t[white_index, j]]])
        m[:, j] = np.linalg.solve(kkt, rhs)[:c]
    return m


def apply_channel_ccm(img: np.ndarray, ccm: np.ndarray) -> np.ndarray:
    """HxWxC (or NxC) channels -> ... x 3 via ``x @ ccm``."""
    return np.tensordot(np.asarray(img, dtype=np.float64), np.asarray(ccm, dtype=np.float64), axes=([-1], [0]))


def colorchecker_targets(
    repo: Path, illuminant: str = "D65", space: str = "srgb"
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """24-patch targets under ``illuminant`` from the repo's spectral reflectances.

    Returns ``(target (24,3), xyz (24,3), names)``; XYZ is normalised so the perfect diffuser has
    ``Y = 1`` (CIE 1931 2deg).  ``space='srgb'`` gives linear sRGB (IEC 61966-2-1 matrix),
    ``'xyz'`` returns XYZ itself.
    """
    import colour_science as cs  # noqa: PLC0415

    lam = np.arange(380.0, 731.0, 1.0)
    chart = cs.load_colorchecker(DATA_ROOT, lam)
    _, spd = cs.load_illuminant(DATA_ROOT, illuminant, lam)
    xyz = cs.xyz_from_spectra(lam, chart.reflectance, spd)
    tgt = cs.xyz_to_srgb_linear(xyz) if space == "srgb" else xyz
    return tgt, xyz, chart.names


def delta_e2000_after_ccm(
    cam_patches: np.ndarray, xyz_ref: np.ndarray, ccm: np.ndarray, *, space: str = "srgb", white_xyz=None
) -> np.ndarray:
    """CIEDE2000 per patch of ``cam_patches @ ccm`` against reference XYZ (Y_white = 1)."""
    import colour_science as cs  # noqa: PLC0415

    out = apply_channel_ccm(cam_patches, ccm)
    xyz = cs.srgb_linear_to_xyz(out) if space == "srgb" else out
    white = cs.WHITE_D65 if white_xyz is None else np.asarray(white_xyz, dtype=np.float64)
    return cs.delta_e_2000(cs.xyz_to_lab(xyz, white), cs.xyz_to_lab(xyz_ref, white))


def colorchecker_spectral_ccm(
    repo: Path,
    layout: CfaLayout,
    *,
    ircf_csv: str | None = None,
    illuminant: str = "D65",
) -> tuple[np.ndarray, dict]:
    """White-preserving ``C x 3`` CCM fitted on the 24 ColorChecker patches *spectrally*.

    Camera responses ``s_c = sum R(lambda) S(lambda) QE_c(lambda) dlambda`` (normalised to the
    perfect diffuser) are regressed on the patches' linear-sRGB values under ``illuminant``,
    constrained so the white patch maps exactly to its target.  This is the CCM a calibration
    on an ideal, noise-free ColorChecker render would give for this QE set.
    """
    import colour_science as cs  # noqa: PLC0415

    lam = np.arange(380.0, 731.0, 1.0)
    chart = cs.load_colorchecker(DATA_ROOT, lam)
    _, spd = cs.load_illuminant(DATA_ROOT, illuminant, lam)
    qe = qe_stack_for_layout(repo, layout, lam, ircf_csv=ircf_csv).astype(np.float64)
    radiance = cs.radiance_spectra(chart.reflectance, spd)
    cam = cs.integrate_spectra(lam, radiance, qe)
    white_cam = cs.integrate_spectra(lam, np.atleast_2d(spd), qe)[0]
    cam = cam / np.maximum(white_cam, 1e-30)[None, :]
    xyz = cs.xyz_from_spectra(lam, chart.reflectance, spd)
    tgt = cs.xyz_to_srgb_linear(xyz)
    wi = int(np.argmax(xyz[:, 1]))
    ccm = fit_channel_ccm(cam, tgt / tgt[wi].mean(), white_index=wi)
    return ccm, {"ccm_source": f"colorchecker_spectral_{illuminant}", "ccm_white_patch": chart.names[wi]}


def channel_wb_gains(img: np.ndarray, method: str = "white_patch", top_pct: float = 1.0) -> np.ndarray:
    """Per-channel gains equalising a neutral reference (brightest ``top_pct`` % or the mean)."""
    x = np.asarray(img, dtype=np.float64).reshape(-1, np.shape(img)[-1])
    if method == "white_patch":
        lum = x.mean(axis=1)
        thr = np.percentile(lum, 100.0 - top_pct)
        sel = x[lum >= thr]
        ref = sel.mean(axis=0) if sel.size else x.mean(axis=0)
    else:
        ref = x.mean(axis=0)
    ref = np.maximum(ref, 1e-9)
    return ref.mean() / ref


def clip_highlights(img: np.ndarray, gains: np.ndarray, clip_level: float | None) -> np.ndarray:
    """Cap every channel at ``clip_level * min(gains)``, the brightest neutral the sensor can record.

    A raw-clipped pixel reads ``clip_level`` in every channel, so per-channel gains (white balance, or
    a CCM's column sums) turn it into ``clip_level * gains``: a colour cast, magenta when green has the
    smallest gain.  Capping at the smallest gained ceiling renders clipped areas white (dcraw ``-H 0``).
    """
    if clip_level is None:
        return img
    g = np.asarray(gains, dtype=np.float64)
    if not np.all(np.isfinite(g)) or np.min(g) <= 0.0:
        return img
    return np.minimum(img, float(clip_level) * float(np.min(g))).astype(np.asarray(img).dtype, copy=False)


def desaturate_near_clip(
    rgb: np.ndarray, raw: np.ndarray, clip_level: float | None, radius: int = 2, start: float = 0.9
) -> np.ndarray:
    """Blend display RGB toward its luminance where the raw mosaic nears its clip level.

    Next to a clipped highlight the demosaic mixes clipped and unclipped samples of different
    channels, so one channel stops growing while the others do: a coloured fringe along window and
    lamp edges. The weight ramps from 0 at ``start * clip_level`` to 1 at the clip level, using the
    largest raw sample within ``radius`` pixels (the demosaic's support). Display only; the raw data
    and the metrics are not touched."""
    if clip_level is None or clip_level <= 0:
        return rgb
    r = np.asarray(raw, dtype=np.float64)
    pad = np.pad(r, radius, mode="edge")
    h, w = r.shape
    m = np.full((h, w), -np.inf)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            m = np.maximum(m, pad[dy : dy + h, dx : dx + w])
    wgt = np.clip((m / float(clip_level) - start) / (1.0 - start), 0.0, 1.0)[:, :, None]
    if not np.any(wgt):
        return rgb
    y = np.tensordot(rgb, np.array([0.2126, 0.7152, 0.0722]), axes=([2], [0]))[:, :, None]
    return (rgb * (1.0 - wgt) + y * wgt).astype(np.asarray(rgb).dtype, copy=False)


def render_cfa_rgb(
    dn_images: Sequence[np.ndarray],
    black_dn: float,
    layout: CfaLayout,
    *,
    repo: Path,
    cfa_cfg: Mapping,
    ircf_csv: str | None = None,
    wb_method: str | None = "white_patch",
    reference_rgb: np.ndarray | None = None,
    clip_dn: float | None = None,
) -> tuple[list[np.ndarray], dict]:
    """Demosaic raw DN mosaics and convert channels -> linear sRGB (DN scale, black re-added).

    ``cfa.ccm``: ``colorchecker_spectral`` (default; :func:`colorchecker_spectral_ccm`),
    ``exr_reference`` (pixel-wise least squares against the EXR's RGB, needs an unbinned
    image) or an explicit ``C x 3`` matrix.  White balance gains (fitted on the first image)
    are applied to the channels before the CCM.  ``clip_dn`` (raw clip level above black) caps clipped
    highlights at the brightest neutral (:func:`clip_highlights`) after white balance and after the CCM.
    """
    method = str(cfa_cfg.get("demosaic", "gradient"))
    chans = [demosaic(np.asarray(d, dtype=np.float64) - black_dn, layout, method) for d in dn_images]
    spec = cfa_cfg.get("ccm", "colorchecker_spectral")
    m = method.lower()
    info: dict = {
        "demosaic": "edge" if m == "edge" and edge_applies(layout) else ("bilinear" if m == "bilinear" else "gradient")
    }
    gains = np.ones(len(layout.channels))
    if wb_method:
        gains = channel_wb_gains(np.clip(chans[0], 0.0, None), wb_method)
    chans = [clip_highlights(c * gains[None, None, :], gains, clip_dn) for c in chans]
    if isinstance(spec, (list, tuple)):
        ccm = np.asarray(spec, dtype=np.float64)
        info["ccm_source"] = "config"
    elif spec == "exr_reference" and reference_rgb is not None and reference_rgb.shape[:2] == chans[0].shape[:2]:
        a, t = chans[0].reshape(-1, chans[0].shape[2]), np.asarray(reference_rgb, dtype=np.float64).reshape(-1, 3)
        t = t * (np.median(a.mean(axis=1)) / max(np.median(t.mean(axis=1)), 1e-12))
        ccm = fit_channel_ccm(a, t)
        info["ccm_source"] = "exr_reference"
    else:
        ccm, meta = colorchecker_spectral_ccm(
            repo, layout, ircf_csv=ircf_csv, illuminant=str(cfa_cfg.get("ccm_illuminant", "D65"))
        )
        info.update(meta)
    if ccm.shape != (len(layout.channels), 3):
        raise ValueError(f"cfa.ccm must be {len(layout.channels)}x3 for channels {layout.channels}")
    info["ccm"] = ccm.tolist()
    info["wb_gains"] = gains.tolist()
    out_clip = None if clip_dn is None else clip_dn * float(np.min(gains))
    rgb = [clip_highlights(np.clip(apply_channel_ccm(c, ccm), 0.0, None), ccm.sum(axis=0), out_clip) for c in chans]
    if clip_dn is not None and bool(cfa_cfg.get("highlight_desaturation", True)):
        rgb = [
            desaturate_near_clip(c, np.asarray(d, dtype=np.float64) - black_dn, clip_dn)
            for c, d in zip(rgb, dn_images, strict=True)
        ]
        info["highlight_desaturation"] = True
    rgb = [c + black_dn for c in rgb]
    return [r.astype(np.float32) for r in rgb], info
