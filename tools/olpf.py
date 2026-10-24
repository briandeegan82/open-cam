#!/usr/bin/env python3
"""Optical low-pass (anti-aliasing) filter: birefringent beam splitting in front of the sensor.

A plate of uniaxial crystal cut with its optic axis at ``axis_angle_deg`` to the plate normal splits
every ray into an ordinary and an extraordinary ray of equal power. The extraordinary ray walks off
sideways by ``d = t tan(rho)``, with (Yariv & Yeh, *Optical Waves in Crystals*, sec. 4.3)

    tan(rho) = (1 - n_o^2/n_e^2) tan(theta) / (1 + (n_o^2/n_e^2) tan^2(theta)).

Two crossed plates (with a depolarising quarter-wave plate between them) give the usual four-spot
filter: point spread 1/4 at (+-dx/2, +-dy/2), transfer function cos(pi fx dx) cos(pi fy dy), with its
first zero at fx = 1 / (2 dx). A split of one pixel pitch puts the zero at the pixel Nyquist frequency.

The spot separation is given either directly (``split_x_um`` / ``split_y_um`` or ``split_pixels``,
wavelength independent) or as a quartz plate thickness (``thickness_x_um`` / ``thickness_y_um``),
for which d(lambda) follows the quartz dispersion of Ghosh, Opt. Commun. 163, 95 (1999).

The filter acts on the spectral EXR before the radiance -> electrons conversion, one EXR pixel per
sensor pixel (as in that conversion). The EXR is already integrated over each pixel, so the filter is
applied as its transfer function on the sampled image. The spots sit at (0, 0), (d, 0), (0, d), (d, d)
as behind a real plate (the ordinary ray is undeviated), so integer splits are exact pixel shifts;
fractional splits (``interpolation: box``, default) take the irradiance as constant over each EXR
pixel, so a spot d = n + f px away re-integrates to weights (1 - f, f) on pixels n and n + 1: a
non-negative kernel that cannot ring around very bright sources (lamps, windows, the sun) and adds a
little blur. ``interpolation: fourier`` is the band-limited alternative (exact cos OTF, but it rings
along rows and columns at edges far above the pixel Nyquist, which pbrt's box-filtered pixels are).
Neither removes aliasing that pbrt already folded into the pixel grid above Nyquist.

Angle of incidence (``angle_dependence``, opt-in, quartz ``thickness_*`` filters only): off axis the
chief ray reaches the plates at the chief-ray angle (CRA). Each plate is then traced exactly: the
ordinary ray refracts with n_o; the extraordinary wave vector satisfies
(k.c)^2 / n_o^2 + (|k|^2 - (k.c)^2) / n_e^2 = 1 for the plate's optic axis c, and its ray runs along
the gradient of that surface. The split is the exit-point difference of the two rays; at normal
incidence it reduces to t tan(rho). The image is filtered per field zone (bilinear weights over a
``field_grid``) with that zone's two 2-D split vectors (off axis each plate also splits slightly
across its own axis). The CRA comes from ``lens.chief_ray_angle_table`` (``cra: table``) or rises
linearly to ``max_cra_deg`` at the frame corner (``cra: linear``).

Light cone (``cone``, opt-in, quartz filters): a pixel receives a cone of rays around the chief ray,
of image-space numerical aperture 1 / (2 N) for the working f-number N. Each plate's split depends on
the incidence angle, so with ``cone.enabled`` the split is traced for ``samples`` rays spread evenly
over the pupil (direction sines in a disk of radius 1 / (2 N) around the chief ray) and the three
displaced spots become clouds of 3 x samples spots. A plane-parallel plate also moves the focus back
by t (1 - 1/n_o) (ordinary ray, paraxial). With ``focus_compensated: true`` (default) the sensor is
taken to sit at that shifted focus, as in a camera designed with its filter stack, so the ordinary
spot stays sharp and only the extraordinary rays' extra spread remains. ``focus_compensated: false``
models a plate added behind a lens focused without it: every ray then also lands t (1 - 1/n_o) x its
direction sine (per unit cos) off, a defocus disc of diameter about t (1 - 1/n_o) / N.

Not modelled: the IR-cut coating and cover-glass of the filter stack (see
``sensor.quantum_efficiency.ircf_csv``).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

# Ghosh (1999) quartz, n^2 - 1 = B0 + B1 l^2 / (l^2 - C1) + B2 l^2 / (l^2 - C2), l in um
# (refractiveindex.info "formula 2", SiO2 Ghosh-o / Ghosh-e, 0.198-2.05 um).
QUARTZ_GHOSH_O = (0.28604141, 1.07044083, 1.00585997e-2, 1.10202242, 100.0)
QUARTZ_GHOSH_E = (0.28851804, 1.09509924, 1.02101864e-2, 1.15662475, 100.0)

DEFAULTS = {
    "enabled": False,
    "split_pixels": None,
    "split_x_um": None,
    "split_y_um": None,
    "thickness_x_um": None,
    "thickness_y_um": None,
    "axis_angle_deg": 45.0,
    "exr_pixel_pitch_um": None,
    # Opt-in: {enabled, cra: table|linear, max_cra_deg, field_grid: [gy, gx]}.
    "angle_dependence": None,
    "interpolation": "box",
    # Opt-in: {enabled, samples, f_number (default: the camera's working f-number), focus_compensated}.
    "cone": None,
}

CONE_DEFAULTS = {"enabled": False, "samples": 16, "f_number": None, "focus_compensated": True}

ANGLE_DEFAULTS = {"enabled": False, "cra": "linear", "max_cra_deg": 25.0, "field_grid": [5, 7]}


def _ghosh(c: tuple[float, ...], lam_nm) -> np.ndarray:
    l2 = (np.asarray(lam_nm, dtype=np.float64) * 1e-3) ** 2
    return np.sqrt(1.0 + c[0] + c[1] * l2 / (l2 - c[2]) + c[3] * l2 / (l2 - c[4]))


def quartz_indices(lam_nm) -> tuple[np.ndarray, np.ndarray]:
    """(n_o, n_e) of crystalline quartz at ``lam_nm``."""
    return _ghosh(QUARTZ_GHOSH_O, lam_nm), _ghosh(QUARTZ_GHOSH_E, lam_nm)


def walkoff_tan(lam_nm, axis_angle_deg: float = 45.0) -> np.ndarray:
    """tan(rho) of the extraordinary ray in quartz with the optic axis at ``axis_angle_deg`` to the normal."""
    n_o, n_e = quartz_indices(lam_nm)
    r = (n_o / n_e) ** 2
    t = math.tan(math.radians(axis_angle_deg))
    return (1.0 - r) * t / (1.0 + r * t * t)


def plate_split_um(thickness_um: float, lam_nm, axis_angle_deg: float, plate: str, sin_x: float, sin_y: float):
    """(dx, dy) [um] of extraordinary minus ordinary exit point for a quartz plate (exact uniaxial trace).

    ``plate`` "x": optic axis in the x-z plane at ``axis_angle_deg`` to the normal (z); "y": in y-z.
    ``sin_x``, ``sin_y``: direction sines of the incident ray in air (0, 0 = normal incidence).
    """
    n_o, n_e = quartz_indices(lam_nm)
    th = math.radians(axis_angle_deg)
    cx, cy = (math.sin(th), 0.0) if plate == "x" else (0.0, math.sin(th))
    cz = math.cos(th)
    a, b = float(sin_x), float(sin_y)
    q = 1.0 / n_o**2 - 1.0 / n_e**2
    m = a * cx + b * cy
    qa = q * cz * cz + 1.0 / n_e**2
    qb = 2.0 * q * m * cz
    qc = q * m * m + (a * a + b * b) / n_e**2 - 1.0
    z = (-qb + np.sqrt(qb * qb - 4.0 * qa * qc)) / (2.0 * qa)
    kc = m + z * cz
    gx, gy, gz = q * kc * cx + a / n_e**2, q * kc * cy + b / n_e**2, q * kc * cz + z / n_e**2
    zo = np.sqrt(n_o**2 - a * a - b * b)
    return thickness_um * (gx / gz - a / zo), thickness_um * (gy / gz - b / zo)


def _cra_deg(h_norm: float, ad: dict, lens_cfg: dict | None) -> float:
    if str(ad.get("cra", "linear")) == "table":
        table = np.asarray((lens_cfg or {}).get("chief_ray_angle_table") or [], dtype=np.float64)
        if table.ndim != 2 or table.shape[1] != 2:
            raise ValueError(
                "olpf.angle_dependence cra: table needs lens.chief_ray_angle_table [[h_norm, cra_deg], ...]"
            )
        return float(np.interp(h_norm, table[:, 0], table[:, 1]))
    return float(ad.get("max_cra_deg", 25.0)) * h_norm


def zone_splits(cfg: dict, lam_nm: float, h: int, w: int, lens_cfg: dict | None = None) -> list[dict]:
    """Per field-zone centre: (row, col, CRA, split px on x and y, cross-axis px) at ``lam_nm``."""
    ad = {**ANGLE_DEFAULTS, **(cfg.get("angle_dependence") or {})}
    gy, gx = (int(v) for v in ad["field_grid"])
    rows = np.linspace(0.0, h - 1.0, gy) if gy > 1 else np.array([0.5 * (h - 1)])
    cols = np.linspace(0.0, w - 1.0, gx) if gx > 1 else np.array([0.5 * (w - 1)])
    cy0, cx0 = 0.5 * (h - 1), 0.5 * (w - 1)
    r_corner = math.hypot(cy0, cx0) or 1.0
    pitch = float(cfg["pixel_pitch_um"])
    out = []
    for r in rows:
        for c in cols:
            ux, uy = c - cx0, r - cy0
            rad = math.hypot(ux, uy)
            cra = _cra_deg(rad / r_corner, ad, lens_cfg)
            s = math.sin(math.radians(cra))
            sx, sy = (s * ux / rad, s * uy / rad) if rad > 0 else (0.0, 0.0)
            ex = ey = (0.0, 0.0)
            if cfg.get("thickness_x_um") is not None:
                ex = plate_split_um(float(cfg["thickness_x_um"]), lam_nm, float(cfg["axis_angle_deg"]), "x", sx, sy)
            if cfg.get("thickness_y_um") is not None:
                ey = plate_split_um(float(cfg["thickness_y_um"]), lam_nm, float(cfg["axis_angle_deg"]), "y", sx, sy)
            out.append(
                {
                    "row": float(r),
                    "col": float(c),
                    "cra_deg": cra,
                    "dx_px": float(ex[0]) / pitch,
                    "dy_px": float(ey[1]) / pitch,
                    "d1_px": (float(ex[0]) / pitch, float(ex[1]) / pitch),
                    "d2_px": (float(ey[0]) / pitch, float(ey[1]) / pitch),
                    "cross_px": max(abs(float(ex[1])), abs(float(ey[0]))) / pitch,
                }
            )
    return out


def pupil_sines(f_number: float, samples: int) -> np.ndarray:
    """(samples, 2) direction-sine offsets spread evenly (sunflower) over a disk of radius 1 / (2 N)."""
    k = np.arange(int(samples), dtype=np.float64)
    r = (0.5 / float(f_number)) * np.sqrt((k + 0.5) / len(k))
    phi = k * math.pi * (3.0 - math.sqrt(5.0))
    return np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)


def focus_shift_um(cfg: dict, lam_nm: float = 550.0) -> float:
    """Paraxial focus shift of the quartz plates, sum of t (1 - 1/n_o) [um]."""
    n_o = float(quartz_indices(lam_nm)[0])
    t = sum(float(cfg[k]) for k in ("thickness_x_um", "thickness_y_um") if cfg.get(k) is not None)
    return t * (1.0 - 1.0 / n_o)


def cone_spots(
    cfg: dict, lam_nm: float, sin_x: float, sin_y: float, f_number: float, samples: int, compensated: bool = True
) -> tuple[list[tuple[float, float]], list[float]]:
    """Four-spot pattern summed over pupil rays around a chief ray with direction sines (sin_x, sin_y).

    Returns (spots in px, weights summing to 1): per pupil ray the ordinary spot and the three
    extraordinary ones (plate x, plate y, both). With ``compensated`` the ordinary spot is the focus
    reference (0, 0); otherwise every spot of a ray is also displaced by the plates' defocus.
    """
    pitch = float(cfg["pixel_pitch_um"])
    ax = float(cfg["axis_angle_deg"])
    n_o = float(quartz_indices(lam_nm)[0])
    tx, ty = cfg.get("thickness_x_um"), cfg.get("thickness_y_um")
    spots, weights = [], []
    pupil = pupil_sines(f_number, samples)
    for du, dv in pupil:
        a, b = sin_x + du, sin_y + dv
        e1 = plate_split_um(float(tx), lam_nm, ax, "x", a, b) if tx is not None else (0.0, 0.0)
        e2 = plate_split_um(float(ty), lam_nm, ax, "y", a, b) if ty is not None else (0.0, 0.0)
        ox = oy = 0.0
        if not compensated:
            # ordinary-ray exit offset relative to the unrefracted ray, minus the chief ray's (pure shift)
            for t in (tx, ty):
                if t is not None:
                    for o, s0, s in ((0, sin_x, a), (1, sin_y, b)):
                        d = float(t) * (s / math.sqrt(1 - a * a - b * b) - s / math.sqrt(n_o**2 - a * a - b * b))
                        d0 = float(t) * (
                            s0 / math.sqrt(1 - sin_x**2 - sin_y**2) - s0 / math.sqrt(n_o**2 - sin_x**2 - sin_y**2)
                        )
                        if o == 0:
                            ox += d - d0
                        else:
                            oy += d - d0
        o = (ox / pitch, oy / pitch)
        d1 = (float(e1[0]) / pitch, float(e1[1]) / pitch)
        d2 = (float(e2[0]) / pitch, float(e2[1]) / pitch)
        for sx, sy in ((0.0, 0.0), d1, d2, (d1[0] + d2[0], d1[1] + d2[1])):
            spots.append((o[0] + sx, o[1] + sy))
            weights.append(1.0 / (4 * len(pupil)))
    return spots, weights


def _hat(n: int, size: int) -> np.ndarray:
    """(n, size) linear hat weights over ``size`` samples; columns sum to 1."""
    if n == 1:
        return np.ones((1, size))
    x = np.arange(size, dtype=np.float64)
    nodes = np.linspace(0.0, size - 1.0, n)
    return np.stack([np.interp(x, nodes, np.eye(n)[k]) for k in range(n)])


def resolve_config(sensor_cfg: dict) -> dict:
    cfg = {**DEFAULTS, **((sensor_cfg or {}).get("olpf") or {})}
    pitch = cfg["exr_pixel_pitch_um"] or (sensor_cfg or {}).get("pixel_pitch_um")
    cfg["pixel_pitch_um"] = None if pitch is None else float(pitch)
    return cfg


def split_px(cfg: dict, axis: str, lam_nm) -> np.ndarray:
    """Spot separation along ``axis`` ("x" or "y") in EXR pixels at each wavelength (0: no split)."""
    lam = np.atleast_1d(np.asarray(lam_nm, dtype=np.float64))
    keys = [k for k in (f"split_{axis}_um", f"thickness_{axis}_um") if cfg.get(k) is not None]
    if cfg.get("split_pixels") is not None:
        if keys:
            raise ValueError(f"olpf: give split_pixels or {' / '.join(keys)}, not both")
        return np.full(lam.shape, float(cfg["split_pixels"]))
    if len(keys) > 1:
        raise ValueError(f"olpf: give split_{axis}_um or thickness_{axis}_um, not both")
    if not keys:
        return np.zeros(lam.shape)
    if cfg.get("pixel_pitch_um") is None:
        raise KeyError("olpf: split in um needs sensor.pixel_pitch_um (or olpf.exr_pixel_pitch_um)")
    if keys[0].startswith("split"):
        d_um = np.full(lam.shape, float(cfg[keys[0]]))
    else:
        d_um = float(cfg[keys[0]]) * walkoff_tan(lam, float(cfg["axis_angle_deg"]))
    return d_um / float(cfg["pixel_pitch_um"])


def transfer(fx, fy, dx_px: float, dy_px: float) -> np.ndarray:
    """OTF of the four-spot filter at frequencies in cycles / pixel (real, may be negative)."""
    return np.cos(np.pi * np.asarray(fx) * dx_px) * np.cos(np.pi * np.asarray(fy) * dy_px)


def spot_sum(img: np.ndarray, spots, weights=None) -> np.ndarray:
    """Mean of copies of ``img`` moved by each (sx, sy) px spot, out(x) = mean a(x - s), with box-pixel
    (bilinear) weights for fractional moves; reflect-padded. Non-negative weights: no ringing.
    ``weights`` (summing to 1) replaces the plain mean."""
    a = np.asarray(img, dtype=np.float64)
    pad = int(math.ceil(max(max(abs(sx), abs(sy)) for sx, sy in spots))) + 2
    p = np.pad(a, pad, mode="reflect")
    h, w = a.shape
    out = np.zeros_like(a)
    ws = [1.0 / len(spots)] * len(spots) if weights is None else list(weights)
    for (sx, sy), ws_k in zip(spots, ws):
        ix, iy = math.floor(sx), math.floor(sy)
        fx, fy = sx - ix, sy - iy
        for oy, wy in ((iy, 1.0 - fy), (iy + 1, fy)):
            for ox, wx in ((ix, 1.0 - fx), (ix + 1, fx)):
                if wy * wx:
                    out += ws_k * wy * wx * p[pad - oy : pad - oy + h, pad - ox : pad - ox + w]
    return out


def apply_plane(img: np.ndarray, dx_px: float, dy_px: float, interpolation: str = "box") -> np.ndarray:
    """Filter one image plane: spots at 0 and +d on each axis, reflect-padded.

    ``box``: ``spot_sum`` (box-pixel weights, no ringing). ``fourier``: ``transfer`` times a d/2 shift
    (band-limited; rings along rows and columns around very bright sources at fractional splits).
    Spots at +-d/2 would need half-pixel shifts even for odd integer splits."""
    if dx_px == 0 and dy_px == 0:
        return img
    if interpolation == "box":
        spots = [(0.0, 0.0), (dx_px, 0.0), (0.0, dy_px), (dx_px, dy_px)]
        return spot_sum(img, spots).astype(img.dtype, copy=False)
    a = np.asarray(img, dtype=np.float64)
    pad = int(math.ceil(max(abs(dx_px), abs(dy_px)))) + 4
    p = np.pad(a, pad, mode="reflect")
    fy = np.fft.fftfreq(p.shape[0])[:, None]
    fx = np.fft.rfftfreq(p.shape[1])[None, :]
    shift = np.exp(-1j * np.pi * (fx * dx_px + fy * dy_px))
    out = np.fft.irfft2(np.fft.rfft2(p) * transfer(fx, fy, dx_px, dy_px) * shift, s=p.shape)
    return out[pad:-pad, pad:-pad].astype(img.dtype, copy=False)


def apply_plane_vec(
    img: np.ndarray, d1: tuple[float, float], d2: tuple[float, float], interpolation: str = "box"
) -> np.ndarray:
    """Four spots at 0, d1, d2, d1 + d2 (2-D split vectors in px): ``apply_plane`` for oblique splits."""
    if not any((*d1, *d2)):
        return img
    if interpolation == "box":
        return spot_sum(img, [(0.0, 0.0), tuple(d1), tuple(d2), (d1[0] + d2[0], d1[1] + d2[1])])
    a = np.asarray(img, dtype=np.float64)
    pad = int(math.ceil(abs(d1[0]) + abs(d2[0]) + abs(d1[1]) + abs(d2[1]))) + 4
    p = np.pad(a, pad, mode="reflect")
    fy = np.fft.fftfreq(p.shape[0])[:, None]
    fx = np.fft.rfftfreq(p.shape[1])[None, :]
    h1 = np.cos(np.pi * (fx * d1[0] + fy * d1[1]))
    h2 = np.cos(np.pi * (fx * d2[0] + fy * d2[1]))
    shift = np.exp(-1j * np.pi * (fx * (d1[0] + d2[0]) + fy * (d1[1] + d2[1])))
    out = np.fft.irfft2(np.fft.rfft2(p) * h1 * h2 * shift, s=p.shape)
    return out[pad:-pad, pad:-pad]


def apply_channels(
    chans: dict[str, np.ndarray], sensor_cfg: dict, lens_cfg: dict | None = None, f_number: float | None = None
) -> tuple[dict[str, np.ndarray], dict]:
    """Apply ``sensor.olpf`` to every image channel (S0.<lambda>nm at its wavelength, others at 550 nm).

    With ``interpolation: fourier``, negative undershoot of the band-limited interpolation at hard edges
    (a few % for fractional splits) is clipped to zero; the clipped energy is reported.
    """
    from exr_multispectral import parse_s0_wavelength_nm

    cfg = resolve_config(sensor_cfg)
    if not cfg["enabled"]:
        return chans, {"enabled": False}
    ad = {**ANGLE_DEFAULTS, **(cfg.get("angle_dependence") or {})}
    angular = bool(ad["enabled"])
    interp = str(cfg.get("interpolation") or "box")
    if interp not in ("box", "fourier"):
        raise ValueError(f"sensor.olpf.interpolation must be box or fourier, got {interp!r}")
    if angular and (cfg.get("split_pixels") is not None or cfg.get("split_x_um") or cfg.get("split_y_um")):
        raise ValueError("olpf.angle_dependence needs the quartz plates (thickness_x_um / thickness_y_um)")
    cone = {**CONE_DEFAULTS, **(cfg.get("cone") or {})}
    use_cone = bool(cone["enabled"])
    if use_cone:
        if cfg.get("thickness_x_um") is None and cfg.get("thickness_y_um") is None:
            raise ValueError("olpf.cone needs the quartz plates (thickness_x_um / thickness_y_um)")
        if interp != "box":
            raise ValueError("olpf.cone needs interpolation: box")
        if cone["f_number"] is not None:
            f_number = float(cone["f_number"])
        elif f_number is None:
            from camera_model import effective_f_number

            f_number = effective_f_number(sensor_cfg or {}, lens_cfg)
        samples = int(cone["samples"])
        if samples < 1 or not f_number > 0:
            raise ValueError("olpf.cone needs samples >= 1 and a positive f_number")
    out, splits, clipped, total = {}, {}, 0.0, 0.0
    zone_info = None
    for name, arr in chans.items():
        if arr.ndim != 2 or name.upper() in ("A", "ALPHA") or name.startswith(("P.", "N.", "Albedo")):
            out[name] = arr
            continue
        lam = parse_s0_wavelength_nm(name) or 550.0
        if use_cone:
            h, w = arr.shape
            if angular:
                zones = zone_splits(cfg, lam, h, w, lens_cfg)
                gy, gx = (int(v) for v in ad["field_grid"])
            else:
                zones = zone_splits({**cfg, "angle_dependence": {"field_grid": [1, 1], "max_cra_deg": 0.0}}, lam, h, w)
                gy = gx = 1
            wy, wx = _hat(gy, h), _hat(gx, w)
            cy0, cx0 = 0.5 * (h - 1), 0.5 * (w - 1)
            f = np.zeros(arr.shape, dtype=np.float64)
            for k, z in enumerate(zones):
                ux, uy = z["col"] - cx0, z["row"] - cy0
                rad = math.hypot(ux, uy)
                s = math.sin(math.radians(z["cra_deg"]))
                sx, sy = (s * ux / rad, s * uy / rad) if rad > 0 else (0.0, 0.0)
                spots, sw = cone_spots(cfg, lam, sx, sy, f_number, samples, bool(cone["focus_compensated"]))
                wz = wy[k // gx][:, None] * wx[k % gx][None, :]
                f += spot_sum(arr.astype(np.float64) * wz, spots, sw)
            dx = min(z["dx_px"] for z in zones)
            dy = min(z["dy_px"] for z in zones)
            if angular and (zone_info is None or abs(lam - 550.0) < abs(zone_info["lambda_nm"] - 550.0)):
                zone_info = {"lambda_nm": lam, "zones": zones}
        elif angular:
            h, w = arr.shape
            zones = zone_splits(cfg, lam, h, w, lens_cfg)
            gy, gx = (int(v) for v in ad["field_grid"])
            wy, wx = _hat(gy, h), _hat(gx, w)
            f = np.zeros(arr.shape, dtype=np.float64)
            for k, z in enumerate(zones):
                wz = wy[k // gx][:, None] * wx[k % gx][None, :]
                f += apply_plane_vec(arr.astype(np.float64) * wz, z["d1_px"], z["d2_px"], interp)
            dx = min(z["dx_px"] for z in zones)
            dy = min(z["dy_px"] for z in zones)
            if zone_info is None or abs(lam - 550.0) < abs(zone_info["lambda_nm"] - 550.0):
                zone_info = {"lambda_nm": lam, "zones": zones}
        else:
            dx, dy = float(split_px(cfg, "x", lam)[0]), float(split_px(cfg, "y", lam)[0])
            f = apply_plane(arr, dx, dy, interp)
        neg = f < 0
        clipped += float(-f[neg].sum())
        total += float(np.abs(arr).sum())
        out[name] = np.where(neg, 0, f).astype(arr.dtype, copy=False)
        splits[name] = (round(dx, 4), round(dy, 4))
    vals = list(splits.values())
    info = {
        "enabled": True,
        "split_px_range": [
            min(v[0] for v in vals),
            max(v[0] for v in vals),
            min(v[1] for v in vals),
            max(v[1] for v in vals),
        ]
        if vals
        else None,
        "clipped_energy_fraction": clipped / total if total else 0.0,
        "interpolation": interp,
    }
    if use_cone:
        dz = focus_shift_um(cfg)
        info["cone"] = {
            "f_number": f_number,
            "samples": samples,
            "focus_compensated": bool(cone["focus_compensated"]),
            "focus_shift_um": dz,
            "defocus_disc_px": 0.0 if cone["focus_compensated"] else dz / f_number / float(cfg["pixel_pitch_um"]),
        }
    if zone_info is not None:
        zs = zone_info["zones"]
        info["angle_dependence"] = {
            "cra": ad["cra"],
            "field_grid": list(ad["field_grid"]),
            "lambda_nm": zone_info["lambda_nm"],
            "dx_px_range": [min(z["dx_px"] for z in zs), max(z["dx_px"] for z in zs)],
            "dy_px_range": [min(z["dy_px"] for z in zs), max(z["dy_px"] for z in zs)],
            "max_cross_px": max(z["cross_px"] for z in zs),
            "max_cra_deg": max(z["cra_deg"] for z in zs),
        }
    return out, info


def apply_cube(
    cube: np.ndarray, lam_nm, sensor_cfg: dict, lens_cfg: dict | None = None, f_number: float | None = None
) -> tuple[np.ndarray, dict]:
    """``apply_channels`` for an H x W x K spectral cube with bucket wavelengths ``lam_nm``."""
    lam = np.asarray(lam_nm, dtype=np.float64).ravel()
    names = [f"S0.{x:g}nm" for x in lam]
    if not resolve_config(sensor_cfg)["enabled"]:
        return cube, {"enabled": False}
    out, info = apply_channels({n: cube[:, :, k] for k, n in enumerate(names)}, sensor_cfg, lens_cfg, f_number)
    return np.stack([out[n] for n in names], axis=2), info


def main() -> None:
    from camera_model import load_camera_model
    from exr_multispectral import read_separate_exr_channels, write_separate_channels_exr

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo-root", type=Path, default=TOOLS.parent)
    ap.add_argument("--camera-model-config", type=Path, required=True)
    ap.add_argument("--exr-in", type=Path, required=True)
    ap.add_argument("--exr-out", type=Path, default=None, help="default: overwrite --exr-in")
    args = ap.parse_args()
    sensor = load_camera_model(args.camera_model_config.resolve()).get("sensor") or {}
    if not resolve_config(sensor)["enabled"]:
        print("sensor.olpf.enabled is false; nothing to do.", file=sys.stderr)
        return
    exr_in = args.exr_in if args.exr_in.is_absolute() else (args.repo_root / args.exr_in).resolve()
    out, info = apply_channels(read_separate_exr_channels(exr_in), sensor)
    exr_out = (args.exr_out or exr_in).resolve()
    exr_out.parent.mkdir(parents=True, exist_ok=True)
    write_separate_channels_exr(exr_out, out)
    print(f"wrote {exr_out} (olpf split px x/y range {info['split_px_range']})")


if __name__ == "__main__":
    main()
