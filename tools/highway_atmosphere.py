#!/usr/bin/env python3
"""Haze / mist / fog for the highway scenes: a spectral pbrt-v4 participating medium.

The medium is parameterised the way visibility is reported in meteorology:

* Meteorological optical range / visibility ``V`` [m] fixes the extinction coefficient at
  550 nm through Koschmieder's law ``sigma_ext = -ln(0.02) / V = 3.912 / V`` (2 % contrast
  threshold; WMO-No. 8, Guide to Meteorological Instruments, ch. 9; Middleton 1952).
* Wavelength dependence of the aerosol part follows the Angstrom law
  ``sigma_aer(lambda) = sigma_aer(550) (lambda / 550 nm)^-alpha`` (Angstrom 1929; alpha ~1.3
  for continental haze, ~0.5 in humid mist, ~0 for fog droplets of 5-20 um; Seinfeld & Pandis,
  Atmospheric Chemistry and Physics, ch. 15).
* Molecular (Rayleigh) scattering of sea-level air is included in the same total
  (1.16e-5 /m at 550 nm, lambda^-4.08; Bucholtz 1995, Appl. Opt. 34, 2765), so ``V`` is the
  total visibility; the aerosol part is ``3.912/V - sigma_R(550)``.
* A single Henyey-Greenstein phase function (pbrt media support one): g ~0.7 for haze
  aerosol (d'Almeida et al. 1991, continental models g = 0.64-0.72), ~0.85 for fog droplets
  (Mie, 5-15 um water drops). Molecular scattering is folded in by the scattering-weighted
  mean g at 550 nm (an approximation of the Rayleigh + HG mixture).
* Single-scattering albedo omega of the aerosol (0.9-0.95 continental haze, ~1 for water
  droplets; Hess, Koepke & Schult 1998, OPAC).

Vertical structure: haze uses an exponential profile with a 1-1.2 km scale height (boundary-
layer aerosol); mist and fog use a uniform ground layer with a cosine taper at the top
(radiation fog is typically 50-300 m deep). pbrt-v4 has no analytic height-falloff medium,
so the profile is a ``uniformgrid`` medium that is uniform horizontally (4 x 4 samples over
+-200 km) and sampled finely in height. The camera, the lights and every surface are inside
it (``MediumInterface "haze" "haze"`` before the camera, so no surface is a medium boundary).

Radiometry. The sun and sky light sources stay outside the scene (pbrt ``distant`` and
``infinite`` lights at infinity), so their illuminance is the illuminance at the *top of the
layer*; pbrt attenuates and scatters it on the way down. The horizontal illuminance at the
road is computed here with a Monte-Carlo plane-parallel radiative-transfer model that uses
exactly the medium given to pbrt (same piecewise-linear density profile, sigma(lambda),
omega, HG g) plus a Lambertian ground of albedo ``ground_albedo``; see :func:`slab_flux_mc`.
That road illuminance is what the scene manifest records as
``lighting.reference_illuminance_lux`` (``--haze-light-reference above``, the default), or
the lights are rescaled so that the road illuminance equals the clear-sky/override value
(``--haze-light-reference road``). Either way the manifest's absolute photometry matches
the rendered EXR (checked against pbrt in tests/test_pbrt_e2e.py).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass, replace

import numpy as np

KOSCHMIEDER = -math.log(0.02)  # 3.912
RAYLEIGH_550_PER_M = 1.16e-5
RAYLEIGH_EXPONENT = 4.08
GRID_HALF_WIDTH_M = 200_000.0
GRID_NXZ = 4
GRID_NY = 256
ROAD_REF_HEIGHT_M = 1.35  # visibility V is defined at camera height


@dataclass(frozen=True)
class HazeParams:
    name: str
    visibility_m: float
    angstrom: float
    g: float
    albedo: float
    profile: str  # "exponential" (scale height) or "layer" (uniform up to height, cosine taper)
    height_m: float
    description: str = ""


PRESETS = {
    # International visibility code: clear 20-50 km, light haze 4-10 km, mist 1-2 km (WMO: mist
    # >= 1 km at high humidity), fog < 1 km (moderate ~200-500 m, thick < 200 m).
    "clear": HazeParams("clear", 30_000.0, 1.3, 0.70, 0.95, "exponential", 1200.0, "clear day, continental aerosol"),
    "hazy": HazeParams("hazy", 8_000.0, 1.0, 0.70, 0.90, "exponential", 1000.0, "light haze"),
    "mist": HazeParams("mist", 2_000.0, 0.5, 0.75, 0.98, "layer", 300.0, "humid mist, 300 m layer"),
    "fog": HazeParams("fog", 150.0, 0.0, 0.85, 0.999, "layer", 150.0, "thick radiation fog, 150 m layer"),
}


def add_cli_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("haze / fog (participating medium, see docs/HIGHWAY_SCENES.md)")
    g.add_argument("--haze", choices=("none", *PRESETS), default="none", help="Atmosphere preset.")
    g.add_argument("--visibility-m", type=float, default=None, help="Meteorological visibility V [m].")
    g.add_argument("--haze-angstrom", type=float, default=None, help="Angstrom exponent alpha.")
    g.add_argument("--haze-g", type=float, default=None, help="Henyey-Greenstein asymmetry g.")
    g.add_argument("--haze-albedo", type=float, default=None, help="Aerosol single-scattering albedo.")
    g.add_argument("--haze-profile", choices=("exponential", "layer"), default=None)
    g.add_argument("--haze-height-m", type=float, default=None, help="Scale height / layer depth [m].")
    g.add_argument(
        "--haze-light-reference",
        choices=("above", "road"),
        default="above",
        help="above: sun/sky illuminance is at the top of the layer (road gets less, physically); "
        "road: rescale the lights so the road illuminance equals the no-medium value.",
    )
    g.add_argument(
        "--haze-ground-albedo", type=float, default=0.15, help="Mean ground albedo for the road-illuminance model."
    )
    g.add_argument(
        "--haze-maxdepth", type=int, default=None, help="volpath maxdepth (default: preset-dependent, >= --maxdepth)."
    )


def params_from_args(args: argparse.Namespace) -> HazeParams | None:
    if getattr(args, "haze", "none") == "none":
        return None
    p = PRESETS[args.haze]
    over = {
        "visibility_m": args.visibility_m,
        "angstrom": args.haze_angstrom,
        "g": args.haze_g,
        "albedo": args.haze_albedo,
        "profile": args.haze_profile,
        "height_m": args.haze_height_m,
    }
    p = replace(p, **{k: v for k, v in over.items() if v is not None})
    if p.visibility_m <= 0 or not 0.0 < p.albedo <= 1.0 or not -1.0 < p.g < 1.0 or p.height_m <= 0:
        raise ValueError(f"invalid haze parameters: {p}")
    return p


def integrator_maxdepth(p: HazeParams, base: int, override: int | None = None) -> int:
    """Path depth that resolves the multiple scattering (each medium event counts as a bounce)."""
    if override is not None:
        return int(override)
    tau = optical_depth_vertical(p, 550.0)
    need = 8 if tau < 0.5 else 24 if tau < 2.0 else 64
    return max(int(base), need)


# ---- optical properties


def coefficients(p: HazeParams, wl_nm) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(sigma_a, sigma_s, sigma_ext) [1/m] at ground level (density 1)."""
    wl = np.asarray(wl_nm, dtype=np.float64)
    sig550 = KOSCHMIEDER / p.visibility_m
    aer550 = max(sig550 - RAYLEIGH_550_PER_M, 0.0)
    aer = aer550 * (wl / 550.0) ** (-p.angstrom)
    ray = RAYLEIGH_550_PER_M * (wl / 550.0) ** (-RAYLEIGH_EXPONENT)
    sa = (1.0 - p.albedo) * aer
    ss = p.albedo * aer + ray
    return sa, ss, sa + ss


def effective_g(p: HazeParams) -> float:
    """Scattering-weighted HG g of aerosol (g) + Rayleigh (0) at 550 nm."""
    _, ss, _ = coefficients(p, 550.0)
    ss_aer = float(ss) - RAYLEIGH_550_PER_M
    return p.g * ss_aer / float(ss) if float(ss) > 0 else 0.0


def medium_top_m(p: HazeParams) -> float:
    return 5.0 * p.height_m if p.profile == "exponential" else 1.3 * p.height_m


def density(p: HazeParams, y) -> np.ndarray:
    """Analytic relative density (1 at the ground)."""
    y = np.clip(np.asarray(y, dtype=np.float64), 0.0, None)
    top = medium_top_m(p)
    if p.profile == "exponential":
        return np.where(y < top, np.exp(-y / p.height_m), 0.0)
    h = p.height_m
    taper = 0.5 * (1.0 + np.cos(np.pi * np.clip((y - h) / (top - h), 0.0, 1.0)))
    return np.where(y <= h, 1.0, np.where(y < top, taper, 0.0))


def grid_y(p: HazeParams) -> tuple[float, float, np.ndarray]:
    """(y0, y1, sample centres) of the pbrt grid; two cells below ground so y>=0 is fully inside."""
    top = medium_top_m(p)
    dy = top / (GRID_NY - 3)
    y0 = -2.0 * dy
    y1 = y0 + GRID_NY * dy
    return y0, y1, y0 + (np.arange(GRID_NY) + 0.5) * dy


def grid_profile(p: HazeParams) -> tuple[np.ndarray, np.ndarray]:
    """(y centres, density samples) exactly as given to pbrt (trilinear -> piecewise linear in y)."""
    _, _, yc = grid_y(p)
    rho = density(p, yc)
    rho[-1] = 0.0
    return yc, rho / np.interp(ROAD_REF_HEIGHT_M, yc, rho)


def profile_density(p: HazeParams, y) -> np.ndarray:
    """Density pbrt evaluates at height y (piecewise-linear interpolation of the grid samples)."""
    yc, rho = grid_profile(p)
    return np.interp(np.asarray(y, dtype=np.float64), yc, rho, left=rho[0], right=0.0)


def column_density_m(p: HazeParams) -> float:
    """Integral of the relative density from the ground to the top [m]."""
    yc, rho = grid_profile(p)
    y = np.linspace(0.0, yc[-1], 20001)
    d = profile_density(p, y)
    return float(np.sum(0.5 * (d[1:] + d[:-1]) * np.diff(y)))


def optical_depth_vertical(p: HazeParams, wl_nm: float = 550.0) -> float:
    return float(coefficients(p, wl_nm)[2]) * column_density_m(p)


# ---- pbrt output


def pbrt_camera_lines() -> list[str]:
    """Before ``Camera``: put the camera (and all later shapes/lights) inside the medium."""
    return ['MediumInterface "haze" "haze"']


def pbrt_medium_lines(p: HazeParams, sigma_a_spd: str, sigma_s_spd: str) -> list[str]:
    """After ``Camera`` and before ``WorldBegin`` (pbrt resolves the camera medium at WorldBegin)."""
    y0, y1, _ = grid_y(p)
    _, rho = grid_profile(p)
    vals = np.tile(np.repeat(rho, GRID_NXZ)[None, :], (GRID_NXZ, 1)).ravel()  # index (z*ny + y)*nx + x
    w = GRID_HALF_WIDTH_M
    return [
        f"# Atmosphere: {p.name} (V = {p.visibility_m:g} m, alpha = {p.angstrom:g}, g = {effective_g(p):.3f},"
        f" omega = {p.albedo:g}, {p.profile} {p.height_m:g} m)",
        "Identity",  # medium in world space (the CTM still holds the LookAt here)
        'MakeNamedMedium "haze" "string type" "uniformgrid"',
        f'    "spectrum sigma_a" "{sigma_a_spd}" "spectrum sigma_s" "{sigma_s_spd}" "float scale" [1]',
        f'    "float g" [{effective_g(p):.6g}]',
        f'    "integer nx" [{GRID_NXZ}] "integer ny" [{GRID_NY}] "integer nz" [{GRID_NXZ}]',
        f'    "point3 p0" [{-w:.6g} {y0:.6g} {-w:.6g}] "point3 p1" [{w:.6g} {y1:.6g} {w:.6g}]',
        '    "float density" [ ' + " ".join(f"{v:.6g}" for v in vals) + " ]",
    ]


# ---- road illuminance (plane-parallel Monte Carlo, same medium as pbrt)


def _sample_hg(g: float, mu: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = mu.size
    u = rng.random(n)
    if abs(g) < 1e-4:
        cos_t = 1.0 - 2.0 * u
    else:
        s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u)
        cos_t = (1.0 + g * g - s * s) / (2.0 * g)
    cos_t = np.clip(cos_t, -1.0, 1.0)
    phi = 2.0 * np.pi * rng.random(n)
    sin_t = np.sqrt(1.0 - cos_t**2)
    return np.clip(mu * cos_t + np.sqrt(np.clip(1.0 - mu * mu, 0.0, None)) * sin_t * np.cos(phi), -1.0, 1.0)


def slab_flux_mc(
    p: HazeParams,
    wl_nm: float,
    mu0: np.ndarray,
    ground_albedo: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """(total, direct) downward flux at the ground per unit incident horizontal flux at the top.

    ``mu0``: cosines of the incidence zenith angle of each photon, drawn from the source's
    horizontal-flux distribution. Delta tracking through the piecewise-linear density profile,
    HG scattering with single-scattering albedo, Lambertian ground reflection with absorption
    weight; photons leaving through the top are lost. Every downward ground crossing counts.
    """
    sa, ss, se = (float(v) for v in coefficients(p, wl_nm))
    omega = ss / se
    g = effective_g(p)
    yc, rho = grid_profile(p)
    top = float(yc[-1])
    sig_maj = se * float(rho.max())
    n = mu0.size
    y = np.full(n, top)
    mu = -np.asarray(mu0, dtype=np.float64)
    w = np.ones(n)
    scattered = np.zeros(n, dtype=bool)
    total = direct = 0.0
    for _ in range(100_000):
        if y.size == 0:
            break
        t = -np.log(1.0 - rng.random(y.size)) / sig_maj
        y_new = y + mu * t
        hit = y_new <= 0.0
        out = y_new >= top
        if hit.any():
            total += float(w[hit].sum())
            direct += float(w[hit & ~scattered].sum())
        coll = ~hit & ~out
        real = coll & (rng.random(y.size) * float(rho.max()) < profile_density(p, y_new))
        y = np.where(hit, 0.0, np.where(coll, y_new, y))
        w = np.where(hit, w * ground_albedo, np.where(real, w * omega, w))
        mu = np.where(hit, np.sqrt(rng.random(y.size)), mu)
        if real.any():
            mu[real] = _sample_hg(g, mu[real], rng)
        scattered = scattered | hit | real
        # Russian roulette on low weights, drop escaped photons.
        low = w < 1e-3
        keep_rr = rng.random(y.size) < 0.1
        w = np.where(low & keep_rr, w * 10.0, w)
        alive = ~out & ~(low & ~keep_rr)
        y, mu, w, scattered = y[alive], mu[alive], w[alive], scattered[alive]
    return total / n, direct / n


def _sample_cos(weights: np.ndarray, cos_z: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    c = np.cumsum(weights)
    idx = np.searchsorted(c, rng.random(n) * c[-1])
    return cos_z[np.clip(idx, 0, cos_z.size - 1)]


def road_illuminance(
    p: HazeParams,
    *,
    sun_elevation_deg: float,
    e_sun_normal_top: float,
    e_sky_horizontal_top: float,
    sky_cos_zenith: np.ndarray,
    sky_weights: np.ndarray,
    ground_albedo: float,
    wl_nm: np.ndarray,
    spectral_weight: np.ndarray,
    photons: int = 20_000,
    seed: int = 1,
) -> dict:
    """Horizontal illuminance at the road under the medium (photometric, spectrally weighted).

    ``sky_cos_zenith``/``sky_weights``: zenith cosine and horizontal-flux weight (luminance x
    cos x solid angle) of the sky map's pixels. ``spectral_weight``: V(lambda) x source SPD
    on ``wl_nm`` (used for both sun and sky).
    """
    rng = np.random.default_rng(seed)
    mu_sun = max(math.sin(math.radians(sun_elevation_deg)), 1e-3)
    sw = np.asarray(spectral_weight, dtype=np.float64)
    sw = sw / sw.sum()
    t_sun = t_sun_dir = t_sky = 0.0
    for wl, wgt in zip(wl_nm, sw):
        if wgt <= 0:
            continue
        tot, dr = slab_flux_mc(p, float(wl), np.full(photons, mu_sun), ground_albedo, rng)
        t_sun += wgt * tot
        t_sun_dir += wgt * dr
        mu_sky = _sample_cos(sky_weights, sky_cos_zenith, photons, rng)
        t_sky += wgt * slab_flux_mc(p, float(wl), mu_sky, ground_albedo, rng)[0]
    e_sun_h_top = e_sun_normal_top * mu_sun
    e_road = e_sun_h_top * t_sun + e_sky_horizontal_top * t_sky
    return {
        "sun_horizontal_top_lux": e_sun_h_top,
        "sky_horizontal_top_lux": e_sky_horizontal_top,
        "sun_direct_road_lux": e_sun_h_top * t_sun_dir,
        "sun_direct_transmittance": t_sun_dir,
        "sun_total_transmittance": t_sun,
        "sky_total_transmittance": t_sky,
        "road_lux": e_road,
        "diffuse_fraction_road": 1.0 - e_sun_h_top * t_sun_dir / max(e_road, 1e-12),
        "mc_photons_per_wavelength": photons,
        "ground_albedo": ground_albedo,
    }


def sky_flux_distribution(sky_rgb_equal_area: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(cos zenith, horizontal-flux weight) of each upper-hemisphere pixel of an equal-area sky map."""
    from highway_sky import LUMA, equal_area_directions

    d = equal_area_directions(sky_rgb_equal_area.shape[0])
    cz = d[..., 2].ravel()
    w = np.clip(sky_rgb_equal_area.reshape(-1, 3) @ LUMA, 0.0, None) * np.clip(cz, 0.0, None)
    up = cz > 0
    return cz[up], w[up]


def manifest_entry(p: HazeParams, wl_nm: np.ndarray) -> dict:
    sa, ss, se = coefficients(p, np.array([450.0, 550.0, 650.0]))
    return {
        **asdict(p),
        "koschmieder_sigma_ext_550_per_m": KOSCHMIEDER / p.visibility_m,
        "sigma_ext_per_m_450_550_650": [float(v) for v in se],
        "single_scattering_albedo_550": float(ss[1] / se[1]),
        "hg_g_effective": effective_g(p),
        "medium_top_m": medium_top_m(p),
        "vertical_optical_depth_550": optical_depth_vertical(p, 550.0),
        "pbrt_medium": "uniformgrid 'haze' (camera, lights and surfaces inside), integrator volpath",
    }


def setup_scene(
    p: HazeParams,
    args: argparse.Namespace,
    *,
    wl: np.ndarray,
    spd_dir,
    sky_rgb_equal_area: np.ndarray,
    sun_spd: np.ndarray,
    elev: float,
    e_dn: float,
    e_sky_h: float,
    e_ref: float,
) -> dict:
    """Write the medium spectra and work out the light levels for the builder.

    Returns the (possibly rescaled) top-of-layer light illuminances, the road illuminance for
    the manifest, the pbrt lines and the manifest entry.
    """
    from colour_science import CMF_WAVELENGTH_NM, CMF_Y

    sa, ss, _ = coefficients(p, wl)
    for name, val in (("haze_sigma_a", sa), ("haze_sigma_s", ss)):
        (spd_dir / f"{name}.spd").write_text("\n".join(f"{w:.1f} {v:.6g}" for w, v in zip(wl, val)) + "\n")
    wl_mc = np.arange(400.0, 701.0, 20.0)
    weight = np.interp(wl_mc, CMF_WAVELENGTH_NM, CMF_Y) * np.interp(wl_mc, wl, sun_spd)
    cz, w = sky_flux_distribution(sky_rgb_equal_area)
    road = road_illuminance(
        p,
        sun_elevation_deg=elev,
        e_sun_normal_top=e_dn,
        e_sky_horizontal_top=e_sky_h,
        sky_cos_zenith=cz,
        sky_weights=w,
        ground_albedo=args.haze_ground_albedo,
        wl_nm=wl_mc,
        spectral_weight=weight,
    )
    k = 1.0
    if args.haze_light_reference == "road":
        k = e_ref / road["road_lux"]
        road = {key: (v * k if key.endswith("_lux") else v) for key, v in road.items()}
    maxdepth = integrator_maxdepth(p, args.maxdepth, args.haze_maxdepth)
    entry = manifest_entry(p, wl)
    entry.update(
        {
            "light_reference": args.haze_light_reference,
            "light_scale_applied": k,
            "no_medium_horizontal_illuminance_lux": e_ref,
            "road_illuminance": road,
            "volpath_maxdepth": maxdepth,
            "spectra": {"sigma_a": "spd/haze_sigma_a.spd", "sigma_s": "spd/haze_sigma_s.spd"},
        }
    )
    return {
        "e_dn": e_dn * k,
        "e_sky_h": e_sky_h * k,
        "e_ref": float(road["road_lux"]),
        "maxdepth": maxdepth,
        "camera_lines": pbrt_camera_lines(),
        "medium_lines": pbrt_medium_lines(p, "spd/haze_sigma_a.spd", "spd/haze_sigma_s.spd"),
        "manifest": entry,
    }
