#!/usr/bin/env python3
"""Spectral materials for highway scenes: analytic reflectance curves + PBRT emitters.

Reflectance curves are smooth analytic approximations of published spectra
(urban spectral libraries for asphalt, TiO2/organic pigments for paints,
chlorophyll well + red edge for vegetation). They are written as CSV files
(``wavelength_nm,reflectance`` with ``#`` comments, same reader conventions as
the illuminant CSVs) under ``spectra/surfaces/highway/`` by ``--write-spectra``.

Retroreflective sign sheeting is approximated as bright diffuse — pbrt has no
retroreflective BRDF. Scene manifests record this approximation.

Also provides GPU-safe PBRT material block emitters (diffuse/coateddiffuse
with spectrum SPDs, conductor with built-in metal spectra, dielectric glass)
and Pillow rendering of sign-legend textures.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# 5 nm grid matches the SPD subsampling default used by the scene builders.
WL_GRID = np.arange(380.0, 780.0 + 1e-9, 5.0)
# Solar SPD written at 1 nm to match spectra/illuminant/interpolated/*.csv.
WL_GRID_1NM = np.arange(380.0, 780.0 + 1e-9, 1.0)


def _sigmoid(wl: np.ndarray, center: float, width: float) -> np.ndarray:
    """Smooth long-pass edge rising around `center` over ~`width` nm."""
    return 1.0 / (1.0 + np.exp(-(wl - center) / max(width, 1e-6)))


def _gauss(wl: np.ndarray, center: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((wl - center) / max(sigma, 1e-6)) ** 2)


# ---------------------------------------------------------------------------
# Surface reflectance registry
# ---------------------------------------------------------------------------

def _asphalt_aged(wl):
    # Herold et al. (2004): aged asphalt ~0.10-0.16, near-flat with gentle rise.
    return 0.10 + 0.06 * (wl - 380.0) / 400.0


def _asphalt_new(wl):
    # Fresh asphalt is much darker (bitumen-rich): ~0.05, slight rise.
    return 0.045 + 0.015 * (wl - 380.0) / 400.0


def _road_paint_white(wl):
    # Waterborne TiO2 traffic paint: UV absorption edge near 400 nm, then flat.
    return 0.25 + 0.50 * _sigmoid(wl, 415.0, 12.0)


def _road_paint_yellow(wl):
    # Lead-free organic yellow: dark below ~480 nm, sigmoid rise to ~0.65.
    return 0.06 + 0.59 * _sigmoid(wl, 505.0, 16.0)


def _sign_sheeting_white(wl):
    # Engineering-grade sheeting (daytime diffuse component): bright, near flat.
    return 0.80 + 0.05 * _sigmoid(wl, 420.0, 15.0)


def _sign_sheeting_red(wl):
    return 0.05 + 0.65 * _sigmoid(wl, 600.0, 12.0)


def _sign_sheeting_green(wl):
    return 0.04 + 0.33 * _gauss(wl, 530.0, 40.0)


def _sign_sheeting_blue(wl):
    return 0.04 + 0.40 * _gauss(wl, 460.0, 35.0)


def _car_paint_white(wl):
    return 0.55 + 0.25 * _sigmoid(wl, 430.0, 18.0)


def _car_paint_black(wl):
    return np.full_like(wl, 0.04)


def _car_paint_silver(wl):
    # Metallic silver base coat: flat mid-gray with slight blue tilt.
    return 0.48 - 0.05 * (wl - 380.0) / 400.0


def _car_paint_gray(wl):
    return np.full_like(wl, 0.25)


def _car_paint_red(wl):
    return 0.05 + 0.60 * _sigmoid(wl, 600.0, 14.0)


def _car_paint_blue(wl):
    return 0.05 + 0.42 * _gauss(wl, 465.0, 38.0) + 0.03 * _sigmoid(wl, 700.0, 25.0)


def _car_glass_tint(wl):
    # Darkened opaque stand-in for glazing (--opaque-glass): slight blue-green.
    return 0.06 + 0.04 * _gauss(wl, 500.0, 60.0)


def _rubber_tire(wl):
    # Carbon black: very dark, nearly flat.
    return 0.035 + 0.01 * (wl - 380.0) / 400.0


def _grass_green(wl):
    # Chlorophyll well at 450/670 nm, green peak at 550, strong NIR red edge.
    base = 0.05
    green_peak = 0.08 * _gauss(wl, 552.0, 32.0)
    chlorophyll_dip = -0.02 * _gauss(wl, 670.0, 18.0)
    red_edge = 0.40 * _sigmoid(wl, 715.0, 12.0)
    return np.clip(base + green_peak + chlorophyll_dip + red_edge, 0.02, 1.0)


def _soil_dry(wl):
    # Dry soil: monotonic ramp typical of iron-oxide-bearing soils.
    return 0.08 + 0.22 * (wl - 380.0) / 400.0


SURFACE_CURVES = {
    "asphalt_aged": _asphalt_aged,
    "asphalt_new": _asphalt_new,
    "road_paint_white": _road_paint_white,
    "road_paint_yellow": _road_paint_yellow,
    "sign_sheeting_white": _sign_sheeting_white,
    "sign_sheeting_red": _sign_sheeting_red,
    "sign_sheeting_green": _sign_sheeting_green,
    "sign_sheeting_blue": _sign_sheeting_blue,
    "car_paint_white": _car_paint_white,
    "car_paint_black": _car_paint_black,
    "car_paint_silver": _car_paint_silver,
    "car_paint_gray": _car_paint_gray,
    "car_paint_red": _car_paint_red,
    "car_paint_blue": _car_paint_blue,
    "car_glass_tint": _car_glass_tint,
    "rubber_tire": _rubber_tire,
    "grass_green": _grass_green,
    "soil_dry": _soil_dry,
}

CAR_PAINTS = ("white", "black", "silver", "gray", "red", "blue")


def synth_spectrum(name: str, wl: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (wavelength_nm, reflectance) for a registered surface."""
    if name not in SURFACE_CURVES:
        raise KeyError(f"unknown surface '{name}'; known: {sorted(SURFACE_CURVES)}")
    w = WL_GRID if wl is None else np.asarray(wl, dtype=np.float64)
    v = np.clip(SURFACE_CURVES[name](w), 0.0, 1.0)
    return w, v


# ---------------------------------------------------------------------------
# Direct-normal solar SPD (ASTM G-173 AM1.5 smooth approximation)
# ---------------------------------------------------------------------------

def solar_direct_am15(wl: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Smooth AM1.5 direct+circumsolar shape, normalized to 1.0 at 560 nm.

    5800 K Planck envelope attenuated by Rayleigh scattering (~lambda^-4,
    stronger at short wavelengths for 1.5 air masses), an ozone Chappuis-band
    dip, and Gaussian notches for the main O2/H2O absorption bands within
    380-780 nm (O2-B 687 nm, H2O 719 nm, O2-A 761 nm).
    """
    w = WL_GRID_1NM if wl is None else np.asarray(wl, dtype=np.float64)
    wm = w * 1e-9
    h, c, kb, t = 6.62607015e-34, 2.99792458e8, 1.380649e-23, 5800.0
    planck = (1.0 / wm**5) / (np.exp(h * c / (wm * kb * t)) - 1.0)

    # Rayleigh optical depth ~ (lambda/550nm)^-4 scaled for AM1.5.
    tau_rayleigh = 0.098 * (w / 550.0) ** -4 * 1.5
    # Chappuis ozone band (broad, weak, centered ~600 nm).
    tau_ozone = 0.035 * _gauss(w, 600.0, 80.0)
    # Molecular absorption notches (depths tuned to G-173 direct spectrum).
    absorption = (
        0.06 * _gauss(w, 687.0, 4.0)    # O2-B
        + 0.12 * _gauss(w, 719.0, 8.0)  # H2O
        + 0.55 * _gauss(w, 761.0, 3.5)  # O2-A
    )
    v = planck * np.exp(-(tau_rayleigh + tau_ozone)) * (1.0 - np.clip(absorption, 0.0, 0.95))
    v /= np.interp(560.0, w, v)
    return w, v


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _write_curve_csv(path: Path, wl: np.ndarray, val: np.ndarray, comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {comment}", "# wavelength_nm,value"]
    lines += [f"{float(w):.6g},{float(v):.6g}" for w, v in zip(wl, val)]
    path.write_text("\n".join(lines) + "\n")


def write_all_surface_csvs(out_dir: Path) -> list[Path]:
    written = []
    for name in sorted(SURFACE_CURVES):
        wl, val = synth_spectrum(name)
        p = out_dir / f"{name}.csv"
        _write_curve_csv(p, wl, val, f"{name} reflectance (analytic approximation), open-cam highway")
        written.append(p)
    return written


def write_solar_csv(path: Path) -> Path:
    wl, val = solar_direct_am15()
    _write_curve_csv(
        path, wl, val,
        "Direct-normal solar SPD, smooth ASTM G-173 AM1.5 approximation, normalized to 1 at 560 nm",
    )
    return path


# ---------------------------------------------------------------------------
# PBRT material block emitters (GPU/wavefront-safe only)
# ---------------------------------------------------------------------------

def diffuse_spd(spd_rel: str) -> list[str]:
    return [f'Material "diffuse" "spectrum reflectance" "{spd_rel}"']


def coateddiffuse_spd(spd_rel: str, roughness: float = 0.15, thickness: float = 0.01) -> list[str]:
    """Clearcoat-over-pigment look for car paint (GPU-supported layered material)."""
    return [
        'Material "coateddiffuse"',
        f'    "spectrum reflectance" "{spd_rel}"',
        f'    "float roughness" [{roughness:g}]',
        f'    "float thickness" [{thickness:g}]',
    ]


def conductor_metal(metal: str = "Al", roughness: float = 0.25) -> list[str]:
    """Rough metal via pbrt built-in spectra (metal-Al ~ galvanized steel stand-in)."""
    return [
        'Material "conductor"',
        f'    "spectrum eta" "metal-{metal}-eta"',
        f'    "spectrum k" "metal-{metal}-k"',
        f'    "float roughness" [{roughness:g}]',
    ]


def dielectric_glass(eta: float = 1.5) -> list[str]:
    return [f'Material "dielectric" "float eta" [{eta:g}]']


def sign_face_textured(texture_name: str) -> list[str]:
    return [f'Material "diffuse" "texture reflectance" "{texture_name}"']


# ---------------------------------------------------------------------------
# Sign legend textures (Pillow)
# ---------------------------------------------------------------------------

def render_sign_face_png(sign_type: str, text: str, out_png: Path,
                         size_px: tuple[int, int] = (512, 512)) -> Path:
    """Render a simple sign face legend. RGB texture — the one non-SPD surface."""
    from PIL import Image, ImageDraw, ImageFont

    w, h = size_px
    if sign_type == "speed_limit":
        bg, fg, border = (255, 255, 255), (0, 0, 0), (0, 0, 0)
        lines = ["SPEED", "LIMIT", text]
    elif sign_type == "exit":
        bg, fg, border = (0, 106, 77), (255, 255, 255), (255, 255, 255)  # highway green
        lines = ["EXIT", text]
    else:
        raise ValueError(f"unknown sign type '{sign_type}'")

    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    m = w // 24
    d.rectangle([m, m, w - m, h - m], outline=border, width=max(2, w // 64))

    def _font(px: int):
        for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"):
            try:
                return ImageFont.truetype(name, px)
            except OSError:
                continue
        return ImageFont.load_default()

    n = len(lines)
    band = (h - 4 * m) / n
    font = _font(int(band * 0.62))
    for i, line in enumerate(lines):
        bbox = d.textbbox((0, 0), line, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        d.text(((w - tw) / 2 - bbox[0], 2 * m + i * band + (band - th) / 2 - bbox[1]),
               line, fill=fg, font=font)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    return out_png


# ---------------------------------------------------------------------------
# CLI: one-time spectra generation
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--write-spectra", action="store_true",
                    help="Write surface reflectance CSVs and the solar SPD CSV.")
    args = ap.parse_args()

    if args.write_spectra:
        repo = args.repo_root.resolve()
        surf = write_all_surface_csvs(repo / "spectra" / "surfaces" / "highway")
        solar = write_solar_csv(repo / "spectra" / "illuminant" / "interpolated" / "solar_direct_am15.csv")
        for p in surf + [solar]:
            print(f"wrote {p.relative_to(repo)}")
    else:
        print("nothing to do (use --write-spectra)")


if __name__ == "__main__":
    main()
