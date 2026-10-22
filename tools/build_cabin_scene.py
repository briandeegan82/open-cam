#!/usr/bin/env python3
"""In-cabin driver-monitoring (DMS) scene at 850/940 nm: procedural driver, NIR LED, night cabin.

Needs the NIR pbrt build (``PBRT_LAMBDA_MAX_NM=1100``, patch 0002; docs/CFA_NIR.md).

* Camera on the steering column, ~0.65 m from the face, looking up at it; 940 nm (or 850 nm)
  Gaussian LED spot light next to the lens, absolute radiometric units (``"bool photometric" false``).
* Driver: ellipsoid head + neck + torso. Skin = measured ColorChecker skin patch to 730 nm,
  extended with the relative NIR shape of ``nir_skin.skin_reflectance`` (water dip at 970 nm).
* Eyes: sclera sphere, iris and pupil discs, and a dielectric cornea cap (eta 1.376,
  radius 7.8 mm) that gives the corneal glint used for gaze tracking. Optional spectacles.
* Corneal glint: the LED's virtual image in the cornea is ~15 um across, far below a pixel, so a
  smooth cornea is almost never hit by a camera sample. ``--cornea-roughness`` (GGX alpha, default
  0.02) spreads it over ~1-3 px so pbrt samples it; the total glint power is unchanged (checked
  against the mirror-sphere result J = rho E R^2 / 4, tests/test_cabin_dms.py). The manifest gives
  each glint's analytic pixel position and camera-plane irradiance.
* Seat/headrest/dashboard/steering wheel use USGS splib07a spectra measured to 1100 nm.
* Night only: no sunlight through the windows.

The manifest records eye, cornea and pupil centres in world space as gaze ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nir_skin  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
WL = np.arange(400.0, 1101.0, 10.0)
HEAD = np.array([0.0, 1.15, 0.0])
HEAD_AXES = (0.075, 0.115, 0.095)
EYE_OFF = (0.032, 0.03, -0.086)
EYE_R, CORNEA_R, CORNEA_ETA = 0.012, 0.0078, 1.376
CORNEA_CAP_COS = 0.55  # cap is z in [-R, -0.55 R] of the cornea sphere
SKIN = {"light": ("02_light_skin", "II"), "dark": ("01_dark_skin", "V")}
#: nominal NIR reflectances (flat): sclera, iris, pupil (dark-pupil; no retinal retroreflection)
EYE_REFL = {"sclera": 0.55, "iris": 0.25, "pupil": 0.02}


def _spd(wl, v) -> str:
    v = np.where(np.abs(v) < 1e-6, 0.0, v)
    return "[ " + " ".join(f"{a:g} {b:.5g}" for a, b in zip(wl, v)) + " ]"


def _usgs(name: str) -> np.ndarray:
    d = np.loadtxt(REPO / "spectra/usgs_nir" / f"{name}.csv", delimiter=",", comments="#")
    return np.clip(np.interp(WL, d[:, 0], d[:, 1]), 0.0, 1.0)


def skin_spectrum(tone: str) -> np.ndarray:
    patch, fitz = SKIN[tone]
    d = np.loadtxt(REPO / "spectra/xrite" / f"{patch}.csv", delimiter=",")
    return nir_skin.extend_measured(WL, d[:, 0], d[:, 1], f_mel=nir_skin.MELANIN[fitz])


def led_spectrum(peak_nm: int) -> tuple[np.ndarray, float]:
    """Peak-normalised SPD on WL and its integral [nm], for absolute radiant intensity."""
    d = np.loadtxt(REPO / "spectra/illuminant/nir" / f"LED_{peak_nm}nm.csv", delimiter=",", comments="#")
    v = np.interp(WL, d[:, 0], d[:, 1])
    return v / v.max(), float(np.trapezoid(np.interp(d[:, 0], d[:, 0], d[:, 1] / d[:, 1].max()), d[:, 0]))


def mat(name: str, refl: np.ndarray) -> str:
    return f'MakeNamedMaterial "{name}" "string type" "diffuse" "spectrum reflectance" {_spd(WL, refl)}'


def lens_slab_transmittance(cos_theta: float, eta: float = 1.5) -> float:
    """Thin-slab transmittance 1 - R_slab, incoherent multiple reflections (pbrt ThinDielectricBxDF)."""
    c = cos_theta
    s2 = max(0.0, 1 - c * c) / eta**2
    if s2 >= 1:
        return 0.0
    ct = math.sqrt(1 - s2)
    rs = ((c - eta * ct) / (c + eta * ct)) ** 2
    rp = ((eta * c - ct) / (eta * c + ct)) ** 2
    r = 0.5 * (rs + rp)
    t = 1 - r
    return 1 - (r + t * t * r / (1 - r * r))


def box(c, s) -> str:
    x, y, z = c
    a, b, d = (v / 2 for v in s)
    p = [(x + i * a, y + j * b, z + k * d) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]
    idx = [0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3]
    pts = " ".join(f"{u:.5g}" for q in p for u in q)
    return f'Shape "trianglemesh" "integer indices" [{" ".join(map(str, idx))}] "point3 P" [{pts}]'


def fresnel_dielectric(cos_i: float, eta: float) -> float:
    """Unpolarised Fresnel reflectance from air onto a dielectric of index ``eta``."""
    sin_t = math.sqrt(max(0.0, 1.0 - cos_i * cos_i)) / eta
    cos_t = math.sqrt(1.0 - sin_t * sin_t)
    rs = ((cos_i - eta * cos_t) / (cos_i + eta * cos_t)) ** 2
    rp = ((eta * cos_i - cos_t) / (eta * cos_i + cos_t)) ** 2
    return 0.5 * (rs + rp)


def project(cam: np.ndarray, look: np.ndarray, p: np.ndarray, fov_deg: float, xres: int, yres: int) -> np.ndarray:
    """Raster (x, y) of world point ``p`` for pbrt's ``LookAt cam look 0 1 0`` perspective camera."""
    fwd = (look - cam) / np.linalg.norm(look - cam)
    right = np.cross([0.0, 1.0, 0.0], fwd)
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    q = p - cam
    w = 2.0 * math.tan(math.radians(fov_deg) / 2) / min(xres, yres)
    z = q @ fwd
    return np.array([xres / 2 + (q @ right) / z / w, yres / 2 - (q @ up) / z / w])


def corneal_glint(
    centre: np.ndarray, axis: np.ndarray, cam: np.ndarray, led: np.ndarray, led_normal: np.ndarray, led_w_sr: float
) -> dict:
    """First-surface (Purkinje I) glint of a small Lambertian LED in a spherical cornea.

    Reflection point from the half vector (iterated); the camera-plane irradiance follows from the
    mirror-sphere intensity J = rho * E * R^2 / 4 (E = LED irradiance at the cornea). ``on_cornea`` is
    False when the reflection point falls outside the corneal cap (it would then be on the sclera).
    """
    p = centre.copy()
    for _ in range(5):
        h = (led - p) / np.linalg.norm(led - p) + (cam - p) / np.linalg.norm(cam - p)
        n = h / np.linalg.norm(h)
        p = centre + CORNEA_R * n
    ds, dc = np.linalg.norm(led - p), np.linalg.norm(cam - p)
    cos_e = max(0.0, float(np.dot(p - led, led_normal)) / ds)
    rho = fresnel_dielectric(float(np.dot(n, (cam - p) / dc)), CORNEA_ETA)
    e_cam = rho * (led_w_sr * cos_e / ds**2) * CORNEA_R**2 / 4 / dc**2
    return {
        "point": p.round(6).tolist(),
        "fresnel": round(rho, 5),
        "on_cornea": bool(np.dot(n, axis) > CORNEA_CAP_COS),
        "camera_irradiance_w_m2": float(f"{e_cam:.4g}"),
    }


def build(args) -> dict:
    out = Path(args.out_dir)
    (out / "out").mkdir(parents=True, exist_ok=True)
    cam = HEAD + np.array([0.0, -args.cam_below_m, -args.cam_dist_m])
    led_pos = cam + np.array([args.led_offset_m, 0.0, 0.0])
    led, led_int = led_spectrum(args.led_nm)
    gaze = np.array([math.radians(args.gaze_yaw_deg), math.radians(args.gaze_pitch_deg)])
    led_normal = (HEAD - led_pos) * np.array([0.0, 1.0, 1.0])  # disc tilted about x towards the head
    led_normal /= np.linalg.norm(led_normal)
    gdir = np.array([math.sin(gaze[0]) * math.cos(gaze[1]), math.sin(gaze[1]), -math.cos(gaze[0]) * math.cos(gaze[1])])
    L = [
        "# Generated by tools/build_cabin_scene.py (in-cabin DMS) -- do not hand-edit.",
        f"LookAt {cam[0]:g} {cam[1]:g} {cam[2]:g}  {HEAD[0]:g} {HEAD[1]:g} {HEAD[2]:g}  0 1 0",
        f'Camera "perspective" "float fov" [{args.fov:g}]',
        f'Sampler "zsobol" "integer pixelsamples" [{args.pixelsamples}]',
        'Integrator "volpath" "integer maxdepth" [8]',
        'PixelFilter "box" "float xradius" [0.5] "float yradius" [0.5]',
        f'Film "spectral" "string filename" ["out/cabin_spectral.exr"] "integer xresolution" [{args.xres}]',
        f'  "integer yresolution" [{args.yres}] "bool savefp16" false "integer nbuckets" [16]',
        '  "float lambdamin" [400] "float lambdamax" [1100]',
        "WorldBegin",
        f"# {args.led_nm} nm LED: {args.led_w_sr:g} W/sr peak-band radiant intensity on axis",
        f"AttributeBegin Translate {led_pos[0]:g} {led_pos[1]:g} {led_pos[2]:g}",
        f"Rotate {math.degrees(math.atan2(HEAD[1] - led_pos[1], HEAD[2] - led_pos[2])):.3f} -1 0 0",
        # Lambertian disc emitter (finite size so the cornea can mirror it: a point light gives no glint)
        f'AreaLightSource "diffuse" "spectrum L" {_spd(WL, led)} "bool photometric" false',
        f'  "float scale" [{args.led_w_sr / (led_int * math.pi * (args.led_radius_mm * 1e-3) ** 2):.6g}]',
        'Material "diffuse" "rgb reflectance" [0 0 0]',
        f'Shape "disk" "float radius" [{args.led_radius_mm * 1e-3:g}]',
        "AttributeEnd",
        mat("skin", skin_spectrum(args.skin)),
        mat("sclera", np.full_like(WL, EYE_REFL["sclera"])),
        mat("iris", np.full_like(WL, EYE_REFL["iris"])),
        mat("pupil", np.full_like(WL, EYE_REFL["pupil"])),
        mat("shirt", _usgs(args.shirt)),
        mat("seat", _usgs("16_hdpe_black")),
        mat("dash", _usgs("23_tar_paper_black")),
        f'MakeNamedMaterial "cornea" "string type" "dielectric" "float eta" [{CORNEA_ETA}]'
        + (
            f' "float roughness" [{args.cornea_roughness:g}] "bool remaproughness" false'
            if args.cornea_roughness > 0
            else ""
        ),
        (
            'MakeNamedMaterial "lens" "string type" "thindielectric" "float eta" [1.5] "bool passthrough" true'
            if args.glasses_model == "thin"
            else 'MakeNamedMaterial "lens" "string type" "dielectric" "float eta" [1.5]'
        ),
        "AttributeBegin",
        'NamedMaterial "skin"',
        f"Translate {HEAD[0]:g} {HEAD[1]:g} {HEAD[2]:g} Scale {HEAD_AXES[0]} {HEAD_AXES[1]} {HEAD_AXES[2]}",
        'Shape "sphere" "float radius" [1]',
        "AttributeEnd",
        'AttributeBegin NamedMaterial "skin"',
        f'Translate 0 {HEAD[1] - 0.2:g} 0.01 Rotate -90 1 0 0 Shape "cylinder" "float radius" [0.05] "float zmin" [0] "float zmax" [0.1]',
        "AttributeEnd",
        'AttributeBegin NamedMaterial "shirt"',
        box((0, HEAD[1] - 0.45, 0.05), (0.42, 0.5, 0.22)),
        "AttributeEnd",
        'AttributeBegin NamedMaterial "seat"',
        box((0, HEAD[1] - 0.35, 0.22), (0.5, 0.9, 0.1)),
        box((0, HEAD[1] + 0.02, 0.17), (0.26, 0.2, 0.08)),
        "AttributeEnd",
        'AttributeBegin NamedMaterial "dash"',
        box((0, cam[1] - 0.12, cam[2] - 0.15), (1.4, 0.06, 0.4)),
        "AttributeEnd",
    ]
    # steering-wheel rim: ring of short cylinders around the column, below the camera
    wc, wr = cam + np.array([0, -0.05, 0.12]), 0.18
    L.append('AttributeBegin NamedMaterial "seat"')
    for k in range(24):
        a0 = 2 * math.pi * k / 24
        if abs(math.sin(a0)) > 0.85 and math.sin(a0) > 0:
            continue  # gap at the top so the camera sees through
        L.append(
            f"AttributeBegin Translate {wc[0] + wr * math.cos(a0):.4f} {wc[1] + wr * math.sin(a0):.4f} {wc[2]:.4f} "
            f"Rotate {math.degrees(a0) + 90:.2f} 0 0 1 Rotate 90 0 1 0 "
            f'Shape "cylinder" "float radius" [0.015] "float zmin" [-0.025] "float zmax" [0.025] AttributeEnd'
        )
    L.append("AttributeEnd")
    eyes = []
    for sx in (-1, 1):
        c = HEAD + np.array([sx * EYE_OFF[0], EYE_OFF[1], EYE_OFF[2]])
        front = c + gdir * EYE_R
        cornea_c = c + gdir * (EYE_R - CORNEA_R + 0.0025)
        yaw, pitch = math.degrees(gaze[0]), math.degrees(gaze[1])
        L += [
            f"AttributeBegin Translate {c[0]:.5f} {c[1]:.5f} {c[2]:.5f}",
            f'NamedMaterial "sclera" Shape "sphere" "float radius" [{EYE_R}]',
            f"Rotate {-yaw:.3f} 0 1 0 Rotate {pitch:.3f} 1 0 0",
            f'AttributeBegin Translate 0 0 {-(EYE_R + 1e-4):.5f} NamedMaterial "pupil" Shape "disk" "float radius" [{args.pupil_mm / 2e3:g}] AttributeEnd',
            f'AttributeBegin Translate 0 0 {-(EYE_R + 5e-5):.5f} NamedMaterial "iris" Shape "disk" "float radius" [0.006] "float innerradius" [{args.pupil_mm / 2e3:g}] AttributeEnd',
            f'AttributeBegin Translate 0 0 {-(EYE_R - CORNEA_R + 0.0025):.5f} NamedMaterial "cornea"',
            f'  Shape "sphere" "float radius" [{CORNEA_R}] "float zmin" [{-CORNEA_R:g}] "float zmax" [{-CORNEA_R * CORNEA_CAP_COS:g}]',
            "AttributeEnd",
            "AttributeEnd",
        ]
        glint = corneal_glint(cornea_c, gdir, cam, led_pos, led_normal, args.led_w_sr)
        glint["pixel_xy"] = (
            project(cam, HEAD, np.array(glint["point"]), args.fov, args.xres, args.yres).round(2).tolist()
        )
        eyes.append(
            {
                "side": "right" if sx < 0 else "left",
                "eye_centre": c.round(5).tolist(),
                "cornea_centre": cornea_c.round(5).tolist(),
                "pupil_centre": front.round(5).tolist(),
                "pupil_pixel_xy": project(cam, HEAD, front, args.fov, args.xres, args.yres).round(2).tolist(),
                "glint": glint,
            }
        )
    if args.glasses:
        L.append('AttributeBegin NamedMaterial "lens"')
        for e in eyes:
            c = np.array(e["eye_centre"]) + np.array([0, 0, -0.02])
            if args.glasses_model == "thin":  # one surface stands for both faces of the 2 mm lens
                x, y, z = c
                pts = " ".join(
                    f"{u:.5g}"
                    for i, j in ((-1, -1), (1, -1), (1, 1), (-1, 1))
                    for u in (x + i * 0.025, y + j * 0.0175, z)
                )
                L.append(f'Shape "trianglemesh" "integer indices" [0 1 2 0 2 3] "point3 P" [{pts}]')
            else:
                L.append(box(tuple(c), (0.05, 0.035, 0.002)))
        L.append("AttributeEnd")
    (out / "cabin.pbrt").write_text("\n".join(L) + "\n")
    manifest = {
        "scene": "cabin_dms",
        "requires": "NIR pbrt build (PBRT_LAMBDA_MAX_NM=1100)",
        "camera": {"position": cam.round(4).tolist(), "look_at": HEAD.tolist(), "fov_deg": args.fov},
        "led": {
            "peak_nm": args.led_nm,
            "position": led_pos.round(4).tolist(),
            "radiant_intensity_w_sr": args.led_w_sr,
            "emitter": "Lambertian disc",
            "radius_mm": args.led_radius_mm,
            "photometric": False,
        },
        "driver": {
            "skin": args.skin,
            "skin_940nm_reflectance": round(float(np.interp(940, WL, skin_spectrum(args.skin))), 3),
            "gaze_yaw_deg": args.gaze_yaw_deg,
            "gaze_pitch_deg": args.gaze_pitch_deg,
            "pupil_mm": args.pupil_mm,
            "glasses": args.glasses,
            "glasses_model": args.glasses_model if args.glasses else None,
            "lens_transmittance_normal": round(lens_slab_transmittance(1.0), 4) if args.glasses else None,
            "cornea_roughness_alpha": args.cornea_roughness,
            "eyes": eyes,
        },
        "nominal": ["head/eye geometry", "sclera/iris/pupil reflectance", "LED intensity", "skin NIR shape (model)"],
        "not_modelled": [
            "sunlight",
            "retinal retroreflection (bright pupil)",
            "eyelids/brows/hair",
            "skin subsurface blur",
        ],
    }
    (out / "cabin_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, default=REPO / "scenes/cabin_dms")
    ap.add_argument("--xres", type=int, default=640)
    ap.add_argument("--yres", type=int, default=480)
    ap.add_argument("--pixelsamples", type=int, default=256)
    ap.add_argument("--fov", type=float, default=40.0)
    ap.add_argument("--cam-dist-m", type=float, default=0.65)
    ap.add_argument("--cam-below-m", type=float, default=0.18)
    ap.add_argument("--led-nm", type=int, choices=(850, 940), default=940)
    ap.add_argument("--led-w-sr", type=float, default=0.5, help="Radiant intensity [W/sr], nominal.")
    ap.add_argument("--led-offset-m", type=float, default=0.04, help="LED offset from the lens (+x).")
    ap.add_argument("--led-radius-mm", type=float, default=2.5, help="Emitting disc radius (Lambertian).")
    ap.add_argument("--skin", choices=sorted(SKIN), default="light")
    ap.add_argument("--shirt", default="11_cotton_white", help="USGS splib07a chart material name.")
    ap.add_argument("--gaze-yaw-deg", type=float, default=0.0)
    ap.add_argument("--gaze-pitch-deg", type=float, default=0.0)
    ap.add_argument("--pupil-mm", type=float, default=6.0, help="Pupil diameter (night: ~6 mm).")
    ap.add_argument("--glasses", action="store_true", help="Flat n = 1.5 spectacle lenses, 20 mm in front of the eyes.")
    ap.add_argument(
        "--glasses-model",
        choices=("thin", "slab"),
        default="thin",
        help="thin: one thin-dielectric surface that light, including shadow rays, passes with the "
        "two-face Fresnel transmittance (pbrt patch 0017); slab: the old 2 mm dielectric box, which "
        "blocks shadow rays (eyes dark and noisy).",
    )
    ap.add_argument(
        "--cornea-roughness",
        type=float,
        default=0.02,
        help="GGX alpha of the cornea; spreads the sub-pixel LED glint over ~1-3 px so it is sampled "
        "(glint power unchanged). 0 = smooth (glint almost never sampled).",
    )
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    m = build(args)
    print(f"wrote {args.out_dir}/cabin.pbrt; skin {args.skin} R(940) = {m['driver']['skin_940nm_reflectance']}")


if __name__ == "__main__":
    main()
