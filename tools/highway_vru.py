"""Vulnerable road users for the highway scene: Euro NCAP-style AEB pedestrian / cyclist targets.

Opt-in (``--vru pedestrian|child|cyclist|motorcyclist``; ``--vru-bike`` puts the rider on an imported
bicycle, motorbike, moped or scooter from ``tools/two_wheelers.py``). The targets are procedural articulated meshes built to
nominal Euro NCAP / ISO 19206 test-target sizes (adult 1.80 m, child 1.154 m, bicycle ~1.8 m long
with 0.70 m wheels); the standards are paywalled, so the sizes here are nominal and not checked
against the standard text. Placement follows the AEB VRU nearside scenarios: the target stands at
``--vru-distance-m`` ahead of the camera, ``--vru-offset-m`` from the right edge of the carriageway
(negative = into the lane), optionally between two parked cars on the hard shoulder
(``--vru-obstruction parked-cars``, as in the Euro NCAP obstructed-child test). Without ``--exposure-s``
the scene is static (posed at ``--vru-gait-phase``). With it the target moves at ``--vru-speed-kmh`` during
the exposure (rigid motion blur) and the procedural bicycle's wheels roll; the limbs keep their pose.

Spectra (all diffuse, no subsurface scattering):

* skin: measured ColorChecker "light skin" reflectance with a melanin optical-depth tilt
  (``iqlab.skin.melanin_reflectance``, ``--vru-skin-tau``). The ColorChecker data end at 730 nm and are
  held constant beyond, so skin NIR reflectance (which really rises to a ~800 nm plateau) is low.
* clothing: measured USGS splib07a fabrics/plastics from ``spectra/usgs_nir`` (350-1100 nm, public
  domain); ``--vru-clothing ncap`` = black top + blue trousers, as the Euro NCAP targets are dressed.
* bicycle: car-paint frame, rubber tyres, aluminium rims.

The manifest records the target's world-space bounding box and anchor point, i.e. 3-D ground truth.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import two_wheelers as tw

REPO = Path(__file__).resolve().parents[1]
USGS = REPO / "spectra" / "usgs_nir"

HEIGHT_M = {"pedestrian": 1.80, "child": 1.154, "cyclist": 1.80, "motorcyclist": 1.80, "escooter": 1.80}
RIDERS = ("cyclist", "motorcyclist", "escooter")
# Nominal target speeds [km/h] during an exposure (typical Euro NCAP VRU test speeds; not checked
# against the protocol text).
NOMINAL_SPEED_KMH = {"pedestrian": 5.0, "child": 5.0, "cyclist": 15.0, "motorcyclist": 30.0, "escooter": 20.0}
MAX_WHEEL_ROTATION_DEG = 150.0  # pbrt interpolates the rotation between shutter open and close
DEFAULT_MOTO = "moto_80s"
DEFAULT_KICK = "kickscooter_electric"
G = 9.81
WHEEL_D, WHEELBASE = 0.70, 1.05
CLOTHING = {
    # preset: (top, trousers) USGS chart files
    "ncap": ("10_nylon_black_coated", "09_nylon_blue"),
    "dark": ("10_nylon_black_coated", "16_hdpe_black"),
    "light": ("11_cotton_white", "09_nylon_blue"),
    "red": ("07_nylon_red", "09_nylon_blue"),
    "yellow": ("14_hdpe_yellow", "16_hdpe_black"),
}


def add_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("vulnerable road users (tools/highway_vru.py)")
    g.add_argument("--vru", choices=("none", *HEIGHT_M), default="none", help="AEB target (opt-in).")
    g.add_argument("--vru-distance-m", type=float, default=30.0, help="Target distance ahead of the camera [m].")
    g.add_argument(
        "--vru-offset-m",
        type=float,
        default=0.4,
        help="Target centre from the right carriageway edge [m]; negative = into the lane.",
    )
    g.add_argument(
        "--vru-motion",
        choices=("crossing", "along"),
        default="crossing",
        help="crossing: moving right-to-left across the road; along: moving away in the road direction.",
    )
    g.add_argument("--vru-gait-phase", type=float, default=0.25, help="Stride / pedal phase in [0, 1).")
    g.add_argument(
        "--vru-speed-kmh",
        type=float,
        default=None,
        help="Target speed during the exposure (with --exposure-s; motion blur, rolling wheels). Default: "
        "nominal 5 km/h walking, 15 km/h cycling, 30 km/h motorcycling; 0 = static.",
    )
    g.add_argument(
        "--vru-bike",
        choices=sorted(tw.RIDER),
        default=None,
        help="Imported Objaverse CC-BY model under the rider: a bicycle for --vru cyclist (default: procedural "
        f"bicycle), a motorbike, moped or scooter for --vru motorcyclist (default: {DEFAULT_MOTO}).",
    )
    g.add_argument(
        "--vru-turn-radius-m",
        type=float,
        default=None,
        help="Riders: steady turn of this radius (+ = to the rider's left, - = right). Sets the lean "
        "atan(v^2 / (g R)) at the target speed and the kinematic steer atan(wheelbase / R) unless "
        "--vru-lean-deg / --vru-steer-deg are given. The path stays straight during the exposure.",
    )
    g.add_argument(
        "--vru-lean-deg",
        type=float,
        default=None,
        help="Riders: roll of bike + rider about the tyre contact line (+ = left).",
    )
    g.add_argument(
        "--vru-steer-deg",
        type=float,
        default=None,
        help="Riders: handlebar angle about the steering axis (+ = left). Procedural bicycle only; imported "
        "models are single meshes and keep straight bars.",
    )
    g.add_argument("--vru-obstruction", choices=("none", "parked-cars"), default="none")
    g.add_argument("--vru-clothing", choices=sorted(CLOTHING), default="ncap")
    g.add_argument("--vru-skin-tau", type=float, default=0.35, help="Melanin optical depth (iqlab.skin).")


def enabled(args: argparse.Namespace) -> bool:
    return args.vru != "none"


def check_args(args: argparse.Namespace) -> None:
    args.vru_bike = getattr(args, "vru_bike", None)
    if args.vru == "motorcyclist" and args.vru_bike is None:
        args.vru_bike = DEFAULT_MOTO
    if args.vru == "escooter" and args.vru_bike is None:
        args.vru_bike = DEFAULT_KICK
    if args.vru_bike is not None:
        if args.vru not in RIDERS:
            raise SystemExit("--vru-bike needs --vru cyclist, motorcyclist or escooter")
        kind = {"bicycle": "cyclist", "kickscooter": "escooter"}.get(tw.MODELS[args.vru_bike][0], "motorcyclist")
        if kind != args.vru:
            raise SystemExit(
                "--vru cyclist takes a bicycle model; --vru escooter a kick e-scooter; --vru motorcyclist a "
                "motorbike, moped or scooter"
            )
    for k in ("vru_turn_radius_m", "vru_lean_deg", "vru_steer_deg"):
        if getattr(args, k, None) is not None and args.vru not in RIDERS:
            raise SystemExit(f"--{k.replace('_', '-')} needs a rider (--vru cyclist, motorcyclist or escooter)")
    r = getattr(args, "vru_turn_radius_m", None)
    if r is not None and abs(r) < 2.0:
        raise SystemExit("--vru-turn-radius-m: |R| must be >= 2 m")
    if not enabled(args):
        return
    if args.vru_distance_m <= 2.0:
        raise SystemExit("--vru-distance-m must be > 2 m")
    if not 0.0 <= args.vru_gait_phase < 1.0:
        raise SystemExit("--vru-gait-phase must be in [0, 1)")
    if getattr(args, "vru_speed_kmh", None) is not None and args.vru_speed_kmh < 0:
        raise SystemExit("--vru-speed-kmh must be >= 0")


def speed_kmh(args: argparse.Namespace) -> float:
    v = getattr(args, "vru_speed_kmh", None)
    return NOMINAL_SPEED_KMH[args.vru] if v is None else v


def turn(args: argparse.Namespace, wheelbase_m: float) -> dict:
    """Lean and steer [deg] (+ = left) from --vru-turn-radius-m and the explicit overrides.

    Steady turn of a point-mass rider: tan(lean) = v^2 / (g R); kinematic steer tan(delta) = wheelbase / R
    (no trail, caster or tyre slip)."""
    r = getattr(args, "vru_turn_radius_m", None)
    v = speed_kmh(args) / 3.6
    lean = math.degrees(math.atan(v * v / (G * r))) if r else 0.0
    steer = math.degrees(math.atan(wheelbase_m / r)) if r else 0.0
    if getattr(args, "vru_lean_deg", None) is not None:
        lean = args.vru_lean_deg
    if getattr(args, "vru_steer_deg", None) is not None:
        steer = args.vru_steer_deg
    return {"lean_deg": lean, "steer_deg": steer, "turn_radius_m": r, "wheelbase_m": wheelbase_m}


def _rotate_parts(src: Parts, dst: Parts, origin, axis, deg: float) -> None:
    """Merge ``src`` into ``dst`` rotated by ``deg`` about the line through ``origin`` along ``axis``."""
    rot = _axis_rotation(axis, deg)
    o = np.asarray(origin, float)
    for mat, items in src.parts.items():
        for p, t, n in items:
            dst.add(mat, (p - o) @ rot.T + o, t, n @ rot.T)


def _axis_rotation(axis, deg: float) -> np.ndarray:
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    a = math.radians(deg)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


# ---------------------------------------------------------------- spectra


def usgs_reflectance(name: str, wl: np.ndarray) -> np.ndarray:
    d = np.loadtxt(USGS / f"{name}.csv", delimiter=",")
    return np.clip(np.interp(wl, d[:, 0], d[:, 1]), 0.0, 1.0)


def write_spectra(args, spd: Path, wl: np.ndarray, write_spd, reflectance) -> dict[str, str]:
    from iqlab import skin

    xr = np.loadtxt(REPO / "spectra" / "xrite" / "02_light_skin.csv", delimiter=",")
    top, trousers = CLOTHING[args.vru_clothing]
    curves = {
        "vru_skin": skin.melanin_reflectance(wl, np.interp(wl, xr[:, 0], xr[:, 1]), args.vru_skin_tau),
        "vru_hair": np.full_like(wl, 0.04),
        "vru_top": usgs_reflectance(top, wl),
        "vru_trousers": usgs_reflectance(trousers, wl),
        "vru_shoes": usgs_reflectance("16_hdpe_black", wl),
        "vru_frame": reflectance("carpaint_red", wl),
        "vru_rim": np.full_like(wl, 0.6),
    }
    for k, v in curves.items():
        write_spd(spd / f"{k}.spd", wl, v)
    return {
        "skin": f"ColorChecker 02 light skin, melanin tau {args.vru_skin_tau:g} (held constant > 730 nm)",
        "top": f"USGS splib07a {top}",
        "trousers": f"USGS splib07a {trousers}",
        "shoes": "USGS splib07a 16_hdpe_black",
    }


def material_lines() -> list[str]:
    out = []
    for m in ("skin", "hair", "top", "trousers", "shoes"):
        out.append(f'MakeNamedMaterial "vru:{m}" "string type" "diffuse" "spectrum reflectance" "spd/vru_{m}.spd"')
    out += [
        'MakeNamedMaterial "vru:frame" "string type" "coateddiffuse" "spectrum reflectance" "spd/vru_frame.spd"'
        ' "float roughness" [0.05]',
        'MakeNamedMaterial "vru:tyre" "string type" "diffuse" "spectrum reflectance" "spd/rubber.spd"',
        'MakeNamedMaterial "vru:rim" "string type" "conductor" "spectrum eta" "metal-Al-eta"'
        ' "spectrum k" "metal-Al-k" "float roughness" [0.15]',
    ]
    return out


# ---------------------------------------------------------------- geometry (local frame: +y up, +z forward)


class Parts:
    """Triangle meshes grouped by material."""

    def __init__(self) -> None:
        self.parts: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}

    def add(self, mat: str, p: np.ndarray, t: np.ndarray, n: np.ndarray) -> None:
        self.parts.setdefault(mat, []).append((p, t, n))

    def plate(self, mat: str, c, u, v, w) -> None:
        """Box centred on ``c`` with half-extent vectors ``u``, ``v``, ``w`` (mutually orthogonal)."""
        c, axes = np.asarray(c, float), [np.asarray(a, float) for a in (u, v, w)]
        pts, nrm, tri = [], [], []
        for k in range(3):
            a, b, n = axes[k], axes[(k + 1) % 3], axes[(k + 2) % 3]
            for sgn in (-1.0, 1.0):
                f = c + sgn * n
                q = (
                    [f - a - b, f + a - b, f + a + b, f - a + b]
                    if sgn > 0
                    else [f - a - b, f - a + b, f + a + b, f + a - b]
                )
                i = len(pts)
                pts += q
                nrm += [sgn * n / np.linalg.norm(n)] * 4
                tri += [(i, i + 1, i + 2), (i, i + 2, i + 3)]
        self.add(mat, np.array(pts), np.array(tri), np.array(nrm))

    def ellipsoid(self, mat: str, c, r, n_u: int = 18, n_v: int = 10) -> None:
        u = np.linspace(0.0, 2 * np.pi, n_u, endpoint=False)
        v = np.linspace(0.0, np.pi, n_v + 1)
        vv, uu = np.meshgrid(v, u, indexing="ij")
        d = np.stack([np.sin(vv) * np.cos(uu), np.cos(vv), np.sin(vv) * np.sin(uu)], -1).reshape(-1, 3)
        r = np.asarray(r, float)
        p = np.asarray(c, float) + d * r
        nrm = d / r
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        t = []
        for i in range(n_v):
            for j in range(n_u):
                a, b = i * n_u + j, i * n_u + (j + 1) % n_u
                t += [(a, a + n_u, b), (b, a + n_u, b + n_u)]
        self.add(mat, p, np.array(t), nrm)

    def tube(self, mat: str, p0, p1, r0: float, r1: float | None = None, n_u: int = 12, caps: bool = True) -> None:
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        r1 = r0 if r1 is None else r1
        axis = p1 - p0
        ln = float(np.linalg.norm(axis))
        if ln < 1e-9:
            return
        w = axis / ln
        a = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(w, a)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(w, e1)
        ang = np.linspace(0.0, 2 * np.pi, n_u, endpoint=False)
        ring = np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2
        p = np.concatenate([p0 + r0 * ring, p1 + r1 * ring])
        t = []
        for j in range(n_u):
            k = (j + 1) % n_u
            t += [(j, k, n_u + j), (k, n_u + k, n_u + j)]
        self.add(mat, p, np.array(t), np.concatenate([ring, ring]))
        if caps:
            self.ellipsoid(mat, p0, (r0, r0, r0), 10, 6)
            self.ellipsoid(mat, p1, (r1, r1, r1), 10, 6)

    def torus(self, mat: str, c, radius: float, r: float, n_u: int = 48, n_v: int = 8) -> None:
        """Wheel in the local y-z plane (axle along x)."""
        u = np.linspace(0.0, 2 * np.pi, n_u, endpoint=False)
        v = np.linspace(0.0, 2 * np.pi, n_v, endpoint=False)
        uu, vv = np.meshgrid(u, v, indexing="ij")
        radial = np.stack([np.zeros_like(uu), np.cos(uu), np.sin(uu)], -1)
        nrm = np.cos(vv)[..., None] * radial + np.sin(vv)[..., None] * np.array([1.0, 0.0, 0.0])
        p = np.asarray(c, float) + radius * radial + r * nrm
        t = []
        for i in range(n_u):
            for j in range(n_v):
                a, b = i * n_v + j, i * n_v + (j + 1) % n_v
                a2, b2 = ((i + 1) % n_u) * n_v + j, ((i + 1) % n_u) * n_v + (j + 1) % n_v
                t += [(a, a2, b), (b, a2, b2)]
        self.add(mat, p.reshape(-1, 3), np.array(t), nrm.reshape(-1, 3))

    def stand(self, height: float) -> Parts:
        """Put the lowest point on y = 0 and scale uniformly so the top is at ``height``."""
        lo, hi = self.bounds()
        k = height / float(hi[1] - lo[1])
        for meshes in self.parts.values():
            for i, (p, t, n) in enumerate(meshes):
                meshes[i] = ((p - [0.0, lo[1], 0.0]) * k, t, n)
        return self

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        allp = np.concatenate([p for v in self.parts.values() for p, _, _ in v])
        return allp.min(0), allp.max(0)

    def lines(self) -> list[str]:
        out = []
        for mat, meshes in self.parts.items():
            p = np.concatenate([m[0] for m in meshes])
            n = np.concatenate([m[2] for m in meshes])
            off = np.cumsum([0] + [len(m[0]) for m in meshes[:-1]])
            t = np.concatenate([m[1] + o for m, o in zip(meshes, off)])
            out += [
                f'NamedMaterial "{mat}"',
                'Shape "trianglemesh"',
                '    "point3 P" [ ' + " ".join(f"{x:.5g}" for x in p.ravel()) + " ]",
                '    "normal N" [ ' + " ".join(f"{x:.4g}" for x in n.ravel()) + " ]",
                '    "integer indices" [ ' + " ".join(str(int(i)) for i in t.ravel()) + " ]",
            ]
        return out


def _limb(parts: Parts, mat: str, root, angles_deg, lengths, radii) -> np.ndarray:
    """Chain of tubes hanging down from ``root``, each rotated about the local x axis (sagittal plane)."""
    p = np.asarray(root, float)
    a = 0.0
    for ang, ln, (r0, r1) in zip(angles_deg, lengths, radii):
        a += math.radians(ang)
        q = p + ln * np.array([0.0, -math.cos(a), math.sin(a)])
        parts.tube(mat, p, q, r0, r1)
        p = q
    return p


def pedestrian(height: float, phase: float, child: bool = False) -> Parts:
    """Walking adult (or child) facing +z, feet on y = 0, mid-stride at ``phase``."""
    s = height / 1.80
    hs = 1.25 if child else 1.0  # children's heads are relatively larger
    swing = math.sin(2 * math.pi * phase)
    knee = 8.0 + 22.0 * max(0.0, -swing)
    parts = Parts()
    hip_y, sh_y = 0.95 * s, 1.47 * s
    if child:  # shorter legs relative to stature
        hip_y, sh_y = 0.90 * s, 1.43 * s
    leg_len = hip_y - 0.07 * s
    thigh, shank = 0.52 * leg_len, 0.48 * leg_len
    for side, sgn in ((-1.0, 1.0), (1.0, -1.0)):
        th = sgn * 25.0 * swing
        kn = knee if sgn * swing < 0 else 6.0
        hip = (side * 0.09 * s, hip_y, 0.0)
        ankle = _limb(
            parts, "vru:trousers", hip, (th, -kn), (thigh, shank), ((0.075 * s, 0.06 * s), (0.055 * s, 0.045 * s))
        )
        parts.ellipsoid("vru:shoes", ankle + np.array([0.0, -0.035 * s, 0.06 * s]), (0.05 * s, 0.04 * s, 0.13 * s))
        sh = (side * 0.19 * s, sh_y - 0.04 * s, 0.0)
        wrist = _limb(
            parts,
            "vru:top",
            sh,
            (-sgn * 20.0 * swing, 15.0),
            (0.30 * s, 0.26 * s),
            ((0.05 * s, 0.045 * s), (0.04 * s, 0.035 * s)),
        )
        parts.ellipsoid("vru:skin", wrist + np.array([0.0, -0.06 * s, 0.0]), (0.03 * s, 0.08 * s, 0.045 * s))
    # torso: pelvis + chest
    parts.ellipsoid("vru:trousers", (0.0, hip_y + 0.02 * s, 0.0), (0.17 * s, 0.11 * s, 0.11 * s))
    parts.tube("vru:top", (0.0, hip_y + 0.08 * s, 0.0), (0.0, sh_y - 0.10 * s, 0.0), 0.15 * s, 0.17 * s)
    parts.ellipsoid("vru:top", (0.0, sh_y - 0.08 * s, 0.0), (0.21 * s, 0.09 * s, 0.11 * s))
    neck_top = sh_y + 0.07 * s
    parts.tube("vru:skin", (0.0, sh_y - 0.02 * s, 0.0), (0.0, neck_top, 0.0), 0.05 * s)
    head_c = np.array([0.0, neck_top + 0.105 * s * hs, 0.01 * s])
    parts.ellipsoid("vru:skin", head_c, (0.075 * s * hs, 0.11 * s * hs, 0.095 * s * hs))
    parts.ellipsoid(
        "vru:hair",
        head_c + np.array([0.0, 0.025 * s * hs, -0.012 * s]),
        (0.079 * s * hs, 0.093 * s * hs, 0.093 * s * hs),
    )
    return parts.stand(height)


# 16 CFR 1512.16 bicycle reflectors: name -> (colour, REFLEX_CIL_MCD table, face width, face height) [m].
REFLECTORS = {
    "rear": ("red", "cpsc1512", 0.06, 0.04),
    "front": ("white", "cpsc1512", 0.06, 0.04),
    "pedal": ("amber", "cpsc1512_pedal", 0.05, 0.015),
    "spoke": ("amber", "cpsc1512", 0.07, 0.025),
}


def reflector_material_lines(spd_dir: Path, out_dir: Path, wl: np.ndarray) -> list[str]:
    """Retroreflective ``vru:reflex_<name>`` materials for the procedural bicycle's REFLECTORS."""
    import highway_night as night
    import highway_taillamps as taillamps

    lens = taillamps.lens_reflectances(wl)
    out = []
    for name, (colour, table, w, h) in REFLECTORS.items():
        r = lens["clear" if colour == "white" else colour]
        night._write_spd(spd_dir / f"vru_reflex_{colour}.spd", wl, r)
        cil = night.REFLEX_CIL_MCD[table][colour]
        p = night.REFLEX_PEDAL if name == "pedal" else night.REFLEX_BICYCLE
        out += night.reflex_material_lines(
            f"vru:reflex_{name}", f"vru_reflex_{colour}", cil, w * h, spd_dir, out_dir, wl, p, spectrum=r
        )
    return out


def _bicycle_reflectors(parts: Parts, rear, front, rw: float, seat, bar, pedals, wheels=None) -> None:
    """Rear (red), front (white), pedal and spoke (amber) reflectors; faces sized as REFLECTORS.

    With ``cyclist``'s ``wheels`` list the spoke reflectors go on the wheel parts (about their axles), so they roll."""

    def half(name: str) -> tuple[float, float]:
        return REFLECTORS[name][2] / 2, REFLECTORS[name][3] / 2

    w, h = half("rear")
    parts.plate("vru:reflex_rear", seat + [0.0, -0.17, -0.07], (w, 0, 0), (0, h, 0), (0, 0, 0.006))
    w, h = half("front")
    parts.plate("vru:reflex_front", bar + [0.0, -0.06, 0.03], (w, 0, 0), (0, h, 0), (0, 0, 0.006))
    w, h = half("pedal")
    for side, p in zip((-1.0, 1.0), pedals):
        parts.plate("vru:reflex_pedal", p + [side * 0.03, -0.02, 0.0], (w, 0, 0), (0, h, 0), (0, 0, 0.045))
    w, h = half("spoke")
    hubs = [(w_parts, np.zeros(3)) for _, w_parts in wheels] if wheels else [(parts, rear), (parts, front)]
    for p_, c in hubs:
        d = np.array([0.0, math.cos(math.pi / 6), math.sin(math.pi / 6)])
        e = np.array([0.0, -d[2], d[1]])
        p_.plate("vru:reflex_spoke", c + d * (rw - 0.12), d * w, e * h, (0.006, 0, 0))


def cyclist(
    height: float, phase: float, wheels: list | None = None, reflectors: bool = False, steer_deg: float = 0.0
) -> Parts:
    """Bicycle (axles along x, travelling +z) with a seated rider; total height ~``height``.

    With a ``wheels`` list the wheels are built about their own axle centres and appended to it as
    ``(centre, Parts)`` (for rolling them), instead of being part of the returned mesh.
    ``reflectors``: add the 16 CFR 1512 reflector set (materials from reflector_material_lines).
    ``steer_deg``: fork, front wheel and handlebar turn about the head-tube axis (+ = left); wheel entries
    are ``(centre, Parts, steer)`` with ``steer = (deg, axis)`` or None.
    """
    s = height / 1.80
    parts = Parts()
    fe = Parts() if steer_deg else parts  # front end (steered); built in place when straight
    rw = WHEEL_D / 2
    rear, front = np.array([0.0, rw, -WHEELBASE / 2]), np.array([0.0, rw, WHEELBASE / 2])
    head = np.array([0.0, 0.98, 0.38])
    steer_axis = head - front
    for centre in (rear, front):
        tgt = fe if centre is front else parts
        w, c = (Parts(), np.zeros(3)) if wheels is not None else (tgt, centre)
        w.torus("vru:tyre", c, rw - 0.016, 0.016)
        w.torus("vru:rim", c, rw - 0.04, 0.008, n_v=6)
        for k in range(6):  # spoke stand-ins
            a = math.pi * k / 6
            d = np.array([0.0, math.cos(a), math.sin(a)]) * (rw - 0.045)
            w.tube("vru:rim", c - d, c + d, 0.003, caps=False, n_u=4)
        if wheels is not None:
            wheels.append((centre, w, (steer_deg, steer_axis) if centre is front and steer_deg else None))
    bb = np.array([0.0, 0.28, -0.05])
    seat = np.array([0.0, 0.88, -0.25])
    for a, b in (
        (rear, bb),
        (rear, seat - [0, 0.08, 0]),
        (bb, seat - [0, 0.08, 0]),
        (bb, head),
        (seat - [0, 0.08, 0], head),
    ):
        parts.tube("vru:frame", a, b, 0.017)
    fe.tube("vru:frame", head, front, 0.014)
    fe.tube("vru:frame", head, head + [0.0, 0.08, -0.02], 0.014)
    bar = head + [0.0, 0.08, -0.02]
    fe.tube("vru:frame", bar - [0.24, 0, 0], bar + [0.24, 0, 0], 0.012)
    if steer_deg:
        _rotate_parts(fe, parts, head, steer_axis, steer_deg)
    steer_rot = _axis_rotation(steer_axis, steer_deg)
    parts.ellipsoid("vru:tyre", seat, (0.07, 0.025, 0.13))
    # rider
    hip = seat + [0.0, 0.07, 0.0]
    a = 2 * math.pi * phase
    pedals = [
        bb + np.array([side * 0.11, -0.17 * math.cos(a + ph), 0.17 * math.sin(a + ph)])
        for side, ph in ((-1.0, 0.0), (1.0, math.pi))
    ]
    grips = [head + steer_rot @ (bar + [side * 0.22, 0.0, 0.0] - head) for side in (-1.0, 1.0)]
    _seated_rider(parts, s, hip, pedals, grips, 40.0)
    if reflectors:
        _bicycle_reflectors(parts, rear, front, rw, seat, bar, pedals, wheels)
    return parts


def _seated_rider(parts: Parts, s: float, hip, feet, grips, lean_deg: float, helmet: bool = False) -> None:
    """Rider seated at ``hip``: legs to ``feet`` (pedals or pegs), hands on ``grips`` (left, right)."""
    for side, pedal in zip((-1.0, 1.0), feet):
        h = hip + [side * 0.1 * s, 0, 0]
        d = pedal - h
        L = float(np.linalg.norm(d))
        th, sh = 0.47 * s, 0.45 * s
        mid = (h + pedal) / 2
        bend = math.sqrt(max(0.0, (th + sh) ** 2 / 4 - (L / 2) ** 2 + 1e-6))
        perp = np.cross(d / L, [1.0, 0.0, 0.0])
        perp *= np.sign(perp[2]) if perp[2] != 0 else 1.0
        knee = mid + bend * perp
        parts.tube("vru:trousers", h, knee, 0.07 * s, 0.055 * s)
        parts.tube("vru:trousers", knee, pedal + [0, 0.04, 0], 0.05 * s, 0.04 * s)
        parts.ellipsoid("vru:shoes", pedal + [0, 0.03, 0.04], (0.045 * s, 0.04 * s, 0.12 * s))
    lean = math.radians(lean_deg)
    torso_len = 0.55 * s
    shoulder = hip + torso_len * np.array([0.0, math.cos(lean), math.sin(lean)])
    parts.tube("vru:top", hip + [0, 0.05, 0], shoulder, 0.15 * s, 0.17 * s)
    for side, grip in zip((-1.0, 1.0), grips):
        sh_p = shoulder + [side * 0.18 * s, -0.03, 0.0]
        elbow = (sh_p + grip) / 2 + [0.0, 0.04, -0.03]
        parts.tube("vru:top", sh_p, elbow, 0.05 * s, 0.045 * s)
        parts.tube("vru:top", elbow, grip, 0.04 * s, 0.035 * s)
        parts.ellipsoid("vru:skin", grip, (0.035, 0.035, 0.05))
    neck = shoulder + [0.0, 0.08 * s, 0.02]
    parts.tube("vru:skin", shoulder, neck, 0.05 * s)
    head_c = neck + np.array([0.0, 0.10 * s, 0.05 * s])
    parts.ellipsoid("vru:skin", head_c, (0.075 * s, 0.11 * s, 0.095 * s))
    if helmet:  # full-face motorcycle helmet, painted shell
        parts.ellipsoid("vru:frame", head_c + [0.0, 0.01 * s, 0.0], (0.13 * s, 0.145 * s, 0.15 * s))
    else:
        parts.ellipsoid("vru:hair", head_c + [0.0, 0.03 * s, -0.01], (0.085 * s, 0.09 * s, 0.11 * s))  # helmet-ish


def imported_rider(aid: str, phase: float) -> Parts:
    """1.80 m rider posed on imported two-wheeler ``aid`` from its ``two_wheelers.RIDER`` contact points."""
    s = 1.0
    seat, (gz, gy, gw), (fz, fy, fw) = tw.RIDER[aid]
    standing = seat is None  # kick e-scooter: standing on the deck, one foot ahead of the other
    hip = np.array([0.0, fy + 0.86, fz]) if standing else np.array([0.0, seat[1] + 0.07, seat[0]])
    if standing:
        feet = [np.array([-fw, fy, fz + 0.10]), np.array([fw, fy, fz - 0.10])]
    elif tw.MODELS[aid][0] == "bicycle":
        a = 2 * math.pi * phase
        bb = np.array([0.0, fy, fz])
        feet = [
            bb + np.array([side * fw, -0.17 * math.cos(a + ph), 0.17 * math.sin(a + ph)])
            for side, ph in ((-1.0, 0.0), (1.0, math.pi))
        ]
    else:
        feet = [np.array([side * fw, fy, fz]) for side in (-1.0, 1.0)]
    grips = [np.array([side * gw, gy, gz]) for side in (-1.0, 1.0)]

    def reach_error(lean: float) -> float:
        sh = hip + 0.55 * s * np.array([0.0, math.cos(math.radians(lean)), math.sin(math.radians(lean))])
        return abs(float(np.linalg.norm(sh + [0.18 * s, -0.03, 0.0] - grips[1])) - 0.5 * s)

    lean = min(np.arange(0.0 if standing else 5.0, 70.5, 1.0), key=reach_error)  # arms reach the grips
    parts = Parts()
    _seated_rider(parts, s, hip, feet, grips, float(lean), helmet=tw.MODELS[aid][0] not in ("bicycle", "kickscooter"))
    return parts


# ---------------------------------------------------------------- scene


CLEAR_M = 12.0


def clear_traffic(args, traffic: list[tuple]) -> tuple[list[tuple], list[dict]]:
    """Drop same-direction traffic within CLEAR_M of the target so it is not hidden by a moving car."""
    if not enabled(args):
        return traffic, []
    keep, removed = [], []
    for c in traffic:
        lane, z = c[0], c[1]
        if lane >= 0 and abs(z - args.vru_distance_m) < CLEAR_M:
            removed.append({"lane": lane, "distance_m": z, "model": c[2]})
        else:
            keep.append(c)
    return keep, removed


def scene_lines(
    args,
    road_edge_x: float,
    place,
    car_body,
    asset=None,
    out_dir=None,
    reflex_lines: list[str] | None = None,
    span: float | None = None,
) -> tuple[list[str], dict]:
    """pbrt lines for the target (+ obstruction cars) and manifest metadata.

    ``car_body(inst, model, paint) -> (include_rel, yaw_model_deg)`` builds a parked car include.
    ``reflex_lines``: reflector_material_lines() output (night); the procedural bicycle then carries reflectors.
    ``span``: shutter-open to last-row-close time [s] of an animated scene (None: static).
    """
    kind = args.vru
    height = HEIGHT_M[kind]
    bike, bike_meta, defs = None, None, []
    if kind in RIDERS and args.vru_bike and asset is not None:
        placer = tw.Placer(asset, out_dir, upright=True)
        bike, defs = placer.variant(args.vru_bike), placer.lines
        bike_meta = {
            "asset": args.vru_bike,
            "kind": tw.MODELS[args.vru_bike][0],
            "length_m": tw.MODELS[args.vru_bike][2],
            "scale": None if bike is None else round(float(bike["scale"]), 5),
            "missing": bike is None,  # --allow-missing-assets: procedural bicycle instead
        }
    v = speed_kmh(args) / 3.6 if span else 0.0
    wheels = [] if v > 0 and kind in RIDERS and bike is None else None
    tr = None
    if kind in RIDERS:
        tr = turn(args, WHEELBASE if bike is None else 0.62 * tw.MODELS[args.vru_bike][2])
    if bike is not None:
        parts = imported_rider(args.vru_bike, args.vru_gait_phase)
    elif kind in RIDERS:
        parts = cyclist(
            height, args.vru_gait_phase, wheels, reflectors=reflex_lines is not None, steer_deg=tr["steer_deg"]
        )
    else:
        parts = pedestrian(height, args.vru_gait_phase, child=kind == "child")
    x, z = road_edge_x + args.vru_offset_m, args.vru_distance_m
    yaw = -90.0 if args.vru_motion == "crossing" else 0.0  # local +z -> world -x (towards the median)
    lo, hi = parts.bounds()
    for centre, w, _ in wheels or []:
        wlo, whi = w.bounds()
        lo, hi = np.minimum(lo, wlo + centre), np.maximum(hi, whi + centre)
    wheel_deg = math.degrees(v * span / (WHEEL_D / 2)) if wheels else None
    if wheel_deg is not None and wheel_deg > MAX_WHEEL_ROTATION_DEG:
        raise SystemExit(
            f"--vru-speed-kmh: the wheels turn {wheel_deg:.0f} deg during the exposure (max "
            f"{MAX_WHEEL_ROTATION_DEG:.0f}); shorten --exposure-s or the rolling-shutter readout"
        )
        lo = np.minimum(lo, (bike["lo"] - bike["c"]) * bike["scale"])
        hi = np.maximum(hi, (bike["hi"] - bike["c"]) * bike["scale"])
    c, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    corners = np.array([[a, b, d] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for d in (lo[2], hi[2])])
    lean = tr["lean_deg"] if tr else 0.0
    if lean:
        corners = corners @ tw.rot_z(-lean).T
    world = np.stack(
        [c * corners[:, 0] + s_ * corners[:, 2], corners[:, 1], -s_ * corners[:, 0] + c * corners[:, 2]], -1
    )
    world += [x, 0.0, z]
    L = [
        "# ---- vulnerable road user (tools/highway_vru.py)",
        *material_lines(),
        *(reflex_lines if reflex_lines is not None and bike is None and kind in RIDERS else []),
        *defs,
        "AttributeBegin",
        *place(x, 0.0, z),
    ]
    L += [f"Rotate {yaw:.6g} 0 1 0"]
    if v > 0:  # rigid motion along the target's heading (local +z) during the exposure
        L += ["ActiveTransform EndTime", f"Translate 0 0 {v * span:.6g}", "ActiveTransform All"]
    if lean:  # roll about the tyre contact line (local z through the origin); + = top to the rider's left (+x)
        L += [f"Rotate {-lean:.6g} 0 0 1"]
    L += parts.lines()
    for centre, w, steer in wheels or []:  # rolling: rotation about the axle (x), top of the wheel moving forward
        L += [
            "AttributeBegin",
            f"Translate {centre[0]:.6g} {centre[1]:.6g} {centre[2]:.6g}",
            *([f"Rotate {steer[0]:.6g} {steer[1][0]:.6g} {steer[1][1]:.6g} {steer[1][2]:.6g}"] if steer else []),
            "ActiveTransform EndTime",
            f"Rotate {wheel_deg:.6g} 1 0 0",
            "ActiveTransform All",
            *w.lines(),
            "AttributeEnd",
        ]
    if bike is not None:
        L += tw.instance_lines(bike, (0.0, 0.0, 0.0), 0.0)
    L += ["AttributeEnd"]
    obstruction = []
    if args.vru_obstruction == "parked-cars":
        # Two parked cars on the shoulder, 1.0 m gap around the target, inner side 0.25 m outside the edge.
        car_x = road_edge_x + 0.25 + 0.95
        for k, (cz, model, paint) in enumerate(
            ((z - 0.5 - 2.3, "car_bmw_m6", "silver"), (z + 0.5 + 2.3, "car_vintage", "blue"))
        ):
            inc, yaw_model = car_body(f"vru_park{k}", model, paint)
            L += [
                "AttributeBegin",
                *place(car_x, 0.0, cz),
                f"Rotate {yaw_model:.6g} 0 1 0",
                f'Include "{inc}"',
                "AttributeEnd",
            ]
            obstruction.append({"model": model, "paint": paint, "x_m": car_x, "z_m": cz})
    meta = {
        "type": kind,
        "nominal_height_m": height,
        "anchor_m": [x, 0.0, z],
        "distance_m": z,
        "offset_from_road_edge_m": args.vru_offset_m,
        "motion": args.vru_motion,
        "gait_phase": args.vru_gait_phase,
        "bbox_world_m": [world.min(0).round(4).tolist(), world.max(0).round(4).tolist()],
        "triangles": int(sum(len(t) for v in parts.parts.values() for _, t, _ in v)),
        "obstruction": obstruction,
        "static": v == 0,
        "speed_kmh": round(v * 3.6, 4),
        "velocity_world_mps": [round(v * s_, 6), 0.0, round(v * c, 6)],
        "displacement_m": round(v * span, 6) if v else 0.0,
        "wheel_rotation_deg": None if wheel_deg is None else round(wheel_deg, 4),
        **(
            {
                "turn": {
                    **{k: (None if x is None else round(x, 4)) for k, x in tr.items()},
                    "steer_applied": bike is None or tr["steer_deg"] == 0,
                    "model": "steady turn: tan(lean) = v^2/(g R), kinematic steer tan(delta) = wheelbase/R; "
                    "straight path during the exposure",
                }
            }
            if tr
            else {}
        ),
        "motion_model": None
        if v == 0
        else "rigid translation over the exposure span (bbox_world_m at shutter open); procedural bicycle "
        "wheels roll; limbs, cranks and imported-model wheels keep their pose",
        **(
            {
                "reflectors": {
                    k: {"colour": c, "cil_table": t, "face_m": [w, h]} for k, (c, t, w, h) in REFLECTORS.items()
                }
            }
            if reflex_lines is not None and bike is None and kind in RIDERS
            else {}
        ),
        "geometry": "procedural, nominal Euro NCAP/ISO 19206 target sizes (not checked against the standard)",
    }
    if bike_meta is not None:
        meta["bike"] = bike_meta
        if bike is not None:
            meta["geometry"] = (
                "procedural rider on an imported Objaverse CC-BY two-wheeler (tools/two_wheelers.py); "
                "rider contact points read off the model's point-cloud profiles"
            )
    return L, meta
