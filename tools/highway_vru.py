"""Vulnerable road users for the highway scene: Euro NCAP-style AEB pedestrian / cyclist targets.

Opt-in (``--vru pedestrian|child|cyclist``). The targets are procedural articulated meshes built to
nominal Euro NCAP / ISO 19206 test-target sizes (adult 1.80 m, child 1.154 m, bicycle ~1.8 m long
with 0.70 m wheels); the standards are paywalled, so the sizes here are nominal and not checked
against the standard text. Placement follows the AEB VRU nearside scenarios: the target stands at
``--vru-distance-m`` ahead of the camera, ``--vru-offset-m`` from the right edge of the carriageway
(negative = into the lane), optionally between two parked cars on the hard shoulder
(``--vru-obstruction parked-cars``, as in the Euro NCAP obstructed-child test). The scene is static
(the target is posed mid-stride at ``--vru-gait-phase``), so motion blur of the target is not modelled.

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

REPO = Path(__file__).resolve().parents[1]
USGS = REPO / "spectra" / "usgs_nir"

HEIGHT_M = {"pedestrian": 1.80, "child": 1.154, "cyclist": 1.80}
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
    g.add_argument("--vru-obstruction", choices=("none", "parked-cars"), default="none")
    g.add_argument("--vru-clothing", choices=sorted(CLOTHING), default="ncap")
    g.add_argument("--vru-skin-tau", type=float, default=0.35, help="Melanin optical depth (iqlab.skin).")


def enabled(args: argparse.Namespace) -> bool:
    return args.vru != "none"


def check_args(args: argparse.Namespace) -> None:
    if not enabled(args):
        return
    if args.vru_distance_m <= 2.0:
        raise SystemExit("--vru-distance-m must be > 2 m")
    if not 0.0 <= args.vru_gait_phase < 1.0:
        raise SystemExit("--vru-gait-phase must be in [0, 1)")


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


def cyclist(height: float, phase: float) -> Parts:
    """Bicycle (axles along x, travelling +z) with a seated rider; total height ~``height``."""
    s = height / 1.80
    parts = Parts()
    rw = WHEEL_D / 2
    rear, front = np.array([0.0, rw, -WHEELBASE / 2]), np.array([0.0, rw, WHEELBASE / 2])
    for c in (rear, front):
        parts.torus("vru:tyre", c, rw - 0.016, 0.016)
        parts.torus("vru:rim", c, rw - 0.04, 0.008, n_v=6)
        for k in range(6):  # spoke stand-ins
            a = math.pi * k / 6
            d = np.array([0.0, math.cos(a), math.sin(a)]) * (rw - 0.045)
            parts.tube("vru:rim", c - d, c + d, 0.003, caps=False, n_u=4)
    bb = np.array([0.0, 0.28, -0.05])
    seat = np.array([0.0, 0.88, -0.25])
    head = np.array([0.0, 0.98, 0.38])
    for a, b in (
        (rear, bb),
        (rear, seat - [0, 0.08, 0]),
        (bb, seat - [0, 0.08, 0]),
        (bb, head),
        (seat - [0, 0.08, 0], head),
    ):
        parts.tube("vru:frame", a, b, 0.017)
    parts.tube("vru:frame", head, front, 0.014)
    parts.tube("vru:frame", head, head + [0.0, 0.08, -0.02], 0.014)
    bar = head + [0.0, 0.08, -0.02]
    parts.tube("vru:frame", bar - [0.24, 0, 0], bar + [0.24, 0, 0], 0.012)
    parts.ellipsoid("vru:tyre", seat, (0.07, 0.025, 0.13))
    # rider
    hip = seat + [0.0, 0.07, 0.0]
    a = 2 * math.pi * phase
    for side, ph in ((-1.0, 0.0), (1.0, math.pi)):
        pedal = bb + np.array([side * 0.11, -0.17 * math.cos(a + ph), 0.17 * math.sin(a + ph)])
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
    lean = math.radians(40.0)
    torso_len = 0.55 * s
    shoulder = hip + torso_len * np.array([0.0, math.cos(lean), math.sin(lean)])
    parts.tube("vru:top", hip + [0, 0.05, 0], shoulder, 0.15 * s, 0.17 * s)
    for side in (-1.0, 1.0):
        sh_p = shoulder + [side * 0.18 * s, -0.03, 0.0]
        grip = bar + [side * 0.22, 0.0, 0.0]
        elbow = (sh_p + grip) / 2 + [0.0, 0.04, -0.03]
        parts.tube("vru:top", sh_p, elbow, 0.05 * s, 0.045 * s)
        parts.tube("vru:top", elbow, grip, 0.04 * s, 0.035 * s)
        parts.ellipsoid("vru:skin", grip, (0.035, 0.035, 0.05))
    neck = shoulder + [0.0, 0.08 * s, 0.02]
    parts.tube("vru:skin", shoulder, neck, 0.05 * s)
    head_c = neck + np.array([0.0, 0.10 * s, 0.05 * s])
    parts.ellipsoid("vru:skin", head_c, (0.075 * s, 0.11 * s, 0.095 * s))
    parts.ellipsoid("vru:hair", head_c + [0.0, 0.03 * s, -0.01], (0.085 * s, 0.09 * s, 0.11 * s))  # helmet-ish
    return parts


# ---------------------------------------------------------------- scene


def scene_lines(args, road_edge_x: float, place, car_body) -> tuple[list[str], dict]:
    """pbrt lines for the target (+ obstruction cars) and manifest metadata.

    ``car_body(inst, model, paint) -> (include_rel, yaw_model_deg)`` builds a parked car include.
    """
    kind = args.vru
    height = HEIGHT_M[kind]
    parts = (
        cyclist(height, args.vru_gait_phase)
        if kind == "cyclist"
        else pedestrian(height, args.vru_gait_phase, child=kind == "child")
    )
    x, z = road_edge_x + args.vru_offset_m, args.vru_distance_m
    yaw = -90.0 if args.vru_motion == "crossing" else 0.0  # local +z -> world -x (towards the median)
    lo, hi = parts.bounds()
    c, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    corners = np.array([[a, b, d] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for d in (lo[2], hi[2])])
    world = np.stack(
        [c * corners[:, 0] + s_ * corners[:, 2], corners[:, 1], -s_ * corners[:, 0] + c * corners[:, 2]], -1
    )
    world += [x, 0.0, z]
    L = ["# ---- vulnerable road user (tools/highway_vru.py)", *material_lines(), "AttributeBegin", *place(x, 0.0, z)]
    L += [f"Rotate {yaw:.6g} 0 1 0", *parts.lines(), "AttributeEnd"]
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
        "static": True,
        "geometry": "procedural, nominal Euro NCAP/ISO 19206 target sizes (not checked against the standard)",
    }
    return L, meta
