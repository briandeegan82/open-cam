"""Seeded scene variety for tools/build_highway_scene.py (dataset generation).

``--seed N`` draws a reproducible scene: traffic in both carriageways (cars, vans, rigid trucks,
articulated trucks, buses) with lane-dependent density and heavy-vehicle share, car paints from
the global colour-popularity distribution as spectral basecoats, a horizontal alignment (tangent
-> clothoid -> circular arc -> clothoid -> tangent) and a graded vertical alignment with
parabolic vertical curves, median lamp posts, an overhead sign gantry, an optional overpass and a
mix of tree/shrub species. Without ``--seed`` nothing here changes the scene except the textured
median barrier / W-beam materials. Every random draw comes from one ``numpy`` generator seeded
with N, in a fixed order, so a seed always gives the same scene.

Sources for parameters (see also docs/HIGHWAY_SCENES.md, "Scene variety"):

* Paint shares: DuPont (Axalta) 2012 Global Automotive Color Popularity Report, world totals:
  white 23 %, black 21 %, silver 18 %, grey 14 %, red 8 %, blue 6 %, brown/beige 6 %, green 1 %,
  other (yellow/gold, orange) 3 %. Commercial vans/trucks are mostly white fleet liveries.
* Alignment: AASHTO "A Policy on Geometric Design of Highways and Streets" (Green Book, 2018):
  minimum radius ~ 600-700 m at 110-120 km/h (e_max 8 %), freeway grades <= 3-4 % (rolling
  terrain), parabolic vertical curves, clothoid (Euler spiral) transitions.
* Traffic: HCM 6th ed. basic freeway segments; LOS A-C densities 7-16 pc/km/ln; default heavy
  vehicle share 10-25 %, heavy vehicles concentrated in the right-hand lanes.
* Lighting columns: BS 5489-1:2020 / EN 13201: 10-12 m mounting height, spacing ~3.5-4 x height.
  Gantry clearance >= 5.5 m (UK DMRB CD 127 5.7 m; MUTCD 17 ft = 5.2 m); overpass headroom
  5.3 m (AASHTO 16 ft minimum, 5.0 m UK).
* Zinc optical constants (galvanised rails): Werner, Glantschnig & Ambrosch-Draxl, J. Phys. Chem.
  Ref. Data 38, 1013 (2009), via refractiveindex.info (CC0). Weathered (zinc carbonate patina)
  galvanised steel diffuse reflectance ~0.25-0.35 (e.g. ASHRAE/LBNL cool-roof solar reflectance
  data for aged galvanised steel); weathered grey concrete 0.2-0.3 (Levinson & Akbari 2002).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fetch_highway_assets import read_ply_positions

PAINT_SHARES = {
    "white": 0.23,
    "black": 0.21,
    "silver": 0.18,
    "gray": 0.14,
    "red": 0.08,
    "blue": 0.06,
    "beige": 0.03,
    "brown": 0.03,
    "darkgreen": 0.01,
    "yellow": 0.015,
    "orange": 0.015,
}
FLEET_PAINT_SHARES = {"white": 0.62, "silver": 0.12, "gray": 0.08, "blue": 0.06, "red": 0.06, "black": 0.06}


def _sig(wl, c, w):
    return 1.0 / (1.0 + np.exp(-(wl - c) / w))


def _g(wl, c, s):
    return np.exp(-0.5 * ((wl - c) / s) ** 2)


# Basecoats not in highway_spectra.CAR_PAINTS (same analytic style: step/band reflectances of
# pigmented automotive basecoats, cf. Kirchner et al., "Digitally reconstructing van Gogh's
# Field with Irises" style pigment curves; USGS splib07 painted-metal samples).
EXTRA_PAINTS = {
    "beige": lambda wl: 0.20 + 0.38 * _sig(wl, 545.0, 28.0),
    "brown": lambda wl: 0.04 + 0.14 * _sig(wl, 585.0, 30.0),
    "yellow": lambda wl: 0.06 + 0.66 * _sig(wl, 512.0, 11.0),
    "orange": lambda wl: 0.05 + 0.64 * _sig(wl, 572.0, 11.0),
}

# Zinc n, k, 360-830 nm (Werner et al. 2009); the spd must span pbrt's whole 360-830 nm range,
# otherwise eta = k = 0 outside it and the conductor Fresnel term returns NaN.
_ZN_WL = np.array([360.0, 410.0, 460.0, 510.0, 560.0, 610.0, 660.0, 710.0, 760.0, 810.0, 830.0])
_ZN_N = np.array([0.53, 0.598, 0.701, 0.823, 0.959, 1.106, 1.266, 1.43, 1.61, 1.789, 1.86])
_ZN_K = np.array([3.056, 3.603, 4.111, 4.597, 5.066, 5.521, 5.96, 6.394, 6.803, 7.213, 7.38])

# name: (kind, scale multiplier relative to the builder's tree/shrub scale draw)
SPECIES = {
    "veg_island_tree_02": ("tree", 1.0),
    "veg_searsia_lucida": ("tree", 0.8),
    "veg_fir_sapling_medium": ("tree", 0.4),
    "veg_pine_sapling_small": ("tree", 1.1),
    "veg_shrub_02": ("shrub", 1.0),
    "veg_shrub_01": ("shrub", 1.5),
    "veg_shrub_04": ("shrub", 3.0),
}
DEFAULT_SPECIES = [("veg_island_tree_02", "tree"), ("veg_shrub_02", "shrub")]

# Heavy/commercial vehicles (manifest kind "glb"): class, model forward axis, real length [m],
# paintable materials and proxy box (width, height) for --allow-missing-assets.
HEAVY = {
    "van_sprinter": {"cls": "van", "forward": "+z", "length_m": 5.91, "paint": ["Carrosserie", "Carrosserie_5"]},
    "truck_box": {"cls": "rigid", "forward": "+z", "length_m": 12.5, "paint": ["bodycolour", "head_paint"]},
    "truck_daf_cf_tractor": {"cls": "tractor", "forward": "+z", "length_m": 6.0, "paint": []},
    "trailer_box": {"cls": "trailer", "forward": "+x", "length_m": 9.9, "paint": []},
    "bus_town": {"cls": "bus", "forward": "+x", "length_m": 12.0, "paint": []},
}
_PROXY_WH = {"van": (2.0, 2.5), "rigid": (2.5, 3.6), "tractor": (2.5, 3.2), "trailer": (2.5, 4.0), "bus": (2.55, 3.1)}
_YAW = {"+z": 0.0, "-z": 180.0, "+x": -90.0, "-x": 90.0}


def add_arguments(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("scene variety (tools/highway_variety.py)")
    g.add_argument("--curve-radius", type=float, default=None, help="Arc radius [m], + = bends right, 0 = straight.")
    g.add_argument("--clothoid-length", type=float, default=None, help="Spiral transition length [m] (0 = none).")
    g.add_argument("--grade", type=float, default=None, help="Road grade [%%], + = uphill.")
    g.add_argument("--lamp-posts", choices=("on", "off"), default=None)
    g.add_argument("--gantry", choices=("on", "off"), default=None)
    g.add_argument("--overpass", choices=("on", "off"), default=None)
    g.add_argument("--barrier-materials", choices=("textured", "flat"), default="textured")


# ---------------------------------------------------------------------------- alignment
@dataclass
class Alignment:
    """Road centre-line: straight frame (x lateral, y up, z = chainage s) -> world."""

    radius_m: float = 0.0  # 0 = straight; sign = bend direction (+ right)
    clothoid_m: float = 0.0
    curve_start_m: float = 80.0
    max_turn_deg: float = 30.0
    grade: float = 0.0  # fraction
    grade_start_m: float = 60.0
    vc_length_m: float = 250.0
    grade_end_m: float = 1300.0
    ds: float = 0.5
    s_min: float = -600.0
    s_max: float = 14000.0

    def __post_init__(self) -> None:
        s = np.arange(self.s_min, self.s_max + self.ds, self.ds)
        k = np.zeros_like(s)
        if self.radius_m:
            k0, lc = 1.0 / self.radius_m, max(self.clothoid_m, 0.0)
            arc = max(math.radians(self.max_turn_deg) * abs(self.radius_m) - lc, 0.0)
            a, b, c = self.curve_start_m, self.curve_start_m + lc, self.curve_start_m + lc + arc
            ramp_in = np.clip((s - a) / lc, 0, 1) if lc > 0 else (s >= a).astype(float)
            ramp_out = np.clip((c + lc - s) / lc, 0, 1) if lc > 0 else (s < c).astype(float)
            k = k0 * np.where(s < b, ramp_in, np.where(s < c, 1.0, ramp_out))
        th = np.concatenate([[0.0], np.cumsum(0.5 * (k[1:] + k[:-1]) * self.ds)])
        th -= np.interp(0.0, s, th)
        cx = np.concatenate([[0.0], np.cumsum(0.5 * (np.sin(th[1:]) + np.sin(th[:-1])) * self.ds)])
        cz = np.concatenate([[0.0], np.cumsum(0.5 * (np.cos(th[1:]) + np.cos(th[:-1])) * self.ds)])
        cx -= np.interp(0.0, s, cx)
        cz -= np.interp(0.0, s, cz)
        # vertical: 0 -> grade over a sag/crest curve, hold, back to 0 (distant terrain stays level)
        g = self.grade * (
            np.clip((s - self.grade_start_m) / self.vc_length_m, 0, 1)
            - np.clip((s - self.grade_end_m) / self.vc_length_m, 0, 1)
        )
        e = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * self.ds)])
        e -= np.interp(0.0, s, e)
        self._s, self._th, self._cx, self._cz, self._e, self._g, self._k = s, th, cx, cz, e, g, k

    @property
    def straight(self) -> bool:
        return not self.radius_m and not self.grade

    def lateral(self, x: np.ndarray) -> np.ndarray:
        """Compress far terrain on the inside of the bend so the warp never folds (|x k| < 0.5)."""
        if not self.radius_m:
            return x
        r, a = abs(self.radius_m), 100.0
        b = 0.5 * r
        inner = np.sign(x) == np.sign(self.radius_m)
        ax = np.abs(x)
        comp = np.sign(x) * (a + (b - a) * np.tanh((ax - a) / (b - a)))
        return np.where(inner & (ax > a), comp, x)

    def frame(self, s: np.ndarray):
        s = np.asarray(s, dtype=np.float64)
        sc = np.clip(s, self._s[0], self._s[-1])
        th = np.interp(sc, self._s, self._th)
        cx = np.interp(sc, self._s, self._cx) + (s - sc) * np.sin(th)
        cz = np.interp(sc, self._s, self._cz) + (s - sc) * np.cos(th)
        e = np.interp(sc, self._s, self._e)
        return th, cx, cz, e, np.interp(sc, self._s, self._g)

    def warp(self, p: np.ndarray) -> np.ndarray:
        th, cx, cz, e, _ = self.frame(p[:, 2])
        x = self.lateral(p[:, 0])
        return np.stack([cx + x * np.cos(th), p[:, 1] + e, cz - x * np.sin(th)], -1)

    def place(self, x: float, y: float, z: float) -> tuple[np.ndarray, float, float]:
        """World position, heading [deg] and pitch [deg] for an object at straight-frame (x, y, z)."""
        p = self.warp(np.array([[x, y, z]], dtype=np.float64))[0]
        th, *_, g = self.frame(np.array([z]))
        return p, math.degrees(float(th[0])), math.degrees(math.atan(float(g[0])))


def refine_along_s(p: np.ndarray, tri: np.ndarray, uv: np.ndarray | None, base: float = 6.0, rel: float = 0.012):
    """Conforming red/green split of triangle edges longer than max(base, rel*|s|) along s (= z)."""
    p, tri = np.asarray(p, np.float64), np.asarray(tri, np.int64)
    uv = None if uv is None else np.asarray(uv, np.float64)
    for _ in range(16):
        e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
        z0, z1 = p[e[:, 0], 2], p[e[:, 1], 2]
        mark = np.abs(z1 - z0) > np.maximum(base, rel * 0.5 * np.abs(z0 + z1))
        if not mark.any():
            break
        key = np.sort(e[mark], 1)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        mid = len(p) + np.arange(len(uniq))
        p = np.concatenate([p, 0.5 * (p[uniq[:, 0]] + p[uniq[:, 1]])])
        if uv is not None:
            uv = np.concatenate([uv, 0.5 * (uv[uniq[:, 0]] + uv[uniq[:, 1]])])
        m = np.full(len(e), -1, np.int64)
        m[np.where(mark)[0]] = mid[inv.ravel()]
        nt = len(tri)
        m3 = np.stack([m[:nt], m[nt : 2 * nt], m[2 * nt :]], 1)  # midpoints of edges 01, 12, 20
        out = [tri[(m3 < 0).all(1)]]
        for i in np.where((m3 >= 0).any(1))[0]:
            v, mm = list(tri[i]), list(m3[i])
            r = next(j for j in range(3) if mm[j] >= 0 and mm[(j + 2) % 3] < 0) if not all(x >= 0 for x in mm) else 0
            a, b, c = v[r], v[(r + 1) % 3], v[(r + 2) % 3]
            mab, mbc, mca = mm[r], mm[(r + 1) % 3], mm[(r + 2) % 3]
            if mbc < 0 and mca < 0:
                out.append(np.array([(a, mab, c), (mab, b, c)]))
            elif mca < 0:
                out.append(np.array([(a, mab, c), (mab, b, mbc), (mab, mbc, c)]))
            else:
                out.append(np.array([(a, mab, mca), (mab, b, mbc), (mca, mbc, c), (mab, mbc, mca)]))
        tri = np.concatenate(out)
    return p, tri, uv


# ---------------------------------------------------------------------------- settings
@dataclass
class Variety:
    seed: int | None = None
    alignment: Alignment = field(default_factory=Alignment)
    lamp_posts: bool = False
    gantry: bool = False
    overpass: bool = False
    gantry_s: float = 150.0
    overpass_s: float = 320.0
    lamp_spacing_m: float = 42.0
    species: list[tuple[str, str]] = field(default_factory=lambda: list(DEFAULT_SPECIES))
    traffic: list | None = None
    paints: dict[str, tuple[str, float]] = field(default_factory=dict)
    barrier_textured: bool = True
    lamp_heads: list[dict] = field(default_factory=list)

    def species_scale(self, vid: str) -> float:
        return SPECIES[vid][1]

    def species_spacing(self, kind: str) -> float:
        return float(sum(k == kind for _, k in self.species))

    def manifest(self) -> dict:
        a = self.alignment
        return {
            "seed": self.seed,
            "alignment": {
                "curve_radius_m": a.radius_m or None,
                "clothoid_length_m": a.clothoid_m,
                "curve_start_m": a.curve_start_m,
                "max_turn_deg": a.max_turn_deg,
                "grade_percent": 100.0 * a.grade,
                "grade_start_m": a.grade_start_m,
                "vertical_curve_m": a.vc_length_m,
            },
            "lamp_posts": {"spacing_m": self.lamp_spacing_m, "heads": self.lamp_heads} if self.lamp_posts else None,
            "gantry_distance_m": self.gantry_s if self.gantry else None,
            "overpass_distance_m": self.overpass_s if self.overpass else None,
            "vegetation_species": [v for v, _ in self.species],
            "barrier_materials": "textured" if self.barrier_textured else "flat",
            "paint_distribution": "DuPont 2012 global colour popularity (white/black/silver/grey 76 %)",
        }


def _pick(rng: np.random.Generator, shares: dict[str, float]) -> str:
    keys = list(shares)
    w = np.array([shares[k] for k in keys])
    return keys[int(rng.choice(len(keys), p=w / w.sum()))]


def resolve(args: argparse.Namespace, car_models: tuple[str, ...], cam_lane: int) -> Variety:
    v = Variety(seed=args.seed, barrier_textured=args.barrier_materials == "textured")
    rng = np.random.default_rng(args.seed) if args.seed is not None else None
    r = rng.uniform if rng is not None else None

    def opt(val, draw, off):
        if val is not None:
            return val
        return draw() if rng is not None else off

    radius = opt(args.curve_radius, lambda: 0.0 if r() < 0.3 else float(np.sign(r(-1, 1)) * r(700, 2500)), 0.0)
    clothoid = opt(args.clothoid_length, lambda: 0.0 if r() < 0.3 else r(60, 180), 0.0)
    grade = opt(args.grade, lambda: 0.0 if r() < 0.3 else r(-4.0, 4.0), 0.0)
    start = r(60, 200) if rng is not None else 80.0
    gstart = r(40, 200) if rng is not None else 60.0
    v.alignment = Alignment(radius_m=radius, clothoid_m=clothoid, curve_start_m=start, grade=grade / 100.0,
                            grade_start_m=gstart)  # fmt: skip
    on = {"on": True, "off": False}
    v.lamp_posts = on[args.lamp_posts] if args.lamp_posts else bool(rng is not None and r() < 0.6)
    v.gantry = on[args.gantry] if args.gantry else bool(rng is not None and r() < 0.5)
    v.overpass = on[args.overpass] if args.overpass else bool(rng is not None and r() < 0.35)
    if rng is not None:
        v.gantry_s, v.overpass_s, v.lamp_spacing_m = r(90, 220), r(240, 480), r(36, 50)
        trees = [k for k, (kind, _) in SPECIES.items() if kind == "tree"]
        shrubs = [k for k, (kind, _) in SPECIES.items() if kind == "shrub"]
        v.species = [(t, "tree") for t in rng.choice(trees, int(rng.integers(1, 4)), replace=False)]
        v.species += [(s, "shrub") for s in rng.choice(shrubs, int(rng.integers(1, 4)), replace=False)]
        v.species = [(str(a), b) for a, b in v.species]
        v.traffic = random_traffic(rng, car_models, cam_lane, v.paints)
    return v


def random_traffic(rng: np.random.Generator, car_models, cam_lane: int, paints: dict) -> list[tuple]:
    """(lane, s, model, paint label, yaw jitter) for both carriageways; lanes < 0 are oncoming."""
    out: list[tuple] = []
    hv_share = rng.uniform(0.05, 0.25)
    lane_hv = {0: 0.2, 1: 1.0, 2: 1.8, -1: 0.2, -2: 1.0, -3: 1.8}
    for lane in (0, 1, 2, -1, -2, -3):
        density = rng.uniform(4.0, 22.0)  # veh/km/lane
        s = (14.0 if lane == cam_lane else 6.0) + rng.exponential(1000.0 / density)
        while s < 320.0:
            hv = rng.uniform() < min(hv_share * lane_hv[lane], 0.6)
            if hv:
                cls = _pick(rng, {"semi": 0.45, "rigid": 0.25, "van": 0.18, "bus": 0.12})
            else:
                cls = "van" if rng.uniform() < 0.08 else "car"
            yaw = float(rng.normal(0.0, 0.6))
            sign = 1.0 if lane >= 0 else -1.0
            if cls == "car":
                model, length = str(rng.choice(car_models)), 4.7
                paint = _pick(rng, PAINT_SHARES)
            else:
                model = {
                    "semi": "truck_daf_cf_tractor",
                    "rigid": "truck_box",
                    "van": "van_sprinter",
                    "bus": "bus_town",
                }[cls]
                length = HEAVY[model]["length_m"]
                paint = _pick(rng, FLEET_PAINT_SHARES)
            label = f"{paint}_{len(paints):02d}"
            paints[label] = (paint, float(rng.uniform(0.9, 1.08)))
            if cls == "semi":  # tractor + box trailer, 1.8 m overlap at the fifth wheel
                lt, tr = length, HEAVY["trailer_box"]["length_m"]
                first, second = (("trailer_box", tr), (model, lt)) if sign > 0 else ((model, lt), ("trailer_box", tr))
                out.append((lane, round(s + first[1] / 2, 2), first[0], label, yaw))
                out.append((lane, round(s + first[1] - 1.8 + second[1] / 2, 2), second[0], label, yaw))
                length = lt + tr - 1.8
            else:
                out.append((lane, round(s + length / 2, 2), model, label, yaw))
            s += length + max(10.0, rng.exponential(1000.0 / density))
    return out


# ---------------------------------------------------------------------------- builder hooks
WARP: Alignment | None = None


def activate(v: Variety) -> None:
    global WARP
    WARP = None if v.alignment.straight else v.alignment


def warp_mesh(p, tri, uv, n):
    """Hook for build_highway_scene.mesh(): refine along s and bend onto the alignment."""
    if WARP is None:
        return p, tri, uv, n
    p, tri, uv = refine_along_s(p, tri, uv)
    return WARP.warp(p), tri, uv, None


def place(x: float, y: float, z: float) -> list[str]:
    """Transform lines for an instance at straight-frame (x, y, z) (identity Translate if straight)."""
    if WARP is None:
        return [f"Translate {x:.6g} {y:.6g} {z:.6g}"]
    p, heading, pitch = WARP.place(x, y, z)
    return [f"Translate {p[0]:.6g} {p[1]:.6g} {p[2]:.6g}", f"Rotate {heading:.6g} 0 1 0", f"Rotate {-pitch:.6g} 1 0 0"]


def paint_reflectance(colour: str, wl: np.ndarray, reflectance) -> np.ndarray:
    if colour in EXTRA_PAINTS:
        return np.clip(EXTRA_PAINTS[colour](wl), 0.0, 1.0)
    return reflectance(f"carpaint_{colour}", wl)


def write_paints(v: Variety, spd: Path, wl: np.ndarray, reflectance, write_spd) -> None:
    """Per-vehicle basecoat SPDs ``spd/carpaint_<label>.spd`` (colour x lightness jitter)."""
    for label, (colour, gain) in v.paints.items():
        write_spd(
            spd / f"carpaint_{label}.spd", wl, np.clip(gain * paint_reflectance(colour, wl, reflectance), 0, 0.95)
        )


def barrier_materials(
    v: Variety, asset, out_dir: Path, wl: np.ndarray, write_spd, luminance_texture
) -> list[str] | None:
    """Textured concrete + weathered galvanised (zinc conductor / patina mix), world-planar mapped."""
    if not v.barrier_textured:
        return None
    con, met = asset("tex_concrete031"), asset("tex_metal032")
    if con is None or met is None:
        return None
    spd = out_dir / "spd"
    write_spd(spd / "zinc_eta.spd", wl, np.interp(wl, _ZN_WL, _ZN_N))
    write_spd(spd / "zinc_k.spd", wl, np.interp(wl, _ZN_WL, _ZN_K))
    write_spd(spd / "zinc_patina.spd", wl, 0.27 + 0.03 * np.clip((wl - 400.0) / 300.0, 0, 1))
    write_spd(spd / "concrete_weathered.spd", wl, 0.22 + 0.06 * np.clip((wl - 400.0) / 300.0, 0, 1))
    tex = out_dir / "textures"
    (ci, craw), (mi, mraw) = con, met
    ct, mt = float(ci["tile_size_m"]), float(mi["tile_size_m"])
    luminance_texture(craw / ci["maps"]["color"], tex / "concrete031_lum.exr")
    amount = tex / "galvanized_patina.png"
    if not amount.is_file():
        from PIL import Image

        im = np.asarray(Image.open(mraw / mi["maps"]["roughness"]).convert("L").resize((512, 512)), np.float64) / 255
        a = np.clip(0.72 + 1.6 * (im - im.mean()), 0.35, 0.97)  # patina fraction, mean ~0.72
        Image.fromarray((a * 255).astype(np.uint8)).save(amount)

    def planar(t: float, a: str, b: str) -> str:
        return f'"string mapping" "planar" "vector3 v1" [{a}] "vector3 v2" [{b}]'.replace("T", f"{1 / t:.6g}")

    return [
        'Texture "concrete:lum" "float" "imagemap" "string filename" "textures/concrete031_lum.exr" '
        + planar(ct, "0 0 T", "T T 0"),
        'Texture "concrete" "spectrum" "scale" "spectrum tex" "spd/concrete_weathered.spd" "texture scale" "concrete:lum"',
        'MakeNamedMaterial "concrete" "string type" "diffuse" "texture reflectance" "concrete"',
        'MakeNamedMaterial "zinc" "string type" "conductor" "spectrum eta" "spd/zinc_eta.spd"'
        ' "spectrum k" "spd/zinc_k.spd" "float roughness" [0.25]',
        'MakeNamedMaterial "zinc_patina" "string type" "diffuse" "spectrum reflectance" "spd/zinc_patina.spd"',
        'Texture "galvanized:amount" "float" "imagemap" "string filename" "textures/galvanized_patina.png"'
        ' "string encoding" "linear" ' + planar(mt, "0 0 T", "T T 0"),
        'MakeNamedMaterial "galvanized" "string type" "mix" "string materials" ["zinc" "zinc_patina"]',
        '    "texture amount" "galvanized:amount"',
    ]


def _robust_box(info: dict, prep: Path) -> tuple[np.ndarray, np.ndarray]:
    cache = prep / "robust_bbox.json"
    if cache.is_file():
        d = json.loads(cache.read_text())
        return np.array(d["lo"]), np.array(d["hi"])
    pts = np.concatenate([read_ply_positions(prep / s["ply"]) for s in info["shapes"]])
    lo, hi = np.percentile(pts, 0.05, axis=0), np.percentile(pts, 99.95, axis=0)
    cache.write_text(json.dumps({"lo": lo.tolist(), "hi": hi.tolist()}))
    return lo, hi


def glb_vehicle_include(model: str, info: dict, prep: Path, raw: Path, out_dir: Path, inst: str, paint_spd: str,
                        luminance_texture) -> tuple[list[str], float]:  # fmt: skip
    """pbrt lines for a prepared glb vehicle (front -> +z after the returned yaw, ground at y=0)."""
    spec = HEAVY[model]
    lines = []
    for k, m in enumerate(info["materials"]):
        mn, name = f"{inst}:{k}", m["name"].lower()
        tex = m.get("base_color_texture")
        rel = os.path.relpath(raw / tex, out_dir) if tex else None
        bc = [min(max(c, 0.02), 0.9) for c in m["base_color_factor"]]
        if m["name"] in spec["paint"]:
            refl = f'"spectrum reflectance" "{paint_spd}"'
            if tex:
                lum = out_dir / "textures" / f"{model}_{k}_lum.exr"
                luminance_texture(raw / tex, lum, size=512)
                lines += [
                    f'Texture "{mn}:lum" "float" "imagemap" "string filename" "{os.path.relpath(lum, out_dir)}"',
                    f'Texture "{mn}:c" "spectrum" "scale" "spectrum tex" "{paint_spd}" "texture scale" "{mn}:lum"',
                ]
                refl = f'"texture reflectance" "{mn}:c"'
            lines.append(f'MakeNamedMaterial "{mn}" "string type" "coateddiffuse" {refl} "float roughness" [0.01]')
        elif m.get("alpha_mode") == "BLEND" or any(w in name for w in ("glass", "vitre", "windglass")):
            lines.append(f'MakeNamedMaterial "{mn}" "string type" "thindielectric" "float eta" [1.52]')
        elif any(w in name for w in ("pneu", "tyre", "tire", "rezina")):
            lines.append(
                f'MakeNamedMaterial "{mn}" "string type" "coateddiffuse" "spectrum reflectance" "spd/rubber.spd"'
                ' "float roughness" [0.45]'
            )
        elif tex:
            lines += [
                f'Texture "{mn}:c" "spectrum" "imagemap" "string filename" "{rel}" "string encoding" "sRGB"',
                f'MakeNamedMaterial "{mn}" "string type" "diffuse" "texture reflectance" "{mn}:c"',
            ]
        elif m.get("metallic", 0.0) >= 0.8 or "chrome" in name:
            lines.append(
                f'MakeNamedMaterial "{mn}" "string type" "conductor" "spectrum eta" "metal-Al-eta"'
                f' "spectrum k" "metal-Al-k" "float roughness" [{max(m.get("roughness", 0.2), 0.05):.3g}]'
            )
        else:
            lines.append(
                f'MakeNamedMaterial "{mn}" "string type" "diffuse" "rgb reflectance" [{bc[0]:.3g} {bc[1]:.3g} {bc[2]:.3g}]'
            )
    lo, hi = _robust_box(info, prep)
    ax = 2 if spec["forward"][1] == "z" else 0
    s = spec["length_m"] / float(hi[ax] - lo[ax])
    c = 0.5 * (lo + hi)
    lines += [f"Scale {s:.6g} {s:.6g} {s:.6g}", f"Translate {-c[0]:.6g} {-lo[1]:.6g} {-c[2]:.6g}"]
    for sh in info["shapes"]:
        ply = os.path.relpath(prep / sh["ply"], out_dir)
        lines += [f'NamedMaterial "{inst}:{sh["material"]}"', f'Shape "plymesh" "string filename" "{ply}"']
    return lines, _YAW[spec["forward"]]


def proxy_vehicle(model: str, paint_spd: str, inst: str, mesh, box) -> list[str] | None:
    """Box stand-in for a missing heavy vehicle (front +z, ground y=0); None for cars."""
    if model not in HEAVY:
        return None
    spec = HEAVY[model]
    w, h = _PROXY_WH[spec["cls"]]
    ln = spec["length_m"]
    return [
        f'MakeNamedMaterial "{inst}:paint" "string type" "coateddiffuse" "spectrum reflectance" "{paint_spd}"'
        ' "float roughness" [0.01]',
        f'MakeNamedMaterial "{inst}:tire" "string type" "diffuse" "spectrum reflectance" "spd/rubber.spd"',
        f'NamedMaterial "{inst}:paint"',
        *mesh(*box(0, 0.5, 0, w, h - 0.5, ln)),
        f'NamedMaterial "{inst}:tire"',
        *mesh(*box(0, 0.0, 0, w - 0.2, 0.5, ln * 0.8)),
    ]


def _quad(cx: float, y0: float, cz: float, w: float, h: float):
    p = np.array([(cx - w / 2, y0, cz), (cx + w / 2, y0, cz), (cx + w / 2, y0 + h, cz), (cx - w / 2, y0 + h, cz)])
    return p, np.array([(0, 2, 1), (0, 3, 2)]), np.array([(0, 0), (1, 0), (1, 1), (0, 1)], np.float64)


def structures(v: Variety, out_dir: Path, layout, median_x: float, mesh, box, sign_legend, lane_w: float) -> list[str]:
    """Lamp posts on the median, overhead sign gantry and overpass (all in the straight frame)."""
    rng = np.random.default_rng(None if v.seed is None else v.seed + 1)
    L: list[str] = []
    if v.lamp_posts:
        L += [
            'MakeNamedMaterial "lamp_housing" "string type" "coateddiffuse" "spectrum reflectance" "spd/zinc_patina.spd"'
            ' "float roughness" [0.3]'
            if (out_dir / "spd" / "zinc_patina.spd").is_file()
            else 'MakeNamedMaterial "lamp_housing" "string type" "diffuse" "spectrum reflectance" "spd/galvanized.spd"',
            'MakeNamedMaterial "luminaire_lens" "string type" "dielectric" "float eta" [1.49]',  # PMMA
            'NamedMaterial "galvanized"',
        ]
        hgt, arm = 12.0, 2.2
        z = float(rng.uniform(5.0, v.lamp_spacing_m))
        posts, heads = [], []
        while z < 1200.0:
            posts.append(box(median_x, 0.0, z, 0.22, hgt, 0.22))
            posts.append(box(median_x, hgt - 0.1, z, 2 * arm, 0.1, 0.1))
            for sgn in (-1.0, 1.0):
                heads.append((median_x + sgn * (arm - 0.3), hgt - 0.2, z, sgn))
            z += v.lamp_spacing_m
        pts = [p for p, _ in posts]
        tri = [t + 8 * i for i, (_, t) in enumerate(posts)]
        L += mesh(np.concatenate(pts), np.concatenate(tri))
        hp, ht, lp, lt = [], [], [], []
        for i, (x, y, z, sgn) in enumerate(heads):
            p, t = box(x, y, z, 0.75, 0.16, 0.32)
            hp.append(p), ht.append(t + 8 * i)
            p, t = box(x, y - 0.03, z, 0.6, 0.03, 0.26)
            lp.append(p), lt.append(t + 8 * i)
            v.lamp_heads.append({"x": x, "y": y - 0.03, "s": z, "aims": "+x" if sgn > 0 else "-x"})
        L += ['NamedMaterial "lamp_housing"', *mesh(np.concatenate(hp), np.concatenate(ht))]
        L += ['NamedMaterial "luminaire_lens"', *mesh(np.concatenate(lp), np.concatenate(lt)), ""]
    if v.gantry:
        zg, x0, x1 = v.gantry_s, median_x, layout.rail_right + 1.2
        xs = (x0 + 0.6, x1)
        L.append('NamedMaterial "galvanized"')
        parts = [box(x, 0.0, zg, 0.4, 7.9, 0.4) for x in xs]
        for y in (6.0, 7.6):
            parts.append(box(0.5 * (x0 + x1), y, zg, x1 - x0, 0.25, 0.25))
            parts.append(box(0.5 * (x0 + x1), y, zg + 1.0, x1 - x0, 0.25, 0.25))
        for x in np.linspace(xs[0], xs[1], 7):
            parts.append(box(float(x), 6.0, zg + 0.5, 0.12, 1.85, 1.0))
        L += mesh(np.concatenate([p for p, _ in parts]), np.concatenate([t + 8 * i for i, (_, t) in enumerate(parts)]))
        towns = ["Galway", "Athlone", "Dublin", "Limerick", "Sligo", "Ennis", "Tuam", "Loughrea", "Shannon"]
        picks = rng.choice(len(towns), 3, replace=False)
        for k, lane in enumerate((0, 1, 2)):
            leg = out_dir / "textures" / f"gantry_sign_{k}.png"
            n1 = int(rng.integers(3, 60))
            sign_legend(leg, [f"{towns[picks[k]]}  {n1}", f"Exit {int(rng.integers(5, 30))}"], (900, 520), 14)
            L += [
                f'Texture "gantry_legend_{k}" "float" "imagemap" "string filename" "textures/{leg.name}"',
                f'MakeNamedMaterial "gantry_sign_{k}" "string type" "mix" "string materials" ["sheet_green" "sheet_white"]',
                f'    "texture amount" "gantry_legend_{k}"',
                f'NamedMaterial "gantry_sign_{k}"',
                *mesh(*_quad((lane + 0.5) * lane_w, 5.6, zg - 0.2, 3.4, 2.0)),
            ]
        L.append("")
    if v.overpass:
        zo, xl, xr = v.overpass_s, layout.rail_left - 40.0, layout.rail_right + 40.0
        clear, depth, width = 5.3, 1.3, 12.0
        L.append('NamedMaterial "concrete"')
        parts = [box(0.5 * (xl + xr), clear, zo, xr - xl, depth, width)]
        for zz in (zo - width / 2 + 0.2, zo + width / 2 - 0.2):
            parts.append(box(0.5 * (xl + xr), clear + depth, zz, xr - xl, 1.0, 0.35))
        parts.append(box(median_x, 0.0, zo, 0.9, clear, 8.0))
        for x in (layout.rail_left - 2.5, layout.rail_right + 2.5):
            parts.append(box(x, -0.5, zo, 1.5, clear + 0.5, width))
        for x in np.concatenate(
            [np.arange(layout.rail_left - 16, xl, -14.0), np.arange(layout.rail_right + 16, xr, 14.0)]
        ):
            parts.append(box(float(x), -1.0, zo, 1.0, clear + 1.0, 6.0))
        L += mesh(np.concatenate([p for p, _ in parts]), np.concatenate([t + 8 * i for i, (_, t) in enumerate(parts)]))
        rp, rt = box(0.5 * (xl + xr), clear + depth, zo, xr - xl, 0.02, width - 0.8)
        L += ['NamedMaterial "asphalt"', *mesh(rp, rt), ""]
    return L
