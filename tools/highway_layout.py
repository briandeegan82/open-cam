#!/usr/bin/env python3
"""Procedural highway geometry for PBRT scene generation (pure geometry, no I/O).

Conventions:
- Units are meters, Y up. The road runs along -Z (camera looks toward -Z).
- pbrt's LookAt places world +X on the LEFT of the image, so driver-view
  semantics are: lane 0 = rightmost (slow) lane at low x, median on the
  high-x side. Lane i is centered at x = (i + 0.5) * lane_width.
- The paved slab spans x in [-shoulder_right, lanes * lane_width + shoulder_median]:
  wide right shoulder at x < 0 (image right), narrow median shoulder beyond
  the last lane (image left).
- Markings (US layout): solid white right edge at x = 0, solid yellow median
  edge at x = lanes * lane_width, dashed white separators between lanes.
- Small fixed Y offsets avoid z-fighting: terrain -0.02, road slab +0.001,
  lane markings +0.003.

All builders return `Mesh` (triangle soup) or transform lists for instancing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Y offsets (meters) chosen to avoid z-fighting at typical viewing distances.
Y_TERRAIN = -0.02
Y_ROAD = 0.001
Y_MARKING = 0.003

# MUTCD-style marking dimensions.
DASH_LENGTH_M = 3.0
DASH_GAP_M = 9.0
MARKING_WIDTH_M = 0.15


@dataclass(frozen=True)
class RoadSpec:
    lanes: int = 3
    lane_width: float = 3.7
    length: float = 400.0      # paved distance ahead of the camera (toward -Z)
    back: float = 20.0         # paved distance behind the camera (+Z)
    shoulder_right: float = 3.0    # wide right shoulder, x < 0 (image right)
    shoulder_median: float = 1.2   # narrow median shoulder, high x (image left)

    @property
    def x_min(self) -> float:
        """Right (low-x) paved edge."""
        return -self.shoulder_right

    @property
    def x_max(self) -> float:
        """Median (high-x) paved edge."""
        return self.lanes * self.lane_width + self.shoulder_median

    def lane_center(self, lane: int) -> float:
        return (lane + 0.5) * self.lane_width


@dataclass(frozen=True)
class CarPlacement:
    model: str
    lane: int
    s_m: float                 # distance ahead of camera along -Z
    paint: str
    lateral_offset_m: float = 0.0
    heading_deg: float = 0.0   # yaw about +Y; 0 = driving toward -Z


@dataclass
class Mesh:
    P: np.ndarray                        # (N, 3) float32
    indices: np.ndarray                  # (M, 3) int32
    uv: np.ndarray | None = field(default=None)  # (N, 2) float32 or None


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def quad(p0, p1, p2, p3, with_uv: bool = False) -> Mesh:
    """Quad from 4 corners (CCW), two triangles. UVs map p0..p3 to unit square."""
    P = np.asarray([p0, p1, p2, p3], dtype=np.float32)
    idx = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    uv = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32) if with_uv else None
    return Mesh(P, idx, uv)


def box(center, size) -> Mesh:
    """Axis-aligned box: center (3,), size (3,) full extents."""
    c = np.asarray(center, dtype=np.float32)
    h = np.asarray(size, dtype=np.float32) / 2.0
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                     dtype=np.float32)
    P = c + signs * h
    # 12 triangles over the 8 corners (indexed by the (sx,sy,sz) ordering above).
    idx = np.asarray([
        [0, 1, 3], [0, 3, 2],  # -x
        [4, 6, 7], [4, 7, 5],  # +x
        [0, 4, 5], [0, 5, 1],  # -y
        [2, 3, 7], [2, 7, 6],  # +y
        [0, 2, 6], [0, 6, 4],  # -z
        [1, 5, 7], [1, 7, 3],  # +z
    ], dtype=np.int32)
    return Mesh(P, idx)


def merge(meshes: list[Mesh]) -> Mesh:
    """Concatenate triangle soups (UVs kept only if every part has them)."""
    if not meshes:
        return Mesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32))
    parts_p, parts_i, parts_uv, off = [], [], [], 0
    all_uv = all(m.uv is not None for m in meshes)
    for m in meshes:
        parts_p.append(m.P)
        parts_i.append(m.indices + off)
        if all_uv:
            parts_uv.append(m.uv)
        off += len(m.P)
    return Mesh(np.concatenate(parts_p), np.concatenate(parts_i),
                np.concatenate(parts_uv) if all_uv else None)


def transformed(mesh: Mesh, translate=(0.0, 0.0, 0.0), yaw_deg: float = 0.0) -> Mesh:
    """Apply yaw-about-Y then translation to a copy of the mesh."""
    P = mesh.P.astype(np.float64)
    if yaw_deg:
        a = np.deg2rad(yaw_deg)
        ca, sa = np.cos(a), np.sin(a)
        rot = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
        P = P @ rot.T
    P = P + np.asarray(translate, dtype=np.float64)
    return Mesh(P.astype(np.float32), mesh.indices.copy(),
                None if mesh.uv is None else mesh.uv.copy())


# ---------------------------------------------------------------------------
# Road surfaces
# ---------------------------------------------------------------------------

def road_slab(spec: RoadSpec) -> Mesh:
    z0, z1 = spec.back, -spec.length
    return quad(
        [spec.x_min, Y_ROAD, z0], [spec.x_max, Y_ROAD, z0],
        [spec.x_max, Y_ROAD, z1], [spec.x_min, Y_ROAD, z1],
    )


def terrain_plane(spec: RoadSpec, margin: float = 250.0) -> Mesh:
    z0, z1 = spec.back + 30.0, -(spec.length + 50.0)
    return quad(
        [spec.x_min - margin, Y_TERRAIN, z0], [spec.x_max + margin, Y_TERRAIN, z0],
        [spec.x_max + margin, Y_TERRAIN, z1], [spec.x_min - margin, Y_TERRAIN, z1],
    )


def _line_quads(x_center: float, z_start: float, z_end: float,
                width: float, dash: float | None = None, gap: float = 0.0) -> list[Mesh]:
    """Marking quads along Z at fixed x; solid when dash is None."""
    xh = width / 2.0
    out = []
    if dash is None:
        segs = [(z_start, z_end)]
    else:
        segs, z = [], z_start
        while z > z_end:
            z_seg_end = max(z - dash, z_end)
            segs.append((z, z_seg_end))
            z -= dash + gap
    for za, zb in segs:
        out.append(quad(
            [x_center - xh, Y_MARKING, za], [x_center + xh, Y_MARKING, za],
            [x_center + xh, Y_MARKING, zb], [x_center - xh, Y_MARKING, zb],
        ))
    return out


def lane_marking_meshes(spec: RoadSpec,
                        dash_m: float = DASH_LENGTH_M,
                        gap_m: float = DASH_GAP_M,
                        width_m: float = MARKING_WIDTH_M) -> dict[str, Mesh]:
    """Batched marking geometry: {"white": Mesh, "yellow": Mesh}.

    US layout: solid white right edge (x=0), solid yellow median edge
    (x = lanes * lane_width), dashed white separators between lanes.
    """
    z0, z1 = spec.back, -spec.length
    white = _line_quads(0.0, z0, z1, width_m)                            # right edge (solid white)
    yellow = _line_quads(spec.lanes * spec.lane_width, z0, z1, width_m)  # median edge (solid yellow)
    for i in range(1, spec.lanes):
        white += _line_quads(i * spec.lane_width, z0, z1, width_m, dash=dash_m, gap=gap_m)
    return {"white": merge(white), "yellow": merge(yellow)}


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------

# Simplified W-beam cross-section: (lateral_offset, height) polyline, outward folds.
_WBEAM_PROFILE = [(0.00, 0.395), (0.04, 0.475), (0.00, 0.550), (0.04, 0.625), (0.00, 0.710)]
GUARDRAIL_POST_SIZE = (0.15, 0.75, 0.10)


def guardrail_meshes(spec: RoadSpec, post_spacing_m: float = 4.0,
                     offset_m: float = 0.5) -> tuple[Mesh, list[np.ndarray]]:
    """Returns (rail ribbon mesh for both sides, post center translations)."""
    z0, z1 = spec.back, -spec.length
    rails: list[Mesh] = []
    posts: list[np.ndarray] = []
    for x_edge, outward in ((spec.x_min - offset_m, -1.0), (spec.x_max + offset_m, +1.0)):
        prof = [(x_edge + outward * dx, y) for dx, y in _WBEAM_PROFILE]
        for (xa, ya), (xb, yb) in zip(prof[:-1], prof[1:]):
            rails.append(quad([xa, ya, z0], [xb, yb, z0], [xb, yb, z1], [xa, ya, z1]))
        z = z0
        while z > z1:
            posts.append(np.array([x_edge + outward * 0.06, GUARDRAIL_POST_SIZE[1] / 2.0, z]))
            z -= post_spacing_m
    return merge(rails), posts


# ---------------------------------------------------------------------------
# Signs
# ---------------------------------------------------------------------------

SIGN_DEFS = {
    # face_w, face_h, bottom_height, n_posts
    "speed_limit": (0.9, 1.2, 2.1, 1),
    "exit": (3.6, 1.8, 2.2, 2),
}


def sign_meshes(sign_type: str, distance_m: float, spec: RoadSpec,
                lateral_offset_m: float = 1.5) -> list[tuple[str, Mesh]]:
    """Roadside sign at `distance_m` ahead on the right shoulder verge
    (low-x side, image right).

    Returns (slot, mesh) pairs; slots: "face" (UV-mapped, toward camera),
    "back", "post". Faces point toward +Z (the camera).
    """
    if sign_type not in SIGN_DEFS:
        raise ValueError(f"unknown sign type '{sign_type}'; known: {sorted(SIGN_DEFS)}")
    fw, fh, hb, n_posts = SIGN_DEFS[sign_type]
    xc = spec.x_min - lateral_offset_m - fw / 2.0
    z = -distance_m
    y0, y1 = hb, hb + fh
    xl, xr = xc - fw / 2.0, xc + fw / 2.0

    # Face UV order: u=0 at high x (image-left, since image-left = +X), so a
    # legend texture reads left-to-right from the camera; v=0 at the bottom.
    face = quad([xr, y0, z], [xl, y0, z], [xl, y1, z], [xr, y1, z], with_uv=True)
    back = quad([xr, y0, z - 0.01], [xl, y0, z - 0.01], [xl, y1, z - 0.01], [xr, y1, z - 0.01])

    posts: list[Mesh] = []
    post_xs = [xc] if n_posts == 1 else [xl + 0.3, xr - 0.3]
    for px in post_xs:
        posts.append(box([px, y1 / 2.0, z - 0.06], [0.1, y1, 0.08]))
    return [("face", face), ("back", back), ("post", merge(posts))]


# ---------------------------------------------------------------------------
# Cars
# ---------------------------------------------------------------------------

def parse_car_spec(s: str) -> list[CarPlacement]:
    """Parse "model:lane:distance_m:paint,..." into placements."""
    out = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 4:
            raise ValueError(f"bad car spec '{item}' (want model:lane:distance:paint)")
        out.append(CarPlacement(model=parts[0], lane=int(parts[1]),
                                s_m=float(parts[2]), paint=parts[3]))
    return out


def place_cars_random(spec: RoadSpec, n: int, rng: np.random.Generator,
                      paints: tuple[str, ...], models: tuple[str, ...] = ("sedan",),
                      min_headway_m: float = 25.0, s_min: float = 15.0,
                      s_max: float | None = None) -> list[CarPlacement]:
    """Random lane/spacing/paint placement with per-lane minimum headway."""
    s_max = s_max if s_max is not None else spec.length * 0.8
    placements: list[CarPlacement] = []
    lane_positions: dict[int, list[float]] = {i: [] for i in range(spec.lanes)}
    attempts = 0
    while len(placements) < n and attempts < n * 50:
        attempts += 1
        lane = int(rng.integers(0, spec.lanes))
        s = float(rng.uniform(s_min, s_max))
        if any(abs(s - other) < min_headway_m for other in lane_positions[lane]):
            continue
        lane_positions[lane].append(s)
        placements.append(CarPlacement(
            model=str(rng.choice(models)), lane=lane, s_m=s,
            paint=str(rng.choice(paints)),
            lateral_offset_m=float(rng.uniform(-0.15, 0.15)),
        ))
    return sorted(placements, key=lambda p: p.s_m)


def procedural_car_meshes(length_m: float = 4.5, width_m: float = 1.82,
                          ) -> dict[str, Mesh]:
    """Low-poly fallback car at origin, forward = -Z, wheels on y=0.

    Slots match fetched-asset semantics: paint, glass, tire, trim.
    """
    L, W = length_m, width_m
    body_y0, body_y1 = 0.30, 0.95         # rocker to beltline
    cabin_y1 = 1.45                        # roof
    cabin_z0, cabin_z1 = -L * 0.35, L * 0.18   # windshield to rear window (z fractions)

    paint = merge([
        box([0, (body_y0 + body_y1) / 2, 0], [W, body_y1 - body_y0, L]),           # body
        box([0, (body_y1 + cabin_y1) / 2, (cabin_z0 + cabin_z1) / 2],
            [W * 0.86, cabin_y1 - body_y1, cabin_z1 - cabin_z0]),                  # cabin
    ])
    # Glass: thin panels proud of the cabin (front/rear/side bands).
    gy0, gy1 = body_y1 + 0.06, cabin_y1 - 0.08
    gh, gyc = gy1 - gy0, (gy0 + gy1) / 2
    glass = merge([
        box([0, gyc, cabin_z0 - 0.015], [W * 0.80, gh, 0.03]),                     # windshield
        box([0, gyc, cabin_z1 + 0.015], [W * 0.80, gh, 0.03]),                     # rear window
        box([-W * 0.435, gyc, (cabin_z0 + cabin_z1) / 2],
            [0.03, gh, (cabin_z1 - cabin_z0) * 0.85]),                             # left band
        box([W * 0.435, gyc, (cabin_z0 + cabin_z1) / 2],
            [0.03, gh, (cabin_z1 - cabin_z0) * 0.85]),                             # right band
    ])
    wheel_r, wheel_w = 0.33, 0.22
    tire = merge([
        box([sx * (W / 2 - wheel_w / 2 + 0.02), wheel_r, sz * L * 0.31],
            [wheel_w, wheel_r * 2, wheel_r * 2])
        for sx in (-1, 1) for sz in (-1, 1)
    ])
    trim = merge([
        box([0, 0.55, -L / 2 - 0.02], [W * 0.9, 0.18, 0.04]),                      # front bumper strip
        box([0, 0.55, L / 2 + 0.02], [W * 0.9, 0.18, 0.04]),                       # rear bumper strip
    ])
    return {"paint": paint, "glass": glass, "tire": tire, "trim": trim}


def car_world_transform(p: CarPlacement, spec: RoadSpec) -> tuple[float, float, float, float]:
    """(tx, ty, tz, yaw_deg) placing a normalized car (origin, fwd=-Z, y=0) on the road."""
    return (spec.lane_center(p.lane) + p.lateral_offset_m, Y_ROAD, -p.s_m, p.heading_deg)


# ---------------------------------------------------------------------------
# PBRT emission
# ---------------------------------------------------------------------------

def emit_trianglemesh(mesh: Mesh, indent: str = "    ") -> list[str]:
    """Emit Shape "trianglemesh" lines (P, indices, and uv when present)."""
    p_flat = " ".join(f"{v:.4f}" for v in mesh.P.reshape(-1))
    i_flat = " ".join(str(int(v)) for v in mesh.indices.reshape(-1))
    lines = [
        f'{indent}Shape "trianglemesh"',
        f'{indent}    "point3 P" [ {p_flat} ]',
        f'{indent}    "integer indices" [ {i_flat} ]',
    ]
    if mesh.uv is not None:
        uv_flat = " ".join(f"{v:.4f}" for v in mesh.uv.reshape(-1))
        lines.append(f'{indent}    "point2 uv" [ {uv_flat} ]')
    return lines
