"""Rear lamp clusters for traffic vehicles at night: lens, light guides / bulbs, stop lamps and CHMSL.

Each lamp follows the car body: a grid over the lamp outline is ray-cast from behind onto the car asset's
meshes (pbrt-v4 PLY, car frame of ``highway_night.vehicle_light_lines``), and the lens and emitters are
bilinear meshes a few millimetres proud of the hit surface. Assets without geometry (proxies, the GLB heavy
vehicles) use their flat bounding-box rear.

Photometry is set per function at H-V (straight behind): tail R1 6 cd, stop S1 80 cd per side and centre
high-mounted stop lamp S3 40 cd (UN ECE R7 ranges 4-17, 60-185 and 25-80 cd); optional rear fog lamp
200 cd (R38 150-300 cd), one on the driver's (left, -x) side as R48 requires for a single lamp. Emitters are Lambertian, so
L = I / A_projected. LED lamps light separate tail and stop elements; incandescent lamps light the whole
chamber (dual filament) with a bulb hot spot behind a red filter (2856 K x red cut-on).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_ROOT = REPO / "scenes" / "assets" / "highway"

TAIL_CD, STOP_CD, CHMSL_CD = 6.0, 80.0, 40.0
FOG_CD = 200.0  # rear fog lamp, UN ECE R38: 150-300 cd at H-V
HOTSPOT_FRACTION = 0.5  # incandescent: share of the intensity from the bulb image
LENS_OFFSET_M, EMITTER_OFFSET_M = 0.002, 0.004
CELL_M = 0.01

_PLY_TYPES = {
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
    "uchar": "u1",
    "uint8": "u1",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
}


def read_ply_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Positions (N, 3) and triangles (M, 3) of a binary little-endian PLY (tri/quad faces)."""
    data = Path(path).read_bytes()
    end = data.index(b"end_header") + len(b"end_header")
    end += 2 if data[end : end + 2] == b"\r\n" else 1
    nv = nf = 0
    vprops, fprop, elem = [], None, None
    for ln in data[:end].decode("latin1").split("\n"):
        t = ln.split()
        if t[:1] == ["element"]:
            elem = t[1]
            if elem == "vertex":
                nv = int(t[2])
            elif elem == "face":
                nf = int(t[2])
        elif t[:1] == ["property"] and elem == "vertex":
            vprops.append((t[2], _PLY_TYPES[t[1]]))
        elif t[:2] == ["property", "list"] and elem == "face":
            fprop = (_PLY_TYPES[t[2]], _PLY_TYPES[t[3]])
    vdt = np.dtype(vprops)
    v = np.frombuffer(data, dtype=vdt, count=nv, offset=end)
    pos = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64)
    off = end + nv * vdt.itemsize
    if not nf or fprop is None:
        return pos, np.zeros((0, 3), dtype=np.int64)
    cdt, idt = np.dtype(fprop[0]), np.dtype(fprop[1])
    k = int(np.frombuffer(data, dtype=cdt, count=1, offset=off)[0])
    rec = np.dtype([("n", cdt), ("i", idt, (k,))])
    f = np.frombuffer(data, dtype=rec, count=nf, offset=off)
    if not (f["n"] == k).all():
        raise ValueError(f"{path}: mixed face sizes not supported")
    idx = f["i"].astype(np.int64)
    tri = idx if k == 3 else np.concatenate([idx[:, [0, 1, 2]], idx[:, [0, 2, 3]]])
    ok = np.isfinite(pos).all(1) & (np.abs(pos) < 100).all(1)
    tri = tri[ok[tri].all(1)]
    return pos, tri


@dataclass
class RearSurface:
    """Car-frame triangles (x lateral, y up, z forward; rear at -length/2) with material ids."""

    tri: np.ndarray  # (M, 3, 3)
    mat: np.ndarray  # (M,) index into materials
    materials: list[str]

    _bins: dict | None = None
    BIN_M = 0.02

    def _index(self) -> dict:
        """Uniform x-y bins of triangle ids (CSR) for the ray casts."""
        if self._bins is None:
            lo, hi = self.tri[:, :, :2].min(1), self.tri[:, :, :2].max(1)
            o = lo.min(0)
            b0 = np.floor((lo - o) / self.BIN_M).astype(np.int64)
            b1 = np.floor((hi - o) / self.BIN_M).astype(np.int64)
            nb = b1.max(0) + 1
            span = b1 - b0
            ids, keys = [], []
            for dx in range(int(span[:, 0].max()) + 1):
                for dy in range(int(span[:, 1].max()) + 1):
                    k = np.nonzero((span[:, 0] >= dx) & (span[:, 1] >= dy))[0]
                    ids.append(k)
                    keys.append((b0[k, 0] + dx) * nb[1] + b0[k, 1] + dy)
            ids, keys = np.concatenate(ids), np.concatenate(keys)
            order = np.argsort(keys, kind="stable")
            ids, keys = ids[order], keys[order]
            starts = np.searchsorted(keys, np.arange(nb[0] * nb[1] + 1))
            self._bins = {"o": o, "nb": nb, "ids": ids, "starts": starts}
        return self._bins

    def cast(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """First hit of rays travelling +z from behind at (x, y): z (nan on miss) and material id (-1)."""
        x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
        zh, mh = np.full(x.size, np.nan), np.full(x.size, -1)
        ix = self._index()
        bx = np.floor((x - ix["o"][0]) / self.BIN_M).astype(np.int64)
        by = np.floor((y - ix["o"][1]) / self.BIN_M).astype(np.int64)
        valid = (bx >= 0) & (by >= 0) & (bx < ix["nb"][0]) & (by < ix["nb"][1])
        for i in np.nonzero(valid)[0]:
            k = bx[i] * ix["nb"][1] + by[i]
            cand = ix["ids"][ix["starts"][k] : ix["starts"][k + 1]]
            if not cand.size:
                continue
            t = self.tri[cand]
            a, e1, e2 = t[:, 0], t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
            det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
            ok = np.abs(det) > 1e-14
            a, e1, e2, det, cand = a[ok], e1[ok], e2[ok], det[ok], cand[ok]
            px, py = x[i] - a[:, 0], y[i] - a[:, 1]
            u = (px * e2[:, 1] - py * e2[:, 0]) / det
            v = (e1[:, 0] * py - e1[:, 1] * px) / det
            inside = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
            if inside.any():
                z = np.where(inside, a[:, 2] + u * e1[:, 2] + v * e2[:, 2], np.inf)
                j = int(np.argmin(z))
                zh[i], mh[i] = z[j], self.mat[cand[j]]
        return zh, mh


@lru_cache(maxsize=16)
def rear_surface(model: str, asset_root: str) -> RearSurface | None:
    """Ray-cast geometry of a prepared ``car_pbrt`` asset in the car frame, or None (no geometry)."""
    p = Path(asset_root) / "prepared" / model / "prepared.json"
    if not p.is_file():
        return None
    info = json.loads(p.read_text())
    if info.get("asset", {}).get("kind") != "car_pbrt" or "length_m" not in info.get("asset", {}):
        return None
    raw = (p.parent / info["raw_dir"]).resolve()
    lo, hi = np.array(info["bbox_min"]), np.array(info["bbox_max"])
    s = info["asset"]["length_m"] / float(hi[0] - lo[0])
    cx, cz = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[2] + hi[2])
    mats = sorted({sh["material"] for sh in info["shapes"]})
    tris, mids = [], []
    for sh in info["shapes"]:
        pos, tri = read_ply_mesh(raw / sh["ply"])
        if not len(tri):
            continue
        # car_include (Scale, Translate) then Rotate 90 about y: X = s (z - cz), Y = s (y - y0), Z = -s (x - cx)
        q = np.stack([s * (pos[:, 2] - cz), s * (pos[:, 1] - lo[1]), -s * (pos[:, 0] - cx)], -1)
        tris.append(q[tri])
        mids.append(np.full(len(tri), mats.index(sh["material"])))
    if not tris:
        return None
    tri, mid = np.concatenate(tris), np.concatenate(mids)
    zmin = tri[..., 2].min(1)
    keep = zmin < zmin.min() + 2.0  # rays come from behind: only the rear 2 m can be hit first
    return RearSurface(tri[keep], mid[keep], mats)


def _box_tris(x0, x1, y0, y1, z0, z1) -> np.ndarray:
    p = np.array([(x, y, z) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)], dtype=np.float64)
    quads = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    return np.array([p[[q[0], q[1], q[2]]] for q in quads] + [p[[q[0], q[2], q[3]]] for q in quads])


@lru_cache(maxsize=1)
def proxy_surface() -> RearSurface:
    """The box-and-glass stand-in of ``build_highway_scene.proxy_car``."""
    body, glass = _box_tris(-0.9, 0.9, 0.3, 1.05, -2.3, 2.3), _box_tris(-0.8, 0.8, 1.05, 1.5, -1.4, 1.0)
    return RearSurface(np.concatenate([body, glass]), np.repeat([0, 1], [len(body), len(glass)]), ["paint", "glass"])


@dataclass(frozen=True)
class LampLayout:
    """Rear lamp cluster of one vehicle type (car frame, metres, mirrored to both sides).

    ``region`` = (|x|0, |x|1, y0, y1); with ``from_edge`` the |x| values are distances in from the side
    (half width - value). Lens cells are the region cells whose ray hits one of ``lens_materials`` (any
    surface when empty). ``chambers`` splits the lens along ``chamber_axis`` (outer->inner or top->bottom)
    into red / amber / clear sections; only red emits. ``chmsl`` = (half width, y0, y1) centre stop lamp.
    ``fog`` = rear fog lamp region (same convention as ``region``, left side only); None = not fitted.
    """

    tech: str
    region: tuple[float, float, float, float]
    lens_materials: tuple[str, ...] = ()
    chmsl: tuple[float, float, float] | None = None
    from_edge: bool = False
    chambers: tuple[str, ...] = ("red",)
    chamber_axis: str = "x"
    fog: tuple[float, float, float, float] | None = None


_TRUCK = LampLayout(
    "incandescent",
    (0.50, 0.06, 0.80, 0.95),
    from_edge=True,
    chambers=("amber", "red", "clear"),
    fog=(0.85, 0.62, 0.80, 0.95),
)
LAYOUTS = {
    # BMW M6: LED contour light guide around the asset's own lamp glass, inner stop area, CHMSL atop the glass.
    "car_bmw_m6": LampLayout(
        "led", (0.40, 0.92, 0.66, 0.94), ("HeadLightGlass",), chmsl=(0.25, 1.265, 1.29), fog=(0.45, 0.70, 0.44, 0.50)
    ),
    # Pontiac GTO 67: incandescent slot lamps behind the chrome louvres; vintage car: round fender lamps.
    # Neither has a rear fog lamp (not fitted to cars of their era).
    "car_pontiac_gto": LampLayout("incandescent", (0.22, 0.80, 0.58, 0.79), ("Chrome",)),
    "car_vintage": LampLayout("incandescent", (0.55, 0.92, 0.42, 0.72), ("Seats",)),
    "proxy": LampLayout("led", (0.45, 0.85, 0.80, 0.97), chmsl=(0.25, 1.43, 1.46), fog=(0.45, 0.75, 0.38, 0.44)),
    "van_sprinter": LampLayout(
        "incandescent",
        (0.17, 0.03, 0.62, 1.12),
        from_edge=True,
        chambers=("amber", "red", "clear"),
        chamber_axis="y",
        fog=(0.17, 0.03, 0.40, 0.50),
    ),
    "bus_town": LampLayout(
        "led",
        (0.20, 0.05, 0.55, 1.15),
        from_edge=True,
        chambers=("amber", "red", "clear"),
        chamber_axis="y",
        fog=(0.45, 0.30, 0.55, 0.70),
    ),
    "truck_box": _TRUCK,
    "trailer_box": _TRUCK,
    "truck_daf_cf_tractor": LampLayout(
        "incandescent",
        (0.55, 0.10, 0.62, 0.76),
        from_edge=True,
        chambers=("amber", "red", "clear"),
        fog=(0.85, 0.67, 0.62, 0.76),
    ),
}

LENS_SPD = {
    "red": "spd/taillamp_lens_red.spd",
    "amber": "spd/taillamp_lens_amber.spd",
    "clear": "spd/taillamp_lens_clear.spd",
}
EMITTER_SPD = {"led": "spd/lamp_red_led.spd", "incandescent": "spd/lamp_red_incandescent.spd"}


def _cutoff(wl: np.ndarray, edge_nm: float, width_nm: float = 7.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(np.asarray(wl, dtype=np.float64) - edge_nm) / width_nm))


def red_incandescent_spectrum(wl: np.ndarray) -> np.ndarray:
    """2856 K filament behind a red lens (long-pass at 610 nm)."""
    from highway_night import planck

    return planck(wl, 2856.0) * _cutoff(wl, 610.0)


def lens_reflectances(wl: np.ndarray) -> dict[str, np.ndarray]:
    """Unlit lens over its reflector/optic (double pass through the filter); nominal, not measured."""
    return {
        "red": 0.03 + 0.42 * _cutoff(wl, 605.0, 8.0),
        "amber": 0.03 + 0.50 * _cutoff(wl, 575.0, 8.0),
        "clear": np.full(np.shape(wl), 0.25),
    }


def _f(v: float) -> str:
    return f"{v:.6g}"


@dataclass
class _Grid:
    """Cells of one lamp on a CELL_M grid: corner positions (ny+1, nx+1, 3), cell mask (ny, nx)."""

    corners: np.ndarray
    mask: np.ndarray
    surface: str

    def area(self, cells: np.ndarray) -> float:
        c = self.corners
        dx = np.abs(c[:-1, 1:, 0] - c[:-1, :-1, 0])
        dy = np.abs(c[1:, :-1, 1] - c[:-1, :-1, 1])
        return float((dx * dy)[cells].sum())


def _grid(
    surf: RearSurface | None, x0: float, x1: float, y0: float, y1: float, rear_z: float, lens: tuple[str, ...]
) -> _Grid:
    nx, ny = max(1, round(abs(x1 - x0) / CELL_M)), max(1, round((y1 - y0) / CELL_M))
    xs, ys = np.linspace(x0, x1, nx + 1), np.linspace(y0, y1, ny + 1)
    gx, gy = np.meshgrid(xs, ys)
    if surf is None:
        z = np.full(gx.shape, rear_z)
        return _Grid(np.stack([gx, gy, z], -1), np.ones((ny, nx), bool), "flat")
    z, _ = surf.cast(gx, gy)
    z = z.reshape(gx.shape)
    cx, cy = 0.5 * (gx[:-1, :-1] + gx[1:, 1:]), 0.5 * (gy[:-1, :-1] + gy[1:, 1:])
    zc, mc = surf.cast(cx, cy)
    quad = np.stack([z[:-1, :-1], z[:-1, 1:], z[1:, :-1], z[1:, 1:]])
    finite = np.isfinite(quad).all(0)
    q = np.where(finite, quad, 0.0)
    ok = finite & (q.max(0) - q.min(0) < 0.02) & np.isfinite(zc.reshape(cx.shape))
    if lens:
        ids = [surf.materials.index(m) for m in lens if m in surf.materials]
        ok &= np.isin(mc.reshape(cx.shape), ids)
    return _Grid(np.stack([gx, gy, np.nan_to_num(z, nan=rear_z)], -1), ok, "mesh")


def _erode(m: np.ndarray, n: int) -> np.ndarray:
    from scipy.ndimage import binary_erosion

    return binary_erosion(m, iterations=n, border_value=0) if n else m.copy()


def _patches(g: _Grid, cells: np.ndarray, offset: float) -> str | None:
    """bilinearmesh over ``cells``; vertex order as ``highway_night._quad_light`` so it emits towards -z."""
    jj, ii = np.nonzero(cells)
    if not len(jj):
        return None
    c = g.corners.copy()
    c[..., 2] -= offset
    # p00 = (x1, y0), p10 = (x0, y0), p01 = (x1, y1), p11 = (x0, y1) for x0 < x1 in the side's own x order
    flip = c[0, -1, 0] < c[0, 0, 0]
    a, b = (ii, ii + 1) if flip else (ii + 1, ii)
    quads = np.stack([c[jj, a], c[jj, b], c[jj + 1, a], c[jj + 1, b]], 1)
    pts = " ".join(_f(v) for v in quads.reshape(-1))
    idx = " ".join(str(v) for v in range(4 * len(jj)))
    return f'    Shape "bilinearmesh" "point3 P" [{pts}] "integer indices" [{idx}]'


def _chambers(g: _Grid, layout: LampLayout, outer_is_high: bool) -> dict[str, np.ndarray]:
    kinds = layout.chambers
    if len(kinds) == 1:
        return {kinds[0]: g.mask}
    ny, nx = g.mask.shape
    if layout.chamber_axis == "x":
        pos = np.broadcast_to(np.arange(nx) / nx, (ny, nx))
        pos = pos if not outer_is_high else 1 - pos  # 0 = outer
    else:
        pos = np.broadcast_to((ny - 1 - np.arange(ny))[:, None] / ny, (ny, nx))  # 0 = top
    sec = np.minimum((pos * len(kinds)).astype(int), len(kinds) - 1)
    out: dict[str, np.ndarray] = {}
    for k, kind in enumerate(kinds):
        out[kind] = out.get(kind, np.zeros_like(g.mask)) | (g.mask & (sec == k))
    return out


def _incandescent_bins(g: _Grid, cells: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """Cells grouped into rings around each lens component's centroid, with relative luminance weights."""
    from scipy.ndimage import label

    cx = 0.5 * (g.corners[:-1, :-1, 0] + g.corners[1:, 1:, 0])
    cy = 0.5 * (g.corners[:-1, :-1, 1] + g.corners[1:, 1:, 1])
    lab, n = label(cells)
    r = np.full(cells.shape, np.inf)
    for k in range(1, n + 1):
        m = lab == k
        r[m] = np.hypot(cx[m] - cx[m].mean(), cy[m] - cy[m].mean())
    edges = (0.0, 0.015, 0.03, 0.045, np.inf)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = cells & (r >= lo) & (r < hi)
        if m.any():
            w = 1.0 + 4.0 * np.exp(-0.5 * (r[m] / 0.02) ** 2)
            out.append((m, float(w.mean())))
    return out


def _lamp_lines(g: _Grid, layout: LampLayout, braking: bool, outer_is_high: bool) -> tuple[list[str], dict]:
    """Lens + emitters of one side; luminances from the per-function intensities (Lambertian, I = L A)."""
    out: list[str] = []
    sections = _chambers(g, layout, outer_is_high)
    red = sections.get("red", np.zeros_like(g.mask))
    groups: list[tuple[np.ndarray, float]] = []  # (cells, luminance cd/m^2)
    if layout.tech == "led":
        inner = _erode(red, 2)
        band = red & ~inner
        if not inner.any() or not band.any():
            groups.append((red, (TAIL_CD + (STOP_CD if braking else 0.0)) / max(g.area(red), 1e-9)))
        else:
            groups.append((band, TAIL_CD / g.area(band)))
            if braking:
                groups.append((inner, STOP_CD / g.area(inner)))
    elif red.any():
        bins = _incandescent_bins(g, red)
        intensity = TAIL_CD + (STOP_CD if braking else 0.0)
        norm = sum(w * g.area(m) for m, w in bins)
        groups += [(m, intensity * w / norm) for m, w in bins]
    for kind, cells in sections.items():
        sh = _patches(g, cells, LENS_OFFSET_M)
        if sh:
            out += ["AttributeBegin", f'    Material "coateddiffuse" "spectrum reflectance" "{LENS_SPD[kind]}"'
                    ' "float roughness" [0]', sh, "AttributeEnd"]  # fmt: skip
    spd = EMITTER_SPD[layout.tech]
    for cells, lum in groups:
        sh = _patches(g, cells, EMITTER_OFFSET_M)
        if sh:
            out += [
                "AttributeBegin",
                f'    Material "coateddiffuse" "spectrum reflectance" "{LENS_SPD["red"]}" "float roughness" [0]',
                f'    AreaLightSource "diffuse" "spectrum L" "{spd}" "float scale" [{_f(lum)}]',
                sh,
                "AttributeEnd",
            ]
    emitted = sum(lum * g.area(c) for c, lum in groups)
    return out, {
        "lens_area_m2": round(g.area(g.mask), 5),
        "emitting_area_m2": round(sum(g.area(c) for c, _ in groups), 5),
        "intensity_cd": round(emitted, 3),
        "peak_luminance_cd_m2": round(max((lum for _, lum in groups), default=0.0), 1),
    }


def _fog_lamp(
    surf: RearSurface | None, layout: LampLayout, half_w: float, rear_z: float
) -> tuple[list[str], dict | None]:
    """Single rear fog lamp on the left (-x) side, or nothing if the layout has none."""
    if layout.fog is None:
        return [], None
    ax0, ax1, y0, y1 = layout.fog
    if layout.from_edge:
        ax0, ax1 = half_w - ax0, half_w - ax1
    g = _grid(surf, -ax0, -ax1, y0, y1, rear_z, ())
    if not g.mask.any():
        g = _grid(None, -ax0, -ax1, y0, y1, rear_z, ())
    if layout.tech == "led":
        groups = [(g.mask, FOG_CD / g.area(g.mask))]
    else:
        bins = _incandescent_bins(g, g.mask)
        norm = sum(w * g.area(m) for m, w in bins)
        groups = [(m, FOG_CD * w / norm) for m, w in bins]
    out = ["AttributeBegin", f'    Material "coateddiffuse" "spectrum reflectance" "{LENS_SPD["red"]}"'
           ' "float roughness" [0]', _patches(g, g.mask, LENS_OFFSET_M), "AttributeEnd"]  # fmt: skip
    for cells, lum in groups:
        out += [
            "AttributeBegin",
            f'    Material "coateddiffuse" "spectrum reflectance" "{LENS_SPD["red"]}" "float roughness" [0]',
            f'    AreaLightSource "diffuse" "spectrum L" "{EMITTER_SPD[layout.tech]}" "float scale" [{_f(lum)}]',
            _patches(g, cells, EMITTER_OFFSET_M),
            "AttributeEnd",
        ]
    return out, {
        "side": "left",
        "intensity_cd": round(sum(lum * g.area(c) for c, lum in groups), 3),
        "area_m2": round(g.area(g.mask), 5),
        "peak_luminance_cd_m2": round(max(lum for _, lum in groups), 1),
        "surface": g.surface,
    }


def rear_lamp_lines(
    car: dict, braking: bool, asset_root: Path | str | None = None, rear_fog: bool = False
) -> tuple[list[str], dict]:
    """pbrt lines (car frame) and manifest entry for one vehicle's rear lamps (``rear_fog``: fog lamp lit)."""
    model = car.get("model", "proxy")
    layout = LAYOUTS.get(model, LAYOUTS["proxy"])
    half_w, half_l = car["width_m"] / 2, car["length_m"] / 2
    rear_z = -half_l - 0.015
    if model == "proxy" or model not in LAYOUTS:
        surf = proxy_surface() if layout is LAYOUTS["proxy"] and model in ("proxy",) else None
    else:
        surf = rear_surface(model, str(asset_root or DEFAULT_ASSET_ROOT))
    lens = layout.lens_materials if surf is not None and surf.materials != ["paint", "glass"] else ()
    ax0, ax1, y0, y1 = layout.region
    if layout.from_edge:
        ax0, ax1 = half_w - ax0, half_w - ax1
    lines = ["# rear lamps (tools/highway_taillamps.py)"]
    sides = []
    for sx in (-1, 1):
        g = _grid(surf, sx * ax0, sx * ax1, y0, y1, rear_z, lens)
        if not g.mask.any():  # asset geometry does not match the layout: flat lamp at the bounding-box rear
            g = _grid(None, sx * ax0, sx * ax1, y0, y1, rear_z, ())
        outer_is_high = abs(g.corners[0, -1, 0]) > abs(g.corners[0, 0, 0])
        ln, meta = _lamp_lines(g, layout, braking, outer_is_high)
        lines += ln
        sides.append(dict(meta, surface=g.surface))
    chmsl = None
    if layout.chmsl:
        hw, cy0, cy1 = layout.chmsl
        g = _grid(surf, hw, -hw, cy0, cy1, rear_z, ())
        if not g.mask.any():
            g = _grid(None, hw, -hw, cy0, cy1, rear_z, ())
        sh_lens, sh_emit = _patches(g, g.mask, LENS_OFFSET_M), _patches(g, g.mask, EMITTER_OFFSET_M)
        lines += ["AttributeBegin", f'    Material "coateddiffuse" "spectrum reflectance" "{LENS_SPD["red"]}"'
                  ' "float roughness" [0]', sh_lens, "AttributeEnd"]  # fmt: skip
        lum = CHMSL_CD / g.area(g.mask) if braking else 0.0
        if braking:
            lines += [
                "AttributeBegin",
                f'    Material "coateddiffuse" "spectrum reflectance" "{LENS_SPD["red"]}" "float roughness" [0]',
                f'    AreaLightSource "diffuse" "spectrum L" "{EMITTER_SPD["led"]}" "float scale" [{_f(lum)}]',
                sh_emit,
                "AttributeEnd",
            ]
        chmsl = {"intensity_cd": CHMSL_CD if braking else 0.0, "area_m2": round(g.area(g.mask), 5)}
    fog_meta = None
    if rear_fog:
        fog_lines, fog_meta = _fog_lamp(surf, layout, half_w, rear_z)
        lines += fog_lines
    return lines, {
        "car": car["id"],
        "braking": braking,
        "layout": model if model in LAYOUTS else "proxy",
        "technology": layout.tech,
        "intensity_cd": sides[0]["intensity_cd"],
        "sides": sides,
        "chmsl": chmsl,
        "rear_fog": fog_meta,
    }
