"""Seeded road-surface ageing for the spectral highway scene (``build_highway_scene.py --road-wear``).

Everything lives on the *material* side of the flat road mesh, so geometry, lights and the manifest's
absolute illuminance are unchanged; only the reflectance field of the road changes (plus a few
centimetre-scale raised pavement markers, which are geometry).

* Anti-tiling: each ambientCG asphalt is re-synthesised into a 6 m periodic tile by variance-preserving
  blending of randomly offset/rotated copies on a cos^2 partition of unity (Heitz & Neyret 2018,
  "High-performance by-example noise using a histogram-preserving blending operator", without the
  histogram transform). Two different asphalts (A = ``--road``, B = Asphalt033) are mixed with a 2.5 m
  correlated noise mask, so no 2 m repeat is visible along the 1.5 km road.
* Wear maps (one 97.5 m x paved-width period, mapped through the road mesh uv, so they follow a warped alignment), all seeded:
  - tyre wheel paths at +-0.88 m from each lane centre (half of a 1.75 m track width), lateral wander
    sigma 0.32 m (MEPDG default traffic-wander SD 10 in = 0.254 m, NCHRP 1-37A 2004, convolved with tyre
    width); slow lanes weighted heaviest. Wheel paths are slightly darker and much smoother (polished
    aggregate, rubber/bitumen film; "polished aggregate" and "bleeding" in the LTPP Distress
    Identification Manual, FHWA-RD-03-031, 2003);
  - oil/drip stains as elongated blotches along lane centres (between the wheel paths);
  - transverse (thermal) and longitudinal (paving-joint, wheel-path) cracks: unsealed cracks 6-15 mm
    wide (LTPP low/moderate severity), sealed ones as 60-120 mm overbands of hot-poured rubberised
    sealant (ASTM D6690 type products, near-black and glossy when fresh);
  - patches (pothole, wheel-path strip and full-lane patches) with younger, darker binder and sealed edges.
* Markings: area-weighted (linear) spectral mixing of traffic paint and the underlying asphalt by a
  seeded wear coverage map (flaking, edge chipping, dash-to-dash variation); intact paint keeps a smoother
  coat (glass-bead sheen in daylight, AASHTO M247 drop-on beads), worn areas take the asphalt roughness.
* Raised pavement markers (MUTCD 2009 Sec. 3B.11-3B.14: 2N = 24.4 m spacing in broken-line gaps, colour
  matching the line, red back on lane lines of a divided road) as 100 x 100 x 18 mm geometry (ASTM D4280
  limits the height to ~20 mm), acrylic lens faces with a smooth coat.

Asphalt reflectance stays within the measured range: the analytic ``asphalt_aged`` spectrum (luminous
reflectance ~0.12, Herold et al. 2004) is only modulated by factors <= ~1 and replaced locally by
patch binder (~0.07) or sealant (~0.04); new asphalt is ~0.05 and aged asphalt ~0.10-0.18 (Pomerantz,
Akbari et al., LBNL cool-pavement reports, 2000-2003). The resulting area-weighted luminous reflectance
is recorded in the scene manifest (``road.wear.luminous_reflectance``).
"""

from __future__ import annotations

import math
import os
import zlib
from pathlib import Path

import numpy as np

# Per-level parameters. Spacings/frequencies are typical of an ageing (~10-15 year old) interstate
# surface; see the LTPP Distress Identification Manual for the distress definitions.
WEAR_LEVELS: dict[str, dict[str, float]] = {
    "light": dict(wheel=0.6, oil=0.5, crack_spacing=35.0, long_frac=0.15, seal_frac=0.85,
                  patches_per_km=12.0, mark_wear=0.10, rpm_missing=0.04),
    "moderate": dict(wheel=0.85, oil=0.8, crack_spacing=15.0, long_frac=0.35, seal_frac=0.7,
                     patches_per_km=30.0, mark_wear=0.25, rpm_missing=0.10),
    "heavy": dict(wheel=1.0, oil=1.0, crack_spacing=7.0, long_frac=0.6, seal_frac=0.55,
                  patches_per_km=70.0, mark_wear=0.45, rpm_missing=0.22),
}  # fmt: skip

WHEEL_OFFSET_M = 0.88
WHEEL_SIGMA_M = 0.32
TILE_M = 6.0  # anti-tiled asphalt tile
TILE_PX = 2000  # 3 mm texels
ROUGH_ASPHALT, ROUGH_POLISHED, ROUGH_BEADS = 0.35, 0.12, 0.15


def _lin(wl: np.ndarray, a: float, b: float) -> np.ndarray:
    return a + b * np.clip((wl - 400.0) / 300.0, 0.0, 1.0)


def _sig(wl: np.ndarray, lo: float, hi: float, edge: float, w: float) -> np.ndarray:
    return lo + (hi - lo) / (1.0 + np.exp(-(wl - edge) / w))


# Analytic reflectances [0-1] for the wear materials.
WEAR_SPECTRA = {
    # Patch binder: a few years old, between new (~0.05) and aged (~0.12) asphalt (Pomerantz et al.).
    "asphalt_patch": lambda wl: _lin(wl, 0.058, 0.025),
    # Hot-poured rubberised crack sealant, near-black and spectrally flat.
    "crack_sealant": lambda wl: _lin(wl, 0.038, 0.006),
    # RPM lenses: acrylic over metallised prisms; daytime diffuse appearance only (no retroreflection).
    "rpm_lens_clear": lambda wl: _lin(wl, 0.30, 0.03),
    "rpm_lens_red": lambda wl: _sig(wl, 0.03, 0.32, 605.0, 10.0),
    "rpm_lens_amber": lambda wl: _sig(wl, 0.03, 0.36, 565.0, 12.0),
}


def cie_y(wl: np.ndarray) -> np.ndarray:
    """CIE 1931 y-bar, multi-lobe Gaussian fit of Wyman, Sloan & Shirley (JCGT 2013)."""

    def g(mu: float, s1: float, s2: float) -> np.ndarray:
        return np.exp(-0.5 * ((wl - mu) / np.where(wl < mu, s1, s2)) ** 2)

    return 0.821 * g(568.8, 46.9, 40.5) + 0.286 * g(530.9, 16.3, 31.1)


def luminous_reflectance(wl: np.ndarray, r: np.ndarray) -> float:
    """Equal-energy luminous (CIE Y) reflectance of a spectral reflectance curve."""
    y = cie_y(wl)
    return float(np.sum(r * y) / np.sum(y))


def _smoothstep(a: float, b: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def periodic_noise(rng: np.random.Generator, shape, spacing, corr) -> np.ndarray:
    """Periodic Gaussian-correlated noise (zero mean, unit SD); corr = per-axis correlation length.

    Smooth fields are synthesised on a grid of ~corr/2 spacing and upsampled (periodic bilinear).
    """
    shape = tuple(int(n) for n in shape)
    cshape = tuple(max(4, min(n, math.ceil(n * d * 2.0 / c))) for n, d, c in zip(shape, spacing, corr))
    csp = tuple(d * n / cn for d, n, cn in zip(spacing, shape, cshape))
    w = rng.standard_normal(cshape)
    k2 = 0.0
    for ax, (n, d, c) in enumerate(zip(cshape, csp, corr)):
        f = np.fft.rfftfreq(n, d) if ax == len(shape) - 1 else np.fft.fftfreq(n, d)
        sh = [1] * len(shape)
        sh[ax] = f.size
        k2 = k2 + (f.reshape(sh) * c) ** 2
    axes = tuple(range(len(shape)))
    out = np.fft.irfftn(np.fft.rfftn(w, axes=axes) * np.exp(-2.0 * np.pi**2 * k2), s=cshape, axes=axes)
    if cshape != shape:
        from scipy.ndimage import zoom

        out = zoom(out, [n / cn for n, cn in zip(shape, cshape)], order=1, mode="grid-wrap", grid_mode=True)
    out -= out.mean()
    return (out / (out.std() + 1e-12)).astype(np.float32)


def write_float_exr(path: Path, a: np.ndarray) -> None:
    import OpenEXR

    path.parent.mkdir(parents=True, exist_ok=True)
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    with OpenEXR.File(header, {"Y": np.ascontiguousarray(a, dtype=np.float16)}) as f:
        f.write(str(path))


def _write_normal_png(path: Path, n: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.round((n + 1.0) * 127.5), 0, 255).astype(np.uint8)).save(path)


def linear_normal_map(src: Path, out_dir: Path) -> Path:
    """Copy a tangent-space normal map to PNG.

    pbrt-v4 decodes 8-bit JPEGs as sRGB even when a linear encoding is requested, which tilts every
    normal and makes the asphalt render ~7x too dark at grazing angles; PNGs are read linearly.
    """
    if src.suffix.lower() == ".png":
        return src
    dst = out_dir / f"{src.stem}.png"
    if not dst.is_file():
        from PIL import Image

        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.open(src).convert("RGB").save(dst)
    return dst


def _load_maps(info: dict, raw: Path, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(luminance, roughness, normal x, normal y) of an ambientCG texture resampled to n x n."""
    from PIL import Image

    def rd(key: str, mode: str) -> np.ndarray:
        im = Image.open(raw / info["maps"][key]).convert(mode).resize((n, n), Image.LANCZOS)
        return np.asarray(im, dtype=np.float64) / 255.0

    c = rd("color", "RGB")
    lum = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4) @ np.array([0.2126, 0.7152, 0.0722])
    rough = rd("roughness", "L") if info["maps"].get("roughness") else np.full((n, n), 0.5)
    nrm = rd("normal", "RGB") * 2.0 - 1.0
    return lum, rough, nrm[..., 0], nrm[..., 1]


def detile(chans: list[np.ndarray], n_out: int, cells: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Variance-preserving stochastic re-tiling of [lum, rough, nx, ny] into an n_out periodic tile."""
    n_src = chans[0].shape[0]
    c = n_out // cells
    d = np.arange(-c, c)
    w1 = np.cos(np.pi * (d + 0.5) / (2 * c)) ** 2
    wgt = np.outer(w1, w1).astype(np.float32)
    means = [float(ch.mean()) for ch in chans[:2]] + [0.0, 0.0]
    acc = [np.zeros((n_out, n_out), np.float32) for _ in chans]
    w2 = np.zeros((n_out, n_out), np.float32)
    for a in range(cells):
        for b in range(cells):
            r = int(rng.integers(4))
            oy, ox = rng.integers(n_src, size=2)
            dst = np.ix_((a * c + d) % n_out, (b * c + d) % n_out)
            src = np.ix_((d + oy) % n_src, (d + ox) % n_src)
            p = [np.rot90(ch, r)[src] - m for ch, m in zip(chans, means)]
            for _ in range(r):  # rotate the tangent-space vectors with the image (CCW: x,y -> -y,x)
                p[2], p[3] = -p[3], p[2]
            for k in range(4):
                acc[k][dst] += wgt * p[k]
            w2[dst] += wgt**2
    s = 1.0 / np.sqrt(w2)
    return [m + a * s for a, m in zip(acc, means)]


class RoadWear:
    """Seeded road ageing. ``geom`` (from build_highway_scene.Layout):

    x0, x1: paved extent; carriageways: [(xmin, xmax, lanes)] with lanes = [(centre x, traffic weight)];
    lane_lines: x of broken lines; period: longitudinal period of the wear maps [m].
    """

    def __init__(self, level: str, seed: int, out_dir: Path, geom: dict, texel_m: float = 0.02):
        self.level, self.seed, self.p = level, seed, WEAR_LEVELS[level]
        self.out_dir, self.geom, self.texel = Path(out_dir), geom, float(texel_m)
        self.tex_dir = self.out_dir / "textures" / "road_wear"
        self.rng = np.random.default_rng([seed, 0x40AD])
        self.tile_m = TILE_M
        self.summary: dict = {"level": level, "seed": seed, "texel_m": self.texel, "period_m": geom["period"]}

    def _rel(self, p: Path) -> str:
        return os.path.relpath(p, self.out_dir)

    # ------------------------------------------------------------------ wear maps
    def wear_maps(self) -> dict[str, np.ndarray]:
        g, p, rng = self.geom, self.p, self.rng
        P, x0, x1 = float(g["period"]), float(g["x0"]), float(g["x1"])
        H, W = max(8, round(P / self.texel)), max(8, round((x1 - x0) / self.texel))
        dz, dx = P / H, (x1 - x0) / W
        X = (x0 + (np.arange(W) + 0.5) * dx)[None, :].astype(np.float32)
        shp, sp = (H, W), (dz, dx)

        def wander(corr: float, amp: float) -> np.ndarray:
            return amp * periodic_noise(rng, (H, 2), (dz, 1.0), (corr, 1.0))[:, :1]

        lanes = [ln for cw in g["carriageways"] for ln in cw[2]]
        wheel = np.zeros(shp, np.float32)
        oil = np.zeros(shp, np.float32)
        blot = _smoothstep(0.9, 2.4, periodic_noise(rng, shp, sp, (0.45, 0.12)))
        big = _smoothstep(1.6, 2.8, periodic_noise(rng, shp, sp, (1.2, 0.35)))
        drips = np.maximum(blot, big)
        for xc, wt in lanes:
            for off in (-WHEEL_OFFSET_M, WHEEL_OFFSET_M):
                wheel += wt * np.exp(-0.5 * ((X - xc - off - wander(18.0, 0.10)) / WHEEL_SIGMA_M) ** 2)
            band = np.exp(-0.5 * ((X - xc - wander(12.0, 0.08)) / 0.28) ** 2)
            oil += wt * band * drips
        wheel = np.clip(wheel, 0, 1) * np.clip(0.8 + 0.2 * periodic_noise(rng, shp, sp, (0.6, 0.6)), 0, 1)
        oil = np.clip(oil * p["oil"], 0, 1)
        macro = 0.6 * periodic_noise(rng, shp, sp, (6.0, 3.0)) + 0.4 * periodic_noise(rng, shp, sp, (1.5, 1.5))
        blend = _smoothstep(-0.6, 0.6, periodic_noise(rng, shp, sp, (2.5, 2.5)))

        # Vector distresses on a 2x supersampled canvas (PIL), wrapped periodically in z.
        from PIL import Image, ImageDraw

        ss = 2
        px = self.texel / ss
        canv = {k: Image.new("L", (W * ss, H * ss), 0) for k in ("crack", "seal", "pmask", "pfresh")}
        dr = {k: ImageDraw.Draw(v) for k, v in canv.items()}

        def to_px(xs, zs):
            return [((x - x0) / dx * ss, z / dz * ss) for x, z in zip(xs, zs)]

        def line(kind: str, xs, zs, width_m: float) -> None:
            wpx = width_m / px
            fill, wid = (255, max(1, round(wpx))) if wpx >= 1 else (round(255 * wpx), 1)
            for o in (-P, 0.0, P):
                dr[kind].line(to_px(xs, np.asarray(zs) + o), fill=fill, width=wid, joint="curve")

        def poly(kind: str, xs, zs, fill: int, outline_m: float | None = None) -> None:
            for o in (-P, 0.0, P):
                pts = to_px(xs, np.asarray(zs) + o)
                if outline_m is None:
                    dr[kind].polygon(pts, fill=fill)
                else:
                    dr[kind].line([*pts, pts[0]], fill=255, width=max(1, round(outline_m / px)))

        def crack(xs, zs) -> bool:
            sealed = rng.random() < p["seal_frac"]
            if sealed:
                line("seal", xs, zs, rng.uniform(0.06, 0.12))
            else:
                line("crack", xs, zs, rng.uniform(0.006, 0.015))
            return sealed

        n_tr = n_long = n_sealed = 0
        for cx0, cx1, cw_lanes in g["carriageways"]:
            z = rng.exponential(p["crack_spacing"])
            while z < P:
                a, b = cx0 + 0.2, cx1 - 0.2
                if rng.random() > 0.6:  # partial-width crack
                    a = rng.uniform(cx0, cx1 - 3.0)
                    b = min(cx1, a + rng.uniform(1.5, 8.0))
                xs = np.arange(a, b, 0.15)
                zs = z + np.cumsum(rng.normal(0, 0.012, xs.size)) + (xs - a) * rng.normal(0, 0.03)
                n_sealed += crack(xs, zs)
                n_tr += 1
                z += max(1.5, rng.exponential(p["crack_spacing"]))
            joints = [x for x in g["lane_lines"] if cx0 < x < cx1]
            joints += [xc + s * WHEEL_OFFSET_M for xc, wt in cw_lanes if wt >= 1.0 for s in (-1, 1)]
            for xj in joints:
                z = rng.uniform(0, 20)
                while z < P:
                    seg = rng.uniform(4.0, 30.0)
                    if rng.random() < p["long_frac"]:
                        zs = np.arange(z, z + seg, 0.2)
                        xs = xj + rng.normal(0, 0.05) + np.cumsum(rng.normal(0, 0.008, zs.size))
                        n_sealed += crack(xs, zs)
                        n_long += 1
                    z += seg + rng.exponential(15.0)

        # Patches erase older cracks/sealant underneath, then get their own sealed edge.
        n_patch = rng.poisson(p["patches_per_km"] * P / 1000.0 * len(g["carriageways"]))
        for _ in range(n_patch):
            cx0, cx1, cw_lanes = g["carriageways"][rng.integers(len(g["carriageways"]))]
            xc, _wt = cw_lanes[rng.integers(len(cw_lanes))]
            kind = rng.choice(["pothole", "wheelpath", "lane"], p=[0.5, 0.35, 0.15])
            if kind == "pothole":
                cx, hw, hl = xc + rng.uniform(-1.2, 1.2), rng.uniform(0.3, 0.8), rng.uniform(0.3, 1.0)
            elif kind == "wheelpath":
                cx, hw, hl = xc + rng.choice([-1, 1]) * WHEEL_OFFSET_M, rng.uniform(0.4, 0.6), rng.uniform(1.5, 7.0)
            else:
                cx, hw, hl = xc, rng.uniform(1.6, 1.8), rng.uniform(1.5, 5.0)
            zc, th = rng.uniform(0, P), math.radians(rng.normal(0, 2.0))
            loc = np.array([(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl)])
            rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
            q = loc @ rot.T
            xs, zs = np.clip(cx + q[:, 0], cx0, cx1), zc + q[:, 1]
            for k in ("crack", "seal"):
                poly(k, xs, zs, 0)
            poly("pmask", xs, zs, 255)
            poly("pfresh", xs, zs, round(255 * rng.uniform(0.35, 1.0)))
            if rng.random() < 0.6:
                poly("seal", xs, zs, 255, outline_m=rng.uniform(0.04, 0.08))

        def down(k: str) -> np.ndarray:
            a = np.asarray(canv[k], dtype=np.float32) / 255.0
            return a.reshape(H, ss, W, ss).mean(axis=(1, 3))

        crk, seal, pm, fresh = down("crack"), down("seal"), down("pmask"), down("pfresh")
        wheel_e = wheel * p["wheel"] * (1.0 - 0.7 * pm)
        oil_e = oil * (1.0 - 0.8 * pm)
        mul = (1.0 + 0.07 * macro) * (1.0 - 0.2 * wheel_e) * (1.0 - 0.45 * oil_e) * (1.0 - 0.6 * crk)
        gloss = np.clip(0.55 * wheel_e + 0.7 * oil_e + 0.3 * fresh, 0.0, 1.0)
        self.summary.update(
            transverse_cracks=n_tr,
            longitudinal_cracks=n_long,
            sealed_fraction=round(n_sealed / max(1, n_tr + n_long), 3),
            patches=int(n_patch),
            patched_area_fraction=round(float(pm.mean()), 4),
            sealant_area_fraction=round(float(seal.mean()), 4),
        )
        return {
            "mul": np.clip(mul, 0.0, 1.3),
            "fresh": fresh,
            "gloss": gloss,
            "seal": np.clip(seal, 0.0, 1.0),
            "blend": blend,
        }

    def reflectance_stats(self, maps: dict[str, np.ndarray], wl: np.ndarray, aged: np.ndarray) -> dict:
        ya = luminous_reflectance(wl, aged)
        yp = luminous_reflectance(wl, WEAR_SPECTRA["asphalt_patch"](wl))
        ys = luminous_reflectance(wl, WEAR_SPECTRA["crack_sealant"](wl))
        f, s = maps["fresh"], maps["seal"]
        r = (1.0 - s) * maps["mul"] * ((1.0 - f) * ya + f * yp) + s * ys
        return {
            "aged_substrate": round(ya, 4),
            "mean": round(float(r.mean()), 4),
            "p05": round(float(np.percentile(r, 5)), 4),
            "p95": round(float(np.percentile(r, 95)), 4),
            "measured_range": "new asphalt ~0.05, aged 0.10-0.18 (Pomerantz et al. LBNL; Herold et al. 2004)",
        }

    # ------------------------------------------------------------------ pbrt
    def material_lines(self, wl: np.ndarray, spd_dir: Path, tiles: list[tuple[str, dict, Path]]) -> list[str]:
        """pbrt texture/material definitions; defines the road material ``asphalt``."""
        for name, fn in WEAR_SPECTRA.items():
            (spd_dir / f"{name}.spd").write_text("\n".join(f"{w:.1f} {v:.6g}" for w, v in zip(wl, fn(wl))) + "\n")
        maps = self.wear_maps()
        g = self.geom
        tag = f"{self.level}_s{self.seed}"
        for k, a in maps.items():
            write_float_exr(self.tex_dir / f"wear_{tag}_{k}.exr", a)
        aged = np.loadtxt(spd_dir / "asphalt_aged.spd")[:, 1]
        self.summary["luminous_reflectance"] = self.reflectance_stats(maps, wl, aged)
        w_m = float(g["x1"] - g["x0"])
        # Road uv is (x, z) / tile_m in the straight frame; map it to (x - x0) / width, -z / period.
        planar = (
            f'"string mapping" "uv" "float uscale" [{self.tile_m / w_m:.8g}] '
            f'"float vscale" [{-self.tile_m / g["period"]:.8g}] "float udelta" [{-g["x0"] / w_m:.8g}]'
        )
        L = [f"# Road wear: {self.level}, seed {self.seed} (tools/highway_road_wear.py)"]
        for k in maps:
            f = self._rel(self.tex_dir / f"wear_{tag}_{k}.exr")
            L.append(f'Texture "rw:{k}" "float" "imagemap" "string filename" "{f}" {planar}')
        names = []
        for i, (aid, info, raw) in enumerate(tiles[:2] or [(None, None, None)]):
            m = "AB"[i]
            names.append(f"rw:asphalt{m}")
            if aid is None:  # no texture assets: procedural wear on the analytic spectrum only
                k, rough, nmap = "rw:mul", f'"texture roughness" "rw:{m}:rough"', ""
                L.append(f'Texture "rw:{m}:rough0" "float" "constant" "float value" [{ROUGH_ASPHALT}]')
            else:
                lum, rmap, npng = self.detiled(aid, info, raw)
                k, rough, nmap = f"rw:{m}:k", f'"texture roughness" "rw:{m}:rough"', f' "string normalmap" "{npng}"'
                L += [
                    f'Texture "rw:{m}:lum" "float" "imagemap" "string filename" "{lum}"',
                    f'Texture "{k}" "float" "scale" "texture tex" "rw:{m}:lum" "texture scale" "rw:mul"',
                    f'Texture "rw:{m}:rough0" "float" "imagemap" "string filename" "{rmap}"'
                    f' "float scale" [{ROUGH_ASPHALT}]',
                ]
            L += [
                f'Texture "rw:{m}:aged" "spectrum" "scale" "spectrum tex" "spd/asphalt_aged.spd" "texture scale" "{k}"',
                f'Texture "rw:{m}:patch" "spectrum" "scale" "spectrum tex" "spd/asphalt_patch.spd"'
                f' "texture scale" "{k}"',
                f'Texture "rw:{m}:refl" "spectrum" "mix" "texture tex1" "rw:{m}:aged" "texture tex2" "rw:{m}:patch"'
                ' "texture amount" "rw:fresh"',
                f'Texture "rw:{m}:rough" "float" "mix" "texture tex1" "rw:{m}:rough0"'
                f' "float tex2" [{ROUGH_POLISHED}] "texture amount" "rw:gloss"',
                f'MakeNamedMaterial "rw:asphalt{m}" "string type" "coateddiffuse" "texture reflectance" "rw:{m}:refl"'
                f' {rough} "float thickness" [0.001]{nmap}',
            ]
        if len(names) == 2:
            L.append(
                'MakeNamedMaterial "rw:asphalt" "string type" "mix" "string materials" ["rw:asphaltA" "rw:asphaltB"]'
                ' "texture amount" "rw:blend"'
            )
        base = "rw:asphalt" if len(names) == 2 else names[0]
        L += [
            'MakeNamedMaterial "rw:sealant" "string type" "coateddiffuse" "spectrum reflectance" "spd/crack_sealant.spd"'
            ' "float roughness" [0.08] "float thickness" [0.001]',
            f'MakeNamedMaterial "asphalt" "string type" "mix" "string materials" ["{base}" "rw:sealant"]'
            ' "texture amount" "rw:seal"',
            "",
        ]
        self.summary["anti_tiling"] = {
            "assets": [t[0] for t in tiles[:2]],
            "tile_m": TILE_M,
            "blend_corr_m": 2.5,
            "method": "variance-preserving cos^2 stochastic re-tiling (Heitz & Neyret 2018) + 2-asphalt noise mix",
        }
        return L

    def detiled(self, aid: str, info: dict, raw: Path) -> tuple[str, str, str]:
        stem = self.tex_dir / f"{aid}_detile{TILE_M:g}m_s{self.seed}"
        paths = [Path(f"{stem}_{s}") for s in ("lum.exr", "rough.exr", "normal.png")]
        if not all(q.is_file() for q in paths):
            src_m = float(info["tile_size_m"])
            n_src = round(TILE_PX * src_m / TILE_M)
            chans = list(_load_maps(info, raw, n_src))
            cells = max(2, round(TILE_M / (0.75 * src_m)))
            n_out = TILE_PX // cells * cells
            lum, rough, nx, ny = detile(
                chans, n_out, cells, np.random.default_rng([self.seed, zlib.crc32(aid.encode())])
            )
            lum = np.clip(lum, 0.02 * lum.mean(), None)
            write_float_exr(paths[0], lum / lum.mean())
            rough = np.clip(rough, 0.05 * rough.mean(), None)
            write_float_exr(paths[1], rough / rough.mean())
            nz = np.sqrt(np.clip(1.0 - nx**2 - ny**2, 0.04, 1.0))
            n = np.stack([nx, ny, nz], axis=-1)
            _write_normal_png(paths[2], n / np.linalg.norm(n, axis=-1, keepdims=True))
        return tuple(self._rel(q) for q in paths)  # type: ignore[return-value]

    # ------------------------------------------------------------------ markings + RPMs
    def marking_lines(
        self, white: list[tuple], yellow: list[tuple], dash: float, gap: float, mesh_fn=None, place_fn=None
    ) -> list[str]:
        """``mesh_fn``/``place_fn``: builder hooks that bend road-frame meshes/instances onto the alignment."""
        """Worn paint (replaces the clean paint_white/paint_yellow quads) and raised pavement markers."""
        rng, p = self.rng, self.p
        pm = 16 * (dash + gap)
        L: list[str] = []
        groups = {
            "dash": [r for r in white if r[3] - r[2] < 50.0],
            "edge_white": [r for r in white if r[3] - r[2] >= 50.0],
            "edge_yellow": yellow,
        }
        for kind, rects in groups.items():
            if not rects:
                continue
            wear = self._marking_wear(p["mark_wear"] * (1.0 if kind == "dash" else 0.6), pm)
            f = self.tex_dir / f"marking_{self.level}_s{self.seed}_{kind}.exr"
            write_float_exr(f, wear)
            paint = "spd/paint_road_yellow.spd" if kind == "edge_yellow" else "spd/paint_road_white.spd"
            L += [
                f'Texture "rw:mw:{kind}" "float" "imagemap" "string filename" "{self._rel(f)}"',
                f'Texture "rw:mp:{kind}" "spectrum" "mix" "spectrum tex1" "{paint}"'
                f' "spectrum tex2" "spd/asphalt_aged.spd" "texture amount" "rw:mw:{kind}"',
                f'Texture "rw:mr:{kind}" "float" "mix" "float tex1" [{ROUGH_BEADS}] "float tex2" [{ROUGH_ASPHALT}]'
                f' "texture amount" "rw:mw:{kind}"',
                f'MakeNamedMaterial "rw:paint_{kind}" "string type" "coateddiffuse" "texture reflectance"'
                f' "rw:mp:{kind}" "texture roughness" "rw:mr:{kind}"',
                f'NamedMaterial "rw:paint_{kind}"',
            ]
            pts, tri, uv = [], [], []
            seg = dash + gap  # split solid lines: 1.5 km sliver triangles render too dark in pbrt
            rects = [
                (x0, x1, z, min(z + seg, z1), off)
                for x0, x1, z0, z1 in rects
                for off in [rng.uniform(0, pm) if kind != "dash" else 0.0]
                for z in np.arange(z0, z1, seg)
            ]
            for i, (x0, x1, z0, z1, off) in enumerate(rects):
                pts += [(x0, 0.004, z0), (x1, 0.004, z0), (x1, 0.004, z1), (x0, 0.004, z1)]
                v0, v1 = -(z0 + off) / pm, -(z1 + off) / pm
                uv += [(0, v0), (1, v0), (1, v1), (0, v1)]
                b = 4 * i
                tri += [(b, b + 2, b + 1), (b, b + 3, b + 2)]
            L += (mesh_fn or _mesh)(np.array(pts), np.array(tri), np.array(uv)) + [""]
        L += self._rpm_lines(groups, dash, gap, place_fn or _translate)
        return L

    def _marking_wear(self, level: float, pm: float) -> np.ndarray:
        """Paint-loss coverage (0 = intact paint, 1 = bare asphalt) across x (u) and along z (v)."""
        rng = self.rng
        nu, h = 16, max(16, round(pm / 0.015))
        sp = (pm / h, 0.15 / nu)
        u = (np.arange(nu) + 0.5) / nu
        f = (
            0.55 * periodic_noise(rng, (h, nu), sp, (0.25, 0.03))
            + 0.45 * periodic_noise(rng, (h, nu), sp, (2.5, 1.0))
            + 1.2 * np.exp(-np.minimum(u, 1 - u) / 0.12)[None, :]
        )
        q = np.quantile(f, 1.0 - level)
        return np.clip((f - q) / 0.25 + 0.5, 0.0, 0.97).astype(np.float32)

    def _rpm_lines(self, groups: dict[str, list[tuple]], dash: float, gap: float, place_fn) -> list[str]:
        rng = self.rng
        L = [
            'MakeNamedMaterial "rw:rpm_lens_clear" "string type" "coateddiffuse"'
            ' "spectrum reflectance" "spd/rpm_lens_clear.spd" "float roughness" [0.02]',
            'MakeNamedMaterial "rw:rpm_lens_red" "string type" "coateddiffuse"'
            ' "spectrum reflectance" "spd/rpm_lens_red.spd" "float roughness" [0.02]',
            'MakeNamedMaterial "rw:rpm_lens_amber" "string type" "coateddiffuse"'
            ' "spectrum reflectance" "spd/rpm_lens_amber.spd" "float roughness" [0.02]',
            'MakeNamedMaterial "rw:rpm_body_white" "string type" "coateddiffuse"'
            ' "spectrum reflectance" "spd/paint_road_white.spd" "float roughness" [0.4]',
            'MakeNamedMaterial "rw:rpm_body_yellow" "string type" "coateddiffuse"'
            ' "spectrum reflectance" "spd/paint_road_yellow.spd" "float roughness" [0.4]',
        ]
        # 100 x 100 mm footprint, 18 mm tall; sloped lens faces towards -z (approaching traffic) and +z.
        b, t, h, zt = 0.05, 0.04, 0.018, 0.02
        front = [(-b, 0, -b), (b, 0, -b), (t, h, -zt), (-t, h, -zt)]
        back = [(b, 0, b), (-b, 0, b), (-t, h, zt), (t, h, zt)]
        body = [
            [(-t, h, -zt), (t, h, -zt), (t, h, zt), (-t, h, zt)],
            [(b, 0, -b), (b, 0, b), (t, h, zt), (t, h, -zt)],
            [(-b, 0, b), (-b, 0, -b), (-t, h, -zt), (-t, h, zt)],
        ]
        for name, lens_f, lens_b, bmat in (
            ("white", "rw:rpm_lens_clear", "rw:rpm_lens_red", "rw:rpm_body_white"),
            ("yellow", "rw:rpm_lens_amber", None, "rw:rpm_body_yellow"),
        ):
            L.append(f'ObjectBegin "rw:rpm_{name}"')
            for mat, quads in ((lens_f, [front]), (lens_b or bmat, [back]), (bmat, body)):
                pts = np.array([v for q in quads for v in q], dtype=float)
                tri = np.array([(4 * i, 4 * i + 1, 4 * i + 2) for i in range(len(quads))]
                               + [(4 * i, 4 * i + 2, 4 * i + 3) for i in range(len(quads))])  # fmt: skip
                L += ["AttributeBegin", f'NamedMaterial "{mat}"', *_mesh(pts, tri), "AttributeEnd"]
            L += ["ObjectEnd", ""]
        z0, z1 = self.geom.get("z0", -1e9), self.geom.get("z1", 1e9)
        placed = missing = 0

        def put(name: str, x: float, y: float, z: float, oncoming: bool) -> None:
            nonlocal placed, missing
            if not z0 + 1.0 < z < z1 - 1.0:
                return
            if rng.random() < self.p["rpm_missing"]:
                missing += 1
                return
            rot = ["Rotate 180 0 1 0"] if oncoming else []
            L.append(
                " ".join(
                    ["AttributeBegin", *place_fn(x, y, z), *rot, f'ObjectInstance "rw:rpm_{name}"', "AttributeEnd"]
                )
            )
            placed += 1

        lines: dict[float, list[tuple]] = {}
        for r in groups["dash"]:
            lines.setdefault(round(0.5 * (r[0] + r[1]), 3), []).append(r)
        for xc, rs in lines.items():
            rs = sorted(rs, key=lambda r: r[2])
            for r in rs[::2]:  # every other gap: 2N = 24.4 m
                put("white", xc, 0.0, r[3] + gap / 2.0, xc < self.geom["median_x"])
        for r in groups["edge_yellow"]:
            xc = 0.5 * (r[0] + r[1])
            z = r[2] + 2.0 + gap / 2.0 + dash
            while z < r[3]:
                put("yellow", xc, 0.004, z, xc < self.geom["median_x"])
                z += 2.0 * (dash + gap)
        self.summary["raised_pavement_markers"] = {
            "placed": placed,
            "missing": missing,
            "spacing_m": 2.0 * (dash + gap),
        }
        return L + [""]


def _translate(x: float, y: float, z: float) -> list[str]:
    return [f"Translate {x:.4f} {y:.4f} {z:.4f}"]


def _mesh(p: np.ndarray, tri: np.ndarray, uv: np.ndarray | None = None) -> list[str]:
    def pts(a: np.ndarray) -> str:
        return " ".join(f"{v:.6g}" for v in np.asarray(a, dtype=np.float64).ravel())

    out = ['Shape "trianglemesh"', f'    "point3 P" [ {pts(p)} ]', f'    "integer indices" [ {pts(tri)} ]']
    if uv is not None:
        out.append(f'    "point2 uv" [ {pts(uv)} ]')
    return out
