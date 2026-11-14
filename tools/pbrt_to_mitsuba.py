"""Convert an open-cam pbrt-v4 scene file to a Mitsuba 3 scene, and render it on the GPU with a spectral film.

Covers what the generated highway scenes use (tools/build_highway_scene.py). pbrt-only features become the plugins
in tools/mitsuba_pbrt_plugins.py, or an approximation that is logged once per kind (``Converter.notes``):

* ``coateddiffuse``  -> ``roughplastic`` (``plastic`` below alpha 1e-3); pbrt's layered model has no Mitsuba plugin
* ``coatedconductor``-> ``roughconductor`` with the conductor's parameters (the clear coat is dropped)
* ``mix`` material   -> ``blendbsdf`` (the mean of pbrt's stochastic per-hit choice)
* RGB ``eta``/``k`` -> their channel means; ``normalmap`` ignored
* emitter selection: Mitsuba picks one emitter in proportion to ``sampling_weight``; this sets it to each
  light's luminous power (like pbrt's ``power`` light sampler; the pbrt scenes use the light BVH). The
  goniometric lamps are one emitter that picks a lamp by power or, with ``--lamp-select exact``, by intensity
  toward the shading point over d^2 (pbrt patch 0024). The sky gets a fixed share of the choices
  (``SKY_SHARE``, as pbrt patch 0035).

Light units follow pbrt: spectra divided by their photometric integral (sum of CIE Y at 1 nm), goniometric
images in cd, ``illuminance`` for the uniform infinite light. The film writes pbrt's spectral EXR layout
(``S0.<lambda>nm`` buckets) so tools/exr_multispectral.py and the analysis scripts read it unchanged.

    venv/bin/python tools/pbrt_to_mitsuba.py out/firefly_exp/highway_path.pbrt --spp 256 --seed 1 \\
        --out out/firefly_exp/out/mitsuba_s1.exr
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

# --------------------------------------------------------------------------------------------- parser

_TOKEN = re.compile(r'"[^"]*"|\[|\]|#[^\n]*|[^\s\[\]"#]+')


@dataclass
class Param:
    type: str
    name: str
    values: list


@dataclass
class Directive:
    name: str
    args: list  # positional values (strings / numbers) before the parameter list
    params: dict[str, Param]
    file: Path


def _value(tok: str):
    if tok.startswith('"'):
        return tok[1:-1]
    if tok in ("true", "false"):
        return tok == "true"
    return float(tok)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text) if not t.startswith("#")]


def parse_file(path: Path) -> list[Directive]:
    """Flatten a pbrt file (and its Includes) into directives."""
    toks = tokenize(path.read_text())
    out: list[Directive] = []
    i, n = 0, len(toks)
    while i < n:
        name = toks[i]
        if not name[0].isalpha():
            raise ValueError(f"{path}: unexpected token {name!r}")
        i += 1
        args: list = []
        params: dict[str, Param] = {}
        while i < n and not _is_decl(toks[i]) and not (toks[i][0].isalpha() and toks[i] not in ("true", "false")):
            if toks[i] == "[":  # Transform [ ... ]
                i += 1
                while toks[i] != "]":
                    args.append(_value(toks[i]))
                    i += 1
            elif toks[i] != "]":
                args.append(_value(toks[i]))
            i += 1
        while i < n and _is_decl(toks[i]):
            ptype, pname = toks[i][1:-1].split()
            i += 1
            vals = []
            if toks[i] == "[":
                i += 1
                while toks[i] != "]":
                    vals.append(_value(toks[i]))
                    i += 1
                i += 1
            else:
                vals.append(_value(toks[i]))
                i += 1
            params[pname] = Param(ptype, pname, vals)
        if name == "Include" or name == "Import":
            out += parse_file((path.parent / args[0]).resolve() if not Path(args[0]).is_absolute() else Path(args[0]))
            continue
        out.append(Directive(name, args, params, path))
    return out


_PARAM_TYPES = {"integer", "float", "point2", "vector2", "point3", "vector3", "normal", "normal3", "point", "vector",
                "spectrum", "rgb", "blackbody", "bool", "string", "texture", "color"}


def _is_decl(tok: str) -> bool:
    """A quoted "type name" parameter declaration."""
    if not tok.startswith('"'):
        return False
    w = tok[1:-1].split()
    return len(w) == 2 and w[0] in _PARAM_TYPES


# fraction of emitter choices that go to the infinite light (the others go by luminous power)
SKY_SHARE = 0.05

# instanced objects up to this many triangles in all are flattened (see Converter.place_instances)
FLATTEN_MAX_TRIANGLES = 4_000_000

# --------------------------------------------------------------------------------------------- transforms

def translate(x, y, z):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def scale(x, y, z):
    return np.diag([x, y, z, 1.0])


def rotate(deg, x, y, z):
    a = np.asarray([x, y, z], float)
    a /= np.linalg.norm(a)
    s, c = math.sin(math.radians(deg)), math.cos(math.radians(deg))
    m = np.eye(4)
    m[:3, :3] = c * np.eye(3) + s * np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]]) \
        + (1 - c) * np.outer(a, a)
    return m


def look_at(eye, target, up):
    """pbrt LookAt: returns camera-from-world."""
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    d = target - eye
    d /= np.linalg.norm(d)
    right = np.cross(up / np.linalg.norm(up), d)
    right /= np.linalg.norm(right)
    new_up = np.cross(d, right)
    w = np.eye(4)
    w[:3, 0], w[:3, 1], w[:3, 2], w[:3, 3] = right, new_up, d, eye
    return np.linalg.inv(w)


# --------------------------------------------------------------------------------------------- converter

@dataclass
class GState:
    ctm: np.ndarray = field(default_factory=lambda: np.eye(4))
    material: str | None = None
    area: Directive | None = None
    reverse: bool = False

    def copy(self):
        return GState(self.ctm.copy(), self.material, self.area, self.reverse)


class Converter:
    def __init__(self, scene: Path, cache: Path, *, max_depth_offset: int = 1, lamp_select: str = "power"):
        import mitsuba as mi  # noqa: PLC0415

        self.mi = mi
        self.scene = scene.resolve()
        self.base = self.scene.parent
        self.cache = cache
        cache.mkdir(parents=True, exist_ok=True)
        self.max_depth_offset = max_depth_offset
        self.notes: dict[str, int] = {}
        self.textures: dict[str, Directive] = {}
        self.tex_ctm: dict[str, np.ndarray] = {}
        self.named_materials: dict[str, Directive] = {}
        self.bsdf_ids: dict[str, str] = {}
        self.objects: dict[str, list] = {}
        self.out: dict = {"type": "scene"}
        self.n = 0
        self.lights: list[dict] = []  # metadata for the report
        self.gonio: list[dict] = []  # goniometric lamps -> one pbrt_gonio_set emitter
        self.instances: dict[str, list[np.ndarray]] = {}
        self.lamp_select = lamp_select
        self.film_info: dict = {}
        self.ybar = np.array([mi.cie1931_y(float(x)) for x in range(360, 831)], dtype=np.float64).ravel()

    # ---- helpers
    def note(self, msg: str):
        self.notes[msg] = self.notes.get(msg, 0) + 1

    def uid(self, prefix: str) -> str:
        self.n += 1
        return f"{prefix}_{self.n}"

    def path(self, f: str) -> str:
        p = Path(f)
        return str(p if p.is_absolute() else (self.base / p).resolve())

    def mtx(self, m: np.ndarray):
        return self.mi.ScalarTransform4f(np.asarray(m, dtype=np.float64).tolist())

    # ---- spectra
    def spd_array(self, p: Param) -> np.ndarray:
        v = p.values
        if len(v) == 1 and isinstance(v[0], str):
            return np.loadtxt(self.path(v[0]), comments="#").reshape(-1, 2)
        a = np.asarray(v, float)
        if len(a) == 1:
            return np.array([[360.0, a[0]], [830.0, a[0]]])
        return a.reshape(-1, 2)

    def irregular(self, a: np.ndarray, k: float = 1.0) -> dict:
        lam, val = a[:, 0], a[:, 1] * k
        keep = np.r_[True, np.diff(lam) > 0]
        return {"type": "irregular", "wavelengths": ",".join(f"{x:.6g}" for x in lam[keep]),
                "values": ",".join(f"{x:.9g}" for x in val[keep])}

    def photometric(self, a: np.ndarray) -> float:
        """pbrt SpectrumToPhotometric: sum over integer nm of CIE Y times the (piecewise-linear) spectrum."""
        lam = np.arange(360, 831, dtype=float)
        s = np.interp(lam, a[:, 0], a[:, 1], left=0.0, right=0.0)
        return float((self.ybar * s).sum())

    def spectrum_value(self, p: Param, *, unbounded=False) -> dict:
        """A spectrum-valued parameter: spd file / inline list / rgb / named metal."""
        if p.type == "rgb":
            if unbounded:
                self.note("RGB eta/k replaced by the channel mean")
                return {"type": "uniform", "value": float(np.mean(p.values))}
            return {"type": "rgb", "value": [float(x) for x in p.values]}
        if p.type in ("float",):
            return {"type": "uniform", "value": float(p.values[0])}
        if p.type == "spectrum" and len(p.values) == 1 and isinstance(p.values[0], str) and \
                re.fullmatch(r"metal-\w+-(eta|k)", p.values[0]):
            raise KeyError(p.values[0])  # handled by the conductor converter
        return self.irregular(self.spd_array(p))

    # ---- textures
    def texture(self, p: Param | None, default=None, *, kind="spectrum"):
        """Mitsuba texture dict for a parameter that may be a constant, a spectrum or a named texture."""
        if p is None:
            return default
        if p.type == "texture":
            return self.named_texture(p.values[0])
        if p.type == "float":
            return {"type": "uniform", "value": float(p.values[0])}
        return self.spectrum_value(p)

    def named_texture(self, name: str) -> dict:
        d = self.textures[name]
        cls = d.args[2]
        P = d.params
        if cls == "imagemap":
            fn = P["filename"].values[0]
            enc = P["encoding"].values[0] if "encoding" in P else None
            raw = fn.lower().endswith(".exr") or enc == "linear" or (d.args[1] == "float" and enc is None
                                                                      and not fn.lower().endswith((".png", ".jpg")))
            mapping = P["mapping"].values[0] if "mapping" in P else "uv"
            tex = {"type": "bitmap", "filename": self.path(fn), "raw": bool(raw)}
            if mapping == "uv":
                su = P["uscale"].values[0] if "uscale" in P else 1.0
                sv = P["vscale"].values[0] if "vscale" in P else 1.0
                du = P["udelta"].values[0] if "udelta" in P else 0.0
                dv = P["vdelta"].values[0] if "vdelta" in P else 0.0
                # pbrt looks images up at (s, 1 - t); Mitsuba's row 0 is the top as well
                tex["to_uv"] = self.mi.ScalarTransform4f([[su, 0, 0, du], [0, -sv, 0, 1 - dv], [0, 0, 1, 0], [0, 0, 0, 1]])
            elif mapping == "planar":
                tex = {"type": "pbrt_planar", "tex": tex,
                       "v1": self.mi.ScalarVector3f(P["v1"].values), "v2": self.mi.ScalarVector3f(P["v2"].values),
                       "udelta": float(P["udelta"].values[0]) if "udelta" in P else 0.0,
                       "vdelta": float(P["vdelta"].values[0]) if "vdelta" in P else 0.0,
                       "to_world": self.mtx(self.tex_ctm[name])}
            else:
                raise NotImplementedError(f"texture mapping {mapping}")
            if "scale" in P and P["scale"].values[0] != 1:
                tex = {"type": "pbrt_scale", "tex": tex, "scale": {"type": "uniform", "value": float(P["scale"].values[0])}}
            if "invert" in P and P["invert"].values[0]:
                raise NotImplementedError("imagemap invert")
            return tex
        if cls == "scale":
            return {"type": "pbrt_scale", "tex": self.texture(P.get("tex"), {"type": "uniform", "value": 1.0}),
                    "scale": self.texture(P.get("scale"), {"type": "uniform", "value": 1.0})}
        if cls == "mix":
            return {"type": "pbrt_mix", "tex1": self.texture(P.get("tex1"), {"type": "uniform", "value": 0.0}),
                    "tex2": self.texture(P.get("tex2"), {"type": "uniform", "value": 1.0}),
                    "amount": self.texture(P.get("amount"), {"type": "uniform", "value": 0.5})}
        if cls == "constant":
            return self.texture(P.get("value"), {"type": "uniform", "value": 1.0})
        raise NotImplementedError(f"texture class {cls}")

    # ---- materials
    @staticmethod
    def alpha(P: dict, key="roughness", default=0.0, remap_default=True) -> float:
        r = P[key].values[0] if key in P else default
        remap = P["remaproughness"].values[0] if "remaproughness" in P else remap_default
        return float(math.sqrt(r) if remap else r)

    def conductor_params(self, P: dict, prefix="") -> dict:
        eta, k = P.get(prefix + "eta"), P.get(prefix + "k")
        if eta is not None and eta.type == "spectrum" and isinstance(eta.values[0], str) and eta.values[0].startswith("metal-"):
            return {"material": eta.values[0].split("-")[1]}
        if eta is None and k is None:
            return {"material": "Cu"}  # pbrt's default conductor is copper
        return {"eta": self.spectrum_value(eta, unbounded=True), "k": self.spectrum_value(k, unbounded=True)}

    def bsdf(self, d: Directive) -> tuple[dict, bool]:
        """(Mitsuba BSDF dict, opaque); opaque BSDFs get wrapped in ``twosided`` by the caller."""
        P = d.params
        mtype = d.args[0] if d.name == "Material" else P["type"].values[0]
        if "normalmap" in P:
            self.note("normalmap ignored")
        if mtype == "diffuse":
            return {"type": "diffuse", "reflectance": self.texture(P.get("reflectance"), {"type": "uniform", "value": 0.5})}, True
        if mtype == "coateddiffuse":
            self.note("coateddiffuse -> roughplastic")
            a = self.alpha(P, default=0.0)
            if "roughness" in P and P["roughness"].type == "texture":
                raise NotImplementedError("textured coateddiffuse roughness")
            eta = float(P["eta"].values[0]) if "eta" in P else 1.5
            refl = self.texture(P.get("reflectance"), {"type": "uniform", "value": 0.5})
            if a < 1e-3:
                return {"type": "plastic", "diffuse_reflectance": refl, "int_ior": eta, "ext_ior": 1.0}, True
            return {"type": "roughplastic", "diffuse_reflectance": refl, "alpha": a, "int_ior": eta,
                    "ext_ior": 1.0, "distribution": "ggx"}, True
        if mtype in ("conductor", "coatedconductor"):
            pre = "conductor." if mtype == "coatedconductor" else ""
            if mtype == "coatedconductor":
                self.note("coatedconductor -> roughconductor (clear coat dropped)")
            au = self.alpha(P, pre + "uroughness", self.alpha(P, pre + "roughness", 0.0)) if pre + "uroughness" in P \
                else self.alpha(P, pre + "roughness", 0.0)
            av = self.alpha(P, pre + "vroughness", au) if pre + "vroughness" in P else au
            c = self.conductor_params(P, pre)
            if max(au, av) < 1e-3:
                return {"type": "conductor", **c}, True
            return {"type": "roughconductor", "distribution": "ggx", "alpha_u": au, "alpha_v": av, **c}, True
        if mtype in ("dielectric", "thindielectric"):
            eta = float(P["eta"].values[0]) if "eta" in P and P["eta"].type == "float" else 1.5
            if "eta" in P and P["eta"].type != "float":
                self.note("spectral dielectric eta -> 1.5")
            if mtype == "thindielectric":
                return {"type": "thindielectric", "int_ior": eta, "ext_ior": 1.0}, False
            a = self.alpha(P, "roughness", 0.0)
            if a < 1e-3:
                return {"type": "dielectric", "int_ior": eta, "ext_ior": 1.0}, False
            return {"type": "roughdielectric", "int_ior": eta, "ext_ior": 1.0, "alpha": a, "distribution": "ggx"}, False
        if mtype == "diffusetransmission":
            return {"type": "pbrt_difftrans",
                    "reflectance": self.texture(P.get("reflectance"), {"type": "uniform", "value": 0.25}),
                    "transmittance": self.texture(P.get("transmittance"), {"type": "uniform", "value": 0.25})}, False
        if mtype == "retroreflective":
            return {"type": "pbrt_retro",
                    "reflectance": self.texture(P.get("reflectance"), {"type": "uniform", "value": 0.0}),
                    "ra": self.texture(P.get("ra"), {"type": "uniform", "value": 0.0}),
                    "alpha0": math.radians(P["observationangle"].values[0] if "observationangle" in P else 0.2),
                    "width": math.radians(P["lobewidth"].values[0] if "lobewidth" in P else 0.31),
                    "q": float(P["entranceexponent"].values[0] if "entranceexponent" in P else 2.7)}, True
        if mtype == "mix":
            names = P["materials"].values
            parts = [self.bsdf(self.named_materials[nm]) for nm in names]
            amount = P.get("amount")
            w = self.texture(amount, {"type": "uniform", "value": 0.5}) if amount is not None and amount.type == "texture" \
                else float(amount.values[0]) if amount is not None else 0.5
            return {"type": "blendbsdf", "weight": w, "bsdf_0": parts[0][0], "bsdf_1": parts[1][0]}, parts[0][1] and parts[1][1]
        if mtype in ("interface", "none", ""):
            return {"type": "null"}, False
        raise NotImplementedError(f"material {mtype}")

    def material_ref(self, key: str, d: Directive) -> dict:
        if key not in self.bsdf_ids:
            b, opaque = self.bsdf(d)
            if opaque:
                b = {"type": "twosided", "bsdf": b}
            bid = self.uid("bsdf")
            self.out[bid] = b
            self.bsdf_ids[key] = bid
        return {"type": "ref", "id": self.bsdf_ids[key]}

    # ---- geometry
    def write_ply(self, P: np.ndarray, F: np.ndarray, N=None, UV=None) -> str:
        fn = self.cache / f"mesh_{self.n:05d}.ply"
        self.n += 1
        props = ["x", "y", "z"] + (["nx", "ny", "nz"] if N is not None else []) + (["u", "v"] if UV is not None else [])
        cols = [P] + ([N] if N is not None else []) + ([UV] if UV is not None else [])
        v = np.concatenate(cols, 1).astype("<f4")
        head = ["ply", "format binary_little_endian 1.0", f"element vertex {len(P)}"] + [f"property float {p}" for p in props] + \
            [f"element face {len(F)}", "property list uchar int vertex_indices", "end_header"]
        f = np.zeros(len(F), dtype=[("n", "u1"), ("i", "<i4", 3)])
        f["n"], f["i"] = 3, F
        with open(fn, "wb") as fh:
            fh.write(("\n".join(head) + "\n").encode())
            fh.write(v.tobytes())
            fh.write(f.tobytes())
        return str(fn)

    @staticmethod
    def ply_header(fn: str) -> dict:
        """Face count and vertex properties from a PLY header."""
        info = {"faces": 0, "normals": False, "uv": False}
        with open(fn, "rb") as fh:
            for raw in fh:
                line = raw.decode("ascii", "replace").strip()
                if line.startswith("element face"):
                    info["faces"] = int(line.split()[2])
                elif line.startswith("property") and line.split()[-1] == "nx":
                    info["normals"] = True
                elif line.startswith("property") and line.split()[-1] in ("u", "s", "texture_u"):
                    info["uv"] = True
                elif line == "end_header":
                    break
        return info

    def shape(self, d: Directive, g: GState) -> tuple[dict, np.ndarray | None, float | None]:
        """(Mitsuba shape dict without bsdf/emitter, summed object-space normal for area lights or None, world area
        or None)."""
        P = d.params
        kind = d.args[0]
        swaps = np.linalg.det(g.ctm[:3, :3]) < 0
        flip = bool(swaps ^ g.reverse)
        nrm = None
        area = None
        if kind == "plymesh":
            fn = self.path(P["filename"].values[0])
            s = {"type": "ply", "filename": fn, "to_world": self.mtx(g.ctm)}
            if not self.ply_header(fn)["normals"]:
                s["face_normals"] = True  # pbrt shades meshes without normals flat; Mitsuba would smooth them
        elif kind in ("trianglemesh", "bilinearmesh"):
            pts = np.asarray(P["P"].values, float).reshape(-1, 3)
            idx = np.asarray(P["indices"].values, int) if "indices" in P else np.arange(len(pts))
            uv = np.asarray(P["uv"].values, float).reshape(-1, 2) if "uv" in P else None
            N = np.asarray(P["N"].values, float).reshape(-1, 3) if "N" in P else None
            if kind == "bilinearmesh":
                q = idx.reshape(-1, 4)  # p00 p10 p01 p11
                F = np.concatenate([q[:, [0, 1, 3]], q[:, [0, 3, 2]]])
                if uv is None and len(pts) == 4:
                    uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
            else:
                F = idx.reshape(-1, 3)
            if uv is not None:
                uv = uv.copy()  # stored as pbrt (s, t); the bitmap to_uv does the flip
            fn = self.write_ply(pts, F, N, uv)
            s = {"type": "ply", "filename": fn, "to_world": self.mtx(g.ctm)}
            if N is None:
                s["face_normals"] = True
            tri = pts[F]
            nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).sum(0)
            tw = np.c_[tri.reshape(-1, 3), np.ones(len(F) * 3)] @ g.ctm.T
            tw = tw[:, :3].reshape(-1, 3, 3)
            area = 0.5 * float(np.linalg.norm(np.cross(tw[:, 1] - tw[:, 0], tw[:, 2] - tw[:, 0]), axis=1).sum())
        elif kind == "cylinder":
            r = float(P["radius"].values[0]) if "radius" in P else 1.0
            z0 = float(P["zmin"].values[0]) if "zmin" in P else -1.0
            z1 = float(P["zmax"].values[0]) if "zmax" in P else 1.0
            if "phimax" in P and P["phimax"].values[0] < 360:
                self.note("cylinder phimax ignored")
            s = {"type": "cylinder", "p0": [0, 0, z0], "p1": [0, 0, z1], "radius": r, "to_world": self.mtx(g.ctm)}
        elif kind == "sphere":
            r = float(P["radius"].values[0]) if "radius" in P else 1.0
            s = {"type": "sphere", "radius": r, "to_world": self.mtx(g.ctm)}
        elif kind == "disk":
            r = float(P["radius"].values[0]) if "radius" in P else 1.0
            h = float(P["height"].values[0]) if "height" in P else 0.0
            s = {"type": "disk", "to_world": self.mtx(g.ctm @ translate(0, 0, h) @ scale(r, r, 1))}
            nrm = np.array([0, 0, 1.0])
        else:
            raise NotImplementedError(f"shape {kind}")
        if "alpha" in P:
            self.note("shape alpha ignored")
        if flip:
            s["flip_normals"] = True
        return s, nrm, area

    def area_emitter(self, a: Directive, g: GState, nrm: np.ndarray | None, area: float | None) -> dict:
        P = a.params
        spd = self.spd_array(P["L"]) if "L" in P else np.array([[360, 1.0], [830, 1.0]])
        k = float(P["scale"].values[0]) if "scale" in P else 1.0
        if not ("photometric" in P and not P["photometric"].values[0]):
            k /= self.photometric(spd)
        if "power" in P:
            raise NotImplementedError("area light power")
        if "twosided" in P and P["twosided"].values[0]:
            self.note("two-sided area light emits one side")
        if area is None:
            self.note("area light on a shape without a known area: sampling weight from 0.01 m^2")
            area = 0.01
        lum = k * self.photometric(spd)  # luminance [nit] before the angular map
        if "angularfilename" in P:
            fn = P["angularfilename"].values[0]
            # "lamp_id": Mitsuba's loader shares identical plugin dicts, but an emitter has one shape
            e = {"type": "pbrt_area_angular", "radiance": self.irregular(spd, k), "filename": self.path(fn),
                 "to_world": self.mtx(g.ctm), "lamp_id": self.uid("lamp")}
            phi = lum * area * math.pi * self.angular_factor(fn, g, nrm)
        else:
            e = {"type": "area", "radiance": self.irregular(spd, k)}
            phi = lum * area * math.pi
        e["sampling_weight"] = phi
        self.lights.append({"kind": "area", "phi_v": phi})
        return e

    def angular_factor(self, fn: str, g: GState, nrm: np.ndarray | None) -> float:
        from exr_multispectral import read_separate_exr_channels  # noqa: PLC0415
        from highway_sky import equal_area_directions  # noqa: PLC0415

        ch = read_separate_exr_channels(Path(self.path(fn)))
        img = (ch["Y"] if "Y" in ch else next(iter(ch.values()))).astype(np.float64)
        d = equal_area_directions(img.shape[0])
        if nrm is None or np.linalg.norm(nrm) == 0:
            return float(img.mean())
        n = nrm / np.linalg.norm(nrm)
        if np.linalg.det(g.ctm[:3, :3]) < 0:
            n = -n
        if g.reverse:
            n = -n
        cos = np.clip(d @ n, 0, None)
        # (1/pi) * sum a(w) cos(w) dw with dw = 4 pi / N
        return float((img * cos).sum() * 4 / img.size)

    # ---- lights
    def light(self, d: Directive, g: GState):
        P = d.params
        kind = d.args[0]
        if kind == "infinite":
            if "filename" in P:
                raise NotImplementedError("infinite light with an image")
            spd = self.spd_array(P["L"])
            k = float(P["scale"].values[0]) if "scale" in P else 1.0
            k /= self.photometric(spd)
            if "illuminance" in P:
                k *= P["illuminance"].values[0] / math.pi
            e = {"type": "constant", "radiance": self.irregular(spd, k)}
            self.out[self.uid("sky")] = e
            self.lights.append({"kind": "infinite", "k": k})
            return
        if kind in ("goniometric", "point"):
            spd = self.spd_array(P["I"]) if "I" in P else np.array([[360, 1.0], [830, 1.0]])
            k = float(P["scale"].values[0]) if "scale" in P else 1.0
            if not ("photometric" in P and not P["photometric"].values[0]):
                k /= self.photometric(spd)
            if "power" in P:
                raise NotImplementedError("light power")
            frm = np.asarray(P["from"].values, float) if "from" in P else np.zeros(3)
            ctm = g.ctm @ translate(*frm)
            if kind == "point":
                e = {"type": "point", "to_world": self.mtx(ctm), "intensity": self.irregular(spd, k)}
                phi = 4 * math.pi * k * self.photometric(spd)
                e["sampling_weight"] = phi
                self.out[self.uid("light")] = e
            else:
                swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
                fn = self.path(P["filename"].values[0])
                phi = 4 * math.pi * self.image_mean(P["filename"].values[0]) * k * self.photometric(spd)
                self.gonio.append({"to_world": (ctm @ swap).tolist(), "image": fn,
                                   "spd": np.c_[spd[:, 0], spd[:, 1] * k].tolist(),
                                   "weight_scale": k * self.photometric(spd), "phi": phi})
            self.lights.append({"kind": kind, "phi_v": phi})
            return
        raise NotImplementedError(f"light {kind}")

    def image_mean(self, fn: str) -> float:
        from exr_multispectral import read_separate_exr_channels  # noqa: PLC0415

        ch = read_separate_exr_channels(Path(self.path(fn)))
        return float((ch["Y"] if "Y" in ch else next(iter(ch.values()))).mean())

    # ---- instancing
    def place_instances(self) -> None:
        """Flatten small instanced objects into one mesh per part; keep big ones as Mitsuba instances.

        Mitsuba re-traces every instance on each ``mi.render`` call (about 40 ms each): 964 instances cost
        40 s per call against 0.2 s without them. Flattening everything would be 114 M triangles here.
        """
        for name, ctms in self.instances.items():
            parts = self.objects[name]
            meshes = all(s["type"] == "ply" for s, _ in parts)
            tris = sum(self.ply_header(s["filename"])["faces"] for s, _ in parts) * len(ctms) if meshes else None
            if meshes and tris <= FLATTEN_MAX_TRIANGLES:
                self.note("small instanced objects flattened")
                for s, m_def in parts:
                    self.out[self.uid("shape")] = self.flatten(s, m_def, ctms)
                continue
            gid = "group_" + re.sub(r"\W", "_", name)
            grp = {"type": "shapegroup"}
            for i, (s, _) in enumerate(parts):
                grp[f"s{i}"] = s
            self.out[gid] = grp
            for ctm in ctms:
                self.out[self.uid("inst")] = {"type": "instance", "shapegroup": {"type": "ref", "id": gid},
                                              "to_world": self.mtx(ctm)}

    def flatten(self, s: dict, m_def: np.ndarray, ctms: list[np.ndarray]) -> dict:
        mi = self.mi
        hdr = self.ply_header(s["filename"])
        mesh = mi.load_dict({"type": "ply", "filename": s["filename"], "face_normals": not hdr["normals"]})
        P = np.array(mesh.vertex_positions_buffer(), np.float64).reshape(-1, 3)
        F = np.array(mesh.faces_buffer(), np.int64).reshape(-1, 3)
        N = np.array(mesh.vertex_normals_buffer(), np.float64).reshape(-1, 3) if hdr["normals"] else None
        UV = np.array(mesh.vertex_texcoords_buffer(), np.float64).reshape(-1, 2) if mesh.has_vertex_texcoords() else None
        Ps, Fs, Ns, UVs = [], [], [], []
        for k, ctm in enumerate(ctms):
            M = ctm @ m_def
            Ps.append(P @ M[:3, :3].T + M[:3, 3])
            Fs.append(F + k * len(P))
            if N is not None:
                n = N @ np.linalg.inv(M[:3, :3])
                Ns.append(n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12))
            if UV is not None:
                UVs.append(UV)
        out = {k: v for k, v in s.items() if k not in ("filename", "to_world", "face_normals")}
        out["filename"] = self.write_ply(np.concatenate(Ps), np.concatenate(Fs), np.concatenate(Ns) if Ns else None,
                                         np.concatenate(UVs) if UVs else None)
        if N is None:
            out["face_normals"] = True
        return out

    # ---- top level
    def convert(self, *, spp: int, width: int | None = None, height: int | None = None) -> dict:
        ds = parse_file(self.scene)
        g = GState()
        stack: list[GState] = []
        named_coords: dict[str, np.ndarray] = {}
        cam = None
        film = sampler = integ = None
        cur_obj: list | None = None
        for d in ds:
            n, A, P = d.name, d.args, d.params
            if n == "AttributeBegin":
                stack.append(g.copy())
            elif n == "AttributeEnd":
                g = stack.pop()
            elif n == "Identity":
                g.ctm = np.eye(4)
            elif n == "Translate":
                g.ctm = g.ctm @ translate(*A)
            elif n == "Scale":
                g.ctm = g.ctm @ scale(*A)
            elif n == "Rotate":
                g.ctm = g.ctm @ rotate(*A)
            elif n == "LookAt":
                g.ctm = g.ctm @ look_at(A[0:3], A[3:6], A[6:9])
            elif n == "Transform":
                g.ctm = np.asarray(A, float).reshape(4, 4).T
            elif n == "ConcatTransform":
                g.ctm = g.ctm @ np.asarray(A, float).reshape(4, 4).T
            elif n == "CoordinateSystem":
                named_coords[A[0]] = g.ctm.copy()
            elif n == "CoordSysTransform":
                g.ctm = named_coords[A[0]].copy()
            elif n == "ReverseOrientation":
                g.reverse = not g.reverse
            elif n == "Camera":
                cam = (d, g.ctm.copy())
                named_coords["camera"] = np.linalg.inv(g.ctm)
            elif n == "Film":
                film = d
            elif n == "Sampler":
                sampler = d
            elif n == "Integrator":
                integ = d
            elif n == "WorldBegin":
                g = GState()
            elif n == "Texture":
                self.textures[A[0]] = d
                self.tex_ctm[A[0]] = g.ctm.copy()
            elif n == "MakeNamedMaterial":
                self.named_materials[A[0]] = d
            elif n == "NamedMaterial":
                g.material = "named:" + A[0]
            elif n == "Material":
                key = "anon:" + json.dumps([A, {k: [p.type, p.values] for k, p in P.items()}], sort_keys=True)
                self.named_materials[key] = d
                g.material = key
            elif n == "AreaLightSource":
                g.area = d
            elif n == "LightSource":
                self.light(d, g)
            elif n == "Shape":
                s, nrm, area = self.shape(d, g)
                key = g.material or "default"
                md = self.named_materials.get(g.material[6:] if g.material and g.material.startswith("named:") else g.material) \
                    if g.material else None
                if md is None:
                    md = Directive("Material", ["diffuse"], {}, d.file)
                s["bsdf"] = self.material_ref(key, md)
                if g.area is not None:
                    if cur_obj is not None:
                        raise NotImplementedError("area light inside an object instance")
                    s["emitter"] = self.area_emitter(g.area, g, nrm, area)
                if cur_obj is not None:
                    cur_obj.append((s, g.ctm.copy()))
                else:
                    self.out[self.uid("shape")] = s
            elif n == "ObjectBegin":
                stack.append(g.copy())
                cur_obj = []
                self.objects[A[0]] = cur_obj
            elif n == "ObjectEnd":
                cur_obj = None
                g = stack.pop()
            elif n == "ObjectInstance":
                self.instances.setdefault(A[0], []).append(g.ctm.copy())
            elif n in ("ColorSpace", "PixelFilter", "Option", "Accelerator", "WorldEnd", "AttributeEnd"):
                pass
            else:
                raise NotImplementedError(f"directive {n}")
        self.place_instances()
        others = sum(lp["phi_v"] for lp in self.lights if lp["kind"] != "infinite")
        for k, v in self.out.items():
            if isinstance(v, dict) and v.get("type") == "constant":
                v["sampling_weight"] = SKY_SHARE / (1 - SKY_SHARE) * others
        if self.gonio:
            fn = self.cache / "gonio_lamps.json"
            fn.write_text(json.dumps(self.gonio))
            self.out["gonio_lamps"] = {"type": "pbrt_gonio_set", "filename": str(fn), "select": self.lamp_select,
                                       "sampling_weight": float(sum(lp["phi"] for lp in self.gonio))}
        # shapegroups must precede their instances in the dict: rebuild with groups first
        groups = {k: v for k, v in self.out.items() if k.startswith("group_")}
        rest = {k: v for k, v in self.out.items() if not k.startswith("group_")}
        self.out = {"type": "scene", **{k: v for k, v in rest.items() if k.startswith("bsdf_")}, **groups,
                    **{k: v for k, v in rest.items() if not k.startswith("bsdf_") and k != "type"}}
        self.out["sensor"] = self.sensor(cam, film, spp, width, height)
        depth = int(integ.params["maxdepth"].values[0]) if integ and "maxdepth" in integ.params else 5
        if integ and integ.args[0] != "path":
            self.note(f"integrator {integ.args[0]} -> path")
        self.out["integrator"] = {"type": "path", "max_depth": depth + self.max_depth_offset}
        if sampler is not None and sampler.args[0] != "independent":
            self.note(f"sampler {sampler.args[0]} -> independent")
        return self.out

    def sensor(self, cam, film: Directive, spp: int, width, height) -> dict:
        d, cam_from_world = cam
        P = d.params
        if d.args[0] != "perspective":
            raise NotImplementedError(f"camera {d.args[0]}")
        F = film.params
        W = width or int(F["xresolution"].values[0])
        H = height or int(F["yresolution"].values[0])
        nb = int(F["nbuckets"].values[0]) if "nbuckets" in F else 16
        lo = float(F["lambdamin"].values[0]) if "lambdamin" in F else 360.0
        hi = float(F["lambdamax"].values[0]) if "lambdamax" in F else 830.0
        edges = np.linspace(lo, hi, nb + 1)
        self.film_info = {"edges": edges.tolist(), "width": W, "height": H}
        film_d = {"type": "specfilm", "width": W, "height": H, "rfilter": {"type": "box"}, "component_format": "float32"}
        eps = 1e-3
        for i in range(nb):
            a, b = edges[i], edges[i + 1]
            film_d[f"band_{i:02d}"] = {"type": "irregular", "wavelengths": f"{a - eps},{a},{b},{b + eps}",
                                       "values": "0,1,1,0"}
        world_from_cam = np.linalg.inv(cam_from_world)
        # Mitsuba's camera x points to the image left, pbrt's to the right
        to_world = world_from_cam @ scale(-1, 1, 1)
        return {"type": "perspective", "fov": float(P["fov"].values[0]) if "fov" in P else 90.0, "fov_axis": "smaller",
                "near_clip": 1e-3, "far_clip": 1e5, "to_world": self.mtx(to_world),
                "sampler": {"type": "independent", "sample_count": spp}, "film": film_d}


# --------------------------------------------------------------------------------------------- render

def load_scene(scene: Path, spp_per_pass: int, cache: Path, variant: str = "cuda_ad_spectral", lamp_select: str = "power"):
    import mitsuba as mi  # noqa: PLC0415

    mi.set_variant(variant)
    from mitsuba_pbrt_plugins import register  # noqa: PLC0415

    register()
    conv = Converter(scene, cache, lamp_select=lamp_select)
    t0 = time.time()
    d = conv.convert(spp=spp_per_pass)
    t1 = time.time()
    sc = mi.load_dict(d)
    t2 = time.time()
    return sc, conv, {"convert_s": t1 - t0, "load_s": t2 - t1}


def render(sc, conv: Converter, spp: int, seed: int, spp_per_pass: int) -> tuple[np.ndarray, dict]:
    """Average of ``spp / spp_per_pass`` passes; returns (H x W x nbuckets, timings)."""
    import drjit as dr  # noqa: PLC0415
    import mitsuba as mi  # noqa: PLC0415

    passes = max(1, spp // spp_per_pass)
    acc = None
    t0 = time.time()
    first = None
    for p in range(passes):
        img = mi.render(sc, spp=spp_per_pass, seed=seed * 100003 + p)
        a = np.asarray(img, dtype=np.float64)
        dr.sync_thread()
        if first is None:
            first = time.time() - t0
        acc = a if acc is None else acc + a
    total = time.time() - t0
    return (acc / passes).astype(np.float32), {"first_pass_s": first, "render_s": total, "passes": passes,
                                               "spp": passes * spp_per_pass}


def write_exr(path: Path, img: np.ndarray, edges: list[float]) -> None:
    """pbrt spectral-film layout and units: S0.<center>nm (comma decimal, as pbrt writes) plus R, G, B zeros.

    Mitsuba's specfilm band is the integral of L over the bucket; pbrt's SpectralFilm stores the bucket mean of
    L times CIE_Y_integral.
    """
    from exr_multispectral import write_separate_channels_exr  # noqa: PLC0415
    from pbrt_spectral_exr_to_electrons import PBRT_CIE_Y_INTEGRAL  # noqa: PLC0415

    e = np.asarray(edges)
    centers = 0.5 * (e[1:] + e[:-1])
    img = img * (PBRT_CIE_Y_INTEGRAL / np.diff(e))[None, None, :].astype(np.float32)
    ch = {"S0." + f"{c:.3f}".replace(".", ",") + "nm": img[..., i] for i, c in enumerate(centers)}
    z = np.zeros(img.shape[:2], np.float32)
    ch.update({"R": z, "G": z, "B": z})
    path.parent.mkdir(parents=True, exist_ok=True)
    write_separate_channels_exr(path, ch)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scene", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--spp-per-pass", type=int, default=16)
    ap.add_argument("--seed", type=int, nargs="+", default=[1], help="one EXR per seed ({seed} in --out)")
    ap.add_argument("--variant", default="cuda_ad_spectral")
    ap.add_argument("--lamp-select", choices=("power", "exact"), default="power")
    ap.add_argument("--cache", type=Path, default=None, help="converted meshes (default: <scene dir>/mitsuba_cache)")
    ap.add_argument("--times", type=Path, default=None, help="JSON file to merge render times into")
    a = ap.parse_args(argv)
    sc, conv, t = load_scene(a.scene, a.spp_per_pass, a.cache or a.scene.parent / "mitsuba_cache", a.variant,
                              a.lamp_select)
    print(f"converted in {t['convert_s']:.1f}s, loaded in {t['load_s']:.1f}s")
    for msg, cnt in conv.notes.items():
        print(f"  approx: {msg} (x{cnt})")
    times = json.loads(a.times.read_text()) if a.times and a.times.exists() else {}
    for s in a.seed:
        img, rt = render(sc, conv, a.spp, s, a.spp_per_pass)
        out = Path(str(a.out).format(seed=s))
        write_exr(out, img, conv.film_info["edges"])
        print(f"seed {s}: {rt['spp']} spp in {rt['render_s']:.1f}s (first pass {rt['first_pass_s']:.1f}s) -> {out}", flush=True)
        if a.times:
            times[out.stem] = rt["render_s"]
            a.times.write_text(json.dumps(times, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
