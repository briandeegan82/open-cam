"""Opt-in measured / spectral materials for the highway scenes (tools/build_highway_scene.py).

Three independent options (all default off; the scene is byte-identical when off):

``--car-paint measured``
    Car basecoats use pbrt-v4's ``measured`` material with spectral BRDFs from the RGL
    material database (Dupuy & Jakob 2018, "An adaptive parameterization for efficient
    material acquisition and rendering", ACM TOG 37(6) 274, doi:10.1145/3272127.3275059;
    https://rgl.epfl.ch/materials, every entry licensed CC0 1.0). The ``*_spec.bsdf`` files
    (195 bands, 360-1000 nm, isotropic, 8 incidence angles) are fetched at build time into
    the gitignored ``scenes/assets/measured/`` cache (sha256-pinned; ~7 MB each).
``--spectral-library usgs``
    Measured spectral reflectances from the USGS Spectral Library v7 (Kokaly et al. 2017,
    USGS Data Series 1035, doi:10.5066/F7RR1WDJ; public domain) replace the analytic curves
    of the road asphalt, concrete barrier and galvanised steel, and the car assets' near-black
    RGB trim (black ABS plastic spectral shape, authors' luminance kept); the assets' RGB
    ``rgb eta/k`` chrome conductors use the Cr optical constants of Rakic et al. 1998
    (Appl. Opt. 37, 5271; refractiveindex.info, CC0).
``--fluorescent-sign {yellow-green,orange}``
    Adds an ASTM D4956 Type XI fluorescent warning sign (0.9 m diamond) on the right verge,
    rendered with the ``fluorescent`` material of third_party/patches/0002-fluorescent-material.patch:
    true wavelength-shifting reradiation through a separable Donaldson matrix. The dye spectra
    are NOT measured: logistic absorption + Gaussian emission bands fitted so that the
    bispectral D65 colour lies at the centroid of the 23 CFR 655 Subpart F (Appendix, Table 3)
    chromaticity box with the "typical" fluorescence luminance factor Y_F (Table 3a).
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CACHE_REL = "scenes/assets/measured"

RGL_URL = "https://d38rqfq1h7iukm.cloudfront.net/media/materials/{0}/{0}_spec.bsdf"
#: RGL material -> (sha256 of ``<name>_spec.bsdf``, RGL description).
RGL_MATERIALS = {
    "ilm_solo_m_68": (
        "eba7178e2058278882765e2946d2c49c0f47429c41eab70f098f0af359ea3a76",
        "Blue metallic m68-speeder material (ILM)",
    ),
    "ilm_l3_37_metallic": (
        "b07911afbede810e27d75824084c30135f4687ae6628a4a6b0b0eafd46aba4b7",
        "Metallic paint from L3-37 robot (ILM)",
    ),
    "ilm_l3_37_dark_green": (
        "113f7e8b5ec3c164cdd9283a7b60473cfa185dec4dc64e2ca5d1236b40d83a52",
        "Dark green paint from L3-37 robot (ILM)",
    ),
    "irid_flake_paint1": (
        "57d25024fd3f58c99cd8510c86de1ddc910b1aa15537ff982398db1d6b1f36e7",
        "Iridescent flake car paint (Dupli-color flip flop ultra) on aluminium, black primer",
    ),
    "irid_flake_paint2": (
        "6a581962799bd23e1ebcb16c9d5e38a93f2d7e71bfc28c9d407f0ee7bf33d4e5",
        "Iridescent car paint with flakes",
    ),
    "cm_white": (
        "55823e81337790eee58c8b7c6b5fa918bc56aea619e0a8e4bb31b4ee295ffed8",
        "TeckWrap vinyl wrapping film (White CM02)",
    ),
    "vch_dragon_eye_red": (
        "f6bbecfc63ae2bd54ba1b912e858177a523690d832c8d61e286f8078f3b101b1",
        "TeckWrap vinyl wrapping film (Dragon Eye Red VCH501N)",
    ),
    "spectralon": (
        "932fb97b99d69907168cb37c2d287b058dc9138c7307e483788cfe5f4c858b2c",
        "Labsphere SRS-99-020 reflectance standard",
    ),
}
#: Analytic paint colour (highway_spectra.CAR_PAINTS) -> RGL material. Colours without a
#: measured counterpart (black, gray) keep the analytic coated-diffuse basecoat. White and
#: red are vinyl wrap films, not paints (no measured solid white/red car paint in RGL).
MEASURED_PAINT = {
    "blue": "ilm_solo_m_68",
    "silver": "ilm_l3_37_metallic",
    "darkgreen": "ilm_l3_37_dark_green",
    "white": "cm_white",
    "red": "vch_dragon_eye_red",
}

USGS_ZIP_URL = (
    "https://www.sciencebase.gov/catalog/file/get/586e8c88e4b0f5ce109fccae"
    "?f=__disk__a7%2F4f%2F91%2Fa74f913e0b7d1b8123ad059e52506a02b75a2832"
)
USGS_ZIP_SHA256 = "d232645740869a82aafcad5839448c50b1dc72965ce042d1374f29b7a798a91c"
_USGS_DIR = "ASCIIdata_splib07a/"
USGS_WAVELENGTHS = _USGS_DIR + "splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt"
_USGS_A = _USGS_DIR + "ChapterA_ArtificialMaterials/splib07a_"
#: scene surface -> USGS splib07a record (ASD FieldSpec, 350-2500 nm, absolute reflectance).
USGS_SPECTRA = {
    "asphalt": _USGS_A + "Asphalt_GDS376_Blck_Road_old_ASDFRa_AREF.txt",
    "concrete": _USGS_A + "Concrete_GDS375_Lt_Gry_Road_ASDFRa_AREF.txt",
    "galvanized": _USGS_A + "GalvanizedSheetMetal_GDS334_ASDFRa_AREF.txt",
    "black_plastic": _USGS_A + "Plastic_ABS_GDS341_BlackPipe_ASDFRa_AREF.txt",
}
#: builder SPD file overwritten by each USGS spectrum (the rest are new files).
USGS_REPLACES = {"asphalt": "asphalt_aged", "concrete": "concrete", "galvanized": "galvanized"}

CR_NK_URL = "https://refractiveindex.info/database/data/main/Cr/nk/Rakic-BB.yml"
CR_NK_SHA256 = "944dbf43022886249df59e4b7eaca92dcd454f1c2661e20f93cbb0a7057c119f"

# ---------------------------------------------------------------------------- fluorescence
#: 23 CFR 655 Subpart F Appendix: Table 3 daytime chromaticity corners (CIE 1931 2 deg, D65,
#: 45/0) and Table 3a luminance factor Y (min) / typical fluorescence luminance factor Y_F, %.
TYPE_XI_LIMITS = {
    "yellow-green": {
        "xy": ((0.387, 0.610), (0.369, 0.546), (0.428, 0.496), (0.460, 0.540)),
        "y_min": 60.0,
        "yf_typical": 20.0,
    },
    "orange": {
        "xy": ((0.583, 0.416), (0.535, 0.400), (0.595, 0.351), (0.645, 0.355)),
        "y_min": 25.0,
        "yf_typical": 15.0,
    },
}


def _flush(v: np.ndarray) -> np.ndarray:
    """Zero values below 1e-20 (pbrt parses SPD files as 32-bit floats)."""
    return np.where(np.abs(v) < 1e-20, 0.0, v)


@dataclass(frozen=True)
class FluorescentDye:
    """Parametric fluorescent sheeting (fitted to TYPE_XI_LIMITS, not measured).

    Absorptance A = a/(1+exp((l-lc)/w)); reflectance R = rho (1-A)(1 - b/(1+exp(-(l-lr)/15)));
    excitation x = qy A (quantum yield x absorptance); photon emission ~ N(mu, sigma).
    """

    a: float
    lc: float
    w: float
    qy: float
    mu: float
    sigma: float
    b: float
    lr: float
    rho: float = 0.9

    def absorptance(self, wl: np.ndarray) -> np.ndarray:
        return _flush(self.a / (1.0 + np.exp((wl - self.lc) / self.w)))

    def reflectance(self, wl: np.ndarray) -> np.ndarray:
        return _flush(self.rho * (1.0 - self.absorptance(wl)) * (1.0 - self.b / (1.0 + np.exp(-(wl - self.lr) / 15.0))))

    def excitation(self, wl: np.ndarray) -> np.ndarray:
        return self.qy * self.absorptance(wl)

    def emission(self, wl: np.ndarray) -> np.ndarray:
        return _flush(np.exp(-0.5 * ((wl - self.mu) / self.sigma) ** 2))


FLUORESCENT_DYES = {
    "yellow-green": FluorescentDye(0.991, 509.479, 5.638, 0.278, 548.678, 42.712, 0.574, 614.4),
    "orange": FluorescentDye(0.989, 569.982, 28.441, 0.400, 630.186, 8.053, 0.0, 663.081),
}


def donaldson_matrix(dye: FluorescentDye, wl: np.ndarray) -> np.ndarray:
    """Energy Donaldson matrix M[em, ex] [nm^-1] exactly as the pbrt patch evaluates it.

    M = x(lex) e(lem) lex / lem for lex < lem, e normalised to unit area over 360-830 nm at
    1 nm (Stokes shift; photon-number yield qy converted to energy by lex/lem).
    """
    grid = np.arange(360.0, 831.0, 1.0)
    e = np.clip(dye.emission(wl), 0, None) / np.clip(dye.emission(grid), 0, None).sum()
    x = np.clip(dye.excitation(wl), 0, 1)
    return np.where(wl[None, :] < wl[:, None], e[:, None] * x[None, :] * wl[None, :] / wl[:, None], 0.0)


def radiance_factors(dye: FluorescentDye, wl: np.ndarray, illuminant: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reflected and fluorescent radiance factors (beta_R, beta_F) of a Lambertian sheet.

    beta_F(lem) = sum_ex M(lem, lex) E(lex) dlex / E(lem); wl must be a uniform grid.
    """
    d = float(wl[1] - wl[0])
    bf = donaldson_matrix(dye, wl) @ illuminant * d / np.where(illuminant > 0, illuminant, np.inf)
    return dye.reflectance(wl), bf


def sheeting_colour(colour: str, repo: Path = REPO) -> dict:
    """Bispectral D65 colour (xy, Y, Y_F in %) of a fitted Type XI sheeting."""
    from colour_science import cmf_on_grid, load_illuminant

    wl = np.arange(360.0, 831.0, 1.0)
    _, d65 = load_illuminant(repo, "D65", wl)
    cmf = np.asarray(cmf_on_grid(wl))
    cmf = cmf if cmf.shape[0] == wl.size else cmf.T
    br, bf = radiance_factors(FLUORESCENT_DYES[colour], wl, d65)
    norm = d65 @ cmf[:, 1]
    xyz = 100.0 * ((br + bf) * d65) @ cmf / norm
    return {"xy": xyz[:2] / xyz.sum(), "Y": float(xyz[1]), "YF": float(100.0 * (bf * d65) @ cmf[:, 1] / norm)}


def in_polygon(pt, corners) -> bool:
    from matplotlib.path import Path as MplPath

    return bool(MplPath(np.asarray(corners)).contains_point(tuple(pt)))


# ---------------------------------------------------------------------------- data cache
def cache_dir(repo: Path = REPO) -> Path:
    return (Path(repo) / CACHE_REL).resolve()


def fetch(repo: Path = REPO, rgl: tuple[str, ...] | None = None, usgs: bool = True, cr: bool = True) -> Path:
    """Download (sha256-checked) the RGL / USGS / Cr data into the gitignored cache."""
    from fetch_highway_assets import _download

    root = cache_dir(repo)
    for name in RGL_MATERIALS if rgl is None else rgl:
        _download(RGL_URL.format(name), root / "rgl" / f"{name}.bsdf", RGL_MATERIALS[name][0])
    if usgs:
        members = [USGS_WAVELENGTHS, *USGS_SPECTRA.values()]
        if not all((root / "usgs" / Path(m).name).is_file() for m in members):
            zpath = root / "usgs" / "splib07a_ASCII.zip"
            _download(USGS_ZIP_URL, zpath, USGS_ZIP_SHA256)
            with zipfile.ZipFile(zpath) as z:
                for m in members:
                    (root / "usgs" / Path(m).name).write_bytes(z.read(m))
            zpath.unlink()
    if cr:
        _download(CR_NK_URL, root / "Cr_Rakic-BB.yml", CR_NK_SHA256)
    return root


def rgl_path(name: str, repo: Path = REPO) -> Path:
    p = cache_dir(repo) / "rgl" / f"{name}.bsdf"
    if not p.is_file():
        fetch(repo, rgl=(name,), usgs=False, cr=False)
    return p


def read_rgl_tensor(path: Path) -> dict[str, np.ndarray]:
    """Read an RGL/pbrt ``tensor_file`` (format of pbrt-v4 src/pbrt/bxdfs.cpp)."""
    import struct

    dtypes = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16, 5: np.int32, 6: np.uint32, 7: np.int64}
    dtypes |= {8: np.uint64, 9: np.float16, 10: np.float32, 11: np.float64}
    b = Path(path).read_bytes()
    if b[:12] != b"tensor_file\0":
        raise ValueError(f"{path}: not a tensor file")
    (n,) = struct.unpack_from("<I", b, 14)
    p, out = 18, {}
    for _ in range(n):
        (ln,) = struct.unpack_from("<H", b, p)
        name = b[p + 2 : p + 2 + ln].decode()
        p += 2 + ln
        nd, dt, off = struct.unpack_from("<HBQ", b, p)
        p += 11
        shape = struct.unpack_from("<" + "Q" * nd, b, p)
        p += 8 * nd
        out[name] = np.frombuffer(b, dtype=dtypes[dt], count=int(np.prod(shape)), offset=off).reshape(shape)
    return out


def read_usgs(key: str, wl: np.ndarray, repo: Path = REPO) -> np.ndarray:
    """USGS splib07a reflectance resampled to ``wl`` [nm] (bad channels dropped)."""
    root = cache_dir(repo) / "usgs"
    if not (root / Path(USGS_SPECTRA[key]).name).is_file():
        fetch(repo, rgl=(), usgs=True, cr=False)
    lam = np.loadtxt(root / Path(USGS_WAVELENGTHS).name, skiprows=1) * 1000.0
    val = np.loadtxt(root / Path(USGS_SPECTRA[key]).name, skiprows=1)
    ok = (val > -1.0) & (val < 2.0)
    return np.interp(np.asarray(wl, float), lam[ok], val[ok])


def chromium_nk(wl: np.ndarray, repo: Path = REPO) -> tuple[np.ndarray, np.ndarray]:
    """Cr refractive index n, k at ``wl`` [nm] (Rakic et al. 1998 Brendel-Bormann fit)."""
    p = cache_dir(repo) / "Cr_Rakic-BB.yml"
    if not p.is_file():
        fetch(repo, rgl=(), usgs=False, cr=True)
    rows = [ln.split() for ln in p.read_text().splitlines() if re.match(r"^\s+[0-9.]+e[-+]\d+\s", ln)]
    d = np.array(rows, dtype=float)
    w = np.asarray(wl, float)
    return np.interp(w, d[:, 0] * 1000.0, d[:, 1]), np.interp(w, d[:, 0] * 1000.0, d[:, 2])


# ---------------------------------------------------------------------------- scene hooks
def add_arguments(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("measured / spectral materials (tools/highway_materials.py)")
    g.add_argument("--car-paint", choices=("analytic", "measured"), default="analytic", help="RGL measured basecoats.")
    g.add_argument("--spectral-library", choices=("analytic", "usgs"), default="analytic", help="USGS reflectances.")
    g.add_argument("--fluorescent-sign", choices=("none", *FLUORESCENT_DYES), default="none")


_RGB_REFL = re.compile(r'"rgb reflectance"\s*\[\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\]')
_RGB_NK = re.compile(r'\s*"rgb (?:eta|k)"\s*\[[^\]]*\]')


def _statements(lines: list[str]) -> list[list[str]]:
    out: list[list[str]] = []
    for ln in lines:
        if out and ln[:1].isspace() and out[-1][0].startswith("MakeNamedMaterial"):
            out[-1].append(ln)
        else:
            out.append([ln])
    return out


class Materials:
    """Per-scene state for the opt-in material options (no-op with the defaults)."""

    def __init__(self, args: argparse.Namespace, out_dir: Path, wl: np.ndarray, write_spd, repo: Path = REPO):
        self.car_paint = getattr(args, "car_paint", "analytic")
        self.library = getattr(args, "spectral_library", "analytic")
        self.sign = getattr(args, "fluorescent_sign", "none")
        self.out_dir, self.wl, self.write_spd, self.repo = Path(out_dir), np.asarray(wl, float), write_spd, repo
        self._black: dict[str, str] = {}

    @property
    def active(self) -> bool:
        return (self.car_paint, self.library, self.sign) != ("analytic", "analytic", "none")

    def write_spectra(self) -> None:
        """Overwrite / add the SPD files used by the options (call after the analytic ones)."""
        spd = self.out_dir / "spd"
        if self.library == "usgs":
            for key, name in USGS_REPLACES.items():
                self.write_spd(spd / f"{name}.spd", self.wl, read_usgs(key, self.wl, self.repo))
            n, k = chromium_nk(self.wl, self.repo)
            self.write_spd(spd / "cr_eta.spd", self.wl, n)
            self.write_spd(spd / "cr_k.spd", self.wl, k)
        if self.sign != "none":
            dye, tag = FLUORESCENT_DYES[self.sign], self.sign.replace("-", "")
            for part in ("reflectance", "excitation", "emission"):
                self.write_spd(spd / f"fluor_{tag}_{part}.spd", self.wl, getattr(dye, part)(self.wl))

    def _black_spd(self, rgb: tuple[float, float, float]) -> str:
        """USGS black ABS spectral shape scaled to the RGB's luminance (Rec.709 Y)."""
        y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        tag = f"{y:.4f}"
        if tag not in self._black:
            from colour_science import cmf_on_grid

            shape = read_usgs("black_plastic", self.wl, self.repo)
            ybar = np.asarray(cmf_on_grid(self.wl))
            ybar = (ybar if ybar.shape[0] == self.wl.size else ybar.T)[:, 1]
            scale = y / float((shape * ybar).sum() / ybar.sum())
            self._black[tag] = f"spd/black_plastic_{tag}.spd"
            self.write_spd(self.out_dir / self._black[tag], self.wl, np.clip(shape * scale, 0, 1))
        return self._black[tag]

    def car_body(self, lines: list[str], colour: str) -> list[str]:
        """Rewrite one car's material statements (paint -> RGL; RGB chrome/black trim -> spectral)."""
        if self.car_paint == "analytic" and self.library == "analytic":
            return lines
        out: list[str] = []
        for st in _statements(lines):
            text = "\n".join(st)
            if not st[0].startswith("MakeNamedMaterial"):
                out += st
                continue
            name = re.match(r'MakeNamedMaterial\s+"([^"]+)"', text).group(1)
            is_paint = '"spectrum reflectance" "spd/carpaint_' in text and '"coateddiffuse"' in text
            if is_paint and self.car_paint == "measured" and colour in MEASURED_PAINT:
                rgl = MEASURED_PAINT[colour]
                dest = self.out_dir / "measured" / f"{rgl}.bsdf"
                if not dest.is_file():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(rgl_path(rgl, self.repo), dest)
                out.append(
                    f'MakeNamedMaterial "{name}" "string type" "measured" "string filename" "measured/{rgl}.bsdf"'
                )
                continue
            if self.library == "usgs":
                if '"conductor"' in text and _RGB_NK.search(text):
                    text = _RGB_NK.sub("", text) + ' "spectrum eta" "spd/cr_eta.spd" "spectrum k" "spd/cr_k.spd"'
                m = _RGB_REFL.search(text)
                if m:
                    rgb = tuple(float(v) for v in m.groups())
                    if 0 < max(rgb) <= 0.12 and max(rgb) - min(rgb) <= 0.1 * max(rgb) + 0.005:
                        text = text[: m.start()] + f'"spectrum reflectance" "{self._black_spd(rgb)}"' + text[m.end() :]
            out += text.split("\n")
        return out

    def sign_lines(self, x: float, z: float, mesh, box, side: float = 0.9, y0: float = 1.5) -> list[str]:
        """A Type XI fluorescent diamond warning sign (plain face) on a galvanised post."""
        if self.sign == "none":
            return []
        tag = self.sign.replace("-", "")
        h = side / np.sqrt(2.0)
        cy = y0 + h
        p = np.array([(x, cy - h, z), (x + h, cy, z), (x, cy + h, z), (x - h, cy, z)])
        return [
            f'MakeNamedMaterial "fluor_sheet" "string type" "fluorescent"'
            f' "spectrum reflectance" "spd/fluor_{tag}_reflectance.spd"'
            f' "spectrum excitation" "spd/fluor_{tag}_excitation.spd"'
            f' "spectrum emission" "spd/fluor_{tag}_emission.spd"',
            'NamedMaterial "fluor_sheet"',
            *mesh(p, np.array([(0, 2, 1), (0, 3, 2)])),
            'NamedMaterial "galvanized"',
            *mesh(*box(x, -0.1, z + 0.06, 0.08, cy, 0.08)),
            "",
        ]

    def manifest(self) -> dict:
        if not self.active:
            return {}
        m = {"car_paint": self.car_paint, "spectral_library": self.library, "fluorescent_sign": self.sign}
        if self.car_paint == "measured":
            m["measured_paints"] = MEASURED_PAINT
        return {"materials": m}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("fetch", "colours"))
    ap.add_argument("--repo-root", type=Path, default=REPO)
    a = ap.parse_args(argv)
    if a.command == "fetch":
        print(f"measured-material cache: {fetch(a.repo_root)}")
    else:
        for c, lim in TYPE_XI_LIMITS.items():
            r = sheeting_colour(c, a.repo_root)
            print(f"{c}: xy={r['xy'].round(4)} Y={r['Y']:.1f} YF={r['YF']:.1f} in_box={in_polygon(r['xy'], lim['xy'])}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
