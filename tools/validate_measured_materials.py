"""Validate the opt-in measured / spectral materials (tools/highway_materials.py) with pbrt.

Renders (needs the patched pbrt of tools/build_pbrt.sh and the RGL cache, fetched on demand):

1. White furnace: each RGL ``measured`` BRDF on a sphere under a uniform unit environment
   (orthographic camera, one bounce) -> directional albedo rho(theta_o, lambda) per spectral
   bucket, relative to a Lambertian white sphere. Energy conservation requires rho <= 1;
   Spectralon (Labsphere SRS-99: ~0.99 over 400-1500 nm) checks the absolute spectral output.
2. Fluorescence: a ``fluorescent`` sheet and a white Lambertian sheet under a D65 distant
   light at normal incidence; the per-bucket ratio is the total radiance factor
   beta_R + beta_F, compared with the analytic bispectral value (path and volpath).
3. Swatches: analytic coated-diffuse vs RGL measured paints (+ iridescent paints, Spectralon,
   fluorescent sheetings vs their reflected part only) as spheres under D65 sun + sky; sphere-
   mean CIE XYZ relative to a white Lambertian sphere -> CIELAB and CIEDE2000.

Usage: venv/bin/python tools/validate_measured_materials.py --out out/measured_materials
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import highway_materials as hm  # noqa: E402
from colour_science import (  # noqa: E402
    cmf_on_grid,
    delta_e_2000,
    load_illuminant,
    xyz_to_lab,
    xyz_to_srgb_linear,
)
from exr_multispectral import spectral_buckets_from_exr  # noqa: E402
from highway_spectra import reflectance  # noqa: E402

REPO = hm.REPO
PBRT = Path(os.environ.get("OPENCAM_PBRT", REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"))
PAINTS = ("white", "silver", "blue", "darkgreen", "red")
FURNACE_RGL = ("spectralon", "ilm_solo_m_68", "ilm_l3_37_metallic", "irid_flake_paint1", "cm_white")


def _spd(path: Path, wl: np.ndarray, v: np.ndarray) -> str:
    path.write_text("".join(f"{a:.6g} {b:.8g}\n" for a, b in zip(wl, v)))
    return path.name


def render(td: Path, body: list[str], *, header: list[str], nbuckets: int, spp: int, name: str) -> tuple:
    s = [*header, 'Sampler "zsobol" "integer pixelsamples" [%d]' % spp]
    s += [f'Film "spectral" "integer nbuckets" [{nbuckets}] "string filename" "{name}.exr" ' + header_film(header)]
    s += ["WorldBegin", *body]
    (td / f"{name}.pbrt").write_text("\n".join(s) + "\n")
    r = subprocess.run([str(PBRT), "--quiet", "--seed", "5", f"{name}.pbrt"], cwd=td, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"pbrt failed on {name}.pbrt:\n{r.stderr[-2000:]}")
    return spectral_buckets_from_exr(td / f"{name}.exr")


def header_film(header: list[str]) -> str:
    res = next(h for h in header if h.startswith("#res"))
    x, y = res.split()[1:]
    return f'"integer xresolution" [{x}] "integer yresolution" [{y}]'


def _rgl(td: Path, name: str) -> str:
    dst = td / f"{name}.bsdf"
    if not dst.exists():
        dst.symlink_to(hm.rgl_path(name))
    return dst.name


# ---------------------------------------------------------------------------- 1. furnace
def furnace(td: Path, names=FURNACE_RGL, res: int = 96, spp: int = 64) -> dict:
    hdr = [f"#res {res} {res}", "LookAt 0 0 5  0 0 0  0 1 0", 'Camera "orthographic" "float screenwindow" [-1 1 -1 1]']
    hdr.append('Integrator "path" "integer maxdepth" [1]')
    light = 'LightSource "infinite" "spectrum L" [360 1 830 1]'

    def one(mat: str, tag: str):
        return render(td, [light, mat, 'Shape "sphere" "float radius" [1]'], header=hdr, nbuckets=16, spp=spp, name=tag)

    white, lam = one('Material "diffuse" "spectrum reflectance" [360 1 830 1]', "furnace_white")
    yy, xx = (np.mgrid[0:res, 0:res] + 0.5) / res * 2 - 1
    r2 = xx**2 + yy**2
    mask = r2 < 0.97**2  # theta_o < ~76 deg (the RGL data cover incidence up to ~80 deg)
    w = white[mask].mean(axis=0)
    mu = np.sqrt(np.clip(1 - r2, 0, 1))[mask]
    out = {"bucket_nm": lam.tolist()}
    for n in names:
        img, _ = one(f'Material "measured" "string filename" "{_rgl(td, n)}"', f"furnace_{n}")
        rho = img[mask] / w
        bins = np.digitize(mu, [0.25, 0.5, 0.75, 0.9])
        prof = np.array([rho[bins == b].mean(axis=0) for b in range(5)])
        vis = (lam >= 400) & (lam <= 800)
        out[n] = {
            "max_albedo": float(prof.max()),
            "max_albedo_visible": float(prof[:, vis].max()),
            "mean_albedo_visible": float(prof[:, vis].mean()),
            "albedo_vs_mu_bins": prof.mean(axis=1).round(4).tolist(),
            "spectral_albedo_mu_gt_0.9": prof[4].round(4).tolist(),
        }
    return out


# ---------------------------------------------------------------------------- 2. fluorescence
def fluorescence(td: Path, colour: str, integrator: str = "path", nbuckets: int = 24, spp: int = 256) -> dict:
    dye = hm.FLUORESCENT_DYES[colour]
    g = np.arange(360.0, 831.0, 1.0)
    files = {
        p: _spd(td / f"{colour}_{p}.spd", g, getattr(dye, p)(g)) for p in ("reflectance", "excitation", "emission")
    }
    hdr = ["#res 64 32", "LookAt 0 0 5  0 0 0  0 1 0", 'Camera "orthographic" "float screenwindow" [-2 2 -1 1]']
    hdr.append(f'Integrator "{integrator}" "integer maxdepth" [1]')
    body = ['LightSource "distant" "point3 from" [0 0 5] "point3 to" [0 0 0] "spectrum L" "stdillum-D65"']
    body += [
        'AttributeBegin Translate 1 0 0 Material "fluorescent"'  # left half of the image
        f' "spectrum reflectance" "{files["reflectance"]}" "spectrum excitation" "{files["excitation"]}"'
        f' "spectrum emission" "{files["emission"]}"',
        'Shape "bilinearmesh" "point3 P" [-0.9 -0.9 0 0.9 -0.9 0 -0.9 0.9 0 0.9 0.9 0] AttributeEnd',
        'AttributeBegin Translate -1 0 0 Material "diffuse" "spectrum reflectance" [360 1 830 1]',
        'Shape "bilinearmesh" "point3 P" [-0.9 -0.9 0 0.9 -0.9 0 -0.9 0.9 0 0.9 0.9 0] AttributeEnd',
    ]
    img, lam = render(td, body, header=hdr, nbuckets=nbuckets, spp=spp, name=f"fluor_{colour}_{integrator}")
    measured = img[6:26, 4:28].mean(axis=(0, 1)) / img[6:26, 36:60].mean(axis=(0, 1))
    _, d65 = load_illuminant(REPO, "D65", g)
    br, bf = hm.radiance_factors(dye, g, d65)
    edges = np.linspace(360.0, 830.0, nbuckets + 1)
    idx = np.clip(np.digitize(g, edges) - 1, 0, nbuckets - 1)
    ana = np.array([((br + bf) * d65)[idx == k].sum() / d65[idx == k].sum() for k in range(nbuckets)])
    ana_r = np.array([(br * d65)[idx == k].sum() / d65[idx == k].sum() for k in range(nbuckets)])
    return {
        "bucket_nm": lam.round(1).tolist(),
        "rendered_total_factor": measured.round(4).tolist(),
        "analytic_total_factor": ana.round(4).tolist(),
        "analytic_reflected_factor": ana_r.round(4).tolist(),
        "max_abs_error": float(np.max(np.abs(measured - ana))),
        "peak_total_factor": float(ana.max()),
    }


# ---------------------------------------------------------------------------- 3. swatches
def swatches(td: Path, res_per_cell: int = 90, spp: int = 128) -> dict:
    wl = np.arange(360.0, 831.0, 5.0)
    cols = 7
    cells: list[tuple[str, str]] = []
    for c in PAINTS:
        cells.append(
            (f"analytic {c}", f'"coateddiffuse" "spectrum reflectance" "{_paint(td, wl, c)}" "float roughness" [0.008]')
        )
    cells += [
        (
            "analytic black",
            f'"coateddiffuse" "spectrum reflectance" "{_paint(td, wl, "black")}" "float roughness" [0.008]',
        )
    ]
    cells += [
        (
            "analytic gray",
            f'"coateddiffuse" "spectrum reflectance" "{_paint(td, wl, "gray")}" "float roughness" [0.008]',
        )
    ]
    for c in PAINTS:
        cells.append((f"measured {c}", f'"measured" "string filename" "{_rgl(td, hm.MEASURED_PAINT[c])}"'))
    for n in ("irid_flake_paint1", "irid_flake_paint2"):
        cells.append((n, f'"measured" "string filename" "{_rgl(td, n)}"'))
    cells.append(("spectralon", f'"measured" "string filename" "{_rgl(td, "spectralon")}"'))
    g = np.arange(360.0, 831.0, 1.0)
    for colour, dye in hm.FLUORESCENT_DYES.items():
        f = {
            p: _spd(td / f"sw_{colour}_{p}.spd", g, getattr(dye, p)(g))
            for p in ("reflectance", "excitation", "emission")
        }
        cells.append(
            (
                f"fluorescent {colour}",
                f'"fluorescent" "spectrum reflectance" "{f["reflectance"]}" "spectrum excitation" "{f["excitation"]}"'
                f' "spectrum emission" "{f["emission"]}"',
            )
        )
        cells.append((f"{colour} reflected only", f'"diffuse" "spectrum reflectance" "{f["reflectance"]}"'))
    rows = -(-len(cells) // cols)
    hdr = [f"#res {cols * res_per_cell} {rows * res_per_cell}", "LookAt 0 0 10  0 0 0  0 1 0"]
    hdr += [f'Camera "orthographic" "float screenwindow" [{-cols / 2} {cols / 2} {-rows / 2} {rows / 2}]']
    hdr += ['Integrator "volpath" "integer maxdepth" [5]']
    lights = [
        'LightSource "distant" "point3 from" [-3 4 10] "point3 to" [0 0 0] "spectrum L" "stdillum-D65" "float scale" [3]',
        'LightSource "infinite" "spectrum L" "stdillum-D65" "float scale" [0.3]',
    ]

    def centre(i: int) -> tuple[float, float]:
        return (cols - 1) / 2 - i % cols, (rows - 1) / 2 - i // cols  # image x = -world x here

    def scene(mats: list[str]) -> list[str]:
        b = list(lights)
        for i, m in enumerate(mats):
            x, y = centre(i)
            b += [
                f"AttributeBegin Translate {x} {y} 0 Material {m}",
                'Shape "sphere" "float radius" [0.42] AttributeEnd',
            ]
        return b

    img, lam = render(td, scene([m for _, m in cells]), header=hdr, nbuckets=47, spp=spp, name="swatches")
    ref, _ = render(td, scene(['"diffuse" "spectrum reflectance" [360 1 830 1]'] * len(cells)), header=hdr,
                    nbuckets=47, spp=max(16, spp // 4), name="swatches_white")  # fmt: skip
    cmf = np.asarray(cmf_on_grid(lam))
    cmf = cmf if cmf.shape[0] == lam.size else cmf.T
    yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]] + 0.5
    px = res_per_cell
    results: dict[str, dict] = {}
    for i, (label, _) in enumerate(cells):
        cx, cy = (i % cols + 0.5) * px, (i // cols + 0.5) * px
        m = (xx - cx) ** 2 + (yy - cy) ** 2 < (0.38 * px) ** 2
        white_y = ref[m].mean(axis=0) @ cmf[:, 1]
        xyz = 100.0 * (img[m].mean(axis=0) @ cmf) / white_y
        wxyz = 100.0 * (ref[m].mean(axis=0) @ cmf) / white_y
        lab = xyz_to_lab(xyz[None], wxyz)[0]
        results[label] = {
            "XYZ": xyz.round(3).tolist(),
            "xy": (xyz[:2] / xyz.sum()).round(4).tolist(),
            "Lab": lab.round(2).tolist(),
        }
    for c in PAINTS:
        a, b = np.array(results[f"analytic {c}"]["Lab"]), np.array(results[f"measured {c}"]["Lab"])
        results[f"measured {c}"]["dE00_vs_analytic"] = float(delta_e_2000(a[None], b[None])[0])
    for c in hm.FLUORESCENT_DYES:
        a, b = np.array(results[f"{c} reflected only"]["Lab"]), np.array(results[f"fluorescent {c}"]["Lab"])
        results[f"fluorescent {c}"]["dE00_vs_reflected_only"] = float(delta_e_2000(a[None], b[None])[0])
        results[f"fluorescent {c}"]["in_type_xi_box"] = hm.in_polygon(
            results[f"fluorescent {c}"]["xy"], hm.TYPE_XI_LIMITS[c]["xy"]
        )
    wscale = (ref.reshape(-1, lam.size) @ cmf[:, 1]).max()
    srgb = xyz_to_srgb_linear((img @ cmf).reshape(-1, 3) / wscale).reshape(*img.shape[:2], 3)
    return {"cells": [lbl for lbl, _ in cells], "cols": cols, "results": results, "_srgb": srgb}


def _paint(td: Path, wl: np.ndarray, c: str) -> str:
    return _spd(td / f"carpaint_{c}.spd", wl, reflectance(f"carpaint_{c}", wl))


def figure(out: Path, sw: dict, fur: dict, fl: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 9.5))
    ax = fig.add_axes([0.02, 0.36, 0.96, 0.62])
    img = np.clip(sw["_srgb"], 0, None)
    img = np.where(img <= 0.0031308, 12.92 * img, 1.055 * np.power(img, 1 / 2.4) - 0.055)
    ax.imshow(np.clip(img, 0, 1))
    ax.set_axis_off()
    n = sw["img_cell"] if "img_cell" in sw else img.shape[1] / sw["cols"]
    for i, lbl in enumerate(sw["cells"]):
        r = sw["results"][lbl]
        txt = lbl + (f"\ndE00={r['dE00_vs_analytic']:.1f}" if "dE00_vs_analytic" in r else "")
        txt += f"\ndE00={r['dE00_vs_reflected_only']:.1f}" if "dE00_vs_reflected_only" in r else ""
        ax.text(
            (i % sw["cols"] + 0.5) * n,
            (i // sw["cols"] + 0.97) * n,
            txt,
            color="w",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_title("pbrt spectral render (D65 sun+sky): row 1 analytic paints, row 2 RGL measured, rows 3-4 extras")
    a1 = fig.add_axes([0.06, 0.06, 0.4, 0.25])
    for k, v in fur.items():
        if k != "bucket_nm":
            a1.plot(fur["bucket_nm"], v["spectral_albedo_mu_gt_0.9"], label=k)
    a1.axhline(1.0, color="k", lw=0.8, ls="--")
    a1.set(xlabel="wavelength [nm]", ylabel="albedo (theta_o < 26 deg)", title="white furnace, pbrt measured BRDF")
    a1.legend(fontsize=7)
    a2 = fig.add_axes([0.55, 0.06, 0.4, 0.25])
    for c, v in fl.items():
        a2.plot(v["bucket_nm"], v["analytic_total_factor"], "-", label=f"{c} analytic total")
        a2.plot(v["bucket_nm"], v["rendered_total_factor"], "o", ms=3, label=f"{c} pbrt (path)")
        a2.plot(v["bucket_nm"], v["analytic_reflected_factor"], ":", label=f"{c} reflected only")
    a2.set(xlabel="wavelength [nm]", ylabel="radiance factor (D65, 0/0)", title="fluorescent sheeting, bispectral")
    a2.legend(fontsize=6)
    p = out / "measured_materials_validation.png"
    fig.savefig(p, dpi=110)
    return p


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=REPO / "out" / "measured_materials")
    ap.add_argument("--spp", type=int, default=128)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as t:
        td = Path(t)
        fur = furnace(td)
        fl = {c: fluorescence(td, c) for c in hm.FLUORESCENT_DYES}
        fl_vol = {c: fluorescence(td, c, "volpath")["max_abs_error"] for c in hm.FLUORESCENT_DYES}
        sw = swatches(td, spp=a.spp)
    fig = figure(a.out, sw, fur, fl)
    sw.pop("_srgb")
    res = {"furnace": fur, "fluorescence_path": fl, "fluorescence_volpath_max_abs_error": fl_vol, "swatches": sw}
    res["type_xi_analytic"] = {
        c: {k: np.asarray(v).round(4).tolist() for k, v in hm.sheeting_colour(c).items()} for c in hm.FLUORESCENT_DYES
    }
    (a.out / "measured_materials_validation.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"wrote {fig}")
    for n, v in fur.items():
        if n != "bucket_nm":
            print(
                f"furnace {n}: max albedo {v['max_albedo']:.4f} (400-800 nm {v['max_albedo_visible']:.4f}), mean {v['mean_albedo_visible']:.4f}"
            )
    for c, v in fl.items():
        print(
            f"fluorescence {c}: max |rendered-analytic| = {v['max_abs_error']:.4f} (path), {fl_vol[c]:.4f} (volpath); peak {v['peak_total_factor']:.3f}"
        )
    for k, v in sw["results"].items():
        extra = {kk: vv for kk, vv in v.items() if kk not in ("XYZ",)}
        print(f"swatch {k}: {extra}")


if __name__ == "__main__":
    main()
